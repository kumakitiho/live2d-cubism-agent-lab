from __future__ import annotations

import threading
import time

import pytest

from tools.resource_scheduler import (
    ResourceLimits,
    ResourceScheduler,
    ScheduledTask,
    resource_kind_for_device,
)


@pytest.mark.parametrize(
    ("backend", "device", "expected"),
    [
        ("mock", "cpu", "cpu"),
        ("mock", "cuda", "cpu"),
        ("sam2", "cpu", "cpu"),
        ("sam2", "cuda", "gpu"),
        ("sam2", "cuda:0", "gpu"),
        ("diffusers", "cpu", "cpu"),
        ("diffusers", "cuda", "gpu"),
        ("flux_fill", "mps", "gpu"),
        ("diffusers", "xpu", "gpu"),
    ],
)
def test_resource_kind_follows_backend_and_device(
    backend: str,
    device: str,
    expected: str,
) -> None:
    assert resource_kind_for_device(backend, device) == expected


def test_resource_kind_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        resource_kind_for_device("diffusers", "tpu")


def test_cpu_real_backend_ignores_gpu_budget_and_model_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingLock:
        def __enter__(self) -> None:
            raise AssertionError("CPU task must not acquire the model lock")

        def __exit__(self, *_args: object) -> None:
            return None

    scheduler = ResourceScheduler(ResourceLimits(gpu_memory_budget_mb=8192))
    monkeypatch.setattr(scheduler, "_model_lock", ExplodingLock())
    results = scheduler.run(
        [
            ScheduledTask(
                "diffusers-cpu",
                lambda: "ok",
                resource=resource_kind_for_device("diffusers", "cpu"),
                gpu_memory_mb=0,
            )
        ]
    )

    assert results["diffusers-cpu"].status == "completed"
    assert results["diffusers-cpu"].value == "ok"


@pytest.mark.parametrize(
    ("estimate", "expected_status", "error_fragment"),
    [
        (0, "failed", "unknown GPU memory usage"),
        (4096, "completed", None),
        (12288, "failed", "exceeds gpu memory budget"),
    ],
)
def test_gpu_real_backend_applies_memory_budget(
    estimate: int,
    expected_status: str,
    error_fragment: str | None,
) -> None:
    results = ResourceScheduler(ResourceLimits(gpu_memory_budget_mb=8192)).run(
        [
            ScheduledTask(
                "diffusers-cuda",
                lambda: "ok",
                resource=resource_kind_for_device("diffusers", "cuda"),
                gpu_memory_mb=estimate,
            )
        ]
    )

    result = results["diffusers-cuda"]
    assert result.status == expected_status
    if error_fragment is not None:
        assert error_fragment in str(result.error)


def test_independent_cpu_tasks_run_in_parallel() -> None:
    barrier = threading.Barrier(2)

    def operation() -> str:
        barrier.wait(timeout=2)
        return "ok"

    results = ResourceScheduler(ResourceLimits(max_cpu_workers=2)).run(
        [ScheduledTask("a", operation), ScheduledTask("b", operation)]
    )

    assert {result.status for result in results.values()} == {"completed"}


def test_gpu_worker_limit_is_enforced() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def operation() -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    scheduler = ResourceScheduler(ResourceLimits(max_gpu_workers=1, model_exclusive_lock=False))
    scheduler.run(
        [
            ScheduledTask("gpu-a", operation, resource="gpu"),
            ScheduledTask("gpu-b", operation, resource="gpu"),
        ]
    )

    assert maximum == 1


def test_model_exclusive_lock_serializes_gpu_models() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def operation() -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    scheduler = ResourceScheduler(ResourceLimits(max_gpu_workers=2, model_exclusive_lock=True))
    scheduler.run(
        [
            ScheduledTask("segmentation", operation, resource="gpu"),
            ScheduledTask("inpainting", operation, resource="gpu"),
        ]
    )

    assert maximum == 1


def test_dependency_orders_segmentation_before_inpainting() -> None:
    calls: list[str] = []
    scheduler = ResourceScheduler()

    results = scheduler.run(
        [
            ScheduledTask("segmentation", lambda: calls.append("segmentation")),
            ScheduledTask(
                "inpainting",
                lambda: calls.append("inpainting"),
                dependencies=("segmentation",),
            ),
        ]
    )

    assert calls == ["segmentation", "inpainting"]
    assert results["inpainting"].status == "completed"


def test_stage_failure_blocks_dependent_stage() -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    results = ResourceScheduler().run(
        [
            ScheduledTask("first", fail),
            ScheduledTask("later", lambda: None, dependencies=("first",)),
        ]
    )

    assert results["first"].status == "failed"
    assert results["later"].status == "blocked"


def test_backend_release_runs_after_success_and_failure() -> None:
    class Backend:
        releases = 0

        def release(self) -> None:
            self.releases += 1

    success = Backend()
    failure = Backend()

    def fail() -> None:
        raise RuntimeError("boom")

    ResourceScheduler().run(
        [
            ScheduledTask("success", lambda: None, backend=success),
            ScheduledTask("failure", fail, backend=failure),
        ]
    )

    assert success.releases == 1
    assert failure.releases == 1


def test_active_gpu_budget_rejects_unknown_memory_estimate() -> None:
    results = ResourceScheduler(ResourceLimits(gpu_memory_budget_mb=4096)).run(
        [ScheduledTask("gpu", lambda: None, resource="gpu")]
    )

    assert results["gpu"].status == "failed"
    assert "unknown GPU memory usage" in str(results["gpu"].error)


def test_active_gpu_budget_accepts_known_estimate_within_budget() -> None:
    results = ResourceScheduler(ResourceLimits(gpu_memory_budget_mb=4096)).run(
        [
            ScheduledTask(
                "gpu",
                lambda: "ok",
                resource="gpu",
                gpu_memory_mb=2048,
            )
        ]
    )

    assert results["gpu"].status == "completed"
    assert results["gpu"].value == "ok"


def test_active_gpu_budget_rejects_estimate_over_budget() -> None:
    results = ResourceScheduler(ResourceLimits(gpu_memory_budget_mb=4096)).run(
        [
            ScheduledTask(
                "gpu",
                lambda: None,
                resource="gpu",
                gpu_memory_mb=6144,
            )
        ]
    )

    assert results["gpu"].status == "failed"
    assert "exceeds gpu memory budget" in str(results["gpu"].error)
