from __future__ import annotations

import threading
import time

from tools.resource_scheduler import ResourceLimits, ResourceScheduler, ScheduledTask


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
