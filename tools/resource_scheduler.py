from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from tools.backend_registry import release_backend

ResourceKind = Literal["cpu", "gpu"]
TaskStatus = Literal["completed", "failed", "blocked"]


@dataclass(frozen=True)
class ResourceLimits:
    max_cpu_workers: int = 4
    max_gpu_workers: int = 1
    gpu_memory_budget_mb: int = 0
    model_exclusive_lock: bool = True

    def __post_init__(self) -> None:
        if self.max_cpu_workers <= 0:
            raise ValueError("max_cpu_workers must be positive")
        if self.max_gpu_workers <= 0:
            raise ValueError("max_gpu_workers must be positive")
        if self.gpu_memory_budget_mb < 0:
            raise ValueError("gpu_memory_budget_mb must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ResourceLimits:
        raw = dict(value or {})
        return cls(
            max_cpu_workers=int(raw.get("max_cpu_workers", 4)),
            max_gpu_workers=int(raw.get("max_gpu_workers", 1)),
            gpu_memory_budget_mb=int(raw.get("gpu_memory_budget_mb", 0)),
            model_exclusive_lock=bool(raw.get("model_exclusive_lock", True)),
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "max_cpu_workers": self.max_cpu_workers,
            "max_gpu_workers": self.max_gpu_workers,
            "gpu_memory_budget_mb": self.gpu_memory_budget_mb,
            "model_exclusive_lock": self.model_exclusive_lock,
        }


@dataclass(frozen=True)
class ScheduledTask:
    name: str
    operation: Callable[[], Any]
    resource: ResourceKind = "cpu"
    dependencies: tuple[str, ...] = ()
    backend: object | None = None
    gpu_memory_mb: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scheduled task name must be non-empty")
        if self.gpu_memory_mb < 0:
            raise ValueError("gpu_memory_mb must be non-negative")


@dataclass
class TaskResult:
    name: str
    status: TaskStatus
    value: Any = None
    error: str | None = None


class ResourceScheduler:
    """Dependency-aware CPU/GPU scheduler with an optional global model lock."""

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self._model_lock = threading.Lock()

    def _invoke(self, task: ScheduledTask) -> Any:
        if task.resource == "gpu" and self.limits.gpu_memory_budget_mb and task.gpu_memory_mb == 0:
            raise RuntimeError(
                f"task {task.name} has unknown GPU memory usage while a budget is active; "
                "provide an estimated gpu_memory_mb"
            )
        if (
            task.resource == "gpu"
            and self.limits.gpu_memory_budget_mb
            and task.gpu_memory_mb > self.limits.gpu_memory_budget_mb
        ):
            raise RuntimeError(
                f"task {task.name} exceeds gpu memory budget: "
                f"{task.gpu_memory_mb} > {self.limits.gpu_memory_budget_mb} MB"
            )
        lock = (
            self._model_lock
            if task.resource == "gpu" and self.limits.model_exclusive_lock
            else None
        )
        try:
            if lock is None:
                return task.operation()
            with lock:
                return task.operation()
        finally:
            if task.backend is not None:
                release_backend(task.backend)

    def run(self, tasks: Sequence[ScheduledTask]) -> dict[str, TaskResult]:
        by_name = {task.name: task for task in tasks}
        if len(by_name) != len(tasks):
            raise ValueError("scheduled task names must be unique")
        unknown = {
            dependency
            for task in tasks
            for dependency in task.dependencies
            if dependency not in by_name
        }
        if unknown:
            raise ValueError(f"unknown task dependencies: {sorted(unknown)}")

        pending = set(by_name)
        results: dict[str, TaskResult] = {}
        with (
            ThreadPoolExecutor(max_workers=self.limits.max_cpu_workers) as cpu_executor,
            ThreadPoolExecutor(max_workers=self.limits.max_gpu_workers) as gpu_executor,
        ):
            while pending:
                for name in sorted(pending):
                    task = by_name[name]
                    if any(
                        dependency in results
                        and results[dependency].status in {"failed", "blocked"}
                        for dependency in task.dependencies
                    ):
                        results[name] = TaskResult(
                            name,
                            "blocked",
                            error="a dependency did not complete",
                        )
                pending -= set(results)
                ready = [
                    by_name[name]
                    for name in sorted(pending)
                    if all(
                        dependency in results and results[dependency].status == "completed"
                        for dependency in by_name[name].dependencies
                    )
                ]
                if not ready:
                    if pending:
                        raise ValueError(f"scheduled task dependency cycle: {sorted(pending)}")
                    break
                futures: dict[Future[Any], ScheduledTask] = {}
                for task in ready:
                    executor = gpu_executor if task.resource == "gpu" else cpu_executor
                    futures[executor.submit(self._invoke, task)] = task
                for future, task in futures.items():
                    try:
                        results[task.name] = TaskResult(
                            task.name,
                            "completed",
                            value=future.result(),
                        )
                    except Exception as exc:
                        results[task.name] = TaskResult(
                            task.name,
                            "failed",
                            error=str(exc),
                        )
                pending -= {task.name for task in ready}
        return results

    def run_stage(
        self,
        name: str,
        operation: Callable[[], Any],
        *,
        resource: ResourceKind = "cpu",
        backend: object | None = None,
        gpu_memory_mb: int = 0,
    ) -> Any:
        result = self.run(
            [
                ScheduledTask(
                    name,
                    operation,
                    resource=resource,
                    backend=backend,
                    gpu_memory_mb=gpu_memory_mb,
                )
            ]
        )[name]
        if result.status != "completed":
            raise RuntimeError(result.error or f"stage {name} did not complete")
        return result.value


__all__ = [
    "ResourceLimits",
    "ResourceScheduler",
    "ScheduledTask",
    "TaskResult",
]
