# -*- coding: utf-8 -*-
from typing import Optional, List, Callable, Tuple, NoReturn, Union

from ts_benchmark.utils.design_pattern import Singleton
from ts_benchmark.utils.parallel.base import TaskResult, SharedStorage
from ts_benchmark.utils.parallel.ray_backend import RayBackend
from ts_benchmark.utils.parallel.sequential_backend import SequentialBackend


__all__ = ["ParallelBackend", "SharedStorage"]


class ParallelBackend(metaclass=Singleton):
    #: all available backends
    BACKEND_DICT = {
        "ray": RayBackend,
        "sequential": SequentialBackend,
    }

    def __init__(self):
        self.backend = None
        self.default_timeout = None

    def init(
        self,
        backend: str = "ray",
        n_workers: Optional[int] = None,
        n_cpus: Optional[int] = None,
        gpu_devices: Optional[List[int]] = None,
        default_timeout: float = -1,
        max_tasks_per_child: Optional[int] = None,
        worker_initializers: Optional[Union[List[Callable], Callable]] = None,
    ):
        if backend not in self.BACKEND_DICT:
            raise ValueError(f"Unknown backend name {backend}")
        if self.backend is not None:
            raise RuntimeError("Please close the backend before re-initializing")
        self.backend = self.BACKEND_DICT[backend](
            n_workers=n_workers,
            n_cpus=n_cpus,
            gpu_devices=gpu_devices,
            max_tasks_per_child=max_tasks_per_child,
            worker_initializers=worker_initializers,
        )
        self.backend.init()
        self.default_timeout = default_timeout

    def schedule(
        self, fn: Callable, args: Tuple, timeout: Optional[float] = None
    ) -> TaskResult:
        if self.backend is None:
            raise RuntimeError(
                "Please initialize parallel backend before calling schedule"
            )
        if timeout is None:
            timeout = self.default_timeout
        return self.backend.schedule(fn, args, timeout)

    def close(self, force: bool = False):
        if self.backend is not None:
            self.backend.close(force)
            self.backend = None

    @property
    def shared_storage(self) -> SharedStorage:
        return self.backend.shared_storage

    def add_worker_initializer(self, func: Callable) -> NoReturn:
        self.backend.add_worker_initializer(func)

    def execute_on_workers(self, func: Callable) -> NoReturn:
        self.backend.execute_on_workers(func)
