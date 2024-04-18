# -*- coding: utf-8 -*-
from __future__ import absolute_import

import itertools
import logging
import os
import queue
import sys
import threading
import time
from typing import Callable, Tuple, Any, List, NoReturn, Optional, Dict, Union

import ray
from ray import ObjectRef
from ray.actor import ActorHandle
from ray.exceptions import RayActorError

from ts_benchmark.utils.parallel.base import TaskResult, SharedStorage

logger = logging.getLogger(__name__)


def is_actor() -> bool:
    """
    判断当前是否正在 actor 中运行
    """
    return ray.get_runtime_context().worker.mode == ray.WORKER_MODE


class RayActor:
    def __init__(self, env: Dict, initializers: Optional[List[Callable]] = None):
        self._idle = True
        self._start_time = None
        if initializers is not None:
            for func in initializers:
                func(env)

    def run(self, fn: Callable, args: Tuple) -> Any:
        self._start_time = time.time()
        self._idle = False
        res = fn(*args)
        self._idle = True
        return res

    def start_time(self) -> Optional[float]:
        return None if self._idle or self._start_time is None else self._start_time


@ray.remote(max_restarts=-1)
class ObjectRefStorageActor:
    def __init__(self):
        self.storage = {}

    def put(self, name: str, value: List[ObjectRef]) -> NoReturn:
        self.storage[name] = value

    def get(self, name: str) -> Optional[List[ObjectRef]]:
        return self.storage.get(name)


class RaySharedStorage(SharedStorage):
    def __init__(self, object_ref_actor):
        self.object_ref_actor = object_ref_actor

    def put(self, name: str, value: Any) -> NoReturn:
        if is_actor():
            raise RuntimeError("put is not supported to be called by actors")
        obj_ref = ray.put(value)
        ray.get(self.object_ref_actor.put.remote(name, [obj_ref]))

    def get(self, name: str, default_value: Any = None) -> Any:
        obj_ref = ray.get(self.object_ref_actor.get.remote(name))
        if obj_ref is None:
            logger.info("data '%s' does not exist in shared storage", name)
            return default_value
        return ray.get(obj_ref[0])


class RayResult(TaskResult):
    __slots__ = ["_event", "_result"]

    def __init__(self, event: threading.Event):
        self._event = event
        self._result = None

    def put(self, value: Any) -> NoReturn:
        self._result = value
        self._event.set()

    def result(self) -> Any:
        self._event.wait()
        if isinstance(self._result, Exception):
            raise self._result
        else:
            return self._result


class RayTask:
    __slots__ = ["result", "actor_id", "timeout", "start_time"]

    def __init__(
        self, result: Any = None, actor_id: Optional[int] = None, timeout: float = -1
    ):
        self.result = result
        self.actor_id = actor_id
        self.timeout = timeout
        self.start_time = None


class RayActorPool:
    """
    ray actor 资源池

    和 ray 的内置 ActorPool 不同，本实现试图支持为每个任务限时
    """

    def __init__(
        self,
        n_workers: int,
        env: Dict,
        per_worker_resources: Optional[Dict] = None,
        max_tasks_per_child: Optional[int] = None,
        worker_initializers: Optional[List[Callable]] = None,
    ):
        if per_worker_resources is None:
            per_worker_resources = {}

        self.env = env
        self.per_worker_resources = per_worker_resources
        self.max_tasks_per_child = max_tasks_per_child
        self.worker_initializers = worker_initializers
        self.actor_class = ray.remote(
            max_restarts=0,
            num_cpus=per_worker_resources.get("num_cpus", 1),
            num_gpus=per_worker_resources.get("num_gpus", 0),
        )(RayActor)
        self.actors = [self._new_actor() for _ in range(n_workers)]

        # these data are only accessed in the main thread
        self._task_counter = itertools.count()

        # these data are only accessed in the loop thread
        self._task_info = {}
        self._ray_task_to_id = {}
        self._active_tasks = []
        self._idle_actors = list(range(len(self.actors)))
        self._restarting_actor_pool = {}
        self._actor_tasks = [0] * len(self.actors)

        # message path between threads
        self._is_closed = False
        self._idle_event = threading.Event()
        self._pending_queue = queue.Queue(maxsize=1000000)

        self._main_loop_thread = threading.Thread(target=self._main_loop)
        self._main_loop_thread.start()

    def _new_actor(self) -> ActorHandle:
        # TODO: On the Windows platform, when GPU resources are allocated in the actors,
        #  the ray tasks sometimes terminate ungracefully, leaving the concurrency counter in a wrong state
        #  As a temporary workaround, we are removing the limit of max_concurrency in this case,
        #  which may lead to unexpected overhead.
        max_concurrency = (
            100
            if sys.platform == "win32"
            and self.per_worker_resources.get("num_gpus", 0) > 0
            else 2
        )
        handle = self.actor_class.options(max_concurrency=max_concurrency).remote(
            self.env,
            self.worker_initializers,
        )

        return handle

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> RayResult:
        self._idle_event.clear()
        task_id = next(self._task_counter)
        result = RayResult(threading.Event())
        self._pending_queue.put((fn, args, timeout, task_id, result), block=True)
        return result

    def _handle_ready_tasks(self, tasks: List) -> NoReturn:
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            try:
                task_info.result.put(ray.get(task_obj))
            except RayActorError as e:
                logger.info(
                    "task %d died unexpectedly on actor %d: %s",
                    task_id,
                    task_info.actor_id,
                    e,
                )
                task_info.result.put(RuntimeError(f"task died unexpectedly: {e}"))
                self._restart_actor(task_info.actor_id)
                del self._task_info[task_id]
                del self._ray_task_to_id[task_obj]
                continue

            self._actor_tasks[task_info.actor_id] += 1
            if (
                self.max_tasks_per_child is not None
                and self._actor_tasks[task_info.actor_id] >= self.max_tasks_per_child
            ):
                logger.info(
                    "max_tasks_per_child reached in actor %s, restarting",
                    task_info.actor_id,
                )
                self._restart_actor(task_info.actor_id)
            else:
                self._idle_actors.append(task_info.actor_id)
            del self._task_info[task_id]
            del self._ray_task_to_id[task_obj]

    def _get_duration(self, task_info: RayTask) -> Optional[float]:
        if task_info.start_time is None:
            try:
                task_info.start_time = ray.get(
                    self.actors[task_info.actor_id].start_time.remote()
                )
            except RayActorError as e:
                logger.info(
                    "actor %d died unexpectedly: %s, restarting...",
                    task_info.actor_id,
                    e,
                )
                return None

        return (
            -1 if task_info.start_time is None else time.time() - task_info.start_time
        )

    def _handle_unfinished_tasks(self, tasks: List) -> NoReturn:
        new_active_tasks = []
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            duration = self._get_duration(task_info)
            if duration is None or 0 < task_info.timeout < duration:
                if duration is not None:
                    logger.info(
                        "actor %d killed after timeout %f",
                        task_info.actor_id,
                        task_info.timeout,
                    )
                self._restart_actor(task_info.actor_id)
                task_info.result.put(
                    TimeoutError(f"time limit exceeded: {task_info.timeout}")
                )
                del self._task_info[task_id]
                del self._ray_task_to_id[task_obj]
            else:
                new_active_tasks.append(task_obj)
        self._active_tasks = new_active_tasks

    def _restart_actor(self, actor_id: int) -> NoReturn:
        cur_actor = self.actors[actor_id]
        ray.kill(cur_actor, no_restart=True)
        del cur_actor
        self.actors[actor_id] = self._new_actor()
        self._actor_tasks[actor_id] = 0
        self._restarting_actor_pool[actor_id] = time.time()

    def _check_restarting_actors(self):
        new_restarting_pool = {}
        for actor_id, restart_time in self._restarting_actor_pool.items():
            if time.time() - restart_time > 5:
                ready_tasks = ray.wait(
                    [self.actors[actor_id].start_time.remote()], timeout=0.5
                )[0]
                if ready_tasks:
                    logger.debug("restarted actor %d is now ready", actor_id)
                    self._idle_actors.append(actor_id)
                    continue
                else:
                    logger.debug(
                        "restarted actor %d is not ready, resetting timer", actor_id
                    )
                    self._restarting_actor_pool[actor_id] = time.time()
            new_restarting_pool[actor_id] = restart_time
        self._restarting_actor_pool = new_restarting_pool

    def _main_loop(self) -> NoReturn:
        while not self._is_closed:
            self._check_restarting_actors()

            logger.debug(
                "%d active tasks, %d idle actors, %d restarting actors",
                len(self._active_tasks),
                len(self._idle_actors),
                len(self._restarting_actor_pool),
            )

            if not self._active_tasks and self._pending_queue.empty():
                self._idle_event.set()
                time.sleep(1)
                continue

            if self._active_tasks:
                ready_tasks, unfinished_tasks = ray.wait(self._active_tasks, timeout=1)
                self._handle_ready_tasks(ready_tasks)
                self._handle_unfinished_tasks(unfinished_tasks)
            else:
                time.sleep(1)

            while self._idle_actors and not self._pending_queue.empty():
                fn, args, timeout, task_id, result = self._pending_queue.get_nowait()
                cur_actor = self._idle_actors.pop()
                task_obj = self.actors[cur_actor].run.remote(fn, args)
                self._task_info[task_id] = RayTask(
                    result=result, actor_id=cur_actor, timeout=timeout
                )
                self._ray_task_to_id[task_obj] = task_id
                self._active_tasks.append(task_obj)
                logger.debug("task %d assigned to actor %d", task_id, cur_actor)

    def wait(self) -> NoReturn:
        if self._is_closed:
            return
        if self._pending_queue.empty() and not self._active_tasks:
            return
        self._idle_event.clear()
        self._idle_event.wait()
        while self._restarting_actor_pool:
            time.sleep(1)

    def close(self) -> NoReturn:
        self._is_closed = True
        for actor in self.actors:
            ray.kill(actor)
        self._main_loop_thread.join()


class RayBackend:
    def __init__(
        self,
        n_workers: Optional[int] = None,
        n_cpus: Optional[int] = None,
        gpu_devices: Optional[List[int]] = None,
        max_tasks_per_child: Optional[int] = None,
        worker_initializers: Optional[Union[List[Callable], Callable]] = None,
    ):
        self.n_cpus = n_cpus if n_cpus is not None else os.cpu_count()
        self.n_workers = n_workers if n_workers is not None else self.n_cpus
        self.gpu_devices = gpu_devices if gpu_devices is not None else []
        self.max_tasks_per_child = max_tasks_per_child
        self.worker_initializers = (
            worker_initializers
            if isinstance(worker_initializers, list)
            else [worker_initializers]
        )
        self.pool = None
        self._storage = None
        self.initialized = False

    def init(self) -> NoReturn:
        if self.initialized:
            return

        cpu_per_worker = self._get_cpus_per_worker(self.n_cpus, self.n_workers)
        gpu_per_worker, gpu_devices = self._get_gpus_per_worker(
            self.gpu_devices, self.n_workers
        )

        if not ray.is_initialized():
            # in the main process
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
            ray.init(num_cpus=self.n_cpus, num_gpus=len(gpu_devices))
        else:
            raise RuntimeError("init is not allowed to be called in ray actors")

        # 'put' can only be called in the main process now,
        # so the storage is safe to be accessed in multiple processes
        self._storage = RaySharedStorage(ObjectRefStorageActor.remote())

        self.pool = RayActorPool(
            self.n_workers,
            self.env,
            {
                "num_cpus": cpu_per_worker,
                "num_gpus": gpu_per_worker,
            },
            max_tasks_per_child=self.max_tasks_per_child,
            worker_initializers=self.worker_initializers,
        )
        self.initialized = True

    def add_worker_initializer(self, func: Callable) -> NoReturn:
        # TODO: check if the pool is idle before updating initializer
        if self.worker_initializers is None:
            self.worker_initializers = []
        self.worker_initializers.append(func)
        self.pool.worker_initializers = self.worker_initializers

    @property
    def env(self) -> Dict:
        return {
            "storage": self.shared_storage,
        }

    def _get_cpus_per_worker(self, n_cpus: int, n_workers: int) -> Union[int, float]:
        if n_cpus > n_workers and n_cpus % n_workers != 0:
            cpus_per_worker = n_cpus // n_workers
            logger.info(
                "only %d among %d cpus are used to match the number of workers",
                cpus_per_worker * n_workers,
                n_cpus,
            )
        else:
            cpus_per_worker = n_cpus / n_workers
        return cpus_per_worker

    def _get_gpus_per_worker(
        self, gpu_devices: List[int], n_workers: int
    ) -> Tuple[Union[int, float], List[int]]:
        n_gpus = len(gpu_devices)
        if n_gpus > n_workers and n_gpus % n_workers != 0:
            gpus_per_worker = n_gpus // n_workers
            used_gpu_devices = gpu_devices[: gpus_per_worker * n_workers]
            logger.info(
                "only %s gpus are used to match the number of workers", used_gpu_devices
            )
        else:
            gpus_per_worker = n_gpus / n_workers
            used_gpu_devices = gpu_devices
        return gpus_per_worker, used_gpu_devices

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> RayResult:
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        return self.pool.schedule(fn, args, timeout)

    def close(self, force: bool = False) -> NoReturn:
        if not self.initialized:
            return

        if not force:
            self.pool.wait()
        self.pool.close()
        ray.shutdown()

    @property
    def shared_storage(self) -> RaySharedStorage:
        return self._storage

    def execute_on_workers(self, func: Callable) -> NoReturn:
        """
        execute function on all workers when the pool is in idle mode
        """
        # TODO: sharing data while there are active tasks is not yet supported
        self.pool.wait()

        # TODO: fix encapsulation problem by implementing pool.map
        tasks = []
        for actor in self.pool.actors:
            tasks.append(actor.run.remote(func, (self.env,)))
        ray.wait(tasks, num_returns=len(tasks))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    backend = RayBackend(3, max_tasks_per_child=1)
    backend.init()

    def sleep_func(t):
        time.sleep(t)
        print(f"sleep after {t}")
        return t

    results = []
    results.append(backend.schedule(sleep_func, (10,), timeout=5))
    results.append(backend.schedule(sleep_func, (10,), timeout=20))
    results.append(backend.schedule(sleep_func, (1,), timeout=5))
    results.append(backend.schedule(sleep_func, (2,), timeout=5))
    results.append(backend.schedule(sleep_func, (3,), timeout=5))
    results.append(backend.schedule(sleep_func, (4,), timeout=5))
    results.append(backend.schedule(sleep_func, (5,), timeout=5))
    results.append(backend.schedule(sleep_func, (6,), timeout=5))

    for i, res in enumerate(results):
        try:
            print(f"{i}-th task result: {res.result()}")
        except TimeoutError:
            print(f"{i}-th task fails after timeout")

    backend.close()
    # time.sleep(100)
    # time.sleep(1)
    # pool.wait()
    # pool.close()
