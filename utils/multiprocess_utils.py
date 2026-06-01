import os
import platform
import re
import traceback

import torch
from torch.multiprocessing import Manager, Process, current_process, get_context

is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))


def main_process_print(self, *args, sep=' ', end='\n', file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)


def _get_mp_context():
    """获取 multiprocessing context：CUDA 环境必须用 spawn"""
    if torch.cuda.is_available():
        return get_context('spawn')
    if platform.system().lower() == 'windows':
        return get_context('spawn')
    return get_context('fork')


def chunked_worker_run(map_func, args, results_queue=None):
    for a in args:
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            results_queue.put(None)


def chunked_multiprocess_run(map_func, args, num_workers, q_max_size=1000):
    num_jobs = len(args)
    if num_jobs < num_workers:
        num_workers = num_jobs

    # 使用 Manager().Queue()：Manager 单独 spawn 管理进程，不涉及 CUDA，避免 Segfault
    queues = [Manager().Queue(maxsize=q_max_size // num_workers) for _ in range(num_workers)]
    ctx = _get_mp_context()

    workers = []
    for i in range(num_workers):
        worker = ctx.Process(
            target=chunked_worker_run, args=(map_func, args[i::num_workers], queues[i]), daemon=True
        )
        workers.append(worker)
        worker.start()

    for i in range(num_jobs):
        yield queues[i % num_workers].get()

    for worker in workers:
        worker.join()
        worker.close()


def chunked_worker_run_optimized(map_func, args, results_queue, init_func=None, init_args=None):
    """优化版 worker：支持初始化函数"""
    if init_func is not None:
        if init_args is not None:
            init_func(*init_args)
        else:
            init_func()

    for a in args:
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            results_queue.put(None)


def chunked_multiprocess_run_v2(map_func, args, num_workers,
                                  init_func=None, init_args=None):
    """
    优化版多进程并行（基于原版 Manager().Queue()，兼容 CUDA）：
    1. 使用 Manager().Queue() 避免 Segfault
    2. 自动选择 spawn/fork context（CUDA 必须用 spawn）
    3. 支持 per-worker 初始化函数（预加载模型）
    4. 按块分配任务（而非轮询），负载更均匀
    """
    num_jobs = len(args)
    if num_workers < 1:
        num_workers = 1
    if num_jobs < num_workers:
        num_workers = num_jobs

    ctx = _get_mp_context()
    # 使用 Manager().Queue()：Manager 单独 spawn 管理进程，不涉及 CUDA，避免 Segfault
    queues = [Manager().Queue(maxsize=max(1, 1000 // num_workers)) for _ in range(num_workers)]

    # 按块分配任务（而非轮询），减少负载不均
    chunk_size = (num_jobs + num_workers - 1) // num_workers
    workers = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_jobs)
        if start_idx >= num_jobs:
            break
        worker_args = args[start_idx:end_idx]
        worker = ctx.Process(
            target=chunked_worker_run_optimized,
            args=(map_func, worker_args, queues[i], init_func, init_args),
            daemon=True
        )
        workers.append(worker)
        worker.start()

    # 轮询收集结果
    for i in range(num_jobs):
        yield queues[i % num_workers].get()

    for worker in workers:
        worker.join(timeout=5)
        worker.close()
