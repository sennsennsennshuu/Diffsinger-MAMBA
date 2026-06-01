"""
CPU 并行化优化脚本
解决 GIL 锁和线程分配不均的问题
"""
import os
import sys
import multiprocessing as mp
import torch

def get_cpu_info():
    """获取 CPU 信息"""
    physical_cores = os.cpu_count() // 2  # 假设超线程比例为 2
    logical_cores = os.cpu_count()
    return physical_cores, logical_cores

def set_optimal_threads():
    """设置最优线程数"""
    physical_cores, logical_cores = get_cpu_info()
    
    # 设置 OpenMP 线程数（PyTorch 内部使用）
    # 使用物理核心数，避免超线程竞争
    optimal_threads = physical_cores
    
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)
    
    # 设置 PyTorch 线程数
    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(min(4, physical_cores))
    
    print(f"CPU 信息:")
    print(f"  物理核心: {physical_cores}")
    print(f"  逻辑核心: {logical_cores}")
    print(f"  OpenMP 线程数: {optimal_threads}")
    print(f"  PyTorch 线程数: {torch.get_num_threads()}")
    print(f"  PyTorch 互操作线程: {torch.get_num_interop_threads()}")
    
    return optimal_threads

def optimize_binarization_workers():
    """优化二值化工作进程数"""
    physical_cores, logical_cores = get_cpu_info()
    
    # 二值化使用多进程，每个进程使用单线程
    # 这样可以避免 GIL 锁和 OpenMP 线程竞争
    optimal_workers = min(physical_cores, 8)  # 最多 8 个 worker
    
    print(f"\n二值化优化:")
    print(f"  推荐 worker 数: {optimal_workers}")
    print(f"  每个 worker 线程数: 1")
    
    return optimal_workers

def create_optimized_config():
    """创建优化的配置"""
    physical_cores, logical_cores = get_cpu_info()
    
    # 训练配置
    train_config = {
        'max_batch_frames': 32000,
        'max_batch_size': 8,
        'ds_workers': min(physical_cores, 8),
        'accumulate_grad_batches': 2,
    }
    
    # 二值化配置
    binarize_config = {
        'num_workers': min(physical_cores, 8),
    }
    
    print(f"\n推荐配置:")
    print(f"  训练:")
    print(f"    max_batch_frames: {train_config['max_batch_frames']}")
    print(f"    max_batch_size: {train_config['max_batch_size']}")
    print(f"    ds_workers: {train_config['ds_workers']}")
    print(f"    accumulate_grad_batches: {train_config['accumulate_grad_batches']}")
    print(f"  二值化:")
    print(f"    num_workers: {binarize_config['num_workers']}")
    
    return train_config, binarize_config

if __name__ == "__main__":
    print("=" * 50)
    print("CPU 并行化优化分析")
    print("=" * 50)
    
    optimal_threads = set_optimal_threads()
    optimal_workers = optimize_binarization_workers()
    train_config, binarize_config = create_optimized_config()
    
    print("\n" + "=" * 50)
    print("优化建议:")
    print("=" * 50)
    print("1. 二值化阶段:")
    print(f"   - 使用 {optimal_workers} 个 worker 进程")
    print("   - 每个进程使用单线程（避免 GIL 竞争）")
    print("   - 已在 acoustic_binarizer.py 中设置 OMP_NUM_THREADS=1")
    print()
    print("2. 训练阶段:")
    print(f"   - PyTorch 使用 {optimal_threads} 个线程")
    print(f"   - DataLoader 使用 {train_config['ds_workers']} 个 worker")
    print("   - 使用梯度累积模拟大 batch")
    print()
    print("3. 环境变量设置:")
    print(f"   OMP_NUM_THREADS={optimal_threads}")
    print(f"   MKL_NUM_THREADS={optimal_threads}")
    print(f"   OPENBLAS_NUM_THREADS={optimal_threads}")
