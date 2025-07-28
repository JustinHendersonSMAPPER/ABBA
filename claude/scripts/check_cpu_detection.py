#!/usr/bin/env python3
"""Check CPU detection on the system."""

import multiprocessing
import os
import platform
import psutil


def main():
    """Check various methods of CPU detection."""
    
    print("CPU Detection Test")
    print("="*60)
    
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")
    print()
    
    # Method 1: multiprocessing.cpu_count()
    print("Method 1: multiprocessing.cpu_count()")
    try:
        cpu_count = multiprocessing.cpu_count()
        print(f"  CPU count: {cpu_count}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Method 2: os.cpu_count()
    print("\nMethod 2: os.cpu_count()")
    try:
        cpu_count = os.cpu_count()
        print(f"  CPU count: {cpu_count}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Method 3: psutil (if available)
    print("\nMethod 3: psutil")
    try:
        print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"  Logical cores: {psutil.cpu_count(logical=True)}")
        print(f"  CPU percent: {psutil.cpu_percent(interval=1)}%")
        
        # CPU frequency
        freq = psutil.cpu_freq()
        if freq:
            print(f"  Current frequency: {freq.current:.0f} MHz")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Method 4: Environment variables (Windows)
    print("\nMethod 4: Environment variables")
    if platform.system() == "Windows":
        num_processors = os.environ.get('NUMBER_OF_PROCESSORS')
        print(f"  NUMBER_OF_PROCESSORS: {num_processors}")
    
    # Test parallel execution
    print("\nParallel Execution Test")
    print("-"*40)
    
    # Test with ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import time
    
    def cpu_bound_task(n):
        """Simple CPU-bound task."""
        total = 0
        for i in range(n):
            total += i * i
        return total
    
    # Test different worker counts
    test_sizes = [1, 2, 4, 8]
    n = 1000000
    
    print(f"Testing CPU-bound task with {n:,} iterations...")
    
    for workers in test_sizes:
        if workers > multiprocessing.cpu_count():
            continue
            
        start = time.time()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(cpu_bound_task, n) for _ in range(workers)]
            results = [f.result() for f in futures]
        duration = time.time() - start
        
        print(f"  {workers} workers: {duration:.2f}s")
    
    print("\nRecommendation:")
    print(f"  Detected cores: {multiprocessing.cpu_count()}")
    print(f"  Recommended workers for I/O: {multiprocessing.cpu_count() * 2}")
    print(f"  Recommended workers for CPU: {multiprocessing.cpu_count()}")


if __name__ == "__main__":
    main()