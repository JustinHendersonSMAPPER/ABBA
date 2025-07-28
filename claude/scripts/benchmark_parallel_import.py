#!/usr/bin/env python3
"""Benchmark parallel import performance."""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.parallel_import import ParallelImporter, benchmark_import_methods
from abba.database.sqlite_manager import SQLiteManager


def main():
    """Run parallel import benchmarks."""
    
    # Paths
    source_db = Path("bible_data/bible.db")
    dest_db = Path("bible_data/abba.db")
    
    if not source_db.exists():
        print(f"Source database not found: {source_db}")
        return
    
    # Test with a subset of translations
    test_translations = ["KJV", "ESV", "NIV", "NASB", "NKJV"]
    
    print("="*60)
    print("Parallel Import Performance Test")
    print("="*60)
    print(f"Testing with {len(test_translations)} translations")
    print(f"CPU count: {import multiprocessing; multiprocessing.cpu_count()}")
    print()
    
    # Quick test with different worker counts
    importer = ParallelImporter(source_db, dest_db)
    
    # Test 1: Sequential (baseline)
    print("Test 1: Sequential import (baseline)")
    start = time.time()
    results_seq = {}
    for tid in test_translations[:2]:  # Just 2 for quick test
        job = ImportJob(tid, str(source_db), str(dest_db))
        results_seq[tid] = ParallelImporter._import_single_translation(job)
    seq_time = time.time() - start
    seq_verses = sum(r.verse_count for r in results_seq.values())
    print(f"  Time: {seq_time:.2f}s")
    print(f"  Verses: {seq_verses:,}")
    print(f"  Rate: {seq_verses/seq_time:.0f} verses/sec")
    print()
    
    # Test 2: Parallel with threads
    print("Test 2: Parallel import with threads")
    start = time.time()
    results_thread = importer.import_translations_parallel(
        test_translations[:2],
        use_processes=False,
        show_progress=True
    )
    thread_time = time.time() - start
    thread_verses = sum(r.verse_count for r in results_thread.values())
    print(f"  Time: {thread_time:.2f}s")
    print(f"  Verses: {thread_verses:,}")
    print(f"  Rate: {thread_verses/thread_time:.0f} verses/sec")
    print(f"  Speedup: {seq_time/thread_time:.2f}x")
    print()
    
    # Test 3: Parallel with processes
    print("Test 3: Parallel import with processes")
    start = time.time()
    results_proc = importer.import_translations_parallel(
        test_translations[:2],
        use_processes=True,
        show_progress=True
    )
    proc_time = time.time() - start
    proc_verses = sum(r.verse_count for r in results_proc.values())
    print(f"  Time: {proc_time:.2f}s")
    print(f"  Verses: {proc_verses:,}")
    print(f"  Rate: {proc_verses/proc_time:.0f} verses/sec")
    print(f"  Speedup: {seq_time/proc_time:.2f}x")
    print()
    
    # Full benchmark if requested
    if "--full" in sys.argv:
        print("="*60)
        print("Running full benchmark...")
        print("="*60)
        
        results = benchmark_import_methods(
            source_db,
            dest_db,
            test_translations
        )
        
        # Print results table
        print(f"\n{'Method':<15} {'Time':<10} {'Verses/s':<12} {'Speedup':<10}")
        print("-" * 50)
        
        baseline = None
        for method, data in results.items():
            if baseline is None:
                baseline = data['duration']
                speedup = 1.0
            else:
                speedup = baseline / data['duration']
            
            print(f"{method:<15} {data['duration']:<10.2f} "
                  f"{data['verses_per_second']:<12.0f} {speedup:<10.2f}x")
        
        # Find optimal configuration
        best_method = max(results.items(), key=lambda x: x[1]['verses_per_second'])
        print(f"\nBest method: {best_method[0]}")
        print(f"Performance: {best_method[1]['verses_per_second']:.0f} verses/second")


if __name__ == "__main__":
    import multiprocessing
    from abba.parallel_import import ImportJob
    
    main()