#!/usr/bin/env python3
"""Test parallel processing on Windows."""

import sys
import multiprocessing
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Test parallel processing setup."""
    
    print("Windows Parallel Processing Test")
    print("="*60)
    
    # IMPORTANT: For Windows multiprocessing
    if sys.platform == 'win32':
        # Windows requires this for multiprocessing to work properly
        multiprocessing.freeze_support()
    
    # Test CPU detection
    print(f"Platform: {sys.platform}")
    print(f"CPU count: {multiprocessing.cpu_count()}")
    print()
    
    # Test configuration
    from abba.config import ABBAConfig
    
    config = ABBAConfig()
    print(f"Default parallel_workers: {config.parallel_workers}")
    print(f"get_parallel_workers(): {config.get_parallel_workers()}")
    print()
    
    # Test with different settings
    config.parallel_workers = 4
    print(f"After setting to 4: {config.get_parallel_workers()}")
    
    config.parallel_workers = None
    print(f"After setting to None: {config.get_parallel_workers()}")
    
    # Test parallel import
    print("\nTesting parallel import configuration...")
    from abba.parallel_import import ParallelImporter
    
    importer = ParallelImporter(
        source_db_path=Path("bible_data/bible.db"),
        dest_db_path=Path("bible_data/test.db"),
        max_workers=None  # Auto-detect
    )
    
    print(f"ParallelImporter max_workers: {importer.max_workers}")
    
    print("\n✓ All tests complete!")


if __name__ == "__main__":
    # Critical for Windows!
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    
    main()