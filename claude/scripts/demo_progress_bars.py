#!/usr/bin/env python3
"""Demo script to show progress bars during import."""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.parallel_import import ParallelImporter
from abba.database.sqlite_manager import SQLiteManager
from abba.operation_manager import OperationManager


def main():
    """Demo progress bars during import."""
    
    # Paths
    source_db = Path("bible_data/bible.db")
    dest_db = Path("bible_data/abba_demo.db")
    state_file = Path("bible_data/.abba_state_demo.json")
    
    if not source_db.exists():
        print(f"Source database not found: {source_db}")
        print("Please run main.py first to download bible.db")
        return
    
    # Initialize components
    print("="*60)
    print("Progress Bar Demo")
    print("="*60)
    
    # Initialize database
    db_manager = SQLiteManager(dest_db)
    db_manager.initialize_database()
    
    # Initialize operation manager
    op_manager = OperationManager(state_file, source_db)
    
    # Initialize parallel importer
    importer = ParallelImporter(
        source_db_path=source_db,
        dest_db_path=dest_db,
        operation_manager=op_manager,
        max_workers=4  # Use 4 workers for demo
    )
    
    # Test translations
    test_translations = ["KJV", "ESV", "NIV"]
    
    print(f"\nDemo 1: Single translation import with detailed progress")
    print("-"*60)
    
    # Import single translation to show detailed progress
    results = importer.import_translations_parallel(
        translation_ids=["KJV"],
        use_processes=False,  # Use threads for I/O
        show_progress=True
    )
    
    if results["KJV"].success:
        print(f"\n✓ Successfully imported KJV: {results['KJV'].verse_count:,} verses")
    
    print(f"\n\nDemo 2: Multiple translations with overall progress")
    print("-"*60)
    
    # Import multiple translations
    results = importer.import_translations_parallel(
        translation_ids=["ESV", "NIV"],
        use_processes=False,
        show_progress=True
    )
    
    # Summary
    print(f"\n\nImport Summary:")
    print("-"*30)
    for tid, result in results.items():
        if result.success:
            rate = result.verse_count / result.duration if result.duration > 0 else 0
            print(f"✓ {tid}: {result.verse_count:,} verses in {result.duration:.1f}s ({rate:.0f} verses/sec)")
        else:
            print(f"✗ {tid}: Failed - {result.error}")
    
    print(f"\n\nDemo 3: Verification with progress")
    print("-"*60)
    
    # Create extractor for verification
    from abba.bible_extractor import BibleExtractor
    from abba.config import ABBAConfig
    
    config = ABBAConfig()
    config.parallel_workers = 4
    
    extractor = BibleExtractor(str(source_db.parent), config=config)
    
    # Verify imports
    verify_results = extractor.verify_import_parallel(
        db_manager=db_manager,
        translation_ids=test_translations
    )
    
    print(f"\n\nDemo 4: Sequential vs Parallel comparison")
    print("-"*60)
    
    # Test translation for comparison
    test_tid = "NKJV"
    
    # Sequential
    print("\nSequential import (1 worker):")
    start = time.time()
    seq_importer = ParallelImporter(source_db, dest_db, op_manager, max_workers=1)
    seq_results = seq_importer.import_translations_parallel(
        [test_tid], use_processes=False, show_progress=True
    )
    seq_time = time.time() - start
    
    # Parallel
    print("\nParallel import (4 workers):")
    # Note: For single translation, parallel won't help much
    # This is just for demo purposes
    
    # Clean up demo database
    if dest_db.exists():
        dest_db.unlink()
    if state_file.exists():
        state_file.unlink()
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()