#!/usr/bin/env python3
"""Test script to debug progress bar issues."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.parallel_import import ParallelImporter
from abba.database.sqlite_manager import SQLiteManager
from abba.config import ABBAConfig


def main():
    """Test progress bars during import."""
    
    # Paths
    source_db = Path("bible_data/bible.db")
    dest_db = Path("bible_data/test_progress.db")
    
    if not source_db.exists():
        print(f"Source database not found: {source_db}")
        return
    
    print("Testing progress bars for Bible import...")
    print("="*60)
    
    # Initialize database
    db_manager = SQLiteManager(dest_db)
    db_manager.initialize_database()
    
    # Test 1: Direct parallel importer
    print("\nTest 1: Direct ParallelImporter with single translation")
    importer = ParallelImporter(
        source_db_path=source_db,
        dest_db_path=dest_db,
        max_workers=None  # Auto-detect
    )
    
    results = importer.import_translations_parallel(
        translation_ids=["KJV"],
        use_processes=False,
        show_progress=True
    )
    
    print(f"\nResult: {results['KJV'].success}, Verses: {results['KJV'].verse_count}")
    
    # Test 2: With config
    print("\n\nTest 2: With config (parallel_workers=4)")
    config = ABBAConfig()
    config.parallel_workers = 4
    
    importer2 = ParallelImporter(
        source_db_path=source_db,
        dest_db_path=dest_db,
        max_workers=config.get_parallel_workers()
    )
    
    results2 = importer2.import_translations_parallel(
        translation_ids=["ESV"],
        use_processes=False,
        show_progress=True
    )
    
    print(f"\nResult: {results2['ESV'].success}, Verses: {results2['ESV'].verse_count}")
    
    # Test 3: Multiple translations
    print("\n\nTest 3: Multiple translations")
    results3 = importer2.import_translations_parallel(
        translation_ids=["NIV", "NASB"],
        use_processes=False,
        show_progress=True
    )
    
    for tid, result in results3.items():
        print(f"{tid}: {result.success}, Verses: {result.verse_count}")
    
    # Clean up
    if dest_db.exists():
        dest_db.unlink()
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()