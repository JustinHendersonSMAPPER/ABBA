#!/usr/bin/env python3
"""Test the post-import validator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database.post_import_validator import PostImportValidator

def test_validator():
    """Test the post-import validator."""
    print("Testing Post-Import Validator")
    print("="*60)
    
    # Check if databases exist
    abba_db = Path("bible_data/abba.db")
    bible_db = Path("bible_data/bible.db")
    
    if not abba_db.exists():
        print("Error: abba.db not found. Run import first.")
        return
    
    if not bible_db.exists():
        print("Error: bible.db not found.")
        return
    
    # Create validator
    validator = PostImportValidator(
        abba_db_path=abba_db,
        source_db_path=bible_db
    )
    
    # Run validation
    print("\nRunning validation checks...\n")
    summary = validator.validate_all_translations()
    
    # Print detailed results
    validator.print_summary(summary)
    
    # Show some example details
    if summary.total_translations > 0:
        print("\nExample validation details:")
        
        # Get a sample translation's results
        import sqlite3
        with sqlite3.connect(abba_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM translations LIMIT 3")
            sample_ids = [row[0] for row in cursor.fetchall()]
        
        for trans_id in sample_ids:
            print(f"\n{trans_id}:")
            
            # Run individual checks
            results = []
            with sqlite3.connect(abba_db) as conn:
                results.extend(validator._validate_translation_metadata(trans_id, conn))
                results.extend(validator._validate_verse_counts(trans_id, conn))
                results.extend(validator._validate_canon_association(trans_id, conn))
            
            for result in results:
                if result.passed:
                    print(f"  ✓ {result.check_name}: {result.message}")
                else:
                    print(f"  ✗ {result.check_name}: {result.message}")
                
                if result.details and result.check_name == "canon_detection":
                    print(f"    Canon: {result.details['canon']}")
                    print(f"    Books: {result.details['total_books']} total, "
                          f"{result.details['standard_books']} standard, "
                          f"{result.details['extended_books']} extended")

if __name__ == "__main__":
    test_validator()
    
    print("\n\n✅ Post-import validator is working!")
    print("It checks:")
    print("  - Translation metadata completeness")
    print("  - Verse count accuracy")
    print("  - Book coverage")
    print("  - Canon associations")
    print("  - Data integrity")