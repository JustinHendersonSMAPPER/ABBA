#!/usr/bin/env python3
"""Analyze how the current import handles verse numbering issues."""

import sqlite3
from pathlib import Path
import sys

# Add the abba package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "abba"))

from database.sqlite_manager import SQLiteManager

def analyze_current_implementation():
    """Analyze how our current implementation handles verse edge cases."""
    
    # Check our test database
    test_db = Path("test_abba.db")
    if test_db.exists():
        test_db.unlink()
    
    print("Creating test database...")
    db_manager = SQLiteManager(test_db)
    db_manager.initialize_database()
    
    # Import eng_web translation
    from bible_extractor import BibleExtractor
    extractor = BibleExtractor("bible_data")
    
    # Get translation metadata first
    translations = extractor.list_translations()
    eng_web = next((t for t in translations if t["id"] == "eng_web"), None)
    
    if not eng_web:
        print("eng_web translation not found")
        return
    
    print("Inserting translation metadata...")
    db_manager.insert_translation(eng_web)
    
    print("Importing eng_web verses...")
    result = extractor.import_translation_to_db("eng_web", db_manager)
    
    if not result:
        print("Failed to import translation")
        return
    
    # Check how blank/empty verses are handled
    print("\nChecking how blank verses are stored...")
    
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        # Look for Acts 8:37 (known blank verse)
        cursor.execute("""
            SELECT translation_id, book_id, chapter, verse, text, length(text) as text_length
            FROM verses 
            WHERE translation_id = 'eng_web' AND book_id = 44 AND chapter = 8 AND verse = 37
        """)
        
        result = cursor.fetchone()
        if result:
            trans_id, book_id, chapter, verse, text, length = result
            print(f"Acts 8:37 found: '{text}' (length: {length})")
        else:
            print("Acts 8:37 not found in database")
        
        # Check for other potential blank verses
        cursor.execute("""
            SELECT translation_id, book_id, chapter, verse, text, length(text) as text_length
            FROM verses 
            WHERE translation_id = 'eng_web' AND (text = '' OR text IS NULL OR length(text) <= 2)
            ORDER BY book_id, chapter, verse
            LIMIT 10
        """)
        
        blank_verses = cursor.fetchall()
        print(f"\nFound {len(blank_verses)} blank or very short verses:")
        for trans_id, book_id, chapter, verse, text, length in blank_verses:
            print(f"  Book {book_id}, Chapter {chapter}, Verse {verse}: '{text}' (length: {length})")
        
        # Check for verse gaps (missing verse numbers)
        print("\nChecking for verse numbering gaps...")
        
        # Look at Acts 8 specifically
        cursor.execute("""
            SELECT verse, text
            FROM verses 
            WHERE translation_id = 'eng_web' AND book_id = 44 AND chapter = 8 
            ORDER BY verse
        """)
        
        acts8_verses = cursor.fetchall()
        verse_numbers = [v[0] for v in acts8_verses]
        print(f"Acts 8 verses: {verse_numbers}")
        
        # Check if there are any gaps
        if verse_numbers:
            expected_range = list(range(1, max(verse_numbers) + 1))
            missing = [v for v in expected_range if v not in verse_numbers]
            if missing:
                print(f"Missing verse numbers in Acts 8: {missing}")
            else:
                print("No missing verse numbers in Acts 8")
    
    # Check book_id mapping
    print("\nChecking book_id mapping...")
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.book_id, b.name, COUNT(*) as verse_count
            FROM verses v
            JOIN books b ON v.book_id = b.book_id AND v.translation_id = b.translation_id
            WHERE v.translation_id = 'eng_web'
            GROUP BY b.book_id, b.name
            ORDER BY b.book_id
            LIMIT 10
        """)
        
        book_stats = cursor.fetchall()
        print("Book verse counts:")
        for book_id, name, count in book_stats:
            print(f"  {book_id}: {name} ({count} verses)")
    
    # Cleanup
    test_db.unlink()
    print("\nAnalysis complete.")

def check_potential_issues():
    """Check for potential issues with our current approach."""
    
    print("\n" + "="*50)
    print("POTENTIAL VERSE NUMBERING ISSUES TO CONSIDER:")
    print("="*50)
    
    issues = [
        {
            "issue": "Empty/Blank Verses",
            "description": "Some translations include empty verses (like Acts 8:37 in eng_web)",
            "current_handling": "Stored as empty strings - preserves verse numbering",
            "concern": "Takes up space but maintains verse alignment across translations"
        },
        {
            "issue": "Skipped Verses",
            "description": "Some translations completely omit certain verses",
            "current_handling": "Not stored at all - creates gaps in verse numbering",
            "concern": "Cross-translation verse lookup becomes complex"
        },
        {
            "issue": "Merged Verses",
            "description": "Some translations combine multiple verse numbers into one text",
            "current_handling": "Unknown - depends on source data structure",
            "concern": "May lose verse-level granularity"
        },
        {
            "issue": "Book Ordering",
            "description": "Different traditions may order books differently",
            "current_handling": "Uses source database book order",
            "concern": "Book IDs may not align across translations"
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['issue']}")
        print(f"   Description: {issue['description']}")
        print(f"   Current handling: {issue['current_handling']}")
        print(f"   Concern: {issue['concern']}")
        print()

if __name__ == "__main__":
    analyze_current_implementation()
    check_potential_issues()