#!/usr/bin/env python3
"""Check how translations handle verse numbering edge cases."""

import sqlite3
from pathlib import Path

def check_verse_gaps():
    """Check for gaps in verse numbering that might indicate skipped verses."""
    
    bible_db = Path("bible_data/bible.db")
    if not bible_db.exists():
        print("bible.db not found")
        return
    
    with sqlite3.connect(bible_db) as conn:
        cursor = conn.cursor()
        
        # Check some known problematic verses - test both English and non-English
        test_cases = [
            # Matthew 17:21 - often omitted in modern translations
            ("eng_web", "MAT", 17, [19, 20, 21, 22, 23]),
            # Test non-English translations
            ("deu_sch", "MAT", 17, [19, 20, 21, 22, 23]),
            ("fra_lsg", "MAT", 17, [19, 20, 21, 22, 23]),
            # Acts 8:37 - often omitted
            ("eng_web", "ACT", 8, [36, 37, 38, 39]),
            ("deu_sch", "ACT", 8, [36, 37, 38, 39]),
            ("fra_lsg", "ACT", 8, [36, 37, 38, 39]),
        ]
        
        for trans_id, book_id, chapter, verses in test_cases:
            print(f"\nChecking {trans_id} {book_id} {chapter}:{verses}")
            
            # Get verses in this range
            cursor.execute("""
                SELECT number, text 
                FROM ChapterVerse 
                WHERE translationId = ? AND bookId = ? AND chapterNumber = ? AND number IN ({})
                ORDER BY number
            """.format(','.join(['?'] * len(verses))), 
            [trans_id, book_id, chapter] + verses)
            
            results = cursor.fetchall()
            found_verses = {v[0] for v in results}
            
            # Check for gaps
            for v in verses:
                if v in found_verses:
                    text = next(r[1] for r in results if r[0] == v)
                    print(f"  {v}: {text[:80]}{'...' if len(text) > 80 else ''}")
                else:
                    print(f"  {v}: [MISSING/SKIPPED]")
    
    print("\n" + "="*50)
    print("Checking for merged verses (verse ranges like '1-2')")
    
    # Look for verse ranges in the text or unusual numbering
    cursor.execute("""
        SELECT translationId, bookId, chapterNumber, number, text
        FROM ChapterVerse 
        WHERE translationId = 'eng_web' 
        AND (text LIKE '%verse%' OR text LIKE '%omit%' OR text LIKE '%see%')
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    for trans_id, book_id, chapter, verse, text in results:
        print(f"{trans_id} {book_id} {chapter}:{verse} - {text[:100]}...")

def check_verse_numbering_patterns():
    """Check for unusual verse numbering patterns."""
    
    bible_db = Path("bible_data/bible.db")
    with sqlite3.connect(bible_db) as conn:
        cursor = conn.cursor()
        
        print("\nChecking verse numbering patterns...")
        
        # Look for chapters with gaps in verse numbering
        cursor.execute("""
            SELECT translationId, bookId, chapterNumber, 
                   GROUP_CONCAT(number ORDER BY number) as verse_numbers,
                   COUNT(*) as verse_count
            FROM ChapterVerse 
            WHERE translationId = 'eng_web' AND bookId IN ('MAT', 'MRK', 'LUK')
            GROUP BY translationId, bookId, chapterNumber
            HAVING verse_count < 20  -- Look at shorter chapters first
            ORDER BY bookId, chapterNumber
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        for trans_id, book_id, chapter, verse_nums, count in results:
            verse_list = [int(v) for v in verse_nums.split(',')]
            expected_range = list(range(1, max(verse_list) + 1))
            missing = [v for v in expected_range if v not in verse_list]
            
            print(f"{book_id} {chapter}: verses {verse_nums} (count: {count})")
            if missing:
                print(f"  Missing verses: {missing}")

if __name__ == "__main__":
    check_verse_gaps()
    check_verse_numbering_patterns()