#!/usr/bin/env python3
"""Check the strongs_primary field issue in the database."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database import SQLiteManager

db = SQLiteManager("bible_data/abba.db")
with db.get_connection() as conn:
    cursor = conn.cursor()
    
    # Find entries where strongs_primary doesn't look like a Strong's number
    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN strongs_primary LIKE 'H%' OR strongs_primary LIKE 'G%' THEN 1 ELSE 0 END) as valid_strongs,
               SUM(CASE WHEN strongs_primary NOT LIKE 'H%' AND strongs_primary NOT LIKE 'G%' THEN 1 ELSE 0 END) as invalid_strongs
        FROM words
        WHERE strongs_primary IS NOT NULL AND strongs_primary != ''
    """)
    
    result = cursor.fetchone()
    print(f"Total words with strongs_primary: {result[0]:,}")
    print(f"Valid Strong's numbers (H*/G*): {result[1]:,}")
    print(f"Invalid entries: {result[2]:,}")
    
    # Sample invalid entries
    cursor.execute("""
        SELECT book, chapter, verse, word_num, strongs_primary, translation, language
        FROM words
        WHERE strongs_primary NOT LIKE 'H%' 
        AND strongs_primary NOT LIKE 'G%'
        AND strongs_primary IS NOT NULL
        AND strongs_primary != ''
        LIMIT 10
    """)
    
    print("\nSample invalid entries:")
    for row in cursor.fetchall():
        book, chapter, verse, word_num, strongs, trans, lang = row
        print(f"  {book} {chapter}:{verse}.{word_num} ({lang})")
        print(f"    strongs_primary: '{strongs}'")
        print(f"    translation: '{trans}'")
        
    # Check if this is a specific translation issue
    cursor.execute("""
        SELECT DISTINCT book, COUNT(*) as count
        FROM words
        WHERE strongs_primary NOT LIKE 'H%' 
        AND strongs_primary NOT LIKE 'G%'
        AND strongs_primary IS NOT NULL
        AND strongs_primary != ''
        GROUP BY book
        ORDER BY count DESC
        LIMIT 10
    """)
    
    print("\nBooks with invalid strongs_primary entries:")
    for book, count in cursor.fetchall():
        print(f"  {book}: {count:,} entries")