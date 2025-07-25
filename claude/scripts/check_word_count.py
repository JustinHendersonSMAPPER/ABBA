#!/usr/bin/env python3
"""Check word count after parser fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database import SQLiteManager

db = SQLiteManager('bible_data/abba.db')
with db.get_connection() as conn:
    cursor = conn.cursor()
    
    # Check total unique words
    cursor.execute("""
        SELECT COUNT(DISTINCT strongs_primary || ':' || COALESCE(morphology_code, ''))
        FROM words
        WHERE strongs_primary IS NOT NULL
        AND strongs_primary != ''
    """)
    unique_words = cursor.fetchone()[0]
    
    # Check valid vs invalid Strong's numbers
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN strongs_primary LIKE 'H%' OR strongs_primary LIKE 'G%' THEN 1 ELSE 0 END) as valid_strongs,
            SUM(CASE WHEN strongs_primary NOT LIKE 'H%' AND strongs_primary NOT LIKE 'G%' THEN 1 ELSE 0 END) as invalid_strongs
        FROM words
        WHERE strongs_primary IS NOT NULL AND strongs_primary != ''
    """)
    
    valid, invalid = cursor.fetchone()
    
    print(f'Total unique words available for embedding: {unique_words:,}')
    print(f'Valid Strongs numbers: {valid:,}')
    print(f'Invalid entries: {invalid:,}')
    print(f'Embeddings generated: 67,838')
    print(f'Expected embeddings vs actual: {unique_words - 67838:,} difference')
    
    # Sample some of the words to see if they look correct now
    cursor.execute("""
        SELECT book, chapter, verse, word_num, 
               CASE WHEN language = 'greek' THEN greek_text ELSE hebrew_text END as word_text,
               transliteration, translation, strongs_primary
        FROM words
        WHERE book = 'Mat' AND chapter = 1 AND verse = 1
        ORDER BY word_num
        LIMIT 5
    """)
    
    print('\nSample Greek words (Matthew 1:1):')
    for row in cursor.fetchall():
        book, chapter, verse, word_num, word_text, trans, translation, strongs = row
        print(f'  Word {word_num}: {word_text} ({trans}) = "{translation}" -> {strongs}')