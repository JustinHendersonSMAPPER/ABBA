#!/usr/bin/env python3
"""Debug the words table to understand the data structure."""

import sqlite3
from pathlib import Path

db_path = Path("bible_data/abba.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== WORDS TABLE INVESTIGATION ===\n")

# Check total words
cursor.execute("SELECT COUNT(*) as count FROM words")
print(f"Total words in database: {cursor.fetchone()['count']:,}\n")

# Check language distribution
print("Language distribution:")
cursor.execute("SELECT language, COUNT(*) as count FROM words GROUP BY language")
for row in cursor.fetchall():
    print(f"  {row['language']}: {row['count']:,}")

# Check some Greek words
print("\n5 Sample Greek words:")
cursor.execute("""
    SELECT book, chapter, verse, word_num, greek_text, transliteration, 
           strongs_primary, translation, morphology_code
    FROM words 
    WHERE language = 'greek' AND greek_text IS NOT NULL
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"\n  {row['book']} {row['chapter']}:{row['verse']} (word {row['word_num']})")
    print(f"  Greek: {row['greek_text']}")
    print(f"  Transliteration: {row['transliteration']}")
    print(f"  Strong's: {row['strongs_primary']}")
    print(f"  Translation: {row['translation']}")
    print(f"  Morphology: {row['morphology_code']}")

# Check John 1:1
print("\n\nJohn 1:1 words:")
cursor.execute("""
    SELECT word_num, greek_text, transliteration, strongs_primary, translation
    FROM words 
    WHERE book = 'Joh' AND chapter = 1 AND verse = 1
    ORDER BY word_num
""")
results = cursor.fetchall()
if results:
    for row in results:
        print(f"  {row['word_num']}. {row['greek_text']} ({row['transliteration']}) - {row['translation']} [{row['strongs_primary']}]")
else:
    print("  No data found")

# Check available Strong's numbers for agape
print("\n\nSearching for 'agape' related Strong's numbers:")
cursor.execute("""
    SELECT DISTINCT strongs_primary, COUNT(*) as count
    FROM words 
    WHERE strongs_primary LIKE 'G26%' OR translation LIKE '%love%'
    GROUP BY strongs_primary
    ORDER BY count DESC
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"  {row['strongs_primary']}: {row['count']} occurrences")

# Check morphology codes in Matthew 5
print("\n\nMorphology codes in Matthew 5:")
cursor.execute("""
    SELECT DISTINCT morphology_code, COUNT(*) as count
    FROM words 
    WHERE book = 'Mat' AND chapter = 5 
    AND morphology_code LIKE '%V%'
    GROUP BY morphology_code
    ORDER BY count DESC
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"  {row['morphology_code']}: {row['count']} occurrences")

# Check Hebrew Strong's numbers
print("\n\nHebrew Strong's numbers (H6942-H6948):")
cursor.execute("""
    SELECT DISTINCT strongs_primary, hebrew_text, transliteration, translation, COUNT(*) as count
    FROM words 
    WHERE strongs_primary BETWEEN 'H6942' AND 'H6948'
    GROUP BY strongs_primary
    ORDER BY strongs_primary
""")
for row in cursor.fetchall():
    print(f"  {row['strongs_primary']}: {row['hebrew_text']} ({row['transliteration']}) - {row['translation']} [{row['count']} times]")

conn.close()