#!/usr/bin/env python3
"""Test the fixed STEPBible parser to verify it correctly handles Greek text."""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.bible_extractor import BibleExtractor
from abba.database import SQLiteManager

# Create a test database
test_db_path = Path("claude/test_fixed_parser.db")
if test_db_path.exists():
    test_db_path.unlink()

# Initialize components
db_manager = SQLiteManager(test_db_path)
db_manager.initialize_database()

# Create extractor with the STEPBible directory
extractor = BibleExtractor(config_file=None)
extractor.stepbible_dir = Path("bible_data/stepbible")

# Test with a small sample of Greek text to see if it's fixed
print("Testing Greek text parsing...")

# First, let's manually check what the parsing produces
text_path = extractor.stepbible_dir / "tagnt_mat_jhn.txt"
with open(text_path, "r", encoding="utf-8") as f:
    content = f.read()

# Get first few data lines
data_lines = []
for line in content.split("\n"):
    line = line.strip()
    if line and not line.startswith("#") and "Mat.1." in line and "\t" in line:
        data_lines.append(line)
        if len(data_lines) >= 5:
            break

print("\nSample Greek lines:")
for i, line in enumerate(data_lines):
    parts = line.split("\t")
    print(f"\nLine {i+1}: {parts[0]}")
    print(f"  Parts count: {len(parts)}")
    if len(parts) > 8:
        print(f"  Part 1 (Greek+trans): {parts[1][:50]}...")
        print(f"  Part 2 (English): {parts[2]}")
        print(f"  Part 3 (Strongs+morph): {parts[3]}")
        print(f"  Part 8 (Spanish?): {parts[8]}")

# Now test the parser
print("\n\nParsing with fixed parser...")
result = extractor.parse_stepbible_text("tagnt_mat_jhn.txt", db_manager)
print(f"Parse result: {result}")

# Check what was inserted
with db_manager.get_connection() as conn:
    cursor = conn.cursor()
    
    # Check Matthew 1:1 words
    cursor.execute("""
        SELECT word_num, greek_text, transliteration, translation, strongs_primary
        FROM words
        WHERE book = 'Mat' AND chapter = 1 AND verse = 1
        ORDER BY word_num
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    print(f"\nMatthew 1:1 words in database ({len(results)} words):")
    for word_num, greek, trans, translation, strongs in results:
        print(f"  Word {word_num}:")
        print(f"    Greek: {greek}")
        print(f"    Transliteration: {trans}")
        print(f"    Translation: {translation}")
        print(f"    strongs_primary: {strongs}")
        print()

# Clean up
test_db_path.unlink()