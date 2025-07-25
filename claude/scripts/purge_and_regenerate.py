#!/usr/bin/env python3
"""Purge bad data and regenerate everything with the fixed parser."""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=== ABBA Data Purge and Regeneration Script ===\n")

# Check current status
print("1. Checking current data issues...")
from abba.database import SQLiteManager

db = SQLiteManager("bible_data/abba.db")
with db.get_connection() as conn:
    cursor = conn.cursor()
    
    # Count invalid strongs entries
    cursor.execute("""
        SELECT COUNT(*) 
        FROM words
        WHERE strongs_primary NOT LIKE 'H%' 
        AND strongs_primary NOT LIKE 'G%'
        AND strongs_primary IS NOT NULL
        AND strongs_primary != ''
    """)
    
    invalid_count = cursor.fetchone()[0]
    print(f"   Found {invalid_count:,} words with invalid Strong's numbers (Spanish text)")

print("\n2. Steps to fix:")
print("   a) Delete word data from database")
print("   b) Reset import tracking for STEPBible files")
print("   c) Delete ChromaDB vector database")
print("   d) Re-import STEPBible data with fixed parser")
print("   e) Regenerate embeddings")

response = input("\nProceed with purge and regeneration? (yes/no): ")
if response.lower() != 'yes':
    print("Aborted.")
    sys.exit(0)

print("\n3. Purging bad data...")

# Delete words from database
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM words")
    conn.commit()
    print("   ✓ Deleted all words from database")

# Reset import tracking
progress_file = Path("bible_data/.import_progress.json")
if progress_file.exists():
    import json
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    # Reset STEPBible files
    if 'stepbible_files' in progress:
        progress['stepbible_files'] = {}
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print("   ✓ Reset STEPBible import tracking")

# Delete ChromaDB
vectors_path = Path("bible_data/vectors")
if vectors_path.exists():
    shutil.rmtree(vectors_path)
    print("   ✓ Deleted ChromaDB vector database")

# Reset embedding progress
embedding_progress_file = Path("bible_data/.embedding_progress.json")
if embedding_progress_file.exists():
    embedding_progress_file.unlink()
    print("   ✓ Reset embedding progress tracking")

print("\n4. Ready for regeneration!")
print("\nNow run these commands in order:")
print("\n   # Re-import STEPBible data with fixed parser")
print("   python abba/main.py")
print("\n   # Generate embeddings (this will take a while)")
print("   python abba/main.py --embed-verses --embed-words")
print("\nThe system will:")
print("   - Import STEPBible data with correct Strong's numbers for Greek")
print("   - Generate verse embeddings for BSB translation")
print("   - Generate word embeddings with proper glosses")
print("\nEstimated time: ~15-30 minutes depending on GPU")