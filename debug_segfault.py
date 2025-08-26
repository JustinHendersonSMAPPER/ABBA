#!/usr/bin/env python3
"""Debug the segmentation fault issue."""

import sqlite3
from pathlib import Path
import json

db_path = Path("bible_data/abba.db")
progress_path = Path("bible_data/.embedding_progress.json")

# Check progress
if progress_path.exists():
    with open(progress_path) as f:
        progress = json.load(f)
    
    print("Embedding progress:")
    for trans_id, info in progress.get('verses', {}).items():
        if info.get('complete'):
            print(f"  {trans_id}: Complete ({info.get('count', 0)} verses)")
        else:
            print(f"  {trans_id}: In progress (last count: {info.get('last_count', 0)})")

# Check HBOMAS specifically
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    
    # Get HBOMAS info
    cursor.execute("""
        SELECT COUNT(*) 
        FROM verses 
        WHERE translation_id = 'HBOMAS' 
        AND text IS NOT NULL AND text != ''
    """)
    hbomas_count = cursor.fetchone()[0]
    print(f"\nHBOMAS has {hbomas_count} verses")
    
    # Check if there's something special about verse 19000
    cursor.execute("""
        SELECT book_id, chapter, verse, LENGTH(text) as text_len
        FROM verses
        WHERE translation_id = 'HBOMAS'
        ORDER BY id
        LIMIT 5 OFFSET 18995
    """)
    
    print("\nVerses around position 19000:")
    for row in cursor.fetchall():
        print(f"  Book {row[0]}, Chapter {row[1]}, Verse {row[2]}, Text length: {row[3]}")

# Check ChromaDB status
try:
    import chromadb
    client = chromadb.PersistentClient(path="bible_data/vectors")
    verses_collection = client.get_collection("verses")
    
    # Get HBOMAS embeddings count
    try:
        # This might fail if collection is corrupted
        result = verses_collection.get(
            where={"translation_id": "HBOMAS"},
            limit=1
        )
        print(f"\nChromaDB query successful")
    except Exception as e:
        print(f"\nChromaDB query failed: {e}")
        
except Exception as e:
    print(f"\nError accessing ChromaDB: {e}")