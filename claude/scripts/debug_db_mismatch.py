#!/usr/bin/env python3
"""Debug the database mismatch issue when importing translations."""

import sqlite3
from pathlib import Path
from abba.config import config_manager
from abba.bible_extractor import BibleExtractor

def main():
    config = config_manager.load_config([])
    
    print("Debugging database mismatch issue...")
    print("=" * 50)
    
    # Check bible.db structure
    bible_db_path = config.db_path
    if not bible_db_path.exists():
        print(f"bible.db not found at {bible_db_path}")
        return
    
    print(f"Examining bible.db at: {bible_db_path}")
    
    with sqlite3.connect(bible_db_path) as conn:
        cursor = conn.cursor()
        
        # Check Translation table structure
        print("\nTranslation table schema:")
        cursor.execute("PRAGMA table_info(Translation)")
        translation_cols = cursor.fetchall()
        for col in translation_cols:
            print(f"  {col}")
        
        # Check Verse table structure  
        print("\nVerse table schema:")
        cursor.execute("PRAGMA table_info(Verse)")
        verse_cols = cursor.fetchall()
        for col in verse_cols:
            print(f"  {col}")
            
        # Sample a few verses to see the data types
        print("\nSample verse data:")
        cursor.execute("SELECT * FROM Verse LIMIT 3")
        verses = cursor.fetchall()
        for verse in verses:
            print(f"  {verse}")
            print(f"  Types: {[type(v).__name__ for v in verse]}")
            
        # Check Book table structure
        print("\nBook table schema:")
        cursor.execute("PRAGMA table_info(Book)")
        book_cols = cursor.fetchall()
        for col in book_cols:
            print(f"  {col}")
            
        # Sample book data
        print("\nSample book data:")
        cursor.execute("SELECT * FROM Book LIMIT 3")
        books = cursor.fetchall()
        for book in books:
            print(f"  {book}")
            print(f"  Types: {[type(v).__name__ for v in book]}")
    
    # Check our ABBA database schema
    abba_db_path = config.abba_db_path
    if abba_db_path.exists():
        print(f"\nExamining ABBA database at: {abba_db_path}")
        
        with sqlite3.connect(abba_db_path) as conn:
            cursor = conn.cursor()
            
            print("\nABBA verses table schema:")
            cursor.execute("PRAGMA table_info(verses)")
            abba_verse_cols = cursor.fetchall()
            for col in abba_verse_cols:
                print(f"  {col}")
                
            print("\nABBA books table schema:")
            cursor.execute("PRAGMA table_info(books)")
            abba_book_cols = cursor.fetchall()
            for col in abba_book_cols:
                print(f"  {col}")
    
    # Test the actual import process step by step
    print("\nTesting verse import process...")
    extractor = BibleExtractor(str(config.data_dir))
    
    # Get a specific translation
    translations = extractor.list_translations()
    if translations:
        test_translation = translations[0]
        print(f"Testing with translation: {test_translation}")
        
        # Try to extract verses for this translation  
        try:
            translation_data = extractor.extract_translation(test_translation['id'])
            if translation_data and 'verses' in translation_data:
                sample_verse = list(translation_data['verses'].values())[0]
                print(f"Sample verse structure: {sample_verse}")
                print(f"Sample verse types: {[(k, type(v).__name__) for k, v in sample_verse.items()]}")
        except Exception as e:
            print(f"Error extracting translation: {e}")

if __name__ == "__main__":
    main()