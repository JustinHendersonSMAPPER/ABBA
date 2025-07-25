#!/usr/bin/env python3
"""Simple test script to debug database import issues."""

import sys
from pathlib import Path

# Add the abba package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "abba"))

from database.sqlite_manager import SQLiteManager
from bible_extractor import BibleExtractor

def main():
    print("Testing database import...")
    
    # Initialize database
    db_path = Path("test_abba.db")
    if db_path.exists():
        db_path.unlink()
    
    print("Creating database...")
    db_manager = SQLiteManager(db_path)
    db_manager.initialize_database()
    
    print("Creating extractor...")
    extractor = BibleExtractor("bible_data")
    
    print("Getting translations list...")
    translations = extractor.list_translations()
    print(f"Found {len(translations)} translations")
    
    # Find eng_web translation
    eng_web = None
    for trans in translations:
        if trans["id"] == "eng_web":
            eng_web = trans
            break
    
    if not eng_web:
        print("eng_web translation not found!")
        print("Available English translations:")
        for trans in translations:
            if "eng" in trans["id"]:
                print(f"  {trans['id']}: {trans['english_name']}")
        return
    
    print(f"Found eng_web: {eng_web}")
    print(f"Data types: {[(k, type(v)) for k, v in eng_web.items()]}")
    
    # Test inserting translation metadata
    print("Inserting translation metadata...")
    try:
        db_manager.insert_translation(eng_web)
        print("✓ Translation metadata inserted successfully")
    except Exception as e:
        print(f"✗ Failed to insert translation metadata: {e}")
        return
    
    # Test importing verses
    print("Importing verses...")
    try:
        result = extractor.import_translation_to_db("eng_web", db_manager)
        if result:
            print("✓ Verses imported successfully")
        else:
            print("✗ Failed to import verses")
    except Exception as e:
        print(f"✗ Error importing verses: {e}")
        import traceback
        traceback.print_exc()
    
    # Check database stats
    print("Database stats:")
    stats = db_manager.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    db_path.unlink()
    print("Test complete.")

if __name__ == "__main__":
    main()