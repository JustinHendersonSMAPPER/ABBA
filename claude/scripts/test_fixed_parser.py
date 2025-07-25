#!/usr/bin/env python3
"""Test the fixed STEPBible parser."""

from pathlib import Path
from abba.bible_extractor import BibleExtractor
from abba.database import SQLiteManager
from abba.config import config_manager

def main():
    config = config_manager.load_config([])
    
    # Create test database
    test_db_path = config.data_dir / "test_fixed.db"
    if test_db_path.exists():
        test_db_path.unlink()
        
    db_manager = SQLiteManager(test_db_path)
    db_manager.initialize_database()
    
    extractor = BibleExtractor(str(config.data_dir))
    
    print("Testing fixed STEPBible parser...")
    print("=" * 50)
    
    # Test Hebrew parsing
    print("Testing tahot_gen_deu.txt...")
    result = extractor.parse_stepbible_text("tahot_gen_deu.txt", db_manager)
    print(f"Parse result: {result}")
    
    # Check results
    stats = db_manager.get_database_stats()
    print(f"Words imported: {stats.get('words', 0):,}")
    
    if stats.get('words', 0) > 0:
        # Show some sample words
        print("\nSample words from Gen 1:1:")
        words = db_manager.get_words_for_verse("Gen", 1, 1)
        for word in words[:5]:
            print(f"  {word['word_num']}: {word['hebrew_text']} = {word['translation']} ({word['strongs_primary']})")
    
    # Clean up
    test_db_path.unlink()

if __name__ == "__main__":
    main()