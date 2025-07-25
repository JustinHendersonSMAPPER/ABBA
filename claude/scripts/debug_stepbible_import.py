#!/usr/bin/env python3
"""Debug STEPBible import to see why words count is 0."""

import sys
from pathlib import Path
from abba.bible_extractor import BibleExtractor
from abba.database import SQLiteManager
from abba.config import config_manager

def main():
    # Load config
    config = config_manager.load_config([])
    
    # Check if STEPBible files exist
    stepbible_dir = config.data_dir / "stepbible"
    print(f"STEPBible directory: {stepbible_dir}")
    print(f"Directory exists: {stepbible_dir.exists()}")
    
    if stepbible_dir.exists():
        files = list(stepbible_dir.glob("*.txt"))
        print(f"Found {len(files)} .txt files:")
        for f in files:
            print(f"  {f.name} ({f.stat().st_size} bytes)")
            
        # Check specific TAHOT/TAGNT files
        text_files = [
            "tahot_gen_deu.txt",
            "tahot_jos_est.txt", 
            "tahot_job_sng.txt",
            "tahot_isa_mal.txt",
            "tagnt_mat_jhn.txt",
            "tagnt_act_rev.txt"
        ]
        
        print("\nTAHOT/TAGNT files status:")
        for filename in text_files:
            filepath = stepbible_dir / filename
            exists = filepath.exists()
            size = filepath.stat().st_size if exists else 0
            print(f"  {filename}: {'EXISTS' if exists else 'MISSING'} ({size} bytes)")
            
        # Try to read a small sample from one file
        sample_file = stepbible_dir / "tahot_gen_deu.txt"
        if sample_file.exists():
            print(f"\nFirst 10 lines of {sample_file.name}:")
            with open(sample_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    print(f"  {i+1}: {line.strip()}")
                    
        # Test parsing a single file
        print(f"\nTesting parse of tahot_gen_deu.txt...")
        extractor = BibleExtractor(str(config.data_dir))
        
        # Setup test database
        test_db_path = config.data_dir / "debug_test.db"
        if test_db_path.exists():
            test_db_path.unlink()
            
        db_manager = SQLiteManager(test_db_path)
        db_manager.initialize_database()
        
        # Try parsing
        result = extractor.parse_stepbible_text("tahot_gen_deu.txt", db_manager)
        print(f"Parse result: {result}")
        
        # Check what was inserted
        stats = db_manager.get_database_stats()
        print(f"Words inserted: {stats.get('words', 0)}")
        
        # Clean up
        test_db_path.unlink()
    else:
        print("STEPBible directory doesn't exist - run with --force-download to download data")

if __name__ == "__main__":
    main()