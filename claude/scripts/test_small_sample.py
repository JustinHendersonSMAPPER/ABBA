#!/usr/bin/env python3
"""Test STEPBible parser with a small sample."""

import tempfile
from pathlib import Path
from abba.bible_extractor import BibleExtractor
from abba.database import SQLiteManager

def main():
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir)
    stepbible_dir = data_dir / "stepbible"
    stepbible_dir.mkdir(exist_ok=True)
    
    # Create sample STEPBible data (real format)
    sample_content = """TAHOT Gen-Deu - Sample data

FIELD DESCRIPTIONS:
...

Gen.1.1#01=L	בְּ/רֵאשִׁ֖ית	be./re.Shit	in/ beginning	H9003/{H7225G}	HR/Ncfsa			H7225G			H9003=ב=in/{H7225G=רֵאשִׁית=: beginning»first:1_beginning}
Gen.1.1#02=L	בָּרָ֣א	ba.Ra'	he created	{H1254A}	HVqp3ms			H1254A			{H1254A=בָּרָא=to create}
Gen.1.1#03=L	אֱלֹהִ֑ים	'E.lo.Him	God	{H0430G}	HNcmpa			H0430G			{H0430G=אֱלֹהִים=God»LORD@Gen.1.1-Heb}
"""
    
    sample_file = stepbible_dir / "tahot_gen_deu.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    # Set up database
    db_path = Path(temp_dir) / "test.db"
    db_manager = SQLiteManager(db_path)
    db_manager.initialize_database()
    
    # Test parser
    extractor = BibleExtractor(str(data_dir))
    print("Testing STEPBible parser with sample data...")
    
    result = extractor.parse_stepbible_text("tahot_gen_deu.txt", db_manager)
    print(f"Parse result: {result}")
    
    # Check results
    stats = db_manager.get_database_stats()
    print(f"Words imported: {stats.get('words', 0)}")
    
    if stats.get('words', 0) > 0:
        words = db_manager.get_words_for_verse("Gen", 1, 1)
        print(f"Found {len(words)} words for Gen 1:1:")
        for word in words:
            print(f"  {word['word_num']}: {word['hebrew_text']} = {word['translation']} ({word['strongs_primary']})")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()