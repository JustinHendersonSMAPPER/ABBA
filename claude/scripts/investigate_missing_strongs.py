#!/usr/bin/env python3
"""
Investigate Missing Strong's Numbers

This script investigates why certain Strong's numbers from concepts.yaml
are not being found in the database.
"""

import sys
import sqlite3
from pathlib import Path
from collections import defaultdict

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database.sqlite_manager import SQLiteManager
from abba.logging_setup import logger


def investigate_strongs_formats(db_path: Path):
    """Investigate how Strong's numbers are actually stored in the database."""
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        print("\n🔍 Investigating Strong's Number Formats in Database")
        print("=" * 80)
        
        # 1. Check Greek Strong's format patterns
        print("\n📊 Greek Strong's Number Patterns:")
        cursor.execute("""
            SELECT strongs_lexical, COUNT(*) as count
            FROM stepbible_verses
            WHERE language = 'greek' AND strongs_lexical IS NOT NULL
            GROUP BY strongs_lexical
            ORDER BY count DESC
            LIMIT 20
        """)
        
        greek_patterns = defaultdict(int)
        for strongs, count in cursor.fetchall():
            # Analyze pattern
            if strongs.startswith('G'):
                num_part = strongs[1:]
                pattern = f"G + {len(num_part)} digits"
                greek_patterns[pattern] += 1
                if len(num_part) < 4:
                    print(f"  • Short format: {strongs} ({count} occurrences)")
                elif len(num_part) > 4:
                    print(f"  • Long format: {strongs} ({count} occurrences)")
        
        print("\n📈 Greek Pattern Summary:")
        for pattern, count in sorted(greek_patterns.items()):
            print(f"  • {pattern}: {count} unique Strong's")
        
        # 2. Check Hebrew Strong's format patterns
        print("\n\n📊 Hebrew Strong's Number Patterns:")
        cursor.execute("""
            SELECT strongs_lexical, COUNT(*) as count
            FROM stepbible_verses
            WHERE language = 'hebrew' AND strongs_lexical IS NOT NULL
            GROUP BY strongs_lexical
            ORDER BY count DESC
            LIMIT 20
        """)
        
        hebrew_patterns = defaultdict(int)
        for strongs, count in cursor.fetchall():
            if strongs.startswith('H'):
                num_part = strongs[1:]
                pattern = f"H + {len(num_part)} digits"
                hebrew_patterns[pattern] += 1
                if len(num_part) < 4:
                    print(f"  • Short format: {strongs} ({count} occurrences)")
                elif len(num_part) > 4:
                    print(f"  • Long format: {strongs} ({count} occurrences)")
        
        print("\n📈 Hebrew Pattern Summary:")
        for pattern, count in sorted(hebrew_patterns.items()):
            print(f"  • {pattern}: {count} unique Strong's")
        
        # 3. Look for specific missing Strong's numbers
        missing_strongs = [
            "G25", "G26", "G266", "G458", "G859", "G863",
            "G629", "G932", "G935", "G936", "G1656", "G3340",
            "G3341", "G4102", "G4103", "G5485", "G4991", "G1343",
            "G2889", "G1208", "G1654", "G1655", "G1831", "G4717",
            "G4716", "G4957"
        ]
        
        print("\n\n🔍 Searching for Missing Strong's Numbers:")
        print("=" * 80)
        
        for strongs in missing_strongs:
            print(f"\n📌 Searching for {strongs}...")
            
            # Try exact match
            cursor.execute("""
                SELECT COUNT(*) FROM stepbible_verses 
                WHERE strongs_lexical = ?
            """, (strongs,))
            exact_count = cursor.fetchone()[0]
            
            if exact_count > 0:
                print(f"  ✅ Found exact match: {exact_count} occurrences")
            else:
                print(f"  ❌ No exact match")
                
                # Try padded versions
                padded_versions = []
                if strongs.startswith('G'):
                    # Try different padding lengths for Greek
                    num_part = strongs[1:]
                    padded_versions = [
                        f"G{num_part.zfill(3)}",  # G025
                        f"G{num_part.zfill(4)}",  # G0025
                        f"G{num_part.zfill(5)}"   # G00025
                    ]
                
                for padded in padded_versions:
                    cursor.execute("""
                        SELECT COUNT(*) FROM stepbible_verses 
                        WHERE strongs_lexical = ?
                    """, (padded,))
                    padded_count = cursor.fetchone()[0]
                    
                    if padded_count > 0:
                        print(f"  ✅ Found as {padded}: {padded_count} occurrences")
                        break
                
                # Check if it exists in lexicon
                cursor.execute("""
                    SELECT COUNT(*) FROM lexicon 
                    WHERE strongs_number = ? OR strongs_number LIKE ?
                """, (strongs, f"{strongs[0]}%{strongs[1:]}"))
                lex_count = cursor.fetchone()[0]
                
                if lex_count > 0:
                    cursor.execute("""
                        SELECT strongs_number, gloss, definition 
                        FROM lexicon 
                        WHERE strongs_number = ? OR strongs_number LIKE ?
                        LIMIT 5
                    """, (strongs, f"{strongs[0]}%{strongs[1:]}"))
                    
                    print(f"  📚 Found in lexicon:")
                    for lex_strongs, gloss, definition in cursor.fetchall():
                        print(f"     • {lex_strongs}: {gloss or 'No gloss'}")
                        if definition:
                            print(f"       Definition: {definition[:100]}...")
                
                # Check if it exists with LIKE pattern
                cursor.execute("""
                    SELECT DISTINCT strongs_lexical, COUNT(*) as count
                    FROM stepbible_verses 
                    WHERE strongs_lexical LIKE ?
                    GROUP BY strongs_lexical
                    LIMIT 5
                """, (f"{strongs[0]}%{strongs[1:]}%",))
                
                similar = cursor.fetchall()
                if similar:
                    print(f"  🔍 Similar Strong's numbers found:")
                    for similar_strongs, count in similar:
                        print(f"     • {similar_strongs}: {count} occurrences")
        
        # 4. Check total coverage
        print("\n\n📊 Database Coverage Statistics:")
        print("=" * 80)
        
        cursor.execute("""
            SELECT 
                language,
                COUNT(DISTINCT strongs_lexical) as unique_strongs,
                COUNT(*) as total_words
            FROM stepbible_verses
            WHERE strongs_lexical IS NOT NULL
            GROUP BY language
        """)
        
        for lang, unique, total in cursor.fetchall():
            print(f"\n{lang.title()}:")
            print(f"  • Unique Strong's numbers: {unique:,}")
            print(f"  • Total word occurrences: {total:,}")
        
        # 5. Check for potential extraction issues
        print("\n\n🔍 Checking for Extraction Issues:")
        print("=" * 80)
        
        # Look for cases where strongs_primary has values but strongs_lexical is null
        cursor.execute("""
            SELECT COUNT(*)
            FROM stepbible_verses
            WHERE language = 'greek' 
            AND strongs_primary IS NOT NULL 
            AND strongs_lexical IS NULL
        """)
        orphaned_greek = cursor.fetchone()[0]
        
        if orphaned_greek > 0:
            print(f"\n⚠️  Found {orphaned_greek} Greek verses with strongs_primary but no strongs_lexical!")
            
            cursor.execute("""
                SELECT id, book, chapter, verse, strongs_primary, strongs_raw
                FROM stepbible_verses
                WHERE language = 'greek' 
                AND strongs_primary IS NOT NULL 
                AND strongs_lexical IS NULL
                LIMIT 10
            """)
            
            print("\nSample cases:")
            for row in cursor.fetchall():
                print(f"  • {row[1]} {row[2]}:{row[3]} - primary: {row[4]}, raw: {row[5]}")


def check_specific_verse_examples(db_path: Path):
    """Check specific verses that should contain missing Strong's numbers."""
    
    print("\n\n🔍 Checking Specific Verse Examples")
    print("=" * 80)
    
    # Known verses with specific Strong's
    test_cases = [
        ("G26", "John", 3, 16, "For God so loved (ἠγάπησεν)"),
        ("G25", "John", 13, 34, "Love (ἀγαπᾶτε) one another"),
        ("G932", "Matthew", 6, 10, "Your kingdom (βασιλεία) come"),
        ("G4102", "Hebrews", 11, 1, "Now faith (πίστις) is"),
    ]
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        for strongs, book, chapter, verse, description in test_cases:
            print(f"\n📖 {description}")
            print(f"   Looking for {strongs} in {book} {chapter}:{verse}")
            
            cursor.execute("""
                SELECT 
                    original_word,
                    strongs_primary,
                    strongs_raw,
                    strongs_lexical,
                    morphology
                FROM stepbible_verses
                WHERE book = ? AND chapter = ? AND verse = ?
                AND language = 'greek'
                ORDER BY word_order
            """, (book, chapter, verse))
            
            words = cursor.fetchall()
            if words:
                print(f"   Found {len(words)} Greek words:")
                found_target = False
                for word in words:
                    if word[3] and strongs[1:] in word[3]:
                        print(f"   ✅ {word[0]} - lexical: {word[3]}")
                        found_target = True
                    elif word[1] and strongs[1:] in word[1]:
                        print(f"   ⚠️  {word[0]} - primary: {word[1]} (but lexical: {word[3]})")
                
                if not found_target:
                    print(f"   ❌ Strong's {strongs} not found in this verse!")
            else:
                print(f"   ❌ No Greek words found for this verse!")


def suggest_fixes(db_path: Path):
    """Suggest fixes for the missing Strong's numbers."""
    
    print("\n\n💡 Suggested Fixes")
    print("=" * 80)
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        # Check if we need to re-run the schema fix
        cursor.execute("""
            SELECT COUNT(*)
            FROM stepbible_verses
            WHERE language = 'greek' 
            AND strongs_primary IS NOT NULL 
            AND strongs_primary != ''
            AND (strongs_lexical IS NULL OR strongs_lexical = '')
        """)
        
        missing_extractions = cursor.fetchone()[0]
        
        if missing_extractions > 0:
            print(f"\n🔧 Need to fix {missing_extractions} Greek verses with missing strongs_lexical!")
            print("\nSuggested action:")
            print("1. Re-run the fix_stepbible_schema.py script")
            print("2. Or create a targeted fix for Greek Strong's extraction")
        
        # Check padding issues
        print("\n\n🔧 Checking if padding adjustment needed...")
        
        # See if G0026 exists when G26 doesn't
        test_nums = ["26", "25", "932", "4102"]
        for num in test_nums:
            variants = [f"G{num}", f"G0{num}", f"G00{num}", f"G000{num}", f"G{num.zfill(4)}"]
            
            cursor.execute("""
                SELECT strongs_lexical, COUNT(*) as count
                FROM stepbible_verses
                WHERE strongs_lexical IN ({})
                GROUP BY strongs_lexical
            """.format(','.join(['?' for _ in variants])), variants)
            
            results = cursor.fetchall()
            if results:
                print(f"\nStrong's G{num} variants found:")
                for strongs, count in results:
                    print(f"  • {strongs}: {count} occurrences")


def main():
    """Run the investigation."""
    config = config_manager.load_config()
    db_path = config.data_dir / "abba.db"
    
    print("🔍 Investigating Missing Strong's Numbers")
    print("=" * 80)
    
    try:
        investigate_strongs_formats(db_path)
        check_specific_verse_examples(db_path)
        suggest_fixes(db_path)
        
        print("\n\n✅ Investigation complete!")
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)