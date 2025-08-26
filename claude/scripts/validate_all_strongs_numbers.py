#!/usr/bin/env python3
"""
Validate ALL Strong's Numbers in Database

This script validates that we have complete coverage of all Strong's numbers
that should exist according to the standard Strong's Concordance ranges.
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


def get_expected_strongs_ranges():
    """
    Define the expected ranges for Strong's numbers based on the
    standard Strong's Exhaustive Concordance.
    """
    return {
        'hebrew': {
            'prefix': 'H',
            'start': 1,
            'end': 8674,  # Standard Hebrew range
            'description': 'Hebrew/Aramaic Old Testament'
        },
        'greek': {
            'prefix': 'G',
            'start': 1,
            'end': 5624,  # Standard Greek range
            'description': 'Greek New Testament'
        }
    }


def validate_strongs_coverage(db_path: Path):
    """Validate that all expected Strong's numbers exist in the database."""
    
    ranges = get_expected_strongs_ranges()
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        print("🔍 Validating Complete Strong's Number Coverage")
        print("=" * 80)
        
        total_missing = 0
        total_expected = 0
        
        for language, info in ranges.items():
            print(f"\n📚 Checking {info['description']} ({info['prefix']}{info['start']:04d}-{info['prefix']}{info['end']:04d})")
            print("-" * 60)
            
            # Get all Strong's numbers from the database
            cursor.execute("""
                SELECT DISTINCT strongs_lexical
                FROM stepbible_verses
                WHERE language = ? AND strongs_lexical IS NOT NULL
                ORDER BY strongs_lexical
            """, (language,))
            
            db_strongs = {row[0] for row in cursor.fetchall()}
            
            # Also check lexicon table
            cursor.execute("""
                SELECT DISTINCT strongs_number
                FROM lexicon
                WHERE strongs_number LIKE ?
                ORDER BY strongs_number
            """, (f"{info['prefix']}%",))
            
            lexicon_strongs = {row[0] for row in cursor.fetchall()}
            
            # Check each expected Strong's number
            missing_strongs = []
            missing_in_verses = []
            missing_in_lexicon = []
            found_count = 0
            
            for num in range(info['start'], info['end'] + 1):
                # Check both padded and unpadded formats
                variants = [
                    f"{info['prefix']}{num}",
                    f"{info['prefix']}{num:04d}",  # Standard padding
                    f"{info['prefix']}{num:05d}"   # Extended padding
                ]
                
                found_in_verses = any(v in db_strongs for v in variants)
                found_in_lexicon = any(v in lexicon_strongs for v in variants)
                
                if found_in_verses or found_in_lexicon:
                    found_count += 1
                    if not found_in_verses:
                        missing_in_verses.append(f"{info['prefix']}{num:04d}")
                    elif not found_in_lexicon:
                        missing_in_lexicon.append(f"{info['prefix']}{num:04d}")
                else:
                    missing_strongs.append(f"{info['prefix']}{num:04d}")
            
            expected_count = info['end'] - info['start'] + 1
            total_expected += expected_count
            total_missing += len(missing_strongs)
            
            # Report statistics
            print(f"\n📊 Statistics for {language.title()}:")
            print(f"  • Expected Strong's numbers: {expected_count:,}")
            print(f"  • Found in database: {found_count:,}")
            print(f"  • Missing completely: {len(missing_strongs):,}")
            print(f"  • Missing from verses (but in lexicon): {len(missing_in_verses):,}")
            print(f"  • Missing from lexicon (but in verses): {len(missing_in_lexicon):,}")
            print(f"  • Coverage: {found_count/expected_count*100:.1f}%")
            
            # Show samples of missing numbers
            if missing_strongs:
                print(f"\n❌ Sample missing Strong's numbers (first 20):")
                for strongs in missing_strongs[:20]:
                    # Try to find why it's missing
                    cursor.execute("""
                        SELECT COUNT(*) FROM lexicon 
                        WHERE strongs_number LIKE ?
                    """, (f"%{strongs[1:]}%",))
                    similar_count = cursor.fetchone()[0]
                    
                    if similar_count > 0:
                        print(f"  • {strongs} - Found {similar_count} similar entries")
                    else:
                        print(f"  • {strongs} - No similar entries found")
                
                if len(missing_strongs) > 20:
                    print(f"  ... and {len(missing_strongs) - 20} more")
            
            # Check for unusual patterns
            print(f"\n🔍 Checking for unusual patterns in {language}:")
            
            # Find Strong's numbers outside expected range
            if language == 'hebrew':
                cursor.execute("""
                    SELECT DISTINCT strongs_lexical
                    FROM stepbible_verses
                    WHERE language = ? 
                    AND strongs_lexical IS NOT NULL
                    AND (
                        CAST(SUBSTR(strongs_lexical, 2) AS INTEGER) > ?
                        OR CAST(SUBSTR(strongs_lexical, 2) AS INTEGER) < ?
                    )
                    LIMIT 10
                """, (language, info['end'], info['start']))
            else:
                cursor.execute("""
                    SELECT DISTINCT strongs_lexical
                    FROM stepbible_verses
                    WHERE language = ? 
                    AND strongs_lexical IS NOT NULL
                    AND strongs_lexical NOT LIKE 'G%'
                    LIMIT 10
                """, (language,))
            
            unusual = cursor.fetchall()
            if unusual:
                print(f"  ⚠️  Found Strong's numbers outside expected range:")
                for (strongs,) in unusual:
                    print(f"     • {strongs}")
        
        # Overall summary
        print("\n" + "=" * 80)
        print("📊 OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total expected Strong's numbers: {total_expected:,}")
        print(f"Total missing Strong's numbers: {total_missing:,}")
        print(f"Overall coverage: {(total_expected - total_missing)/total_expected*100:.1f}%")
        
        if total_missing > 0:
            print("\n⚠️  WARNING: Not all Strong's numbers are present in the database!")
            print("This may be due to:")
            print("1. Numbers assigned to variant readings not in the main text")
            print("2. Numbers that were later determined to be duplicates")
            print("3. Numbers reserved but never used")
            print("4. Incomplete data import")
            
        # Check specific known issues
        print("\n🔍 Checking specific known Strong's numbers:")
        test_cases = [
            ("G0025", "ἀγαπάω - to love"),
            ("G0026", "ἀγάπη - love"),
            ("H0157", "אהב - to love"),
            ("H2617", "חסד - lovingkindness"),
            ("G4102", "πίστις - faith"),
            ("G0932", "βασιλεία - kingdom")
        ]
        
        for strongs, description in test_cases:
            cursor.execute("""
                SELECT COUNT(*) FROM stepbible_verses
                WHERE strongs_lexical = ?
            """, (strongs,))
            verse_count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT gloss, definition FROM lexicon
                WHERE strongs_number = ?
            """, (strongs,))
            lexicon_data = cursor.fetchone()
            
            status = "✅" if verse_count > 0 else "❌"
            print(f"{status} {strongs} - {description}")
            print(f"   Verses: {verse_count}, Lexicon: {'Yes' if lexicon_data else 'No'}")
            if lexicon_data:
                print(f"   Gloss: {lexicon_data[0] or 'N/A'}")


def check_data_integrity(db_path: Path):
    """Check for data integrity issues that might affect Strong's coverage."""
    
    print("\n\n🔍 Checking Data Integrity")
    print("=" * 80)
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        # Check for null Strong's numbers where we'd expect them
        cursor.execute("""
            SELECT language, COUNT(*) as count
            FROM stepbible_verses
            WHERE strongs_lexical IS NULL
            AND strongs_primary IS NOT NULL
            GROUP BY language
        """)
        
        null_issues = cursor.fetchall()
        if null_issues:
            print("\n⚠️  Found verses with strongs_primary but no strongs_lexical:")
            for lang, count in null_issues:
                print(f"  • {lang}: {count:,} verses")
        
        # Check for formatting inconsistencies
        print("\n📊 Strong's Number Format Distribution:")
        
        for lang in ['hebrew', 'greek']:
            prefix = 'H' if lang == 'hebrew' else 'G'
            cursor.execute("""
                SELECT 
                    LENGTH(strongs_lexical) - 1 as num_digits,
                    COUNT(*) as count
                FROM stepbible_verses
                WHERE language = ?
                AND strongs_lexical IS NOT NULL
                GROUP BY LENGTH(strongs_lexical)
                ORDER BY num_digits
            """, (lang,))
            
            print(f"\n{lang.title()} digit distribution:")
            for digits, count in cursor.fetchall():
                print(f"  • {prefix} + {digits} digits: {count:,} occurrences")


def suggest_remediation(db_path: Path):
    """Suggest steps to improve Strong's number coverage."""
    
    print("\n\n💡 Remediation Suggestions")
    print("=" * 80)
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        # Check if we need to re-extract Strong's numbers
        cursor.execute("""
            SELECT COUNT(*)
            FROM stepbible_verses
            WHERE strongs_primary IS NOT NULL
            AND strongs_primary != ''
            AND (strongs_lexical IS NULL OR strongs_lexical = '')
        """)
        
        missing_extractions = cursor.fetchone()[0]
        
        if missing_extractions > 0:
            print(f"\n1. Re-extract Strong's numbers:")
            print(f"   • Found {missing_extractions:,} verses with unextracted Strong's")
            print(f"   • Run: python claude/scripts/fix_stepbible_schema.py")
        
        # Check if lexicon is complete
        cursor.execute("""
            SELECT 
                (SELECT COUNT(DISTINCT strongs_lexical) FROM stepbible_verses WHERE strongs_lexical IS NOT NULL) as verse_strongs,
                (SELECT COUNT(DISTINCT strongs_number) FROM lexicon) as lexicon_strongs
        """)
        
        verse_count, lexicon_count = cursor.fetchone()
        
        if verse_count > lexicon_count:
            print(f"\n2. Update lexicon data:")
            print(f"   • Verses reference {verse_count:,} unique Strong's")
            print(f"   • Lexicon contains {lexicon_count:,} entries")
            print(f"   • Missing: {verse_count - lexicon_count:,} lexicon entries")
        
        print("\n3. Consider alternative Strong's numbering schemes:")
        print("   • Some editions use different numbering (e.g., Strong's Enhanced)")
        print("   • Verify which Strong's system STEPBible uses")
        
        print("\n4. Investigate specific missing ranges:")
        print("   • Many missing numbers might be intentional gaps")
        print("   • Check Strong's concordance documentation for reserved/unused numbers")


def main():
    """Run the complete Strong's validation."""
    config = config_manager.load_config()
    db_path = config.data_dir / "abba.db"
    
    try:
        validate_strongs_coverage(db_path)
        check_data_integrity(db_path)
        suggest_remediation(db_path)
        
        print("\n\n✅ Validation complete!")
        print("\nNote: Not all Strong's numbers from 1-8674 (Hebrew) or 1-5624 (Greek)")
        print("were actually used in Strong's original concordance. Many gaps are expected.")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)