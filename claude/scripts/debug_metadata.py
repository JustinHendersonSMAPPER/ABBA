#!/usr/bin/env python3
"""Debug script to identify None values in word metadata."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database import SQLiteManager


def debug_word_metadata():
    """Debug word data to find None values."""
    print("=== Debugging Word Metadata ===\n")
    
    # Load configuration
    config = config_manager.load_config()
    db_manager = SQLiteManager(config.abba_db_path)
    
    # Get sample words to check for None values
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT 
                w.strongs_primary,
                w.morphology_code,
                w.language,
                w.greek_text,
                w.hebrew_text,
                w.transliteration,
                l.gloss,
                l.part_of_speech,
                COUNT(*) as frequency
            FROM words w
            LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
            WHERE w.strongs_primary IS NOT NULL
              AND w.strongs_primary != ''
            GROUP BY w.strongs_primary, w.morphology_code, w.language
            ORDER BY frequency DESC
            LIMIT 20
        """
        
        cursor.execute(query)
        words = cursor.fetchall()
        
        print(f"Found {len(words)} unique word combinations\n")
        
        print("Sample words with their metadata:")
        print("-" * 120)
        print(f"{'Strong\'s':<10} {'Morph':<15} {'Lang':<6} {'Greek':<15} {'Hebrew':<15} {'Trans':<15} {'Gloss':<20} {'POS':<10}")
        print("-" * 120)
        
        none_fields = {
            'strongs_primary': 0,
            'morphology_code': 0,
            'language': 0,
            'greek_text': 0,
            'hebrew_text': 0,
            'transliteration': 0,
            'gloss': 0,
            'part_of_speech': 0
        }
        
        for i, word in enumerate(words):
            strongs_primary = word[0]
            morphology_code = word[1]
            language = word[2]
            greek_text = word[3]
            hebrew_text = word[4]
            transliteration = word[5]
            gloss = word[6]
            part_of_speech = word[7]
            frequency = word[8]
            
            # Count None values
            if strongs_primary is None:
                none_fields['strongs_primary'] += 1
            if morphology_code is None:
                none_fields['morphology_code'] += 1
            if language is None:
                none_fields['language'] += 1
            if greek_text is None:
                none_fields['greek_text'] += 1
            if hebrew_text is None:
                none_fields['hebrew_text'] += 1
            if transliteration is None:
                none_fields['transliteration'] += 1
            if gloss is None:
                none_fields['gloss'] += 1
            if part_of_speech is None:
                none_fields['part_of_speech'] += 1
            
            # Show first 10 with None indicators
            if i < 10:
                def format_field(value, width):
                    if value is None:
                        return f"{'<NULL>':<{width}}"
                    return f"{str(value)[:width-1]:<{width}}"
                
                print(f"{format_field(strongs_primary, 10)} "
                      f"{format_field(morphology_code, 15)} "
                      f"{format_field(language, 6)} "
                      f"{format_field(greek_text, 15)} "
                      f"{format_field(hebrew_text, 15)} "
                      f"{format_field(transliteration, 15)} "
                      f"{format_field(gloss, 20)} "
                      f"{format_field(part_of_speech, 10)}")
        
        print("\n" + "=" * 60)
        print("None Value Counts:")
        print("=" * 60)
        for field, count in none_fields.items():
            if count > 0:
                print(f"  {field}: {count} None values")
            else:
                print(f"  {field}: No None values ✓")
        
        print(f"\nTotal words analyzed: {len(words)}")
        
        # Test metadata creation
        print("\n" + "=" * 60)
        print("Testing Metadata Creation:")
        print("=" * 60)
        
        if words:
            word = words[0]  # Test first word
            word_dict = {
                'strongs_primary': word[0],
                'morphology_code': word[1],
                'language': word[2],
                'greek_text': word[3],
                'hebrew_text': word[4],
                'transliteration': word[5],
                'gloss': word[6],
                'part_of_speech': word[7]
            }
            
            print("Original word data:")
            for key, value in word_dict.items():
                print(f"  {key}: {value} ({type(value).__name__})")
            
            # Test metadata creation with old logic
            print("\nOld metadata (might have None):")
            old_metadata = {
                "strongs": word_dict['strongs_primary'],
                "morphology": word_dict.get('morphology_code', ''),
                "language": word_dict['language'],
                "word": word_dict.get('greek_text') or word_dict.get('hebrew_text', ''),
                "transliteration": word_dict.get('transliteration', ''),
                "gloss": word_dict.get('gloss', ''),
                "part_of_speech": word_dict.get('part_of_speech', '')
            }
            
            for key, value in old_metadata.items():
                print(f"  {key}: {value} ({type(value).__name__})")
                if value is None:
                    print(f"    ⚠️  {key} is None!")
            
            # Test metadata creation with new logic
            print("\nNew metadata (no None values):")
            new_metadata = {
                "strongs": word_dict.get('strongs_primary', '') or '',
                "morphology": word_dict.get('morphology_code', '') or '',
                "language": word_dict.get('language', '') or '',
                "word": word_dict.get('greek_text', '') or word_dict.get('hebrew_text', '') or '',
                "transliteration": word_dict.get('transliteration', '') or '',
                "gloss": word_dict.get('gloss', '') or '',
                "part_of_speech": word_dict.get('part_of_speech', '') or ''
            }
            
            for key, value in new_metadata.items():
                print(f"  {key}: {value} ({type(value).__name__})")
                if value is None:
                    print(f"    ❌ {key} is still None!")
                else:
                    print(f"    ✓ {key} is safe")
    
    print("\n✓ Metadata debugging complete!")


if __name__ == "__main__":
    debug_word_metadata()