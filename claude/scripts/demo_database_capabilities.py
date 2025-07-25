#!/usr/bin/env python3
"""
Demonstrates how the SQLite database supports:
1. Retrieving specific book verses
2. Accessing original language meanings
3. Morphological analysis
4. Cross-translation comparison
"""

import sqlite3
import json
from pathlib import Path
from abba.config import ABBAConfig
from abba.database.sqlite_manager import SQLiteManager
from abba.api.search import SearchAPI
from abba.api.analysis import AnalysisAPI

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)

def demo_verse_retrieval():
    """Demonstrate basic verse retrieval from different translations."""
    print_section("1. Basic Verse Retrieval")
    
    # Initialize database connection
    config = ABBAConfig()
    config.database_path = "bible_data/abba.db"
    db = SQLiteManager(config)
    api = SearchAPI(db)
    
    # Example: Get John 3:16 in multiple translations
    verse_ref = "John 3:16"
    translations = ["KJV", "ESV", "NIV", "WEB"]
    
    print(f"\nRetrieving {verse_ref} in multiple translations:")
    print("-" * 60)
    
    for translation in translations:
        try:
            result = api.get_verse("JHN", 3, 16, translation=translation)
            if result:
                print(f"\n{translation}:")
                print(f"  {result['text']}")
        except Exception as e:
            print(f"\n{translation}: Not available ({e})")

def demo_original_language_access():
    """Demonstrate accessing original Hebrew/Greek text with meanings."""
    print_section("2. Original Language Access")
    
    # Initialize database connection
    config = ABBAConfig()
    config.database_path = "bible_data/abba.db"
    db = SQLiteManager(config)
    api = SearchAPI(db)
    
    # Example 1: Hebrew from Genesis 1:1
    print("\nHebrew Example - Genesis 1:1:")
    print("-" * 60)
    
    # Get the Hebrew words for Genesis 1:1
    hebrew_words = db.execute("""
        SELECT 
            word_id,
            hebrew,
            transliteration,
            strongs_number,
            morphology,
            english_gloss
        FROM words
        WHERE book = 'GEN' AND chapter = 1 AND verse = 1
        ORDER BY word_position
    """)
    
    print("Word-by-word breakdown:")
    for i, word in enumerate(hebrew_words, 1):
        print(f"\n{i}. {word['hebrew']} ({word['transliteration']})")
        print(f"   Strong's: {word['strongs_number']}")
        print(f"   Morphology: {word['morphology']}")
        print(f"   English: {word['english_gloss']}")
        
        # Get detailed lexicon entry
        if word['strongs_number']:
            lexicon = db.execute("""
                SELECT brief_gloss, full_definition
                FROM lexicon
                WHERE strongs_number = ?
            """, (word['strongs_number'],))
            
            if lexicon:
                lex = lexicon[0]
                print(f"   Definition: {lex['brief_gloss']}")
                if lex['full_definition']:
                    # Show first 100 chars of full definition
                    full_def = lex['full_definition'][:100]
                    if len(lex['full_definition']) > 100:
                        full_def += "..."
                    print(f"   Full: {full_def}")
    
    # Example 2: Greek from John 1:1
    print("\n\nGreek Example - John 1:1:")
    print("-" * 60)
    
    greek_words = db.execute("""
        SELECT 
            word_id,
            greek,
            transliteration,
            strongs_number,
            morphology,
            english_gloss
        FROM words
        WHERE book = 'JHN' AND chapter = 1 AND verse = 1
        ORDER BY word_position
        LIMIT 5
    """)
    
    print("First 5 words breakdown:")
    for i, word in enumerate(greek_words, 1):
        print(f"\n{i}. {word['greek']} ({word['transliteration']})")
        print(f"   Strong's: {word['strongs_number']}")
        print(f"   Morphology: {word['morphology']}")
        print(f"   English: {word['english_gloss']}")

def demo_morphological_analysis():
    """Demonstrate morphological pattern searching."""
    print_section("3. Morphological Analysis")
    
    config = ABBAConfig()
    config.database_path = "bible_data/abba.db"
    db = SQLiteManager(config)
    api = SearchAPI(db)
    analysis = AnalysisAPI(db)
    
    # Example: Find all Qal perfect verbs in Genesis 1
    print("\nFinding Qal perfect verbs in Genesis 1:")
    print("-" * 60)
    
    # Search for Hebrew verbs with Qal stem and perfect aspect
    results = db.execute("""
        SELECT DISTINCT
            hebrew,
            transliteration,
            strongs_number,
            morphology,
            english_gloss,
            verse
        FROM words
        WHERE book = 'GEN' 
        AND chapter = 1
        AND morphology LIKE '%VqP%'  -- Verb, Qal, Perfect
        ORDER BY verse, word_position
        LIMIT 10
    """)
    
    for word in results:
        print(f"\nVerse {word['verse']}: {word['hebrew']} ({word['transliteration']})")
        print(f"  Strong's: {word['strongs_number']}")
        print(f"  Morphology: {word['morphology']}")
        print(f"  English: {word['english_gloss']}")
        
        # Decode morphology
        morph_info = db.execute("""
            SELECT description
            FROM morphology_codes
            WHERE code = ?
        """, (word['morphology'],))
        
        if morph_info:
            print(f"  Parsed: {morph_info[0]['description']}")

def demo_semantic_search():
    """Demonstrate searching by Strong's numbers and semantic domains."""
    print_section("4. Semantic Search Capabilities")
    
    config = ABBAConfig()
    config.database_path = "bible_data/abba.db"
    db = SQLiteManager(config)
    api = SearchAPI(db)
    
    # Example: Search for all uses of "agape" (G26)
    print("\nSearching for 'agape' (love) - Strong's G26:")
    print("-" * 60)
    
    results = api.search_strongs("G26", limit=5)
    
    for i, result in enumerate(results, 1):
        verse = result['verses'][0]  # Get first verse from results
        print(f"\n{i}. {verse['reference']}")
        print(f"   Greek: {verse['greek_text']}")
        print(f"   Context: {verse['english_text']}")

def demo_word_relationships():
    """Demonstrate finding related words and concepts."""
    print_section("5. Word Relationships & Families")
    
    config = ABBAConfig()
    config.database_path = "bible_data/abba.db"
    db = SQLiteManager(config)
    
    # Example: Find word families based on Hebrew roots
    print("\nFinding words from the root ברא (create):")
    print("-" * 60)
    
    # Search for words with similar Strong's numbers (Hebrew root system)
    results = db.execute("""
        SELECT DISTINCT
            strongs_number,
            hebrew,
            transliteration,
            brief_gloss,
            COUNT(*) as usage_count
        FROM words w
        JOIN lexicon l ON w.strongs_number = l.strongs_number
        WHERE w.strongs_number LIKE 'H125%'  -- ברא root family
        GROUP BY w.strongs_number
        ORDER BY usage_count DESC
    """)
    
    for word in results:
        print(f"\n{word['strongs_number']}: {word['hebrew']} ({word['transliteration']})")
        print(f"  Meaning: {word['brief_gloss']}")
        print(f"  Used {word['usage_count']} times")

def demo_translation_comparison():
    """Demonstrate comparing translations with original text."""
    print_section("6. Translation Comparison with Original")
    
    config = ABBAConfig()
    config.database_path = "bible_data/abba.db"
    db = SQLiteManager(config)
    analysis = AnalysisAPI(db)
    
    # Example: Compare how different translations handle a complex verse
    print("\nComparing translations of Genesis 1:2 (complex Hebrew):")
    print("-" * 60)
    
    # Get the Hebrew text structure
    hebrew_words = db.execute("""
        SELECT 
            hebrew,
            transliteration,
            strongs_number,
            english_gloss
        FROM words
        WHERE book = 'GEN' AND chapter = 1 AND verse = 2
        ORDER BY word_position
    """)
    
    print("Original Hebrew structure:")
    hebrew_text = " ".join([w['hebrew'] for w in hebrew_words])
    translit = " ".join([w['transliteration'] for w in hebrew_words])
    print(f"Hebrew: {hebrew_text}")
    print(f"Transliteration: {translit}")
    print(f"Word-for-word: {' '.join([w['english_gloss'] for w in hebrew_words])}")
    
    # Compare translations
    print("\nTranslation renderings:")
    translations = ["KJV", "ESV", "NIV", "NLT"]
    
    for trans in translations:
        try:
            verse = db.execute("""
                SELECT text
                FROM verses
                WHERE book = 'GEN' AND chapter = 1 AND verse = 2
                AND translation = ?
            """, (trans,))
            
            if verse:
                print(f"\n{trans}: {verse[0]['text']}")
        except:
            pass

def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("ABBA SQLite Database Capabilities Demonstration")
    print("="*60)
    
    try:
        # Check if database exists
        db_path = Path("bible_data/abba.db")
        if not db_path.exists():
            print("\nError: Database not found at bible_data/abba.db")
            print("Please run 'python abba/main.py' first to create the database.")
            return
        
        # Run demonstrations
        demo_verse_retrieval()
        demo_original_language_access()
        demo_morphological_analysis()
        demo_semantic_search()
        demo_word_relationships()
        demo_translation_comparison()
        
        print("\n" + "="*60)
        print("Demonstration Complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()