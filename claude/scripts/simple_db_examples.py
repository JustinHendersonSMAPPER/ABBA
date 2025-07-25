#!/usr/bin/env python3
"""
Simple examples showing SQLite database capabilities for:
1. Verse retrieval
2. Original language access
"""

import sqlite3
from pathlib import Path

def example_1_basic_verse_retrieval():
    """Example 1: Retrieve a verse from different translations."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Verse Retrieval")
    print("="*60)
    
    # Connect to database
    db = sqlite3.connect("bible_data/abba.db")
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    
    # Get John 3:16 in different translations
    # Note: book_id 43 is John in standard Bible ordering
    query = """
    SELECT translation_id, text 
    FROM verses 
    WHERE book_id = 43 AND chapter = 3 AND verse = 16
    AND translation_id IN ('eng_kjv', 'eng_asv', 'ENGWEBP', 'eng_bbe')
    """
    
    print("\nJohn 3:16 in different translations:")
    print("-" * 60)
    
    for row in cursor.execute(query):
        print(f"\n{row['translation_id']}:")
        print(f"{row['text']}")
    
    db.close()

def example_2_original_language_meaning():
    """Example 2: Access original Greek/Hebrew with meanings."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Complete Linguistic Analysis")
    print("="*60)
    
    # Connect to database
    db = sqlite3.connect("bible_data/abba.db")
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    
    # Get ALL linguistic data for John 1:1
    query = """
    SELECT 
        w.word_num,
        w.word_ref,
        w.greek_text as greek,
        w.transliteration,
        w.strongs_primary as english_gloss,
        w.translation as full_code,
        w.strongs_raw,
        w.morphology_code,
        SUBSTR(w.translation, 1, 5) as strongs_clean,
        l.gloss as lexicon_gloss,
        l.definition as lexicon_definition,
        l.transliteration as lexicon_translit,
        l.part_of_speech as lexicon_pos,
        l.original_word as lexicon_original,
        m.description as morph_description
    FROM words w
    LEFT JOIN lexicon l ON SUBSTR(w.translation, 1, 5) = l.strongs_number
    LEFT JOIN morphology m ON w.morphology_code = m.code
    WHERE w.book = 'Jhn' AND w.chapter = 1 AND w.verse = 1
    ORDER BY w.word_num
    """
    
    print("\nJohn 1:1 - Complete verse breakdown:")
    print("=" * 80)
    
    results = list(cursor.execute(query))
    
    if results:
        # Display the full Greek text
        greek_text = " ".join([row['greek'] for row in results])
        print(f"\nGreek text: {greek_text}")
        
        # Display transliteration
        translit_text = " ".join([row['transliteration'] for row in results])
        print(f"Transliteration: {translit_text}")
        
        # Display word-for-word English
        english_text = " ".join([row['english_gloss'] for row in results])
        print(f"Word-for-word: {english_text}")
        
        print("\n" + "=" * 80)
        print("DETAILED WORD-BY-WORD ANALYSIS:")
        print("=" * 80)
        
        for row in results:
            print(f"\n{'─' * 70}")
            print(f"WORD {row['word_num']}: {row['greek']} ({row['transliteration']})")
            print(f"{'─' * 70}")
            
            # Basic Information
            print("\n▶ BASIC INFORMATION:")
            print(f"  Reference ID: {row['word_ref']}")
            print(f"  English gloss: {row['english_gloss']}")
            print(f"  Full parsing code: {row['full_code']}")
            if row['strongs_raw'] and row['strongs_raw'] != row['full_code']:
                print(f"  Raw Strong's data: {row['strongs_raw']}")
            
            # Strong's Number Analysis
            print("\n▶ STRONG'S NUMBER:")
            print(f"  Number: {row['strongs_clean']}")
            if row['lexicon_gloss']:
                print(f"  Lexicon gloss: {row['lexicon_gloss']}")
            if row['lexicon_pos']:
                print(f"  Part of speech: {row['lexicon_pos']}")
            if row['lexicon_original'] and row['lexicon_original'] != row['greek']:
                print(f"  Dictionary form: {row['lexicon_original']}")
            if row['lexicon_translit'] and row['lexicon_translit'] != row['transliteration']:
                print(f"  Dictionary transliteration: {row['lexicon_translit']}")
            
            # Full Lexicon Definition
            if row['lexicon_definition']:
                print("\n▶ LEXICON DEFINITION:")
                # Split long definitions into readable chunks
                def_lines = []
                definition = row['lexicon_definition']
                while len(definition) > 70:
                    # Find a good break point
                    break_point = definition[:70].rfind(' ')
                    if break_point == -1:
                        break_point = 70
                    def_lines.append(definition[:break_point])
                    definition = definition[break_point:].strip()
                if definition:
                    def_lines.append(definition)
                for line in def_lines:
                    print(f"  {line}")
            
            # Morphological Analysis
            print("\n▶ MORPHOLOGICAL ANALYSIS:")
            if '=' in row['full_code']:
                strongs, morph_code = row['full_code'].split('=', 1)
                print(f"  Morphology code: {morph_code}")
                
                # Decode the morphology code
                if morph_code.startswith('N-'):
                    print("  Part of speech: NOUN")
                    parts = morph_code.split('-')
                    if len(parts) > 1:
                        case_info = parts[1]
                        if len(case_info) >= 3:
                            # Parse case, number, gender
                            cases = {'N': 'Nominative', 'G': 'Genitive', 'D': 'Dative', 
                                    'A': 'Accusative', 'V': 'Vocative'}
                            numbers = {'S': 'Singular', 'P': 'Plural'}
                            genders = {'M': 'Masculine', 'F': 'Feminine', 'N': 'Neuter'}
                            
                            if case_info[0] in cases:
                                print(f"  Case: {cases.get(case_info[0], case_info[0])}")
                            if len(case_info) > 1 and case_info[1] in numbers:
                                print(f"  Number: {numbers.get(case_info[1], case_info[1])}")
                            if len(case_info) > 2 and case_info[2] in genders:
                                print(f"  Gender: {genders.get(case_info[2], case_info[2])}")
                            
                            # Check for additional markers
                            if len(parts) > 2:
                                if 'P' in parts[2]:
                                    print("  Type: Proper noun")
                                if 'T' in parts[2]:
                                    print("  Type: Title")
                                    
                elif morph_code.startswith('V-'):
                    print("  Part of speech: VERB")
                    parts = morph_code.split('-')
                    if len(parts) > 1:
                        verb_info = parts[1]
                        # Decode tense, voice, mood
                        if len(verb_info) >= 3:
                            tenses = {'P': 'Present', 'I': 'Imperfect', 'F': 'Future', 
                                     'A': 'Aorist', 'X': 'Perfect', 'Y': 'Pluperfect'}
                            voices = {'A': 'Active', 'M': 'Middle', 'P': 'Passive'}
                            moods = {'I': 'Indicative', 'S': 'Subjunctive', 'O': 'Optative',
                                    'M': 'Imperative', 'N': 'Infinitive', 'P': 'Participle'}
                            
                            # Special handling for different verb patterns
                            if verb_info == 'IAI':
                                print("  Tense: Imperfect")
                                print("  Voice: Active") 
                                print("  Mood: Indicative")
                            elif len(verb_info) >= 3:
                                if verb_info[0] in tenses:
                                    print(f"  Tense: {tenses.get(verb_info[0], verb_info[0])}")
                                if len(verb_info) > 1 and verb_info[1] in voices:
                                    print(f"  Voice: {voices.get(verb_info[1], verb_info[1])}")
                                if len(verb_info) > 2 and verb_info[2] in moods:
                                    print(f"  Mood: {moods.get(verb_info[2], verb_info[2])}")
                            
                            # Person and number for finite verbs
                            if len(parts) > 2:
                                person_info = parts[2]
                                persons = {'1': '1st person', '2': '2nd person', '3': '3rd person'}
                                numbers = {'S': 'Singular', 'P': 'Plural'}
                                if len(person_info) >= 2:
                                    if person_info[0] in persons:
                                        print(f"  Person: {persons[person_info[0]]}")
                                    if person_info[1] in numbers:
                                        print(f"  Number: {numbers[person_info[1]]}")
                                        
                elif morph_code.startswith('P-'):
                    print("  Part of speech: PREPOSITION")
                    if '-' in morph_code:
                        parts = morph_code.split('-')
                        if len(parts) > 1 and parts[1]:
                            print(f"  Additional info: {parts[1]}")
                            
                elif morph_code.startswith('T-'):
                    print("  Part of speech: ARTICLE")
                    parts = morph_code.split('-')
                    if len(parts) > 1:
                        art_info = parts[1]
                        if len(art_info) >= 3:
                            cases = {'N': 'Nominative', 'G': 'Genitive', 'D': 'Dative', 
                                    'A': 'Accusative'}
                            numbers = {'S': 'Singular', 'P': 'Plural'}
                            genders = {'M': 'Masculine', 'F': 'Feminine', 'N': 'Neuter'}
                            
                            if art_info[0] in cases:
                                print(f"  Case: {cases.get(art_info[0], art_info[0])}")
                            if len(art_info) > 1 and art_info[1] in numbers:
                                print(f"  Number: {numbers.get(art_info[1], art_info[1])}")
                            if len(art_info) > 2 and art_info[2] in genders:
                                print(f"  Gender: {genders.get(art_info[2], art_info[2])}")
                                
                elif morph_code.startswith('C-'):
                    print("  Part of speech: CONJUNCTION")
                elif morph_code.startswith('D-'):
                    print("  Part of speech: ADVERB")
                elif morph_code.startswith('A-'):
                    print("  Part of speech: ADJECTIVE")
                    
            if row['morphology_code']:
                print(f"  Morphology table code: {row['morphology_code']}")
            if row['morph_description']:
                print(f"  Morphology description: {row['morph_description']}")
    else:
        print("No data found for John 1:1")
    
    # Show how different translations render John 1:1
    print("\n\nTranslation comparison for John 1:1:")
    print("-" * 80)
    
    translation_query = """
    SELECT translation_id, text
    FROM verses
    WHERE book_id = 43 AND chapter = 1 AND verse = 1
    AND translation_id IN ('eng_kjv', 'eng_asv', 'ENGWEBP', 'eng_bbe', 'eng_dby')
    ORDER BY translation_id
    """
    
    for row in cursor.execute(translation_query):
        print(f"\n{row['translation_id']}:")
        print(f"   {row['text']}")
    
    db.close()

def example_3_word_usage_frequency():
    """Example 3: Track usage of 'logos' (Word) from John 1:1."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Usage of 'logos' (Word) Throughout Scripture")
    print("="*60)
    
    # Connect to database
    db = sqlite3.connect("bible_data/abba.db")
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    
    # Find usage of "logos" (Word) - Strong's G3056 from John 1:1
    query = """
    SELECT 
        COUNT(*) as total_uses,
        COUNT(DISTINCT book || ':' || chapter) as chapters_used,
        COUNT(DISTINCT book) as books_used,
        GROUP_CONCAT(DISTINCT book) as books
    FROM words
    WHERE translation LIKE 'G3056%'
    """
    
    result = cursor.execute(query).fetchone()
    
    print(f"\nUsage of 'logos' (λόγος) - Strong's G3056:")
    print("-" * 60)
    print(f"Total occurrences: {result['total_uses']}")
    print(f"Appears in {result['chapters_used']} different chapters")
    print(f"Appears in {result['books_used']} different books")
    
    # Show breakdown by book
    book_breakdown = """
    SELECT book, COUNT(*) as count
    FROM words
    WHERE translation LIKE 'G3056%'
    GROUP BY book
    ORDER BY count DESC
    LIMIT 10
    """
    
    print("\nTop 10 books using 'logos':")
    for row in cursor.execute(book_breakdown):
        print(f"  {row['book']}: {row['count']} times")
    
    # Show different meanings/translations
    meanings = """
    SELECT DISTINCT strongs_primary as meaning, COUNT(*) as count
    FROM words
    WHERE translation LIKE 'G3056%'
    GROUP BY strongs_primary
    ORDER BY count DESC
    LIMIT 10
    """
    
    print("\nDifferent translations of 'logos':")
    for row in cursor.execute(meanings):
        print(f"  '{row['meaning']}': {row['count']} times")
    
    # Show some example uses in context
    examples = """
    SELECT 
        book || ' ' || chapter || ':' || verse as reference,
        greek_text as greek,
        strongs_primary as english_gloss,
        transliteration
    FROM words
    WHERE translation LIKE 'G3056%'
    AND book IN ('Jhn', 'Mat', 'Luk', 'Rev')
    ORDER BY 
        CASE book 
            WHEN 'Jhn' THEN 1 
            WHEN 'Mat' THEN 2 
            WHEN 'Luk' THEN 3 
            WHEN 'Rev' THEN 4 
        END,
        chapter, verse
    LIMIT 5
    """
    
    print("\n\nExample occurrences in key passages:")
    print("-" * 60)
    for row in cursor.execute(examples):
        print(f"\n{row['reference']}: {row['greek']} ({row['transliteration']})")
        print(f"   Translation: {row['english_gloss']}")
    
    db.close()

def main():
    """Run all examples."""
    # Check if database exists
    if not Path("bible_data/abba.db").exists():
        print("Error: Database not found at bible_data/abba.db")
        print("Please run 'python abba/main.py' first to create the database.")
        return
    
    print("\n" + "="*60)
    print("SQLite Database Examples - ABBA Project")
    print("="*60)
    
    example_1_basic_verse_retrieval()
    example_2_original_language_meaning()
    example_3_word_usage_frequency()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)

if __name__ == "__main__":
    main()