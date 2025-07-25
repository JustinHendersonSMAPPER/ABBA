#!/usr/bin/env python3
"""Analyze cross-linguistic verse alignment challenges."""

import sqlite3
from pathlib import Path
from collections import defaultdict

def analyze_verse_divisions():
    """Analyze how verse divisions differ across languages."""
    
    bible_db = Path("bible_data/bible.db")
    if not bible_db.exists():
        print("bible.db not found")
        return
    
    with sqlite3.connect(bible_db) as conn:
        cursor = conn.cursor()
        
        print("VERSE DIVISION ANALYSIS")
        print("="*60)
        
        # Compare verse counts for the same book/chapter across languages
        test_books = [
            ('GEN', 1, "Genesis 1"),  # Creation account
            ('PSA', 23, "Psalm 23"),  # Well-known psalm
            ('MAT', 5, "Matthew 5"),  # Sermon on the Mount
        ]
        
        for book_id, chapter, desc in test_books:
            print(f"\n{desc}:")
            
            # Get verse counts by language
            cursor.execute("""
                SELECT 
                    t.language,
                    t.id as translation_id,
                    COUNT(DISTINCT cv.number) as verse_count,
                    MIN(cv.number) as min_verse,
                    MAX(cv.number) as max_verse
                FROM ChapterVerse cv
                JOIN Translation t ON cv.translationId = t.id
                WHERE cv.bookId = ? AND cv.chapterNumber = ?
                GROUP BY t.language, t.id
                HAVING verse_count > 0
                ORDER BY t.language
                LIMIT 20
            """, (book_id, chapter))
            
            results = cursor.fetchall()
            
            # Group by language
            by_language = defaultdict(list)
            for lang, trans_id, count, min_v, max_v in results:
                by_language[lang].append((trans_id, count, min_v, max_v))
            
            # Show variations
            for lang, translations in sorted(by_language.items())[:10]:
                counts = {count for _, count, _, _ in translations}
                if len(counts) > 1:
                    print(f"  {lang}: VARIES - {counts} verses")
                else:
                    count = translations[0][1]
                    print(f"  {lang}: {count} verses")

def analyze_text_length_variations():
    """Analyze how text length varies across languages for the same verse."""
    
    bible_db = Path("bible_data/bible.db")
    with sqlite3.connect(bible_db) as conn:
        cursor = conn.cursor()
        
        print("\n\nTEXT LENGTH VARIATION ANALYSIS")
        print("="*60)
        print("Comparing John 3:16 across languages:")
        
        # Get John 3:16 in multiple languages
        cursor.execute("""
            SELECT 
                t.language,
                t.englishName,
                cv.text,
                LENGTH(cv.text) as text_length
            FROM ChapterVerse cv
            JOIN Translation t ON cv.translationId = t.id
            WHERE cv.bookId = 'JHN' AND cv.chapterNumber = 3 AND cv.number = 16
            AND cv.text != ''
            ORDER BY LENGTH(cv.text)
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        for lang, name, text, length in results:
            print(f"\n{lang} ({name}): {length} chars")
            print(f"  {text[:100]}{'...' if len(text) > 100 else ''}")

def analyze_original_language_alignment():
    """Analyze challenges in aligning original language texts."""
    
    print("\n\nORIGINAL LANGUAGE ALIGNMENT CHALLENGES")
    print("="*60)
    
    challenges = [
        {
            "issue": "Word Order Differences",
            "example": "Hebrew VSO vs English SVO",
            "impact": "Single Hebrew word may map to multiple English words in different positions"
        },
        {
            "issue": "Morphological Complexity",
            "example": "Hebrew וּבְתוֹרָתוֹ = 'and-in-law-his' = 'and in his law'",
            "impact": "One Hebrew word requires 4 English words"
        },
        {
            "issue": "Semantic Range",
            "example": "Greek λόγος (logos) = word/reason/account/speech",
            "impact": "Context determines translation, not 1:1 mapping"
        },
        {
            "issue": "Cultural Concepts",
            "example": "Hebrew חֶסֶד (chesed) = lovingkindness/mercy/loyalty",
            "impact": "No single word equivalent in many languages"
        },
        {
            "issue": "Verse Division Traditions",
            "example": "Masoretic vs Septuagint verse numbering",
            "impact": "Same text has different verse numbers in different traditions"
        }
    ]
    
    for i, challenge in enumerate(challenges, 1):
        print(f"\n{i}. {challenge['issue']}")
        print(f"   Example: {challenge['example']}")
        print(f"   Impact: {challenge['impact']}")

def propose_alignment_strategies():
    """Propose strategies for accurate cross-linguistic alignment."""
    
    print("\n\nPROPOSED ALIGNMENT STRATEGIES")
    print("="*60)
    
    strategies = [
        {
            "name": "Multi-Level Alignment",
            "description": "Align at word, phrase, and verse levels simultaneously",
            "implementation": """
            - Word-level: Strong's numbers for Hebrew/Greek
            - Phrase-level: Syntactic units (subject, verb, object)
            - Verse-level: Traditional verse numbers as fallback
            """,
            "pros": "Flexible, preserves granularity",
            "cons": "Complex to implement"
        },
        {
            "name": "Reference-Based Alignment",
            "description": "Use canonical references (book.chapter.verse.word)",
            "implementation": """
            - Each word gets unique reference: Gen.1.1.1, Gen.1.1.2
            - Translations map their words to source references
            - Allows many-to-many mappings
            """,
            "pros": "Precise alignment possible",
            "cons": "Requires extensive mapping data"
        },
        {
            "name": "Semantic Unit Alignment",
            "description": "Align by meaning units rather than verses",
            "implementation": """
            - Identify semantic boundaries (complete thoughts)
            - May span multiple verses or split single verses
            - Language-specific segmentation
            """,
            "pros": "More natural for non-Western languages",
            "cons": "Subjective, requires linguistic expertise"
        },
        {
            "name": "Parallel Text Alignment",
            "description": "Statistical alignment using parallel corpora",
            "implementation": """
            - Use algorithms like IBM Model 1-5, GIZA++
            - Learn word alignments from parallel texts
            - Probabilistic rather than deterministic
            """,
            "pros": "Can handle any language pair",
            "cons": "Requires training data, not 100% accurate"
        }
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']}")
        print(f"Description: {strategy['description']}")
        print(f"Implementation:{strategy['implementation']}")
        print(f"Pros: {strategy['pros']}")
        print(f"Cons: {strategy['cons']}")

def analyze_current_schema_limitations():
    """Analyze limitations of current database schema for alignment."""
    
    print("\n\nCURRENT SCHEMA LIMITATIONS")
    print("="*60)
    
    limitations = [
        "1. Assumes verse-level granularity only",
        "2. No provision for word-level alignment",
        "3. No support for many-to-many verse mappings",
        "4. Cannot handle split or merged verses",
        "5. No semantic unit boundaries",
        "6. Language-specific features not captured"
    ]
    
    for limitation in limitations:
        print(limitation)
    
    print("\n\nSUGGESTED SCHEMA ENHANCEMENTS")
    print("-"*40)
    
    print("""
    -- Word-level alignment table
    CREATE TABLE word_alignments (
        id INTEGER PRIMARY KEY,
        source_word_id INTEGER,  -- from 'words' table
        target_translation_id TEXT,
        target_book_id INTEGER,
        target_chapter INTEGER,
        target_verse INTEGER,
        target_word_position INTEGER,
        target_text TEXT,
        alignment_confidence REAL,
        alignment_method TEXT,  -- manual, statistical, rule-based
        FOREIGN KEY (source_word_id) REFERENCES words(id)
    );
    
    -- Semantic unit boundaries
    CREATE TABLE semantic_units (
        id INTEGER PRIMARY KEY,
        translation_id TEXT,
        book_id INTEGER,
        start_chapter INTEGER,
        start_verse INTEGER,
        start_word INTEGER,
        end_chapter INTEGER,
        end_verse INTEGER,
        end_word INTEGER,
        unit_type TEXT,  -- sentence, paragraph, pericope, discourse
        description TEXT
    );
    
    -- Cross-reference mappings for different versification systems
    CREATE TABLE versification_mappings (
        id INTEGER PRIMARY KEY,
        from_system TEXT,  -- LXX, MT, Vulgate, etc.
        to_system TEXT,
        from_book TEXT,
        from_chapter INTEGER,
        from_verse INTEGER,
        to_book TEXT,
        to_chapter INTEGER,
        to_verse INTEGER,
        mapping_type TEXT  -- split, merge, reorder, equivalent
    );
    """)

if __name__ == "__main__":
    analyze_verse_divisions()
    analyze_text_length_variations()
    analyze_original_language_alignment()
    propose_alignment_strategies()
    analyze_current_schema_limitations()