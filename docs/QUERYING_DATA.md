# Querying Biblical Data - A Comprehensive Guide

This guide provides detailed examples and patterns for querying the ABBA SQLite database to extract biblical texts, perform linguistic analysis, and conduct word studies.

## Table of Contents

1. [Basic Verse Queries](#basic-verse-queries)
2. [Original Language Analysis](#original-language-analysis)
3. [Strong's Concordance Searches](#strongs-concordance-searches)
4. [Morphological Analysis](#morphological-analysis)
5. [Cross-Translation Comparisons](#cross-translation-comparisons)
6. [Advanced Query Patterns](#advanced-query-patterns)
7. [Python Integration Examples](#python-integration-examples)
8. [Performance Tips](#performance-tips)

## Basic Verse Queries

### Single Verse Lookup

```sql
-- Get John 3:16 in KJV
SELECT text 
FROM verses 
WHERE translation_id = 'eng_kjv' 
  AND book_id = 43      -- John is book 43
  AND chapter = 3 
  AND verse = 16;
```

### Multiple Verses

```sql
-- Get John 3:16-18
SELECT verse, text 
FROM verses 
WHERE translation_id = 'eng_kjv' 
  AND book_id = 43 
  AND chapter = 3 
  AND verse BETWEEN 16 AND 18
ORDER BY verse;
```

### Entire Chapter

```sql
-- Get all of Genesis 1
SELECT verse, text 
FROM verses 
WHERE translation_id = 'eng_kjv' 
  AND book_id = 1       -- Genesis is book 1
  AND chapter = 1
ORDER BY verse;
```

### Book Reference Table

Common book IDs for reference:
- Genesis: 1
- Psalms: 19
- Isaiah: 23
- Matthew: 40
- John: 43
- Romans: 45
- Revelation: 66

### Using Book Names

```sql
-- Join with books table to use names instead of IDs
SELECT v.chapter, v.verse, v.text
FROM verses v
JOIN books b ON v.translation_id = b.translation_id 
             AND v.book_id = b.book_id
WHERE v.translation_id = 'eng_kjv'
  AND b.name = 'John'
  AND v.chapter = 1
  AND v.verse <= 5;
```

## Original Language Analysis

### Greek Text Analysis

```sql
-- Get complete Greek analysis for John 1:1
SELECT 
    w.word_num as position,
    w.greek_text as greek,
    w.transliteration,
    w.strongs_primary as strongs,
    w.translation as parsing_code,
    l.gloss as meaning,
    l.definition
FROM words w
LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
WHERE w.book = 'Jhn' 
  AND w.chapter = 1 
  AND w.verse = 1
ORDER BY w.word_num;
```

Result example:
```
position | greek | transliteration | strongs | meaning
---------+-------+-----------------+---------+---------
1        | Ἐν    | En              | G1722   | in
2        | ἀρχῇ   | archē           | G0746   | beginning
3        | ἦν    | ēn              | G1510   | was
...
```

### Hebrew Text Analysis

```sql
-- Get Hebrew analysis for Genesis 1:1
SELECT 
    w.word_num,
    w.hebrew_text,
    w.transliteration,
    w.strongs_primary,
    l.gloss,
    CASE 
        WHEN w.morphology_code LIKE 'N%' THEN 'Noun'
        WHEN w.morphology_code LIKE 'V%' THEN 'Verb'
        WHEN w.morphology_code LIKE 'P%' THEN 'Preposition'
        ELSE 'Other'
    END as part_of_speech
FROM words w
LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
WHERE w.book = 'Gen' 
  AND w.chapter = 1 
  AND w.verse = 1
ORDER BY w.word_num;
```

### Detailed Morphological Breakdown

```sql
-- Parse morphology codes for Greek verbs
SELECT 
    w.greek_text,
    w.strongs_primary,
    w.translation,
    CASE 
        WHEN w.translation LIKE '%V-P%' THEN 'Present'
        WHEN w.translation LIKE '%V-I%' THEN 'Imperfect'
        WHEN w.translation LIKE '%V-F%' THEN 'Future'
        WHEN w.translation LIKE '%V-A%' THEN 'Aorist'
        WHEN w.translation LIKE '%V-X%' THEN 'Perfect'
        WHEN w.translation LIKE '%V-Y%' THEN 'Pluperfect'
    END as tense,
    CASE
        WHEN w.translation LIKE '%A%I%' THEN 'Active'
        WHEN w.translation LIKE '%M%I%' THEN 'Middle'
        WHEN w.translation LIKE '%P%I%' THEN 'Passive'
    END as voice,
    m.description
FROM words w
LEFT JOIN morphology m ON w.morphology_code = m.code
WHERE w.book = 'Jhn' 
  AND w.chapter = 1
  AND w.translation LIKE '%V-%'  -- Verbs only
ORDER BY w.chapter, w.verse, w.word_num;
```

## Strong's Concordance Searches

### Find All Occurrences of a Word

```sql
-- Find all occurrences of "logos" (G3056)
SELECT 
    w.book,
    w.chapter,
    w.verse,
    w.greek_text,
    w.transliteration,
    w.strongs_primary
FROM words w
WHERE w.translation LIKE 'G3056%'
ORDER BY w.id
LIMIT 20;
```

### Count Occurrences by Book

```sql
-- Count how many times "agape" (G26) appears in each book
SELECT 
    w.book,
    COUNT(*) as occurrences,
    GROUP_CONCAT(DISTINCT w.chapter || ':' || w.verse) as references
FROM words w
WHERE w.translation LIKE 'G0026%'
GROUP BY w.book
ORDER BY occurrences DESC;
```

### Find Word Variations

```sql
-- Find all forms of "agape" love (G25, G26)
SELECT DISTINCT
    w.strongs_primary,
    w.greek_text,
    l.gloss,
    l.part_of_speech,
    COUNT(*) as frequency
FROM words w
LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
WHERE w.strongs_primary IN ('G0025', 'G0026')  -- agapao, agape
GROUP BY w.strongs_primary, w.greek_text, l.gloss, l.part_of_speech
ORDER BY frequency DESC;
```

### Semantic Word Studies

```sql
-- Study "faith" related words
SELECT DISTINCT
    w.strongs_primary,
    l.gloss,
    l.original_word,
    COUNT(*) as uses
FROM words w
JOIN lexicon l ON w.strongs_primary = l.strongs_number
WHERE l.gloss LIKE '%faith%' 
   OR l.gloss LIKE '%believe%'
   OR l.gloss LIKE '%trust%'
GROUP BY w.strongs_primary, l.gloss, l.original_word
ORDER BY uses DESC
LIMIT 15;
```

## Morphological Analysis

### Find Specific Grammatical Forms

```sql
-- Find all present active indicative verbs in Romans 1
SELECT 
    verse,
    word_num,
    greek_text,
    strongs_primary,
    translation
FROM words
WHERE book = 'Rom' 
  AND chapter = 1
  AND translation LIKE '%V-PAI%'  -- Verb-Present Active Indicative
ORDER BY verse, word_num;
```

### Noun Analysis

```sql
-- Analyze all nominative case nouns in John 1:1-5
SELECT 
    verse,
    greek_text,
    strongs_primary,
    CASE
        WHEN translation LIKE '%NSM%' THEN 'Nominative Singular Masculine'
        WHEN translation LIKE '%NSF%' THEN 'Nominative Singular Feminine'
        WHEN translation LIKE '%NSN%' THEN 'Nominative Singular Neuter'
        WHEN translation LIKE '%NPM%' THEN 'Nominative Plural Masculine'
        WHEN translation LIKE '%NPF%' THEN 'Nominative Plural Feminine'
        WHEN translation LIKE '%NPN%' THEN 'Nominative Plural Neuter'
    END as case_number_gender
FROM words
WHERE book = 'Jhn' 
  AND chapter = 1 
  AND verse <= 5
  AND translation LIKE '%N-N%'  -- Noun-Nominative
ORDER BY verse, word_num;
```

### Imperative Commands

```sql
-- Find all imperative commands in Matthew 5-7 (Sermon on the Mount)
SELECT 
    chapter,
    verse,
    greek_text,
    transliteration,
    strongs_primary,
    l.gloss
FROM words w
JOIN lexicon l ON w.strongs_primary = l.strongs_number
WHERE w.book = 'Mat' 
  AND w.chapter BETWEEN 5 AND 7
  AND w.translation LIKE '%V-%M%'  -- Verb-Imperative Mood
ORDER BY w.chapter, w.verse, w.word_num;
```

## Cross-Translation Comparisons

### Compare Verse Translations

```sql
-- Compare how different translations render John 1:1
SELECT 
    t.english_name,
    t.language,
    v.text
FROM verses v
JOIN translations t ON v.translation_id = t.translation_id
WHERE v.book_id = 43 
  AND v.chapter = 1 
  AND v.verse = 1
  AND v.translation_id IN ('eng_kjv', 'eng_asv', 'eng_bbe', 'ENGWEBP', 'eng_dby')
ORDER BY t.year;
```

### Analyze Translation Choices

```sql
-- See how different translations handle a specific Greek word
WITH word_context AS (
    SELECT book, chapter, verse, word_num
    FROM words
    WHERE strongs_primary = 'G3056'  -- logos
    AND book = 'Jhn' AND chapter = 1 AND verse = 1
)
SELECT 
    v.translation_id,
    v.text,
    -- Extract the word at approximate position
    SUBSTR(v.text, 1, 50) as beginning_of_verse
FROM verses v
JOIN word_context w ON v.book_id = 43  -- John
                   AND v.chapter = w.chapter 
                   AND v.verse = w.verse
WHERE v.translation_id IN ('eng_kjv', 'eng_asv', 'eng_bbe')
ORDER BY v.translation_id;
```

### Language Family Comparisons

```sql
-- Compare Germanic language translations
SELECT 
    t.translation_id,
    t.language,
    t.english_name,
    v.text
FROM verses v
JOIN translations t ON v.translation_id = t.translation_id
WHERE v.book_id = 43 AND v.chapter = 3 AND v.verse = 16
  AND t.language IN ('eng', 'deu', 'nld', 'dan', 'swe')
ORDER BY t.language;
```

## Advanced Query Patterns

### Phrase Searches

```sql
-- Find "love one another" phrases
SELECT 
    v1.translation_id,
    v1.book_id,
    v1.chapter,
    v1.verse,
    v1.text
FROM verses v1
WHERE v1.translation_id = 'eng_kjv'
  AND v1.text LIKE '%love one another%'
ORDER BY v1.book_id, v1.chapter, v1.verse;
```

### Parallel Passage Detection

```sql
-- Find parallel accounts (e.g., Sermon on the Mount)
SELECT 
    b.name as book,
    v.chapter,
    MIN(v.verse) as start_verse,
    MAX(v.verse) as end_verse,
    COUNT(*) as verse_count
FROM verses v
JOIN books b ON v.translation_id = b.translation_id AND v.book_id = b.book_id
WHERE v.translation_id = 'eng_kjv'
  AND v.text LIKE '%Blessed are%'
GROUP BY b.name, v.chapter
HAVING verse_count > 3
ORDER BY v.book_id;
```

### Word Proximity Analysis

```sql
-- Find verses where "faith" and "works" appear together
WITH faith_verses AS (
    SELECT DISTINCT book, chapter, verse
    FROM words
    WHERE strongs_primary IN ('G4102', 'G4100')  -- pistis, pisteuo
),
works_verses AS (
    SELECT DISTINCT book, chapter, verse  
    FROM words
    WHERE strongs_primary IN ('G2041')  -- ergon
)
SELECT 
    f.book,
    f.chapter,
    f.verse,
    v.text
FROM faith_verses f
JOIN works_verses w ON f.book = w.book 
                   AND f.chapter = w.chapter 
                   AND f.verse = w.verse
JOIN verses v ON v.book_id = (
    SELECT book_id FROM books 
    WHERE name = f.book 
    AND translation_id = 'eng_kjv'
    LIMIT 1
)
AND v.chapter = f.chapter 
AND v.verse = f.verse
WHERE v.translation_id = 'eng_kjv'
ORDER BY f.book, f.chapter, f.verse;
```

### Linguistic Pattern Analysis

```sql
-- Find chiastic structures (ABBA patterns) in Psalms
WITH verse_words AS (
    SELECT 
        chapter,
        verse,
        GROUP_CONCAT(strongs_primary ORDER BY word_num) as word_pattern
    FROM words
    WHERE book = 'Psa'
    GROUP BY chapter, verse
)
SELECT 
    v1.chapter,
    v1.verse as verse1,
    v2.verse as verse2,
    v1.word_pattern
FROM verse_words v1
JOIN verse_words v2 ON v1.chapter = v2.chapter 
                   AND v1.verse < v2.verse
                   AND v1.word_pattern = v2.word_pattern
LIMIT 10;
```

## Python Integration Examples

### Basic Query Function

```python
import sqlite3
from typing import List, Dict, Any

def get_verse(translation: str, book: int, chapter: int, verse: int) -> str:
    """Get a single verse text."""
    conn = sqlite3.connect('bible_data/abba.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT text 
        FROM verses 
        WHERE translation_id = ? 
          AND book_id = ? 
          AND chapter = ? 
          AND verse = ?
    """, (translation, book, chapter, verse))
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

# Usage
verse_text = get_verse('eng_kjv', 43, 3, 16)
print(verse_text)
```

### Word Study Class

```python
class WordStudy:
    def __init__(self, db_path='bible_data/abba.db'):
        self.db_path = db_path
    
    def find_word_occurrences(self, strongs_number: str) -> List[Dict[str, Any]]:
        """Find all occurrences of a Strong's number."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                w.book,
                w.chapter,
                w.verse,
                w.greek_text,
                w.hebrew_text,
                w.transliteration,
                w.strongs_primary,
                l.gloss,
                l.definition
            FROM words w
            LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
            WHERE w.translation LIKE ? || '%'
            ORDER BY w.id
            LIMIT 100
        """, (strongs_number,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_word_frequency(self, strongs_number: str) -> Dict[str, int]:
        """Get frequency count by book."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT book, COUNT(*) as count
            FROM words
            WHERE translation LIKE ? || '%'
            GROUP BY book
            ORDER BY count DESC
        """, (strongs_number,))
        
        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return results

# Usage
study = WordStudy()
occurrences = study.find_word_occurrences('G3056')  # logos
frequency = study.get_word_frequency('G3056')

print(f"Found {len(occurrences)} occurrences")
print(f"Top books: {list(frequency.items())[:5]}")
```

### Morphological Analysis Tool

```python
def analyze_verb_morphology(book: str, chapter: int) -> pd.DataFrame:
    """Analyze all verbs in a chapter."""
    import pandas as pd
    
    conn = sqlite3.connect('bible_data/abba.db')
    
    query = """
    SELECT 
        verse,
        word_num,
        greek_text,
        transliteration,
        strongs_primary,
        translation,
        CASE 
            WHEN translation LIKE '%V-P%' THEN 'Present'
            WHEN translation LIKE '%V-I%' THEN 'Imperfect'
            WHEN translation LIKE '%V-F%' THEN 'Future'
            WHEN translation LIKE '%V-A%' THEN 'Aorist'
            WHEN translation LIKE '%V-X%' THEN 'Perfect'
            ELSE 'Other'
        END as tense,
        CASE
            WHEN translation LIKE '%A%' THEN 'Active'
            WHEN translation LIKE '%M%' THEN 'Middle'
            WHEN translation LIKE '%P%' THEN 'Passive'
            ELSE 'Unknown'
        END as voice
    FROM words
    WHERE book = ?
      AND chapter = ?
      AND translation LIKE '%V-%'
    ORDER BY verse, word_num
    """
    
    df = pd.read_sql_query(query, conn, params=(book, chapter))
    conn.close()
    
    return df

# Usage
verb_analysis = analyze_verb_morphology('Jhn', 1)
print(verb_analysis.groupby(['tense', 'voice']).size())
```

### Parallel Text Viewer

```python
def get_parallel_verses(book_id: int, chapter: int, verse: int, 
                       translations: List[str]) -> Dict[str, str]:
    """Get the same verse in multiple translations."""
    conn = sqlite3.connect('bible_data/abba.db')
    cursor = conn.cursor()
    
    placeholders = ','.join(['?' for _ in translations])
    
    cursor.execute(f"""
        SELECT translation_id, text
        FROM verses
        WHERE book_id = ?
          AND chapter = ?
          AND verse = ?
          AND translation_id IN ({placeholders})
    """, [book_id, chapter, verse] + translations)
    
    results = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    
    return results

# Usage
parallel = get_parallel_verses(
    43, 1, 1, 
    ['eng_kjv', 'eng_asv', 'eng_bbe', 'grc_byzantine']
)

for translation, text in parallel.items():
    print(f"{translation}: {text}\n")
```

## Performance Tips

### Use Indexes Effectively

```sql
-- Good: Uses index on (book, chapter, verse)
SELECT * FROM words 
WHERE book = 'Jhn' AND chapter = 1 AND verse = 1;

-- Bad: Can't use index effectively
SELECT * FROM words 
WHERE verse = 1;  -- Missing book and chapter
```

### Limit Large Result Sets

```sql
-- Always use LIMIT for exploratory queries
SELECT * FROM words 
WHERE strongs_primary = 'G3056'
LIMIT 100;

-- Get count first if needed
SELECT COUNT(*) FROM words WHERE strongs_primary = 'G3056';
```

### Optimize Joins

```sql
-- Efficient: Join only needed tables
SELECT w.greek_text, l.gloss
FROM words w
JOIN lexicon l ON w.strongs_primary = l.strongs_number
WHERE w.book = 'Jhn' AND w.chapter = 1;

-- Less efficient: Unnecessary joins
SELECT w.*, l.*, m.*, v.*
FROM words w
LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
LEFT JOIN morphology m ON w.morphology_code = m.code
LEFT JOIN verses v ON v.book_id = 43
WHERE w.book = 'Jhn';
```

### Batch Operations

```python
# Good: Single query with multiple results
def get_chapter_words(book: str, chapter: int):
    conn = sqlite3.connect('bible_data/abba.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT verse, word_num, greek_text, strongs_primary
        FROM words
        WHERE book = ? AND chapter = ?
        ORDER BY verse, word_num
    """, (book, chapter))
    
    results = cursor.fetchall()
    conn.close()
    return results

# Bad: Multiple queries in a loop
def get_chapter_words_slow(book: str, chapter: int):
    results = []
    for verse in range(1, 32):  # Assume max 31 verses
        for word in range(1, 50):  # Assume max 50 words
            # DON'T DO THIS - too many queries!
            result = get_single_word(book, chapter, verse, word)
            if result:
                results.append(result)
    return results
```

### Create Summary Tables for Analytics

```sql
-- Create a summary table for word frequencies
CREATE TABLE word_frequency AS
SELECT 
    strongs_primary,
    COUNT(*) as total_occurrences,
    COUNT(DISTINCT book) as books_used,
    COUNT(DISTINCT book || ':' || chapter) as chapters_used
FROM words
WHERE strongs_primary IS NOT NULL
GROUP BY strongs_primary;

CREATE INDEX idx_word_freq_strongs ON word_frequency(strongs_primary);
CREATE INDEX idx_word_freq_count ON word_frequency(total_occurrences DESC);

-- Now queries are much faster
SELECT * FROM word_frequency 
WHERE strongs_primary = 'G3056';
```

## Common Query Templates

### Template 1: Verse Range

```sql
-- Get a range of verses
SELECT chapter, verse, text
FROM verses
WHERE translation_id = ?
  AND book_id = ?
  AND ((chapter = ? AND verse >= ?) 
       OR (chapter > ? AND chapter < ?)
       OR (chapter = ? AND verse <= ?))
ORDER BY chapter, verse;
```

### Template 2: Word Context

```sql
-- Get context around a specific word
WITH target_word AS (
    SELECT book, chapter, verse, word_num
    FROM words
    WHERE strongs_primary = ?
    LIMIT 1
)
SELECT 
    w.word_num,
    w.greek_text,
    w.transliteration,
    w.strongs_primary,
    CASE 
        WHEN w.word_num = t.word_num THEN '***'
        ELSE ''
    END as marker
FROM words w
CROSS JOIN target_word t
WHERE w.book = t.book
  AND w.chapter = t.chapter
  AND w.verse = t.verse
ORDER BY w.word_num;
```

### Template 3: Statistical Analysis

```sql
-- Get statistical summary for a book
SELECT 
    'Total Verses' as metric,
    COUNT(DISTINCT chapter || ':' || verse) as value
FROM verses
WHERE translation_id = 'eng_kjv' AND book_id = ?
UNION ALL
SELECT 
    'Total Words',
    COUNT(*)
FROM words
WHERE book = ?
UNION ALL
SELECT 
    'Unique Greek/Hebrew Words',
    COUNT(DISTINCT strongs_primary)
FROM words
WHERE book = ?;
```

## Conclusion

This guide covers the most common query patterns for biblical data analysis. The key principles are:

1. **Use indexes** - Structure queries to leverage existing indexes
2. **Join wisely** - Only join tables you need
3. **Limit results** - Use LIMIT for exploration
4. **Batch operations** - Minimize database round trips
5. **Know your data** - Understand the schema and relationships

For more examples, see `claude/scripts/simple_db_examples.py` in the repository.