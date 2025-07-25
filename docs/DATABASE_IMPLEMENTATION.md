# Database Implementation Guide

This document provides comprehensive details about the ABBA SQLite database implementation, including schema design, data flow, and usage patterns.

## Overview

ABBA uses SQLite as its primary data store for biblical texts, linguistic analysis, and cross-references. The database design prioritizes:

- **Fast queries**: Optimized indexes for verse lookups and Strong's number searches
- **Data integrity**: Foreign key constraints and normalized tables
- **Extensibility**: Schema supports future enhancements
- **Efficiency**: Minimal redundancy while maintaining query performance

## Database Schema

### Core Tables

#### 1. verses
Stores all biblical text across multiple translations.

```sql
CREATE TABLE verses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    translation_id TEXT NOT NULL,
    book_id INTEGER NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(translation_id, book_id, chapter, verse)
);

CREATE INDEX idx_verses_reference ON verses(book_id, chapter, verse);
CREATE INDEX idx_verses_translation ON verses(translation_id);
```

**Fields:**
- `translation_id`: Unique identifier for the translation (e.g., 'eng_kjv', 'heb_wlc')
- `book_id`: Canonical book number (1-66, where 1=Genesis, 66=Revelation)
- `chapter`: Chapter number within the book
- `verse`: Verse number within the chapter
- `text`: The actual verse text

**Example Data:**
```
id | translation_id | book_id | chapter | verse | text
---+---------------+---------+---------+-------+--------------------------------
1  | eng_kjv       | 43      | 3       | 16    | For God so loved the world...
2  | eng_asv       | 43      | 3       | 16    | For God so loved the world...
```

#### 2. words
Contains word-by-word analysis of original Hebrew and Greek texts.

```sql
CREATE TABLE words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_ref TEXT UNIQUE NOT NULL,
    book TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    word_num INTEGER NOT NULL,
    hebrew_text TEXT,
    greek_text TEXT,
    transliteration TEXT,
    translation TEXT,
    strongs_raw TEXT,
    strongs_primary TEXT,
    morphology_code TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek'))
);

CREATE INDEX idx_words_location ON words(book, chapter, verse);
CREATE INDEX idx_words_strongs ON words(strongs_primary);
CREATE INDEX idx_words_morph ON words(morphology_code);
```

**Fields:**
- `word_ref`: Unique reference format: "Book.Chapter.Verse.WordNum" (e.g., "Gen.1.1.1")
- `book`: Three-letter book code (e.g., 'Gen', 'Mat', 'Rev')
- `hebrew_text`/`greek_text`: Original text (only one populated per word)
- `transliteration`: Romanized representation
- `translation`: Full parsing code including Strong's + morphology (e.g., "G3056=N-NSM")
- `strongs_primary`: Clean Strong's number (e.g., "G3056")
- `morphology_code`: Grammatical parsing code
- `language`: Either 'hebrew' or 'greek'

**Example Data:**
```
word_ref    | greek_text | transliteration | translation  | strongs_primary
------------+------------+-----------------+--------------+----------------
Jhn.1.1.1   | Ἐν         | En              | G1722=P      | G1722
Jhn.1.1.2   | ἀρχῇ        | archē           | G0746=N-DSF  | G0746
Jhn.1.1.3   | ἦν         | ēn              | G1510=V-IAI  | G1510
```

#### 3. lexicon
Dictionary definitions for Hebrew and Greek words.

```sql
CREATE TABLE lexicon (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strongs_number TEXT UNIQUE NOT NULL,
    original_word TEXT,
    transliteration TEXT,
    part_of_speech TEXT,
    gloss TEXT,
    definition TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek'))
);

CREATE INDEX idx_lexicon_strongs ON lexicon(strongs_number);
CREATE INDEX idx_lexicon_language ON lexicon(language);
```

**Fields:**
- `strongs_number`: Strong's concordance number (e.g., "H1234", "G5678")
- `original_word`: Dictionary form in Hebrew/Greek
- `gloss`: Brief one-word definition
- `definition`: Full lexical definition
- `part_of_speech`: Grammatical category

**Example Data:**
```
strongs_number | original_word | gloss | definition
---------------+--------------+-------+----------------------------------------
G3056          | λόγος        | word  | A word, speech, divine utterance...
G0026          | ἀγάπη        | love  | Love, especially divine love...
```

#### 4. morphology
Explanations for morphological parsing codes.

```sql
CREATE TABLE morphology (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    description TEXT,
    components TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek'))
);

CREATE INDEX idx_morphology_code ON morphology(code);
CREATE INDEX idx_morphology_language ON morphology(language);
```

**Fields:**
- `code`: Morphology code (e.g., "N-NSM", "V-IAI-3S")
- `description`: Human-readable explanation
- `components`: Breakdown of code components

**Common Greek Morphology Codes:**
- `N-`: Noun
- `V-`: Verb
- `P-`: Preposition
- `T-`: Article
- `A-`: Adjective
- `D-`: Adverb
- `C-`: Conjunction

**Example Parsing:**
- `N-NSM`: Noun - Nominative Singular Masculine
- `V-IAI-3S`: Verb - Imperfect Active Indicative - 3rd person Singular
- `P-G`: Preposition - Genitive

#### 5. books
Metadata about biblical books.

```sql
CREATE TABLE books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    translation_id TEXT NOT NULL,
    book_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    common_name TEXT,
    book_order INTEGER,
    number_of_chapters INTEGER,
    testament TEXT CHECK(testament IN ('old', 'new')),
    UNIQUE(translation_id, book_id)
);

CREATE INDEX idx_books_translation ON books(translation_id);
CREATE INDEX idx_books_order ON books(book_order);
```

**Standard Book IDs (1-66):**
- Old Testament: 1-39 (Genesis to Malachi)
- New Testament: 40-66 (Matthew to Revelation)

#### 6. translations
Translation metadata.

```sql
CREATE TABLE translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    translation_id TEXT UNIQUE NOT NULL,
    name TEXT,
    english_name TEXT,
    language TEXT,
    year INTEGER,
    license TEXT
);
```

### Data Relationships

```
translations <--1:N--> verses
                         |
                         v
                       books

words <--N:1--> lexicon (via strongs_number)
  |
  +-----N:1--> morphology (via morphology_code)
```

## Import Process

### 1. Translation Import Flow

```python
# Simplified flow from bible_extractor.py
def import_translation(translation_id):
    # 1. Get translation metadata from bible.db
    translation_info = get_translation_info(translation_id)
    
    # 2. Insert translation record
    insert_translation(translation_info)
    
    # 3. Get all books for translation
    books = get_books_for_translation(translation_id)
    
    # 4. For each book:
    for book in books:
        # Insert book metadata
        insert_book(book)
        
        # Get all verses
        verses = get_verses_for_book(book)
        
        # Batch insert verses
        insert_verses_batch(verses)
```

### 2. STEPBible Import Flow

STEPBible data arrives in tab-separated format:

```
Gen.1.1#01=L<TAB>בְּ/רֵאשִׁ֖ית<TAB>be./re.Shit<TAB>in/beginning<TAB>H9003/{H7225}<TAB>HR/Ncfsa
```

Processing steps:
1. Parse reference (Gen.1.1#01)
2. Extract Hebrew/Greek text
3. Parse Strong's numbers and morphology
4. Clean and normalize data
5. Insert into words table

### 3. Import Tracking

The system uses `.import_status.json` to track progress:

```json
{
  "translations": {
    "eng_kjv": {
      "imported": true,
      "timestamp": "2024-01-15T10:30:00Z",
      "verse_count": 31102
    }
  },
  "stepbible_files": {
    "tahot": ["tahot_gen_deu.txt", "tahot_jos_est.txt"],
    "tagnt": ["tagnt_mat_jhn.txt", "tagnt_act_rev.txt"],
    "lexicon": ["tbesh.txt", "tbesg.txt"],
    "morphology": ["tehmc.txt", "tegmc.txt"]
  }
}
```

## Query Patterns

### Basic Verse Retrieval

```sql
-- Get a single verse
SELECT text FROM verses 
WHERE translation_id = 'eng_kjv' 
  AND book_id = 43 
  AND chapter = 3 
  AND verse = 16;

-- Get a chapter
SELECT verse, text FROM verses
WHERE translation_id = 'eng_kjv'
  AND book_id = 1
  AND chapter = 1
ORDER BY verse;

-- Compare translations
SELECT translation_id, text FROM verses
WHERE book_id = 43 AND chapter = 1 AND verse = 1
AND translation_id IN ('eng_kjv', 'eng_asv', 'eng_bbe');
```

### Original Language Analysis

```sql
-- Get Greek text with morphology for John 1:1
SELECT 
    w.word_num,
    w.greek_text,
    w.transliteration,
    w.strongs_primary,
    w.translation,
    l.gloss,
    l.definition,
    m.description as morphology
FROM words w
LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
LEFT JOIN morphology m ON w.morphology_code = m.code
WHERE w.book = 'Jhn' AND w.chapter = 1 AND w.verse = 1
ORDER BY w.word_num;
```

### Strong's Number Searches

```sql
-- Find all occurrences of a word (e.g., logos - G3056)
SELECT book, chapter, verse, greek_text, strongs_primary 
FROM words 
WHERE translation LIKE 'G3056%'
ORDER BY id
LIMIT 20;

-- Count occurrences by book
SELECT book, COUNT(*) as occurrences
FROM words
WHERE translation LIKE 'G3056%'
GROUP BY book
ORDER BY occurrences DESC;

-- Find related Strong's numbers (same root)
SELECT DISTINCT strongs_primary, greek_text, COUNT(*) as count
FROM words
WHERE strongs_primary LIKE 'G305%'  -- Same prefix
GROUP BY strongs_primary, greek_text
ORDER BY count DESC;
```

### Morphological Searches

```sql
-- Find all present tense verbs in John 1
SELECT verse, word_num, greek_text, transliteration
FROM words
WHERE book = 'Jhn' AND chapter = 1
  AND translation LIKE '%V-P%'  -- V = Verb, P = Present
ORDER BY verse, word_num;

-- Find all nominative nouns
SELECT book, chapter, verse, greek_text, strongs_primary
FROM words
WHERE translation LIKE '%N-N%'  -- N = Noun, N = Nominative
  AND book = 'Rom'
LIMIT 20;
```

### Cross-Translation Word Studies

```sql
-- See how different translations handle a specific verse
SELECT 
    t.translation_id,
    t.language,
    v.text
FROM verses v
JOIN translations t ON v.translation_id = t.translation_id
WHERE v.book_id = 43 AND v.chapter = 1 AND v.verse = 1
ORDER BY t.language, t.translation_id;
```

## Performance Optimization

### Indexes

The database includes strategic indexes for common query patterns:

1. **Verse lookups**: `idx_verses_reference` on (book_id, chapter, verse)
2. **Translation filtering**: `idx_verses_translation` on translation_id
3. **Strong's searches**: `idx_words_strongs` on strongs_primary
4. **Morphology queries**: `idx_words_morph` on morphology_code
5. **Word location**: `idx_words_location` on (book, chapter, verse)

### Query Optimization Tips

1. **Use indexes**: Structure WHERE clauses to utilize indexes
2. **Limit results**: Always use LIMIT for exploratory queries
3. **Join carefully**: Left joins to lexicon/morphology only when needed
4. **Batch operations**: Insert verses in batches of 1000
5. **Prepared statements**: Use parameterized queries to prevent SQL injection

### Database Maintenance

```sql
-- Analyze database for query optimization
ANALYZE;

-- Check database integrity
PRAGMA integrity_check;

-- Vacuum to reclaim space
VACUUM;

-- Get database statistics
SELECT 
    'verses' as table_name, COUNT(*) as row_count FROM verses
UNION ALL
SELECT 'words', COUNT(*) FROM words
UNION ALL
SELECT 'lexicon', COUNT(*) FROM lexicon
UNION ALL
SELECT 'morphology', COUNT(*) FROM morphology;
```

## Advanced Usage

### Full-Text Search Setup

```sql
-- Create virtual table for full-text search
CREATE VIRTUAL TABLE verses_fts USING fts5(
    translation_id, 
    book_id, 
    chapter, 
    verse, 
    text,
    content=verses
);

-- Populate FTS table
INSERT INTO verses_fts SELECT translation_id, book_id, chapter, verse, text FROM verses;

-- Search example
SELECT book_id, chapter, verse, highlight(verses_fts, 4, '<b>', '</b>') as text
FROM verses_fts
WHERE verses_fts MATCH 'love NEAR/5 world'
  AND translation_id = 'eng_kjv'
LIMIT 10;
```

### Custom Views

```sql
-- Create view for easy verse access with book names
CREATE VIEW verse_view AS
SELECT 
    v.translation_id,
    b.name as book_name,
    v.chapter,
    v.verse,
    v.text
FROM verses v
JOIN books b ON v.translation_id = b.translation_id AND v.book_id = b.book_id;

-- Usage
SELECT * FROM verse_view 
WHERE translation_id = 'eng_kjv' 
  AND book_name = 'John' 
  AND chapter = 1 
  AND verse = 1;
```

## Troubleshooting

### Common Issues

1. **Slow queries**: Run EXPLAIN QUERY PLAN to check index usage
2. **Lock errors**: Ensure single writer, multiple readers pattern
3. **Character encoding**: UTF-8 is enforced for all text fields
4. **Missing data**: Check import_status.json for incomplete imports

### Debugging Queries

```sql
-- Check query performance
EXPLAIN QUERY PLAN
SELECT * FROM words WHERE strongs_primary = 'G3056';

-- Get table statistics
SELECT name, COUNT(*) FROM sqlite_master WHERE type='index' GROUP BY name;

-- Check foreign key constraints
PRAGMA foreign_keys = ON;
PRAGMA foreign_key_check;
```