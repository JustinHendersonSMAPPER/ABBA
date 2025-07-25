# ABBA SQLite Database Examples

The ABBA project uses two databases:
- **`bible.db`** - Downloaded source containing Bible translations only
- **`abba.db`** - ABBA's processed database with original languages, morphology, lexicon, and all translations

These examples show how to use `abba.db` which contains the enriched biblical data.

## 1. Basic Verse Retrieval

Retrieve any verse from multiple translations:

```sql
SELECT translation, text 
FROM verses 
WHERE book = 'JHN' AND chapter = 3 AND verse = 16
AND translation IN ('KJV', 'ESV', 'NIV')
```

**Example Output:**
```
KJV: For God so loved the world, that he gave his only begotten Son...
ESV: For God so loved the world, that he gave his only Son...
NIV: For God so loved the world that he gave his one and only Son...
```

## 2. Original Language Access with Meanings

Get Greek/Hebrew text with Strong's numbers and definitions:

```sql
SELECT 
    w.greek,
    w.transliteration,
    w.strongs_number,
    w.english_gloss,
    l.brief_gloss,
    l.full_definition
FROM words w
LEFT JOIN lexicon l ON w.strongs_number = l.strongs_number
WHERE w.book = 'JHN' AND w.chapter = 1 AND w.verse = 1
ORDER BY w.word_position
```

**Example Output (John 1:1 first words):**
```
1. Ἐν (En)
   Strong's: G1722
   Basic meaning: in/on/at
   Lexicon: in, on, at, by, with
   Full definition: A primary preposition denoting position...

2. ἀρχῇ (archē)
   Strong's: G746
   Basic meaning: beginning
   Lexicon: beginning, origin, first
   Full definition: From G756; a commencement, or chief...
```

## 3. Morphological Analysis

Search for specific grammatical patterns:

```sql
-- Find all imperative verbs (commands) in Matthew 5
SELECT greek, transliteration, english_gloss, chapter, verse
FROM words
WHERE book = 'MAT' AND chapter = 5
AND morphology LIKE '%V_RAI%'  -- Verb, Imperative Active
```

**Common Morphology Codes:**
- `V` = Verb
- `N` = Noun
- `A` = Adjective
- `P` = Pronoun
- `RAI` = Aorist Active Imperative
- `VqP` = Verb, Qal, Perfect (Hebrew)

## 4. Word Usage Analysis

Track how often and where specific words appear:

```sql
-- Find all uses of "agape" (love) - Strong's G26
SELECT 
    COUNT(*) as total_uses,
    COUNT(DISTINCT book || ':' || chapter) as chapters_used
FROM words
WHERE strongs_number = 'G26'
```

## 5. Hebrew Word Families

Find related words by Hebrew root:

```sql
-- Find words from the קדש (holy) root
SELECT DISTINCT
    w.strongs_number,
    w.hebrew,
    w.transliteration,
    l.brief_gloss,
    COUNT(*) as usage_count
FROM words w
JOIN lexicon l ON w.strongs_number = l.strongs_number
WHERE w.strongs_number BETWEEN 'H6942' AND 'H6948'
GROUP BY w.strongs_number
```

## 6. Translation Comparison

Compare how different translations render the same original text:

```sql
-- Get Hebrew structure
SELECT hebrew, transliteration, english_gloss
FROM words
WHERE book = 'GEN' AND chapter = 1 AND verse = 2
ORDER BY word_position

-- Then compare translations
SELECT translation, text
FROM verses
WHERE book = 'GEN' AND chapter = 1 AND verse = 2
AND translation IN ('KJV', 'ESV', 'NLT')
```

## Database Schema Overview

### Main Tables:

1. **verses** - Complete verses in all translations
   - `book`, `chapter`, `verse`, `translation`, `text`

2. **words** - Original language words with analysis
   - `word_id`, `book`, `chapter`, `verse`, `word_position`
   - `hebrew`/`greek`, `transliteration`, `english_gloss`
   - `strongs_number`, `morphology`

3. **lexicon** - Strong's dictionary definitions
   - `strongs_number`, `brief_gloss`, `full_definition`
   - `language` (Hebrew/Greek)

4. **morphology_codes** - Grammatical code explanations
   - `code`, `description`, `language`

5. **books** - Book metadata
   - `book_code`, `book_name`, `testament`

## Using the Python API

The project provides high-level APIs for easier access:

```python
from abba.config import ABBAConfig
from abba.database.sqlite_manager import SQLiteManager
from abba.api.search import SearchAPI
from abba.api.analysis import AnalysisAPI

# Initialize
config = ABBAConfig()
config.database_path = "bible_data/abba.db"
db = SQLiteManager(config)
search_api = SearchAPI(db)
analysis_api = AnalysisAPI(db)

# Get a verse
verse = search_api.get_verse("JHN", 3, 16, translation="KJV")

# Search by Strong's number
results = search_api.search_strongs("G26")  # agape

# Analyze morphology patterns
patterns = analysis_api.analyze_morphology_patterns(
    language="greek", 
    pattern="V_RAI"  # Imperative verbs
)

# Find hapax legomena (words used only once)
rare_words = analysis_api.find_hapax_legomena(language="hebrew")
```

## Performance Tips

1. **Use indexes** - The database has indexes on common query patterns
2. **Limit results** - Always use LIMIT for exploration queries
3. **Cache frequent queries** - Use the built-in caching layer
4. **Batch operations** - Process multiple verses in one query when possible

## Database File Locations

After running `python abba/main.py`:
- `bible_data/bible.db` - Downloaded source database (translations only)
- `bible_data/abba.db` - Processed database with all linguistic data

## Example Scripts

Two demonstration scripts are provided:

1. `claude/scripts/simple_db_examples.py` - Basic examples with direct SQL
2. `claude/scripts/demo_database_capabilities.py` - Comprehensive API demonstrations

Run them with:
```bash
# First, create the database
python abba/main.py

# Then run examples
python claude/scripts/simple_db_examples.py
```