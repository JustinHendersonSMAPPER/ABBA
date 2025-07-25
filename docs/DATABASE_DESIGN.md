# STEPBible Database Design

## Recommended Architecture: SQLite with Specialized Indexes

### Why SQLite?

1. **Perfect fit for query patterns:**
   - Fast verse lookups: `SELECT * FROM words WHERE book=? AND chapter=? AND verse=?`
   - Strong's number searches: `SELECT * FROM words WHERE strongs_number=?`
   - Morphology queries: `SELECT * FROM words WHERE morphology LIKE ?`
   - Cross-referencing: JOIN operations between words, lexicon, and morphology tables

2. **Performance characteristics:**
   - With proper indexes, verse lookups will be sub-millisecond
   - B-tree indexes perfect for our access patterns
   - Can handle the 100MB dataset easily
   - Memory-mapped I/O for efficiency

3. **Development benefits:**
   - SQL is well-understood
   - Excellent Python support via sqlite3
   - Easy to backup, version, and distribute
   - Built-in full-text search capabilities

### Proposed Schema

```sql
-- Biblical text words (from TAHOT/TAGNT)
CREATE TABLE words (
    id INTEGER PRIMARY KEY,
    book TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    word_num INTEGER NOT NULL,
    word_ref TEXT NOT NULL,  -- e.g., "Gen.1.1#01=L"
    hebrew_text TEXT,
    greek_text TEXT,
    transliteration TEXT,
    translation TEXT,
    strongs_raw TEXT,        -- e.g., "H9003/{H7225G}"
    morphology_code TEXT,    -- e.g., "HR/Ncfsa"
    strongs_primary TEXT,    -- extracted primary Strong's
    language TEXT CHECK(language IN ('hebrew', 'greek', 'aramaic')),
    UNIQUE(book, chapter, verse, word_num)
);

-- Indexes for common queries
CREATE INDEX idx_verse ON words(book, chapter, verse);
CREATE INDEX idx_strongs ON words(strongs_primary);
CREATE INDEX idx_morphology ON words(morphology_code);

-- Lexicon entries (from TBESH/TBESG)
CREATE TABLE lexicon (
    strongs_number TEXT PRIMARY KEY,
    extended_strongs TEXT,
    disambiguated_strongs TEXT,
    unified_strongs TEXT,
    original_word TEXT,
    transliteration TEXT,
    part_of_speech TEXT,
    gloss TEXT,
    definition TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek'))
);

-- Morphology codes (from TEHMC/TEGMC)
CREATE TABLE morphology (
    code TEXT PRIMARY KEY,
    description TEXT,
    components TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek'))
);

-- Future: Semantic taxonomy
CREATE TABLE semantic_concepts (
    id INTEGER PRIMARY KEY,
    concept TEXT NOT NULL,
    category TEXT,
    description TEXT
);

CREATE TABLE word_concepts (
    word_id INTEGER REFERENCES words(id),
    concept_id INTEGER REFERENCES semantic_concepts(id),
    confidence REAL,
    source TEXT,  -- 'manual' or 'ollama'
    PRIMARY KEY (word_id, concept_id)
);

-- Full-text search virtual table
CREATE VIRTUAL TABLE words_fts USING fts5(
    word_ref, 
    translation, 
    content=words
);
```

### Additional Components

1. **Cache Layer** (Optional)
   - Use Redis or in-memory dict for frequently accessed verses
   - Cache decoded morphology descriptions
   - Cache lexicon lookups

2. **Semantic Analysis Storage**
   - Store Ollama embeddings in a separate numpy array file
   - Use FAISS or Annoy for similarity searches
   - Link back to SQLite via word IDs

3. **API Design**
   ```python
   class BibleDatabase:
       def get_verse(self, book: str, chapter: int, verse: int) -> List[Word]
       def lookup_strongs(self, number: str) -> LexiconEntry
       def decode_morphology(self, code: str) -> MorphologyInfo
       def search_concept(self, concept: str) -> List[VerseMatch]
       def get_word_analysis(self, book: str, chapter: int, verse: int, word_num: int) -> CompleteAnalysis
   ```

### Migration Path

1. Parse STEPBible text files once
2. Populate SQLite database
3. Build indexes
4. Verify data integrity
5. Implement caching layer if needed
6. Add Ollama semantic analysis iteratively

### Performance Expectations

- Database size: ~150-200MB with indexes
- Verse lookup: <1ms
- Strong's lookup: <1ms
- Concept search: <100ms (with proper indexes)
- Full-text search: <50ms for common terms

This approach provides the best balance of:
- Query flexibility
- Performance
- Development speed
- Maintainability
- Future extensibility