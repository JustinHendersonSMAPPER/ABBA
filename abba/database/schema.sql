-- ABBA Database Schema
-- SQLite schema for storing biblical texts, lexicons, and morphology data

-- Biblical text words (from TAHOT/TAGNT)
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    language TEXT CHECK(language IN ('hebrew', 'greek', 'aramaic')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book, chapter, verse, word_num)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_words_verse ON words(book, chapter, verse);
CREATE INDEX IF NOT EXISTS idx_words_strongs ON words(strongs_primary);
CREATE INDEX IF NOT EXISTS idx_words_morphology ON words(morphology_code);
CREATE INDEX IF NOT EXISTS idx_words_language ON words(language);

-- Lexicon entries (from TBESH/TBESG)
CREATE TABLE IF NOT EXISTS lexicon (
    strongs_number TEXT PRIMARY KEY,
    extended_strongs TEXT,
    disambiguated_strongs TEXT,
    unified_strongs TEXT,
    original_word TEXT,
    transliteration TEXT,
    part_of_speech TEXT,
    gloss TEXT,
    definition TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lexicon_language ON lexicon(language);
CREATE INDEX IF NOT EXISTS idx_lexicon_original_word ON lexicon(original_word);

-- Supplementary lexicon definitions (multiple sources per Strong's number)
CREATE TABLE IF NOT EXISTS lexicon_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strongs_number TEXT NOT NULL,
    source_lexicon TEXT NOT NULL,  -- 'bdb', 'dodson', 'thayers', etc.
    original_word TEXT,
    transliteration TEXT,
    part_of_speech TEXT,
    gloss TEXT,                    -- short meaning
    definition TEXT,               -- full scholarly definition
    language TEXT CHECK(language IN ('hebrew', 'greek')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strongs_number, source_lexicon)
);

CREATE INDEX IF NOT EXISTS idx_lexdef_strongs ON lexicon_definitions(strongs_number);
CREATE INDEX IF NOT EXISTS idx_lexdef_source ON lexicon_definitions(source_lexicon);
CREATE INDEX IF NOT EXISTS idx_lexdef_language ON lexicon_definitions(language);

-- Morphology codes (from TEHMC/TEGMC)
CREATE TABLE IF NOT EXISTS morphology (
    code TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    components TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_morphology_language ON morphology(language);

-- Bible translations from bible.db
CREATE TABLE IF NOT EXISTS translations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    english_name TEXT,
    language TEXT,
    canon TEXT CHECK(canon IN ('hebrew', 'protestant', 'catholic', 'orthodox', 'ethiopian')),
    is_partial_canon BOOLEAN DEFAULT 0,
    apocrypha_count INTEGER DEFAULT 0,
    has_import_failures BOOLEAN DEFAULT 0,
    failed_verse_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Books metadata
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    translation_id TEXT NOT NULL,
    book_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    common_name TEXT,
    book_order INTEGER,
    number_of_chapters INTEGER,
    testament TEXT CHECK(testament IN ('old', 'new')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (translation_id) REFERENCES translations(id),
    UNIQUE(translation_id, book_id)
);

CREATE INDEX IF NOT EXISTS idx_books_translation ON books(translation_id);
CREATE INDEX IF NOT EXISTS idx_books_order ON books(book_order);

-- Verses from translations
CREATE TABLE IF NOT EXISTS verses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    translation_id TEXT NOT NULL,
    book_id INTEGER NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (translation_id) REFERENCES translations(id),
    UNIQUE(translation_id, book_id, chapter, verse)
);

CREATE INDEX IF NOT EXISTS idx_verses_reference ON verses(translation_id, book_id, chapter, verse);

-- Full-text search virtual table for verses
CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(
    translation_id,
    book_id,
    chapter,
    verse,
    text,
    content=verses
);

-- Future: Semantic concepts
CREATE TABLE IF NOT EXISTS semantic_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL UNIQUE,
    category TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Future: Word-concept mappings
CREATE TABLE IF NOT EXISTS word_concepts (
    word_id INTEGER NOT NULL,
    concept_id INTEGER NOT NULL,
    confidence REAL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    source TEXT,  -- 'manual' or 'ollama'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (word_id, concept_id),
    FOREIGN KEY (word_id) REFERENCES words(id),
    FOREIGN KEY (concept_id) REFERENCES semantic_concepts(id)
);

-- STEPBible verses (from TAHOT/TAGNT files)
CREATE TABLE IF NOT EXISTS stepbible_verses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,     -- source file name
    book TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    word_number INTEGER NOT NULL,
    original_word TEXT,
    transliteration TEXT,
    english TEXT,
    strongs_raw TEXT,
    strongs_primary TEXT,
    morphology TEXT,
    language TEXT CHECK(language IN ('hebrew', 'greek', 'aramaic')) NOT NULL,
    data_hash INTEGER,  -- MurmurHash3 of the word data for integrity checking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_file, book, chapter, verse, word_number)
);

CREATE INDEX IF NOT EXISTS idx_stepbible_verses_reference ON stepbible_verses(book, chapter, verse);
CREATE INDEX IF NOT EXISTS idx_stepbible_verses_source ON stepbible_verses(source_file);
CREATE INDEX IF NOT EXISTS idx_stepbible_verses_strongs ON stepbible_verses(strongs_primary);
CREATE INDEX IF NOT EXISTS idx_stepbible_verses_language ON stepbible_verses(language);

-- Track failed verse imports
CREATE TABLE IF NOT EXISTS failed_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    translation_id TEXT NOT NULL,
    book_id INTEGER NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (translation_id) REFERENCES translations(id),
    UNIQUE(translation_id, book_id, chapter, verse)
);

CREATE INDEX IF NOT EXISTS idx_failed_imports_translation ON failed_imports(translation_id);

-- STEPBible validation results
CREATE TABLE IF NOT EXISTS stepbible_validation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_checks INTEGER,
    passed_checks INTEGER,
    failed_checks INTEGER,
    success_rate REAL,
    details TEXT  -- JSON with detailed results
);

-- Database metadata
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial metadata
INSERT OR REPLACE INTO db_metadata (key, value) VALUES 
    ('schema_version', '1.0'),
    ('created_at', datetime('now')),
    ('description', 'ABBA Biblical Analysis Database');