"""Database migration utilities for ABBA."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def add_canon_column(db_path: Path) -> bool:
    """Add canon column to translations table if it doesn't exist.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if column exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM pragma_table_info('translations')
                WHERE name='canon'
            """
            )

            if cursor.fetchone()[0] > 0:
                logger.debug("Canon column already exists")
                return False

            # Add column
            cursor.execute(
                """
                ALTER TABLE translations
                ADD COLUMN canon TEXT CHECK(canon IN ('hebrew', 'protestant', 'catholic', 'orthodox', 'ethiopian'))
            """
            )

            # Update existing rows with detected canons
            from ..parallel_import import get_translation_canon  # pylint: disable=import-outside-toplevel

            cursor.execute("SELECT id FROM translations")
            translation_ids = [row[0] for row in cursor.fetchall()]

            source_db_path = db_path.parent / "bible.db"
            if source_db_path.exists():
                for trans_id in translation_ids:
                    canon_enum = get_translation_canon(trans_id, str(source_db_path))
                    cursor.execute("UPDATE translations SET canon = ? WHERE id = ?", (canon_enum.value, trans_id))

            conn.commit()
            logger.info("Added canon column and updated %d translations", len(translation_ids))
            return True

    except Exception as e:
        logger.error("Failed to add canon column: %s", e)
        raise


def add_stepbible_verses_table(db_path: Path) -> bool:
    """Add stepbible_verses table if it doesn't exist.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='table' AND name='stepbible_verses'
            """
            )

            if cursor.fetchone()[0] > 0:
                logger.debug("stepbible_verses table already exists")
                return False

            # Create table
            cursor.execute(
                """
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_file, book, chapter, verse, word_number)
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_stepbible_verses_reference ON stepbible_verses(book, chapter, verse)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stepbible_verses_source ON stepbible_verses(source_file)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_stepbible_verses_strongs ON stepbible_verses(strongs_primary)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stepbible_verses_language ON stepbible_verses(language)")

            conn.commit()
            logger.info("Added stepbible_verses table")
            return True

    except Exception as e:
        logger.error("Failed to add stepbible_verses table: %s", e)
        raise


def add_partial_canon_columns(db_path: Path) -> bool:
    """Add partial canon tracking columns to translations table.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if columns exist
            cursor.execute(
                """
                SELECT COUNT(*) FROM pragma_table_info('translations')
                WHERE name IN ('is_partial_canon', 'apocrypha_count')
            """
            )

            if cursor.fetchone()[0] >= 2:
                logger.debug("Partial canon columns already exist")
                return False

            # Add columns
            try:
                cursor.execute("ALTER TABLE translations ADD COLUMN is_partial_canon BOOLEAN DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column might already exist

            try:
                cursor.execute("ALTER TABLE translations ADD COLUMN apocrypha_count INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column might already exist

            conn.commit()
            logger.info("Added partial canon tracking columns")
            return True

    except Exception as e:
        logger.error("Failed to add partial canon columns: %s", e)
        raise


def add_import_failure_tracking(db_path: Path) -> bool:
    """Add import failure tracking columns and table.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if columns exist
            cursor.execute(
                """
                SELECT COUNT(*) FROM pragma_table_info('translations')
                WHERE name IN ('has_import_failures', 'failed_verse_count')
            """
            )

            columns_exist = cursor.fetchone()[0] >= 2

            # Check if table exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='table' AND name='failed_imports'
            """
            )

            table_exists = cursor.fetchone()[0] > 0

            if columns_exist and table_exists:
                logger.debug("Import failure tracking already exists")
                return False

            # Add columns if needed
            if not columns_exist:
                try:
                    cursor.execute("ALTER TABLE translations ADD COLUMN has_import_failures BOOLEAN DEFAULT 0")
                except sqlite3.OperationalError:
                    pass

                try:
                    cursor.execute("ALTER TABLE translations ADD COLUMN failed_verse_count INTEGER DEFAULT 0")
                except sqlite3.OperationalError:
                    pass

            # Create table if needed
            if not table_exists:
                cursor.execute(
                    """
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
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_failed_imports_translation
                    ON failed_imports(translation_id)
                """
                )

            conn.commit()
            logger.info("Added import failure tracking")
            return True

    except Exception as e:
        logger.error("Failed to add import failure tracking: %s", e)
        raise


def add_stepbible_hash_column(db_path: Path) -> bool:
    """Add data_hash column to stepbible_verses table if it doesn't exist.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if column exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM pragma_table_info('stepbible_verses')
                WHERE name='data_hash'
            """
            )

            if cursor.fetchone()[0] > 0:
                logger.debug("data_hash column already exists in stepbible_verses")
                return False

            # Add column
            cursor.execute(
                """
                ALTER TABLE stepbible_verses
                ADD COLUMN data_hash INTEGER
            """
            )

            conn.commit()
            logger.info("Added data_hash column to stepbible_verses table")
            return True

    except Exception as e:
        logger.error("Failed to add data_hash column: %s", e)
        raise


def add_book_metadata_table(db_path: Path) -> bool:
    """Add book_metadata table for genre, authorship, and literary features.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='book_metadata'")
            if cursor.fetchone()[0] > 0:
                logger.debug("book_metadata table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS book_metadata (
                    book_id INTEGER NOT NULL PRIMARY KEY,
                    primary_genre TEXT NOT NULL,
                    secondary_genres TEXT,
                    author_traditional TEXT,
                    date_range_start INTEGER,
                    date_range_end INTEGER,
                    original_audience TEXT,
                    original_language TEXT,
                    literary_features TEXT,
                    reading_context TEXT,
                    canonical_section TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()
            logger.info("Added book_metadata table")
            return True
    except Exception as e:
        logger.error("Failed to add book_metadata table: %s", e)
        raise


def add_passages_table(db_path: Path) -> bool:
    """Add passages table for pericope/passage boundaries.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='passages'")
            if cursor.fetchone()[0] > 0:
                logger.debug("passages table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS passages (
                    passage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    start_chapter INTEGER NOT NULL,
                    start_verse INTEGER NOT NULL,
                    end_chapter INTEGER NOT NULL,
                    end_verse INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    genre TEXT,
                    literary_type TEXT,
                    structural_features TEXT,
                    parent_passage_id INTEGER,
                    display_order INTEGER,
                    FOREIGN KEY (parent_passage_id) REFERENCES passages(passage_id)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_passages_book ON passages(book_id, start_chapter, start_verse)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_passages_genre ON passages(genre)")
            conn.commit()
            logger.info("Added passages table")
            return True
    except Exception as e:
        logger.error("Failed to add passages table: %s", e)
        raise


def add_literary_structures_table(db_path: Path) -> bool:
    """Add literary_structures table for chiasmus, parallelism, etc.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='literary_structures'")
            if cursor.fetchone()[0] > 0:
                logger.debug("literary_structures table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS literary_structures (
                    structure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    start_chapter INTEGER NOT NULL,
                    start_verse INTEGER NOT NULL,
                    end_chapter INTEGER NOT NULL,
                    end_verse INTEGER NOT NULL,
                    structure_type TEXT NOT NULL,
                    description TEXT,
                    significance TEXT,
                    elements TEXT,
                    scholarly_source TEXT
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_lit_struct_book "
                "ON literary_structures(book_id, start_chapter, start_verse)"
            )
            conn.commit()
            logger.info("Added literary_structures table")
            return True
    except Exception as e:
        logger.error("Failed to add literary_structures table: %s", e)
        raise


def add_cultural_context_table(db_path: Path) -> bool:
    """Add cultural_context table for historical/cultural annotations.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cultural_context'")
            if cursor.fetchone()[0] > 0:
                logger.debug("cultural_context table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cultural_context (
                    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    start_chapter INTEGER,
                    start_verse INTEGER,
                    end_chapter INTEGER,
                    end_verse INTEGER,
                    context_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    detailed_content TEXT,
                    time_period TEXT,
                    geographic_region TEXT,
                    confidence TEXT,
                    sources TEXT,
                    display_priority INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_cultural_context_scope "
                "ON cultural_context(book_id, start_chapter, start_verse)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cultural_context_type ON cultural_context(context_type)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_cultural_context_priority ON cultural_context(display_priority)"
            )
            conn.commit()
            logger.info("Added cultural_context table")
            return True
    except Exception as e:
        logger.error("Failed to add cultural_context table: %s", e)
        raise


def add_cross_references_table(db_path: Path) -> bool:
    """Add cross_references table for inter-passage references.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cross_references'")
            if cursor.fetchone()[0] > 0:
                logger.debug("cross_references table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cross_references (
                    ref_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_book_id INTEGER NOT NULL,
                    source_chapter INTEGER NOT NULL,
                    source_verse INTEGER NOT NULL,
                    target_book_id INTEGER NOT NULL,
                    target_chapter INTEGER NOT NULL,
                    target_verse INTEGER NOT NULL,
                    ref_type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    source_dataset TEXT,
                    notes TEXT,
                    UNIQUE(source_book_id, source_chapter, source_verse,
                           target_book_id, target_chapter, target_verse, ref_type)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_xref_source "
                "ON cross_references(source_book_id, source_chapter, source_verse)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_xref_target "
                "ON cross_references(target_book_id, target_chapter, target_verse)"
            )
            conn.commit()
            logger.info("Added cross_references table")
            return True
    except Exception as e:
        logger.error("Failed to add cross_references table: %s", e)
        raise


def add_word_richness_table(db_path: Path) -> bool:
    """Add word_richness table for precomputed meaning-loss scores.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='word_richness'")
            if cursor.fetchone()[0] > 0:
                logger.debug("word_richness table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS word_richness (
                    richness_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book TEXT NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    word_num INTEGER NOT NULL,
                    strongs_number TEXT NOT NULL,
                    gloss_coverage REAL,
                    morphology_significance TEXT,
                    untranslatable_nuances TEXT,
                    cultural_significance TEXT,
                    richness_score REAL NOT NULL,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_richness_verse ON word_richness(book, chapter, verse)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_richness_score ON word_richness(richness_score DESC)")
            conn.commit()
            logger.info("Added word_richness table")
            return True
    except Exception as e:
        logger.error("Failed to add word_richness table: %s", e)
        raise


def add_life_topics_tables(db_path: Path) -> bool:
    """Add life_topics, life_topic_concepts, and topic_study_steps tables.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='life_topics'")
            if cursor.fetchone()[0] > 0:
                logger.debug("life_topics tables already exist")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS life_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    icon TEXT,
                    display_order INTEGER DEFAULT 0
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS life_topic_concepts (
                    topic_id INTEGER NOT NULL,
                    concept_name TEXT NOT NULL,
                    relevance_aspect TEXT,
                    display_order INTEGER DEFAULT 0,
                    FOREIGN KEY (topic_id) REFERENCES life_topics(id),
                    PRIMARY KEY (topic_id, concept_name)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_study_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    step_order INTEGER NOT NULL,
                    step_type TEXT NOT NULL,
                    verse_reference TEXT NOT NULL,
                    insight TEXT,
                    translation_lens_focus TEXT,
                    FOREIGN KEY (topic_id) REFERENCES life_topics(id)
                )
            """
            )

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_life_topics_category ON life_topics(category)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_topic_steps_topic ON topic_study_steps(topic_id, step_order)"
            )
            conn.commit()
            logger.info("Added life_topics, life_topic_concepts, and topic_study_steps tables")
            return True
    except Exception as e:
        logger.error("Failed to add life_topics tables: %s", e)
        raise


def add_genre_shifts_table(db_path: Path) -> bool:
    """Add genre_shifts table for tracking genre transitions within books."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='genre_shifts'")
            if cursor.fetchone()[0] > 0:
                return False
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS genre_shifts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    from_genre TEXT NOT NULL,
                    to_genre TEXT NOT NULL,
                    description TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_genre_shifts_book ON genre_shifts(book_id, chapter, verse)")
            conn.commit()
            logger.info("Added genre_shifts table")
            return True
    except Exception as e:
        logger.error("Failed to add genre_shifts table: %s", e)
        raise


def add_speaker_attributions_table(db_path: Path) -> bool:
    """Add speaker_attributions table for quoted speech."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='speaker_attributions'")
            if cursor.fetchone()[0] > 0:
                return False
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS speaker_attributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    start_chapter INTEGER NOT NULL,
                    start_verse INTEGER NOT NULL,
                    end_chapter INTEGER NOT NULL,
                    end_verse INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    context_note TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_speaker_attr_book "
                "ON speaker_attributions(book_id, start_chapter, start_verse)"
            )
            conn.commit()
            logger.info("Added speaker_attributions table")
            return True
    except Exception as e:
        logger.error("Failed to add speaker_attributions table: %s", e)
        raise


def add_word_explanations_table(db_path: Path) -> bool:
    """Add word_explanations table for plain-English word explanations."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='word_explanations'")
            if cursor.fetchone()[0] > 0:
                return False
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS word_explanations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strongs_number TEXT UNIQUE NOT NULL,
                    language TEXT NOT NULL,
                    explanation TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_word_explanations_strongs ON word_explanations(strongs_number)"
            )
            conn.commit()
            logger.info("Added word_explanations table")
            return True
    except Exception as e:
        logger.error("Failed to add word_explanations table: %s", e)
        raise


def add_concept_quality_tables(db_path: Path) -> bool:
    """Add tables for concept quality review metadata."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='semantic_range_warnings'")
            if cursor.fetchone()[0] > 0:
                return False
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_range_warnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strongs_number TEXT UNIQUE NOT NULL,
                    warning_text TEXT NOT NULL,
                    frequency_note TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_review_flags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT UNIQUE NOT NULL,
                    flag_type TEXT NOT NULL,
                    review_note TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_temporal_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT UNIQUE NOT NULL,
                    temporal_period TEXT NOT NULL,
                    period_note TEXT
                )
                """
            )
            conn.commit()
            logger.info("Added concept quality tables")
            return True
    except Exception as e:
        logger.error("Failed to add concept quality tables: %s", e)
        raise


def add_verse_annotations_cache_table(db_path: Path) -> bool:
    """Add verse_annotations_cache table for precomputed annotation data.

    Materializes all STANDARD/DEEP-level annotation queries into a single
    row per verse, reducing 8-12 queries per deep request to a single lookup.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='verse_annotations_cache'")
            if cursor.fetchone()[0] > 0:
                logger.debug("verse_annotations_cache table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS verse_annotations_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    words_json TEXT,
                    richness_flags_json TEXT,
                    cross_references_json TEXT,
                    cultural_context_json TEXT,
                    passage_info_json TEXT,
                    literary_structures_json TEXT,
                    speaker_json TEXT,
                    active_genre TEXT,
                    cache_version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(book_id, chapter, verse)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_annotation_cache_verse "
                "ON verse_annotations_cache(book_id, chapter, verse)"
            )
            conn.commit()
            logger.info("Added verse_annotations_cache table")
            return True
    except Exception as e:
        logger.error("Failed to add verse_annotations_cache table: %s", e)
        raise


def add_range_query_indexes(db_path: Path) -> bool:
    """Add composite indexes optimizing range-based verse lookups.

    Covers the expensive WHERE (start_chapter < ? OR ...) patterns
    used by passages, literary_structures, speaker_attributions,
    and genre_shifts queries.

    Args:
        db_path: Path to the database

    Returns:
        True if any indexes were added, False if all exist
    """
    indexes = [
        (
            "idx_passages_range",
            "passages",
            "book_id, start_chapter, start_verse, end_chapter, end_verse",
        ),
        (
            "idx_literary_structures_range",
            "literary_structures",
            "book_id, start_chapter, start_verse, end_chapter, end_verse",
        ),
        (
            "idx_speaker_attr_range",
            "speaker_attributions",
            "book_id, start_chapter, start_verse, end_chapter, end_verse",
        ),
        (
            "idx_genre_shifts_lookup",
            "genre_shifts",
            "book_id, chapter, verse",
        ),
        (
            "idx_cross_refs_source",
            "cross_references",
            "source_book_id, source_chapter, source_verse",
        ),
        (
            "idx_cross_refs_target",
            "cross_references",
            "target_book_id, target_chapter, target_verse",
        ),
        (
            "idx_word_richness_verse",
            "word_richness",
            "book, chapter, verse",
        ),
        (
            "idx_cultural_context_book",
            "cultural_context",
            "book_id, start_chapter",
        ),
    ]
    added = False
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for idx_name, table, columns in indexes:
                # Check if table exists before adding index
                cursor.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?",
                    (idx_name,),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})")
                    added = True
            conn.commit()
            if added:
                logger.info("Added range query optimization indexes")
    except Exception as e:
        logger.error("Failed to add range query indexes: %s", e)
        raise
    return added


def add_lexicon_definitions_table(db_path: Path) -> bool:
    """Add lexicon_definitions table for multi-source lexicon data.

    Stores definitions from supplementary lexicons (BDB, Dodson, etc.)
    keyed to Strong's numbers, enabling multi-source comparison.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='lexicon_definitions'")
            if cursor.fetchone()[0] > 0:
                logger.debug("lexicon_definitions table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS lexicon_definitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strongs_number TEXT NOT NULL,
                    source_lexicon TEXT NOT NULL,
                    original_word TEXT,
                    transliteration TEXT,
                    part_of_speech TEXT,
                    gloss TEXT,
                    definition TEXT,
                    language TEXT CHECK(language IN ('hebrew', 'greek')) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strongs_number, source_lexicon)
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lexdef_strongs ON lexicon_definitions(strongs_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lexdef_source ON lexicon_definitions(source_lexicon)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lexdef_language ON lexicon_definitions(language)")
            conn.commit()
            logger.info("Added lexicon_definitions table")
            return True
    except Exception as e:
        logger.error("Failed to add lexicon_definitions table: %s", e)
        raise


def add_user_annotation_tables(db_path: Path) -> bool:
    """Add tables for user notes, collections, and sharing."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='verse_notes'")
            if cursor.fetchone()[0] > 0:
                return False
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS verse_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    note_type TEXT DEFAULT 'personal',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_verse ON verse_notes(book_id, chapter, verse)")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_collections (
                    collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    book_id INTEGER NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    note TEXT DEFAULT '',
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (collection_id) REFERENCES user_collections(collection_id),
                    UNIQUE(collection_id, book_id, chapter, verse)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_collection_items ON collection_items(collection_id)")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    share_token TEXT UNIQUE NOT NULL,
                    share_type TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    content_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_shared_token ON shared_items(share_token)")
            conn.commit()
            logger.info("Added user annotation tables")
            return True
    except Exception as e:
        logger.error("Failed to add user annotation tables: %s", e)
        raise


_MIGRATIONS = [
    (add_canon_column, "canon column"),
    (add_stepbible_verses_table, "stepbible_verses table"),
    (add_partial_canon_columns, "partial canon columns"),
    (add_import_failure_tracking, "import failure tracking"),
    (add_stepbible_hash_column, "stepbible hash column"),
    # Phase 5 enrichment migrations
    (add_book_metadata_table, "book_metadata table"),
    (add_passages_table, "passages table"),
    (add_literary_structures_table, "literary_structures table"),
    (add_cultural_context_table, "cultural_context table"),
    (add_cross_references_table, "cross_references table"),
    (add_word_richness_table, "word_richness table"),
    (add_life_topics_tables, "life_topics tables"),
    # Phase 5-6 quality + contextual intelligence
    (add_genre_shifts_table, "genre_shifts table"),
    (add_speaker_attributions_table, "speaker_attributions table"),
    (add_word_explanations_table, "word_explanations table"),
    (add_concept_quality_tables, "concept quality tables"),
    # Supplementary lexicon support
    (add_lexicon_definitions_table, "lexicon_definitions table"),
    # Phase 7 performance optimization
    (add_verse_annotations_cache_table, "verse_annotations_cache table"),
    (add_range_query_indexes, "range query optimization indexes"),
    # Phase 8 user features
    (add_user_annotation_tables, "user annotation tables"),
]


def run_migrations(db_path: Path) -> None:
    """Run all necessary database migrations.

    Args:
        db_path: Path to the database
    """
    if not db_path.exists():
        logger.debug("No database to migrate")
        return

    logger.info("Checking for database migrations...")

    migrations_run = []
    for migrate_fn, label in _MIGRATIONS:
        if migrate_fn(db_path):
            migrations_run.append(label)

    if migrations_run:
        logger.info("Database migrations completed: %s", ", ".join(migrations_run))
    else:
        logger.debug("No migrations needed")
