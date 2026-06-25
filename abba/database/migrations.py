"""Database migration utilities for ABBA."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

logger = logging.getLogger(__name__)


@contextmanager
def _connect(db_path: Union[Path, str], *args: Any, **kwargs: Any) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection that commits, rolls back, and always closes.

    Python's built-in ``sqlite3`` connection context manager commits (or rolls
    back on error) the active transaction but does **not** close the connection,
    so it lingers until garbage collection. On Windows a lingering connection
    holds an OS file lock, causing ``PermissionError`` on temp-DB teardown. This
    helper preserves the commit/rollback semantics and additionally closes the
    connection on exit.

    Args:
        db_path: Path to the database.
        *args: Extra positional arguments forwarded to ``sqlite3.connect``.
        **kwargs: Extra keyword arguments forwarded to ``sqlite3.connect``.

    Yields:
        An open ``sqlite3.Connection``.
    """
    conn = sqlite3.connect(db_path, *args, **kwargs)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def add_canon_column(db_path: Path) -> bool:
    """Add canon column to translations table if it doesn't exist.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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


def add_cross_reference_candidates_table(db_path: Path) -> bool:
    """Add cross_reference_candidates staging table for TSK and other sources.

    Stores raw cross-reference data from public-domain datasets (TSK, OpenBible,
    etc.) before editorial review.  A unique constraint on the 6-tuple
    (source_book_id, source_chapter, source_verse, target_book_id, target_chapter,
    target_verse) deduplicates repeated imports; INSERT OR IGNORE makes every
    import idempotent.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cross_reference_candidates'"
            )
            if cursor.fetchone()[0] > 0:
                logger.debug("cross_reference_candidates table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cross_reference_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_book_id INTEGER NOT NULL,
                    source_chapter INTEGER NOT NULL,
                    source_verse INTEGER NOT NULL,
                    target_book_id INTEGER NOT NULL,
                    target_chapter INTEGER NOT NULL,
                    target_verse INTEGER NOT NULL,
                    anchor_phrase TEXT,
                    source_dataset TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_book_id, source_chapter, source_verse,
                           target_book_id, target_chapter, target_verse)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_xref_candidates_source "
                "ON cross_reference_candidates(source_book_id, source_chapter, source_verse)"
            )
            conn.commit()
            logger.info("Added cross_reference_candidates table")
            return True
    except Exception as e:
        logger.error("Failed to add cross_reference_candidates table: %s", e)
        raise


def add_dictionary_entries_table(db_path: Path) -> bool:
    """Add the dictionary_entries table for public-domain reference works.

    Stores entry-keyed articles (headword -> article) from public-domain Bible
    dictionaries such as Easton's 1897 (decision D5), the raw material for later
    entity-linking of a verse's proper nouns / key terms to dictionary headwords.

    Idempotent: ``INSERT OR IGNORE`` against the ``UNIQUE(source, headword)``
    constraint makes re-imports add zero rows.

    NOTE: This migration is intentionally **not** registered in ``_MIGRATIONS``;
    it is applied explicitly by ``easton_importer.import_easton_entries`` (and by
    tests against temp databases) so the live database is only touched when a
    dictionary is actually ingested.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='dictionary_entries'")
            if cursor.fetchone()[0] > 0:
                logger.debug("dictionary_entries table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dictionary_entries (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    headword TEXT NOT NULL,
                    headword_normalized TEXT NOT NULL,
                    article TEXT NOT NULL,
                    ref_targets TEXT,
                    source TEXT NOT NULL,
                    license TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, headword)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_dict_entries_headword ON dictionary_entries(headword_normalized)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dict_entries_source ON dictionary_entries(source)")
            conn.commit()
            logger.info("Added dictionary_entries table")
            return True
    except Exception as e:
        logger.error("Failed to add dictionary_entries table: %s", e)
        raise


def add_stepbible_lexical_strongs_column(db_path: Path) -> bool:
    """Add lexical_strongs column to stepbible_verses table if it doesn't exist.

    Stores the normalized (unpadded) canonical Strong's key computed from
    ``extract_lexical_strongs`` + ``normalize_strongs``, enabling fast concordance
    lookups without per-query derivation.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists or
        if stepbible_verses table does not exist yet.
    """
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()

            # If the table doesn't exist yet, nothing to do
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='stepbible_verses'")
            if cursor.fetchone()[0] == 0:
                logger.debug("stepbible_verses table not present; skipping lexical_strongs migration")
                return False

            # Idempotent: check whether column already exists
            cursor.execute("PRAGMA table_info(stepbible_verses)")
            columns = {row[1] for row in cursor.fetchall()}
            if "lexical_strongs" in columns:
                logger.debug("lexical_strongs column already exists in stepbible_verses")
                return False

            cursor.execute("ALTER TABLE stepbible_verses ADD COLUMN lexical_strongs TEXT")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_stepbible_lexical_strongs ON stepbible_verses(lexical_strongs)"
            )
            conn.commit()
            logger.info("Added lexical_strongs column and index to stepbible_verses")
            return True

    except Exception as e:
        logger.error("Failed to add lexical_strongs column: %s", e)
        raise


def add_provenance_table(db_path: Path) -> bool:
    """Add the central provenance table for auditable data attribution.

    One uniform audit record per enrichment element, keyed by
    (entity_type, entity_id): where it came from, whether it is trusted, why,
    and — for AI output — a 0.00-1.00 confidence.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='provenance'")
            if cursor.fetchone()[0] > 0:
                logger.debug("provenance table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_detail TEXT,
                    trust_tier TEXT NOT NULL CHECK(trust_tier IN ('A', 'B', 'C')),
                    trust_rationale TEXT NOT NULL,
                    generated_by TEXT,
                    grounding_json TEXT,
                    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
                    pipeline_version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entity_type, entity_id)
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_entity ON provenance(entity_type, entity_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_tier ON provenance(trust_tier)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_source ON provenance(source)")
            conn.commit()
            logger.info("Added provenance table")
            return True
    except Exception as e:
        logger.error("Failed to add provenance table: %s", e)
        raise


def add_word_richness_table(db_path: Path) -> bool:
    """Add word_richness table for precomputed meaning-loss scores.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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
        with _connect(db_path) as conn:
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


def add_semantic_domain_tables(db_path: Path) -> bool:
    """Add semantic_domains and strongs_domain_mappings tables for Louw-Nida classification."""
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='semantic_domains'")
            if cursor.fetchone()[0] > 0:
                logger.debug("semantic_domains table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_domains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain_code TEXT UNIQUE NOT NULL,
                    domain_name TEXT NOT NULL,
                    parent_domain TEXT,
                    description TEXT,
                    level INTEGER DEFAULT 1
                )
            """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strongs_domain_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strongs_number TEXT NOT NULL,
                    domain_code TEXT NOT NULL,
                    confidence REAL DEFAULT 0.9,
                    UNIQUE(strongs_number, domain_code),
                    FOREIGN KEY (domain_code) REFERENCES semantic_domains(domain_code)
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain_strongs ON strongs_domain_mappings(strongs_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain_code ON strongs_domain_mappings(domain_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain_parent ON semantic_domains(parent_domain)")
            conn.commit()
            logger.info("Added semantic_domains and strongs_domain_mappings tables")
            return True
    except Exception as e:
        logger.error("Failed to add semantic domain tables: %s", e)
        raise


def add_syntax_tree_table(db_path: Path) -> bool:
    """Add syntax_trees table for MACULA treebank data."""
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='syntax_trees'")
            if cursor.fetchone()[0] > 0:
                logger.debug("syntax_trees table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS syntax_trees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL UNIQUE,
                    book_id INTEGER NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    word_num INTEGER,
                    node_type TEXT NOT NULL,
                    role TEXT,
                    parent_id TEXT,
                    clause_type TEXT,
                    relation TEXT,
                    depth INTEGER DEFAULT 0,
                    text_content TEXT
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_syntax_verse ON syntax_trees(book_id, chapter, verse)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_syntax_parent ON syntax_trees(parent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_syntax_clause ON syntax_trees(clause_type)")
            conn.commit()
            logger.info("Added syntax_trees table")
            return True
    except Exception as e:
        logger.error("Failed to add syntax_trees table: %s", e)
        raise


def add_discourse_annotation_table(db_path: Path) -> bool:
    """Add discourse_annotations table for OpenText.org discourse data."""
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='discourse_annotations'")
            if cursor.fetchone()[0] > 0:
                logger.debug("discourse_annotations table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS discourse_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    start_chapter INTEGER NOT NULL,
                    start_verse INTEGER NOT NULL,
                    end_chapter INTEGER NOT NULL,
                    end_verse INTEGER NOT NULL,
                    discourse_type TEXT NOT NULL,
                    function_label TEXT,
                    relation_to_context TEXT,
                    description TEXT,
                    prominence INTEGER DEFAULT 0
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_discourse_verse "
                "ON discourse_annotations(book_id, start_chapter, start_verse)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_discourse_type ON discourse_annotations(discourse_type)")
            conn.commit()
            logger.info("Added discourse_annotations table")
            return True
    except Exception as e:
        logger.error("Failed to add discourse_annotations table: %s", e)
        raise


def add_manuscript_variant_table(db_path: Path) -> bool:
    """Add manuscript_variants table for textual criticism data."""
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='manuscript_variants'")
            if cursor.fetchone()[0] > 0:
                logger.debug("manuscript_variants table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS manuscript_variants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    variant_type TEXT NOT NULL,
                    base_text TEXT,
                    variant_text TEXT,
                    manuscripts TEXT,
                    explanation TEXT,
                    significance TEXT CHECK(significance IN ('major', 'minor', 'orthographic'))
                        DEFAULT 'minor',
                    confidence REAL DEFAULT 0.8
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_variants_verse ON manuscript_variants(book_id, chapter, verse)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_type ON manuscript_variants(variant_type)")
            conn.commit()
            logger.info("Added manuscript_variants table")
            return True
    except Exception as e:
        logger.error("Failed to add manuscript_variants table: %s", e)
        raise


def add_community_contribution_tables(db_path: Path) -> bool:
    """Add community_contributions and contribution_reviews tables."""
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='community_contributions'")
            if cursor.fetchone()[0] > 0:
                logger.debug("community_contributions table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS community_contributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    chapter INTEGER,
                    verse INTEGER,
                    contribution_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author_id TEXT DEFAULT 'anonymous',
                    status TEXT CHECK(status IN ('pending', 'approved', 'rejected')) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS contribution_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contribution_id INTEGER NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    decision TEXT CHECK(decision IN ('approve', 'reject', 'request_changes')) NOT NULL,
                    review_note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (contribution_id) REFERENCES community_contributions(id)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_contributions_book ON community_contributions(book_id, chapter, verse)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contributions_status ON community_contributions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_contrib ON contribution_reviews(contribution_id)")
            conn.commit()
            logger.info("Added community_contributions and contribution_reviews tables")
            return True
    except Exception as e:
        logger.error("Failed to add community contribution tables: %s", e)
        raise


def add_concept_proposal_table(db_path: Path) -> bool:
    """Add concept_proposals table for collaborative concept editing."""
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='concept_proposals'")
            if cursor.fetchone()[0] > 0:
                logger.debug("concept_proposals table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT NOT NULL,
                    proposed_by TEXT DEFAULT 'anonymous',
                    proposal_type TEXT CHECK(proposal_type IN ('new', 'edit', 'merge', 'delete')) NOT NULL,
                    description TEXT NOT NULL,
                    hebrew_terms_json TEXT,
                    greek_terms_json TEXT,
                    verse_mappings_json TEXT,
                    status TEXT CHECK(status IN ('pending', 'approved', 'rejected')) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_proposals_concept ON concept_proposals(concept_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_proposals_status ON concept_proposals(status)")
            conn.commit()
            logger.info("Added concept_proposals table")
            return True
    except Exception as e:
        logger.error("Failed to add concept_proposals table: %s", e)
        raise


def add_ml_and_graph_tables(db_path: Path) -> bool:
    """Add concept_feedback and semantic_relationship_graph tables for ML and visualization."""
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='concept_feedback'")
            if cursor.fetchone()[0] > 0:
                logger.debug("concept_feedback table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT NOT NULL,
                    verse_id TEXT NOT NULL,
                    feedback_type TEXT CHECK(feedback_type IN ('relevant', 'irrelevant', 'partial')) NOT NULL,
                    user_id TEXT DEFAULT 'anonymous',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_relationship_graph (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_concept TEXT NOT NULL,
                    target_concept TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    evidence_count INTEGER DEFAULT 0,
                    shared_strongs_json TEXT,
                    UNIQUE(source_concept, target_concept, relationship_type)
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_concept ON concept_feedback(concept_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_source ON semantic_relationship_graph(source_concept)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_target ON semantic_relationship_graph(target_concept)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_type ON semantic_relationship_graph(relationship_type)"
            )
            conn.commit()
            logger.info("Added concept_feedback and semantic_relationship_graph tables")
            return True
    except Exception as e:
        logger.error("Failed to add ML and graph tables: %s", e)
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
    # Phase 9 extended capabilities
    (add_semantic_domain_tables, "semantic domain tables"),
    (add_syntax_tree_table, "syntax_trees table"),
    (add_discourse_annotation_table, "discourse_annotations table"),
    (add_manuscript_variant_table, "manuscript_variants table"),
    (add_community_contribution_tables, "community contribution tables"),
    (add_concept_proposal_table, "concept_proposals table"),
    (add_ml_and_graph_tables, "ML and graph tables"),
    # Phase 8 user features
    (add_user_annotation_tables, "user annotation tables"),
    # Phase 0a provenance foundation
    (add_provenance_table, "provenance table"),
    # Strong's concordance: pre-computed lexical key for fast concordance queries
    (add_stepbible_lexical_strongs_column, "stepbible lexical_strongs column"),
    # TSK cross-reference staging table
    (add_cross_reference_candidates_table, "cross_reference_candidates table"),
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
