"""Database migration utilities for ABBA."""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

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
            cursor.execute("""
                SELECT COUNT(*) FROM pragma_table_info('translations') 
                WHERE name='canon'
            """)

            if cursor.fetchone()[0] > 0:
                logger.debug("Canon column already exists")
                return False

            # Add column
            cursor.execute("""
                ALTER TABLE translations 
                ADD COLUMN canon TEXT CHECK(canon IN ('hebrew', 'protestant', 'catholic', 'orthodox', 'ethiopian'))
            """)

            # Update existing rows with detected canons
            from ..parallel_import import get_translation_canon

            cursor.execute("SELECT id FROM translations")
            translation_ids = [row[0] for row in cursor.fetchall()]

            source_db_path = db_path.parent / "bible.db"
            if source_db_path.exists():
                for trans_id in translation_ids:
                    canon_enum = get_translation_canon(trans_id, str(source_db_path))
                    cursor.execute("UPDATE translations SET canon = ? WHERE id = ?", (canon_enum.value, trans_id))

            conn.commit()
            logger.info(f"Added canon column and updated {len(translation_ids)} translations")
            return True

    except Exception as e:
        logger.error(f"Failed to add canon column: {e}")
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
            cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name='stepbible_verses'
            """)

            if cursor.fetchone()[0] > 0:
                logger.debug("stepbible_verses table already exists")
                return False

            # Create table
            cursor.execute("""
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
            """)

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
        logger.error(f"Failed to add stepbible_verses table: {e}")
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
            cursor.execute("""
                SELECT COUNT(*) FROM pragma_table_info('translations') 
                WHERE name IN ('is_partial_canon', 'apocrypha_count')
            """)

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
        logger.error(f"Failed to add partial canon columns: {e}")
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
            cursor.execute("""
                SELECT COUNT(*) FROM pragma_table_info('translations') 
                WHERE name IN ('has_import_failures', 'failed_verse_count')
            """)

            columns_exist = cursor.fetchone()[0] >= 2

            # Check if table exists
            cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name='failed_imports'
            """)

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
                cursor.execute("""
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
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_failed_imports_translation 
                    ON failed_imports(translation_id)
                """)

            conn.commit()
            logger.info("Added import failure tracking")
            return True

    except Exception as e:
        logger.error(f"Failed to add import failure tracking: {e}")
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
            cursor.execute("""
                SELECT COUNT(*) FROM pragma_table_info('stepbible_verses') 
                WHERE name='data_hash'
            """)

            if cursor.fetchone()[0] > 0:
                logger.debug("data_hash column already exists in stepbible_verses")
                return False

            # Add column
            cursor.execute("""
                ALTER TABLE stepbible_verses 
                ADD COLUMN data_hash INTEGER
            """)

            conn.commit()
            logger.info("Added data_hash column to stepbible_verses table")
            return True

    except Exception as e:
        logger.error(f"Failed to add data_hash column: {e}")
        raise


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

    # Add canon column if needed
    if add_canon_column(db_path):
        migrations_run.append("canon column")

    # Add stepbible_verses table if needed
    if add_stepbible_verses_table(db_path):
        migrations_run.append("stepbible_verses table")

    # Add partial canon tracking columns if needed
    if add_partial_canon_columns(db_path):
        migrations_run.append("partial canon columns")

    # Add import failure tracking if needed
    if add_import_failure_tracking(db_path):
        migrations_run.append("import failure tracking")

    # Add hash column to stepbible_verses if needed
    if add_stepbible_hash_column(db_path):
        migrations_run.append("stepbible hash column")

    if migrations_run:
        logger.info(f"Database migrations completed: {', '.join(migrations_run)}")
    else:
        logger.debug("No migrations needed")
