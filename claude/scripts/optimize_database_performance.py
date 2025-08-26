#!/usr/bin/env python3
"""
Database Performance Optimization

This script adds indexes and optimizations to improve semantic search performance.
"""

import sys
import sqlite3
import time
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database.sqlite_manager import SQLiteManager
from abba.logging_setup import logger


def add_performance_indexes(db_path: Path):
    """Add indexes to improve query performance."""
    
    logger.info("🔧 Adding performance indexes to database...")
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        # Get existing indexes
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        existing_indexes = {row[0] for row in cursor.fetchall()}
        logger.info(f"Found {len(existing_indexes)} existing indexes")
        
        # Define indexes to create
        indexes = [
            # For Strong's concordance searches
            ("idx_strongs_lexical", "stepbible_verses(strongs_lexical)"),
            ("idx_normalized_word", "stepbible_verses(normalized_word)"),
            ("idx_language", "stepbible_verses(language)"),
            ("idx_verse_location", "stepbible_verses(book, chapter, verse)"),
            
            # For concept mappings
            ("idx_concept_name", "concept_mappings(concept_name)"),
            ("idx_concept_verse", "concept_mappings(verse_id)"),
            ("idx_concept_confidence", "concept_mappings(concept_name, confidence DESC)"),
            ("idx_concept_type", "concept_mappings(match_type)"),
            
            # For lexicon lookups
            ("idx_lexicon_strongs", "lexicon(strongs_number)"),
            
            # For import tracking
            ("idx_operation_state", "operation_state(operation_type, target_id)"),
            ("idx_import_status", "import_status(translation_id)"),
            
            # Composite indexes for common queries
            ("idx_strongs_language", "stepbible_verses(strongs_lexical, language)"),
            ("idx_book_strongs", "stepbible_verses(book, strongs_lexical)"),
        ]
        
        # Create indexes
        created_count = 0
        for index_name, index_def in indexes:
            if index_name not in existing_indexes:
                try:
                    start_time = time.time()
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {index_def}")
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Created index {index_name} ({elapsed:.2f}s)")
                    created_count += 1
                except sqlite3.Error as e:
                    logger.warning(f"⚠️  Failed to create index {index_name}: {e}")
            else:
                logger.debug(f"   Index {index_name} already exists")
        
        # Analyze tables to update statistics
        logger.info("\n📊 Analyzing tables for query optimization...")
        tables = ['stepbible_verses', 'concept_mappings', 'lexicon', 'translations']
        for table in tables:
            try:
                cursor.execute(f"ANALYZE {table}")
                logger.info(f"✅ Analyzed table: {table}")
            except sqlite3.Error:
                pass  # Table might not exist
        
        conn.commit()
        
        logger.info(f"\n✅ Created {created_count} new indexes")


def optimize_database_settings(db_path: Path):
    """Apply database optimization settings."""
    
    logger.info("\n⚙️  Applying database optimization settings...")
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        # Set pragmas for performance
        optimizations = [
            ("PRAGMA journal_mode = WAL", "Write-Ahead Logging for better concurrency"),
            ("PRAGMA synchronous = NORMAL", "Balanced durability/performance"),
            ("PRAGMA cache_size = -64000", "64MB cache for better performance"),
            ("PRAGMA temp_store = MEMORY", "Use memory for temporary tables"),
            ("PRAGMA mmap_size = 268435456", "256MB memory-mapped I/O"),
        ]
        
        for pragma, description in optimizations:
            try:
                cursor.execute(pragma)
                logger.info(f"✅ {description}")
            except sqlite3.Error as e:
                logger.warning(f"⚠️  Failed to set {pragma}: {e}")


def check_database_stats(db_path: Path):
    """Display database statistics."""
    
    logger.info("\n📊 Database Statistics")
    logger.info("=" * 60)
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        
        # Table sizes
        tables = ['stepbible_verses', 'concept_mappings', 'lexicon', 'translations']
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"{table:<20} {count:>10,} rows")
            except sqlite3.Error:
                logger.info(f"{table:<20} {'N/A':>10}")
        
        # Database file size
        db_size = db_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"\nDatabase size: {db_size:.1f} MB")
        
        # Check page count and size
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        logger.info(f"Pages: {page_count:,} x {page_size} bytes")


def vacuum_database(db_path: Path):
    """Vacuum database to reclaim space and optimize layout."""
    
    logger.info("\n🧹 Vacuuming database...")
    logger.info("This may take a while for large databases...")
    
    start_time = time.time()
    
    with SQLiteManager(db_path).get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("VACUUM")
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Vacuum completed in {elapsed:.1f} seconds")


def main():
    """Run all database optimizations."""
    
    logger.info("🚀 Database Performance Optimization")
    logger.info("=" * 60)
    
    # Load configuration
    config = config_manager.load_config()
    db_path = config.data_dir / "abba.db"
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False
    
    try:
        # Show initial stats
        check_database_stats(db_path)
        
        # Add indexes
        add_performance_indexes(db_path)
        
        # Apply optimizations
        optimize_database_settings(db_path)
        
        # Vacuum database (optional - can be slow)
        response = input("\nVacuum database? This can take several minutes for large databases (y/N): ")
        if response.lower() == 'y':
            vacuum_database(db_path)
        
        # Show final stats
        logger.info("\n" + "=" * 60)
        check_database_stats(db_path)
        
        logger.info("\n✅ Database optimization complete!")
        logger.info("\nPerformance improvements:")
        logger.info("• Faster Strong's number searches")
        logger.info("• Faster concept mapping queries")
        logger.info("• Better join performance")
        logger.info("• Improved concurrent access")
        
        return True
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)