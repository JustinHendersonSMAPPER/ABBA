"""
SQLite database exporter for ABBA canonical data.

Exports canonical biblical data to optimized SQLite database with FTS5 support,
suitable for mobile apps, offline access, and embedded systems.
"""

import sqlite3
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
import json
import gzip

from .base import (
    DataExporter,
    CanonicalDataset,
    ExportConfig,
    ExportResult,
    ExportStatus,
    ExportError,
    ValidationResult,
    ExportUtilities,
    StreamingDataProcessor,
)
from ..parsers.translation_parser import TranslationVerse
from ..annotations.models import Annotation
from ..timeline.models import Event, TimePeriod


@dataclass
class SQLiteConfig(ExportConfig):
    """SQLite-specific export configuration."""

    # Database settings
    enable_fts5: bool = True
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True

    # Optimization settings
    vacuum_on_completion: bool = True
    analyze_on_completion: bool = True

    # FTS5 settings
    fts5_tokenizer: str = "unicode61"
    fts5_remove_diacritics: bool = True

    # Compression
    compress_large_text: bool = True
    compression_threshold: int = 1024  # bytes

    def __post_init__(self):
        """Ensure SQLite format type is set."""
        from .base import ExportFormat

        self.format_type = ExportFormat.SQLITE


class SQLiteExporter(DataExporter):
    """Exports canonical data to SQLite database."""

    def __init__(self, config: SQLiteConfig):
        """Initialize SQLite exporter."""
        super().__init__(config)
        self.config: SQLiteConfig = config
        self.db_path = Path(config.output_path)
        self.processor = StreamingDataProcessor(batch_size=config.batch_size)

        # SQL schemas
        self._schema_sql = self._get_schema_sql()
        self._index_sql = self._get_index_sql()
        self._fts5_sql = self._get_fts5_sql()

    def validate_config(self) -> ValidationResult:
        """Validate SQLite exporter configuration."""
        validation = ValidationResult(is_valid=True)

        # Check output path
        if not self.config.output_path:
            validation.add_error("Output path is required")
            return validation

        # Ensure output directory exists
        try:
            ExportUtilities.ensure_directory(self.db_path.parent)
        except Exception as e:
            validation.add_error(f"Cannot create output directory: {str(e)}")

        # Check SQLite availability
        try:
            sqlite3.sqlite_version
        except Exception as e:
            validation.add_error(f"SQLite not available: {str(e)}")

        # Validate batch size
        if self.config.batch_size <= 0:
            validation.add_error("Batch size must be positive")

        return validation

    def get_supported_features(self) -> List[str]:
        """Get list of features supported by SQLite exporter."""
        return [
            "full_text_search",
            "offline_access",
            "mobile_optimized",
            "verse_lookup",
            "cross_references",
            "timeline_queries",
            "annotations",
            "compressed_storage",
        ]

    async def export(self, data: CanonicalDataset) -> ExportResult:
        """Export canonical data to SQLite database."""
        await self.prepare_export(data)

        try:
            # Initialize database
            await self._initialize_database()

            # Export core data
            await self._export_verses(data.verses)

            # Export enrichment data
            if data.annotations:
                await self._export_annotations(data.annotations)

            if data.cross_references:
                await self._export_cross_references(data.cross_references)

            if data.timeline_events:
                await self._export_timeline_events(data.timeline_events)

            if data.timeline_periods:
                await self._export_timeline_periods(data.timeline_periods)

            # Finalize database
            await self._finalize_database(data.metadata)

            result = self.create_result(ExportStatus.COMPLETED)

        except Exception as e:
            self.logger.error(f"SQLite export failed: {str(e)}")
            result = self.create_result(
                ExportStatus.FAILED, ExportError(f"SQLite export failed: {str(e)}", stage="export")
            )

        return await self.finalize_export(result)

    async def _initialize_database(self):
        """Initialize SQLite database with schema."""
        self.logger.info(f"Initializing SQLite database: {self.db_path}")

        # Remove existing database
        if self.db_path.exists():
            self.db_path.unlink()

        # Create new database
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")

            # Enable foreign keys
            if self.config.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys=ON")

            # Set performance pragmas
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")

            # Create schema
            conn.executescript(self._schema_sql)

            # Create indices
            conn.executescript(self._index_sql)

            # Create FTS5 tables if enabled
            if self.config.enable_fts5:
                conn.executescript(self._fts5_sql)

            conn.commit()

    async def _export_verses(self, verses: Iterator[TranslationVerse]):
        """Export verses to database."""
        self.logger.info("Exporting verses to SQLite")

        verse_count = 0

        async def process_verse_batch(verse_batch: List[TranslationVerse]):
            nonlocal verse_count

            with sqlite3.connect(self.db_path) as conn:
                # Prepare verse data
                verse_data = []
                original_data = []
                translation_data = []

                for verse in verse_batch:
                    # Handle different verse types
                    translations_json = None
                    metadata_json = None
                    
                    # Check if it's a TranslationVerse (simple) or more complex type
                    if hasattr(verse, 'text') and not hasattr(verse, 'translations'):
                        # Simple TranslationVerse - create translations dict from text
                        translations_json = json.dumps({"default": verse.text})
                    elif hasattr(verse, 'translations'):
                        # Complex verse with multiple translations
                        translations_json = json.dumps(verse.translations) if verse.translations else None
                    
                    if hasattr(verse, 'metadata'):
                        metadata_json = json.dumps(verse.metadata) if verse.metadata else None
                    
                    # Main verse record
                    verse_data.append(
                        (
                            str(verse.verse_id),
                            verse.verse_id.book,
                            verse.verse_id.chapter,
                            verse.verse_id.verse,
                            translations_json,
                            metadata_json,
                        )
                    )

                    # Original language data
                    if hasattr(verse, 'hebrew_tokens') and verse.hebrew_tokens:
                        for token in verse.hebrew_tokens:
                            original_data.append(
                                (
                                    str(verse.verse_id),
                                    "hebrew",
                                    token.get("word", ""),
                                    token.get("lemma", ""),
                                    token.get("morph", ""),
                                    token.get("strongs", ""),
                                    json.dumps(token),
                                )
                            )

                    if hasattr(verse, 'greek_tokens') and verse.greek_tokens:
                        for token in verse.greek_tokens:
                            original_data.append(
                                (
                                    str(verse.verse_id),
                                    "greek",
                                    token.get("word", ""),
                                    token.get("lemma", ""),
                                    token.get("morph", ""),
                                    token.get("strongs", ""),
                                    json.dumps(token),
                                )
                            )

                    # Translation data
                    verse_translations = None
                    if hasattr(verse, 'text') and not hasattr(verse, 'translations'):
                        # Simple TranslationVerse
                        verse_translations = {"default": verse.text}
                    elif hasattr(verse, 'translations'):
                        verse_translations = verse.translations
                    
                    if verse_translations:
                        for translation_id, text in verse_translations.items():
                            # Compress large text if enabled
                            if (
                                self.config.compress_large_text
                                and len(text.encode("utf-8")) > self.config.compression_threshold
                            ):
                                text = gzip.compress(text.encode("utf-8"))
                                is_compressed = True
                            else:
                                is_compressed = False

                            translation_data.append(
                                (str(verse.verse_id), translation_id, text, is_compressed)
                            )

                # Insert data
                conn.executemany(
                    """
                    INSERT INTO verses (verse_id, book, chapter, verse, translations_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    verse_data,
                )

                if original_data:
                    conn.executemany(
                        """
                        INSERT INTO original_language (verse_id, language, word, lemma, morph, strongs, token_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        original_data,
                    )

                if translation_data:
                    conn.executemany(
                        """
                        INSERT INTO verse_translations (verse_id, translation_id, text, is_compressed)
                        VALUES (?, ?, ?, ?)
                    """,
                        translation_data,
                    )

                # Update FTS5 if enabled
                if self.config.enable_fts5 and translation_data:
                    fts_data = []
                    for verse_id, translation_id, text, is_compressed in translation_data:
                        if not is_compressed:
                            fts_data.append((verse_id, translation_id, text))

                    if fts_data:
                        conn.executemany(
                            """
                            INSERT INTO verses_fts (verse_id, translation_id, text)
                            VALUES (?, ?, ?)
                        """,
                            fts_data,
                        )

                conn.commit()
                verse_count += len(verse_batch)
                self.update_progress(verse_count, "verses")

        # Process verses in batches
        async for batch in self.processor.process_in_batches(
            verses, process_verse_batch, lambda count: self.update_progress(count, "verses")
        ):
            pass

        self.logger.info(f"Exported {verse_count} verses")

    async def _export_annotations(self, annotations: Iterator[Annotation]):
        """Export annotations to database."""
        self.logger.info("Exporting annotations to SQLite")

        annotation_count = 0

        async def process_annotation_batch(annotation_batch: List[Annotation]):
            nonlocal annotation_count

            with sqlite3.connect(self.db_path) as conn:
                annotation_data = []
                topic_data = []

                for annotation in annotation_batch:
                    annotation_data.append(
                        (
                            annotation.id,
                            str(annotation.start_verse),
                            (
                                annotation.annotation_type.value
                                if annotation.annotation_type
                                else None
                            ),
                            annotation.level.value if annotation.level else None,
                            annotation.confidence.overall_score if annotation.confidence else 0.0,
                            json.dumps({"source": annotation.source, "verified": annotation.verified}),
                        )
                    )

                    # Extract topics
                    if annotation.topic_id:
                        topic_data.append(
                            (
                                annotation.id,
                                annotation.topic_id,
                                annotation.topic_name or "",
                                annotation.confidence.overall_score if annotation.confidence else 1.0,
                            )
                        )

                # Insert data
                conn.executemany(
                    """
                    INSERT INTO annotations (id, verse_id, type, level, confidence, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    annotation_data,
                )

                if topic_data:
                    conn.executemany(
                        """
                        INSERT INTO annotation_topics (annotation_id, topic_id, topic_name, confidence)
                        VALUES (?, ?, ?, ?)
                    """,
                        topic_data,
                    )

                conn.commit()
                annotation_count += len(annotation_batch)
                self.update_progress(annotation_count, "annotations")

        # Process annotations in batches
        async for batch in self.processor.process_in_batches(
            annotations,
            process_annotation_batch,
            lambda count: self.update_progress(count, "annotations"),
        ):
            pass

        self.logger.info(f"Exported {annotation_count} annotations")

    async def _export_cross_references(self, cross_references: Iterator[Dict[str, Any]]):
        """Export cross-references to database."""
        self.logger.info("Exporting cross-references to SQLite")

        ref_count = 0

        async def process_ref_batch(ref_batch: List[Dict[str, Any]]):
            nonlocal ref_count

            with sqlite3.connect(self.db_path) as conn:
                ref_data = []

                for ref in ref_batch:
                    ref_data.append(
                        (
                            str(ref.get("source_verse_id", "")),
                            str(ref.get("target_verse_id", "")),
                            ref.get("type", "reference"),
                            ref.get("confidence", 1.0),
                            ref.get("source", ""),
                            json.dumps(ref.get("metadata", {})),
                        )
                    )

                conn.executemany(
                    """
                    INSERT INTO cross_references (source_verse_id, target_verse_id, type, confidence, source, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    ref_data,
                )

                conn.commit()
                ref_count += len(ref_batch)
                self.update_progress(ref_count, "cross_references")

        # Process cross-references in batches
        async for batch in self.processor.process_in_batches(
            cross_references,
            process_ref_batch,
            lambda count: self.update_progress(count, "cross_references"),
        ):
            pass

        self.logger.info(f"Exported {ref_count} cross-references")

    async def _export_timeline_events(self, events: Iterator[Event]):
        """Export timeline events to database."""
        self.logger.info("Exporting timeline events to SQLite")

        event_count = 0

        async def process_event_batch(event_batch: List[Event]):
            nonlocal event_count

            with sqlite3.connect(self.db_path) as conn:
                event_data = []

                for event in event_batch:
                    # Convert event to database format
                    event_dict = event.to_dict()

                    event_data.append(
                        (
                            event.id,
                            event.name,
                            event.description,
                            event.event_type.value,
                            event.certainty_level.value,
                            json.dumps(event_dict.get("time_point")) if event.time_point else None,
                            json.dumps(event_dict.get("time_range")) if event.time_range else None,
                            json.dumps(event_dict.get("location")) if event.location else None,
                            json.dumps(event_dict.get("participants")),
                            json.dumps(event.categories),
                            json.dumps([str(v) for v in event.verse_refs]) if event.verse_refs else None,
                            json.dumps(event_dict),
                        )
                    )

                conn.executemany(
                    """
                    INSERT INTO timeline_events (
                        id, name, description, event_type, certainty_level,
                        time_point_json, time_range_json, location_json,
                        participants_json, categories_json, verse_refs_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    event_data,
                )

                conn.commit()
                event_count += len(event_batch)
                self.update_progress(event_count, "events")

        # Process events in batches
        async for batch in self.processor.process_in_batches(
            events, process_event_batch, lambda count: self.update_progress(count, "events")
        ):
            pass

        self.logger.info(f"Exported {event_count} timeline events")

    async def _export_timeline_periods(self, periods: Iterator[TimePeriod]):
        """Export timeline periods to database."""
        self.logger.info("Exporting timeline periods to SQLite")

        period_count = 0

        async def process_period_batch(period_batch: List[TimePeriod]):
            nonlocal period_count

            with sqlite3.connect(self.db_path) as conn:
                period_data = []

                for period in period_batch:
                    period_dict = period.to_dict()

                    period_data.append(
                        (
                            period.id,
                            period.name,
                            period.description,
                            period.parent_period,
                            json.dumps(period.child_periods),
                            (
                                json.dumps(period_dict.get("time_range"))
                                if period.time_range
                                else None
                            ),
                            json.dumps(period.events),
                            json.dumps(period_dict),
                        )
                    )

                conn.executemany(
                    """
                    INSERT INTO timeline_periods (
                        id, name, description, parent_period_id,
                        child_periods_json, time_range_json, events_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    period_data,
                )

                conn.commit()
                period_count += len(period_batch)

        # Process periods in batches
        async for batch in self.processor.process_in_batches(periods, process_period_batch):
            pass

        self.logger.info(f"Exported {period_count} timeline periods")

    async def _finalize_database(self, metadata: Dict[str, Any]):
        """Finalize database with metadata and optimization."""
        self.logger.info("Finalizing SQLite database")

        with sqlite3.connect(self.db_path) as conn:
            # Insert export metadata
            conn.execute(
                """
                INSERT INTO export_metadata (key, value, created_at)
                VALUES (?, ?, datetime('now'))
            """,
                (
                    "export_info",
                    json.dumps(
                        {
                            "format": "sqlite",
                            "version": "1.0",
                            "exporter": "ABBA SQLite Exporter",
                            "metadata": metadata,
                        }
                    ),
                ),
            )

            # Insert statistics
            stats = {
                "total_verses": self.stats.processed_verses,
                "total_annotations": self.stats.processed_annotations,
                "total_events": self.stats.processed_events,
                "total_cross_references": self.stats.processed_cross_references,
            }

            conn.execute(
                """
                INSERT INTO export_metadata (key, value, created_at)
                VALUES (?, ?, datetime('now'))
            """,
                ("statistics", json.dumps(stats)),
            )

            # Optimize database
            if self.config.analyze_on_completion:
                self.logger.info("Running ANALYZE for query optimization")
                conn.execute("ANALYZE")

            conn.commit()
            
            if self.config.vacuum_on_completion:
                self.logger.info("Running VACUUM for storage optimization")
                # VACUUM must be run outside of a transaction
                conn.isolation_level = None  # Auto-commit mode
                conn.execute("VACUUM")
                conn.isolation_level = ''  # Restore default

        self.logger.info("SQLite database finalized")

    def _get_schema_sql(self) -> str:
        """Get database schema SQL."""
        return """
        -- Main verses table
        CREATE TABLE verses (
            verse_id TEXT PRIMARY KEY,
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            translations_json TEXT,
            metadata_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Original language tokens
        CREATE TABLE original_language (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id TEXT NOT NULL,
            language TEXT NOT NULL, -- 'hebrew' or 'greek'
            word TEXT NOT NULL,
            lemma TEXT,
            morph TEXT,
            strongs TEXT,
            token_json TEXT,
            FOREIGN KEY (verse_id) REFERENCES verses(verse_id)
        );
        
        -- Verse translations (separate for efficiency)
        CREATE TABLE verse_translations (
            verse_id TEXT NOT NULL,
            translation_id TEXT NOT NULL,
            text TEXT NOT NULL,
            is_compressed BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (verse_id, translation_id),
            FOREIGN KEY (verse_id) REFERENCES verses(verse_id)
        );
        
        -- Annotations
        CREATE TABLE annotations (
            id TEXT PRIMARY KEY,
            verse_id TEXT NOT NULL,
            type TEXT,
            level TEXT,
            confidence REAL DEFAULT 1.0,
            metadata_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (verse_id) REFERENCES verses(verse_id)
        );
        
        -- Annotation topics
        CREATE TABLE annotation_topics (
            annotation_id TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            topic_name TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            PRIMARY KEY (annotation_id, topic_id),
            FOREIGN KEY (annotation_id) REFERENCES annotations(id)
        );
        
        -- Cross-references
        CREATE TABLE cross_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_verse_id TEXT NOT NULL,
            target_verse_id TEXT NOT NULL,
            type TEXT DEFAULT 'reference',
            confidence REAL DEFAULT 1.0,
            source TEXT,
            metadata_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_verse_id) REFERENCES verses(verse_id),
            FOREIGN KEY (target_verse_id) REFERENCES verses(verse_id)
        );
        
        -- Timeline events
        CREATE TABLE timeline_events (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            event_type TEXT NOT NULL,
            certainty_level TEXT NOT NULL,
            time_point_json TEXT,
            time_range_json TEXT,
            location_json TEXT,
            participants_json TEXT,
            categories_json TEXT,
            verse_refs_json TEXT,
            metadata_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Timeline periods
        CREATE TABLE timeline_periods (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            parent_period_id TEXT,
            child_periods_json TEXT,
            time_range_json TEXT,
            events_json TEXT,
            metadata_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_period_id) REFERENCES timeline_periods(id)
        );
        
        -- Export metadata
        CREATE TABLE export_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Books metadata
        CREATE TABLE books (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            testament TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            chapter_count INTEGER,
            verse_count INTEGER,
            metadata_json TEXT
        );
        """

    def _get_index_sql(self) -> str:
        """Get database indices SQL."""
        return """
        -- Verse lookup indices
        CREATE INDEX idx_verses_book_chapter ON verses(book, chapter);
        CREATE INDEX idx_verses_book_chapter_verse ON verses(book, chapter, verse);
        
        -- Original language indices
        CREATE INDEX idx_original_language_verse ON original_language(verse_id);
        CREATE INDEX idx_original_language_lemma ON original_language(lemma);
        CREATE INDEX idx_original_language_strongs ON original_language(strongs);
        
        -- Translation indices
        CREATE INDEX idx_verse_translations_translation ON verse_translations(translation_id);
        
        -- Annotation indices
        CREATE INDEX idx_annotations_verse ON annotations(verse_id);
        CREATE INDEX idx_annotations_type ON annotations(type);
        CREATE INDEX idx_annotation_topics_topic ON annotation_topics(topic_id);
        
        -- Cross-reference indices
        CREATE INDEX idx_cross_references_source ON cross_references(source_verse_id);
        CREATE INDEX idx_cross_references_target ON cross_references(target_verse_id);
        CREATE INDEX idx_cross_references_type ON cross_references(type);
        
        -- Timeline indices
        CREATE INDEX idx_timeline_events_type ON timeline_events(event_type);
        CREATE INDEX idx_timeline_events_certainty ON timeline_events(certainty_level);
        CREATE INDEX idx_timeline_periods_parent ON timeline_periods(parent_period_id);
        """

    def _get_fts5_sql(self) -> str:
        """Get FTS5 virtual table SQL."""
        tokenizer = f"tokenize='{self.config.fts5_tokenizer}"
        if self.config.fts5_remove_diacritics:
            tokenizer += " remove_diacritics 1"
        tokenizer += "'"

        return f"""
        -- Full-text search for verses
        CREATE VIRTUAL TABLE verses_fts USING fts5(
            verse_id UNINDEXED,
            translation_id UNINDEXED, 
            text,
            {tokenizer}
        );
        
        -- Full-text search for timeline events
        CREATE VIRTUAL TABLE events_fts USING fts5(
            event_id UNINDEXED,
            name,
            description,
            {tokenizer}
        );
        """

    async def validate_output(self, result: ExportResult) -> ValidationResult:
        """Validate SQLite database output."""
        validation = await super().validate_output(result)

        if not validation.is_valid:
            return validation

        try:
            # Check database integrity
            with sqlite3.connect(self.db_path) as conn:
                # Check database integrity
                integrity_result = conn.execute("PRAGMA integrity_check").fetchone()
                if integrity_result[0] != "ok":
                    validation.add_error(f"Database integrity check failed: {integrity_result[0]}")

                # Check table existence
                tables = conn.execute(
                    """
                    SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
                ).fetchall()

                expected_tables = {
                    "verses",
                    "original_language",
                    "verse_translations",
                    "annotations",
                    "annotation_topics",
                    "cross_references",
                    "timeline_events",
                    "timeline_periods",
                    "export_metadata",
                    "books",
                }

                actual_tables = {table[0] for table in tables}
                missing_tables = expected_tables - actual_tables

                if missing_tables:
                    validation.add_error(f"Missing tables: {missing_tables}")

                # Check data counts
                verse_count = conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
                if verse_count == 0:
                    validation.add_error("No verses found in database")
                elif verse_count != self.stats.processed_verses:
                    validation.add_warning(
                        f"Verse count mismatch: expected {self.stats.processed_verses}, "
                        f"found {verse_count}"
                    )

                # Check FTS5 if enabled
                if self.config.enable_fts5:
                    try:
                        fts_count = conn.execute("SELECT COUNT(*) FROM verses_fts").fetchone()[0]
                        if fts_count == 0:
                            validation.add_warning("FTS5 table is empty")
                    except sqlite3.OperationalError as e:
                        validation.add_error(f"FTS5 validation failed: {str(e)}")

        except Exception as e:
            validation.add_error(f"Database validation failed: {str(e)}")

        return validation
