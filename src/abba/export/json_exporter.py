"""
Static JSON exporter for ABBA canonical data.

Exports canonical biblical data to hierarchical JSON files optimized for
CDNs, static websites, and serverless architectures with progressive loading.
"""

import asyncio
import logging
import json
import gzip
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator, Set
from pathlib import Path
from collections import defaultdict

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
from ..book_codes import BOOK_INFO


@dataclass
class JSONConfig(ExportConfig):
    """Static JSON export configuration."""

    # File organization
    chunk_size: int = 100  # verses per chunk file
    create_book_indices: bool = True
    create_search_indices: bool = True

    # Progressive loading
    enable_progressive_loading: bool = True
    manifest_version: str = "1.0"

    # Compression
    gzip_output: bool = True
    minify_json: bool = True

    # Content organization
    split_large_books: bool = True
    large_book_threshold: int = 50  # chapters

    # Search optimization
    create_inverted_index: bool = True
    index_stemming: bool = True
    index_stopwords: bool = False

    def __post_init__(self):
        """Ensure JSON format type is set."""
        from .base import ExportFormat

        self.format_type = ExportFormat.STATIC_JSON
    
    def validate(self) -> ValidationResult:
        """Validate JSON-specific configuration settings."""
        # Start with base validation
        result = super().validate()
        errors = list(result.errors)
        warnings = list(result.warnings)
        
        # Add JSON-specific validations
        if self.chunk_size <= 0:
            errors.append("Chunk size must be positive")
            
        if self.large_book_threshold <= 0:
            errors.append("Large book threshold must be positive")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class StaticJSONExporter(DataExporter):
    """Exports canonical data to hierarchical static JSON files."""

    def __init__(self, config: JSONConfig):
        """Initialize JSON exporter."""
        super().__init__(config)
        self.config: JSONConfig = config
        self.output_dir = Path(config.output_path)
        self.processor = StreamingDataProcessor(batch_size=config.batch_size)

        # File structure
        self.api_dir = self.output_dir / "api" / "v1"
        self.meta_dir = self.api_dir / "meta"
        self.books_dir = self.api_dir / "books"
        self.search_dir = self.api_dir / "search"
        self.timeline_dir = self.api_dir / "timeline"

        # Data collectors for processing
        self.book_data = defaultdict(lambda: defaultdict(list))
        self.book_metadata = {}
        self.all_topics = set()
        self.search_index = defaultdict(lambda: defaultdict(set))
        self.timeline_data = {"events": [], "periods": []}

    def validate_config(self) -> ValidationResult:
        """Validate JSON exporter configuration."""
        validation = ValidationResult(is_valid=True)

        # Check output path
        if not self.config.output_path:
            validation.add_error("Output path is required")
            return validation

        # Validate chunk size
        if self.config.chunk_size <= 0:
            validation.add_error("Chunk size must be positive")

        # Validate large book threshold
        if self.config.large_book_threshold <= 0:
            validation.add_error("Large book threshold must be positive")

        return validation

    def get_supported_features(self) -> List[str]:
        """Get list of features supported by JSON exporter."""
        return [
            "progressive_loading",
            "cdn_optimized",
            "client_side_search",
            "hierarchical_navigation",
            "compression",
            "static_hosting",
            "offline_capable",
            "mobile_friendly",
        ]

    async def export(self, data: CanonicalDataset) -> ExportResult:
        """Export canonical data to static JSON files."""
        await self.prepare_export(data)

        try:
            # Initialize directory structure
            await self._initialize_directories()

            # Collect and organize data
            await self._collect_verses(data.verses)

            if data.annotations:
                await self._collect_annotations(data.annotations)

            if data.cross_references:
                await self._collect_cross_references(data.cross_references)

            if data.timeline_events:
                await self._collect_timeline_events(data.timeline_events)

            if data.timeline_periods:
                await self._collect_timeline_periods(data.timeline_periods)

            # Generate all JSON files
            await self._generate_metadata_files(data.metadata)
            await self._generate_book_files()
            await self._generate_search_files()
            await self._generate_timeline_files()
            await self._generate_manifest()

            result = self.create_result(ExportStatus.COMPLETED)

        except Exception as e:
            self.logger.error(f"JSON export failed: {str(e)}")
            result = self.create_result(
                ExportStatus.FAILED, ExportError(f"JSON export failed: {str(e)}", stage="export")
            )

        return await self.finalize_export(result)

    async def _initialize_directories(self):
        """Initialize directory structure."""
        self.logger.info(f"Initializing JSON directory structure: {self.output_dir}")

        # Create all directories
        for directory in [
            self.api_dir,
            self.meta_dir,
            self.books_dir,
            self.search_dir,
            self.timeline_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create book subdirectories
        for book_id in BOOK_INFO.keys():
            book_dir = self.books_dir / book_id
            book_dir.mkdir(exist_ok=True)

            chapters_dir = book_dir / "chapters"
            chapters_dir.mkdir(exist_ok=True)

    async def _collect_verses(self, verses: Iterator[TranslationVerse]):
        """Collect and organize verses by book and chapter."""
        self.logger.info("Collecting verses for JSON export")

        verse_count = 0

        for verse in verses:
            book_id = verse.verse_id.book
            chapter = verse.verse_id.chapter

            # Add to book data
            self.book_data[book_id][chapter].append(verse)

            # Update book metadata
            if book_id not in self.book_metadata:
                book_info = BOOK_INFO.get(book_id, {})
                self.book_metadata[book_id] = {
                    "id": book_id,
                    "name": book_info.get("name", book_id),
                    "testament": book_info.get("testament", "unknown").value if hasattr(book_info.get("testament", "unknown"), "value") else book_info.get("testament", "unknown"),
                    "order": book_info.get("order", 999),
                    "chapters": set(),
                    "verse_count": 0,
                    "translations": set(),
                }

            # Update metadata
            self.book_metadata[book_id]["chapters"].add(chapter)
            self.book_metadata[book_id]["verse_count"] += 1

            # Handle different verse types
            if hasattr(verse, 'text') and not hasattr(verse, 'translations'):
                # Simple TranslationVerse
                self.book_metadata[book_id]["translations"].add("default")
            elif hasattr(verse, 'translations') and verse.translations:
                self.book_metadata[book_id]["translations"].update(verse.translations.keys())

            # Build search index
            if self.config.create_search_indices:
                self._index_verse_for_search(verse)

            verse_count += 1
            if verse_count % 1000 == 0:
                self.update_progress(verse_count, "verses")

        # Convert sets to lists for JSON serialization
        for book_meta in self.book_metadata.values():
            book_meta["chapters"] = sorted(book_meta["chapters"])
            book_meta["translations"] = sorted(book_meta["translations"])
            book_meta["chapter_count"] = len(book_meta["chapters"])

        self.update_progress(verse_count, "verses")
        self.logger.info(f"Collected {verse_count} verses from {len(self.book_data)} books")

    async def _collect_annotations(self, annotations: Iterator[Annotation]):
        """Collect annotations organized by verse."""
        self.logger.info("Collecting annotations for JSON export")

        annotation_count = 0
        annotations_by_verse = defaultdict(list)

        for annotation in annotations:
            verse_id = str(annotation.start_verse)
            annotations_by_verse[verse_id].append(annotation)

            # Collect topics
            if annotation.topic_id:
                self.all_topics.add(annotation.topic_id)

            annotation_count += 1
            if annotation_count % 1000 == 0:
                self.update_progress(annotation_count, "annotations")

        # Attach annotations to verses
        for book_id, chapters in self.book_data.items():
            for chapter, verses in chapters.items():
                for verse in verses:
                    verse_id = str(verse.verse_id)
                    if verse_id in annotations_by_verse:
                        verse.annotations = annotations_by_verse[verse_id]

        self.update_progress(annotation_count, "annotations")
        self.logger.info(f"Collected {annotation_count} annotations")

    async def _collect_cross_references(self, cross_references: Iterator[Dict[str, Any]]):
        """Collect cross-references organized by verse."""
        self.logger.info("Collecting cross-references for JSON export")

        ref_count = 0
        refs_by_verse = defaultdict(list)

        for ref in cross_references:
            source_id = ref.get("source_verse_id")
            if source_id:
                refs_by_verse[source_id].append(ref)

            ref_count += 1
            if ref_count % 1000 == 0:
                self.update_progress(ref_count, "cross_references")

        # Attach cross-references to verses
        for book_id, chapters in self.book_data.items():
            for chapter, verses in chapters.items():
                for verse in verses:
                    verse_id = str(verse.verse_id)
                    if verse_id in refs_by_verse:
                        verse.cross_references = refs_by_verse[verse_id]

        self.update_progress(ref_count, "cross_references")
        self.logger.info(f"Collected {ref_count} cross-references")

    async def _collect_timeline_events(self, events: Iterator[Event]):
        """Collect timeline events."""
        self.logger.info("Collecting timeline events for JSON export")

        event_count = 0

        for event in events:
            self.timeline_data["events"].append(event.to_dict())
            event_count += 1

            if event_count % 100 == 0:
                self.update_progress(event_count, "events")

        self.update_progress(event_count, "events")
        self.logger.info(f"Collected {event_count} timeline events")

    async def _collect_timeline_periods(self, periods: Iterator[TimePeriod]):
        """Collect timeline periods."""
        self.logger.info("Collecting timeline periods for JSON export")

        period_count = 0

        for period in periods:
            self.timeline_data["periods"].append(period.to_dict())
            period_count += 1

        self.logger.info(f"Collected {period_count} timeline periods")

    def _index_verse_for_search(self, verse: TranslationVerse):
        """Add verse to search index."""
        verse_id = str(verse.verse_id)

        # Index by book
        self.search_index["books"][verse.verse_id.book].add(verse_id)

        # Index by text content
        verse_translations = None
        if hasattr(verse, 'text') and not hasattr(verse, 'translations'):
            # Simple TranslationVerse
            verse_translations = {"default": verse.text}
        elif hasattr(verse, 'translations'):
            verse_translations = verse.translations
            
        if verse_translations:
            for translation_id, text in verse_translations.items():
                words = self._extract_search_words(text)
                for word in words:
                    self.search_index["words"][word].add(verse_id)
                    self.search_index["translations"][translation_id].add(verse_id)

    def _extract_search_words(self, text: str) -> Set[str]:
        """Extract searchable words from text."""
        import re

        # Simple word extraction
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter stopwords if disabled
        if not self.config.index_stopwords:
            stopwords = {
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "shall",
                "must",
                "ought",
                "said",
                "says",
                "he",
                "she",
                "it",
                "they",
                "them",
                "their",
                "his",
                "her",
                "its",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "we",
                "us",
                "me",
                "him",
            }
            words = [w for w in words if w not in stopwords]

        # Apply stemming if enabled
        if self.config.index_stemming:
            words = [self._simple_stem(w) for w in words]

        return set(words)

    def _simple_stem(self, word: str) -> str:
        """Simple stemming algorithm."""
        # Very basic stemming - remove common suffixes
        suffixes = ["ing", "ed", "er", "est", "ly", "s"]

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]

        return word

    async def _generate_metadata_files(self, metadata: Dict[str, Any]):
        """Generate metadata files."""
        self.logger.info("Generating metadata files")

        # Books metadata
        books_meta = {
            "books": list(self.book_metadata.values()),
            "total_books": len(self.book_metadata),
            "total_verses": sum(book["verse_count"] for book in self.book_metadata.values()),
            "available_translations": sorted(
                set(trans for book in self.book_metadata.values() for trans in book["translations"])
            ),
        }
        await self._write_json_file(self.meta_dir / "books.json", books_meta)

        # Topics metadata
        if self.all_topics:
            topics_meta = {"topics": sorted(self.all_topics), "total_topics": len(self.all_topics)}
            await self._write_json_file(self.meta_dir / "topics.json", topics_meta)

        # Timeline metadata
        if self.timeline_data["events"] or self.timeline_data["periods"]:
            timeline_meta = {
                "total_events": len(self.timeline_data["events"]),
                "total_periods": len(self.timeline_data["periods"]),
                "event_types": list(
                    set(
                        event.get("event_type")
                        for event in self.timeline_data["events"]
                        if event.get("event_type")
                    )
                ),
                "date_range": self._calculate_timeline_date_range(),
            }
            await self._write_json_file(self.meta_dir / "timeline.json", timeline_meta)

        # Export metadata
        export_meta = {
            "format": "static_json",
            "version": self.config.manifest_version,
            "generated_at": self.stats.start_time.isoformat() if self.stats.start_time else None,
            "exporter": "ABBA Static JSON Exporter",
            "configuration": {
                "chunk_size": self.config.chunk_size,
                "compression": self.config.gzip_output,
                "progressive_loading": self.config.enable_progressive_loading,
            },
            "custom_metadata": metadata,
        }
        await self._write_json_file(self.meta_dir / "export.json", export_meta)

    async def _generate_book_files(self):
        """Generate book and chapter files."""
        self.logger.info("Generating book files")

        for book_id, chapters in self.book_data.items():
            book_meta = self.book_metadata[book_id]

            # Create book metadata file
            book_meta_file = {
                **book_meta,
                "chapters": [
                    {
                        "chapter": ch,
                        "verse_count": len(verses),
                        "file_path": f"chapters/{ch}/verses.json",
                    }
                    for ch, verses in sorted(chapters.items())
                ],
            }

            book_dir = self.books_dir / book_id
            await self._write_json_file(book_dir / "meta.json", book_meta_file)

            # Generate chapter files
            for chapter, verses in chapters.items():
                await self._generate_chapter_files(book_id, chapter, verses)

        self.logger.info(f"Generated files for {len(self.book_data)} books")

    async def _generate_chapter_files(self, book_id: str, chapter: int, verses: List[TranslationVerse]):
        """Generate files for a chapter."""
        chapter_dir = self.books_dir / book_id / "chapters" / str(chapter)
        chapter_dir.mkdir(parents=True, exist_ok=True)

        # Decide whether to chunk large chapters
        if len(verses) > self.config.chunk_size:
            await self._generate_chunked_chapter(chapter_dir, verses)
        else:
            await self._generate_single_chapter_file(chapter_dir, verses)

    async def _generate_single_chapter_file(self, chapter_dir: Path, verses: List[TranslationVerse]):
        """Generate single file for chapter."""
        chapter_data = {
            "verses": [self._serialize_verse(verse) for verse in verses],
            "verse_count": len(verses),
            "is_chunked": False,
        }

        await self._write_json_file(chapter_dir / "verses.json", chapter_data)

    async def _generate_chunked_chapter(self, chapter_dir: Path, verses: List[TranslationVerse]):
        """Generate chunked files for large chapter."""
        chunks = []

        # Split into chunks
        for i in range(0, len(verses), self.config.chunk_size):
            chunk_verses = verses[i : i + self.config.chunk_size]
            chunk_num = i // self.config.chunk_size + 1

            chunk_data = {
                "verses": [self._serialize_verse(verse) for verse in chunk_verses],
                "chunk_number": chunk_num,
                "verse_range": {
                    "start": chunk_verses[0].verse_id.verse,
                    "end": chunk_verses[-1].verse_id.verse,
                },
            }

            chunk_file = f"chunk_{chunk_num}.json"
            await self._write_json_file(chapter_dir / chunk_file, chunk_data)

            chunks.append(
                {
                    "chunk_number": chunk_num,
                    "file_path": chunk_file,
                    "verse_range": chunk_data["verse_range"],
                    "verse_count": len(chunk_verses),
                }
            )

        # Create chapter index
        chapter_index = {
            "is_chunked": True,
            "total_verses": len(verses),
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

        await self._write_json_file(chapter_dir / "verses.json", chapter_index)

    def _get_verse_translations(self, verse) -> Dict[str, str]:
        """Get translations from verse object."""
        if hasattr(verse, 'text') and not hasattr(verse, 'translations'):
            # Simple TranslationVerse
            return {"default": verse.text}
        elif hasattr(verse, 'translations'):
            return verse.translations or {}
        return {}
    
    def _serialize_verse(self, verse: TranslationVerse) -> Dict[str, Any]:
        """Serialize verse to JSON-compatible format."""
        verse_data = {
            "verse_id": str(verse.verse_id),
            "book": verse.verse_id.book,
            "chapter": verse.verse_id.chapter,
            "verse": verse.verse_id.verse,
            "translations": self._get_verse_translations(verse),
        }

        # Add original language data
        if hasattr(verse, 'hebrew_tokens') and verse.hebrew_tokens:
            verse_data["hebrew"] = verse.hebrew_tokens

        if hasattr(verse, 'greek_tokens') and verse.greek_tokens:
            verse_data["greek"] = verse.greek_tokens

        # Add annotations if present
        if hasattr(verse, "annotations") and verse.annotations:
            verse_data["annotations"] = [
                {
                    "id": ann.id,
                    "type": ann.annotation_type.value if ann.annotation_type else None,
                    "level": ann.level.value if ann.level else None,
                    "confidence": ann.confidence.overall_score if ann.confidence else None,
                    "topic": (
                        {"id": ann.topic_id, "name": ann.topic_name} if ann.topic_id else None
                    ),
                }
                for ann in verse.annotations
            ]

        # Add cross-references if present
        if hasattr(verse, "cross_references") and verse.cross_references:
            verse_data["cross_references"] = [
                {
                    "target": ref.get("target_verse_id"),
                    "type": ref.get("type", "reference"),
                    "confidence": ref.get("confidence", 1.0),
                }
                for ref in verse.cross_references
            ]

        # Add metadata if present
        if hasattr(verse, 'metadata') and verse.metadata:
            verse_data["metadata"] = verse.metadata

        return verse_data

    async def _generate_search_files(self):
        """Generate search index files."""
        if not self.config.create_search_indices:
            return

        self.logger.info("Generating search index files")

        # Convert defaultdict to regular dict and sets to lists for JSON
        search_data = {}
        for category, index in self.search_index.items():
            search_data[category] = {key: sorted(verse_ids) for key, verse_ids in index.items()}

        # Main search index
        await self._write_json_file(self.search_dir / "index.json", search_data)

        # Create separate index files for large categories
        if "words" in search_data and len(search_data["words"]) > 1000:
            # Split word index into alphabetical files
            word_index = search_data["words"]
            letters = {}

            for word, verse_ids in word_index.items():
                first_letter = word[0].upper() if word else "OTHER"
                if first_letter not in letters:
                    letters[first_letter] = {}
                letters[first_letter][word] = verse_ids

            # Write letter-based index files
            for letter, words in letters.items():
                await self._write_json_file(
                    self.search_dir / f"words_{letter}.json", {"letter": letter, "words": words}
                )

            # Create word index manifest
            word_manifest = {
                "index_type": "alphabetical",
                "available_letters": sorted(letters.keys()),
                "total_words": len(word_index),
            }
            await self._write_json_file(self.search_dir / "words_manifest.json", word_manifest)

    async def _generate_timeline_files(self):
        """Generate timeline files."""
        if not (self.timeline_data["events"] or self.timeline_data["periods"]):
            return

        self.logger.info("Generating timeline files")

        # Events file
        if self.timeline_data["events"]:
            events_data = {
                "events": self.timeline_data["events"],
                "total_events": len(self.timeline_data["events"]),
            }
            await self._write_json_file(self.timeline_dir / "events.json", events_data)

        # Periods file
        if self.timeline_data["periods"]:
            periods_data = {
                "periods": self.timeline_data["periods"],
                "total_periods": len(self.timeline_data["periods"]),
            }
            await self._write_json_file(self.timeline_dir / "periods.json", periods_data)

        # Combined timeline file
        timeline_combined = {
            "events": self.timeline_data["events"],
            "periods": self.timeline_data["periods"],
            "metadata": {
                "total_events": len(self.timeline_data["events"]),
                "total_periods": len(self.timeline_data["periods"]),
                "date_range": self._calculate_timeline_date_range(),
            },
        }
        await self._write_json_file(self.timeline_dir / "timeline.json", timeline_combined)

    def _calculate_timeline_date_range(self) -> Optional[Dict[str, Any]]:
        """Calculate date range for timeline data."""
        if not self.timeline_data["events"]:
            return None

        # Extract dates from events (simplified)
        dates = []
        for event in self.timeline_data["events"]:
            if event.get("time_point", {}).get("year"):
                dates.append(event["time_point"]["year"])

        if dates:
            return {
                "earliest_year": min(dates),
                "latest_year": max(dates),
                "span_years": max(dates) - min(dates),
            }

        return None

    async def _generate_manifest(self):
        """Generate progressive loading manifest."""
        if not self.config.enable_progressive_loading:
            return

        self.logger.info("Generating progressive loading manifest")

        # Collect all generated files
        files = []

        for file_path in self.output_dir.rglob("*.json*"):
            relative_path = file_path.relative_to(self.output_dir)
            file_info = {
                "path": str(relative_path),
                "size": file_path.stat().st_size,
                "compressed": file_path.suffix == ".gz",
            }

            # Categorize file
            if "meta" in str(relative_path):
                file_info["category"] = "metadata"
                file_info["priority"] = "high"
            elif "books" in str(relative_path):
                file_info["category"] = "content"
                file_info["priority"] = "medium"
            elif "search" in str(relative_path):
                file_info["category"] = "search"
                file_info["priority"] = "low"
            elif "timeline" in str(relative_path):
                file_info["category"] = "timeline"
                file_info["priority"] = "low"
            else:
                file_info["category"] = "other"
                file_info["priority"] = "low"

            files.append(file_info)

        # Create manifest
        manifest = {
            "format": "static_json",
            "version": self.config.manifest_version,
            "progressive_loading": True,
            "generated_at": self.stats.start_time.isoformat() if self.stats.start_time else None,
            "files": files,
            "statistics": {
                "total_files": len(files),
                "total_size": sum(f["size"] for f in files),
                "categories": {
                    category: len([f for f in files if f["category"] == category])
                    for category in set(f["category"] for f in files)
                },
            },
            "api_endpoints": {
                "books": "/api/v1/meta/books.json",
                "search": "/api/v1/search/index.json",
                "timeline": "/api/v1/timeline/timeline.json",
            },
        }

        await self._write_json_file(self.output_dir / "manifest.json", manifest)

    async def _write_json_file(self, file_path: Path, data: Any):
        """Write JSON data to file with optional compression."""
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize JSON
        json_str = json.dumps(
            data,
            ensure_ascii=False,
            indent=None if self.config.minify_json else 2,
            separators=(",", ":") if self.config.minify_json else None,
        )

        # Write file
        if self.config.gzip_output:
            # Write compressed file
            gz_path = file_path.with_suffix(file_path.suffix + ".gz")
            with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                f.write(json_str)
        else:
            # Write uncompressed file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)

    async def validate_output(self, result: ExportResult) -> ValidationResult:
        """Validate JSON export output."""
        validation = await super().validate_output(result)

        if not validation.is_valid:
            return validation

        try:
            # Check directory structure
            required_dirs = [self.api_dir, self.meta_dir, self.books_dir]
            for directory in required_dirs:
                if not directory.exists():
                    validation.add_error(f"Required directory missing: {directory}")

            # Check essential files
            essential_files = [self.meta_dir / "books.json", self.meta_dir / "export.json"]

            if self.config.gzip_output:
                essential_files = [f.with_suffix(f.suffix + ".gz") for f in essential_files]

            for file_path in essential_files:
                if not file_path.exists():
                    validation.add_error(f"Essential file missing: {file_path}")

            # Validate JSON syntax in sample files
            sample_files = list(self.output_dir.rglob("*.json"))[:5]  # Check first 5 files

            for file_path in sample_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    validation.add_error(f"Invalid JSON in {file_path}: {str(e)}")

            # Check manifest if progressive loading enabled
            if self.config.enable_progressive_loading:
                manifest_path = self.output_dir / "manifest.json"
                if self.config.gzip_output:
                    manifest_path = manifest_path.with_suffix(".json.gz")

                if not manifest_path.exists():
                    validation.add_error("Progressive loading manifest missing")

            # Check book files
            if not any(self.books_dir.iterdir()):
                validation.add_error("No book files generated")

            # Validate file counts
            json_files = list(self.output_dir.rglob("*.json*"))
            if len(json_files) == 0:
                validation.add_error("No JSON files generated")
            elif len(json_files) < len(self.book_metadata):
                validation.add_warning("Fewer JSON files than expected based on book count")

        except Exception as e:
            validation.add_error(f"JSON validation failed: {str(e)}")

        return validation
