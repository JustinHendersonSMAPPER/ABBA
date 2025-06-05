"""
OpenSearch exporter for ABBA canonical data.

Exports canonical biblical data to OpenSearch with optimized mappings,
custom analyzers, and search templates for large-scale applications.
"""

import asyncio
import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator, Union
from datetime import datetime
import aiohttp
import ssl

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
class OpenSearchConfig(ExportConfig):
    """OpenSearch export configuration."""

    # Connection settings
    cluster_url: str = "http://localhost:9200"
    username: Optional[str] = None
    password: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 30

    # Index settings
    index_prefix: str = "abba"
    index_version: str = "v1"
    create_aliases: bool = True

    # Bulk import settings
    bulk_size: int = 500
    max_retries: int = 3
    refresh_after_import: bool = True

    # Index optimization
    number_of_shards: int = 1
    number_of_replicas: int = 0
    refresh_interval: str = "30s"

    # Custom analyzers
    enable_biblical_analyzer: bool = True
    enable_multilingual_support: bool = True

    def __post_init__(self):
        """Ensure OpenSearch format type is set."""
        from .base import ExportFormat

        self.format_type = ExportFormat.OPENSEARCH
    
    def validate(self) -> ValidationResult:
        """Validate OpenSearch-specific configuration settings."""
        # Start with base validation
        result = super().validate()
        errors = list(result.errors)
        warnings = list(result.warnings)
        
        # Add OpenSearch-specific validations
        if not self.cluster_url:
            errors.append("OpenSearch cluster URL is required")
            
        if self.bulk_size <= 0:
            errors.append("Bulk size must be positive")
            
        if self.number_of_shards <= 0:
            errors.append("Number of shards must be positive")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class OpenSearchExporter(DataExporter):
    """Exports canonical data to OpenSearch cluster."""

    def __init__(self, config: OpenSearchConfig):
        """Initialize OpenSearch exporter."""
        super().__init__(config)
        self.config: OpenSearchConfig = config
        self.processor = StreamingDataProcessor(batch_size=config.bulk_size)

        # Index names
        self.verses_index = f"{config.index_prefix}_verses_{config.index_version}"
        self.annotations_index = f"{config.index_prefix}_annotations_{config.index_version}"
        self.events_index = f"{config.index_prefix}_events_{config.index_version}"
        self.periods_index = f"{config.index_prefix}_periods_{config.index_version}"

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # Mappings and settings
        self._analyzers = self._get_custom_analyzers()
        self._mappings = self._get_index_mappings()
        self._settings = self._get_index_settings()

    def validate_config(self) -> ValidationResult:
        """Validate OpenSearch exporter configuration."""
        validation = ValidationResult(is_valid=True)

        # Check cluster URL
        if not self.config.cluster_url:
            validation.add_error("OpenSearch cluster URL is required")

        # Validate bulk size
        if self.config.bulk_size <= 0:
            validation.add_error("Bulk size must be positive")
        elif self.config.bulk_size > 10000:
            validation.add_warning("Very large bulk size may cause memory issues")

        # Validate timeout
        if self.config.timeout <= 0:
            validation.add_error("Timeout must be positive")

        return validation

    def get_supported_features(self) -> List[str]:
        """Get list of features supported by OpenSearch exporter."""
        return [
            "full_text_search",
            "faceted_search",
            "aggregations",
            "real_time_search",
            "scalable_search",
            "multilingual_analysis",
            "biblical_text_analysis",
            "complex_queries",
            "analytics",
        ]

    async def export(self, data: CanonicalDataset) -> ExportResult:
        """Export canonical data to OpenSearch."""
        await self.prepare_export(data)

        try:
            # Initialize connection
            await self._initialize_connection()

            # Setup indices
            await self._setup_indices()

            # Export data
            await self._export_verses(data.verses)

            if data.annotations:
                await self._export_annotations(data.annotations)

            if data.timeline_events:
                await self._export_events(data.timeline_events)

            if data.timeline_periods:
                await self._export_periods(data.timeline_periods)

            # Finalize export
            await self._finalize_export(data.metadata)

            result = self.create_result(ExportStatus.COMPLETED)

        except Exception as e:
            self.logger.error(f"OpenSearch export failed: {str(e)}")
            result = self.create_result(
                ExportStatus.FAILED,
                ExportError(f"OpenSearch export failed: {str(e)}", stage="export"),
            )

        finally:
            await self._cleanup_connection()

        return await self.finalize_export(result)

    async def _initialize_connection(self):
        """Initialize HTTP connection to OpenSearch."""
        self.logger.info(f"Connecting to OpenSearch cluster: {self.config.cluster_url}")

        # Create SSL context
        ssl_context = ssl.create_default_context()
        if not self.config.verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        # Create authentication
        auth = None
        if self.config.username and self.config.password:
            auth = aiohttp.BasicAuth(self.config.username, self.config.password)

        # Create session
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            auth=auth,
            headers={"Content-Type": "application/json"},
        )

        # Test connection
        try:
            async with self.session.get(f"{self.config.cluster_url}/_cluster/health") as response:
                if response.status != 200:
                    raise ExportError(
                        f"OpenSearch cluster health check failed: {response.status}",
                        stage="connection",
                    )

                health = await response.json()
                self.logger.info(f"Connected to OpenSearch cluster: {health.get('cluster_name')}")

        except aiohttp.ClientError as e:
            raise ExportError(f"Failed to connect to OpenSearch: {str(e)}", stage="connection")

    async def _cleanup_connection(self):
        """Cleanup HTTP connection."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _setup_indices(self):
        """Setup OpenSearch indices with mappings and settings."""
        self.logger.info("Setting up OpenSearch indices")

        indices = [
            (self.verses_index, self._mappings["verses"]),
            (self.annotations_index, self._mappings["annotations"]),
            (self.events_index, self._mappings["events"]),
            (self.periods_index, self._mappings["periods"]),
        ]

        for index_name, mapping in indices:
            await self._create_index(index_name, mapping)

        # Create aliases if enabled
        if self.config.create_aliases:
            await self._create_aliases()

    async def _create_index(self, index_name: str, mapping: Dict[str, Any]):
        """Create individual index with mapping."""
        self.logger.info(f"Creating index: {index_name}")

        # Check if index exists
        async with self.session.head(f"{self.config.cluster_url}/{index_name}") as response:
            if response.status == 200:
                # Index exists, delete it
                self.logger.warning(f"Index {index_name} exists, deleting...")
                async with self.session.delete(
                    f"{self.config.cluster_url}/{index_name}"
                ) as del_response:
                    if del_response.status not in [200, 404]:
                        raise ExportError(f"Failed to delete existing index: {del_response.status}")

        # Create index with settings and mapping
        index_config = {"settings": self._settings, "mappings": mapping}

        async with self.session.put(
            f"{self.config.cluster_url}/{index_name}", data=json.dumps(index_config)
        ) as response:
            if response.status not in [200, 201]:
                error_text = await response.text()
                raise ExportError(
                    f"Failed to create index {index_name}: {response.status} - {error_text}",
                    stage="index_creation",
                )

        self.logger.info(f"Created index: {index_name}")

    async def _create_aliases(self):
        """Create index aliases for easier access."""
        aliases = [
            (self.verses_index, f"{self.config.index_prefix}_verses"),
            (self.annotations_index, f"{self.config.index_prefix}_annotations"),
            (self.events_index, f"{self.config.index_prefix}_events"),
            (self.periods_index, f"{self.config.index_prefix}_periods"),
        ]

        alias_actions = []
        for index_name, alias_name in aliases:
            alias_actions.append({"add": {"index": index_name, "alias": alias_name}})

        alias_request = {"actions": alias_actions}

        async with self.session.post(
            f"{self.config.cluster_url}/_aliases", data=json.dumps(alias_request)
        ) as response:
            if response.status != 200:
                self.logger.warning(f"Failed to create aliases: {response.status}")
            else:
                self.logger.info("Created index aliases")

    async def _export_verses(self, verses: Iterator[TranslationVerse]):
        """Export verses to OpenSearch."""
        self.logger.info("Exporting verses to OpenSearch")

        verse_count = 0

        async def process_verse_batch(verse_batch: List[TranslationVerse]):
            nonlocal verse_count

            # Prepare bulk request
            bulk_body = []

            for verse in verse_batch:
                # Index action
                index_action = {"index": {"_index": self.verses_index, "_id": str(verse.verse_id)}}
                bulk_body.append(json.dumps(index_action))

                # Document
                doc = self._serialize_verse_for_opensearch(verse)
                bulk_body.append(json.dumps(doc))

            # Send bulk request
            await self._send_bulk_request(bulk_body)

            verse_count += len(verse_batch)
            self.update_progress(verse_count, "verses")

        # Process verses in batches
        async for batch in self.processor.process_in_batches(
            verses, process_verse_batch, lambda count: self.update_progress(count, "verses")
        ):
            pass

        self.logger.info(f"Exported {verse_count} verses to OpenSearch")

    async def _export_annotations(self, annotations: Iterator[Annotation]):
        """Export annotations to OpenSearch."""
        self.logger.info("Exporting annotations to OpenSearch")

        annotation_count = 0

        async def process_annotation_batch(annotation_batch: List[Annotation]):
            nonlocal annotation_count

            bulk_body = []

            for annotation in annotation_batch:
                # Index action
                index_action = {"index": {"_index": self.annotations_index, "_id": annotation.id}}
                bulk_body.append(json.dumps(index_action))

                # Document
                doc = self._serialize_annotation_for_opensearch(annotation)
                bulk_body.append(json.dumps(doc))

            await self._send_bulk_request(bulk_body)

            annotation_count += len(annotation_batch)
            self.update_progress(annotation_count, "annotations")

        # Process annotations in batches
        async for batch in self.processor.process_in_batches(
            annotations,
            process_annotation_batch,
            lambda count: self.update_progress(count, "annotations"),
        ):
            pass

        self.logger.info(f"Exported {annotation_count} annotations to OpenSearch")

    async def _export_events(self, events: Iterator[Event]):
        """Export timeline events to OpenSearch."""
        self.logger.info("Exporting timeline events to OpenSearch")

        event_count = 0

        async def process_event_batch(event_batch: List[Event]):
            nonlocal event_count

            bulk_body = []

            for event in event_batch:
                # Index action
                index_action = {"index": {"_index": self.events_index, "_id": event.id}}
                bulk_body.append(json.dumps(index_action))

                # Document
                doc = self._serialize_event_for_opensearch(event)
                bulk_body.append(json.dumps(doc))

            await self._send_bulk_request(bulk_body)

            event_count += len(event_batch)
            self.update_progress(event_count, "events")

        # Process events in batches
        async for batch in self.processor.process_in_batches(
            events, process_event_batch, lambda count: self.update_progress(count, "events")
        ):
            pass

        self.logger.info(f"Exported {event_count} timeline events to OpenSearch")

    async def _export_periods(self, periods: Iterator[TimePeriod]):
        """Export timeline periods to OpenSearch."""
        self.logger.info("Exporting timeline periods to OpenSearch")

        period_count = 0

        async def process_period_batch(period_batch: List[TimePeriod]):
            nonlocal period_count

            bulk_body = []

            for period in period_batch:
                # Index action
                index_action = {"index": {"_index": self.periods_index, "_id": period.id}}
                bulk_body.append(json.dumps(index_action))

                # Document
                doc = self._serialize_period_for_opensearch(period)
                bulk_body.append(json.dumps(doc))

            await self._send_bulk_request(bulk_body)

            period_count += len(period_batch)

        # Process periods in batches
        async for batch in self.processor.process_in_batches(periods, process_period_batch):
            pass

        self.logger.info(f"Exported {period_count} timeline periods to OpenSearch")

    async def _send_bulk_request(self, bulk_body: List[str]):
        """Send bulk request to OpenSearch with retry logic."""
        bulk_data = "\n".join(bulk_body) + "\n"

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(
                    f"{self.config.cluster_url}/_bulk",
                    data=bulk_data,
                    headers={"Content-Type": "application/x-ndjson"},
                ) as response:

                    if response.status == 200:
                        result = await response.json()

                        # Check for errors in bulk response
                        if result.get("errors"):
                            error_items = [
                                item
                                for item in result.get("items", [])
                                if "error" in item.get("index", {})
                            ]

                            if error_items:
                                self.logger.warning(f"Bulk request had {len(error_items)} errors")
                                for error_item in error_items[:3]:  # Log first 3 errors
                                    error = error_item["index"]["error"]
                                    self.logger.warning(
                                        f"Bulk error: {error.get('type')} - {error.get('reason')}"
                                    )

                        return result

                    else:
                        error_text = await response.text()
                        if attempt < self.config.max_retries:
                            self.logger.warning(
                                f"Bulk request failed (attempt {attempt + 1}): {response.status} - {error_text}"
                            )
                            await asyncio.sleep(2**attempt)  # Exponential backoff
                            continue
                        else:
                            raise ExportError(
                                f"Bulk request failed after {self.config.max_retries} retries: {response.status}",
                                stage="bulk_import",
                            )

            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Network error (attempt {attempt + 1}): {str(e)}")
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    raise ExportError(
                        f"Network error after {self.config.max_retries} retries: {str(e)}",
                        stage="bulk_import",
                    )

        raise ExportError("Bulk request failed after all retries", stage="bulk_import")

    def _serialize_verse_for_opensearch(self, verse: TranslationVerse) -> Dict[str, Any]:
        """Serialize verse for OpenSearch indexing."""
        doc = {
            "verse_id": str(verse.verse_id),
            "book": verse.verse_id.book,
            "chapter": verse.verse_id.chapter,
            "verse": verse.verse_id.verse,
            "book_order": self._get_book_order(verse.verse_id.book),
            "indexed_at": datetime.utcnow().isoformat(),
        }

        # Add translations
        translations = {}
        if hasattr(verse, 'translations') and verse.translations:
            translations = verse.translations
        elif hasattr(verse, 'text'):
            # For simple TranslationVerse, create a basic translation entry
            translations = {"default": verse.text}
            
        if translations:
            doc["translations"] = translations

            # Create combined text for search
            all_text = " ".join(translations.values())
            doc["all_text"] = all_text

            # Add individual translation fields for faceting
            for trans_id, text in translations.items():
                doc[f"text_{trans_id}"] = text

        # Add original language data
        if hasattr(verse, 'hebrew_tokens') and verse.hebrew_tokens:
            doc["hebrew"] = {
                "tokens": verse.hebrew_tokens,
                "words": [token.get("word", "") for token in verse.hebrew_tokens],
                "lemmas": [token.get("lemma", "") for token in verse.hebrew_tokens],
                "strongs": [
                    token.get("strongs", "")
                    for token in verse.hebrew_tokens
                    if token.get("strongs")
                ],
            }

        if hasattr(verse, 'greek_tokens') and verse.greek_tokens:
            doc["greek"] = {
                "tokens": verse.greek_tokens,
                "words": [token.get("word", "") for token in verse.greek_tokens],
                "lemmas": [token.get("lemma", "") for token in verse.greek_tokens],
                "strongs": [
                    token.get("strongs", "") for token in verse.greek_tokens if token.get("strongs")
                ],
            }

        # Add metadata
        if hasattr(verse, 'metadata') and verse.metadata:
            doc["metadata"] = verse.metadata

        return doc

    def _serialize_annotation_for_opensearch(self, annotation: Annotation) -> Dict[str, Any]:
        """Serialize annotation for OpenSearch indexing."""
        doc = {
            "annotation_id": annotation.id,
            "verse_id": str(annotation.start_verse),
            "book": annotation.start_verse.book,
            "chapter": annotation.start_verse.chapter,
            "verse": annotation.start_verse.verse,
            "type": annotation.annotation_type.value if annotation.annotation_type else None,
            "level": annotation.level.value if annotation.level else None,
            "confidence": annotation.confidence.overall_score if annotation.confidence else 0.0,
            "indexed_at": datetime.utcnow().isoformat(),
        }

        # Add topics
        if annotation.topic_id:
            doc["topics"] = [
                {
                    "id": annotation.topic_id,
                    "name": annotation.topic_name or "",
                    "category": None,  # Not available in simple annotation
                }
            ]

            # Create searchable topic fields
            doc["topic_names"] = [annotation.topic_name] if annotation.topic_name else []
            doc["topic_ids"] = [annotation.topic_id]

        # Add metadata
        doc["metadata"] = {
            "source": annotation.source,
            "verified": annotation.verified,
            "created_date": annotation.created_date.isoformat() if annotation.created_date else None,
        }

        return doc

    def _serialize_event_for_opensearch(self, event: Event) -> Dict[str, Any]:
        """Serialize timeline event for OpenSearch indexing."""
        doc = {
            "event_id": event.id,
            "name": event.name,
            "description": event.description,
            "event_type": event.event_type.value,
            "certainty_level": event.certainty_level.value,
            "categories": event.categories,
            "indexed_at": datetime.utcnow().isoformat(),
        }

        # Add temporal data
        if event.time_point:
            time_data = event.time_point.to_dict()
            doc["time_point"] = time_data

            # Extract searchable date fields
            if event.time_point.exact_date:
                doc["exact_date"] = event.time_point.exact_date.isoformat()
                doc["year"] = event.time_point.exact_date.year
            elif event.time_point.earliest_date and event.time_point.latest_date:
                doc["date_range"] = {
                    "gte": event.time_point.earliest_date.isoformat(),
                    "lte": event.time_point.latest_date.isoformat(),
                }
                doc["earliest_year"] = event.time_point.earliest_date.year
                doc["latest_year"] = event.time_point.latest_date.year

        if event.time_range:
            doc["time_range"] = event.time_range.to_dict()

        # Add location
        if event.location:
            doc["location"] = {
                "name": event.location.name,
                "modern_name": event.location.modern_name,
                "region": event.location.region,
            }

            if event.location.latitude and event.location.longitude:
                doc["location"]["coordinates"] = {
                    "lat": event.location.latitude,
                    "lon": event.location.longitude,
                }

        # Add participants
        if event.participants:
            doc["participants"] = [
                {"id": p.id, "name": p.name, "type": p.entity_type, "role": p.role}
                for p in event.participants
            ]

            doc["participant_names"] = [p.name for p in event.participants]

        # Add verse references
        if event.verse_refs:
            doc["verse_refs"] = [str(v) for v in event.verse_refs]

        # Add scholars and sources for filtering
        doc["scholars"] = event.scholars
        doc["sources"] = event.sources
        doc["methodologies"] = event.methodologies
        doc["traditions"] = event.traditions

        return doc

    def _serialize_period_for_opensearch(self, period: TimePeriod) -> Dict[str, Any]:
        """Serialize timeline period for OpenSearch indexing."""
        doc = {
            "period_id": period.id,
            "name": period.name,
            "description": period.description,
            "parent_period": period.parent_period,
            "child_periods": period.child_periods,
            "events": period.events,
            "indexed_at": datetime.utcnow().isoformat(),
        }

        # Add time range
        if period.time_range:
            doc["time_range"] = period.time_range.to_dict()

        # Add characteristics
        if period.characteristics:
            doc["characteristics"] = period.characteristics

        if period.political_entity:
            doc["political_entity"] = period.political_entity

        return doc

    def _get_book_order(self, book_id: str) -> int:
        """Get canonical order for book (for sorting)."""
        from ..book_codes import get_book_order

        order = get_book_order(book_id)
        return order if order is not None else 999

    async def _finalize_export(self, metadata: Dict[str, Any]):
        """Finalize OpenSearch export."""
        self.logger.info("Finalizing OpenSearch export")

        # Refresh indices if requested
        if self.config.refresh_after_import:
            indices = [
                self.verses_index,
                self.annotations_index,
                self.events_index,
                self.periods_index,
            ]

            for index_name in indices:
                async with self.session.post(
                    f"{self.config.cluster_url}/{index_name}/_refresh"
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Failed to refresh index {index_name}")

        # Store export metadata
        export_meta = {
            "format": "opensearch",
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "exporter": "ABBA OpenSearch Exporter",
            "statistics": {
                "verses": self.stats.processed_verses,
                "annotations": self.stats.processed_annotations,
                "events": self.stats.processed_events,
            },
            "indices": {
                "verses": self.verses_index,
                "annotations": self.annotations_index,
                "events": self.events_index,
                "periods": self.periods_index,
            },
            "custom_metadata": metadata,
        }

        # Store in a metadata index
        meta_index = f"{self.config.index_prefix}_metadata_{self.config.index_version}"

        try:
            async with self.session.put(
                f"{self.config.cluster_url}/{meta_index}/_doc/export_info",
                data=json.dumps(export_meta),
            ) as response:
                if response.status not in [200, 201]:
                    self.logger.warning("Failed to store export metadata")

        except Exception as e:
            self.logger.warning(f"Failed to store export metadata: {str(e)}")

    def _get_index_settings(self) -> Dict[str, Any]:
        """Get index settings."""
        settings = {
            "number_of_shards": self.config.number_of_shards,
            "number_of_replicas": self.config.number_of_replicas,
            "refresh_interval": self.config.refresh_interval,
        }

        # Add custom analyzers if enabled
        if self.config.enable_biblical_analyzer:
            settings["analysis"] = self._analyzers

        return settings

    def _get_custom_analyzers(self) -> Dict[str, Any]:
        """Get custom analyzer definitions."""
        return {
            "analyzer": {
                "biblical_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "biblical_synonyms", "biblical_stemmer", "stop"],
                },
                "biblical_names": {
                    "type": "custom",
                    "tokenizer": "keyword",
                    "filter": ["lowercase", "trim"],
                },
                "original_language": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                },
            },
            "filter": {
                "biblical_synonyms": {
                    "type": "synonym",
                    "synonyms": [
                        "God,LORD,Yahweh,Jehovah",
                        "Christ,Jesus,Messiah",
                        "Holy Spirit,Spirit of God,Comforter",
                        "Israel,Jacob",
                        "Jerusalem,Zion,City of David",
                    ],
                },
                "biblical_stemmer": {"type": "stemmer", "language": "english"},
            },
        }

    def _get_index_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get index mapping definitions."""
        return {
            "verses": {
                "properties": {
                    "verse_id": {"type": "keyword"},
                    "book": {"type": "keyword"},
                    "chapter": {"type": "integer"},
                    "verse": {"type": "integer"},
                    "book_order": {"type": "integer"},
                    "translations": {"type": "object", "enabled": False},
                    "all_text": {
                        "type": "text",
                        "analyzer": (
                            "biblical_text" if self.config.enable_biblical_analyzer else "standard"
                        ),
                        "fields": {
                            "exact": {"type": "keyword"},
                            "ngram": {"type": "text", "analyzer": "ngram_analyzer"},
                        },
                    },
                    "hebrew": {
                        "properties": {
                            "words": {"type": "text", "analyzer": "original_language"},
                            "lemmas": {"type": "keyword"},
                            "strongs": {"type": "keyword"},
                        }
                    },
                    "greek": {
                        "properties": {
                            "words": {"type": "text", "analyzer": "original_language"},
                            "lemmas": {"type": "keyword"},
                            "strongs": {"type": "keyword"},
                        }
                    },
                    "indexed_at": {"type": "date"},
                },
                "dynamic_templates": [
                    {
                        "text_translations": {
                            "path_match": "text_*",
                            "mapping": {
                                "type": "text",
                                "analyzer": (
                                    "biblical_text"
                                    if self.config.enable_biblical_analyzer
                                    else "standard"
                                ),
                            },
                        }
                    }
                ],
            },
            "annotations": {
                "properties": {
                    "annotation_id": {"type": "keyword"},
                    "verse_id": {"type": "keyword"},
                    "book": {"type": "keyword"},
                    "chapter": {"type": "integer"},
                    "verse": {"type": "integer"},
                    "type": {"type": "keyword"},
                    "level": {"type": "keyword"},
                    "confidence": {"type": "float"},
                    "topics": {
                        "type": "nested",
                        "properties": {
                            "id": {"type": "keyword"},
                            "name": {"type": "text", "analyzer": "biblical_text"},
                            "category": {"type": "keyword"},
                        },
                    },
                    "topic_names": {"type": "text", "analyzer": "biblical_text"},
                    "topic_ids": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                }
            },
            "events": {
                "properties": {
                    "event_id": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "biblical_text"},
                    "description": {"type": "text", "analyzer": "biblical_text"},
                    "event_type": {"type": "keyword"},
                    "certainty_level": {"type": "keyword"},
                    "categories": {"type": "keyword"},
                    "exact_date": {"type": "date"},
                    "year": {"type": "integer"},
                    "date_range": {"type": "date_range"},
                    "earliest_year": {"type": "integer"},
                    "latest_year": {"type": "integer"},
                    "location": {
                        "properties": {
                            "name": {"type": "text", "analyzer": "biblical_names"},
                            "modern_name": {"type": "text"},
                            "region": {"type": "keyword"},
                            "coordinates": {"type": "geo_point"},
                        }
                    },
                    "participants": {
                        "type": "nested",
                        "properties": {
                            "id": {"type": "keyword"},
                            "name": {"type": "text", "analyzer": "biblical_names"},
                            "type": {"type": "keyword"},
                            "role": {"type": "keyword"},
                        },
                    },
                    "participant_names": {"type": "text", "analyzer": "biblical_names"},
                    "verse_refs": {"type": "keyword"},
                    "scholars": {"type": "keyword"},
                    "sources": {"type": "keyword"},
                    "methodologies": {"type": "keyword"},
                    "traditions": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                }
            },
            "periods": {
                "properties": {
                    "period_id": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "biblical_text"},
                    "description": {"type": "text", "analyzer": "biblical_text"},
                    "parent_period": {"type": "keyword"},
                    "child_periods": {"type": "keyword"},
                    "events": {"type": "keyword"},
                    "political_entity": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                }
            },
        }

    async def validate_output(self, result: ExportResult) -> ValidationResult:
        """Validate OpenSearch export output."""
        validation = ValidationResult(is_valid=True)

        if not self.session:
            validation.add_error("No OpenSearch connection available for validation")
            return validation

        try:
            # Check cluster health
            async with self.session.get(f"{self.config.cluster_url}/_cluster/health") as response:
                if response.status != 200:
                    validation.add_error(f"Cluster health check failed: {response.status}")
                    return validation

            # Check indices existence and document counts
            indices = [
                (self.verses_index, self.stats.processed_verses),
                (self.annotations_index, self.stats.processed_annotations),
                (self.events_index, self.stats.processed_events),
            ]

            for index_name, expected_count in indices:
                # Check index exists
                async with self.session.head(f"{self.config.cluster_url}/{index_name}") as response:
                    if response.status != 200:
                        validation.add_error(f"Index {index_name} does not exist")
                        continue

                # Check document count
                async with self.session.get(
                    f"{self.config.cluster_url}/{index_name}/_count"
                ) as response:
                    if response.status == 200:
                        count_result = await response.json()
                        actual_count = count_result.get("count", 0)

                        if actual_count == 0 and expected_count > 0:
                            validation.add_error(f"Index {index_name} is empty")
                        elif actual_count != expected_count:
                            validation.add_warning(
                                f"Document count mismatch in {index_name}: "
                                f"expected {expected_count}, found {actual_count}"
                            )
                    else:
                        validation.add_warning(f"Could not verify document count for {index_name}")

            # Test search functionality
            test_query = {"query": {"match_all": {}}, "size": 1}

            async with self.session.post(
                f"{self.config.cluster_url}/{self.verses_index}/_search",
                data=json.dumps(test_query),
            ) as response:
                if response.status != 200:
                    validation.add_error("Search test failed")
                else:
                    search_result = await response.json()
                    if search_result.get("hits", {}).get("total", {}).get("value", 0) == 0:
                        validation.add_warning("Search returned no results")

        except Exception as e:
            validation.add_error(f"OpenSearch validation failed: {str(e)}")

        return validation
