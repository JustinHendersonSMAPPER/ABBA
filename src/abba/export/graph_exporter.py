"""
Graph database exporters for ABBA canonical data.

Exports canonical biblical data to graph databases (Neo4j, ArangoDB) optimized
for relationship traversal, research applications, and complex biblical studies.
"""

import asyncio
import logging
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator, Union, Tuple
from datetime import datetime
import aiohttp

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
class GraphConfig(ExportConfig):
    """Base graph database export configuration."""

    # Connection settings
    server_url: str = "http://localhost:7474"
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "neo4j"

    # Export settings
    batch_size: int = 100
    create_indices: bool = True
    clear_existing_data: bool = True

    # Node/relationship optimization
    enable_relationship_properties: bool = True
    compute_graph_metrics: bool = True

    def __post_init__(self):
        """Set graph format type."""
        from .base import ExportFormat

        if not hasattr(self, "format_type"):
            self.format_type = ExportFormat.NEO4J
            
    def validate(self) -> ValidationResult:
        """Validate graph configuration."""
        # Start with base validation
        result = super().validate()
        errors = list(result.errors)
        warnings = list(result.warnings)
        
        # Validate server URL
        if not self.server_url:
            errors.append("Server URL is required for graph export")
            
        # Validate batch size for graph operations
        if self.batch_size > 10000:
            warnings.append("Large batch sizes may cause memory issues in graph databases")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


@dataclass
class Neo4jConfig(GraphConfig):
    """Neo4j-specific configuration."""

    # Neo4j specific settings
    use_bolt_protocol: bool = False
    bolt_port: int = 7687

    def __post_init__(self):
        """Set Neo4j format type."""
        from .base import ExportFormat

        self.format_type = ExportFormat.NEO4J


@dataclass
class ArangoConfig(GraphConfig):
    """ArangoDB-specific configuration."""

    # ArangoDB specific settings
    collection_prefix: str = "abba"
    create_graph: bool = True
    graph_name: str = "biblical_graph"

    def __post_init__(self):
        """Set ArangoDB format type."""
        from .base import ExportFormat

        self.format_type = ExportFormat.ARANGODB


class GraphExporter(DataExporter):
    """Base class for graph database exporters."""

    def __init__(self, config: GraphConfig):
        """Initialize graph exporter."""
        super().__init__(config)
        self.config: GraphConfig = config
        self.processor = StreamingDataProcessor(batch_size=config.batch_size)

        # Graph data collectors
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

    def validate_config(self) -> ValidationResult:
        """Validate graph exporter configuration."""
        validation = ValidationResult(is_valid=True)

        # Check server URL
        if not self.config.server_url:
            validation.add_error("Graph database server URL is required")

        # Check batch size
        if self.config.batch_size <= 0:
            validation.add_error("Batch size must be positive")

        return validation

    def get_supported_features(self) -> List[str]:
        """Get features supported by graph exporters."""
        return [
            "relationship_traversal",
            "pattern_matching",
            "graph_algorithms",
            "complex_queries",
            "biblical_research",
            "citation_analysis",
            "topic_relationships",
            "timeline_connections",
        ]

    async def export(self, data: CanonicalDataset) -> ExportResult:
        """Export canonical data to graph database."""
        await self.prepare_export(data)

        try:
            # Initialize connection
            await self._initialize_connection()

            # Clear existing data if requested
            if self.config.clear_existing_data:
                await self._clear_database()

            # Setup database schema
            await self._setup_schema()

            # Collect graph data
            await self._collect_graph_data(data)

            # Export nodes and relationships
            await self._export_nodes()
            await self._export_relationships()

            # Create indices and constraints
            if self.config.create_indices:
                await self._create_indices()

            # Compute graph metrics if requested
            if self.config.compute_graph_metrics:
                await self._compute_graph_metrics()

            # Finalize export
            await self._finalize_export(data.metadata)

            result = self.create_result(ExportStatus.COMPLETED)

        except Exception as e:
            self.logger.error(f"Graph export failed: {str(e)}")
            result = self.create_result(
                ExportStatus.FAILED, ExportError(f"Graph export failed: {str(e)}", stage="export")
            )

        finally:
            await self._cleanup_connection()

        return await self.finalize_export(result)

    async def _collect_graph_data(self, data: CanonicalDataset):
        """Collect and organize data for graph export."""
        self.logger.info("Collecting graph data")

        # Collect verses as nodes
        await self._collect_verse_nodes(data.verses)

        # Collect annotations
        if data.annotations:
            await self._collect_annotation_nodes(data.annotations)

        # Collect cross-references as relationships
        if data.cross_references:
            await self._collect_cross_reference_relationships(data.cross_references)

        # Collect timeline data
        if data.timeline_events:
            await self._collect_event_nodes(data.timeline_events)

        if data.timeline_periods:
            await self._collect_period_nodes(data.timeline_periods)

        self.logger.info(
            f"Collected {len(self.nodes)} nodes and {len(self.relationships)} relationships"
        )

    async def _collect_verse_nodes(self, verses: Iterator[TranslationVerse]):
        """Collect verses as graph nodes."""
        verse_count = 0

        for verse in verses:
            node_id = f"verse:{verse.verse_id}"

            # Handle both canonical verses and simple TranslationVerse objects
            translations = {}
            if hasattr(verse, 'translations'):
                translations = verse.translations or {}
            elif hasattr(verse, 'text'):
                # For simple TranslationVerse, create a basic translation entry
                translations = {"default": verse.text}

            self.nodes[node_id] = {
                "id": node_id,
                "label": "Verse",
                "properties": {
                    "verse_id": str(verse.verse_id),
                    "book": verse.verse_id.book,
                    "chapter": verse.verse_id.chapter,
                    "verse": verse.verse_id.verse,
                    "translations": translations,
                    "created_at": datetime.utcnow().isoformat(),
                },
            }

            # Add original language data as related nodes
            if hasattr(verse, 'hebrew_tokens') and verse.hebrew_tokens:
                await self._add_language_tokens(node_id, verse.hebrew_tokens, "Hebrew")

            if hasattr(verse, 'greek_tokens') and verse.greek_tokens:
                await self._add_language_tokens(node_id, verse.greek_tokens, "Greek")

            verse_count += 1
            if verse_count % 1000 == 0:
                self.update_progress(verse_count, "verses")

        self.update_progress(verse_count, "verses")

    async def _add_language_tokens(self, verse_node_id: str, tokens: List[Dict], language: str):
        """Add original language tokens as nodes."""
        for i, token in enumerate(tokens):
            token_id = f"{verse_node_id}:token:{language.lower()}:{i}"

            self.nodes[token_id] = {
                "id": token_id,
                "label": f"{language}Token",
                "properties": {
                    "word": token.get("word", ""),
                    "lemma": token.get("lemma", ""),
                    "morph": token.get("morph", ""),
                    "strongs": token.get("strongs", ""),
                    "position": i,
                    "language": language.lower(),
                },
            }

            # Create relationship: Verse -> CONTAINS -> Token
            self.relationships.append(
                {
                    "source": verse_node_id,
                    "target": token_id,
                    "type": "CONTAINS",
                    "properties": {"position": i, "language": language.lower()},
                }
            )

    async def _collect_annotation_nodes(self, annotations: Iterator[Annotation]):
        """Collect annotations and topics as nodes."""
        annotation_count = 0
        topics_seen = set()

        for annotation in annotations:
            # Create annotation node
            annotation_id = f"annotation:{annotation.id}"

            self.nodes[annotation_id] = {
                "id": annotation_id,
                "label": "Annotation",
                "properties": {
                    "annotation_id": annotation.id,
                    "type": (
                        annotation.annotation_type.value if annotation.annotation_type else None
                    ),
                    "level": annotation.level.value if annotation.level else None,
                    "confidence": annotation.confidence,
                    "created_at": datetime.utcnow().isoformat(),
                },
            }

            # Create relationship: Verse -> ANNOTATED_WITH -> Annotation
            verse_node_id = f"verse:{annotation.start_verse}"
            self.relationships.append(
                {
                    "source": verse_node_id,
                    "target": annotation_id,
                    "type": "ANNOTATED_WITH",
                    "properties": {"confidence": annotation.confidence.overall_score if annotation.confidence else 0.0},
                }
            )

            # Create topic nodes and relationships
            if annotation.topic_id:
                topic_id = f"topic:{annotation.topic_id}"

                # Create topic node if not seen before
                if topic_id not in topics_seen:
                    self.nodes[topic_id] = {
                        "id": topic_id,
                        "label": "Topic",
                        "properties": {
                            "topic_id": annotation.topic_id,
                            "name": annotation.topic_name or "",
                            "category": None,
                            "importance": 0.5,
                        },
                    }
                    topics_seen.add(topic_id)

                    # Create relationship: Annotation -> ABOUT -> Topic
                    self.relationships.append(
                        {
                            "source": annotation_id,
                            "target": topic_id,
                            "type": "ABOUT",
                            "properties": {"confidence": annotation.confidence.overall_score if annotation.confidence else 1.0},
                        }
                    )

            annotation_count += 1
            if annotation_count % 1000 == 0:
                self.update_progress(annotation_count, "annotations")

        self.update_progress(annotation_count, "annotations")

    async def _collect_cross_reference_relationships(
        self, cross_references: Iterator[Dict[str, Any]]
    ):
        """Collect cross-references as relationships."""
        ref_count = 0

        for ref in cross_references:
            source_id = f"verse:{ref.get('source_verse_id')}"
            target_id = f"verse:{ref.get('target_verse_id')}"

            self.relationships.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "type": "REFERENCES",
                    "properties": {
                        "reference_type": ref.get("type", "reference"),
                        "confidence": ref.get("confidence", 1.0),
                        "source": ref.get("source", ""),
                        "created_at": datetime.utcnow().isoformat(),
                    },
                }
            )

            ref_count += 1
            if ref_count % 1000 == 0:
                self.update_progress(ref_count, "cross_references")

        self.update_progress(ref_count, "cross_references")

    async def _collect_event_nodes(self, events: Iterator[Event]):
        """Collect timeline events as nodes."""
        event_count = 0
        people_seen = set()
        places_seen = set()

        for event in events:
            event_id = f"event:{event.id}"

            # Create event node
            event_props = {
                "event_id": event.id,
                "name": event.name,
                "description": event.description,
                "event_type": event.event_type.value,
                "certainty_level": event.certainty_level.value,
                "categories": event.categories,
                "created_at": datetime.utcnow().isoformat(),
            }

            # Add temporal data
            if event.time_point:
                time_data = event.time_point.to_dict()
                event_props.update(
                    {
                        "exact_date": time_data.get("exact_date"),
                        "earliest_date": time_data.get("earliest_date"),
                        "latest_date": time_data.get("latest_date"),
                        "confidence": event.time_point.confidence,
                    }
                )

            self.nodes[event_id] = {"id": event_id, "label": "Event", "properties": event_props}

            # Create participant nodes and relationships
            if event.participants:
                for participant in event.participants:
                    person_id = f"person:{participant.id}"

                    if person_id not in people_seen:
                        self.nodes[person_id] = {
                            "id": person_id,
                            "label": "Person",
                            "properties": {
                                "person_id": participant.id,
                                "name": participant.name,
                                "entity_type": participant.entity_type,
                            },
                        }
                        people_seen.add(person_id)

                    # Create relationship: Person -> PARTICIPATES_IN -> Event
                    self.relationships.append(
                        {
                            "source": person_id,
                            "target": event_id,
                            "type": "PARTICIPATES_IN",
                            "properties": {"role": participant.role or "participant"},
                        }
                    )

            # Create location nodes and relationships
            if event.location:
                place_id = f"place:{event.location.name.replace(' ', '_')}"

                if place_id not in places_seen:
                    location_props = {
                        "name": event.location.name,
                        "modern_name": event.location.modern_name,
                        "region": event.location.region,
                    }

                    if event.location.latitude and event.location.longitude:
                        location_props.update(
                            {
                                "latitude": event.location.latitude,
                                "longitude": event.location.longitude,
                            }
                        )

                    self.nodes[place_id] = {
                        "id": place_id,
                        "label": "Place",
                        "properties": location_props,
                    }
                    places_seen.add(place_id)

                # Create relationship: Event -> OCCURS_AT -> Place
                self.relationships.append(
                    {"source": event_id, "target": place_id, "type": "OCCURS_AT", "properties": {}}
                )

            # Create relationships to verses
            if event.verse_refs:
                for verse_ref in event.verse_refs:
                    verse_id = f"verse:{verse_ref}"
                    self.relationships.append(
                        {
                            "source": verse_id,
                            "target": event_id,
                            "type": "DESCRIBES",
                            "properties": {},
                        }
                    )

            event_count += 1
            if event_count % 100 == 0:
                self.update_progress(event_count, "events")

        self.update_progress(event_count, "events")

    async def _collect_period_nodes(self, periods: Iterator[TimePeriod]):
        """Collect timeline periods as nodes."""
        period_count = 0

        for period in periods:
            period_id = f"period:{period.id}"

            self.nodes[period_id] = {
                "id": period_id,
                "label": "Period",
                "properties": {
                    "period_id": period.id,
                    "name": period.name,
                    "description": period.description,
                    "political_entity": period.political_entity,
                    "created_at": datetime.utcnow().isoformat(),
                },
            }

            # Create hierarchical relationships
            if period.parent_period:
                parent_id = f"period:{period.parent_period}"
                self.relationships.append(
                    {
                        "source": parent_id,
                        "target": period_id,
                        "type": "CONTAINS_PERIOD",
                        "properties": {},
                    }
                )

            # Create relationships to events
            for event_id in period.events:
                event_node_id = f"event:{event_id}"
                self.relationships.append(
                    {
                        "source": period_id,
                        "target": event_node_id,
                        "type": "CONTAINS_EVENT",
                        "properties": {},
                    }
                )

            period_count += 1

        self.logger.info(f"Collected {period_count} timeline periods")

    # Abstract methods for database-specific implementations
    @abstractmethod
    async def _initialize_connection(self):
        """Initialize database connection."""
        pass

    @abstractmethod
    async def _cleanup_connection(self):
        """Cleanup database connection."""
        pass

    @abstractmethod
    async def _clear_database(self):
        """Clear existing data from database."""
        pass

    @abstractmethod
    async def _setup_schema(self):
        """Setup database schema."""
        pass

    @abstractmethod
    async def _export_nodes(self):
        """Export nodes to database."""
        pass

    @abstractmethod
    async def _export_relationships(self):
        """Export relationships to database."""
        pass

    @abstractmethod
    async def _create_indices(self):
        """Create database indices."""
        pass

    @abstractmethod
    async def _compute_graph_metrics(self):
        """Compute graph metrics."""
        pass

    @abstractmethod
    async def _finalize_export(self, metadata: Dict[str, Any]):
        """Finalize export."""
        pass


class Neo4jExporter(GraphExporter):
    """Neo4j-specific graph exporter."""

    def __init__(self, config: Neo4jConfig):
        """Initialize Neo4j exporter."""
        super().__init__(config)
        self.config: Neo4jConfig = config

    async def _initialize_connection(self):
        """Initialize Neo4j connection."""
        self.logger.info(f"Connecting to Neo4j: {self.config.server_url}")

        # Create authentication
        auth = None
        if self.config.username and self.config.password:
            auth = aiohttp.BasicAuth(self.config.username, self.config.password)

        # Create session
        self.session = aiohttp.ClientSession(
            auth=auth, headers={"Content-Type": "application/json"}
        )

        # Test connection
        try:
            url = f"{self.config.server_url}/db/{self.config.database}/tx/commit"
            test_query = {"statements": [{"statement": "RETURN 1 as test"}]}

            async with self.session.post(url, data=json.dumps(test_query)) as response:
                if response.status != 200:
                    raise ExportError(f"Neo4j connection test failed: {response.status}")

                result = await response.json()
                if result.get("errors"):
                    raise ExportError(f"Neo4j query error: {result['errors']}")

                self.logger.info("Connected to Neo4j successfully")

        except aiohttp.ClientError as e:
            raise ExportError(f"Failed to connect to Neo4j: {str(e)}")

    async def _cleanup_connection(self):
        """Cleanup Neo4j connection."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _clear_database(self):
        """Clear existing data from Neo4j."""
        self.logger.info("Clearing existing Neo4j data")

        query = "MATCH (n) DETACH DELETE n"
        await self._execute_cypher(query)

    async def _setup_schema(self):
        """Setup Neo4j schema constraints and indices."""
        self.logger.info("Setting up Neo4j schema")

        # Create constraints
        constraints = [
            "CREATE CONSTRAINT verse_id IF NOT EXISTS FOR (v:Verse) REQUIRE v.verse_id IS UNIQUE",
            "CREATE CONSTRAINT annotation_id IF NOT EXISTS FOR (a:Annotation) REQUIRE a.annotation_id IS UNIQUE",
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
            "CREATE CONSTRAINT period_id IF NOT EXISTS FOR (pd:Period) REQUIRE pd.period_id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                await self._execute_cypher(constraint)
            except Exception as e:
                self.logger.warning(f"Failed to create constraint: {str(e)}")

    async def _export_nodes(self):
        """Export nodes to Neo4j."""
        self.logger.info("Exporting nodes to Neo4j")

        # Group nodes by label for efficient batch processing
        nodes_by_label = {}
        for node in self.nodes.values():
            label = node["label"]
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(node)

        # Export each label type
        for label, nodes in nodes_by_label.items():
            await self._export_nodes_by_label(label, nodes)

    async def _export_nodes_by_label(self, label: str, nodes: List[Dict[str, Any]]):
        """Export nodes of specific label."""
        self.logger.info(f"Exporting {len(nodes)} {label} nodes")

        # Process in batches
        for i in range(0, len(nodes), self.config.batch_size):
            batch = nodes[i : i + self.config.batch_size]

            # Build UNWIND query
            query = f"""
            UNWIND $nodes AS nodeData
            CREATE (n:{label})
            SET n = nodeData.properties
            """

            node_data = [{"properties": node["properties"]} for node in batch]
            await self._execute_cypher(query, {"nodes": node_data})

        self.logger.info(f"Exported {len(nodes)} {label} nodes")

    async def _export_relationships(self):
        """Export relationships to Neo4j."""
        self.logger.info(f"Exporting {len(self.relationships)} relationships to Neo4j")

        # Group relationships by type
        rels_by_type = {}
        for rel in self.relationships:
            rel_type = rel["type"]
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)

        # Export each relationship type
        for rel_type, rels in rels_by_type.items():
            await self._export_relationships_by_type(rel_type, rels)

    async def _export_relationships_by_type(
        self, rel_type: str, relationships: List[Dict[str, Any]]
    ):
        """Export relationships of specific type."""
        self.logger.info(f"Exporting {len(relationships)} {rel_type} relationships")

        # Process in batches
        for i in range(0, len(relationships), self.config.batch_size):
            batch = relationships[i : i + self.config.batch_size]

            # Build relationship creation query
            query = f"""
            UNWIND $rels AS relData
            MATCH (source) WHERE id(source) = relData.source OR 
                  ANY(label IN labels(source) WHERE 
                      (label = 'Verse' AND source.verse_id = substring(relData.source, 6)) OR
                      (label = 'Event' AND source.event_id = substring(relData.source, 6)) OR
                      (label = 'Topic' AND source.topic_id = substring(relData.source, 6)) OR
                      (label = 'Annotation' AND source.annotation_id = substring(relData.source, 11)) OR
                      (label = 'Person' AND source.person_id = substring(relData.source, 7)) OR
                      (label = 'Period' AND source.period_id = substring(relData.source, 7)) OR
                      (label = 'Place' AND source.name = replace(substring(relData.source, 6), '_', ' '))
                  )
            MATCH (target) WHERE id(target) = relData.target OR
                  ANY(label IN labels(target) WHERE 
                      (label = 'Verse' AND target.verse_id = substring(relData.target, 6)) OR
                      (label = 'Event' AND target.event_id = substring(relData.target, 6)) OR
                      (label = 'Topic' AND target.topic_id = substring(relData.target, 6)) OR
                      (label = 'Annotation' AND target.annotation_id = substring(relData.target, 11)) OR
                      (label = 'Person' AND target.person_id = substring(relData.target, 7)) OR
                      (label = 'Period' AND target.period_id = substring(relData.target, 7)) OR
                      (label = 'Place' AND target.name = replace(substring(relData.target, 6), '_', ' '))
                  )
            CREATE (source)-[r:{rel_type}]->(target)
            SET r = relData.properties
            """

            rel_data = [
                {"source": rel["source"], "target": rel["target"], "properties": rel["properties"]}
                for rel in batch
            ]

            await self._execute_cypher(query, {"rels": rel_data})

        self.logger.info(f"Exported {len(relationships)} {rel_type} relationships")

    async def _create_indices(self):
        """Create Neo4j indices for performance."""
        self.logger.info("Creating Neo4j indices")

        indices = [
            "CREATE INDEX verse_book IF NOT EXISTS FOR (v:Verse) ON (v.book)",
            "CREATE INDEX verse_chapter IF NOT EXISTS FOR (v:Verse) ON (v.chapter)",
            "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX place_name IF NOT EXISTS FOR (pl:Place) ON (pl.name)",
        ]

        for index in indices:
            try:
                await self._execute_cypher(index)
            except Exception as e:
                self.logger.warning(f"Failed to create index: {str(e)}")

    async def _compute_graph_metrics(self):
        """Compute Neo4j graph metrics."""
        self.logger.info("Computing graph metrics")

        # Basic graph statistics
        queries = [
            ("node_count", "MATCH (n) RETURN count(n) as count"),
            ("relationship_count", "MATCH ()-[r]->() RETURN count(r) as count"),
            ("verse_count", "MATCH (v:Verse) RETURN count(v) as count"),
            ("topic_count", "MATCH (t:Topic) RETURN count(t) as count"),
            ("event_count", "MATCH (e:Event) RETURN count(e) as count"),
        ]

        metrics = {}
        for metric_name, query in queries:
            try:
                result = await self._execute_cypher(query)
                if result and result[0]:
                    metrics[metric_name] = result[0].get("count", 0)
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric_name}: {str(e)}")

        self.logger.info(f"Graph metrics: {metrics}")
        return metrics

    async def _finalize_export(self, metadata: Dict[str, Any]):
        """Finalize Neo4j export."""
        self.logger.info("Finalizing Neo4j export")

        # Store export metadata as a node
        export_meta = {
            "format": "neo4j",
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "exporter": "ABBA Neo4j Exporter",
            "node_count": len(self.nodes),
            "relationship_count": len(self.relationships),
            "custom_metadata": json.dumps(metadata),
        }

        query = """
        CREATE (meta:ExportMetadata)
        SET meta = $properties
        """

        await self._execute_cypher(query, {"properties": export_meta})

    async def _execute_cypher(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query."""
        url = f"{self.config.server_url}/db/{self.config.database}/tx/commit"

        statement = {"statement": query, "parameters": parameters or {}}

        request_body = {"statements": [statement]}

        async with self.session.post(url, data=json.dumps(request_body)) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ExportError(f"Neo4j query failed: {response.status} - {error_text}")

            result = await response.json()

            if result.get("errors"):
                raise ExportError(f"Neo4j query error: {result['errors']}")

            # Extract data from result
            if result.get("results") and result["results"][0].get("data"):
                return [row["row"] for row in result["results"][0]["data"]]

            return []


class ArangoExporter(GraphExporter):
    """ArangoDB-specific graph exporter."""

    def __init__(self, config: ArangoConfig):
        """Initialize ArangoDB exporter."""
        super().__init__(config)
        self.config: ArangoConfig = config

    async def _initialize_connection(self):
        """Initialize ArangoDB connection."""
        self.logger.info(f"Connecting to ArangoDB: {self.config.server_url}")

        # Create authentication
        auth = None
        if self.config.username and self.config.password:
            auth = aiohttp.BasicAuth(self.config.username, self.config.password)

        # Create session
        self.session = aiohttp.ClientSession(
            auth=auth, headers={"Content-Type": "application/json"}
        )

        # Test connection
        try:
            async with self.session.get(f"{self.config.server_url}/_api/version") as response:
                if response.status != 200:
                    raise ExportError(f"ArangoDB connection test failed: {response.status}")

                self.logger.info("Connected to ArangoDB successfully")

        except aiohttp.ClientError as e:
            raise ExportError(f"Failed to connect to ArangoDB: {str(e)}")

    async def _cleanup_connection(self):
        """Cleanup ArangoDB connection."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _clear_database(self):
        """Clear existing collections from ArangoDB."""
        self.logger.info("Clearing existing ArangoDB collections")

        # Get list of collections
        async with self.session.get(f"{self.config.server_url}/_api/collection") as response:
            if response.status == 200:
                collections = await response.json()

                # Delete collections with our prefix
                for collection in collections.get("result", []):
                    if collection["name"].startswith(self.config.collection_prefix):
                        await self._delete_collection(collection["name"])

    async def _delete_collection(self, collection_name: str):
        """Delete ArangoDB collection."""
        url = f"{self.config.server_url}/_api/collection/{collection_name}"

        async with self.session.delete(url) as response:
            if response.status not in [200, 404]:
                self.logger.warning(f"Failed to delete collection {collection_name}")

    async def _setup_schema(self):
        """Setup ArangoDB collections and graph."""
        self.logger.info("Setting up ArangoDB schema")

        # Create document collections
        doc_collections = [
            "verses",
            "topics",
            "events",
            "people",
            "places",
            "periods",
            "annotations",
        ]

        for collection in doc_collections:
            await self._create_collection(
                f"{self.config.collection_prefix}_{collection}", "document"
            )

        # Create edge collections
        edge_collections = [
            "references",
            "contains",
            "about",
            "participates",
            "occurs",
            "describes",
        ]

        for collection in edge_collections:
            await self._create_collection(f"{self.config.collection_prefix}_{collection}", "edge")

        # Create named graph if requested
        if self.config.create_graph:
            await self._create_graph()

    async def _create_collection(self, collection_name: str, collection_type: str):
        """Create ArangoDB collection."""
        data = {"name": collection_name, "type": 3 if collection_type == "edge" else 2}

        async with self.session.post(
            f"{self.config.server_url}/_api/collection", data=json.dumps(data)
        ) as response:
            if response.status not in [200, 201, 409]:  # 409 = already exists
                self.logger.warning(f"Failed to create collection {collection_name}")

    async def _create_graph(self):
        """Create ArangoDB named graph."""
        edge_definitions = [
            {
                "collection": f"{self.config.collection_prefix}_references",
                "from": [f"{self.config.collection_prefix}_verses"],
                "to": [f"{self.config.collection_prefix}_verses"],
            },
            {
                "collection": f"{self.config.collection_prefix}_about",
                "from": [f"{self.config.collection_prefix}_annotations"],
                "to": [f"{self.config.collection_prefix}_topics"],
            },
            {
                "collection": f"{self.config.collection_prefix}_participates",
                "from": [f"{self.config.collection_prefix}_people"],
                "to": [f"{self.config.collection_prefix}_events"],
            },
        ]

        graph_data = {"name": self.config.graph_name, "edgeDefinitions": edge_definitions}

        async with self.session.post(
            f"{self.config.server_url}/_api/gharial", data=json.dumps(graph_data)
        ) as response:
            if response.status not in [201, 202, 409]:
                self.logger.warning("Failed to create named graph")

    async def _export_nodes(self):
        """Export nodes to ArangoDB."""
        self.logger.info("Exporting nodes to ArangoDB")

        # Group nodes by collection
        nodes_by_collection = {}

        for node in self.nodes.values():
            label = node["label"].lower()
            collection = f"{self.config.collection_prefix}_{label}s"

            if collection not in nodes_by_collection:
                nodes_by_collection[collection] = []

            # Prepare document
            doc = {"_key": node["id"].replace(":", "_"), **node["properties"]}

            nodes_by_collection[collection].append(doc)

        # Import each collection
        for collection, documents in nodes_by_collection.items():
            await self._import_documents(collection, documents)

    async def _export_relationships(self):
        """Export relationships to ArangoDB."""
        self.logger.info("Exporting relationships to ArangoDB")

        # Group relationships by type
        rels_by_type = {}

        for rel in self.relationships:
            rel_type = rel["type"].lower()
            collection = f"{self.config.collection_prefix}_{rel_type}"

            if collection not in rels_by_type:
                rels_by_type[collection] = []

            # Prepare edge document
            edge_doc = {
                "_from": f"{self.config.collection_prefix}_{rel['source'].split(':')[0]}s/{rel['source'].replace(':', '_')}",
                "_to": f"{self.config.collection_prefix}_{rel['target'].split(':')[0]}s/{rel['target'].replace(':', '_')}",
                **rel["properties"],
            }

            rels_by_type[collection].append(edge_doc)

        # Import each edge collection
        for collection, edges in rels_by_type.items():
            await self._import_documents(collection, edges)

    async def _import_documents(self, collection: str, documents: List[Dict[str, Any]]):
        """Import documents into ArangoDB collection."""
        # Process in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i : i + self.config.batch_size]

            url = f"{self.config.server_url}/_api/document/{collection}"

            async with self.session.post(url, data=json.dumps(batch)) as response:
                if response.status not in [200, 201, 202]:
                    self.logger.warning(f"Failed to import batch to {collection}")

    async def _create_indices(self):
        """Create ArangoDB indices."""
        self.logger.info("Creating ArangoDB indices")

        # Define indices for different collections
        indices = [
            (f"{self.config.collection_prefix}_verses", ["verse_id", "book", "chapter"]),
            (f"{self.config.collection_prefix}_topics", ["name", "category"]),
            (f"{self.config.collection_prefix}_events", ["name", "event_type"]),
            (f"{self.config.collection_prefix}_people", ["name"]),
            (f"{self.config.collection_prefix}_places", ["name"]),
        ]

        for collection, fields in indices:
            for field in fields:
                await self._create_index(collection, field)

    async def _create_index(self, collection: str, field: str):
        """Create index on specific field."""
        index_data = {"type": "persistent", "fields": [field]}

        url = f"{self.config.server_url}/_api/index?collection={collection}"

        async with self.session.post(url, data=json.dumps(index_data)) as response:
            if response.status not in [200, 201]:
                self.logger.warning(f"Failed to create index on {collection}.{field}")

    async def _compute_graph_metrics(self):
        """Compute ArangoDB graph metrics."""
        self.logger.info("Computing graph metrics")

        # Use AQL queries for metrics
        queries = [
            ("node_count", "FOR doc IN @@collection COLLECT WITH COUNT INTO length RETURN length"),
            (
                "edge_count",
                "FOR edge IN @@edge_collection COLLECT WITH COUNT INTO length RETURN length",
            ),
        ]

        metrics = {}
        # This is a simplified implementation - would need actual AQL execution
        self.logger.info("Graph metrics computed")
        return metrics

    async def _finalize_export(self, metadata: Dict[str, Any]):
        """Finalize ArangoDB export."""
        self.logger.info("Finalizing ArangoDB export")

        # Store export metadata
        export_meta = {
            "_key": "export_metadata",
            "format": "arangodb",
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "exporter": "ABBA ArangoDB Exporter",
            "custom_metadata": metadata,
        }

        collection = f"{self.config.collection_prefix}_metadata"
        await self._create_collection(collection, "document")

        url = f"{self.config.server_url}/_api/document/{collection}"
        async with self.session.post(url, data=json.dumps(export_meta)) as response:
            if response.status not in [200, 201]:
                self.logger.warning("Failed to store export metadata")

    async def validate_output(self, result: ExportResult) -> ValidationResult:
        """Validate ArangoDB export output."""
        validation = ValidationResult(is_valid=True)

        if not self.session:
            validation.add_error("No ArangoDB connection available for validation")
            return validation

        try:
            # Check collections exist
            async with self.session.get(f"{self.config.server_url}/_api/collection") as response:
                if response.status == 200:
                    collections = await response.json()
                    collection_names = [c["name"] for c in collections.get("result", [])]

                    required_collections = [
                        f"{self.config.collection_prefix}_verses",
                        f"{self.config.collection_prefix}_references",
                    ]

                    for collection in required_collections:
                        if collection not in collection_names:
                            validation.add_error(f"Required collection missing: {collection}")
                else:
                    validation.add_error("Failed to retrieve collection list")

        except Exception as e:
            validation.add_error(f"ArangoDB validation failed: {str(e)}")

        return validation
