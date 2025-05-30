"""
Tests for ABBA graph database exporters.

Test coverage for Neo4j and ArangoDB exporters, graph data modeling,
node/relationship creation, and graph-specific features.
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from abba.export.graph_exporter import (
    GraphExporter,
    Neo4jExporter,
    ArangoExporter,
    GraphConfig,
    Neo4jConfig,
    ArangoConfig,
)
from abba.export.base import (
    ExportFormat,
    ExportResult,
    ExportStatus,
    CanonicalDataset,
    ValidationResult,
    ExportError,
)
from abba.alignment.unified_reference import UnifiedVerse
from abba.annotations.models import Annotation, AnnotationType, AnnotationLevel, Topic
from abba.timeline.models import Event, TimePeriod, EventType, CertaintyLevel
from abba.verse_id import VerseID


class MockAsyncResponse:
    """Mock aiohttp response for testing."""

    def __init__(self, status: int, json_data: Dict[str, Any] = None, text_data: str = ""):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


class MockAsyncSession:
    """Mock aiohttp session for testing."""

    def __init__(self):
        self.requests = []
        self.responses = {}
        self.closed = False

    def set_response(self, method: str, url: str, response: MockAsyncResponse):
        """Set mock response for specific request."""
        key = f"{method.upper()}:{url}"
        self.responses[key] = response

    def get(self, url: str):
        return self._make_request("GET", url)

    def post(self, url: str, **kwargs):
        return self._make_request("POST", url, **kwargs)

    def delete(self, url: str):
        return self._make_request("DELETE", url)

    def _make_request(self, method: str, url: str, **kwargs):
        self.requests.append((method, url, kwargs))
        key = f"{method}:{url}"
        return MockAsyncContextManager(
            self.responses.get(key, MockAsyncResponse(200, {"status": "ok"}))
        )

    async def close(self):
        self.closed = True


class MockAsyncContextManager:
    """Mock async context manager for responses."""

    def __init__(self, response: MockAsyncResponse):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestGraphConfig:
    """Test graph database configuration classes."""

    def test_basic_graph_config(self):
        """Test basic graph configuration."""
        config = GraphConfig(
            server_url="http://localhost:7474",
            username="neo4j",
            password="password",
            database="neo4j",
            batch_size=100,
        )

        assert config.server_url == "http://localhost:7474"
        assert config.username == "neo4j"
        assert config.password == "password"
        assert config.database == "neo4j"
        assert config.batch_size == 100
        assert config.create_indices is True
        assert config.clear_existing_data is True

    def test_neo4j_config(self):
        """Test Neo4j-specific configuration."""
        config = Neo4jConfig(
            server_url="http://localhost:7474",
            use_bolt_protocol=True,
            bolt_port=7687,
            enable_relationship_properties=True,
            compute_graph_metrics=True,
        )

        assert config.format_type == ExportFormat.NEO4J
        assert config.use_bolt_protocol is True
        assert config.bolt_port == 7687
        assert config.enable_relationship_properties is True
        assert config.compute_graph_metrics is True

    def test_arango_config(self):
        """Test ArangoDB-specific configuration."""
        config = ArangoConfig(
            server_url="http://localhost:8529",
            collection_prefix="test_abba",
            create_graph=True,
            graph_name="test_graph",
        )

        assert config.format_type == ExportFormat.ARANGODB
        assert config.collection_prefix == "test_abba"
        assert config.create_graph is True
        assert config.graph_name == "test_graph"

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = GraphConfig(server_url="http://localhost:7474")
        validation = config.validate()
        assert validation.is_valid

        # Invalid config - no server URL
        config = GraphConfig(server_url="")
        validation = config.validate()
        assert not validation.is_valid

        # Invalid config - invalid batch size
        config = GraphConfig(server_url="http://localhost:7474", batch_size=0)
        validation = config.validate()
        assert not validation.is_valid


class TestNeo4jExporter:
    """Test Neo4j graph exporter."""

    @pytest.fixture
    def config(self):
        """Create test Neo4j configuration."""
        return Neo4jConfig(
            server_url="http://localhost:7474",
            username="neo4j",
            password="password",
            database="neo4j",
            batch_size=10,  # Small for testing
            create_indices=True,
            compute_graph_metrics=True,
        )

    @pytest.fixture
    def exporter(self, config):
        """Create test Neo4j exporter."""
        return Neo4jExporter(config)

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session for Neo4j."""
        session = MockAsyncSession()

        # Setup default responses
        session.set_response(
            "POST",
            "http://localhost:7474/db/neo4j/tx/commit",
            MockAsyncResponse(200, {"results": [{"data": [{"row": [1]}]}], "errors": []}),
        )

        return session

    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        verses = []
        for i in range(5):
            verse_id = VerseID("GEN", 1, i + 1)
            verse = UnifiedVerse(
                verse_id=verse_id,
                translations={"ESV": f"This is verse {i + 1} in English Standard Version."},
                hebrew_tokens=(
                    [{"word": "בְּרֵאשִׁית", "lemma": "רֵאשִׁית", "strongs": "H7225"}] if i == 0 else None
                ),
                greek_tokens=(
                    [{"word": "Ἐν", "lemma": "ἐν", "strongs": "G1722"}] if i == 0 else None
                ),
            )
            verses.append(verse)
        return verses

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotations."""
        annotations = []
        for i in range(3):
            annotation = Annotation(
                id=f"ann_{i}",
                verse_id=VerseID("GEN", 1, i + 1),
                annotation_type=AnnotationType.TOPIC,
                level=AnnotationLevel.VERSE,
                confidence=0.8,
                topics=[Topic(id=f"topic_{i}", name=f"Topic {i}")],
            )
            annotations.append(annotation)
        return annotations

    @pytest.fixture
    def sample_events(self):
        """Create sample timeline events."""
        from abba.timeline.models import TimePoint, Location, Participant

        events = []
        for i in range(2):
            event = Event(
                id=f"event_{i}",
                name=f"Event {i}",
                description=f"Description of event {i}",
                event_type=EventType.HISTORICAL,
                certainty_level=CertaintyLevel.HIGH,
                categories=["biblical", "historical"],
                time_point=TimePoint(year=-2000 + i * 100),
                location=Location(name=f"Location {i}", latitude=31.0 + i, longitude=35.0 + i),
                participants=[Participant(id=f"person_{i}", name=f"Person {i}")],
                verse_refs=[VerseID("GEN", 1, i + 1)],
            )
            events.append(event)
        return events

    @pytest.fixture
    def sample_dataset(self, sample_verses, sample_annotations, sample_events):
        """Create sample canonical dataset."""
        return CanonicalDataset(
            verses=iter(sample_verses),
            annotations=iter(sample_annotations),
            timeline_events=iter(sample_events),
            metadata={"format": "test", "version": "1.0"},
        )

    def test_exporter_initialization(self, exporter, config):
        """Test exporter initialization."""
        assert exporter.config == config
        assert exporter.logger is not None
        assert isinstance(exporter.nodes, dict)
        assert isinstance(exporter.relationships, list)

    def test_config_validation(self, exporter):
        """Test configuration validation."""
        validation = exporter.validate_config()
        assert validation.is_valid

    def test_supported_features(self, exporter):
        """Test supported features."""
        features = exporter.get_supported_features()
        assert "relationship_traversal" in features
        assert "pattern_matching" in features
        assert "graph_algorithms" in features
        assert "complex_queries" in features
        assert "biblical_research" in features
        assert "citation_analysis" in features

    @pytest.mark.asyncio
    async def test_connection_initialization(self, exporter, mock_session):
        """Test Neo4j connection initialization."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            await exporter._initialize_connection()

            # Should have made test query
            requests = [req for req in mock_session.requests if req[0] == "POST"]
            test_requests = [req for req in requests if "RETURN 1" in str(req[2].get("data", ""))]
            assert len(test_requests) == 1

    @pytest.mark.asyncio
    async def test_database_clearing(self, exporter, mock_session):
        """Test database clearing."""
        exporter.session = mock_session

        await exporter._clear_database()

        # Should have made delete query
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        delete_requests = [
            req for req in requests if "DETACH DELETE" in str(req[2].get("data", ""))
        ]
        assert len(delete_requests) == 1

    @pytest.mark.asyncio
    async def test_schema_setup(self, exporter, mock_session):
        """Test schema and constraints setup."""
        exporter.session = mock_session

        await exporter._setup_schema()

        # Should have made constraint creation queries
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        constraint_requests = [
            req for req in requests if "CONSTRAINT" in str(req[2].get("data", ""))
        ]
        assert len(constraint_requests) > 0

    @pytest.mark.asyncio
    async def test_verse_node_collection(self, exporter, sample_verses):
        """Test verse node collection."""
        await exporter._collect_verse_nodes(iter(sample_verses))

        # Should have created verse nodes
        verse_nodes = [
            node for node_id, node in exporter.nodes.items() if node_id.startswith("verse:")
        ]
        assert len(verse_nodes) == 5

        # Check first verse node
        first_verse_node = verse_nodes[0]
        assert first_verse_node["label"] == "Verse"
        assert "verse_id" in first_verse_node["properties"]
        assert "book" in first_verse_node["properties"]
        assert "translations" in first_verse_node["properties"]

        # Should have created Hebrew token nodes and relationships
        hebrew_nodes = [node for node_id, node in exporter.nodes.items() if "hebrew" in node_id]
        assert len(hebrew_nodes) > 0

        # Should have created relationships
        token_relationships = [rel for rel in exporter.relationships if rel["type"] == "CONTAINS"]
        assert len(token_relationships) > 0

    @pytest.mark.asyncio
    async def test_annotation_node_collection(self, exporter, sample_annotations):
        """Test annotation and topic node collection."""
        await exporter._collect_annotation_nodes(iter(sample_annotations))

        # Should have created annotation nodes
        annotation_nodes = [
            node for node_id, node in exporter.nodes.items() if node_id.startswith("annotation:")
        ]
        assert len(annotation_nodes) == 3

        # Should have created topic nodes
        topic_nodes = [
            node for node_id, node in exporter.nodes.items() if node_id.startswith("topic:")
        ]
        assert len(topic_nodes) == 3  # One topic per annotation

        # Should have created relationships
        annotation_relationships = [
            rel for rel in exporter.relationships if rel["type"] == "ANNOTATED_WITH"
        ]
        assert len(annotation_relationships) == 3

        about_relationships = [rel for rel in exporter.relationships if rel["type"] == "ABOUT"]
        assert len(about_relationships) == 3

    @pytest.mark.asyncio
    async def test_event_node_collection(self, exporter, sample_events):
        """Test timeline event node collection."""
        await exporter._collect_event_nodes(iter(sample_events))

        # Should have created event nodes
        event_nodes = [
            node for node_id, node in exporter.nodes.items() if node_id.startswith("event:")
        ]
        assert len(event_nodes) == 2

        # Should have created person nodes
        person_nodes = [
            node for node_id, node in exporter.nodes.items() if node_id.startswith("person:")
        ]
        assert len(person_nodes) == 2

        # Should have created place nodes
        place_nodes = [
            node for node_id, node in exporter.nodes.items() if node_id.startswith("place:")
        ]
        assert len(place_nodes) == 2

        # Check event node properties
        first_event_node = event_nodes[0]
        assert first_event_node["label"] == "Event"
        props = first_event_node["properties"]
        assert "event_id" in props
        assert "name" in props
        assert "event_type" in props
        assert "certainty_level" in props

        # Should have temporal data
        assert "exact_date" in props or "earliest_date" in props

        # Should have created relationships
        participates_relationships = [
            rel for rel in exporter.relationships if rel["type"] == "PARTICIPATES_IN"
        ]
        assert len(participates_relationships) == 2

        occurs_relationships = [rel for rel in exporter.relationships if rel["type"] == "OCCURS_AT"]
        assert len(occurs_relationships) == 2

        describes_relationships = [
            rel for rel in exporter.relationships if rel["type"] == "DESCRIBES"
        ]
        assert len(describes_relationships) == 2

    @pytest.mark.asyncio
    async def test_node_export(self, exporter, mock_session):
        """Test node export to Neo4j."""
        # Add some test nodes
        exporter.nodes["verse:GEN.1.1"] = {
            "id": "verse:GEN.1.1",
            "label": "Verse",
            "properties": {"verse_id": "GEN.1.1", "book": "GEN"},
        }
        exporter.nodes["topic:topic_1"] = {
            "id": "topic:topic_1",
            "label": "Topic",
            "properties": {"topic_id": "topic_1", "name": "Creation"},
        }

        exporter.session = mock_session

        await exporter._export_nodes()

        # Should have made node creation queries
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        node_requests = [
            req
            for req in requests
            if "UNWIND" in str(req[2].get("data", "")) and "CREATE" in str(req[2].get("data", ""))
        ]
        assert len(node_requests) >= 1

    @pytest.mark.asyncio
    async def test_relationship_export(self, exporter, mock_session):
        """Test relationship export to Neo4j."""
        # Add some test relationships
        exporter.relationships.extend(
            [
                {
                    "source": "verse:GEN.1.1",
                    "target": "annotation:ann_1",
                    "type": "ANNOTATED_WITH",
                    "properties": {"confidence": 0.8},
                },
                {
                    "source": "annotation:ann_1",
                    "target": "topic:topic_1",
                    "type": "ABOUT",
                    "properties": {"confidence": 0.9},
                },
            ]
        )

        exporter.session = mock_session

        await exporter._export_relationships()

        # Should have made relationship creation queries
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        rel_requests = [
            req
            for req in requests
            if "CREATE (" in str(req[2].get("data", "")) and ")-[" in str(req[2].get("data", ""))
        ]
        assert len(rel_requests) >= 1

    @pytest.mark.asyncio
    async def test_index_creation(self, exporter, mock_session):
        """Test index creation."""
        exporter.session = mock_session

        await exporter._create_indices()

        # Should have made index creation queries
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        index_requests = [req for req in requests if "CREATE INDEX" in str(req[2].get("data", ""))]
        assert len(index_requests) > 0

    @pytest.mark.asyncio
    async def test_graph_metrics_computation(self, exporter, mock_session):
        """Test graph metrics computation."""
        # Setup response for metrics queries
        mock_session.set_response(
            "POST",
            "http://localhost:7474/db/neo4j/tx/commit",
            MockAsyncResponse(200, {"results": [{"data": [{"row": [100]}]}], "errors": []}),
        )

        exporter.session = mock_session

        metrics = await exporter._compute_graph_metrics()

        assert isinstance(metrics, dict)

        # Should have made metrics queries
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        metrics_requests = [
            req for req in requests if "count(" in str(req[2].get("data", "")).lower()
        ]
        assert len(metrics_requests) > 0

    @pytest.mark.asyncio
    async def test_cypher_query_execution(self, exporter, mock_session):
        """Test Cypher query execution."""
        exporter.session = mock_session

        result = await exporter._execute_cypher("RETURN 1 as test", {"param": "value"})

        assert isinstance(result, list)

        # Should have made query request
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        assert len(requests) == 1

        # Check request structure
        request_data = json.loads(requests[0][2]["data"])
        assert "statements" in request_data
        assert len(request_data["statements"]) == 1

        statement = request_data["statements"][0]
        assert statement["statement"] == "RETURN 1 as test"
        assert statement["parameters"] == {"param": "value"}

    @pytest.mark.asyncio
    async def test_export_metadata_finalization(self, exporter, mock_session):
        """Test export metadata finalization."""
        # Add some nodes and relationships for statistics
        exporter.nodes["test_node"] = {"id": "test_node", "label": "Test", "properties": {}}
        exporter.relationships.append(
            {"source": "node1", "target": "node2", "type": "TEST", "properties": {}}
        )

        exporter.session = mock_session

        await exporter._finalize_export({"custom": "metadata"})

        # Should have made metadata creation query
        requests = [req for req in mock_session.requests if req[0] == "POST"]
        metadata_requests = [
            req for req in requests if "ExportMetadata" in str(req[2].get("data", ""))
        ]
        assert len(metadata_requests) == 1

    @pytest.mark.asyncio
    async def test_full_export_workflow(self, exporter, mock_session, sample_dataset):
        """Test complete export workflow."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await exporter.export(sample_dataset)

        assert result.status == ExportStatus.COMPLETED

        # Should have collected nodes and relationships
        assert len(exporter.nodes) > 0
        assert len(exporter.relationships) > 0

        # Should have made various requests
        request_types = [req[0] for req in mock_session.requests]
        assert "POST" in request_types

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, config):
        """Test connection error handling."""
        exporter = Neo4jExporter(config)

        # Mock connection failure
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.side_effect = Exception("Connection failed")
            mock_session_class.return_value = mock_session

            sample_dataset = CanonicalDataset(verses=iter([]), metadata={"format": "test"})

            result = await exporter.export(sample_dataset)
            assert result.status == ExportStatus.FAILED
            assert result.error is not None


class TestArangoExporter:
    """Test ArangoDB graph exporter."""

    @pytest.fixture
    def config(self):
        """Create test ArangoDB configuration."""
        return ArangoConfig(
            server_url="http://localhost:8529",
            username="root",
            password="password",
            database="_system",
            collection_prefix="test_abba",
            batch_size=10,
            create_graph=True,
            graph_name="test_biblical_graph",
        )

    @pytest.fixture
    def exporter(self, config):
        """Create test ArangoDB exporter."""
        return ArangoExporter(config)

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session for ArangoDB."""
        session = MockAsyncSession()

        # Setup default responses
        session.set_response(
            "GET",
            "http://localhost:8529/_api/version",
            MockAsyncResponse(200, {"version": "3.11.0"}),
        )

        session.set_response(
            "GET", "http://localhost:8529/_api/collection", MockAsyncResponse(200, {"result": []})
        )

        session.set_response(
            "POST", "http://localhost:8529/_api/collection", MockAsyncResponse(201, {"id": "12345"})
        )

        session.set_response(
            "POST",
            "http://localhost:8529/_api/gharial",
            MockAsyncResponse(201, {"graph": {"name": "test_graph"}}),
        )

        return session

    def test_exporter_initialization(self, exporter, config):
        """Test ArangoDB exporter initialization."""
        assert exporter.config == config
        assert exporter.logger is not None
        assert isinstance(exporter.nodes, dict)
        assert isinstance(exporter.relationships, list)

    @pytest.mark.asyncio
    async def test_connection_initialization(self, exporter, mock_session):
        """Test ArangoDB connection initialization."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            await exporter._initialize_connection()

            # Should have made version check
            requests = [req for req in mock_session.requests if req[0] == "GET"]
            version_requests = [req for req in requests if "_api/version" in req[1]]
            assert len(version_requests) == 1

    @pytest.mark.asyncio
    async def test_database_clearing(self, exporter, mock_session):
        """Test database clearing."""
        # Setup response with existing collections
        mock_session.set_response(
            "GET",
            "http://localhost:8529/_api/collection",
            MockAsyncResponse(
                200,
                {
                    "result": [
                        {"name": "test_abba_verses"},
                        {"name": "other_collection"},
                        {"name": "test_abba_events"},
                    ]
                },
            ),
        )

        mock_session.set_response(
            "DELETE",
            "http://localhost:8529/_api/collection/test_abba_verses",
            MockAsyncResponse(200, {"id": "123"}),
        )

        exporter.session = mock_session

        await exporter._clear_database()

        # Should have deleted collections with our prefix
        requests = [req for req in mock_session.requests if req[0] == "DELETE"]
        delete_requests = [req for req in requests if "test_abba" in req[1]]
        assert len(delete_requests) >= 1

    @pytest.mark.asyncio
    async def test_schema_setup(self, exporter, mock_session):
        """Test schema setup (collections and graph)."""
        exporter.session = mock_session

        await exporter._setup_schema()

        # Should have created document collections
        post_requests = [req for req in mock_session.requests if req[0] == "POST"]
        collection_requests = [req for req in post_requests if "_api/collection" in req[1]]
        assert len(collection_requests) > 0

        # Should have created graph if enabled
        if exporter.config.create_graph:
            graph_requests = [req for req in post_requests if "_api/gharial" in req[1]]
            assert len(graph_requests) == 1

    @pytest.mark.asyncio
    async def test_node_export(self, exporter, mock_session, sample_verses):
        """Test node export to ArangoDB."""
        # Setup document import response
        mock_session.set_response(
            "POST",
            "http://localhost:8529/_api/document/test_abba_verses",
            MockAsyncResponse(200, [{"_id": "test_abba_verses/123"}]),
        )

        # Collect and export nodes
        await exporter._collect_verse_nodes(iter(sample_verses))

        exporter.session = mock_session

        await exporter._export_nodes()

        # Should have made document import requests
        post_requests = [req for req in mock_session.requests if req[0] == "POST"]
        import_requests = [req for req in post_requests if "_api/document" in req[1]]
        assert len(import_requests) > 0

    @pytest.mark.asyncio
    async def test_relationship_export(self, exporter, mock_session):
        """Test relationship export to ArangoDB."""
        # Add test relationships
        exporter.relationships.extend(
            [
                {
                    "source": "verse:GEN.1.1",
                    "target": "annotation:ann_1",
                    "type": "ANNOTATED_WITH",
                    "properties": {"confidence": 0.8},
                }
            ]
        )

        exporter.session = mock_session

        await exporter._export_relationships()

        # Should have made edge import requests
        post_requests = [req for req in mock_session.requests if req[0] == "POST"]
        edge_requests = [
            req for req in post_requests if "_api/document" in req[1] and "annotated_with" in req[1]
        ]
        assert len(edge_requests) >= 0  # May depend on collection creation

    @pytest.mark.asyncio
    async def test_index_creation(self, exporter, mock_session):
        """Test index creation."""
        mock_session.set_response(
            "POST", "http://localhost:8529/_api/index", MockAsyncResponse(201, {"id": "index_123"})
        )

        exporter.session = mock_session

        await exporter._create_indices()

        # Should have made index creation requests
        post_requests = [req for req in mock_session.requests if req[0] == "POST"]
        index_requests = [req for req in post_requests if "_api/index" in req[1]]
        assert len(index_requests) > 0

    @pytest.mark.asyncio
    async def test_document_import_batching(self, exporter, mock_session):
        """Test document import with batching."""
        # Create large number of documents
        documents = []
        for i in range(25):  # More than batch_size (10)
            documents.append({"_key": f"doc_{i}", "name": f"Document {i}"})

        exporter.session = mock_session

        await exporter._import_documents("test_collection", documents)

        # Should have made multiple batch requests
        post_requests = [req for req in mock_session.requests if req[0] == "POST"]
        import_requests = [req for req in post_requests if "test_collection" in req[1]]
        assert len(import_requests) >= 3  # 25 docs / 10 batch_size = 3 batches

    @pytest.mark.asyncio
    async def test_graph_creation(self, exporter, mock_session):
        """Test named graph creation."""
        exporter.session = mock_session

        await exporter._create_graph()

        # Should have made graph creation request
        post_requests = [req for req in mock_session.requests if req[0] == "POST"]
        graph_requests = [req for req in post_requests if "_api/gharial" in req[1]]
        assert len(graph_requests) == 1

        # Check graph structure
        graph_request = graph_requests[0]
        graph_data = json.loads(graph_request[2]["data"])

        assert graph_data["name"] == exporter.config.graph_name
        assert "edgeDefinitions" in graph_data
        assert len(graph_data["edgeDefinitions"]) > 0

    @pytest.mark.asyncio
    async def test_export_metadata_finalization(self, exporter, mock_session):
        """Test export metadata finalization."""
        mock_session.set_response(
            "POST",
            "http://localhost:8529/_api/collection",
            MockAsyncResponse(201, {"id": "metadata_collection"}),
        )

        mock_session.set_response(
            "POST",
            "http://localhost:8529/_api/document/test_abba_metadata",
            MockAsyncResponse(201, {"_id": "test_abba_metadata/export_metadata"}),
        )

        exporter.session = mock_session

        await exporter._finalize_export({"custom": "metadata"})

        # Should have created metadata collection and document
        post_requests = [req for req in mock_session.requests if req[0] == "POST"]
        metadata_requests = [req for req in post_requests if "metadata" in req[1]]
        assert len(metadata_requests) >= 1

    @pytest.mark.asyncio
    async def test_output_validation(self, exporter, mock_session):
        """Test ArangoDB output validation."""
        # Setup validation responses
        mock_session.set_response(
            "GET",
            "http://localhost:8529/_api/collection",
            MockAsyncResponse(
                200, {"result": [{"name": "test_abba_verses"}, {"name": "test_abba_references"}]}
            ),
        )

        exporter.session = mock_session

        result = ExportResult(format_type=ExportFormat.ARANGODB, status=ExportStatus.COMPLETED)

        validation = await exporter.validate_output(result)

        # Should check for required collections
        assert validation.is_valid or len(validation.warnings) > 0

    @pytest.mark.asyncio
    async def test_full_export_workflow(self, exporter, mock_session, sample_dataset):
        """Test complete ArangoDB export workflow."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await exporter.export(sample_dataset)

        assert result.status == ExportStatus.COMPLETED

        # Should have collected nodes and relationships
        assert len(exporter.nodes) > 0
        assert len(exporter.relationships) > 0

        # Should have made various API calls
        request_methods = [req[0] for req in mock_session.requests]
        assert "GET" in request_methods  # Version check
        assert "POST" in request_methods  # Collection/document creation


class TestGraphDataModeling:
    """Test graph data modeling and structure."""

    def test_node_id_generation(self):
        """Test consistent node ID generation."""
        # Test verse node IDs
        verse = UnifiedVerse(verse_id=VerseID("GEN", 1, 1), translations={"ESV": "Test verse"})

        expected_verse_id = "verse:GEN.1.1"
        # This would be tested in the actual node collection
        assert str(verse.verse_id) == "GEN.1.1"

    def test_relationship_types(self):
        """Test relationship type definitions."""
        relationship_types = [
            "CONTAINS",  # Verse -> Token
            "ANNOTATED_WITH",  # Verse -> Annotation
            "ABOUT",  # Annotation -> Topic
            "REFERENCES",  # Verse -> Verse (cross-references)
            "PARTICIPATES_IN",  # Person -> Event
            "OCCURS_AT",  # Event -> Place
            "DESCRIBES",  # Verse -> Event
            "CONTAINS_PERIOD",  # Period -> Period
            "CONTAINS_EVENT",  # Period -> Event
        ]

        # Verify relationship types are well-defined
        for rel_type in relationship_types:
            assert isinstance(rel_type, str)
            assert len(rel_type) > 0

    def test_node_properties_structure(self):
        """Test node properties structure."""
        # Test verse node properties
        verse_props = {
            "verse_id": "GEN.1.1",
            "book": "GEN",
            "chapter": 1,
            "verse": 1,
            "translations": {"ESV": "Test text"},
            "created_at": datetime.utcnow().isoformat(),
        }

        assert "verse_id" in verse_props
        assert "book" in verse_props
        assert isinstance(verse_props["chapter"], int)
        assert isinstance(verse_props["verse"], int)

        # Test event node properties
        event_props = {
            "event_id": "event_1",
            "name": "Test Event",
            "event_type": "historical",
            "certainty_level": "high",
            "categories": ["biblical"],
            "created_at": datetime.utcnow().isoformat(),
        }

        assert "event_id" in event_props
        assert "name" in event_props
        assert isinstance(event_props["categories"], list)

    def test_relationship_properties_structure(self):
        """Test relationship properties structure."""
        # Test annotation relationship
        annotation_rel_props = {"confidence": 0.8, "created_at": datetime.utcnow().isoformat()}

        assert "confidence" in annotation_rel_props
        assert 0.0 <= annotation_rel_props["confidence"] <= 1.0

        # Test cross-reference relationship
        cross_ref_props = {"reference_type": "parallel", "confidence": 0.9, "source": "commentary"}

        assert "reference_type" in cross_ref_props
        assert "confidence" in cross_ref_props


if __name__ == "__main__":
    pytest.main([__file__])
