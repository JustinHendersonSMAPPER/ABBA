"""
Tests for ABBA OpenSearch exporter.

Test coverage for OpenSearch cluster export, index creation,
bulk import, custom analyzers, and search optimization.
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from abba.export.opensearch_exporter import OpenSearchExporter, OpenSearchConfig
from abba.export.base import (
    ExportFormat,
    ExportResult,
    ExportStatus,
    CanonicalDataset,
    ValidationResult,
    ExportError,
)
from abba.parsers.translation_parser import TranslationVerse
from abba.annotations.models import Annotation, AnnotationType, AnnotationLevel, Topic, TopicCategory
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

    def put(self, url: str, **kwargs):
        return self._make_request("PUT", url, **kwargs)

    def head(self, url: str):
        return self._make_request("HEAD", url)

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


class TestOpenSearchConfig:
    """Test OpenSearch exporter configuration."""

    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        config = OpenSearchConfig(
            output_path="/tmp/opensearch_export",
            format_type=ExportFormat.OPENSEARCH,
            cluster_url="https://localhost:9200",
            username="admin",
            password="admin",
            index_prefix="test_abba",
            bulk_size=1000,
        )

        assert config.cluster_url == "https://localhost:9200"
        assert config.username == "admin"
        assert config.password == "admin"
        assert config.index_prefix == "test_abba"
        assert config.bulk_size == 1000
        assert config.format_type == ExportFormat.OPENSEARCH

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = OpenSearchConfig(output_path="/tmp/opensearch_export", format_type=ExportFormat.OPENSEARCH, cluster_url="http://localhost:9200")
        validation = config.validate()
        assert validation.is_valid

        # Invalid config - no cluster URL
        config = OpenSearchConfig(output_path="/tmp/opensearch_export", format_type=ExportFormat.OPENSEARCH, cluster_url="")
        validation = config.validate()
        assert not validation.is_valid

        # Invalid config - invalid bulk size
        config = OpenSearchConfig(output_path="/tmp/opensearch_export", format_type=ExportFormat.OPENSEARCH, cluster_url="http://localhost:9200", bulk_size=0)
        validation = config.validate()
        assert not validation.is_valid

    def test_ssl_configuration(self):
        """Test SSL configuration options."""
        config = OpenSearchConfig(
            output_path="/tmp/opensearch_export",
            format_type=ExportFormat.OPENSEARCH,
            cluster_url="https://secure-cluster:9200",
            verify_ssl=False,
            timeout=60
        )

        assert config.verify_ssl is False
        assert config.timeout == 60

    def test_index_configuration(self):
        """Test index configuration options."""
        config = OpenSearchConfig(
            output_path="/tmp/opensearch_export",
            format_type=ExportFormat.OPENSEARCH,
            cluster_url="http://localhost:9200",
            index_prefix="custom_abba",
            index_version="v2",
            number_of_shards=3,
            number_of_replicas=1,
            refresh_interval="10s",
        )

        assert config.index_prefix == "custom_abba"
        assert config.index_version == "v2"
        assert config.number_of_shards == 3
        assert config.number_of_replicas == 1
        assert config.refresh_interval == "10s"


class TestOpenSearchExporter:
    """Test OpenSearch exporter functionality."""

    @pytest.fixture
    def config(self):
        """Create test OpenSearch configuration."""
        return OpenSearchConfig(
            output_path="/tmp/opensearch_export",
            format_type=ExportFormat.OPENSEARCH,
            cluster_url="http://localhost:9200",
            username="admin",
            password="admin",
            index_prefix="test_abba",
            bulk_size=10,  # Small for testing
            enable_biblical_analyzer=True,
            refresh_after_import=True,
        )

    @pytest.fixture
    def exporter(self, config):
        """Create test OpenSearch exporter."""
        return OpenSearchExporter(config)

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = MockAsyncSession()

        # Setup default responses
        session.set_response(
            "GET",
            "http://localhost:9200/_cluster/health",
            MockAsyncResponse(200, {"cluster_name": "test-cluster", "status": "green"}),
        )

        # Index creation responses
        session.set_response(
            "HEAD",
            "http://localhost:9200/test_abba_verses_v1",
            MockAsyncResponse(404),  # Index doesn't exist
        )
        session.set_response(
            "PUT",
            "http://localhost:9200/test_abba_verses_v1",
            MockAsyncResponse(200, {"acknowledged": True}),
        )

        # Bulk import responses
        session.set_response(
            "POST",
            "http://localhost:9200/_bulk",
            MockAsyncResponse(200, {"took": 10, "errors": False, "items": []}),
        )

        # Alias creation
        session.set_response(
            "POST", "http://localhost:9200/_aliases", MockAsyncResponse(200, {"acknowledged": True})
        )

        # Refresh responses
        session.set_response(
            "POST",
            "http://localhost:9200/test_abba_verses_v1/_refresh",
            MockAsyncResponse(200, {"_shards": {"total": 1}}),
        )

        return session

    @pytest.fixture
    def sample_verses(self):
        """Create sample verses for testing."""
        verses = []
        for i in range(20):  # Enough to test batching
            verse_id = VerseID("GEN", 1, i + 1)
            verse = TranslationVerse(
                verse_id=verse_id,
                text=f"This is verse {i + 1} in English Standard Version.",
                original_book_name="Genesis",
                original_chapter=1,
                original_verse=i + 1
            )
            verses.append(verse)
        return verses

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotations."""
        annotations = []
        for i in range(10):
            annotation = Annotation(
                id=f"ann_{i}",
                start_verse=VerseID("GEN", 1, i + 1),
                annotation_type=AnnotationType.THEOLOGICAL_THEME,
                level=AnnotationLevel.VERSE,
                topic_id=f"topic_{i}",
                topic_name=f"Topic {i}",
                content=f"Annotation content for verse {i + 1}"
            )
            annotations.append(annotation)
        return annotations

    @pytest.fixture
    def sample_events(self):
        """Create sample timeline events."""
        from abba.timeline.models import TimePoint, Location, EntityRef, create_bce_date

        events = []
        for i in range(5):
            event = Event(
                id=f"event_{i}",
                name=f"Event {i}",
                description=f"Description of event {i}",
                event_type=EventType.POINT,
                certainty_level=CertaintyLevel.CERTAIN,
                categories=["biblical", "historical"],
                time_point=TimePoint(exact_date=create_bce_date(2000 - i * 100)),
                location=Location(name=f"Location {i}", latitude=31.0 + i, longitude=35.0 + i),
                participants=[EntityRef(id=f"person_{i}", name=f"Person {i}", entity_type="person")],
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
        assert exporter.verses_index == "test_abba_verses_v1"
        assert exporter.annotations_index == "test_abba_annotations_v1"
        assert exporter.events_index == "test_abba_events_v1"

    def test_config_validation(self, exporter):
        """Test configuration validation."""
        validation = exporter.validate_config()
        assert validation.is_valid

    def test_supported_features(self, exporter):
        """Test supported features."""
        features = exporter.get_supported_features()
        assert "full_text_search" in features
        assert "faceted_search" in features
        assert "aggregations" in features
        assert "real_time_search" in features
        assert "scalable_search" in features
        assert "multilingual_analysis" in features

    def test_custom_analyzers(self, exporter):
        """Test custom analyzer definitions."""
        analyzers = exporter._get_custom_analyzers()

        assert "analyzer" in analyzers
        assert "biblical_text" in analyzers["analyzer"]
        assert "biblical_names" in analyzers["analyzer"]
        assert "original_language" in analyzers["analyzer"]

        assert "filter" in analyzers
        assert "biblical_synonyms" in analyzers["filter"]
        assert "biblical_stemmer" in analyzers["filter"]

        # Check biblical synonyms
        synonyms = analyzers["filter"]["biblical_synonyms"]["synonyms"]
        assert any("God,LORD,Yahweh" in synonym for synonym in synonyms)

    def test_index_mappings(self, exporter):
        """Test index mapping definitions."""
        mappings = exporter._get_index_mappings()

        # Check verses mapping
        assert "verses" in mappings
        verses_mapping = mappings["verses"]["properties"]
        assert "verse_id" in verses_mapping
        assert "all_text" in verses_mapping
        assert "hebrew" in verses_mapping
        assert "greek" in verses_mapping

        # Check annotations mapping
        assert "annotations" in mappings
        annotations_mapping = mappings["annotations"]["properties"]
        assert "annotation_id" in annotations_mapping
        assert "topics" in annotations_mapping
        assert annotations_mapping["topics"]["type"] == "nested"

        # Check events mapping
        assert "events" in mappings
        events_mapping = mappings["events"]["properties"]
        assert "event_id" in events_mapping
        assert "location" in events_mapping
        assert "participants" in events_mapping
        assert events_mapping["participants"]["type"] == "nested"

    def test_index_settings(self, exporter):
        """Test index settings generation."""
        settings = exporter._get_index_settings()

        assert "number_of_shards" in settings
        assert "number_of_replicas" in settings
        assert "refresh_interval" in settings

        # Should include custom analyzers
        if exporter.config.enable_biblical_analyzer:
            assert "analysis" in settings

    @pytest.mark.asyncio
    async def test_connection_initialization(self, exporter, mock_session):
        """Test OpenSearch connection initialization."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            await exporter._initialize_connection()

            # Should have made health check request
            requests = [req for req in mock_session.requests if req[0] == "GET"]
            health_requests = [req for req in requests if "_cluster/health" in req[1]]
            assert len(health_requests) == 1

    @pytest.mark.asyncio
    async def test_index_creation(self, exporter, mock_session):
        """Test index creation with mappings and settings."""
        exporter.session = mock_session

        await exporter._setup_indices()

        # Should have checked if indices exist
        head_requests = [req for req in mock_session.requests if req[0] == "HEAD"]
        assert len(head_requests) >= 1

        # Should have created indices
        put_requests = [req for req in mock_session.requests if req[0] == "PUT"]
        assert len(put_requests) >= 1

        # Check that index creation included mappings and settings
        index_creation_request = next(
            req for req in put_requests if "test_abba_verses_v1" in req[1]
        )
        request_data = json.loads(index_creation_request[2]["data"])
        assert "settings" in request_data
        assert "mappings" in request_data

    @pytest.mark.asyncio
    async def test_verse_serialization(self, exporter):
        """Test verse serialization for OpenSearch."""
        # Create a verse with translations attribute
        verse = TranslationVerse(
            verse_id=VerseID("GEN", 1, 1),
            text="In the beginning God created the heavens and the earth.",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        # Add translations attribute
        verse.translations = {
            "ESV": "In the beginning God created the heavens and the earth.",
            "NIV": "In the beginning God created the heavens and the earth."
        }
        # Add Hebrew tokens
        verse.hebrew_tokens = [
            {"word": "בְּרֵאשִׁית", "lemma": "רֵאשִׁית", "strongs": "H7225"},
            {"word": "בָּרָא", "lemma": "בָּרָא", "strongs": "H1254"}
        ]
        # Add metadata
        verse.metadata = {"source": "test"}

        doc = exporter._serialize_verse_for_opensearch(verse)

        # Check basic fields
        assert doc["verse_id"] == "GEN.1.1"
        assert doc["book"] == "GEN"
        assert doc["chapter"] == 1
        assert doc["verse"] == 1

        # Check translations
        assert "translations" in doc
        assert "all_text" in doc
        assert "text_ESV" in doc
        assert "text_NIV" in doc

        # Check Hebrew data
        assert "hebrew" in doc
        assert "tokens" in doc["hebrew"]
        assert "words" in doc["hebrew"]
        assert "lemmas" in doc["hebrew"]
        assert "strongs" in doc["hebrew"]

        # Check metadata
        assert "metadata" in doc
        assert "indexed_at" in doc

    @pytest.mark.asyncio
    async def test_annotation_serialization(self, exporter):
        """Test annotation serialization for OpenSearch."""
        from abba.annotations.models import AnnotationConfidence
        
        annotation = Annotation(
            id="test_ann",
            start_verse=VerseID("GEN", 1, 1),
            annotation_type=AnnotationType.THEOLOGICAL_THEME,
            level=AnnotationLevel.VERSE,
            topic_id="topic_1",
            topic_name="Creation",
            content="Annotation about creation",
            confidence=AnnotationConfidence(
                overall_score=0.9,
                model_confidence=0.9,
                contextual_relevance=0.9,
                semantic_similarity=0.9
            )
        )

        doc = exporter._serialize_annotation_for_opensearch(annotation)

        # Check basic fields
        assert doc["annotation_id"] == "test_ann"
        assert doc["verse_id"] == "GEN.1.1"
        assert doc["book"] == "GEN"
        assert doc["type"] == "theological_theme"
        assert doc["confidence"] == 0.9

        # Check topics
        assert "topics" in doc
        assert len(doc["topics"]) == 1
        assert doc["topics"][0]["name"] == "Creation"

        # Check searchable topic fields
        assert "topic_names" in doc
        assert "Creation" in doc["topic_names"]

        assert "topic_ids" in doc
        assert "topic_1" in doc["topic_ids"]

    @pytest.mark.asyncio
    async def test_event_serialization(self, exporter):
        """Test timeline event serialization for OpenSearch."""
        from abba.timeline.models import TimePoint, Location, create_bce_date

        event = Event(
            id="test_event",
            name="Test Event",
            description="A test event",
            event_type=EventType.POINT,
            certainty_level=CertaintyLevel.CERTAIN,
            categories=["biblical", "historical"],
            time_point=TimePoint(exact_date=create_bce_date(2000)),
            location=Location(
                name="Jerusalem",
                modern_name="Jerusalem",
                region="Judea",
                latitude=31.7683,
                longitude=35.2137,
            ),
            verse_refs=[VerseID("2SA", 5, 7)],
        )

        doc = exporter._serialize_event_for_opensearch(event)

        # Check basic fields
        assert doc["event_id"] == "test_event"
        assert doc["name"] == "Test Event"
        assert doc["event_type"] == "point"
        assert doc["certainty_level"] == "certain"
        assert doc["categories"] == ["biblical", "historical"]

        # Check temporal data
        assert "time_point" in doc
        assert doc["time_point"]["display_date"] == "2000 BCE"

        # Check location
        assert "location" in doc
        assert doc["location"]["name"] == "Jerusalem"

        # Check verse references
        assert "verse_refs" in doc
        assert "2SA.5.7" in doc["verse_refs"]

        # Check filtering fields
        assert "indexed_at" in doc

    @pytest.mark.asyncio
    async def test_bulk_export_processing(self, exporter, mock_session, sample_dataset):
        """Test bulk export processing."""
        exporter.session = mock_session

        await exporter._export_verses(sample_dataset.verses)

        # Should have made bulk requests
        bulk_requests = [
            req for req in mock_session.requests if req[0] == "POST" and "_bulk" in req[1]
        ]
        assert len(bulk_requests) >= 1

        # Check bulk request format
        bulk_request = bulk_requests[0]
        bulk_data = bulk_request[2]["data"]

        # Should contain NDJSON format
        lines = bulk_data.strip().split("\n")
        assert len(lines) % 2 == 0  # Even number (action + document pairs)

        # Check first action line
        action_line = json.loads(lines[0])
        assert "index" in action_line
        assert "_index" in action_line["index"]
        assert "_id" in action_line["index"]

        # Check first document line
        doc_line = json.loads(lines[1])
        assert "verse_id" in doc_line
        assert "book" in doc_line

    @pytest.mark.asyncio
    async def test_error_handling_during_bulk_import(self, exporter, mock_session):
        """Test error handling during bulk import."""
        # Setup session to return bulk errors
        error_response = MockAsyncResponse(
            200,
            {
                "took": 10,
                "errors": True,
                "items": [
                    {"index": {"error": {"type": "mapping_exception", "reason": "Field error"}}},
                    {"index": {"_id": "doc2", "status": 201}},
                ],
            },
        )
        mock_session.set_response("POST", "http://localhost:9200/_bulk", error_response)

        exporter.session = mock_session

        # Create test data
        verses = [TranslationVerse(
            verse_id=VerseID("GEN", 1, 1),
            text="Test verse",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )]

        # Should handle bulk errors gracefully (log warnings but continue)
        await exporter._export_verses(iter(verses))

        # Should have attempted the bulk request
        bulk_requests = [
            req for req in mock_session.requests if req[0] == "POST" and "_bulk" in req[1]
        ]
        assert len(bulk_requests) >= 1

    @pytest.mark.asyncio
    async def test_retry_logic(self, exporter, mock_session):
        """Test retry logic for failed requests."""
        # Setup session to fail then succeed
        fail_response = MockAsyncResponse(503, text_data="Service Unavailable")
        success_response = MockAsyncResponse(200, {"took": 10, "errors": False, "items": []})

        # First call fails, subsequent calls succeed
        call_count = 0

        def get_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockAsyncContextManager(fail_response)
            else:
                return MockAsyncContextManager(success_response)

        with patch.object(mock_session, "post", side_effect=get_response):
            exporter.session = mock_session

            # Should retry and eventually succeed
            bulk_body = ['{"index": {"_index": "test", "_id": "1"}}', '{"field": "value"}']
            result = await exporter._send_bulk_request(bulk_body)

            assert result is not None
            assert call_count > 1  # Should have retried

    @pytest.mark.asyncio
    async def test_alias_creation(self, exporter, mock_session):
        """Test index alias creation."""
        exporter.session = mock_session

        await exporter._create_aliases()

        # Should have made alias creation request
        alias_requests = [
            req for req in mock_session.requests if req[0] == "POST" and "_aliases" in req[1]
        ]
        assert len(alias_requests) == 1

        # Check alias actions
        alias_request = alias_requests[0]
        alias_data = json.loads(alias_request[2]["data"])

        assert "actions" in alias_data
        actions = alias_data["actions"]
        assert len(actions) >= 1

        # Check alias structure
        action = actions[0]
        assert "add" in action
        assert "index" in action["add"]
        assert "alias" in action["add"]

    @pytest.mark.asyncio
    async def test_index_refresh(self, exporter, mock_session):
        """Test index refresh after import."""
        exporter.session = mock_session

        await exporter._finalize_export({"test": "metadata"})

        # Should have made refresh requests
        refresh_requests = [
            req for req in mock_session.requests if req[0] == "POST" and "_refresh" in req[1]
        ]
        assert len(refresh_requests) >= 1

    @pytest.mark.asyncio
    async def test_metadata_storage(self, exporter, mock_session):
        """Test export metadata storage."""
        # Setup response for metadata document creation
        mock_session.set_response(
            "PUT",
            "http://localhost:9200/test_abba_metadata_v1/_doc/export_info",
            MockAsyncResponse(201, {"_id": "export_info", "_version": 1}),
        )

        exporter.session = mock_session
        exporter.stats.processed_verses = 100
        exporter.stats.processed_annotations = 50

        await exporter._finalize_export({"custom": "metadata"})

        # Should have stored metadata
        metadata_requests = [
            req
            for req in mock_session.requests
            if req[0] == "PUT" and "metadata" in req[1] and "export_info" in req[1]
        ]
        assert len(metadata_requests) == 1

        # Check metadata content
        metadata_request = metadata_requests[0]
        metadata_doc = json.loads(metadata_request[2]["data"])

        assert metadata_doc["format"] == "opensearch"
        assert metadata_doc["exporter"] == "ABBA OpenSearch Exporter"
        assert "statistics" in metadata_doc
        assert "indices" in metadata_doc
        assert "custom_metadata" in metadata_doc

    @pytest.mark.asyncio
    async def test_output_validation(self, exporter, mock_session):
        """Test export output validation."""
        # Setup validation responses
        mock_session.set_response(
            "GET",
            "http://localhost:9200/_cluster/health",
            MockAsyncResponse(200, {"status": "green"}),
        )
        mock_session.set_response(
            "HEAD", "http://localhost:9200/test_abba_verses_v1", MockAsyncResponse(200)
        )
        mock_session.set_response(
            "GET",
            "http://localhost:9200/test_abba_verses_v1/_count",
            MockAsyncResponse(200, {"count": 100}),
        )
        mock_session.set_response(
            "POST",
            "http://localhost:9200/test_abba_verses_v1/_search",
            MockAsyncResponse(200, {"hits": {"total": {"value": 100}, "hits": [{}]}}),
        )

        exporter.session = mock_session
        exporter.stats.processed_verses = 100

        result = ExportResult(
            format_type=ExportFormat.OPENSEARCH, 
            status=ExportStatus.COMPLETED,
            output_path="http://localhost:9200"
        )

        validation = await exporter.validate_output(result)
        assert validation.is_valid
        assert len(validation.errors) == 0

    @pytest.mark.asyncio
    async def test_full_export_workflow(self, exporter, mock_session, sample_dataset):
        """Test complete export workflow."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await exporter.export(sample_dataset)

        assert result.status == ExportStatus.COMPLETED

        # Check that all major operations were performed
        request_methods = [req[0] for req in mock_session.requests]

        # Should have made health check
        assert "GET" in request_methods

        # Should have created indices
        assert "PUT" in request_methods

        # Should have imported data
        assert "POST" in request_methods

        # Verify bulk requests were made
        bulk_requests = [
            req for req in mock_session.requests if req[0] == "POST" and "_bulk" in req[1]
        ]
        assert len(bulk_requests) > 0

    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, config):
        """Test handling of connection failures."""
        exporter = OpenSearchExporter(config)

        # Mock connection failure
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.side_effect = Exception("Connection failed")
            mock_session_class.return_value = mock_session

            sample_dataset = CanonicalDataset(verses=iter([]), metadata={"format": "test"})

            result = await exporter.export(sample_dataset)
            assert result.status == ExportStatus.FAILED
            assert result.error is not None

    def test_book_order_calculation(self, exporter):
        """Test book order calculation for sorting."""
        # Test with known book
        gen_order = exporter._get_book_order("GEN")
        assert gen_order == 1

        mat_order = exporter._get_book_order("MAT")
        assert mat_order > gen_order

        # Test with unknown book
        unknown_order = exporter._get_book_order("UNKNOWN")
        assert unknown_order == 999

    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, exporter, mock_session):
        """Test handling of large datasets with multiple batches."""
        # Create large dataset
        large_verses = []
        for i in range(100):  # More than bulk_size (10)
            verse_id = VerseID("GEN", i // 10 + 1, (i % 10) + 1)
            verse = TranslationVerse(
                verse_id=verse_id,
                text=f"Large dataset verse {i}",
                original_book_name="Genesis",
                original_chapter=i // 10 + 1,
                original_verse=(i % 10) + 1
            )
            large_verses.append(verse)

        exporter.session = mock_session

        await exporter._export_verses(iter(large_verses))

        # Should have made multiple bulk requests
        bulk_requests = [
            req for req in mock_session.requests if req[0] == "POST" and "_bulk" in req[1]
        ]
        assert len(bulk_requests) >= 10  # 100 verses / 10 bulk_size = 10 batches


if __name__ == "__main__":
    pytest.main([__file__])
