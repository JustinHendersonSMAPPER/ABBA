"""Unit tests for bridge_tables module."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

from abba.alignment.bridge_tables import (
    MappingData,
    VersificationBridge,
)
from abba.alignment.verse_mapper import MappingConfidence, MappingType
from abba.verse_id import parse_verse_id
from abba.versification import VersificationSystem


class TestMappingData:
    """Test the MappingData dataclass."""

    def test_creation(self) -> None:
        """Test creating MappingData instances."""
        mappings = {"GEN.1.1": ["GEN.1.1"]}
        metadata = {"source": "test", "version": "1.0"}

        data = MappingData(
            from_system=VersificationSystem.MT,
            to_system=VersificationSystem.MODERN,
            mappings=mappings,
            metadata=metadata,
        )

        assert data.from_system == VersificationSystem.MT
        assert data.to_system == VersificationSystem.MODERN
        assert data.mappings == mappings
        assert data.metadata == metadata
        assert data.confidence_scores == {}  # Default value
        assert data.mapping_types == {}  # Default value

    def test_post_init(self) -> None:
        """Test that post_init initializes optional fields."""
        data = MappingData(
            from_system=VersificationSystem.MT,
            to_system=VersificationSystem.MODERN,
            mappings={},
            metadata={},
        )

        assert data.confidence_scores is not None
        assert data.mapping_types is not None


class TestVersificationBridge:
    """Test the VersificationBridge class."""

    def test_initialization_without_data_dir(self) -> None:
        """Test bridge initialization without external data directory."""
        bridge = VersificationBridge()

        assert bridge.data_dir is None
        assert len(bridge.mapping_data) > 0  # Should have built-in mappings
        assert len(bridge.lookup_index) > 0

    def test_initialization_with_nonexistent_data_dir(self) -> None:
        """Test bridge initialization with nonexistent data directory."""
        bridge = VersificationBridge("/nonexistent/path")

        assert bridge.data_dir == Path("/nonexistent/path")
        assert len(bridge.mapping_data) > 0  # Should still have built-in mappings

    def test_initialization_with_external_data(self) -> None:
        """Test bridge initialization with external mapping files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test mapping file
            test_mapping = {
                "mappings": {"TEST.1.1": ["TEST.1.1"]},
                "confidence_scores": {"TEST.1.1": 0.95},
                "mapping_types": {"TEST.1.1": "exact"},
                "metadata": {"source": "test"},
            }

            mapping_file = Path(temp_dir) / "mt_to_modern.json"
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(test_mapping, f)

            bridge = VersificationBridge(temp_dir)

            # Should load the external mapping
            key = (VersificationSystem.MT, VersificationSystem.MODERN)
            assert key in bridge.mapping_data
            assert "TEST.1.1" in bridge.mapping_data[key].mappings

    def test_built_in_mappings_loaded(self) -> None:
        """Test that built-in mappings are loaded correctly."""
        bridge = VersificationBridge()

        # Check that key mapping systems are present
        expected_keys = [
            (VersificationSystem.MT, VersificationSystem.MODERN),
            (VersificationSystem.KJV, VersificationSystem.MODERN),
            (VersificationSystem.LXX, VersificationSystem.MODERN),
        ]

        for key in expected_keys:
            assert key in bridge.mapping_data

    def test_malachi_mappings(self) -> None:
        """Test specific Malachi chapter mappings."""
        bridge = VersificationBridge()

        # Hebrew Malachi 3:19 should map to English 4:1
        mapping = bridge.get_mapping("MAL.3.19", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert len(mapping.target_verses) == 1
        assert str(mapping.target_verses[0]) == "MAL.4.1"
        assert mapping.mapping_type == MappingType.SHIFT
        assert mapping.confidence == MappingConfidence.CERTAIN

    def test_joel_mappings(self) -> None:
        """Test Joel chapter boundary mappings."""
        bridge = VersificationBridge()

        # Hebrew Joel 3:1 should map to English Joel 2:28
        mapping = bridge.get_mapping("JOL.3.1", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert len(mapping.target_verses) == 1
        assert str(mapping.target_verses[0]) == "JOL.2.28"
        assert mapping.mapping_type == MappingType.SHIFT

    def test_kjv_textual_variants(self) -> None:
        """Test KJV textual variant mappings."""
        bridge = VersificationBridge()

        # Acts 8:37 exists in KJV but missing in modern texts
        mapping = bridge.get_mapping(
            "ACT.8.37", VersificationSystem.KJV, VersificationSystem.MODERN
        )

        assert mapping is not None
        assert len(mapping.target_verses) == 0  # Missing in modern
        assert mapping.mapping_type == MappingType.MISSING

    def test_psalm_title_mappings(self) -> None:
        """Test Psalm title mappings from Hebrew to Modern."""
        bridge = VersificationBridge()

        # Test a psalm with title differences
        mapping = bridge.get_mapping("PSA.3.1", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        # Should have appropriate mapping type for title handling

    def test_lxx_mappings(self) -> None:
        """Test basic LXX mappings."""
        bridge = VersificationBridge()

        # Check that LXX mapping exists (even if basic)
        key = (VersificationSystem.LXX, VersificationSystem.MODERN)
        assert key in bridge.mapping_data

    def test_get_mapping_nonexistent_verse(self) -> None:
        """Test getting mapping for verse that doesn't exist in mappings."""
        bridge = VersificationBridge()

        # This verse likely doesn't have specific mapping rules
        mapping = bridge.get_mapping("ECC.3.1", VersificationSystem.MT, VersificationSystem.MODERN)

        # Should return None for unmapped verses
        assert mapping is None

    def test_get_mapping_nonexistent_system_pair(self) -> None:
        """Test getting mapping for unsupported system pair."""
        bridge = VersificationBridge()

        # Use a system pair that doesn't exist
        mapping = bridge.get_mapping(
            "GEN.1.1", VersificationSystem.VULGATE, VersificationSystem.ORTHODOX
        )

        assert mapping is None

    def test_get_mapping_with_verse_id_object(self) -> None:
        """Test getting mapping with VerseID object input."""
        bridge = VersificationBridge()
        verse_id = parse_verse_id("MAL.3.19")

        mapping = bridge.get_mapping(verse_id, VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert mapping.source_verse == verse_id

    def test_get_mapping_invalid_verse_id(self) -> None:
        """Test getting mapping with invalid verse ID."""
        bridge = VersificationBridge()

        mapping = bridge.get_mapping("INVALID", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is None

    def test_has_mapping(self) -> None:
        """Test checking if mapping exists between systems."""
        bridge = VersificationBridge()

        # Should have mapping from MT to Modern
        assert bridge.has_mapping(VersificationSystem.MT, VersificationSystem.MODERN)

        # Should not have mapping between arbitrary systems
        assert not bridge.has_mapping(VersificationSystem.VULGATE, VersificationSystem.ORTHODOX)

    def test_get_supported_systems(self) -> None:
        """Test getting supported versification systems."""
        bridge = VersificationBridge()

        systems = bridge.get_supported_systems()

        assert isinstance(systems, set)
        assert VersificationSystem.MT in systems
        assert VersificationSystem.MODERN in systems
        assert VersificationSystem.KJV in systems

    def test_get_mapping_statistics(self) -> None:
        """Test getting mapping statistics."""
        bridge = VersificationBridge()

        stats = bridge.get_mapping_statistics(VersificationSystem.MT, VersificationSystem.MODERN)

        assert stats is not None
        assert "from_system" in stats
        assert "to_system" in stats
        assert "total_mappings" in stats
        assert "exact_mappings" in stats
        assert "missing_mappings" in stats
        assert "average_confidence" in stats
        assert stats["from_system"] == "MT"
        assert stats["to_system"] == "Modern"

    def test_get_mapping_statistics_nonexistent(self) -> None:
        """Test getting statistics for nonexistent mapping."""
        bridge = VersificationBridge()

        stats = bridge.get_mapping_statistics(
            VersificationSystem.VULGATE, VersificationSystem.ORTHODOX
        )

        assert stats is None

    def test_validate_mappings(self) -> None:
        """Test mapping validation functionality."""
        bridge = VersificationBridge()

        validation_results = bridge.validate_mappings()

        assert isinstance(validation_results, dict)
        assert "total_systems" in validation_results
        assert "total_mappings" in validation_results
        assert "bidirectional_pairs" in validation_results
        assert "inconsistencies" in validation_results
        assert "missing_reverse_mappings" in validation_results

    def test_confidence_score_conversion(self) -> None:
        """Test confidence score conversion to enum values."""
        bridge = VersificationBridge()

        # Get a mapping that should have a confidence score
        mapping = bridge.get_mapping("MAL.3.19", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert isinstance(mapping.confidence, MappingConfidence)
        assert mapping.confidence.value <= 1.0
        assert mapping.confidence.value >= 0.0

    def test_mapping_type_conversion(self) -> None:
        """Test mapping type conversion to enum values."""
        bridge = VersificationBridge()

        mapping = bridge.get_mapping("MAL.3.19", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert isinstance(mapping.mapping_type, MappingType)

    def test_lookup_index_built(self) -> None:
        """Test that lookup index is properly built."""
        bridge = VersificationBridge()

        assert len(bridge.lookup_index) > 0

        # Check that index contains parsed VerseID objects
        for index in bridge.lookup_index.values():
            if index:  # Skip empty indices
                for target_verses in index.values():
                    for verse in target_verses:
                        assert verse is not None
                        # All entries should be VerseID objects
                        assert hasattr(verse, "book")
                        assert hasattr(verse, "chapter")
                        assert hasattr(verse, "verse")

    def test_external_file_loading_error_handling(self) -> None:
        """Test error handling when loading malformed external files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a malformed JSON file
            mapping_file = Path(temp_dir) / "mt_to_modern.json"
            with open(mapping_file, "w", encoding="utf-8") as f:
                f.write("invalid json content")

            # Should not crash, but log warning and continue
            bridge = VersificationBridge(temp_dir)

            # Should still have built-in mappings
            assert len(bridge.mapping_data) > 0

    def test_complex_mapping_statistics(self) -> None:
        """Test statistics calculation for various mapping types."""
        bridge = VersificationBridge()

        stats = bridge.get_mapping_statistics(VersificationSystem.KJV, VersificationSystem.MODERN)

        if stats:  # May not exist if no KJV mappings loaded
            assert stats["complex_mappings"] >= 0
            assert stats["total_mappings"] >= stats["exact_mappings"] + stats["missing_mappings"]

    def test_metadata_preservation(self) -> None:
        """Test that mapping metadata is preserved."""
        bridge = VersificationBridge()

        key = (VersificationSystem.MT, VersificationSystem.MODERN)
        if key in bridge.mapping_data:
            mapping_data = bridge.mapping_data[key]
            assert "description" in mapping_data.metadata
            assert "source" in mapping_data.metadata
            assert "last_updated" in mapping_data.metadata
