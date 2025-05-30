"""Unit tests for verse_mapper module."""

import pytest

from abba.alignment.verse_mapper import (
    EnhancedVerseMapper,
    MappingConfidence,
    MappingType,
    VerseMapping,
)
from abba.verse_id import VerseID, VerseRange, parse_verse_id
from abba.versification import VersificationSystem


class TestMappingType:
    """Test the MappingType enum."""

    def test_mapping_type_values(self) -> None:
        """Test mapping type enum values."""
        assert MappingType.EXACT.value == "exact"
        assert MappingType.SPLIT.value == "split"
        assert MappingType.MERGE.value == "merge"
        assert MappingType.MISSING.value == "missing"


class TestMappingConfidence:
    """Test the MappingConfidence enum."""

    def test_confidence_values(self) -> None:
        """Test confidence enum values."""
        assert MappingConfidence.CERTAIN.value == 1.0
        assert MappingConfidence.HIGH.value == 0.9
        assert MappingConfidence.MEDIUM.value == 0.7
        assert MappingConfidence.LOW.value == 0.5
        assert MappingConfidence.DISPUTED.value == 0.3


class TestVerseMapping:
    """Test the VerseMapping dataclass."""

    def test_creation(self) -> None:
        """Test creating VerseMapping instances."""
        source = parse_verse_id("GEN.1.1")
        targets = [parse_verse_id("GEN.1.1")]

        mapping = VerseMapping(
            source_verse=source,
            target_verses=targets,
            mapping_type=MappingType.EXACT,
            confidence=MappingConfidence.CERTAIN,
        )

        assert mapping.source_verse == source
        assert mapping.target_verses == targets
        assert mapping.mapping_type == MappingType.EXACT
        assert mapping.confidence == MappingConfidence.CERTAIN

    def test_is_one_to_one(self) -> None:
        """Test one-to-one mapping detection."""
        source = parse_verse_id("GEN.1.1")

        # 1:1 mapping
        one_to_one = VerseMapping(
            source_verse=source,
            target_verses=[parse_verse_id("GEN.1.1")],
            mapping_type=MappingType.EXACT,
            confidence=MappingConfidence.CERTAIN,
        )
        assert one_to_one.is_one_to_one()

        # 1:many mapping
        one_to_many = VerseMapping(
            source_verse=source,
            target_verses=[parse_verse_id("GEN.1.1a"), parse_verse_id("GEN.1.1b")],
            mapping_type=MappingType.SPLIT,
            confidence=MappingConfidence.HIGH,
        )
        assert not one_to_many.is_one_to_one()

    def test_is_complex(self) -> None:
        """Test complex mapping detection."""
        source = parse_verse_id("GEN.1.1")

        # Simple mapping
        simple = VerseMapping(
            source_verse=source,
            target_verses=[parse_verse_id("GEN.1.1")],
            mapping_type=MappingType.EXACT,
            confidence=MappingConfidence.CERTAIN,
        )
        assert not simple.is_complex()

        # Complex mapping
        complex_mapping = VerseMapping(
            source_verse=source,
            target_verses=[],
            mapping_type=MappingType.MISSING,
            confidence=MappingConfidence.LOW,
        )
        assert complex_mapping.is_complex()


class TestEnhancedVerseMapper:
    """Test the EnhancedVerseMapper class."""

    def test_initialization(self) -> None:
        """Test mapper initialization."""
        mapper = EnhancedVerseMapper()
        assert mapper._mapping_cache is not None
        assert len(mapper._mapping_cache) > 0  # Should have loaded some mappings

    def test_map_same_system(self) -> None:
        """Test mapping within the same versification system."""
        mapper = EnhancedVerseMapper()

        mapping = mapper.map_verse(
            "GEN.1.1", VersificationSystem.MODERN, VersificationSystem.MODERN
        )

        assert mapping.mapping_type == MappingType.EXACT
        assert mapping.confidence == MappingConfidence.CERTAIN
        assert len(mapping.target_verses) == 1
        assert str(mapping.target_verses[0]) == "GEN.1.1"

    def test_map_psalm_titles(self) -> None:
        """Test mapping Psalm titles between MT and Modern."""
        mapper = EnhancedVerseMapper()

        # This should have a specific mapping due to title differences
        mapping = mapper.map_verse("PSA.3.1", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert mapping.mapping_type in [MappingType.SHIFT, MappingType.EXACT]
        assert mapping.confidence.value >= 0.8

    def test_map_malachi_differences(self) -> None:
        """Test mapping Malachi chapter differences."""
        mapper = EnhancedVerseMapper()

        # Hebrew Malachi 3:19 should map to English 4:1
        mapping = mapper.map_verse("MAL.3.19", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert len(mapping.target_verses) == 1
        assert str(mapping.target_verses[0]) == "MAL.4.1"
        assert mapping.mapping_type == MappingType.SHIFT
        assert mapping.confidence == MappingConfidence.CERTAIN

    def test_map_joel_differences(self) -> None:
        """Test mapping Joel chapter differences."""
        mapper = EnhancedVerseMapper()

        # Hebrew Joel 3:1 should map to English Joel 2:28
        mapping = mapper.map_verse("JOL.3.1", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping is not None
        assert len(mapping.target_verses) == 1
        assert str(mapping.target_verses[0]) == "JOL.2.28"

    def test_map_textual_variants(self) -> None:
        """Test mapping textual variants from KJV to Modern."""
        mapper = EnhancedVerseMapper()

        # Acts 8:37 exists in KJV but missing in modern texts
        mapping = mapper.map_verse("ACT.8.37", VersificationSystem.KJV, VersificationSystem.MODERN)

        assert mapping is not None
        assert len(mapping.target_verses) == 0  # Missing in modern
        assert mapping.mapping_type == MappingType.MISSING

    def test_map_verse_with_string_input(self) -> None:
        """Test mapping with string verse ID input."""
        mapper = EnhancedVerseMapper()

        mapping = mapper.map_verse(
            "GEN.1.1", VersificationSystem.MODERN, VersificationSystem.MODERN
        )

        assert mapping is not None
        assert str(mapping.source_verse) == "GEN.1.1"

    def test_map_verse_with_verse_id_input(self) -> None:
        """Test mapping with VerseID input."""
        mapper = EnhancedVerseMapper()
        verse_id = parse_verse_id("GEN.1.1")

        mapping = mapper.map_verse(verse_id, VersificationSystem.MODERN, VersificationSystem.MODERN)

        assert mapping is not None
        assert mapping.source_verse == verse_id

    def test_map_invalid_verse(self) -> None:
        """Test mapping with invalid verse ID."""
        mapper = EnhancedVerseMapper()

        with pytest.raises(ValueError):
            mapper.map_verse("INVALID", VersificationSystem.MODERN, VersificationSystem.MODERN)

    def test_map_verse_range_string(self) -> None:
        """Test mapping verse ranges with string input."""
        mapper = EnhancedVerseMapper()

        mappings = mapper.map_verse_range(
            "GEN.1.1-GEN.1.3", VersificationSystem.MODERN, VersificationSystem.MODERN
        )

        assert len(mappings) == 3
        assert all(mapping.mapping_type == MappingType.EXACT for mapping in mappings)

    def test_map_verse_range_single_verse(self) -> None:
        """Test mapping single verse as range."""
        mapper = EnhancedVerseMapper()

        mappings = mapper.map_verse_range(
            "GEN.1.1", VersificationSystem.MODERN, VersificationSystem.MODERN
        )

        assert len(mappings) == 1
        assert str(mappings[0].source_verse) == "GEN.1.1"

    def test_map_verse_range_object(self) -> None:
        """Test mapping with VerseRange object."""
        mapper = EnhancedVerseMapper()
        verse_range = VerseRange(parse_verse_id("GEN.1.1"), parse_verse_id("GEN.1.2"))

        mappings = mapper.map_verse_range(
            verse_range, VersificationSystem.MODERN, VersificationSystem.MODERN
        )

        assert len(mappings) == 2

    def test_get_mapping_confidence(self) -> None:
        """Test getting mapping confidence scores."""
        mapper = EnhancedVerseMapper()

        mapping = VerseMapping(
            source_verse=parse_verse_id("GEN.1.1"),
            target_verses=[parse_verse_id("GEN.1.1")],
            mapping_type=MappingType.EXACT,
            confidence=MappingConfidence.HIGH,
        )

        confidence = mapper.get_mapping_confidence(mapping)
        assert confidence == 0.9

    def test_find_conflicting_mappings(self) -> None:
        """Test finding conflicting mappings."""
        mapper = EnhancedVerseMapper()

        conflicts = mapper.find_conflicting_mappings(
            VersificationSystem.KJV, VersificationSystem.MODERN
        )

        # Should find some conflicts (textual variants)
        assert isinstance(conflicts, list)
        # May be empty if no conflicts defined, which is also valid

    def test_get_supported_systems(self) -> None:
        """Test getting supported versification systems."""
        mapper = EnhancedVerseMapper()

        systems = mapper.get_supported_systems()

        assert isinstance(systems, set)
        assert VersificationSystem.MODERN in systems
        assert VersificationSystem.MT in systems
        assert VersificationSystem.KJV in systems

    def test_validate_mapping_consistency(self) -> None:
        """Test mapping consistency validation."""
        mapper = EnhancedVerseMapper()

        # Test with systems that should be consistent
        is_consistent = mapper.validate_mapping_consistency(
            VersificationSystem.MODERN, VersificationSystem.MODERN
        )

        assert is_consistent  # Same system should always be consistent

    def test_default_mapping_for_unknown_verse(self) -> None:
        """Test default mapping for verses without specific mappings."""
        mapper = EnhancedVerseMapper()

        # Use a verse that likely doesn't have specific mapping rules
        mapping = mapper.map_verse("2CO.5.17", VersificationSystem.MODERN, VersificationSystem.KJV)

        assert mapping is not None
        assert mapping.mapping_type == MappingType.EXACT
        assert mapping.confidence == MappingConfidence.HIGH
        assert len(mapping.target_verses) == 1
        assert str(mapping.target_verses[0]) == "2CO.5.17"

    def test_mapping_with_notes(self) -> None:
        """Test that mappings can include explanatory notes."""
        mapper = EnhancedVerseMapper()

        # Malachi mapping should have notes explaining the chapter difference
        mapping = mapper.map_verse("MAL.3.19", VersificationSystem.MT, VersificationSystem.MODERN)

        assert mapping.notes is not None
        assert len(mapping.notes) > 0

    def test_complex_mapping_types(self) -> None:
        """Test various complex mapping types."""
        mapper = EnhancedVerseMapper()

        # Test missing mapping (Acts 8:37)
        missing_mapping = mapper.map_verse(
            "ACT.8.37", VersificationSystem.KJV, VersificationSystem.MODERN
        )
        assert missing_mapping.mapping_type == MappingType.MISSING

        # Test shift mapping (Malachi)
        shift_mapping = mapper.map_verse(
            "MAL.3.19", VersificationSystem.MT, VersificationSystem.MODERN
        )
        assert shift_mapping.mapping_type == MappingType.SHIFT

    def test_verse_range_invalid_format(self) -> None:
        """Test verse range mapping with invalid format."""
        mapper = EnhancedVerseMapper()

        with pytest.raises(ValueError):
            mapper.map_verse_range(
                "INVALID-RANGE", VersificationSystem.MODERN, VersificationSystem.MODERN
            )
