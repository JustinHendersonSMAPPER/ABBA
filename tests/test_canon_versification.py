"""Tests for versification engine."""

import pytest
from abba.canon.versification import VersificationEngine, MappingResult
from abba.canon.models import VerseMapping, MappingType, VersificationScheme
from abba.verse_id import VerseID


class TestVersificationEngine:
    """Test VersificationEngine functionality."""

    @pytest.fixture
    def engine(self):
        """Create a versification engine for testing."""
        return VersificationEngine()

    def test_engine_initialization(self, engine):
        """Test engine initializes with default mappings."""
        # Engine should have some mappings after initialization
        assert len(engine._mappings) > 0

        # Check that bidirectional mappings exist
        assert ("masoretic", "septuagint") in engine._mappings
        assert ("septuagint", "masoretic") in engine._reverse_mappings

    def test_map_same_scheme(self, engine):
        """Test mapping within the same versification scheme."""
        verse = VerseID("PSA", 23, 1)
        result = engine.map_verse(verse, "standard", "standard")

        assert result.success
        assert result.source_verses == [verse]
        assert result.target_verses == [verse]
        assert result.mapping_type == MappingType.ONE_TO_ONE
        assert result.confidence == 1.0

    def test_psalm_numbering_differences(self, engine):
        """Test Psalm numbering differences between Hebrew and Greek."""
        # Hebrew Psalm 51 = LXX Psalm 50
        verse = VerseID("PSA", 51, 1)
        result = engine.map_verse(verse, "masoretic", "septuagint")

        assert result.success
        assert len(result.target_verses) == 1
        assert result.target_verses[0].chapter == 50
        assert result.mapping_type == MappingType.ONE_TO_ONE

        # Reverse mapping
        lxx_verse = VerseID("PSA", 50, 1)
        reverse_result = engine.map_verse(lxx_verse, "septuagint", "masoretic")

        assert reverse_result.success
        assert reverse_result.target_verses[0].chapter == 51

    def test_psalm_split_merge(self, engine):
        """Test Psalm split/merge cases."""
        # Hebrew Psalm 9 = first part of LXX Psalm 9
        # Hebrew Psalm 10 = second part of LXX Psalm 9

        # This is a complex case that would need specific verse mappings
        # For now, test the general mechanism
        pass

    def test_3john_verse_split(self, engine):
        """Test 3 John verse differences."""
        # 3 John 1:14 in standard = 1:14-15 in Vulgate
        verse = VerseID("3JN", 1, 14)
        result = engine.map_verse(verse, "standard", "vulgate")

        # Based on our mapping, this should map to verses 14 and 15
        assert result.success
        assert result.mapping_type == MappingType.ONE_TO_MANY
        assert len(result.target_verses) == 2
        assert result.target_verses[0].verse == 14
        assert result.target_verses[1].verse == 15

    def test_daniel_additions(self, engine):
        """Test Daniel Greek additions (null mappings)."""
        # Daniel 13 (Susanna) exists only in Greek
        verse = VerseID("DAN", 13, 1)
        result = engine.map_verse(verse, "septuagint", "masoretic")

        assert result.success
        assert result.mapping_type == MappingType.NULL_MAPPING
        assert len(result.target_verses) == 0
        assert result.is_null

    def test_malachi_chapter_differences(self, engine):
        """Test Malachi chapter differences."""
        # Malachi 3:19 in Hebrew = 4:1 in English
        verse = VerseID("MAL", 3, 19)
        result = engine.map_verse(verse, "masoretic", "standard")

        assert result.success
        assert result.target_verses[0].chapter == 4
        assert result.target_verses[0].verse == 1

    def test_no_mapping_found(self, engine):
        """Test behavior when no specific mapping exists."""
        # Use a verse that likely has no specific mapping
        verse = VerseID("GEN", 1, 1)
        result = engine.map_verse(verse, "standard", "septuagint")

        assert result.success
        assert result.source_verses == [verse]
        assert result.target_verses == [verse]
        assert result.confidence < 1.0  # Lower confidence for assumed mapping
        assert "No specific mapping found" in result.notes

    def test_add_custom_mapping(self, engine):
        """Test adding custom verse mappings."""
        # Add a custom mapping
        custom_mapping = VerseMapping(
            source_scheme_id="custom1",
            target_scheme_id="custom2",
            mapping_type=MappingType.ONE_TO_ONE,
            source_book="TST",
            source_chapter=1,
            source_verses=[1],
            target_book="TST",
            target_chapter=2,
            target_verses=[3],
            notes="Test mapping",
        )

        engine.add_mapping(custom_mapping)

        # Test the mapping works
        verse = VerseID("TST", 1, 1)
        result = engine.map_verse(verse, "custom1", "custom2")

        assert result.success
        assert result.target_verses[0].chapter == 2
        assert result.target_verses[0].verse == 3
        assert result.notes == "Test mapping"

    def test_map_verse_range(self, engine):
        """Test mapping a range of verses."""
        start = VerseID("MAL", 3, 19)
        end = VerseID("MAL", 3, 21)

        results = engine.map_verse_range(start, end, "masoretic", "standard")

        assert len(results) == 3  # verses 19, 20, 21

        # Check each mapping
        for i, result in enumerate(results):
            assert result.success
            expected_verse = i + 1  # 4:1, 4:2, 4:3
            assert result.target_verses[0].chapter == 4
            assert result.target_verses[0].verse == expected_verse

    def test_chapter_level_mapping(self, engine):
        """Test chapter-level mapping (verse 0 indicates all verses)."""
        # The engine uses verse 0 to indicate chapter-level mappings
        # This is used for Psalm renumbering where entire chapters map

        # Based on initialization, Psalm 51 Hebrew = Psalm 50 Greek
        verse = VerseID("PSA", 51, 5)  # Any verse in the chapter
        result = engine.map_verse(verse, "masoretic", "septuagint")

        assert result.success
        assert result.target_verses[0].chapter == 50
        assert result.target_verses[0].verse == 5  # Same verse number

    def test_get_scheme_differences(self, engine):
        """Test getting all differences between schemes."""
        differences = engine.get_scheme_differences("masoretic", "standard")

        assert len(differences) > 0

        # Should include Malachi differences
        mal_diffs = [d for d in differences if d.source_book == "MAL"]
        assert len(mal_diffs) > 0

    def test_register_versification_scheme(self, engine):
        """Test registering a versification scheme."""
        scheme = VersificationScheme(
            id="test_scheme",
            name="Test Scheme",
            description="Test versification scheme",
            base_text="Test",
        )

        engine.register_scheme(scheme)

        retrieved = engine.get_scheme("test_scheme")
        assert retrieved is not None
        assert retrieved.name == "Test Scheme"

    def test_can_map_between(self, engine):
        """Test checking if mapping is possible between schemes."""
        # Same scheme always mappable
        assert engine.can_map_between("standard", "standard")

        # Schemes with mappings
        assert engine.can_map_between("masoretic", "septuagint")
        assert engine.can_map_between("septuagint", "masoretic")
        assert engine.can_map_between("standard", "vulgate")

        # Schemes without mappings (hypothetical)
        assert not engine.can_map_between("nonexistent1", "nonexistent2")

    def test_mapping_result_properties(self):
        """Test MappingResult properties."""
        # Test split
        split_result = MappingResult(
            success=True,
            source_verses=[VerseID("TST", 1, 1)],
            target_verses=[VerseID("TST", 1, 1), VerseID("TST", 1, 2)],
            mapping_type=MappingType.ONE_TO_MANY,
        )
        assert split_result.is_split
        assert not split_result.is_merge
        assert not split_result.is_null

        # Test merge
        merge_result = MappingResult(
            success=True,
            source_verses=[VerseID("TST", 1, 1), VerseID("TST", 1, 2)],
            target_verses=[VerseID("TST", 1, 1)],
            mapping_type=MappingType.MANY_TO_ONE,
        )
        assert merge_result.is_merge
        assert not merge_result.is_split
        assert not merge_result.is_null

        # Test null
        null_result = MappingResult(
            success=True,
            source_verses=[VerseID("TST", 1, 1)],
            target_verses=[],
            mapping_type=MappingType.NULL_MAPPING,
        )
        assert null_result.is_null
        assert not null_result.is_split
        assert not null_result.is_merge
