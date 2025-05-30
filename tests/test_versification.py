"""Unit tests for versification module."""

from abba import (
    VerseID,
    VersificationDifference,
    VersificationMapper,
    VersificationRules,
    VersificationSystem,
    get_versification_documentation,
)


class TestVersificationSystem:
    """Test the VersificationSystem enum."""

    def test_system_values(self):
        """Test that versification systems have correct values."""
        assert VersificationSystem.MT.value == "MT"
        assert VersificationSystem.LXX.value == "LXX"
        assert VersificationSystem.VULGATE.value == "Vulgate"
        assert VersificationSystem.KJV.value == "KJV"
        assert VersificationSystem.MODERN.value == "Modern"
        assert VersificationSystem.ORTHODOX.value == "Orthodox"
        assert VersificationSystem.CATHOLIC.value == "Catholic"

    def test_system_count(self):
        """Test that we have all expected versification systems."""
        systems = list(VersificationSystem)
        assert len(systems) == 7


class TestVersificationDifference:
    """Test the VersificationDifference dataclass."""

    def test_creation(self):
        """Test creating VersificationDifference instances."""
        diff = VersificationDifference(
            book="PSA",
            chapter=3,
            system1=VersificationSystem.MT,
            system2=VersificationSystem.MODERN,
            difference_type="offset",
            description="Hebrew includes psalm title as verse 1",
            mapping={"offset": 1, "start_verse": 1},
        )

        assert diff.book == "PSA"
        assert diff.chapter == 3
        assert diff.system1 == VersificationSystem.MT
        assert diff.system2 == VersificationSystem.MODERN
        assert diff.difference_type == "offset"
        assert diff.mapping["offset"] == 1

    def test_difference_types(self):
        """Test various difference types."""
        # Split difference
        split_diff = VersificationDifference(
            book="3JN",
            chapter=1,
            system1=VersificationSystem.KJV,
            system2=VersificationSystem.MODERN,
            difference_type="split",
            description="Verse split differently",
            mapping={"14": ["14", "15"]},
        )
        assert split_diff.difference_type == "split"
        assert split_diff.mapping["14"] == ["14", "15"]

        # Reorder difference
        reorder_diff = VersificationDifference(
            book="ROM",
            chapter=16,
            system1=VersificationSystem.KJV,
            system2=VersificationSystem.MODERN,
            difference_type="reorder",
            description="Verses in different order",
            mapping={"16:25-27": ["14:24-26", "16:25-27"]},
        )
        assert reorder_diff.difference_type == "reorder"


class TestVersificationMapper:
    """Test the VersificationMapper class."""

    def test_initialization(self):
        """Test mapper initialization."""
        mapper = VersificationMapper()
        assert mapper.differences is not None
        assert isinstance(mapper.differences, dict)

    def test_build_difference_index(self):
        """Test building the difference index."""
        mapper = VersificationMapper()
        # Should have entries for known differences
        assert ("PSA", 3) in mapper.differences
        assert ("3JN", 1) in mapper.differences
        assert ("MAL", 4) in mapper.differences
        assert ("ROM", 16) in mapper.differences

    def test_get_system_for_translation(self):
        """Test getting versification system for translations."""
        mapper = VersificationMapper()

        # Known translations
        assert mapper.get_system_for_translation("KJV") == VersificationSystem.KJV
        assert (
            mapper.get_system_for_translation("kjv") == VersificationSystem.KJV
        )  # Case insensitive
        assert mapper.get_system_for_translation("ESV") == VersificationSystem.MODERN
        assert mapper.get_system_for_translation("NIV") == VersificationSystem.MODERN
        assert mapper.get_system_for_translation("BHS") == VersificationSystem.MT
        assert mapper.get_system_for_translation("LXX") == VersificationSystem.LXX

        # Unknown translation defaults to MODERN
        assert mapper.get_system_for_translation("UNKNOWN") == VersificationSystem.MODERN
        assert mapper.get_system_for_translation("") == VersificationSystem.MODERN

    def test_map_verse_same_system(self):
        """Test mapping verse within same system."""
        mapper = VersificationMapper()
        verse = VerseID(book="GEN", chapter=1, verse=1)

        result = mapper.map_verse(verse, VersificationSystem.MODERN, VersificationSystem.MODERN)
        assert len(result) == 1
        assert result[0] == verse

    def test_map_verse_string_input(self):
        """Test mapping verse with string input."""
        mapper = VersificationMapper()

        # Valid string
        result = mapper.map_verse("GEN.1.1", VersificationSystem.MODERN, VersificationSystem.MODERN)
        assert len(result) == 1
        assert result[0].book == "GEN"
        assert result[0].chapter == 1
        assert result[0].verse == 1

        # Invalid string
        result = mapper.map_verse("Invalid", VersificationSystem.MODERN, VersificationSystem.KJV)
        assert len(result) == 0

    def test_map_verse_with_offset(self):
        """Test mapping verse with offset difference."""
        mapper = VersificationMapper()

        # Psalm with title (MT verse 2 = Modern verse 1)
        mt_verse = VerseID(book="PSA", chapter=3, verse=2)
        result = mapper.map_verse(mt_verse, VersificationSystem.MT, VersificationSystem.MODERN)
        assert len(result) == 1
        assert result[0].verse == 1  # MT verse 2 maps to Modern verse 1

        # Reverse mapping
        modern_verse = VerseID(book="PSA", chapter=3, verse=1)
        result = mapper.map_verse(modern_verse, VersificationSystem.MODERN, VersificationSystem.MT)
        assert len(result) == 1
        assert result[0].verse == 2  # Modern verse 1 maps to MT verse 2

    def test_map_verse_no_mapping_needed(self):
        """Test mapping verse that doesn't need mapping."""
        mapper = VersificationMapper()

        # Genesis 1:1 is the same in all systems
        verse = VerseID(book="GEN", chapter=1, verse=1)
        result = mapper.map_verse(verse, VersificationSystem.MT, VersificationSystem.MODERN)
        assert len(result) == 1
        assert result[0] == verse

    def test_applies_to_systems(self):
        """Test checking if difference applies to system pair."""
        mapper = VersificationMapper()

        diff = VersificationDifference(
            book="PSA",
            chapter=3,
            system1=VersificationSystem.MT,
            system2=VersificationSystem.MODERN,
            difference_type="offset",
            description="Test",
            mapping={},
        )

        # Should apply in both directions
        assert mapper._applies_to_systems(diff, VersificationSystem.MT, VersificationSystem.MODERN)
        assert mapper._applies_to_systems(diff, VersificationSystem.MODERN, VersificationSystem.MT)

        # Should not apply to unrelated systems
        assert not mapper._applies_to_systems(
            diff, VersificationSystem.KJV, VersificationSystem.LXX
        )

    def test_apply_mapping_offset(self):
        """Test applying offset mapping."""
        mapper = VersificationMapper()

        diff = VersificationDifference(
            book="PSA",
            chapter=3,
            system1=VersificationSystem.MT,
            system2=VersificationSystem.MODERN,
            difference_type="offset",
            description="Test",
            mapping={"offset": 1, "start_verse": 1},
        )

        # MT to Modern (subtract offset)
        verse = VerseID(book="PSA", chapter=3, verse=2)
        result = mapper._apply_mapping(
            verse, diff, VersificationSystem.MT, VersificationSystem.MODERN
        )
        assert len(result) == 1
        assert result[0].verse == 1

        # Modern to MT (add offset)
        verse = VerseID(book="PSA", chapter=3, verse=1)
        result = mapper._apply_mapping(
            verse, diff, VersificationSystem.MODERN, VersificationSystem.MT
        )
        assert len(result) == 1
        assert result[0].verse == 2

    def test_apply_mapping_below_start_verse(self):
        """Test applying mapping for verse below start_verse."""
        mapper = VersificationMapper()

        diff = VersificationDifference(
            book="PSA",
            chapter=3,
            system1=VersificationSystem.MT,
            system2=VersificationSystem.MODERN,
            difference_type="offset",
            description="Test",
            mapping={"offset": 1, "start_verse": 2},  # Only affects verse 2 and above
        )

        # Verse 1 should not be affected
        verse = VerseID(book="PSA", chapter=3, verse=1)
        result = mapper._apply_mapping(
            verse, diff, VersificationSystem.MT, VersificationSystem.MODERN
        )
        assert len(result) == 1
        assert result[0] == verse  # No change

    def test_apply_mapping_invalid_result(self):
        """Test applying mapping that results in invalid verse number."""
        mapper = VersificationMapper()

        # Large offset that would result in verse 0 or negative
        diff = VersificationDifference(
            book="PSA",
            chapter=3,
            system1=VersificationSystem.MT,
            system2=VersificationSystem.MODERN,
            difference_type="offset",
            description="Test",
            mapping={"offset": 5, "start_verse": 1},
        )

        # Verse 3 with offset -5 would be -2 (invalid)
        verse = VerseID(book="PSA", chapter=3, verse=3)
        result = mapper._apply_mapping(
            verse, diff, VersificationSystem.MT, VersificationSystem.MODERN
        )
        assert len(result) == 0  # No valid result

    def test_get_split_verses(self):
        """Test getting split verses for a chapter."""
        mapper = VersificationMapper()

        # 3 John has split verses in KJV
        splits = mapper.get_split_verses("3JN", 1, VersificationSystem.KJV)
        assert len(splits) > 0

        # Should return verse number and parts
        for verse_num, parts in splits:
            assert isinstance(verse_num, int)
            assert isinstance(parts, list)
            assert all(part in ["a", "b", "c"] for part in parts)

    def test_get_split_verses_no_splits(self):
        """Test getting split verses for chapter with no splits."""
        mapper = VersificationMapper()

        # Genesis 1 has no split verses
        splits = mapper.get_split_verses("GEN", 1, VersificationSystem.MODERN)
        assert len(splits) == 0


class TestVersificationRules:
    """Test the VersificationRules documentation class."""

    def test_get_psalm_title_rules(self):
        """Test psalm title rules documentation."""
        rules = VersificationRules.get_psalm_title_rules()

        assert "MT" in rules
        assert "LXX" in rules
        assert "Modern" in rules
        assert "Mapping" in rules

        assert "verse 1" in rules["MT"]
        assert "unnumbered" in rules["Modern"]

    def test_get_chapter_division_rules(self):
        """Test chapter division rules documentation."""
        rules = VersificationRules.get_chapter_division_rules()

        assert "Malachi" in rules
        assert "Joel" in rules

        # Check Malachi rules
        mal_rules = rules["Malachi"][0]
        assert "MT" in mal_rules
        assert "Christian" in mal_rules
        assert "Mapping" in mal_rules
        assert "3:19-24" in mal_rules["Mapping"]
        assert "4:1-6" in mal_rules["Mapping"]

    def test_get_verse_order_variants(self):
        """Test verse order variants documentation."""
        variants = VersificationRules.get_verse_order_variants()

        assert len(variants) >= 2

        # Check Romans doxology
        rom_variant = next(v for v in variants if v["reference"] == "ROM.16.25-27")
        assert rom_variant["issue"] == "Doxology placement"
        assert len(rom_variant["variants"]) >= 3

        # Check Pericope Adulterae
        jhn_variant = next(v for v in variants if v["reference"] == "JHN.7.53-8.11")
        assert "Pericope Adulterae" in jhn_variant["issue"]
        assert len(jhn_variant["variants"]) >= 3

    def test_get_split_verse_rules(self):
        """Test split verse rules documentation."""
        rules = VersificationRules.get_split_verse_rules()

        assert len(rules) >= 3

        # Check rule types
        types = [r["type"] for r in rules]
        assert "Hebrew poetry" in types
        assert "Long verses" in types
        assert "Manuscript variants" in types

        # Each rule should have examples
        for rule in rules:
            assert "examples" in rule
            assert len(rule["examples"]) > 0

    def test_get_canonical_rules(self):
        """Test canonical versification rules."""
        rules = VersificationRules.get_canonical_rules()

        assert "Base System" in rules
        assert "Protestant" in rules["Base System"]
        assert "66 books" in rules["Base System"]

        assert "Verse Parts" in rules
        assert "lowercase letters" in rules["Verse Parts"]

        assert "Missing Verses" in rules
        assert "Added Verses" in rules
        assert "Chapter Boundaries" in rules
        assert "Book Order" in rules
        assert "Apocrypha" in rules


class TestGetVersificationDocumentation:
    """Test the documentation generation function."""

    def test_documentation_structure(self):
        """Test that documentation has expected structure."""
        doc = get_versification_documentation()

        assert "# ABBA Versification Mapping Rules" in doc
        assert "## Overview" in doc
        assert "## Canonical Verse ID Format" in doc
        assert "## Major Versification Systems" in doc
        assert "## Handling Differences" in doc
        assert "## Implementation Guidelines" in doc
        assert "## Common Pitfalls" in doc

    def test_documentation_content(self):
        """Test that documentation includes key content."""
        doc = get_versification_documentation()

        # Versification systems
        assert "Masoretic Text (MT)" in doc
        assert "Septuagint (LXX)" in doc
        assert "Vulgate" in doc
        assert "King James Version (KJV)" in doc
        assert "Modern Critical" in doc

        # Difference types
        assert "Offset Differences" in doc
        assert "Split/Merge Differences" in doc
        assert "Missing/Added Verses" in doc
        assert "Chapter Boundary Differences" in doc

        # Examples
        assert "Psalms with titles" in doc
        assert "3 John 14-15" in doc
        assert "Mark 16:9-20" in doc
        assert "Malachi 3-4" in doc

    def test_documentation_guidelines(self):
        """Test that documentation includes implementation guidelines."""
        doc = get_versification_documentation()

        assert "Always store in canonical format" in doc
        assert "Preserve source information" in doc
        assert "Handle edge cases gracefully" in doc
        assert "Support round-trip conversion" in doc


class TestIntegration:
    """Integration tests for versification functionality."""

    def test_psalm_versification_workflow(self):
        """Test complete workflow for Psalm versification."""
        mapper = VersificationMapper()

        # Create a Psalm verse in MT system (with title as verse 1)
        # Use PSA.3 which has defined offset mapping
        mt_psalm = VerseID(book="PSA", chapter=3, verse=1)  # Actually the title

        # Map to Modern system - title verse doesn't exist in Modern system
        modern_verses = mapper.map_verse(
            mt_psalm, VersificationSystem.MT, VersificationSystem.MODERN
        )
        assert len(modern_verses) == 0  # Title verse maps to nothing in Modern system

        # Map actual content verse
        mt_content = VerseID(book="PSA", chapter=3, verse=2)  # First content verse in MT
        modern_content = mapper.map_verse(
            mt_content, VersificationSystem.MT, VersificationSystem.MODERN
        )
        assert len(modern_content) == 1
        assert modern_content[0].verse == 1  # Maps to verse 1 in Modern

    def test_translation_to_versification_mapping(self):
        """Test mapping from translation to versification system."""
        mapper = VersificationMapper()

        translations = {
            "KJV": VersificationSystem.KJV,
            "NKJV": VersificationSystem.KJV,
            "ESV": VersificationSystem.MODERN,
            "NIV": VersificationSystem.MODERN,
            "BHS": VersificationSystem.MT,
        }

        for trans, expected_system in translations.items():
            system = mapper.get_system_for_translation(trans)
            assert system == expected_system

    def test_round_trip_mapping(self):
        """Test that mappings can be reversed."""
        mapper = VersificationMapper()

        # Start with a verse in MT
        original = VerseID(book="PSA", chapter=3, verse=5)

        # Map to Modern
        modern = mapper.map_verse(original, VersificationSystem.MT, VersificationSystem.MODERN)
        assert len(modern) == 1

        # Map back to MT
        back_to_mt = mapper.map_verse(modern[0], VersificationSystem.MODERN, VersificationSystem.MT)
        assert len(back_to_mt) == 1
        assert back_to_mt[0] == original  # Should match original


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_book_mapping(self):
        """Test mapping with invalid book."""
        mapper = VersificationMapper()

        invalid_verse = VerseID(book="XXX", chapter=1, verse=1)
        result = mapper.map_verse(invalid_verse, VersificationSystem.MT, VersificationSystem.MODERN)
        assert len(result) == 1
        assert result[0] == invalid_verse  # No mapping, returns original

    def test_empty_string_input(self):
        """Test mapping with empty string."""
        mapper = VersificationMapper()

        result = mapper.map_verse("", VersificationSystem.MT, VersificationSystem.MODERN)
        assert len(result) == 0

    def test_none_values(self):
        """Test handling of None values."""
        mapper = VersificationMapper()

        # get_system_for_translation with None should use default
        system = mapper.get_system_for_translation("")
        assert system == VersificationSystem.MODERN
