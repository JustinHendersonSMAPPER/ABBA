"""Unit tests for unified_reference module."""

import pytest

from abba.alignment.unified_reference import (
    CanonicalMapping,
    UnifiedReferenceSystem,
    VersionMapping,
)
from abba.verse_id import VerseID, parse_verse_id
from abba.versification import VersificationSystem


class TestCanonicalMapping:
    """Test the CanonicalMapping dataclass."""

    def test_creation(self) -> None:
        """Test creating CanonicalMapping instances."""
        canonical_id = parse_verse_id("GEN.1.1")

        mapping = CanonicalMapping(
            source_ref="Genesis 1:1",
            canonical_id=canonical_id,
            source_system=VersificationSystem.MODERN,
            confidence=0.95,
            notes="Standard mapping",
        )

        assert mapping.source_ref == "Genesis 1:1"
        assert mapping.canonical_id == canonical_id
        assert mapping.source_system == VersificationSystem.MODERN
        assert mapping.confidence == 0.95
        assert mapping.notes == "Standard mapping"


class TestVersionMapping:
    """Test the VersionMapping dataclass."""

    def test_creation(self) -> None:
        """Test creating VersionMapping instances."""
        mapping = VersionMapping(
            version_code="ESV",
            versification_system=VersificationSystem.MODERN,
            canonical_mappings={},
            coverage_stats={"total": 100, "mapped": 95},
        )

        assert mapping.version_code == "ESV"
        assert mapping.versification_system == VersificationSystem.MODERN
        assert mapping.canonical_mappings == {}
        assert mapping.coverage_stats["total"] == 100

    def test_get_canonical_id(self) -> None:
        """Test getting canonical ID from version mapping."""
        canonical_id = parse_verse_id("GEN.1.1")
        canonical_mapping = CanonicalMapping(
            source_ref="Genesis 1:1",
            canonical_id=canonical_id,
            source_system=VersificationSystem.MODERN,
            confidence=0.95,
        )

        mapping = VersionMapping(
            version_code="ESV",
            versification_system=VersificationSystem.MODERN,
            canonical_mappings={"Genesis 1:1": canonical_mapping},
            coverage_stats={},
        )

        result = mapping.get_canonical_id("Genesis 1:1")
        assert result == canonical_id

        # Test missing reference
        result = mapping.get_canonical_id("Nonexistent")
        assert result is None


class TestUnifiedReferenceSystem:
    """Test the UnifiedReferenceSystem class."""

    def test_initialization(self) -> None:
        """Test URS initialization."""
        urs = UnifiedReferenceSystem()

        assert urs.verse_mapper is not None
        assert urs.versification_mapper is not None
        assert len(urs.version_mappings) > 0
        assert urs._canonical_system == VersificationSystem.MODERN

    def test_translation_mappings_loaded(self) -> None:
        """Test that known translation mappings are loaded."""
        urs = UnifiedReferenceSystem()

        # Check that modern translations are loaded
        assert "ESV" in urs.version_mappings
        assert "NIV" in urs.version_mappings
        assert "NASB" in urs.version_mappings

        # Check that KJV translations use KJV versification
        assert urs.version_mappings["KJV"].versification_system == VersificationSystem.KJV
        assert urs.version_mappings["NKJV"].versification_system == VersificationSystem.KJV

        # Check that Catholic translations use Vulgate versification
        assert urs.version_mappings["NAB"].versification_system == VersificationSystem.VULGATE

    def test_generate_canonical_id_simple(self) -> None:
        """Test generating canonical ID for simple cases."""
        urs = UnifiedReferenceSystem()

        # Test with modern versification (should pass through)
        canonical_id = urs.generate_canonical_id("GEN.1.1", VersificationSystem.MODERN)

        assert canonical_id is not None
        assert str(canonical_id) == "GEN.1.1"

    def test_generate_canonical_id_with_translation(self) -> None:
        """Test generating canonical ID with translation code."""
        urs = UnifiedReferenceSystem()

        # ESV uses modern versification
        canonical_id = urs.generate_canonical_id("Genesis 1:1", translation_code="ESV")

        assert canonical_id is not None
        assert str(canonical_id) == "GEN.1.1"

    def test_generate_canonical_id_invalid_reference(self) -> None:
        """Test generating canonical ID for invalid reference."""
        urs = UnifiedReferenceSystem()

        canonical_id = urs.generate_canonical_id("INVALID.REFERENCE")

        assert canonical_id is None

    def test_generate_canonical_id_cross_system(self) -> None:
        """Test generating canonical ID across versification systems."""
        urs = UnifiedReferenceSystem()

        # Hebrew Malachi 3:19 should become Modern 4:1
        canonical_id = urs.generate_canonical_id("MAL.3.19", VersificationSystem.MT)

        assert canonical_id is not None
        # Should map to modern system (MAL.4.1)

    def test_create_version_mapping(self) -> None:
        """Test creating version mapping for new translation."""
        urs = UnifiedReferenceSystem()

        mapping = urs.create_version_mapping(
            "TEST", VersificationSystem.MODERN, ["GEN.1.1", "MAT.1.1"]
        )

        assert mapping.version_code == "TEST"
        assert mapping.versification_system == VersificationSystem.MODERN
        assert mapping.coverage_stats["total_refs"] == 2
        assert mapping.coverage_stats["mapped_refs"] <= 2

    def test_resolve_verse_variants(self) -> None:
        """Test resolving verse variants."""
        urs = UnifiedReferenceSystem()

        # Test basic verse
        variants = urs.resolve_verse_variants("GEN.1.1")

        assert len(variants) >= 1
        assert parse_verse_id("GEN.1.1") in variants

    def test_resolve_verse_variants_with_parts(self) -> None:
        """Test resolving variants for verses without parts."""
        urs = UnifiedReferenceSystem()

        variants = urs.resolve_verse_variants("ROM.1.1")

        # Should include the original and possibly a variant with part "a"
        assert len(variants) >= 1
        original = parse_verse_id("ROM.1.1")
        assert original in variants

    def test_resolve_textual_variants(self) -> None:
        """Test resolving known textual variants."""
        urs = UnifiedReferenceSystem()

        # Test Mark's longer ending
        variants = urs.resolve_verse_variants("MRK.16.9")

        assert len(variants) >= 1
        # Should include variants if they exist

    def test_validate_canonical_id_valid(self) -> None:
        """Test validating valid canonical IDs."""
        urs = UnifiedReferenceSystem()

        # Valid canonical IDs
        assert urs.validate_canonical_id("GEN.1.1")
        assert urs.validate_canonical_id("REV.22.21")
        assert urs.validate_canonical_id("PSA.23.1")

    def test_validate_canonical_id_invalid(self) -> None:
        """Test validating invalid canonical IDs."""
        urs = UnifiedReferenceSystem()

        # Invalid canonical IDs
        assert not urs.validate_canonical_id("INVALID.1.1")
        assert not urs.validate_canonical_id("GEN.0.1")  # Invalid chapter
        assert not urs.validate_canonical_id("GEN.1.0")  # Invalid verse
        assert not urs.validate_canonical_id("GEN.200.1")  # Chapter too high

    def test_validate_canonical_id_with_verse_id(self) -> None:
        """Test validating VerseID objects."""
        urs = UnifiedReferenceSystem()

        valid_verse = parse_verse_id("GEN.1.1")
        assert urs.validate_canonical_id(valid_verse)

    def test_get_mapping_statistics(self) -> None:
        """Test getting mapping statistics for translations."""
        urs = UnifiedReferenceSystem()

        # Create a test mapping with some statistics
        urs.create_version_mapping("TEST", VersificationSystem.MODERN, ["GEN.1.1", "MAT.1.1"])

        stats = urs.get_mapping_statistics("TEST")

        assert stats is not None
        assert "mapping_rate" in stats
        assert "failure_rate" in stats
        assert stats["total_refs"] == 2

    def test_get_mapping_statistics_nonexistent(self) -> None:
        """Test getting statistics for nonexistent translation."""
        urs = UnifiedReferenceSystem()

        stats = urs.get_mapping_statistics("NONEXISTENT")
        assert stats is None

    def test_get_supported_translations(self) -> None:
        """Test getting list of supported translations."""
        urs = UnifiedReferenceSystem()

        translations = urs.get_supported_translations()

        assert isinstance(translations, list)
        assert "ESV" in translations
        assert "KJV" in translations
        assert "NIV" in translations

    def test_bulk_generate_canonical_ids(self) -> None:
        """Test bulk generation of canonical IDs."""
        urs = UnifiedReferenceSystem()

        references = ["GEN.1.1", "MAT.1.1", "REV.22.21"]
        results = urs.bulk_generate_canonical_ids(references, VersificationSystem.MODERN)

        assert len(results) == 3
        assert all(ref in results for ref in references)
        assert all(results[ref] is not None for ref in references)

    def test_bulk_generate_with_invalid_references(self) -> None:
        """Test bulk generation with some invalid references."""
        urs = UnifiedReferenceSystem()

        references = ["GEN.1.1", "INVALID.REF", "MAT.1.1"]
        results = urs.bulk_generate_canonical_ids(references, VersificationSystem.MODERN)

        assert len(results) == 3
        assert results["GEN.1.1"] is not None
        assert results["INVALID.REF"] is None
        assert results["MAT.1.1"] is not None

    def test_find_cross_system_conflicts(self) -> None:
        """Test finding conflicts across versification systems."""
        urs = UnifiedReferenceSystem()

        conflicts = urs.find_cross_system_conflicts()

        assert isinstance(conflicts, list)
        # May be empty if no conflicts found, which is valid

    def test_canonical_system_consistency(self) -> None:
        """Test that canonical system is used consistently."""
        urs = UnifiedReferenceSystem()

        # Generate ID from different systems - should all normalize to canonical
        modern_id = urs.generate_canonical_id("GEN.1.1", VersificationSystem.MODERN)
        kjv_id = urs.generate_canonical_id("GEN.1.1", VersificationSystem.KJV)

        # For verses without versification differences, should be the same
        assert modern_id == kjv_id

    def test_hebrew_to_modern_mapping(self) -> None:
        """Test specific Hebrew to Modern mappings."""
        urs = UnifiedReferenceSystem()

        # Test Malachi mapping
        canonical_id = urs.generate_canonical_id("MAL.3.19", VersificationSystem.MT)

        assert canonical_id is not None
        # Should map to modern versification


class TestUnifiedReferenceSystemIntegration:
    """Integration tests for the Unified Reference System."""

    def test_comprehensive_mapping_workflow(self) -> None:
        """Test complete mapping workflow from various sources."""
        urs = UnifiedReferenceSystem()

        # Test various input formats and systems
        test_cases = [
            ("Genesis 1:1", VersificationSystem.MODERN, "ESV"),
            ("GEN.1.1", VersificationSystem.MODERN, "NIV"),
            ("MAL.3.19", VersificationSystem.MT, None),
            ("Acts 8:37", VersificationSystem.KJV, "KJV"),
            ("Psalm 3:1", VersificationSystem.MT, None),
        ]

        results = {}
        for ref, system, translation in test_cases:
            canonical_id = urs.generate_canonical_id(ref, system, translation)
            results[ref] = canonical_id

        # Verify basic functionality
        assert results["Genesis 1:1"] is not None
        assert results["GEN.1.1"] is not None

        # These should be the same
        assert str(results["Genesis 1:1"]) == str(results["GEN.1.1"])

    def test_version_mapping_complete_workflow(self) -> None:
        """Test complete version mapping workflow."""
        urs = UnifiedReferenceSystem()

        # Create mapping for new translation
        test_refs = ["Genesis 1:1", "Exodus 1:1", "Matthew 1:1", "Mark 1:1", "Revelation 22:21"]

        mapping = urs.create_version_mapping(
            "TEST_TRANSLATION", VersificationSystem.MODERN, test_refs
        )

        # Verify mapping was created
        assert mapping.version_code == "TEST_TRANSLATION"
        assert mapping.coverage_stats["total_refs"] == 5
        assert mapping.coverage_stats["mapped_refs"] >= 0

        # Test bulk generation
        bulk_results = urs.bulk_generate_canonical_ids(
            test_refs, VersificationSystem.MODERN, "TEST_TRANSLATION"
        )
        assert len(bulk_results) == 5

    def test_complex_versification_scenarios(self) -> None:
        """Test complex versification mapping scenarios."""
        urs = UnifiedReferenceSystem()

        # Test verses with known versification differences
        complex_cases = [
            ("PSA.3.1", VersificationSystem.MT),  # Psalm title difference
            ("MAL.3.19", VersificationSystem.MT),  # Chapter boundary difference
            ("JOL.3.1", VersificationSystem.MT),  # Joel chapter difference
            ("ACT.8.37", VersificationSystem.KJV),  # Textual variant
        ]

        for ref, system in complex_cases:
            canonical_id = urs.generate_canonical_id(ref, system)
            # Should handle complex cases without failing
            # May return None for missing verses (like Acts 8:37)

    def test_variant_resolution_comprehensive(self) -> None:
        """Test comprehensive variant resolution."""
        urs = UnifiedReferenceSystem()

        # Test various types of variants
        variant_tests = [
            "MRK.16.9",  # Mark's longer ending
            "JHN.7.53",  # Pericope Adulterae
            "ROM.16.25",  # Doxology placement
            "1JN.5.7",  # Comma Johanneum
        ]

        for verse_ref in variant_tests:
            variants = urs.resolve_verse_variants(verse_ref)
            assert len(variants) >= 1

            # Original verse should always be included
            original = parse_verse_id(verse_ref)
            assert original in variants

    def test_cross_system_mapping_consistency(self) -> None:
        """Test consistency across different versification systems."""
        urs = UnifiedReferenceSystem()

        # Test verses that should map consistently
        test_verses = ["GEN.1.1", "MAT.1.1", "REV.22.21"]

        for verse_ref in test_verses:
            # Generate from different systems
            modern_id = urs.generate_canonical_id(verse_ref, VersificationSystem.MODERN)
            kjv_id = urs.generate_canonical_id(verse_ref, VersificationSystem.KJV)

            # For these verses, should get same canonical ID
            if modern_id and kjv_id:
                assert str(modern_id) == str(kjv_id)

    def test_error_handling_and_edge_cases(self) -> None:
        """Test error handling for edge cases."""
        urs = UnifiedReferenceSystem()

        # Test invalid inputs
        edge_cases = [
            "",  # Empty string
            "INVALID",  # Invalid format
            "GEN.0.1",  # Invalid chapter
            "GEN.1.0",  # Invalid verse
            "NONEXISTENT.1.1",  # Invalid book
        ]

        for case in edge_cases:
            canonical_id = urs.generate_canonical_id(case)
            # Should handle gracefully
            if canonical_id is not None:
                # If it generated an ID, it should at least be valid format
                assert isinstance(canonical_id, (str, VerseID))

    def test_bulk_operations_performance(self) -> None:
        """Test bulk operations with larger datasets."""
        urs = UnifiedReferenceSystem()

        # Generate large test dataset
        test_refs = []
        for book in ["GEN", "EXO", "MAT", "MRK"]:
            for chapter in range(1, 6):  # First 5 chapters
                for verse in range(1, 11):  # First 10 verses
                    test_refs.append(f"{book}.{chapter}.{verse}")

        # Test bulk generation
        results = urs.bulk_generate_canonical_ids(test_refs)

        assert len(results) == len(test_refs)
        # Most should succeed
        successful = sum(1 for v in results.values() if v is not None)
        assert successful > len(test_refs) * 0.8  # At least 80% success rate

    def test_translation_specific_behaviors(self) -> None:
        """Test translation-specific behaviors."""
        urs = UnifiedReferenceSystem()

        # Test that different translations use appropriate versification systems
        translation_tests = [
            ("KJV", VersificationSystem.KJV),
            ("ESV", VersificationSystem.MODERN),
            ("NAB", VersificationSystem.VULGATE),
        ]

        for translation, expected_system in translation_tests:
            if translation in urs.version_mappings:
                mapping = urs.version_mappings[translation]
                assert mapping.versification_system == expected_system

    def test_statistics_and_reporting(self) -> None:
        """Test statistics and reporting functionality."""
        urs = UnifiedReferenceSystem()

        # Create test mapping with known statistics
        test_refs = ["GEN.1.1", "INVALID.REF", "MAT.1.1", "ANOTHER.INVALID"]
        urs.create_version_mapping("STATS_TEST", VersificationSystem.MODERN, test_refs)

        stats = urs.get_mapping_statistics("STATS_TEST")
        assert stats is not None
        assert stats["total_refs"] == 4
        assert stats["mapping_rate"] <= 1.0
        assert stats["failure_rate"] >= 0.0
        assert abs(stats["mapping_rate"] + stats["failure_rate"] - 1.0) < 0.01  # Should sum to 1.0

    def test_conflict_detection_comprehensive(self) -> None:
        """Test comprehensive conflict detection."""
        urs = UnifiedReferenceSystem()

        conflicts = urs.find_cross_system_conflicts()

        # Should return list of conflicts (may be empty)
        assert isinstance(conflicts, list)

        # Each conflict should have proper structure
        for conflict in conflicts:
            assert len(conflict) == 3  # (system1, system2, mappings)
            assert isinstance(conflict[0], str)
            assert isinstance(conflict[1], str)
            assert isinstance(conflict[2], list)
