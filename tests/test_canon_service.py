"""Tests for canon service."""

import pytest
import json
from pathlib import Path
import tempfile
from abba.canon.service import CanonService
from abba.canon.models import CanonTradition, LicenseType, TranslationPhilosophy, MappingType
from abba.verse_id import VerseID


class TestCanonService:
    """Test CanonService functionality."""

    @pytest.fixture
    def service(self):
        """Create a canon service for testing."""
        return CanonService()

    def test_service_initialization(self, service):
        """Test service initializes with all components."""
        assert service.registry is not None
        assert service.versification is not None
        assert service.translations is not None
        assert service.comparator is not None

    # Canon Operations Tests

    def test_get_canon(self, service):
        """Test getting a canon by ID."""
        protestant = service.get_canon("protestant")
        assert protestant is not None
        assert protestant.name == "Protestant Canon"
        assert protestant.book_count == 66

        assert service.get_canon("invalid") is None

    def test_list_canons(self, service):
        """Test listing canons."""
        # All canons
        all_canons = service.list_canons()
        assert len(all_canons) >= 4  # At least the default ones

        # Filter by tradition
        protestant_canons = service.list_canons(CanonTradition.PROTESTANT)
        assert len(protestant_canons) >= 1
        assert all(c.tradition == CanonTradition.PROTESTANT for c in protestant_canons)

    def test_get_canon_for_translation(self, service):
        """Test getting canon used by a translation."""
        canon = service.get_canon_for_translation("kjv")
        assert canon is not None
        assert canon.id == "protestant"

        canon = service.get_canon_for_translation("drb")
        assert canon is not None
        assert canon.id == "catholic"

    # Book Operations Tests

    def test_get_book_support(self, service):
        """Test checking which canons support a book."""
        # Genesis should be in all canons
        gen_support = service.get_book_support("GEN")
        assert all(gen_support.values())  # All True

        # Tobit should not be in Protestant
        tob_support = service.get_book_support("TOB")
        assert not tob_support["protestant"]
        assert tob_support["catholic"]
        assert tob_support["eastern_orthodox"]

    def test_get_universal_books(self, service):
        """Test getting books in all canons."""
        universal = service.get_universal_books()
        assert len(universal) == 66  # Protestant canon is the common subset
        assert "GEN" in universal
        assert "REV" in universal
        assert "TOB" not in universal  # Not in Protestant

    def test_get_tradition_specific_books(self, service):
        """Test getting books unique to a tradition."""
        # Ethiopian Orthodox should have unique books
        ethiopian_unique = service.get_tradition_specific_books(CanonTradition.ETHIOPIAN_ORTHODOX)
        assert len(ethiopian_unique) > 0
        assert "ENO" in ethiopian_unique  # 1 Enoch
        assert "JUB" in ethiopian_unique  # Jubilees

    # Translation Operations Tests

    def test_get_translation(self, service):
        """Test getting a translation."""
        kjv = service.get_translation("kjv")
        assert kjv is not None
        assert kjv.name == "King James Version"

    def test_find_translations(self, service):
        """Test finding translations by criteria."""
        # English translations
        english = service.find_translations(language="en")
        assert len(english) > 5
        assert all(t.language_code == "en" for t in english)

        # Digital Protestant translations
        digital_protestant = service.find_translations(canon="protestant", digital_only=True)
        assert len(digital_protestant) > 0
        assert all(t.allows_digital_use() for t in digital_protestant)
        assert all(t.canon_id == "protestant" for t in digital_protestant)

    def test_get_translation_options(self, service):
        """Test getting translation options for a verse."""
        verse = VerseID("GEN", 1, 1)

        # All translations should have Genesis
        options = service.get_translation_options(verse)
        assert len(options) > 10

        # Filter by language
        english_options = service.get_translation_options(verse, "en")
        assert len(english_options) > 5
        assert all(t.language_code == "en" for t in english_options)

        # Deuterocanonical book
        tobit_verse = VerseID("TOB", 1, 1)
        tobit_options = service.get_translation_options(tobit_verse)

        # Should not include Protestant translations
        protestant_trans = [t for t in tobit_options if t.canon_id == "protestant"]
        assert len(protestant_trans) == 0

    # Versification Operations Tests

    def test_map_verse(self, service):
        """Test mapping verses between translations."""
        verse = VerseID("PSA", 51, 1)

        # Map between translations with different versification
        # Assuming we have translations with different schemes
        result = service.map_verse(verse, "kjv", "drb")

        assert result.success
        # KJV uses standard, DRB uses vulgate versification
        # The actual mapping depends on the versification schemes

    def test_map_verse_invalid_translation(self, service):
        """Test mapping with invalid translation."""
        verse = VerseID("GEN", 1, 1)
        result = service.map_verse(verse, "invalid1", "invalid2")

        assert not result.success
        assert result.mapping_type == MappingType.NULL_MAPPING
        assert "Translation not found" in result.notes

    def test_check_verse_existence(self, service):
        """Test checking if verse exists in canon."""
        # Genesis 1:1 should exist in all canons
        assert service.check_verse_existence(VerseID("GEN", 1, 1), "protestant")
        assert service.check_verse_existence(VerseID("GEN", 1, 1), "catholic")

        # Tobit should not exist in Protestant
        assert not service.check_verse_existence(VerseID("TOB", 1, 1), "protestant")
        assert service.check_verse_existence(VerseID("TOB", 1, 1), "catholic")

        # Invalid canon
        assert not service.check_verse_existence(VerseID("GEN", 1, 1), "invalid")

    # Comparison Operations Tests

    def test_compare_canons_via_service(self, service):
        """Test comparing canons through service."""
        result = service.compare_canons("protestant", "catholic")

        assert result is not None
        assert len(result.common_books) == 66
        assert len(result.second_only_books) == 7

    def test_find_optimal_canon(self, service):
        """Test finding optimal canon for book set."""
        # Books common to all
        common_books = ["GEN", "EXO", "PSA", "MAT", "ROM"]
        canon = service.find_optimal_canon(common_books)
        assert canon is not None  # Any canon should work

        # Books including deuterocanonical
        with_deutero = ["GEN", "TOB", "WIS", "MAT"]
        canon = service.find_optimal_canon(with_deutero)
        assert canon is not None
        assert canon.id != "protestant"  # Protestant doesn't have TOB/WIS

        # Books that no canon has (hypothetical)
        impossible = ["GEN", "XXX", "YYY"]
        canon = service.find_optimal_canon(impossible)
        assert canon is None

    # Analysis Operations Tests

    def test_analyze_book_coverage(self, service):
        """Test analyzing book coverage."""
        # Analyze Tobit
        analysis = service.analyze_book_coverage("TOB")

        assert analysis["book_id"] == "TOB"

        # Canon coverage
        assert "protestant" not in analysis["canon_coverage"]
        assert "catholic" in analysis["canon_coverage"]

        # Tradition coverage
        assert not analysis["tradition_coverage"]["protestant"]
        assert analysis["tradition_coverage"]["catholic"]

        # Translation coverage
        assert analysis["translation_coverage"]["total"] > 0
        assert "en" in analysis["translation_coverage"]["by_language"]

        # Classification
        assert "deuterocanonical" in analysis["classifications"]

    def test_get_canon_statistics(self, service):
        """Test getting canon statistics."""
        stats = service.get_canon_statistics()

        assert stats["total_canons"] >= 4
        assert stats["total_unique_books"] > 66
        assert stats["versification_schemes"] >= 4
        assert stats["total_translations"] > 10

        assert "protestant" in stats["by_tradition"]
        assert "catholic" in stats["by_tradition"]

        assert stats["most_inclusive"] == "Ethiopian Orthodox Canon"
        assert stats["most_restrictive"] == "Protestant Canon"

    # Export/Import Operations Tests

    def test_export_canon_data(self, service):
        """Test exporting canon data."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            service.export_canon_data(temp_path)

            # Load and verify
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "canons" in data
            assert "versification_schemes" in data
            assert "statistics" in data

            # Check canon data
            assert "protestant" in data["canons"]
            protestant_data = data["canons"]["protestant"]
            assert protestant_data["book_count"] == 66
            assert len(protestant_data["books"]) == 66

            # Check versification data
            assert "standard" in data["versification_schemes"]

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_canon_report(self, service):
        """Test generating canon report."""
        report = service.generate_canon_report("protestant")

        assert "Canon Report: Protestant Canon" in report
        assert "Total Books: 66" in report
        assert "Old Testament" in report
        assert "New Testament" in report
        assert "Genesis (GEN)" in report
        assert "Revelation (REV)" in report

        # Should not have comparison with itself
        assert "Comparison with Protestant Canon" not in report

        # Catholic report should have comparison
        catholic_report = service.generate_canon_report("catholic")
        assert "Comparison with Protestant Canon" in catholic_report
        assert "Books unique to Catholic Canon:" in catholic_report
        assert "TOB" in catholic_report

    def test_invalid_canon_report(self, service):
        """Test generating report for invalid canon."""
        report = service.generate_canon_report("invalid")
        assert report == "Canon 'invalid' not found"

    def test_caching_behavior(self, service):
        """Test that service caches computed values."""
        # First call
        support1 = service.get_book_support("GEN")

        # Second call should use cache
        support2 = service.get_book_support("GEN")

        # Should be the same object
        assert support1 is support2

        # Test verse existence caching
        exists1 = service.check_verse_existence(VerseID("GEN", 1, 1), "protestant")
        exists2 = service.check_verse_existence(VerseID("GEN", 1, 1), "protestant")

        assert exists1 == exists2
