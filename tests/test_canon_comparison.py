"""Tests for canon comparison tools."""

import pytest
from abba.canon.comparison import CanonComparator, ComparisonResult
from abba.canon.registry import CanonRegistry
from abba.canon.models import (
    Canon,
    CanonBook,
    CanonTradition,
    BookSection,
    BookClassification,
    CanonDifference,
)


class TestCanonComparator:
    """Test CanonComparator functionality."""

    @pytest.fixture
    def comparator(self):
        """Create a comparator with default registry."""
        return CanonComparator()

    @pytest.fixture
    def custom_registry(self):
        """Create a custom registry for testing."""
        registry = CanonRegistry()

        # Clear default canons for controlled testing
        registry._canons.clear()

        # Create simple test canons
        canon1 = Canon(
            id="test1",
            name="Test Canon 1",
            tradition=CanonTradition.PROTESTANT,
            description="First test canon",
            book_count=3,
        )
        canon1.books = [
            CanonBook(
                "test1",
                "GEN",
                1,
                BookSection.OLD_TESTAMENT,
                BookClassification.PROTOCANONICAL,
                "Genesis",
            ),
            CanonBook(
                "test1",
                "EXO",
                2,
                BookSection.OLD_TESTAMENT,
                BookClassification.PROTOCANONICAL,
                "Exodus",
            ),
            CanonBook(
                "test1",
                "MAT",
                3,
                BookSection.NEW_TESTAMENT,
                BookClassification.PROTOCANONICAL,
                "Matthew",
            ),
        ]

        canon2 = Canon(
            id="test2",
            name="Test Canon 2",
            tradition=CanonTradition.CATHOLIC,
            description="Second test canon",
            book_count=4,
        )
        canon2.books = [
            CanonBook(
                "test2",
                "GEN",
                1,
                BookSection.OLD_TESTAMENT,
                BookClassification.PROTOCANONICAL,
                "Genesis",
            ),
            CanonBook(
                "test2",
                "TOB",
                2,
                BookSection.OLD_TESTAMENT,
                BookClassification.DEUTEROCANONICAL,
                "Tobit",
            ),
            CanonBook(
                "test2",
                "EXO",
                3,
                BookSection.OLD_TESTAMENT,
                BookClassification.PROTOCANONICAL,
                "Exodus",
            ),
            CanonBook(
                "test2",
                "MAT",
                4,
                BookSection.NEW_TESTAMENT,
                BookClassification.PROTOCANONICAL,
                "Gospel of Matthew",
            ),
        ]

        registry.register_canon(canon1)
        registry.register_canon(canon2)

        return registry

    def test_compare_canons_basic(self, custom_registry):
        """Test basic canon comparison."""
        comparator = CanonComparator(custom_registry)
        result = comparator.compare_canons("test1", "test2")

        assert result is not None
        assert result.first_canon.id == "test1"
        assert result.second_canon.id == "test2"

        # Check book analysis
        assert result.common_books == {"GEN", "EXO", "MAT"}
        assert result.first_only_books == set()
        assert result.second_only_books == {"TOB"}

        # Check similarity score
        assert 0 <= result.similarity_score <= 1

    def test_compare_protestant_catholic(self, comparator):
        """Test comparing Protestant and Catholic canons."""
        result = comparator.compare_canons("protestant", "catholic")

        assert result is not None

        # Protestant has 66 books, all should be in Catholic
        assert len(result.common_books) == 66
        assert len(result.first_only_books) == 0

        # Catholic has 7 additional books
        assert len(result.second_only_books) == 7
        assert "TOB" in result.second_only_books  # Tobit
        assert "WIS" in result.second_only_books  # Wisdom
        assert "SIR" in result.second_only_books  # Sirach

    def test_comparison_summary(self, comparator):
        """Test comparison summary generation."""
        result = comparator.compare_canons("protestant", "catholic")
        summary = result.get_summary()

        assert "Protestant Canon vs Catholic Canon" in summary
        assert "Common books: 66" in summary
        assert "Unique to Catholic Canon: 7" in summary
        assert "Similarity:" in summary

    def test_order_differences(self, custom_registry):
        """Test finding order differences."""
        comparator = CanonComparator(custom_registry)
        result = comparator.compare_canons("test1", "test2")

        # EXO is at position 2 in test1 but 3 in test2
        order_diffs = result.order_differences
        assert len(order_diffs) > 0

        # Check that EXO has different position
        exo_diff = next((d for d in order_diffs if d.book_id == "EXO"), None)
        assert exo_diff is not None
        assert exo_diff.first_canon_position == 2
        assert exo_diff.second_canon_position == 3

    def test_name_differences(self, custom_registry):
        """Test finding name differences."""
        comparator = CanonComparator(custom_registry)
        result = comparator.compare_canons("test1", "test2")

        # MAT has different names: "Matthew" vs "Gospel of Matthew"
        name_diffs = result.name_differences
        mat_diff = next((d for d in name_diffs if d.book_id == "MAT"), None)
        assert mat_diff is not None
        assert mat_diff.first_canon_name == "Matthew"
        assert mat_diff.second_canon_name == "Gospel of Matthew"

    def test_find_book_journey(self, comparator):
        """Test tracing a book across canons."""
        # Test with Tobit (not in Protestant)
        journey = comparator.find_book_journey("TOB")

        assert "protestant" not in journey
        assert "catholic" in journey
        assert "eastern_orthodox" in journey

        catholic_info = journey["catholic"]
        assert catholic_info["tradition"] == "catholic"
        assert catholic_info["section"] == "old_testament"
        assert catholic_info["classification"] == "deuterocanonical"

    def test_compare_traditions(self, comparator):
        """Test comparing all canons between traditions."""
        results = comparator.compare_traditions(CanonTradition.PROTESTANT, CanonTradition.CATHOLIC)

        assert len(results) > 0

        # Should have comparison between Protestant and Catholic canons
        for result in results:
            assert result.first_canon.tradition == CanonTradition.PROTESTANT
            assert result.second_canon.tradition == CanonTradition.CATHOLIC

    def test_find_deuterocanonical_differences(self, comparator):
        """Test identifying deuterocanonical books."""
        deutero_books = comparator.find_deuterocanonical_differences()

        # Catholic should have deuterocanonical books
        assert "catholic" in deutero_books
        catholic_deutero = deutero_books["catholic"]
        assert len(catholic_deutero) == 7
        assert "TOB" in catholic_deutero
        assert "JDT" in catholic_deutero
        assert "WIS" in catholic_deutero

        # Protestant should have none
        assert "protestant" not in deutero_books or len(deutero_books.get("protestant", [])) == 0

    def test_generate_compatibility_matrix(self, comparator):
        """Test generating canon compatibility matrix."""
        # Test with subset of canons
        matrix = comparator.generate_compatibility_matrix(["protestant", "catholic"])

        assert "protestant" in matrix
        assert "catholic" in matrix

        # Self-comparison should be 1.0
        assert matrix["protestant"]["protestant"] == 1.0
        assert matrix["catholic"]["catholic"] == 1.0

        # Protestant-Catholic should have high similarity
        assert matrix["protestant"]["catholic"] > 0.75
        assert matrix["catholic"]["protestant"] > 0.75

    def test_find_most_inclusive_canon(self, comparator):
        """Test finding the most inclusive canon."""
        most_inclusive = comparator.find_most_inclusive_canon()

        assert most_inclusive is not None
        assert most_inclusive.id == "ethiopian_orthodox"  # Has 81 books
        assert most_inclusive.book_count == 81

    def test_get_book_classification_summary(self, comparator):
        """Test getting book classification summary."""
        summary = comparator.get_book_classification_summary()

        assert "protestant" in summary
        assert "catholic" in summary

        # Protestant should only have protocanonical
        protestant_summary = summary["protestant"]
        assert BookClassification.PROTOCANONICAL in protestant_summary
        assert protestant_summary[BookClassification.PROTOCANONICAL] == 66

        # Catholic should have both
        catholic_summary = summary["catholic"]
        assert BookClassification.PROTOCANONICAL in catholic_summary
        assert BookClassification.DEUTEROCANONICAL in catholic_summary
        assert catholic_summary[BookClassification.PROTOCANONICAL] == 66
        assert catholic_summary[BookClassification.DEUTEROCANONICAL] == 7

    def test_comparison_caching(self, comparator):
        """Test that comparisons are cached."""
        # First comparison
        result1 = comparator.compare_canons("protestant", "catholic")

        # Second comparison should use cache
        result2 = comparator.compare_canons("protestant", "catholic")

        # Should be the same object (cached)
        assert result1 is result2

    def test_invalid_canon_comparison(self, comparator):
        """Test comparing invalid canons."""
        result = comparator.compare_canons("protestant", "invalid")
        assert result is None

        result = comparator.compare_canons("invalid1", "invalid2")
        assert result is None

    def test_section_differences(self, comparator):
        """Test finding section differences between canons."""
        # Create custom comparator to test section differences
        registry = CanonRegistry()
        registry._canons.clear()

        # Canon where Daniel is in OT
        canon1 = Canon("test1", "Test 1", CanonTradition.PROTESTANT, "Test", 1)
        canon1.books = [
            CanonBook(
                "test1",
                "DAN",
                1,
                BookSection.OLD_TESTAMENT,
                BookClassification.PROTOCANONICAL,
                "Daniel",
            )
        ]

        # Canon where Daniel is in Apocrypha (hypothetical)
        canon2 = Canon("test2", "Test 2", CanonTradition.CATHOLIC, "Test", 1)
        canon2.books = [
            CanonBook(
                "test2",
                "DAN",
                1,
                BookSection.APOCRYPHA,
                BookClassification.PROTOCANONICAL,
                "Daniel",
            )
        ]

        registry.register_canon(canon1)
        registry.register_canon(canon2)

        comp = CanonComparator(registry)
        result = comp.compare_canons("test1", "test2")

        assert len(result.section_differences) == 1
        dan_diff = result.section_differences[0]
        assert dan_diff.book_id == "DAN"
        assert dan_diff.first_canon_section == BookSection.OLD_TESTAMENT
        assert dan_diff.second_canon_section == BookSection.APOCRYPHA
