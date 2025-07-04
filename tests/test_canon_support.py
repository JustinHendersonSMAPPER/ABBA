"""Unit tests for canon_support module."""

import pytest

from abba.alignment.canon_support import Canon, CanonManager
from abba.book_codes import BookCode


class TestCanon:
    """Test the Canon enum."""

    def test_canon_values(self) -> None:
        """Test canon enum values."""
        assert Canon.PROTESTANT.value == "protestant"
        assert Canon.CATHOLIC.value == "catholic"
        assert Canon.ORTHODOX.value == "orthodox"
        assert Canon.ETHIOPIAN.value == "ethiopian"


class TestCanonManager:
    """Test the CanonManager class."""

    def test_initialization(self) -> None:
        """Test CanonManager initialization."""
        manager = CanonManager()

        assert manager.canon_definitions is not None
        assert len(manager.canon_definitions) > 0

    def test_protestant_canon_books(self) -> None:
        """Test getting Protestant canon books."""
        manager = CanonManager()

        books = manager.get_canon_books(Canon.PROTESTANT)

        assert len(books) == 66
        assert "GEN" in books  # OT book
        assert "MAT" in books  # NT book
        assert "REV" in books  # Last NT book

        # Should not contain deuterocanonical books (they don't exist in BookCode enum)
        # This is correct - Protestant canon only has 66 books

    def test_catholic_canon_books(self) -> None:
        """Test getting Catholic canon books."""
        manager = CanonManager()

        books = manager.get_canon_books(Canon.CATHOLIC)

        # Catholic canon includes deuterocanonical books
        assert len(books) == 71  # 66 Protestant + 5 deutero

        # Should contain all Protestant books
        protestant_books = manager.get_canon_books(Canon.PROTESTANT)
        for book in protestant_books:
            assert book in books

        # Check for deuterocanonical books
        assert "TOB" in books
        assert "JDT" in books
        assert "WIS" in books
        assert "SIR" in books
        assert "BAR" in books

    def test_orthodox_canon_books(self) -> None:
        """Test getting Orthodox canon books."""
        manager = CanonManager()

        books = manager.get_canon_books(Canon.ORTHODOX)

        assert len(books) == 73  # 66 Protestant + 5 Catholic deutero + 2 Orthodox (1ES, 2ES)

        # Should contain Catholic books
        catholic_books = manager.get_canon_books(Canon.CATHOLIC)
        for book in catholic_books:
            assert book in books

        # Check for Orthodox-specific books
        assert "1ES" in books
        assert "2ES" in books

    def test_ethiopian_canon_books(self) -> None:
        """Test getting Ethiopian canon books."""
        manager = CanonManager()

        books = manager.get_canon_books(Canon.ETHIOPIAN)

        assert len(books) >= 66  # At least Protestant books

        # Should contain most standard books
        assert "GEN" in books
        assert "REV" in books

    def test_check_book_in_canon_protestant(self) -> None:
        """Test checking if book is in Protestant canon."""
        manager = CanonManager()

        # Standard Protestant books
        assert manager.is_book_in_canon("GEN", Canon.PROTESTANT)
        assert manager.is_book_in_canon("MAT", Canon.PROTESTANT)
        assert manager.is_book_in_canon("REV", Canon.PROTESTANT)

        # Since deuterocanonical books aren't defined in BookCode, we can't test them
        # But we can test that Protestant books are properly included

    def test_check_book_in_canon_catholic(self) -> None:
        """Test checking if book is in Catholic canon."""
        manager = CanonManager()

        # Protestant books should be in Catholic canon
        assert manager.is_book_in_canon("GEN", Canon.CATHOLIC)
        assert manager.is_book_in_canon("MAT", Canon.CATHOLIC)

        # Since deuterocanonical books aren't in BookCode enum,
        # Catholic canon will only contain Protestant books in this implementation

    def test_get_canon_statistics(self) -> None:
        """Test getting canon statistics."""
        manager = CanonManager()

        stats = manager.get_canon_statistics(Canon.PROTESTANT)

        assert isinstance(stats, dict)
        assert "total_books" in stats
        assert "ot_books" in stats
        assert "nt_books" in stats
        assert stats["total_books"] == 66
        assert stats["ot_books"] == 39
        assert stats["nt_books"] == 27

    def test_get_canon_statistics_catholic(self) -> None:
        """Test getting Catholic canon statistics."""
        manager = CanonManager()

        stats = manager.get_canon_statistics(Canon.CATHOLIC)

        # Catholic has Protestant books plus deuterocanonical
        assert stats["total_books"] == 71  # 66 Protestant + 5 deutero (TOB, JDT, WIS, SIR, BAR)
        assert stats["ot_books"] == 44  # 39 Protestant OT + 5 deutero
        assert stats["nt_books"] == 27
        assert stats["includes_deuterocanonical"] is True

    def test_compare_canons(self) -> None:
        """Test comparing different canons."""
        manager = CanonManager()

        comparison = manager.compare_canons(Canon.PROTESTANT, Canon.CATHOLIC)

        assert isinstance(comparison, dict)
        assert "common_books" in comparison
        assert "unique_to_first" in comparison
        assert "unique_to_second" in comparison
        assert "total_first" in comparison
        assert "total_second" in comparison

        # Protestant has 66, Catholic has 71 (66 + 5 deutero)
        assert len(comparison["common_books"]) == 66

        # Protestant has no unique books, Catholic has 5 deuterocanonical
        assert len(comparison["unique_to_first"]) == 0
        assert len(comparison["unique_to_second"]) == 5  # TOB, JDT, WIS, SIR, BAR

    def test_compare_protestant_orthodox(self) -> None:
        """Test comparing Protestant and Orthodox canons."""
        manager = CanonManager()

        comparison = manager.compare_canons(Canon.PROTESTANT, Canon.ORTHODOX)

        # Orthodox has Catholic books plus 1ES and 2ES
        assert len(comparison["common_books"]) == 66
        assert len(comparison["unique_to_first"]) == 0
        assert len(comparison["unique_to_second"]) == 7  # 5 Catholic deutero + 1ES + 2ES

    def test_get_supported_canons(self) -> None:
        """Test getting list of supported canons."""
        manager = CanonManager()

        canons = manager.get_supported_canons()

        assert isinstance(canons, list)
        assert Canon.PROTESTANT in canons
        assert Canon.CATHOLIC in canons
        assert Canon.ORTHODOX in canons
        assert Canon.ETHIOPIAN in canons

    def test_get_books_by_testament_protestant(self) -> None:
        """Test getting books by testament for Protestant canon."""
        manager = CanonManager()

        ot_books = manager.get_books_by_testament(Canon.PROTESTANT, "ot")
        nt_books = manager.get_books_by_testament(Canon.PROTESTANT, "nt")

        assert len(ot_books) == 39
        assert len(nt_books) == 27

        assert "GEN" in ot_books
        assert "MAL" in ot_books
        assert "MAT" in nt_books
        assert "REV" in nt_books

        # No overlap between OT and NT
        assert len(set(ot_books) & set(nt_books)) == 0

    def test_get_books_by_testament_catholic(self) -> None:
        """Test getting books by testament for Catholic canon."""
        manager = CanonManager()

        ot_books = manager.get_books_by_testament(Canon.CATHOLIC, "ot")
        nt_books = manager.get_books_by_testament(Canon.CATHOLIC, "nt")

        assert len(ot_books) == 44  # 39 Protestant OT + 5 deuterocanonical
        assert len(nt_books) == 27  # Same as Protestant NT

        # Check deuterocanonical books are in OT
        assert "TOB" in ot_books
        assert "JDT" in ot_books

    def test_get_books_by_testament_invalid_testament(self) -> None:
        """Test getting books by testament with invalid testament."""
        manager = CanonManager()

        books = manager.get_books_by_testament(Canon.PROTESTANT, "invalid")

        assert books == []

    def test_validate_canon_coverage(self) -> None:
        """Test validating canon coverage."""
        manager = CanonManager()

        # Test with complete Protestant coverage
        protestant_books = manager.get_canon_books(Canon.PROTESTANT)
        validation = manager.validate_canon_coverage(Canon.PROTESTANT, protestant_books)

        assert validation["is_complete"]
        assert validation["missing_books"] == []
        assert validation["extra_books"] == []
        assert validation["coverage_percentage"] == 1.0

    def test_validate_canon_coverage_incomplete(self) -> None:
        """Test validating incomplete canon coverage."""
        manager = CanonManager()

        # Test with partial coverage (missing some books)
        partial_books = [BookCode.GEN, BookCode.MAT, BookCode.REV]
        validation = manager.validate_canon_coverage(Canon.PROTESTANT, partial_books)

        assert not validation["is_complete"]
        assert len(validation["missing_books"]) == 63  # 66 - 3
        assert validation["extra_books"] == []
        assert validation["coverage_percentage"] < 1.0

    def test_validate_canon_coverage_with_extra_books(self) -> None:
        """Test validating canon coverage with extra books."""
        manager = CanonManager()

        # Test with Protestant books plus some extra non-existent book codes
        protestant_books = manager.get_canon_books(Canon.PROTESTANT)
        books_with_extra = protestant_books + ["TOB", "WIS"]  # String codes, not BookCode enum

        validation = manager.validate_canon_coverage(Canon.PROTESTANT, books_with_extra)

        assert validation["is_complete"]  # Has all required books
        assert validation["missing_books"] == []
        assert len(validation["extra_books"]) == 2
        assert "TOB" in validation["extra_books"]
        assert "WIS" in validation["extra_books"]

    def test_canon_book_ordering(self) -> None:
        """Test that canon books are returned in proper order."""
        manager = CanonManager()

        books = manager.get_canon_books(Canon.PROTESTANT)

        # First book should be Genesis
        assert books[0] == BookCode.GEN

        # Last book should be Revelation
        assert books[-1] == BookCode.REV

        # Matthew should come before Mark
        mat_index = books.index(BookCode.MAT)
        mrk_index = books.index(BookCode.MRK)
        assert mat_index < mrk_index

    def test_deuterocanonical_book_identification(self) -> None:
        """Test identification of deuterocanonical books."""
        manager = CanonManager()

        # Since deuterocanonical books aren't in BookCode enum, test is limited
        # But we can test the deuterocanonical identification methods
        deutero_books = manager.get_deuterocanonical_books()

        # May be empty since books aren't in BookCode enum
        for book in deutero_books:
            assert manager.is_deuterocanonical(book)

    def test_canon_intersection_and_differences(self) -> None:
        """Test finding intersections and differences between canons."""
        manager = CanonManager()

        protestant_books = set(manager.get_canon_books(Canon.PROTESTANT))
        catholic_books = set(manager.get_canon_books(Canon.CATHOLIC))

        # Protestant and Catholic share the 66 core books
        intersection = protestant_books & catholic_books
        assert len(intersection) == 66

        # Catholic has 5 deuterocanonical books
        catholic_only = catholic_books - protestant_books
        assert len(catholic_only) == 5
        assert catholic_only == {"TOB", "JDT", "WIS", "SIR", "BAR"}

    def test_get_canon_description(self) -> None:
        """Test getting canon descriptions."""
        manager = CanonManager()

        # Test that we can get some kind of description for each canon
        for canon in Canon:
            stats = manager.get_canon_statistics(canon)
            assert "total_books" in stats
            assert stats["total_books"] > 0
