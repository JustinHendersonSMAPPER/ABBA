"""Tests for canon models."""

import pytest
from abba.canon.models import (
    Canon,
    CanonBook,
    CanonTradition,
    BookClassification,
    BookSection,
    VersificationScheme,
    VerseMapping,
    MappingType,
    Translation,
    TranslationPhilosophy,
    LicenseType,
    CanonDifference,
)


class TestCanon:
    """Test Canon model."""

    def test_canon_creation(self):
        """Test creating a canon."""
        canon = Canon(
            id="test",
            name="Test Canon",
            tradition=CanonTradition.PROTESTANT,
            description="Test canon",
            book_count=66,
        )

        assert canon.id == "test"
        assert canon.name == "Test Canon"
        assert canon.tradition == CanonTradition.PROTESTANT
        assert canon.book_count == 66
        assert canon.books == []

    def test_canon_book_operations(self):
        """Test canon book operations."""
        canon = Canon(
            id="test",
            name="Test Canon",
            tradition=CanonTradition.PROTESTANT,
            description="Test canon",
            book_count=2,
        )

        # Add books
        canon.books.append(
            CanonBook(
                canon_id="test",
                book_id="GEN",
                order=1,
                section=BookSection.OLD_TESTAMENT,
                classification=BookClassification.PROTOCANONICAL,
                canonical_name="Genesis",
            )
        )

        canon.books.append(
            CanonBook(
                canon_id="test",
                book_id="MAT",
                order=2,
                section=BookSection.NEW_TESTAMENT,
                classification=BookClassification.PROTOCANONICAL,
                canonical_name="Matthew",
            )
        )

        # Test operations
        assert canon.has_book("GEN")
        assert canon.has_book("MAT")
        assert not canon.has_book("REV")

        assert canon.get_book_ids() == ["GEN", "MAT"]
        assert canon.get_book_order("GEN") == 1
        assert canon.get_book_order("MAT") == 2
        assert canon.get_book_order("REV") is None


class TestCanonBook:
    """Test CanonBook model."""

    def test_canon_book_creation(self):
        """Test creating a canon book."""
        book = CanonBook(
            canon_id="protestant",
            book_id="JOB",
            order=18,
            section=BookSection.OLD_TESTAMENT,
            classification=BookClassification.PROTOCANONICAL,
            canonical_name="Job",
        )

        assert book.canon_id == "protestant"
        assert book.book_id == "JOB"
        assert book.order == 18
        assert book.section == BookSection.OLD_TESTAMENT
        assert book.classification == BookClassification.PROTOCANONICAL

    def test_canon_book_sorting(self):
        """Test sorting canon books."""
        books = [
            CanonBook(
                "test", "MAT", 40, BookSection.NEW_TESTAMENT, BookClassification.PROTOCANONICAL
            ),
            CanonBook(
                "test", "GEN", 1, BookSection.OLD_TESTAMENT, BookClassification.PROTOCANONICAL
            ),
            CanonBook(
                "test", "PSA", 19, BookSection.OLD_TESTAMENT, BookClassification.PROTOCANONICAL
            ),
        ]

        sorted_books = sorted(books)
        assert [b.book_id for b in sorted_books] == ["GEN", "PSA", "MAT"]


class TestVersificationScheme:
    """Test VersificationScheme model."""

    def test_versification_scheme_creation(self):
        """Test creating a versification scheme."""
        scheme = VersificationScheme(
            id="lxx",
            name="Septuagint",
            description="Greek Septuagint versification",
            base_text="Septuagint",
            includes_apocrypha=True,
            differences={"psalms": "Different numbering after Psalm 9"},
        )

        assert scheme.id == "lxx"
        assert scheme.name == "Septuagint"
        assert scheme.includes_apocrypha is True
        assert "psalms" in scheme.differences


class TestVerseMapping:
    """Test VerseMapping model."""

    def test_verse_mapping_creation(self):
        """Test creating a verse mapping."""
        mapping = VerseMapping(
            source_scheme_id="masoretic",
            target_scheme_id="septuagint",
            mapping_type=MappingType.ONE_TO_ONE,
            source_book="PSA",
            source_chapter=51,
            source_verses=[1],
            target_book="PSA",
            target_chapter=50,
            target_verses=[3],
        )

        assert mapping.source_scheme_id == "masoretic"
        assert mapping.target_scheme_id == "septuagint"
        assert mapping.mapping_type == MappingType.ONE_TO_ONE

    def test_verse_mapping_references(self):
        """Test getting verse references."""
        mapping = VerseMapping(
            source_scheme_id="standard",
            target_scheme_id="vulgate",
            mapping_type=MappingType.ONE_TO_MANY,
            source_book="3JN",
            source_chapter=1,
            source_verses=[14],
            target_book="3JN",
            target_chapter=1,
            target_verses=[14, 15],
        )

        assert mapping.get_source_references() == ["3JN.1.14"]
        assert mapping.get_target_references() == ["3JN.1.14", "3JN.1.15"]

    def test_null_mapping_references(self):
        """Test null mapping references."""
        mapping = VerseMapping(
            source_scheme_id="septuagint",
            target_scheme_id="masoretic",
            mapping_type=MappingType.NULL_MAPPING,
            source_book="DAN",
            source_chapter=13,
            source_verses=[0],
            target_book="DAN",
            target_chapter=0,
            target_verses=[],
        )

        assert mapping.get_source_references() == ["DAN.13.0"]
        assert mapping.get_target_references() == []


class TestTranslation:
    """Test Translation model."""

    def test_translation_creation(self):
        """Test creating a translation."""
        translation = Translation(
            id="esv",
            name="English Standard Version",
            abbreviation="ESV",
            language_code="en",
            canon_id="protestant",
            versification_scheme_id="standard",
            philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
            year_published=2001,
            license_type=LicenseType.RESTRICTED,
        )

        assert translation.id == "esv"
        assert translation.abbreviation == "ESV"
        assert translation.philosophy == TranslationPhilosophy.FORMAL_EQUIVALENCE

    def test_translation_public_domain(self):
        """Test public domain check."""
        pd_translation = Translation(
            id="kjv",
            name="King James Version",
            abbreviation="KJV",
            language_code="en",
            canon_id="protestant",
            versification_scheme_id="standard",
            philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
            license_type=LicenseType.PUBLIC_DOMAIN,
        )

        assert pd_translation.is_public_domain()
        assert pd_translation.allows_digital_use()

        restricted = Translation(
            id="niv",
            name="New International Version",
            abbreviation="NIV",
            language_code="en",
            canon_id="protestant",
            versification_scheme_id="standard",
            philosophy=TranslationPhilosophy.DYNAMIC_EQUIVALENCE,
            license_type=LicenseType.RESTRICTED,
            digital_distribution=True,
        )

        assert not restricted.is_public_domain()
        assert restricted.allows_digital_use()

    def test_translation_attribution(self):
        """Test attribution text generation."""
        translation = Translation(
            id="esv",
            name="English Standard Version",
            abbreviation="ESV",
            language_code="en",
            canon_id="protestant",
            versification_scheme_id="standard",
            philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
            year_published=2001,
            copyright_holder="Crossway",
            license_type=LicenseType.RESTRICTED,
            attribution_required=True,
        )

        attribution = translation.get_attribution_text()
        assert "English Standard Version" in attribution
        assert "ESV" in attribution
        assert "Crossway" in attribution
        assert "2001" in attribution

        # Test no attribution required
        translation.attribution_required = False
        assert translation.get_attribution_text() == ""


class TestCanonDifference:
    """Test CanonDifference model."""

    def test_book_presence_difference(self):
        """Test book presence difference."""
        diff = CanonDifference(
            difference_type="book_presence",
            book_id="TOB",
            in_first_canon=False,
            in_second_canon=True,
            description="Tobit is in Catholic but not Protestant canon",
        )

        assert diff.difference_type == "book_presence"
        assert diff.book_id == "TOB"
        assert not diff.in_first_canon
        assert diff.in_second_canon

    def test_book_order_difference(self):
        """Test book order difference."""
        diff = CanonDifference(
            difference_type="book_order",
            book_id="MAL",
            first_canon_position=39,
            second_canon_position=46,
            description="Malachi: position 39 vs 46",
        )

        assert diff.difference_type == "book_order"
        assert diff.first_canon_position == 39
        assert diff.second_canon_position == 46

    def test_section_difference(self):
        """Test section difference."""
        diff = CanonDifference(
            difference_type="book_section",
            book_id="DAN",
            first_canon_section=BookSection.OLD_TESTAMENT,
            second_canon_section=BookSection.APOCRYPHA,
            description="Daniel: old_testament vs apocrypha",
        )

        assert diff.difference_type == "book_section"
        assert diff.first_canon_section == BookSection.OLD_TESTAMENT
        assert diff.second_canon_section == BookSection.APOCRYPHA
