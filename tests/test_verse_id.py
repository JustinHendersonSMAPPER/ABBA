"""Unit tests for verse_id module."""

import pytest

from abba import (
    VerseID,
    VerseRange,
    compare_verse_ids,
    create_verse_id,
    get_verse_parts,
    is_valid_verse_id,
    normalize_verse_id,
    parse_verse_id,
    parse_verse_range,
)


class TestVerseID:
    """Test the VerseID dataclass."""

    def test_creation(self):
        """Test creating VerseID instances."""
        verse = VerseID(book="GEN", chapter=1, verse=1)
        assert verse.book == "GEN"
        assert verse.chapter == 1
        assert verse.verse == 1
        assert verse.part is None

        verse_with_part = VerseID(book="ROM", chapter=3, verse=23, part="a")
        assert verse_with_part.part == "a"

    def test_string_representation(self):
        """Test string conversion of VerseID."""
        verse = VerseID(book="GEN", chapter=1, verse=1)
        assert str(verse) == "GEN.1.1"

        verse_with_part = VerseID(book="ROM", chapter=3, verse=23, part="a")
        assert str(verse_with_part) == "ROM.3.23a"

    def test_repr(self):
        """Test repr of VerseID."""
        verse = VerseID(book="GEN", chapter=1, verse=1)
        assert repr(verse) == "VerseID('GEN.1.1')"

    def test_equality(self):
        """Test VerseID equality comparison."""
        verse1 = VerseID(book="GEN", chapter=1, verse=1)
        verse2 = VerseID(book="GEN", chapter=1, verse=1)
        verse3 = VerseID(book="GEN", chapter=1, verse=2)

        assert verse1 == verse2
        assert verse1 != verse3
        assert verse1 != "GEN.1.1"  # Different type

    def test_ordering(self):
        """Test VerseID ordering/comparison."""
        gen11 = VerseID(book="GEN", chapter=1, verse=1)
        gen12 = VerseID(book="GEN", chapter=1, verse=2)
        gen21 = VerseID(book="GEN", chapter=2, verse=1)
        exo11 = VerseID(book="EXO", chapter=1, verse=1)

        # Basic comparisons
        assert gen11 < gen12
        assert gen12 < gen21
        assert gen21 < exo11

        # Reverse comparisons
        assert gen12 > gen11
        assert not gen11 > gen12

        # Equal comparisons
        assert gen11 <= gen11
        assert gen11 >= gen11

    def test_ordering_with_parts(self):
        """Test ordering with verse parts."""
        verse_base = VerseID(book="ROM", chapter=3, verse=23)
        verse_a = VerseID(book="ROM", chapter=3, verse=23, part="a")
        verse_b = VerseID(book="ROM", chapter=3, verse=23, part="b")

        assert verse_base < verse_a
        assert verse_a < verse_b
        assert not verse_b < verse_a

    def test_hash(self):
        """Test that VerseID is hashable."""
        verse1 = VerseID(book="GEN", chapter=1, verse=1)
        verse2 = VerseID(book="GEN", chapter=1, verse=1)
        verse3 = VerseID(book="GEN", chapter=1, verse=2)

        # Same verses should have same hash
        assert hash(verse1) == hash(verse2)

        # Can be used in sets/dicts
        verse_set = {verse1, verse2, verse3}
        assert len(verse_set) == 2  # verse1 and verse2 are same

    def test_to_dict(self):
        """Test converting VerseID to dictionary."""
        verse = VerseID(book="GEN", chapter=1, verse=1)
        d = verse.to_dict()
        assert d == {
            "canonical_id": "GEN.1.1",
            "book": "GEN",
            "chapter": 1,
            "verse": 1,
            "verse_part": None,
        }

        verse_with_part = VerseID(book="ROM", chapter=3, verse=23, part="a")
        d = verse_with_part.to_dict()
        assert d["verse_part"] == "a"

    def test_next_verse(self):
        """Test getting next verse."""
        verse = VerseID(book="GEN", chapter=1, verse=1)
        next_v = verse.next_verse()
        assert next_v == VerseID(book="GEN", chapter=1, verse=2)

        # With parts
        verse_a = VerseID(book="ROM", chapter=3, verse=23, part="a")
        next_v = verse_a.next_verse()
        assert next_v == VerseID(book="ROM", chapter=3, verse=23, part="b")

        # Part z should go to next verse number
        verse_z = VerseID(book="ROM", chapter=3, verse=23, part="z")
        next_v = verse_z.next_verse()
        assert next_v == VerseID(book="ROM", chapter=3, verse=24)

    def test_previous_verse(self):
        """Test getting previous verse."""
        verse = VerseID(book="GEN", chapter=1, verse=2)
        prev_v = verse.previous_verse()
        assert prev_v == VerseID(book="GEN", chapter=1, verse=1)

        # With parts
        verse_b = VerseID(book="ROM", chapter=3, verse=23, part="b")
        prev_v = verse_b.previous_verse()
        assert prev_v == VerseID(book="ROM", chapter=3, verse=23, part="a")

        verse_a = VerseID(book="ROM", chapter=3, verse=23, part="a")
        prev_v = verse_a.previous_verse()
        assert prev_v == VerseID(book="ROM", chapter=3, verse=23)

        # First verse of chapter
        verse = VerseID(book="GEN", chapter=2, verse=1)
        prev_v = verse.previous_verse()
        assert prev_v.chapter == 1

        # First verse of book
        verse = VerseID(book="GEN", chapter=1, verse=1)
        prev_v = verse.previous_verse()
        assert prev_v is None


class TestVerseRange:
    """Test the VerseRange dataclass."""

    def test_creation(self):
        """Test creating VerseRange instances."""
        start = VerseID(book="GEN", chapter=1, verse=1)
        end = VerseID(book="GEN", chapter=1, verse=5)
        range_obj = VerseRange(start=start, end=end)

        assert range_obj.start == start
        assert range_obj.end == end

    def test_string_representation(self):
        """Test string representation of VerseRange."""
        start = VerseID(book="GEN", chapter=1, verse=1)
        end = VerseID(book="GEN", chapter=1, verse=5)
        range_obj = VerseRange(start=start, end=end)

        assert str(range_obj) == "GEN.1.1-GEN.1.5"

    def test_contains(self):
        """Test checking if verse is in range."""
        start = VerseID(book="GEN", chapter=1, verse=1)
        end = VerseID(book="GEN", chapter=1, verse=5)
        range_obj = VerseRange(start=start, end=end)

        assert VerseID(book="GEN", chapter=1, verse=1) in range_obj
        assert VerseID(book="GEN", chapter=1, verse=3) in range_obj
        assert VerseID(book="GEN", chapter=1, verse=5) in range_obj
        assert VerseID(book="GEN", chapter=1, verse=6) not in range_obj
        assert VerseID(book="GEN", chapter=2, verse=1) not in range_obj

    def test_to_list(self):
        """Test expanding range to list of verses."""
        start = VerseID(book="GEN", chapter=1, verse=1)
        end = VerseID(book="GEN", chapter=1, verse=3)
        range_obj = VerseRange(start=start, end=end)

        verses = range_obj.to_list()
        assert len(verses) == 3
        assert verses[0] == VerseID(book="GEN", chapter=1, verse=1)
        assert verses[1] == VerseID(book="GEN", chapter=1, verse=2)
        assert verses[2] == VerseID(book="GEN", chapter=1, verse=3)


class TestParseVerseID:
    """Test parse_verse_id function."""

    def test_canonical_format(self):
        """Test parsing canonical format."""
        verse = parse_verse_id("GEN.1.1")
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

        verse = parse_verse_id("ROM.3.23a")
        assert verse == VerseID(book="ROM", chapter=3, verse=23, part="a")

    def test_canonical_case_insensitive(self):
        """Test that canonical parsing is case-insensitive for book."""
        verse = parse_verse_id("gen.1.1")
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

    def test_common_formats(self):
        """Test parsing common verse formats."""
        # Colon separator
        verse = parse_verse_id("Genesis 1:1")
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

        # Period separator
        verse = parse_verse_id("Genesis 1.1")
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

        # Abbreviated book name
        verse = parse_verse_id("Gen 1:1")
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

        # With part
        verse = parse_verse_id("Rom 3:23a")
        assert verse == VerseID(book="ROM", chapter=3, verse=23, part="a")

    def test_numbered_books(self):
        """Test parsing numbered book references."""
        verse = parse_verse_id("1 John 3:16")
        assert verse == VerseID(book="1JN", chapter=3, verse=16)

        verse = parse_verse_id("2 Samuel 7:12")
        assert verse == VerseID(book="2SA", chapter=7, verse=12)

        verse = parse_verse_id("1Sam 1:1")
        assert verse == VerseID(book="1SA", chapter=1, verse=1)

    def test_invalid_formats(self):
        """Test that invalid formats return None."""
        assert parse_verse_id("") is None
        assert parse_verse_id("Invalid") is None
        assert parse_verse_id("Genesis") is None  # No chapter/verse
        assert parse_verse_id("Gen 1") is None  # No verse
        assert parse_verse_id("XXX 1:1") is None  # Invalid book


class TestCreateVerseID:
    """Test create_verse_id function."""

    def test_with_book_code(self):
        """Test creating with book code."""
        verse = create_verse_id("GEN", 1, 1)
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

        verse = create_verse_id("gen", 1, 1)  # Case insensitive
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

    def test_with_book_name(self):
        """Test creating with book name."""
        verse = create_verse_id("Genesis", 1, 1)
        assert verse == VerseID(book="GEN", chapter=1, verse=1)

        verse = create_verse_id("1 Samuel", 1, 1)
        assert verse == VerseID(book="1SA", chapter=1, verse=1)

    def test_with_part(self):
        """Test creating with verse part."""
        verse = create_verse_id("ROM", 3, 23, "a")
        assert verse == VerseID(book="ROM", chapter=3, verse=23, part="a")

    def test_validation(self):
        """Test validation in create_verse_id."""
        # Invalid book
        assert create_verse_id("XXX", 1, 1) is None

        # Invalid chapter (too high)
        assert create_verse_id("GEN", 51, 1) is None  # Genesis has 50 chapters

        # Invalid chapter (too low)
        assert create_verse_id("GEN", 0, 1) is None

        # Invalid verse
        assert create_verse_id("GEN", 1, 0) is None

        # Invalid part
        assert create_verse_id("GEN", 1, 1, "1") is None  # Not a letter
        assert create_verse_id("GEN", 1, 1, "aa") is None  # Too long


class TestParseVerseRange:
    """Test parse_verse_range function."""

    def test_canonical_format(self):
        """Test parsing canonical range format."""
        range_obj = parse_verse_range("GEN.1.1-GEN.1.5")
        assert range_obj is not None
        assert range_obj.start == VerseID(book="GEN", chapter=1, verse=1)
        assert range_obj.end == VerseID(book="GEN", chapter=1, verse=5)

    def test_simple_format(self):
        """Test parsing simple range format (same chapter)."""
        range_obj = parse_verse_range("Genesis 1:1-5")
        assert range_obj is not None
        assert range_obj.start == VerseID(book="GEN", chapter=1, verse=1)
        assert range_obj.end == VerseID(book="GEN", chapter=1, verse=5)

    def test_full_reference_range(self):
        """Test parsing with full references on both sides."""
        range_obj = parse_verse_range("Genesis 1:1 - Genesis 1:5")
        assert range_obj is not None
        assert range_obj.start == VerseID(book="GEN", chapter=1, verse=1)
        assert range_obj.end == VerseID(book="GEN", chapter=1, verse=5)

    def test_cross_chapter_range(self):
        """Test parsing range across chapters."""
        range_obj = parse_verse_range("GEN.1.31-GEN.2.3")
        assert range_obj is not None
        assert range_obj.start == VerseID(book="GEN", chapter=1, verse=31)
        assert range_obj.end == VerseID(book="GEN", chapter=2, verse=3)

    def test_invalid_ranges(self):
        """Test that invalid ranges return None."""
        # End before start
        assert parse_verse_range("GEN.1.5-GEN.1.1") is None

        # Invalid verse references
        assert parse_verse_range("XXX.1.1-XXX.1.5") is None

        # Empty string
        assert parse_verse_range("") is None


class TestNormalizeVerseID:
    """Test normalize_verse_id function."""

    def test_normalization(self):
        """Test normalizing various verse formats."""
        assert normalize_verse_id("Genesis 1:1") == "GEN.1.1"
        assert normalize_verse_id("Gen 1:1") == "GEN.1.1"
        assert normalize_verse_id("GEN.1.1") == "GEN.1.1"
        assert normalize_verse_id("ROM.3.23a") == "ROM.3.23a"

    def test_invalid_inputs(self):
        """Test that invalid inputs return None."""
        assert normalize_verse_id("Invalid") is None
        assert normalize_verse_id("") is None


class TestIsValidVerseID:
    """Test is_valid_verse_id function."""

    def test_valid_verses(self):
        """Test valid verse IDs."""
        assert is_valid_verse_id("GEN.1.1") is True
        assert is_valid_verse_id("Genesis 1:1") is True
        assert is_valid_verse_id("ROM.3.23a") is True

    def test_invalid_verses(self):
        """Test invalid verse IDs."""
        assert is_valid_verse_id("Invalid") is False
        assert is_valid_verse_id("") is False
        assert is_valid_verse_id("GEN") is False


class TestCompareVerseIDs:
    """Test compare_verse_ids function."""

    def test_comparison_with_objects(self):
        """Test comparing VerseID objects."""
        verse1 = VerseID(book="GEN", chapter=1, verse=1)
        verse2 = VerseID(book="GEN", chapter=1, verse=2)

        assert compare_verse_ids(verse1, verse2) == -1
        assert compare_verse_ids(verse2, verse1) == 1
        assert compare_verse_ids(verse1, verse1) == 0

    def test_comparison_with_strings(self):
        """Test comparing verse strings."""
        assert compare_verse_ids("GEN.1.1", "GEN.1.2") == -1
        assert compare_verse_ids("GEN.1.2", "GEN.1.1") == 1
        assert compare_verse_ids("GEN.1.1", "GEN.1.1") == 0

    def test_mixed_comparison(self):
        """Test comparing mixed types."""
        verse = VerseID(book="GEN", chapter=1, verse=1)
        assert compare_verse_ids(verse, "GEN.1.2") == -1
        assert compare_verse_ids("GEN.1.2", verse) == 1

    def test_invalid_verses(self):
        """Test that invalid verses raise ValueError."""
        with pytest.raises(ValueError):
            compare_verse_ids("Invalid", "GEN.1.1")

        with pytest.raises(ValueError):
            compare_verse_ids("GEN.1.1", "Invalid")


class TestGetVerseParts:
    """Test get_verse_parts function."""

    def test_basic_functionality(self):
        """Test basic verse parts retrieval."""
        # Currently just returns the base verse
        parts = get_verse_parts("ROM.3.23")
        assert len(parts) == 1
        assert parts[0] == VerseID(book="ROM", chapter=3, verse=23)

        verse = VerseID(book="ROM", chapter=3, verse=23)
        parts = get_verse_parts(verse)
        assert len(parts) == 1
        assert parts[0] == verse

    def test_invalid_input(self):
        """Test with invalid input."""
        parts = get_verse_parts("Invalid")
        assert parts == []
