"""Tests for the search query parser."""

import pytest

from abba.api.query_parser import BOOK_NAME_TO_ID, ParsedQuery, parse_query


class TestBasicParsing:
    """Tests for basic query parsing."""

    def test_simple_text_query(self):
        """Simple text returns as-is."""
        result = parse_query("love")
        assert result.text == "love"
        assert result.search_type == "text"
        assert not result.is_exact_phrase

    def test_multi_word_query(self):
        """Multi-word queries are preserved."""
        result = parse_query("living water")
        assert result.text == "living water"

    def test_empty_query(self):
        """Empty query returns empty ParsedQuery."""
        result = parse_query("")
        assert result.text == ""

    def test_whitespace_only_query(self):
        """Whitespace-only query returns empty."""
        result = parse_query("   ")
        assert result.text == ""


class TestExactPhrase:
    """Tests for quoted exact phrase parsing."""

    def test_exact_phrase(self):
        """Quoted phrases set is_exact_phrase."""
        result = parse_query('"living water"')
        assert result.text == "living water"
        assert result.is_exact_phrase is True

    def test_exact_phrase_with_modifiers(self):
        """Quoted phrase with book filter."""
        result = parse_query('"living water" in:john')
        assert result.text == "living water"
        assert result.is_exact_phrase is True
        assert result.book_filter == 43


class TestBookFilter:
    """Tests for book filtering."""

    def test_in_book_filter(self):
        """in:book filters by book."""
        result = parse_query("love in:john")
        assert result.text == "love"
        assert result.book_filter == 43
        assert "book=john" in result.filters_applied

    def test_book_filter_keyword(self):
        """book:name also works."""
        result = parse_query("faith book:romans")
        assert result.book_filter == 45

    def test_book_filter_abbreviation(self):
        """Three-letter abbreviations work."""
        result = parse_query("creation in:gen")
        assert result.book_filter == 1

    def test_unknown_book_ignored(self):
        """Unknown book names are ignored."""
        result = parse_query("test in:notabook")
        assert result.book_filter is None


class TestTestamentFilter:
    """Tests for testament filtering."""

    def test_testament_new(self):
        """testament:new filters to NT."""
        result = parse_query("grace testament:new")
        assert result.testament_filter == "new"

    def test_testament_old(self):
        """testament:old filters to OT."""
        result = parse_query("covenant testament:old")
        assert result.testament_filter == "old"

    def test_shorthand_ot(self):
        """Trailing 'ot' sets old testament filter."""
        result = parse_query("covenant ot")
        assert result.testament_filter == "old"
        assert result.text == "covenant"

    def test_shorthand_nt(self):
        """Trailing 'nt' sets new testament filter."""
        result = parse_query("grace nt")
        assert result.testament_filter == "new"
        assert result.text == "grace"


class TestStrongsDetection:
    """Tests for Strong's number detection."""

    def test_strongs_modifier(self):
        """strongs:H0430 sets strongs_number."""
        result = parse_query("strongs:H0430")
        assert result.strongs_number == "H0430"
        assert result.search_type == "strongs"

    def test_strongs_auto_detect(self):
        """Bare Strong's numbers are auto-detected."""
        result = parse_query("H7225")
        assert result.strongs_number == "H7225"
        assert result.search_type == "strongs"

    def test_greek_strongs(self):
        """Greek Strong's numbers work."""
        result = parse_query("G3056")
        assert result.strongs_number == "G3056"


class TestLanguageFilter:
    """Tests for language filtering."""

    def test_language_hebrew(self):
        """language:hebrew filters by language."""
        result = parse_query("word language:hebrew")
        assert result.language_filter == "hebrew"

    def test_lang_shorthand(self):
        """lang:greek also works."""
        result = parse_query("word lang:greek")
        assert result.language_filter == "greek"


class TestCombinedFilters:
    """Tests for combining multiple filters."""

    def test_book_and_testament(self):
        """Multiple filters can combine."""
        result = parse_query("love in:john testament:new")
        assert result.book_filter == 43
        assert result.testament_filter == "new"
        assert result.text == "love"

    def test_has_filters_property(self):
        """has_filters returns True when filters are set."""
        result = parse_query("love in:john")
        assert result.has_filters is True

    def test_no_filters_property(self):
        """has_filters returns False for plain queries."""
        result = parse_query("love")
        assert result.has_filters is False


class TestBookMapping:
    """Tests for the book name mapping."""

    def test_all_66_books_mapped(self):
        """All 66 books should be accessible."""
        ids = set(BOOK_NAME_TO_ID.values())
        assert len(ids) == 66
        assert min(ids) == 1
        assert max(ids) == 66

    def test_common_names_work(self):
        """Common full names map correctly."""
        assert BOOK_NAME_TO_ID["genesis"] == 1
        assert BOOK_NAME_TO_ID["revelation"] == 66
        assert BOOK_NAME_TO_ID["psalms"] == 19
