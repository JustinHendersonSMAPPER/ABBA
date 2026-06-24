"""Tests for Strong's number normalization utilities."""

import pytest

from abba.strongs import extract_lexical_strongs, normalize_strongs


class TestNormalizeStrongs:
    def test_hebrew_padded(self) -> None:
        assert normalize_strongs("H0430") == "H430"

    def test_greek_padded(self) -> None:
        assert normalize_strongs("G0746") == "G746"

    def test_greek_with_suffix(self) -> None:
        assert normalize_strongs("G0901a") == "G901a"

    def test_hebrew_no_leading_zeros(self) -> None:
        assert normalize_strongs("H9003") == "H9003"

    def test_empty_string(self) -> None:
        assert normalize_strongs("") == ""

    def test_none(self) -> None:
        assert normalize_strongs(None) == ""

    def test_lowercase_letter(self) -> None:
        assert normalize_strongs("h0430") == "H430"

    def test_greek_lowercase_letter(self) -> None:
        assert normalize_strongs("g0746") == "G746"

    def test_already_normalized(self) -> None:
        assert normalize_strongs("H430") == "H430"
        assert normalize_strongs("G746") == "G746"


class TestExtractLexicalStrongs:
    def test_greek_padded_primary(self) -> None:
        # G0746 is a valid lexical primary; returned in source (padded) form (lookups normalize).
        assert extract_lexical_strongs("G0746", None) == "G0746"

    def test_hebrew_step_prefix_with_raw(self) -> None:
        # H9003 is a STEP prefix, fall back to strongs_raw
        assert extract_lexical_strongs("H9003", "H9003/{H7225G}") == "H7225"

    def test_hebrew_elohim_from_raw(self) -> None:
        # H0430G in braces -> strip STEP tag G -> H0430 (source/padded form; lookups normalize to H430)
        assert extract_lexical_strongs("H9001", "{H0430G}") == "H0430"

    def test_hebrew_creation_word(self) -> None:
        # H1254A in braces -> strip STEP tag A -> H1254
        assert extract_lexical_strongs("H9002", "{H1254A}") == "H1254"

    def test_valid_hebrew_primary(self) -> None:
        # H7225 is valid (not H9000-H9999)
        assert extract_lexical_strongs("H7225", None) == "H7225"

    def test_none_primary_none_raw(self) -> None:
        assert extract_lexical_strongs(None, None) == ""

    def test_none_primary_with_raw(self) -> None:
        assert extract_lexical_strongs(None, "{H7225G}") == "H7225"

    def test_greek_no_raw_needed(self) -> None:
        assert extract_lexical_strongs("G3056", None) == "G3056"
