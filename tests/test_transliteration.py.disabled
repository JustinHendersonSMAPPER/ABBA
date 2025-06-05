"""Tests for transliteration engines."""

import pytest

from abba.language.transliteration import (
    TransliterationScheme,
    TransliterationRule,
    HebrewTransliterator,
    GreekTransliterator,
    ArabicTransliterator,
    SyriacTransliterator,
)


class TestTransliterationRule:
    """Test transliteration rules."""

    def test_rule_creation(self):
        """Test creating transliteration rule."""
        rule = TransliterationRule(
            source="א", target="ʾ", context=None, priority=10, reversible=True
        )

        assert rule.source == "א"
        assert rule.target == "ʾ"
        assert rule.priority == 10
        assert rule.reversible


class TestHebrewTransliterator:
    """Test Hebrew transliteration."""

    def test_sbl_hebrew_transliteration(self):
        """Test SBL Hebrew transliteration."""
        trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)

        # Test consonants
        assert trans.transliterate("א") == "ʾ"
        assert trans.transliterate("ב") == "b" or trans.transliterate("ב") == "v"
        assert trans.transliterate("ג") == "g"
        assert trans.transliterate("ד") == "d"
        assert trans.transliterate("ה") == "h"

        # Test with dagesh
        assert trans.transliterate("בּ") == "b"
        assert trans.transliterate("כּ") == "k"
        assert trans.transliterate("פּ") == "p"

        # Test final forms
        assert trans.transliterate("ך") == "ḵ"
        assert trans.transliterate("ם") == "m"
        assert trans.transliterate("ן") == "n"
        assert trans.transliterate("ף") == "p̄"
        assert trans.transliterate("ץ") == "ṣ"

        # Test shin/sin
        assert trans.transliterate("שׁ") == "š"
        assert trans.transliterate("שׂ") == "ś"

    def test_sbl_hebrew_vowels(self):
        """Test SBL Hebrew vowel transliteration."""
        trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)

        # Test vowels
        assert trans.transliterate("בַ") == "ba"  # Patah
        assert trans.transliterate("בָ") == "bā"  # Qamats
        assert trans.transliterate("בֶ") == "be"  # Segol
        assert trans.transliterate("בֵ") == "bē"  # Tsere
        assert trans.transliterate("בִ") == "bi"  # Hiriq
        assert trans.transliterate("בֹ") == "bō"  # Holam
        assert trans.transliterate("בֻ") == "bu"  # Qubuts
        assert trans.transliterate("בְ") == "bĕ"  # Sheva

    def test_simple_hebrew_transliteration(self):
        """Test simple Hebrew transliteration."""
        trans = HebrewTransliterator(TransliterationScheme.SIMPLE_HEBREW)

        # No diacritics in simple scheme
        assert trans.transliterate("א") == "'"
        assert trans.transliterate("ב") == "v"
        assert trans.transliterate("בּ") == "b"
        assert trans.transliterate("ח") == "ch"
        assert trans.transliterate("צ") == "ts"
        assert trans.transliterate("שׁ") == "sh"

    def test_hebrew_word_transliteration(self):
        """Test transliterating Hebrew words."""
        trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)

        # Test a simple word
        result = trans.transliterate("שָׁלוֹם")
        assert "š" in result
        assert "l" in result
        assert "m" in result

    def test_hebrew_reverse_transliteration(self):
        """Test reverse transliteration to Hebrew."""
        trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)

        # Simple test
        hebrew = "שׁ"
        latin = trans.transliterate(hebrew)
        back = trans.reverse_transliterate(latin)

        # May not be identical due to ambiguities
        assert "ש" in back


class TestGreekTransliterator:
    """Test Greek transliteration."""

    def test_sbl_greek_transliteration(self):
        """Test SBL Greek transliteration."""
        trans = GreekTransliterator(TransliterationScheme.SBL_GREEK)

        # Test basic letters
        assert trans.transliterate("α") == "a"
        assert trans.transliterate("β") == "b"
        assert trans.transliterate("γ") == "g"
        assert trans.transliterate("δ") == "d"
        assert trans.transliterate("ε") == "e"
        assert trans.transliterate("η") == "ē"
        assert trans.transliterate("θ") == "th"
        assert trans.transliterate("ω") == "ō"

        # Test diphthongs
        assert trans.transliterate("αι") == "ai"
        assert trans.transliterate("ει") == "ei"
        assert trans.transliterate("οι") == "oi"
        assert trans.transliterate("ου") == "ou"

        # Test special combinations
        assert trans.transliterate("γγ") == "ng"
        assert trans.transliterate("γκ") == "nk"

    def test_simple_greek_transliteration(self):
        """Test simple Greek transliteration."""
        trans = GreekTransliterator(TransliterationScheme.SIMPLE_GREEK)

        # Simple scheme uses basic letters
        assert trans.transliterate("η") == "e"  # Not ē
        assert trans.transliterate("ω") == "o"  # Not ō
        assert trans.transliterate("φ") == "f"  # Not ph

    def test_beta_code_transliteration(self):
        """Test Beta Code transliteration."""
        trans = GreekTransliterator(TransliterationScheme.BETA_CODE)

        # Beta Code uses different mappings
        assert trans.transliterate("α") == "a"
        assert trans.transliterate("η") == "h"
        assert trans.transliterate("θ") == "q"
        assert trans.transliterate("ξ") == "c"
        assert trans.transliterate("ψ") == "y"
        assert trans.transliterate("ω") == "w"

        # Uppercase
        assert trans.transliterate("Α") == "*a"

    def test_greek_breathing_marks(self):
        """Test Greek breathing mark transliteration."""
        trans = GreekTransliterator(TransliterationScheme.SBL_GREEK)

        # Smooth breathing (usually ignored)
        assert trans.transliterate("ἀ") == "a"

        # Rough breathing
        assert trans.transliterate("ἁ") == "ha"

    def test_greek_word_transliteration(self):
        """Test transliterating Greek words."""
        trans = GreekTransliterator(TransliterationScheme.SBL_GREEK)

        # Test a word
        result = trans.transliterate("λόγος")
        assert result == "logos" or result == "lógos"


class TestArabicTransliterator:
    """Test Arabic transliteration."""

    def test_arabic_din_transliteration(self):
        """Test DIN 31635 Arabic transliteration."""
        trans = ArabicTransliterator(TransliterationScheme.ARABIC_DIN)

        # Test basic letters
        assert trans.transliterate("ا") == "ā"  # Alif
        assert trans.transliterate("ب") == "b"  # Ba
        assert trans.transliterate("ت") == "t"  # Ta
        assert trans.transliterate("ث") == "ṯ"  # Tha
        assert trans.transliterate("ج") == "ǧ"  # Jim
        assert trans.transliterate("ح") == "ḥ"  # Ha
        assert trans.transliterate("خ") == "ḫ"  # Kha

        # Test special characters
        assert trans.transliterate("ء") == "ʾ"  # Hamza
        assert trans.transliterate("ة") == "h"  # Ta marbuta

        # Test article
        assert trans.transliterate("ال") == "al-"

    def test_arabic_word_transliteration(self):
        """Test transliterating Arabic words."""
        trans = ArabicTransliterator(TransliterationScheme.ARABIC_DIN)

        # Test with article
        result = trans.transliterate("الكتاب")
        assert result.startswith("al-")
        assert "k" in result


class TestSyriacTransliterator:
    """Test Syriac transliteration."""

    def test_syriac_basic_transliteration(self):
        """Test basic Syriac transliteration."""
        trans = SyriacTransliterator(TransliterationScheme.SYRIAC_ACADEMIC)

        # Test basic letters
        assert trans.transliterate("ܐ") == "ʾ"  # Alaph
        assert trans.transliterate("ܒ") == "b"  # Beth
        assert trans.transliterate("ܓ") == "g"  # Gamal
        assert trans.transliterate("ܕ") == "d"  # Dalath
        assert trans.transliterate("ܗ") == "h"  # He
        assert trans.transliterate("ܘ") == "w"  # Waw
        assert trans.transliterate("ܙ") == "z"  # Zayn


class TestTransliterationEdgeCases:
    """Test edge cases in transliteration."""

    def test_empty_text(self):
        """Test transliterating empty text."""
        trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)

        assert trans.transliterate("") == ""
        assert trans.reverse_transliterate("") == ""

    def test_preserve_unknown(self):
        """Test preserving unknown characters."""
        trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)

        # Mix of Hebrew and English
        mixed = "Hello שלום World"
        result = trans.transliterate(mixed)

        assert "Hello" in result
        assert "World" in result
        assert "שלום" not in result  # Hebrew transliterated

    def test_context_sensitive_rules(self):
        """Test context-sensitive transliteration."""
        trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)

        # Beth with/without dagesh
        with_dagesh = "בּ"
        without_dagesh = "ב"

        assert trans.transliterate(with_dagesh) == "b"
        # Without dagesh might be 'v' depending on context
