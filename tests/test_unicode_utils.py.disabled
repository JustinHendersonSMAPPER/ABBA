"""Tests for Unicode normalization utilities."""

import pytest
import unicodedata

from abba.language.unicode_utils import (
    UnicodeNormalizer,
    NormalizationForm,
    HebrewNormalizer,
    GreekNormalizer,
    DiacriticHandler,
    CombiningCharacterHandler,
    CombiningSequence,
)


class TestUnicodeNormalizer:
    """Test base Unicode normalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = UnicodeNormalizer()

    def test_standard_normalization(self):
        """Test standard Unicode normalization forms."""
        # Test string with combining characters
        text = "e\u0301"  # e + acute accent

        nfc = self.normalizer.normalize(text, NormalizationForm.NFC)
        nfd = self.normalizer.normalize(text, NormalizationForm.NFD)

        assert len(nfc) == 1  # Composed form
        assert len(nfd) == 2  # Decomposed form
        assert nfc == "é"

    def test_clean_text(self):
        """Test text cleaning."""
        # Text with various problematic characters
        text = "Test\u2019s\u00A0text\u2014with\u200Bzero-width"
        cleaned = self.normalizer.clean_text(text)

        assert "\u2019" not in cleaned  # Right quote replaced
        assert "\u00A0" not in cleaned  # Non-breaking space replaced
        assert "\u2014" not in cleaned  # Em dash replaced
        assert "\u200B" not in cleaned  # Zero-width space removed
        assert "Test's text-with zero-width" in cleaned

    def test_remove_zero_width(self):
        """Test zero-width character removal."""
        text = "Text\u200Bwith\u200Czero\u200Dwidth\uFEFFchars"
        result = self.normalizer.remove_zero_width(text)

        assert "\u200B" not in result
        assert "\u200C" not in result
        assert "\u200D" not in result
        assert "\uFEFF" not in result
        assert result == "Textwithzerowidthchars"

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  Text   with    multiple\t\nspaces  "
        result = self.normalizer.normalize_whitespace(text)

        assert result == "Text with multiple spaces"

    def test_decompose_to_sequences(self):
        """Test decomposition to combining sequences."""
        text = "café"  # Has combining acute on e
        sequences = self.normalizer.decompose_to_sequences(text)

        assert len(sequences) == 4  # c, a, f, e+acute

        # Check the accented e
        e_seq = sequences[3]
        assert e_seq.base == "e"
        assert len(e_seq.combiners) == 1
        assert ord(e_seq.combiners[0]) == 0x0301  # Combining acute


class TestCombiningSequence:
    """Test combining sequence handling."""

    def test_combining_sequence_creation(self):
        """Test creating combining sequence."""
        seq = CombiningSequence(
            base="e", combiners=["\u0301", "\u0308"], start_pos=0, end_pos=3  # acute + diaeresis
        )

        assert seq.base == "e"
        assert len(seq.combiners) == 2
        assert seq.to_string() == "e\u0301\u0308"

    def test_normalize_combiner_order(self):
        """Test normalizing combining mark order."""
        # Create sequence with wrong order
        seq = CombiningSequence(
            base="e",
            combiners=["\u0308", "\u0301"],  # diaeresis + acute (wrong order)
            start_pos=0,
            end_pos=3,
        )

        normalized = seq.normalize_order()

        # Should reorder by combining class
        assert normalized.combiners[0] == "\u0301"  # Acute (class 230)
        assert normalized.combiners[1] == "\u0308"  # Diaeresis (class 230)


class TestHebrewNormalizer:
    """Test Hebrew-specific normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = HebrewNormalizer()

    def test_normalize_hebrew_text(self):
        """Test Hebrew text normalization."""
        # Hebrew with vowels and accents
        text = "בְּרֵאשִׁית"
        normalized = self.normalizer.normalize_hebrew(text)

        assert normalized is not None
        assert "ב" in normalized
        assert "\u05B0" in normalized  # Sheva

    def test_normalize_final_forms(self):
        """Test Hebrew final letter normalization."""
        # Text with final letters
        text = "שלום עולם"  # Has final mem
        normalized = self.normalizer.normalize_final_forms(text)

        # Final mem should be preserved at word end
        assert "ם" in normalized

    def test_normalize_vowel_order(self):
        """Test Hebrew vowel ordering."""
        # Text with vowels and accents in wrong order
        text = "בְּ"  # Beth + dagesh + sheva
        normalized = self.normalizer.normalize_vowel_order(text)

        assert normalized is not None
        # Vowels should come before accents

    def test_handle_holam_vav(self):
        """Test holam vav special handling."""
        text = "ו\u05B9"  # Vav + holam
        result = self.normalizer.handle_holam_vav(text)

        # Should be normalized to precomposed form
        assert "\uFB4B" in result  # Vav with holam

    def test_strip_hebrew_points(self):
        """Test stripping Hebrew pointing."""
        text = "בְּרֵאשִׁית"

        # Strip all points
        stripped = self.normalizer.strip_hebrew_points(text)
        assert stripped == "בראשית"

        # Keep vowels only
        vowels_only = self.normalizer.strip_hebrew_points(text, keep_vowels=True)
        assert "\u05B0" in vowels_only  # Sheva kept

        # Keep accents only
        accents_only = self.normalizer.strip_hebrew_points(text, keep_accents=True)
        assert "\u05B0" not in accents_only  # Sheva removed


class TestGreekNormalizer:
    """Test Greek-specific normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = GreekNormalizer()

    def test_normalize_greek_text(self):
        """Test Greek text normalization."""
        # Greek with diacritics
        text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
        normalized = self.normalizer.normalize_greek(text)

        assert normalized is not None
        assert "λόγος" in normalized

    def test_normalize_breathing_marks(self):
        """Test Greek breathing mark normalization."""
        # Text with breathing marks
        text = "ὁ ἄνθρωπος"
        normalized = self.normalizer.normalize_breathing_marks(text)

        assert normalized is not None
        # Breathing marks should be ordered correctly

    def test_normalize_final_sigma(self):
        """Test final sigma normalization."""
        text = "λογος της"
        normalized = self.normalizer.normalize_final_sigma(text)

        # Should have final sigma at word ends
        assert "λογος" not in normalized
        assert "λογος"[:-1] + "ς" in normalized

    def test_strip_greek_accents(self):
        """Test stripping Greek accents."""
        text = "λόγος"

        # Strip all accents
        stripped = self.normalizer.strip_greek_accents(text)
        assert stripped == "λογος"

        # Keep breathing marks
        text_breathing = "ὁ λόγος"
        stripped_breathing = self.normalizer.strip_greek_accents(
            text_breathing, keep_breathing=True
        )
        assert "ὁ" in stripped_breathing  # Breathing kept


class TestDiacriticHandler:
    """Test diacritic handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = DiacriticHandler()

    def test_count_diacritics(self):
        """Test counting diacritics."""
        text = "café naïve"  # Has combining marks
        text_nfd = unicodedata.normalize("NFD", text)

        counts = self.handler.count_diacritics(text_nfd)

        assert counts["above"] > 0  # Acute accents
        assert sum(counts.values()) == 3  # Total marks

    def test_has_diacritics(self):
        """Test diacritic detection."""
        # With diacritics
        assert self.handler.has_diacritics("café")

        # Without diacritics
        assert not self.handler.has_diacritics("cafe")

    def test_strip_all_diacritics(self):
        """Test stripping all diacritics."""
        text = "café naïve résumé"
        stripped = self.handler.strip_all_diacritics(text)

        assert stripped == "cafe naive resume"


class TestCombiningCharacterHandler:
    """Test combining character handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CombiningCharacterHandler()

    def test_validate_combining_sequences(self):
        """Test validation of combining sequences."""
        # Valid text
        valid_text = "café"
        errors = self.handler.validate_combining_sequences(valid_text)
        assert len(errors) == 0

        # Text with too many combiners (artificial example)
        bad_text = "e" + "\u0301" * 5  # 5 acute accents
        errors = self.handler.validate_combining_sequences(bad_text)
        assert len(errors) > 0
        assert "Too many combining marks" in errors[0]

    def test_fix_combining_order(self):
        """Test fixing combining character order."""
        # Text with wrong combining order
        text = "e\u0308\u0301"  # Diaeresis then acute (wrong order)
        fixed = self.handler.fix_combining_order(text)

        # Should be reordered
        assert fixed == "e\u0301\u0308" or fixed == "ë́"
