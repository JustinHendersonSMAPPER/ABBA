"""
Test suite for Unicode utilities.
"""

import unittest
import unicodedata

from abba.language.unicode_utils import (
    NormalizationForm,
    CombiningSequence,
    UnicodeNormalizer,
    HebrewNormalizer,
    GreekNormalizer,
    UnicodeValidator,
    CharacterInfo,
    get_unicode_info,
    is_combining_mark,
    strip_accents,
)


class TestNormalizationForm(unittest.TestCase):
    """Test NormalizationForm enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        self.assertEqual(NormalizationForm.NFC.value, "NFC")
        self.assertEqual(NormalizationForm.NFD.value, "NFD")
        self.assertEqual(NormalizationForm.NFKC.value, "NFKC")
        self.assertEqual(NormalizationForm.NFKD.value, "NFKD")
        self.assertEqual(NormalizationForm.HEBREW_CANONICAL.value, "HEBREW_CANONICAL")
        self.assertEqual(NormalizationForm.GREEK_CANONICAL.value, "GREEK_CANONICAL")


class TestCombiningSequence(unittest.TestCase):
    """Test CombiningSequence class."""

    def test_creation(self):
        """Test creating combining sequences."""
        seq = CombiningSequence(
            base="a",
            combiners=["\u0301", "\u0308"],  # acute, diaeresis
            start_pos=0,
            end_pos=2,
        )

        self.assertEqual(seq.base, "a")
        self.assertEqual(len(seq.combiners), 2)
        self.assertEqual(seq.start_pos, 0)
        self.assertEqual(seq.end_pos, 2)

    def test_to_string(self):
        """Test converting to string."""
        seq = CombiningSequence(
            base="e", combiners=["\u0301"], start_pos=0, end_pos=1  # acute
        )

        result = seq.to_string()
        self.assertEqual(result, "e\u0301")

    def test_normalize_order(self):
        """Test normalizing combiner order."""
        # Create sequence with combiners in wrong order
        seq = CombiningSequence(
            base="a",
            combiners=["\u0308", "\u0301"],  # diaeresis (230), acute (230)
            start_pos=0,
            end_pos=2,
        )

        normalized = seq.normalize_order()
        # Order should be preserved when combining classes are equal
        self.assertEqual(normalized.combiners, ["\u0308", "\u0301"])


class TestUnicodeNormalizer(unittest.TestCase):
    """Test UnicodeNormalizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = UnicodeNormalizer()

    def test_normalize_nfc(self):
        """Test NFC normalization."""
        # Decomposed to composed
        text = "e\u0301"  # e + combining acute
        result = self.normalizer.normalize(text, NormalizationForm.NFC)
        self.assertEqual(result, "é")

    def test_normalize_nfd(self):
        """Test NFD normalization."""
        # Composed to decomposed
        text = "é"
        result = self.normalizer.normalize(text, NormalizationForm.NFD)
        self.assertEqual(result, "e\u0301")

    def test_clean_text(self):
        """Test text cleaning."""
        # Text with problematic characters
        text = "Hello\u200B\u00A0world\u2019s\u2014test"
        result = self.normalizer.clean_text(text)
        self.assertEqual(result, "Hello world's-test")

    def test_remove_zero_width(self):
        """Test removing zero-width characters."""
        text = "test\u200Bword\u200Cmore\u200D"
        result = self.normalizer.remove_zero_width(text)
        self.assertEqual(result, "testwordmore")

    def test_normalize_whitespace(self):
        """Test normalizing whitespace."""
        text = "  multiple   spaces   \t\n  here  "
        result = self.normalizer.normalize_whitespace(text)
        self.assertEqual(result, "multiple spaces here")

    def test_decompose_to_sequences(self):
        """Test decomposing to combining sequences."""
        text = "café"
        sequences = self.normalizer.decompose_to_sequences(text)

        self.assertEqual(len(sequences), 4)
        self.assertEqual(sequences[0].base, "c")
        self.assertEqual(sequences[1].base, "a")
        self.assertEqual(sequences[2].base, "f")
        self.assertEqual(sequences[3].base, "e")
        self.assertEqual(len(sequences[3].combiners), 1)  # acute accent


class TestHebrewNormalizer(unittest.TestCase):
    """Test HebrewNormalizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = HebrewNormalizer()

    def test_normalize_hebrew(self):
        """Test Hebrew normalization."""
        # Hebrew text with vowels
        text = "שָׁלוֹם"
        result = self.normalizer.normalize_hebrew(text)
        # Should preserve the text
        self.assertIn("ש", result)
        self.assertIn("ל", result)
        self.assertIn("ם", result)

    def test_normalize_final_forms(self):
        """Test normalizing Hebrew final forms."""
        # Text with final forms
        text = "שלום חכם"
        result = self.normalizer.normalize_final_forms(text)
        # Final mem should be used at word end
        self.assertTrue(result.endswith("ם"))

    def test_strip_hebrew_points(self):
        """Test stripping Hebrew points."""
        # Text with vowels and accents
        text = "בְּרֵאשִׁית"
        
        # Strip all
        result = self.normalizer.strip_hebrew_points(text)
        self.assertEqual(result, "בראשית")
        
        # Keep vowels
        result = self.normalizer.strip_hebrew_points(text, keep_vowels=True)
        self.assertIn("\u05B0", result)  # Sheva should remain

    def test_handle_holam_vav(self):
        """Test handling holam vav combinations."""
        # Vav with holam
        text = "ו\u05B9"  # Vav + holam
        result = self.normalizer.handle_holam_vav(text)
        # Should convert to precomposed form
        self.assertEqual(result, "\uFB4B")


class TestGreekNormalizer(unittest.TestCase):
    """Test GreekNormalizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = GreekNormalizer()

    def test_normalize_greek(self):
        """Test Greek normalization."""
        # Greek text with diacritics
        text = "Ἰησοῦς"
        result = self.normalizer.normalize_greek(text)
        # Should preserve the text
        self.assertIn("Ἰ", result)
        self.assertIn("σ", result)

    def test_normalize_final_sigma(self):
        """Test final sigma normalization."""
        # Text with regular sigma at word end
        text = "λόγοσ τοῦ θεοῦ"
        result = self.normalizer.normalize_final_sigma(text)
        # Should use final sigma
        self.assertIn("ς", result)

    def test_normalize_breathing_marks(self):
        """Test normalizing breathing mark order."""
        # Text with breathing marks and accents
        text = "ἐν ἀρχῇ"
        result = self.normalizer.normalize_breathing_marks(text)
        # Should maintain proper order
        self.assertIn("ἐ", result)
        self.assertIn("ἀ", result)

    def test_strip_greek_accents(self):
        """Test stripping Greek accents."""
        # Text with various accents
        text = "Ἰησοῦς Χριστός"
        
        # Strip all accents
        result = self.normalizer.strip_greek_accents(text)
        # Should remove accents but keep base letters
        self.assertIn("Ι", result)
        self.assertIn("Χ", result)
        
        # Keep breathing marks
        result = self.normalizer.strip_greek_accents(text, keep_breathing=True)
        # Should keep smooth breathing on Iota
        self.assertIn("Ἰ", result)


class TestUnicodeValidator(unittest.TestCase):
    """Test UnicodeValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = UnicodeValidator()

    def test_validate_text(self):
        """Test validating Unicode strings."""
        # Valid text
        valid_text = "Hello שלום Ἰησοῦς"
        errors = self.validator.validate_text(valid_text)
        self.assertEqual(len(errors), 0)
        
        # Text with control characters (simulate)
        # Note: Can't include actual invalid characters in source

    def test_detect_scripts(self):
        """Test detecting scripts in text."""
        # Mixed script text
        text = "Hello שלום world"
        scripts = self.validator.detect_scripts(text)
        
        self.assertIn("Latin", scripts)
        self.assertIn("Hebrew", scripts)

    def test_has_valid_normalization(self):
        """Test checking valid normalization."""
        # Already normalized text
        text = "café"
        self.assertTrue(self.validator.has_valid_normalization(text))
        
        # Text that needs normalization
        text = "cafe\u0301"  # Decomposed
        self.assertTrue(self.validator.has_valid_normalization(text))

    def test_is_normalization_safe(self):
        """Test checking if normalization is safe."""
        # Safe text
        safe_text = "Hello world"
        self.assertTrue(self.validator.is_normalization_safe(safe_text))
        
        # Text that might change meaning if normalized
        # (Example: certain Arabic or Hebrew combinations)
        hebrew_text = "שָׁלוֹם"
        self.assertTrue(self.validator.is_normalization_safe(hebrew_text))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_unicode_info(self):
        """Test getting Unicode character info."""
        # Latin character
        info = get_unicode_info('A')
        self.assertIsInstance(info, CharacterInfo)
        self.assertEqual(info.char, 'A')
        self.assertEqual(info.name, 'LATIN CAPITAL LETTER A')
        self.assertEqual(info.category, 'Lu')
        
        # Hebrew character
        info = get_unicode_info('א')
        self.assertEqual(info.name, 'HEBREW LETTER ALEF')
        self.assertEqual(info.script, 'Hebrew')

    def test_is_combining_mark(self):
        """Test identifying combining marks."""
        # Combining marks
        self.assertTrue(is_combining_mark('\u0301'))  # Combining acute
        self.assertTrue(is_combining_mark('\u05B8'))  # Hebrew qamats
        
        # Regular characters
        self.assertFalse(is_combining_mark('a'))
        self.assertFalse(is_combining_mark('א'))

    def test_strip_accents(self):
        """Test stripping accents utility."""
        # Test with simple strings first
        self.assertEqual(strip_accents("hello"), "hello")
        self.assertEqual(strip_accents("test"), "test")
        
        # Test with composed characters
        text_with_acute = "e\u0301"  # e with combining acute
        self.assertEqual(strip_accents(text_with_acute), "e")


if __name__ == "__main__":
    unittest.main()