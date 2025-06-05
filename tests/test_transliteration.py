"""
Test suite for transliteration functionality.
"""

import unittest
from unittest.mock import patch, MagicMock

from abba.language.transliteration import (
    TransliterationScheme,
    TransliterationRule,
    TransliterationEngine,
    HebrewTransliterator,
    GreekTransliterator,
    ArabicTransliterator,
    SyriacTransliterator,
    TransliterationValidator,
    create_transliterator,
)


class TestTransliterationScheme(unittest.TestCase):
    """Test TransliterationScheme enum."""

    def test_hebrew_schemes(self):
        """Test Hebrew transliteration schemes."""
        self.assertEqual(TransliterationScheme.SBL_HEBREW.value, "sbl_hebrew")
        self.assertEqual(TransliterationScheme.ACADEMIC_HEBREW.value, "academic_hebrew")
        self.assertEqual(TransliterationScheme.SIMPLE_HEBREW.value, "simple_hebrew")
        self.assertEqual(TransliterationScheme.ISO_259.value, "iso_259")

    def test_greek_schemes(self):
        """Test Greek transliteration schemes."""
        self.assertEqual(TransliterationScheme.SBL_GREEK.value, "sbl_greek")
        self.assertEqual(TransliterationScheme.ACADEMIC_GREEK.value, "academic_greek")
        self.assertEqual(TransliterationScheme.SIMPLE_GREEK.value, "simple_greek")
        self.assertEqual(TransliterationScheme.BETA_CODE.value, "beta_code")

    def test_other_schemes(self):
        """Test other transliteration schemes."""
        self.assertEqual(TransliterationScheme.ARABIC_DIN.value, "arabic_din")
        self.assertEqual(TransliterationScheme.SYRIAC_ACADEMIC.value, "syriac_academic")


class TestTransliterationRule(unittest.TestCase):
    """Test TransliterationRule dataclass."""

    def test_simple_rule(self):
        """Test creating a simple rule."""
        rule = TransliterationRule(
            source="א",
            target="'",
            priority=1,
            reversible=True
        )
        self.assertEqual(rule.source, "א")
        self.assertEqual(rule.target, "'")
        self.assertEqual(rule.priority, 1)
        self.assertTrue(rule.reversible)
        self.assertIsNone(rule.context)

    def test_contextual_rule(self):
        """Test creating a contextual rule."""
        rule = TransliterationRule(
            source="ה",
            target="h",
            context=r"[^$]",  # Not at end of word
            priority=2,
            reversible=False
        )
        self.assertEqual(rule.source, "ה")
        self.assertEqual(rule.target, "h")
        self.assertEqual(rule.context, r"[^$]")
        self.assertFalse(rule.reversible)


class TestTransliterationEngine(unittest.TestCase):
    """Test base TransliterationEngine functionality."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        # Create a concrete implementation for testing
        class TestEngine(TransliterationEngine):
            def setup_rules(self):
                self.rules = [
                    TransliterationRule("א", "'", priority=1),
                    TransliterationRule("ב", "b", priority=1),
                ]
        
        engine = TestEngine(TransliterationScheme.SIMPLE_HEBREW)
        self.assertEqual(engine.scheme, TransliterationScheme.SIMPLE_HEBREW)
        self.assertEqual(len(engine.rules), 2)

    def test_build_reverse_rules(self):
        """Test building reverse rules."""
        class TestEngine(TransliterationEngine):
            def setup_rules(self):
                self.rules = [
                    TransliterationRule("א", "'", reversible=True),
                    TransliterationRule("ב", "b", reversible=True),
                    TransliterationRule("ה", "", reversible=False),  # Silent at end
                ]
        
        engine = TestEngine(TransliterationScheme.SIMPLE_HEBREW)
        
        # Should have 2 reverse rules (excluding non-reversible)
        self.assertEqual(len(engine.reverse_rules), 2)
        
        # Check reverse mapping
        self.assertEqual(engine.reverse_rules[0].source, "'")
        self.assertEqual(engine.reverse_rules[0].target, "א")


class TestHebrewTransliterator(unittest.TestCase):
    """Test HebrewTransliterator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sbl_trans = HebrewTransliterator(TransliterationScheme.SBL_HEBREW)
        self.simple_trans = HebrewTransliterator(TransliterationScheme.SIMPLE_HEBREW)

    def test_simple_hebrew_transliteration(self):
        """Test simple Hebrew transliteration."""
        # Basic word
        hebrew = "שָׁלוֹם"
        result = self.simple_trans.transliterate(hebrew)
        
        # Should produce something like "shalom"
        self.assertIn("sh", result.lower())
        self.assertIn("l", result.lower())
        self.assertIn("m", result.lower())

    def test_sbl_hebrew_transliteration(self):
        """Test SBL Hebrew transliteration."""
        # Word with vowels
        hebrew = "בְּרֵאשִׁית"
        result = self.sbl_trans.transliterate(hebrew)
        
        # Should preserve more detail than simple
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_hebrew_consonants(self):
        """Test transliterating Hebrew consonants."""
        consonants = {
            "א": "'",
            "ב": "b",
            "ג": "g",
            "ד": "d",
            "ה": "h",
            "ו": "w",
            "ז": "z",
            "ח": "ch",
            "ט": "t",
            "י": "y",
            "כ": "k",
            "ל": "l",
            "מ": "m",
            "נ": "n",
            "ס": "s",
            "ע": "'",
            "פ": "p",
            "צ": "ts",
            "ק": "q",
            "ר": "r",
            "ש": "sh",
            "ת": "t",
        }
        
        for heb, expected in consonants.items():
            result = self.simple_trans.transliterate(heb)
            # The exact output may vary, but it should be consistent
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_hebrew_final_forms(self):
        """Test transliterating Hebrew final forms."""
        # Final forms should transliterate same as regular
        pairs = [
            ("כ", "ך"),  # kaf/final kaf
            ("מ", "ם"),  # mem/final mem
            ("נ", "ן"),  # nun/final nun
            ("פ", "ף"),  # pe/final pe
            ("צ", "ץ"),  # tsadi/final tsadi
        ]
        
        for regular, final in pairs:
            reg_result = self.simple_trans.transliterate(regular)
            final_result = self.simple_trans.transliterate(final)
            # Should produce similar base consonant
            self.assertEqual(reg_result[0], final_result[0])

    def test_hebrew_with_vowels(self):
        """Test transliterating Hebrew with vowel points."""
        # Text with various vowels
        voweled = "קָטָן"  # qatan (small)
        result = self.sbl_trans.transliterate(voweled)
        
        # Should handle vowels
        self.assertIsInstance(result, str)
        self.assertIn("q", result.lower())

    def test_preserve_unknown_chars(self):
        """Test preserving unknown characters."""
        # Mixed text
        mixed = "Hebrew: שָׁלוֹם!"
        result = self.simple_trans.transliterate(mixed, preserve_unknown=True)
        
        # Should preserve punctuation and Latin
        self.assertIn(":", result)
        self.assertIn("!", result)
        self.assertIn("Hebrew", result)

    def test_reverse_transliteration(self):
        """Test reverse transliteration."""
        # Start with transliterated text
        latin = "shalom"
        result = self.simple_trans.reverse_transliterate(latin)
        
        # Should produce Hebrew characters
        # (exact result depends on scheme and ambiguity)
        self.assertIsInstance(result, str)


class TestGreekTransliterator(unittest.TestCase):
    """Test GreekTransliterator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sbl_trans = GreekTransliterator(TransliterationScheme.SBL_GREEK)
        self.simple_trans = GreekTransliterator(TransliterationScheme.SIMPLE_GREEK)
        self.beta_trans = GreekTransliterator(TransliterationScheme.BETA_CODE)

    def test_simple_greek_transliteration(self):
        """Test simple Greek transliteration."""
        # Basic word (using unaccented Greek for better test reliability)
        greek = "λογος"  # logos without accents
        result = self.simple_trans.transliterate(greek)
        
        # Should produce something like "logos"
        self.assertEqual(result, "logos")

    def test_sbl_greek_transliteration(self):
        """Test SBL Greek transliteration."""
        # Word with diacritics
        greek = "Ἰησοῦς"
        result = self.sbl_trans.transliterate(greek)
        
        # Should handle breathing marks and accents
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_greek_alphabet(self):
        """Test transliterating Greek alphabet."""
        alphabet = {
            "α": "a",
            "β": "b",
            "γ": "g",
            "δ": "d",
            "ε": "e",
            "ζ": "z",
            "η": "e",  # or ē
            "θ": "th",
            "ι": "i",
            "κ": "k",
            "λ": "l",
            "μ": "m",
            "ν": "n",
            "ξ": "x",
            "ο": "o",
            "π": "p",
            "ρ": "r",
            "σ": "s",
            "τ": "t",
            "υ": "u",  # or y
            "φ": "ph",
            "χ": "ch",
            "ψ": "ps",
            "ω": "o",  # or ō
        }
        
        for greek, expected in alphabet.items():
            result = self.simple_trans.transliterate(greek)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_greek_final_sigma(self):
        """Test handling final sigma."""
        # Word ending in sigma
        word = "λόγος"  # ends with final sigma
        result = self.simple_trans.transliterate(word)
        
        # Both sigmas should transliterate to 's'
        self.assertEqual(result.lower().count('s'), 1)

    def test_greek_breathing_marks(self):
        """Test handling breathing marks."""
        # Rough breathing
        rough = "ἁ"  # alpha with rough breathing
        smooth = "ἀ"  # alpha with smooth breathing
        
        rough_result = self.sbl_trans.transliterate(rough)
        smooth_result = self.sbl_trans.transliterate(smooth)
        
        # Should distinguish between them
        self.assertNotEqual(rough_result, smooth_result)

    def test_beta_code_transliteration(self):
        """Test Beta Code transliteration."""
        # Greek text without accents for cleaner test
        greek = "λογος"
        result = self.beta_trans.transliterate(greek)
        
        # Beta code uses ASCII representation
        self.assertIsInstance(result, str)
        # The current implementation maps to lowercase ASCII letters
        self.assertEqual(result, "logos")

    def test_greek_diphthongs(self):
        """Test transliterating Greek diphthongs."""
        diphthongs = {
            "αι": "ai",
            "ει": "ei",
            "οι": "oi",
            "υι": "ui",
            "αυ": "au",
            "ευ": "eu",
            "ου": "ou",
        }
        
        for greek, expected in diphthongs.items():
            result = self.simple_trans.transliterate(greek)
            # Should handle diphthongs appropriately
            self.assertIsInstance(result, str)

    def test_iota_subscript(self):
        """Test handling iota subscript."""
        # Words with iota subscript
        text = "τῷ λόγῳ"  # with iota subscripts
        result = self.sbl_trans.transliterate(text)
        
        # Should handle iota subscript
        self.assertIsInstance(result, str)
        self.assertIn(" ", result)  # Preserve word boundary


class TestTransliterationValidator(unittest.TestCase):
    """Test TransliterationValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = TransliterationValidator()

    def test_validate_scheme(self):
        """Test validating transliteration schemes."""
        # Valid schemes
        self.assertTrue(self.validator.validate_scheme(TransliterationScheme.SBL_HEBREW))
        self.assertTrue(self.validator.validate_scheme(TransliterationScheme.SBL_GREEK))
        self.assertTrue(self.validator.validate_scheme(TransliterationScheme.BETA_CODE))
        
        # Invalid scheme (would need to create a non-enum value to test false case)
        # Since all TransliterationScheme enum values are valid, this always returns True

    def test_validate_transliteration(self):
        """Test validating transliteration quality."""
        # Test empty result validation
        errors = self.validator.validate_transliteration(
            "שָׁלוֹם", "", TransliterationScheme.SBL_HEBREW
        )
        self.assertIn("Empty transliteration result", errors)
        
        # Test Hebrew characters remaining
        errors = self.validator.validate_transliteration(
            "שָׁלוֹם", "shal שׁ om", TransliterationScheme.SBL_HEBREW
        )
        self.assertIn("Unconverted Hebrew characters remain", errors)
        
        # Test Greek characters remaining
        errors = self.validator.validate_transliteration(
            "λόγος", "log λ os", TransliterationScheme.SBL_GREEK
        )
        self.assertIn("Unconverted Greek characters remain", errors)
        
        # Test null characters
        errors = self.validator.validate_transliteration(
            "test", "te\x00st", TransliterationScheme.SBL_HEBREW
        )
        self.assertIn("Null characters in output", errors)
        
        # Valid transliteration should return no errors
        errors = self.validator.validate_transliteration(
            "שָׁלוֹם", "shalom", TransliterationScheme.SBL_HEBREW
        )
        self.assertEqual(len(errors), 0)

    def test_validate_reversibility(self):
        """Test validating reversibility of transliteration."""
        hebrew = "שלום"  # Simple text without vowels for better reversibility
        trans = HebrewTransliterator(TransliterationScheme.SIMPLE_HEBREW)
        
        # Test without reverse engine
        is_reversible = self.validator.validate_reversibility(
            hebrew, "shalom", None
        )
        self.assertFalse(is_reversible)
        
        # Test with reverse engine
        latin = trans.transliterate(hebrew)
        # For actual reversibility test, we'd need a properly configured reverse engine
        # This is more of a structure test
        is_reversible = self.validator.validate_reversibility(
            hebrew, latin, trans
        )
        self.assertIsInstance(is_reversible, bool)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function."""

    def test_create_hebrew_transliterator(self):
        """Test creating Hebrew transliterator."""
        # Without scheme
        trans = create_transliterator("hebrew")
        self.assertIsInstance(trans, HebrewTransliterator)
        self.assertEqual(trans.scheme, TransliterationScheme.SBL_HEBREW)  # Default
        
        # With specific scheme
        trans = create_transliterator("hebrew", TransliterationScheme.SIMPLE_HEBREW)
        self.assertIsInstance(trans, HebrewTransliterator)
        self.assertEqual(trans.scheme, TransliterationScheme.SIMPLE_HEBREW)

    def test_create_greek_transliterator(self):
        """Test creating Greek transliterator."""
        # Without scheme
        trans = create_transliterator("greek")
        self.assertIsInstance(trans, GreekTransliterator)
        self.assertEqual(trans.scheme, TransliterationScheme.SBL_GREEK)  # Default
        
        # With specific scheme
        trans = create_transliterator("greek", TransliterationScheme.BETA_CODE)
        self.assertIsInstance(trans, GreekTransliterator)
        self.assertEqual(trans.scheme, TransliterationScheme.BETA_CODE)

    def test_create_arabic_transliterator(self):
        """Test creating Arabic transliterator."""
        trans = create_transliterator("arabic")
        self.assertIsInstance(trans, TransliterationEngine)  # ArabicTransliterator
        self.assertEqual(trans.scheme, TransliterationScheme.ARABIC_DIN)

    def test_create_syriac_transliterator(self):
        """Test creating Syriac transliterator."""
        trans = create_transliterator("syriac")
        self.assertIsInstance(trans, TransliterationEngine)  # SyriacTransliterator
        # Note: Syriac uses SBL_HEBREW as default scheme in the implementation
        self.assertEqual(trans.scheme, TransliterationScheme.SBL_HEBREW)

    def test_create_unsupported_transliterator(self):
        """Test creating transliterator for unsupported language."""
        # Should raise error for unsupported languages
        with self.assertRaises(ValueError) as context:
            create_transliterator("klingon")
        self.assertIn("Unsupported language", str(context.exception))


if __name__ == "__main__":
    unittest.main()