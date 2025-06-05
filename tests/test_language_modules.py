"""
Basic tests for language modules to ensure they work.
"""

import unittest
from abba.language.script_detector import Script, ScriptRange, ScriptDetector
from abba.language.rtl import TextDirection, BidiClass, DirectionalRun, RTLHandler
from abba.language.unicode_utils import NormalizationForm, CombiningSequence, UnicodeNormalizer
from abba.language.transliteration import TransliterationScheme, TransliterationRule, TransliterationEngine
from abba.language.font_support import FontFeature, FontRequirements, Font, FontManager


class TestLanguageModulesBasic(unittest.TestCase):
    """Basic tests for language modules."""

    def test_script_detector_basics(self):
        """Test basic script detection."""
        detector = ScriptDetector()
        
        # Basic detection
        self.assertEqual(detector.get_char_script('א'), Script.HEBREW)
        self.assertEqual(detector.get_char_script('α'), Script.GREEK)
        self.assertEqual(detector.get_char_script('a'), Script.LATIN)
        
        # Mixed script detection
        self.assertTrue(detector.is_mixed_script("Hello שָׁלוֹם"))
        self.assertFalse(detector.is_mixed_script("Hello world"))

    def test_rtl_handler_basics(self):
        """Test basic RTL handling."""
        handler = RTLHandler()
        
        # Basic RTL detection
        self.assertEqual(handler.get_char_direction('א'), TextDirection.RTL)
        self.assertEqual(handler.get_char_direction('a'), TextDirection.LTR)
        
        # Direction detection
        self.assertEqual(handler.detect_direction("שָׁלוֹם"), TextDirection.RTL)
        self.assertEqual(handler.detect_direction("Hello"), TextDirection.LTR)

    def test_unicode_normalizer_basics(self):
        """Test basic Unicode normalization."""
        normalizer = UnicodeNormalizer()
        
        # Basic normalization
        text = "café"
        normalized = normalizer.normalize(text, NormalizationForm.NFD)
        self.assertIsInstance(normalized, str)
        
        # Clean text
        text = "Hello—world"
        cleaned = normalizer.clean_text(text)
        self.assertNotIn("—", cleaned)

    def test_font_requirements_basics(self):
        """Test basic font requirements."""
        reqs = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS},
            rtl_support=True
        )
        
        self.assertIn(Script.HEBREW, reqs.scripts)
        self.assertIn(FontFeature.HEBREW_VOWELS, reqs.features)
        self.assertTrue(reqs.rtl_support)

    def test_transliteration_basics(self):
        """Test basic transliteration concepts."""
        # Test enum
        self.assertEqual(TransliterationScheme.SBL_HEBREW.value, "sbl_hebrew")
        
        # Test rule
        rule = TransliterationRule(
            source="א",
            target="'",
            priority=1
        )
        self.assertEqual(rule.source, "א")
        self.assertEqual(rule.target, "'")


if __name__ == "__main__":
    unittest.main()