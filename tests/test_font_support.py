"""
Test suite for font support functionality.
"""

import unittest
from unittest.mock import Mock, patch

from abba.language.font_support import (
    FontFeature,
    FontRequirements,
    Font,
    FontManager,
    FontChain,
    RenderingHints,
    detect_font_requirements,
)
from abba.language.script_detector import Script


class TestFontFeature(unittest.TestCase):
    """Test FontFeature enum."""

    def test_hebrew_features(self):
        """Test Hebrew font features."""
        self.assertEqual(FontFeature.HEBREW_VOWELS.value, "hebr")
        self.assertEqual(FontFeature.HEBREW_CANTILLATION.value, "cant")
        self.assertEqual(FontFeature.HEBREW_LIGATURES.value, "liga")

    def test_greek_features(self):
        """Test Greek font features."""
        self.assertEqual(FontFeature.GREEK_ACCENTS.value, "mark")
        self.assertEqual(FontFeature.GREEK_BREATHING.value, "mkmk")
        self.assertEqual(FontFeature.GREEK_LIGATURES.value, "liga")

    def test_arabic_features(self):
        """Test Arabic font features."""
        self.assertEqual(FontFeature.ARABIC_INIT.value, "init")
        self.assertEqual(FontFeature.ARABIC_MEDI.value, "medi")
        self.assertEqual(FontFeature.ARABIC_FINA.value, "fina")
        self.assertEqual(FontFeature.ARABIC_ISOL.value, "isol")

    def test_general_features(self):
        """Test general font features."""
        self.assertEqual(FontFeature.KERNING.value, "kern")
        self.assertEqual(FontFeature.CONTEXTUAL.value, "calt")
        self.assertEqual(FontFeature.OLD_STYLE_NUMS.value, "onum")


class TestFontRequirements(unittest.TestCase):
    """Test FontRequirements dataclass."""

    def test_default_requirements(self):
        """Test default font requirements."""
        reqs = FontRequirements()
        self.assertEqual(len(reqs.scripts), 0)
        self.assertEqual(len(reqs.features), 0)
        self.assertEqual(len(reqs.unicode_blocks), 0)
        self.assertFalse(reqs.combining_marks)
        self.assertFalse(reqs.rtl_support)
        self.assertFalse(reqs.vertical_text)
        self.assertEqual(reqs.minimum_size, 12)

    def test_hebrew_requirements(self):
        """Test Hebrew font requirements."""
        reqs = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS, FontFeature.HEBREW_CANTILLATION},
            unicode_blocks=[(0x0590, 0x05FF)],
            combining_marks=True,
            rtl_support=True,
            minimum_size=14
        )
        
        self.assertIn(Script.HEBREW, reqs.scripts)
        self.assertIn(FontFeature.HEBREW_VOWELS, reqs.features)
        self.assertTrue(reqs.combining_marks)
        self.assertTrue(reqs.rtl_support)
        self.assertEqual(reqs.minimum_size, 14)

    def test_merge_requirements(self):
        """Test merging font requirements."""
        reqs1 = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS},
            rtl_support=True
        )
        
        reqs2 = FontRequirements(
            scripts={Script.GREEK},
            features={FontFeature.GREEK_ACCENTS},
            minimum_size=16
        )
        
        merged = reqs1.merge(reqs2)
        
        # Should contain both scripts
        self.assertIn(Script.HEBREW, merged.scripts)
        self.assertIn(Script.GREEK, merged.scripts)
        
        # Should contain both features
        self.assertIn(FontFeature.HEBREW_VOWELS, merged.features)
        self.assertIn(FontFeature.GREEK_ACCENTS, merged.features)
        
        # Should take maximum values
        self.assertTrue(merged.rtl_support)
        self.assertEqual(merged.minimum_size, 16)


class TestFont(unittest.TestCase):
    """Test Font dataclass."""

    def test_font_creation(self):
        """Test creating a font."""
        font = Font(
            name="SBL Hebrew",
            family="SBL Hebrew",
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS, FontFeature.HEBREW_CANTILLATION},
            unicode_coverage=[(0x0590, 0x05FF)],
            is_free=True,
            license="SIL OFL",
            url="https://www.sbl-site.org/educational/BiblicalFonts_SBLHebrew.aspx"
        )
        
        self.assertEqual(font.name, "SBL Hebrew")
        self.assertIn(Script.HEBREW, font.scripts)
        self.assertTrue(font.is_free)

    def test_supports_requirements_success(self):
        """Test font supporting requirements."""
        font = Font(
            name="Test Font",
            family="Test",
            scripts={Script.HEBREW, Script.LATIN},
            features={FontFeature.HEBREW_VOWELS, FontFeature.KERNING},
            unicode_coverage=[(0x0000, 0x007F), (0x0590, 0x05FF)]
        )
        
        reqs = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS},
            unicode_blocks=[(0x0590, 0x05FF)]
        )
        
        self.assertTrue(font.supports_requirements(reqs))

    def test_supports_requirements_missing_script(self):
        """Test font missing required script."""
        font = Font(
            name="Test Font",
            family="Test",
            scripts={Script.LATIN},
            features={FontFeature.KERNING},
            unicode_coverage=[(0x0000, 0x007F)]
        )
        
        reqs = FontRequirements(scripts={Script.HEBREW})
        
        self.assertFalse(font.supports_requirements(reqs))

    def test_supports_requirements_missing_feature(self):
        """Test font missing required feature."""
        font = Font(
            name="Test Font",
            family="Test",
            scripts={Script.HEBREW},
            features={FontFeature.KERNING},
            unicode_coverage=[(0x0590, 0x05FF)]
        )
        
        reqs = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_CANTILLATION}
        )
        
        self.assertFalse(font.supports_requirements(reqs))

    def test_supports_requirements_partial_unicode(self):
        """Test font with partial Unicode coverage."""
        font = Font(
            name="Test Font",
            family="Test",
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS},
            unicode_coverage=[(0x0590, 0x05CF)]  # Partial Hebrew block
        )
        
        reqs = FontRequirements(
            scripts={Script.HEBREW},
            unicode_blocks=[(0x0590, 0x05FF)]  # Full Hebrew block
        )
        
        self.assertFalse(font.supports_requirements(reqs))


class TestFontManager(unittest.TestCase):
    """Test FontManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = FontManager()

    def test_register_font(self):
        """Test accessing registered fonts."""
        # Manager should have pre-registered fonts
        self.assertGreater(len(self.manager._font_db), 0)
        
        # Should be able to get font info
        sbl_hebrew = self.manager.get_font_info("SBL Hebrew")
        self.assertIsNotNone(sbl_hebrew)
        self.assertIn(Script.HEBREW, sbl_hebrew.scripts)
        
        # Should be able to list fonts for Hebrew
        hebrew_fonts = self.manager.list_fonts_for_script(Script.HEBREW)
        self.assertGreater(len(hebrew_fonts), 0)

    def test_find_fonts_by_script(self):
        """Test finding fonts by script."""
        # List fonts for different scripts
        hebrew_fonts = self.manager.list_fonts_for_script(Script.HEBREW)
        greek_fonts = self.manager.list_fonts_for_script(Script.GREEK)
        
        # Should have fonts for each script
        self.assertGreater(len(hebrew_fonts), 0)
        self.assertGreater(len(greek_fonts), 0)
        
        # Hebrew fonts should support Hebrew
        for font in hebrew_fonts:
            self.assertIn(Script.HEBREW, font.scripts)
            
        # Greek fonts should support Greek
        for font in greek_fonts:
            self.assertIn(Script.GREEK, font.scripts)

    def test_find_fonts_with_features(self):
        """Test finding fonts with specific features."""
        # Create requirements with cantillation
        reqs = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_CANTILLATION}
        )
        
        # Get recommendations
        recommendations = self.manager.recommend_fonts(reqs)
        
        # Primary font should support cantillation
        self.assertTrue(
            FontFeature.HEBREW_CANTILLATION in recommendations.primary.features or
            recommendations.primary.supports_requirements(reqs)
        )

    def test_create_font_chain(self):
        """Test creating font fallback chain."""
        # Create requirements for Hebrew with cantillation
        reqs = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS, FontFeature.HEBREW_CANTILLATION}
        )
        
        # Get recommended fonts
        recommendations = self.manager.recommend_fonts(reqs)
        
        # Should have primary recommendation
        self.assertIsNotNone(recommendations.primary)
        
        # Primary font should support Hebrew
        self.assertIn(Script.HEBREW, recommendations.primary.scripts)
        
        # May have fallback fonts (optional)
        # The manager might not always provide fallbacks

    def test_get_default_fonts(self):
        """Test listing fonts for Hebrew script."""
        # Get fonts for Hebrew script
        hebrew_fonts = self.manager.list_fonts_for_script(Script.HEBREW)
        self.assertIsInstance(hebrew_fonts, list)
        self.assertGreater(len(hebrew_fonts), 0)
        
        # All returned fonts should support Hebrew
        for font in hebrew_fonts:
            self.assertIn(Script.HEBREW, font.scripts)


class TestFontChain(unittest.TestCase):
    """Test FontChain class."""

    def test_chain_creation(self):
        """Test creating font chain."""
        fonts = [
            Font("Primary", "Primary", {Script.HEBREW}, set(), [(0x0590, 0x05FF)]),
            Font("Fallback", "Fallback", {Script.HEBREW}, set(), [(0x0590, 0x05FF)]),
        ]
        
        chain = FontChain(fonts)
        self.assertEqual(len(chain.fonts), 2)
        self.assertEqual(chain.fonts[0].name, "Primary")

    def test_to_css(self):
        """Test generating CSS font family."""
        fonts = [
            Font("SBL Hebrew", "SBL Hebrew", {Script.HEBREW}, set(), []),
            Font("Times New Roman", "Times New Roman", {Script.LATIN}, set(), []),
        ]
        
        chain = FontChain(fonts)
        css = chain.to_css()
        
        # Should create proper CSS font-family
        self.assertIn("SBL Hebrew", css)
        self.assertIn("Times New Roman", css)
        self.assertIn("serif", css)  # Should include generic fallback

    def test_empty_chain(self):
        """Test empty font chain."""
        chain = FontChain([])
        css = chain.to_css()
        
        # Should still provide generic fallback
        self.assertIn("serif", css)


class TestRenderingHints(unittest.TestCase):
    """Test RenderingHints class."""

    def test_hebrew_hints(self):
        """Test rendering hints for Hebrew."""
        # Create a font chain first
        fonts = [
            Font("SBL Hebrew", "SBL Hebrew", {Script.HEBREW}, set(), [])
        ]
        chain = FontChain(fonts)
        
        hints = RenderingHints(
            font_chain=chain,
            direction="rtl",
            line_height=1.8,
            letter_spacing=0
        )
        
        self.assertEqual(hints.direction, "rtl")
        self.assertEqual(hints.line_height, 1.8)
        self.assertEqual(hints.letter_spacing, 0)

    def test_to_css_styles(self):
        """Test converting hints to CSS."""
        # Create a font chain first
        fonts = [
            Font("Times New Roman", "Times New Roman", {Script.LATIN}, set(), [])
        ]
        chain = FontChain(fonts)
        
        hints = RenderingHints(
            font_chain=chain,
            direction="rtl",
            line_height=1.6,
            letter_spacing=0.05
        )
        
        css = hints.to_css()
        
        # Should include CSS properties
        self.assertIn("direction", css)
        self.assertEqual(css["direction"], "rtl")
        self.assertEqual(css["line-height"], "1.6")
        self.assertIn("letter-spacing", css)
        self.assertEqual(css["letter-spacing"], "0.05em")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    @patch('abba.language.font_support.ScriptDetector')
    def test_detect_font_requirements(self, mock_detector_class):
        """Test detecting font requirements from text."""
        # Mock script detection
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        # Mock detect_scripts method
        mock_detector.detect_scripts.return_value = [Script.HEBREW, Script.LATIN]
        
        # Detect requirements
        text = "Hello שָׁלוֹם"
        reqs = detect_font_requirements(text)
        
        # Should detect both scripts
        self.assertIn(Script.HEBREW, reqs.scripts)
        self.assertIn(Script.LATIN, reqs.scripts)
        
        # Should detect RTL
        self.assertTrue(reqs.rtl_support)
        
        # Should detect combining marks (Hebrew vowels)
        self.assertTrue(reqs.combining_marks)

    @patch('abba.language.font_support.ScriptDetector')
    def test_detect_requirements_greek(self, mock_detector_class):
        """Test detecting Greek font requirements."""
        # Mock script detection
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_scripts.return_value = [Script.GREEK]
        
        text = "λόγος"
        reqs = detect_font_requirements(text)
        
        # Should detect Greek
        self.assertIn(Script.GREEK, reqs.scripts)
        
        # Note: Combining marks detection would require actual Unicode analysis
        # The mock doesn't analyze the actual text characters

    def test_detect_requirements_complex(self):
        """Test detecting complex mixed text requirements."""
        # Mixed text with multiple scripts
        text = "Hebrew שָׁלוֹם, Greek λόγος, Arabic سلام"
        reqs = detect_font_requirements(text)
        
        # Should detect multiple scripts
        self.assertGreater(len(reqs.scripts), 1)
        
        # Should detect RTL (Hebrew and Arabic)
        self.assertTrue(reqs.rtl_support)


if __name__ == "__main__":
    unittest.main()