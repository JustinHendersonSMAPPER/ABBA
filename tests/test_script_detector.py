"""
Test suite for script detection functionality.
"""

import unittest
from abba.language.script_detector import Script, ScriptRange, ScriptDetector


class TestScript(unittest.TestCase):
    """Test Script enum."""

    def test_script_values(self):
        """Test script enum values."""
        self.assertEqual(Script.HEBREW.value, "hebrew")
        self.assertEqual(Script.GREEK.value, "greek")
        self.assertEqual(Script.LATIN.value, "latin")
        self.assertEqual(Script.ARABIC.value, "arabic")

    def test_all_scripts_defined(self):
        """Test that all expected scripts are defined."""
        expected_scripts = [
            "hebrew", "greek", "latin", "arabic", "syriac",
            "coptic", "ethiopic", "armenian", "georgian",
            "cyrillic", "devanagari", "chinese", "common", "unknown"
        ]
        actual_scripts = [script.value for script in Script]
        for expected in expected_scripts:
            self.assertIn(expected, actual_scripts)


class TestScriptRange(unittest.TestCase):
    """Test ScriptRange dataclass."""

    def test_script_range_creation(self):
        """Test creating a script range."""
        range_obj = ScriptRange(
            script=Script.HEBREW,
            start=0,
            end=5,
            text="שָׁלוֹם",
        )
        self.assertEqual(range_obj.script, Script.HEBREW)
        self.assertEqual(range_obj.start, 0)
        self.assertEqual(range_obj.end, 5)
        self.assertEqual(range_obj.text, "שָׁלוֹם")

    def test_script_range_length(self):
        """Test calculating range length."""
        range_obj = ScriptRange(
            script=Script.GREEK,
            start=10,
            end=25,
            text="ἐν ἀρχῇ ἦν ὁ λόγος"
        )
        self.assertEqual(range_obj.length, 15)


class TestScriptDetector(unittest.TestCase):
    """Test ScriptDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = ScriptDetector()

    def test_detect_scripts_hebrew(self):
        """Test detecting Hebrew script."""
        text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        scripts = self.detector.detect_scripts(text)
        
        self.assertIn(Script.HEBREW, scripts)

    def test_detect_scripts_greek(self):
        """Test detecting Greek script."""
        text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
        scripts = self.detector.detect_scripts(text, min_confidence=0.5)
        
        self.assertIn(Script.GREEK, scripts)

    def test_detect_scripts_latin(self):
        """Test detecting Latin script."""
        text = "In principio erat Verbum"
        scripts = self.detector.detect_scripts(text)
        
        self.assertIn(Script.LATIN, scripts)

    def test_is_mixed_script(self):
        """Test detecting mixed scripts."""
        # Single script
        self.assertFalse(self.detector.is_mixed_script("Hello world"))
        self.assertFalse(self.detector.is_mixed_script("שָׁלוֹם"))
        
        # Mixed scripts
        self.assertTrue(self.detector.is_mixed_script("Hello שָׁלוֹם"))
        self.assertTrue(self.detector.is_mixed_script("λόγος and דָּבָר"))

    def test_segment_text(self):
        """Test segmenting text by script."""
        text = "The word אֱלֹהִים (Elohim)"
        segments = self.detector.segment_by_script(text)
        
        self.assertGreater(len(segments), 1)
        
        # Check segments
        scripts = {seg.script for seg in segments}
        self.assertIn(Script.LATIN, scripts)
        self.assertIn(Script.HEBREW, scripts)

    def test_get_char_script(self):
        """Test getting script of single character."""
        # Hebrew
        self.assertEqual(self.detector.get_char_script('א'), Script.HEBREW)
        self.assertEqual(self.detector.get_char_script('ש'), Script.HEBREW)
        
        # Greek
        self.assertEqual(self.detector.get_char_script('α'), Script.GREEK)
        self.assertEqual(self.detector.get_char_script('ω'), Script.GREEK)
        
        # Latin
        self.assertEqual(self.detector.get_char_script('a'), Script.LATIN)
        self.assertEqual(self.detector.get_char_script('Z'), Script.LATIN)
        
        # Arabic
        self.assertEqual(self.detector.get_char_script('ا'), Script.ARABIC)
        
        # Common
        self.assertEqual(self.detector.get_char_script(' '), Script.COMMON)
        self.assertEqual(self.detector.get_char_script('1'), Script.COMMON)

    def test_count_scripts(self):
        """Test counting characters by script."""
        text = "Hello שָׁלוֹם world"
        counts = self.detector.count_scripts(text)
        
        self.assertIn(Script.LATIN, counts)
        self.assertIn(Script.HEBREW, counts)
        self.assertIn(Script.COMMON, counts)  # Spaces
        
        # Check counts make sense
        self.assertGreater(counts[Script.LATIN], 5)
        self.assertGreater(counts[Script.HEBREW], 3)

    def test_segment_with_punctuation(self):
        """Test segmenting with punctuation."""
        text = "Genesis 1:1 - בְּרֵאשִׁית"
        segments = self.detector.segment_by_script(text, merge_common=True)
        
        # Should handle punctuation
        self.assertGreater(len(segments), 0)
        
        # Find Hebrew segment
        hebrew_found = False
        for seg in segments:
            if seg.script == Script.HEBREW:
                hebrew_found = True
                self.assertIn("בְּרֵאשִׁית", seg.text)
        self.assertTrue(hebrew_found)

    def test_empty_text(self):
        """Test with empty text."""
        self.assertEqual(self.detector.detect_scripts(""), [])
        self.assertEqual(self.detector.segment_by_script(""), [])
        self.assertFalse(self.detector.is_mixed_script(""))
        self.assertEqual(self.detector.count_scripts(""), {})

    def test_complex_mixed_text(self):
        """Test complex mixed script text."""
        text = """
        Hebrew: בְּרֵאשִׁית
        Greek: λόγος
        Arabic: في البدء
        """
        
        scripts = self.detector.detect_scripts(text, min_confidence=0.05)
        
        # Should detect all scripts
        self.assertIn(Script.HEBREW, scripts)
        self.assertIn(Script.GREEK, scripts)
        self.assertIn(Script.ARABIC, scripts)
        self.assertIn(Script.LATIN, scripts)
        
        # Should be mixed
        self.assertTrue(self.detector.is_mixed_script(text))


if __name__ == "__main__":
    unittest.main()