"""Tests for script detection."""

import pytest

from abba.language.script_detector import Script, ScriptDetector, ScriptRange


class TestScriptDetector:
    """Test script detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ScriptDetector()

    def test_detect_single_script(self):
        """Test detecting single script texts."""
        # Hebrew
        assert self.detector.detect_script("בְּרֵאשִׁית בָּרָא") == Script.HEBREW

        # Greek
        assert self.detector.detect_script("Ἐν ἀρχῇ ἦν ὁ λόγος") == Script.GREEK

        # Latin
        assert self.detector.detect_script("In principio erat") == Script.LATIN

        # Arabic
        assert self.detector.detect_script("بِسْمِ اللهِ") == Script.ARABIC

        # Syriac
        assert self.detector.detect_script("ܒܫܡܐ ܕܐܒܐ") == Script.SYRIAC

    def test_detect_scripts_multiple(self):
        """Test detecting multiple scripts in text."""
        # Hebrew and English
        mixed = "The word בְּרֵאשִׁית means beginning"
        scripts = self.detector.detect_scripts(mixed, min_confidence=0.1)

        assert Script.HEBREW in scripts
        assert Script.LATIN in scripts

        # Greek and Latin
        mixed_greek = "The Greek word λόγος means word"
        scripts_greek = self.detector.detect_scripts(mixed_greek, min_confidence=0.1)

        assert Script.GREEK in scripts_greek
        assert Script.LATIN in scripts_greek

    def test_get_char_script(self):
        """Test single character script detection."""
        # Hebrew characters
        assert self.detector.get_char_script("א") == Script.HEBREW
        assert self.detector.get_char_script("ת") == Script.HEBREW

        # Greek characters
        assert self.detector.get_char_script("α") == Script.GREEK
        assert self.detector.get_char_script("ω") == Script.GREEK

        # Latin characters
        assert self.detector.get_char_script("A") == Script.LATIN
        assert self.detector.get_char_script("z") == Script.LATIN

        # Common characters
        assert self.detector.get_char_script(" ") == Script.COMMON
        assert self.detector.get_char_script(",") == Script.COMMON
        assert self.detector.get_char_script("1") == Script.COMMON

    def test_segment_by_script(self):
        """Test segmentation by script changes."""
        text = "Genesis 1:1 בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        segments = self.detector.segment_by_script(text)

        assert len(segments) >= 2

        # Check Latin segment
        latin_seg = next(s for s in segments if s.script == Script.LATIN)
        assert "Genesis" in latin_seg.text

        # Check Hebrew segment
        hebrew_seg = next(s for s in segments if s.script == Script.HEBREW)
        assert "בְּרֵאשִׁית" in hebrew_seg.text

    def test_segment_with_merge_common(self):
        """Test segmentation with common character merging."""
        text = "Test (טֶסְט) test"

        # With merging
        segments_merged = self.detector.segment_by_script(text, merge_common=True)

        # Without merging
        segments_unmerged = self.detector.segment_by_script(text, merge_common=False)

        # Merged should have fewer segments
        assert len(segments_merged) <= len(segments_unmerged)

    def test_count_scripts(self):
        """Test counting characters by script."""
        text = "Hello שלום مرحبا"
        counts = self.detector.count_scripts(text)

        assert counts[Script.LATIN] == 5  # Hello
        assert counts[Script.HEBREW] == 4  # שלום
        assert counts[Script.ARABIC] == 5  # مرحبا
        assert counts[Script.COMMON] == 2  # Two spaces

    def test_is_mixed_script(self):
        """Test mixed script detection."""
        # Single script
        assert not self.detector.is_mixed_script("Hello World")
        assert not self.detector.is_mixed_script("שלום עולם")

        # Mixed scripts
        assert self.detector.is_mixed_script("Hello שלום")
        assert self.detector.is_mixed_script("Test λόγος")

    def test_get_script_info(self):
        """Test getting script information."""
        # Hebrew info
        hebrew_info = self.detector.get_script_info(Script.HEBREW)
        assert hebrew_info["name"] == "hebrew"
        assert hebrew_info["direction"] == "rtl"
        assert hebrew_info["combining_marks"] == True

        # Greek info
        greek_info = self.detector.get_script_info(Script.GREEK)
        assert greek_info["name"] == "greek"
        assert greek_info["direction"] == "ltr"
        assert greek_info["combining_marks"] == True

        # Arabic info
        arabic_info = self.detector.get_script_info(Script.ARABIC)
        assert arabic_info["requires_shaping"] == True

    def test_detect_dominant_direction(self):
        """Test dominant direction detection."""
        # RTL dominant
        rtl_text = "שלום Hello עולם"
        assert self.detector.detect_dominant_direction(rtl_text) == "rtl"

        # LTR dominant
        ltr_text = "Hello שלום World"
        assert self.detector.detect_dominant_direction(ltr_text) == "ltr"

        # Mixed
        mixed_text = "Hello שלום"
        direction = self.detector.detect_dominant_direction(mixed_text)
        assert direction in ["ltr", "rtl", "mixed"]

    def test_coptic_detection(self):
        """Test Coptic script detection."""
        # Coptic-specific characters
        coptic_text = "ⲁⲃⲅⲇ"  # Coptic letters
        assert self.detector.detect_script(coptic_text) == Script.COPTIC

    def test_script_range_properties(self):
        """Test ScriptRange properties."""
        range_obj = ScriptRange(script=Script.HEBREW, start=10, end=20, text="טקסט עברי")

        assert range_obj.script == Script.HEBREW
        assert range_obj.length == 10
        assert range_obj.text == "טקסט עברי"
        assert range_obj.confidence == 1.0

    def test_empty_text_handling(self):
        """Test handling empty text."""
        assert self.detector.detect_script("") == Script.UNKNOWN
        assert self.detector.detect_scripts("") == []
        assert self.detector.segment_by_script("") == []
        assert not self.detector.is_mixed_script("")

    def test_unknown_script(self):
        """Test handling unknown scripts."""
        # Characters not in defined ranges
        unknown_text = "𐌀𐌁𐌂"  # Old Italic
        script = self.detector.detect_script(unknown_text)
        assert script == Script.UNKNOWN

    def test_presentation_forms(self):
        """Test handling presentation forms."""
        # Hebrew presentation forms
        hebrew_pres = "ﬠﬡﬢ"  # Hebrew presentation forms
        assert self.detector.detect_script(hebrew_pres) == Script.HEBREW

        # Arabic presentation forms
        arabic_pres = "ﻛﻠﻤﺔ"  # Arabic presentation forms
        assert self.detector.detect_script(arabic_pres) == Script.ARABIC

    def test_normalize_for_script(self):
        """Test script-specific normalization."""
        # Hebrew normalization
        hebrew_text = "בְּרֵאשִׁית"
        normalized = self.detector.normalize_for_script(hebrew_text, Script.HEBREW)
        assert normalized is not None

        # Greek normalization
        greek_text = "λόγος"
        normalized = self.detector.normalize_for_script(greek_text, Script.GREEK)
        assert normalized is not None
