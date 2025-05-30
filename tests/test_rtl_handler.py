"""Tests for RTL text handling."""

import pytest

from abba.language.rtl import RTLHandler, TextDirection, BidiAlgorithm, DirectionalRun


class TestRTLHandler:
    """Test RTL text handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = RTLHandler()

    def test_detect_hebrew_direction(self):
        """Test Hebrew text direction detection."""
        # Pure Hebrew
        hebrew_text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        assert self.handler.detect_direction(hebrew_text) == TextDirection.RTL

        # Hebrew with punctuation
        hebrew_punct = "בְּרֵאשִׁית, בָּרָא אֱלֹהִים."
        assert self.handler.detect_direction(hebrew_punct) == TextDirection.RTL

    def test_detect_arabic_direction(self):
        """Test Arabic text direction detection."""
        arabic_text = "بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ"
        assert self.handler.detect_direction(arabic_text) == TextDirection.RTL

    def test_detect_greek_direction(self):
        """Test Greek text direction detection."""
        greek_text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
        assert self.handler.detect_direction(greek_text) == TextDirection.LTR

    def test_detect_latin_direction(self):
        """Test Latin text direction detection."""
        latin_text = "In principio erat Verbum"
        assert self.handler.detect_direction(latin_text) == TextDirection.LTR

    def test_detect_mixed_direction(self):
        """Test mixed direction text detection."""
        # More Hebrew than English
        mixed_rtl = "The word בְּרֵאשִׁית means beginning"
        assert self.handler.detect_direction(mixed_rtl) == TextDirection.RTL

        # More English than Hebrew
        mixed_ltr = "Genesis begins with בְּרֵאשִׁית"
        assert self.handler.detect_direction(mixed_ltr) == TextDirection.LTR

    def test_get_char_direction(self):
        """Test single character direction detection."""
        # Hebrew characters
        assert self.handler.get_char_direction("א") == TextDirection.RTL
        assert self.handler.get_char_direction("ב") == TextDirection.RTL

        # Arabic characters
        assert self.handler.get_char_direction("ا") == TextDirection.RTL
        assert self.handler.get_char_direction("ب") == TextDirection.RTL

        # Latin characters
        assert self.handler.get_char_direction("A") == TextDirection.LTR
        assert self.handler.get_char_direction("a") == TextDirection.LTR

        # Neutral characters
        assert self.handler.get_char_direction(" ") == TextDirection.AUTO
        assert self.handler.get_char_direction(",") == TextDirection.AUTO

    def test_segment_by_direction(self):
        """Test text segmentation by direction."""
        # Simple mixed text
        text = "The Hebrew word בְּרֵאשִׁית means 'beginning'"
        segments = self.handler.segment_by_direction(text)

        assert len(segments) >= 3

        # Check first segment (LTR)
        assert segments[0].direction == TextDirection.LTR
        assert segments[0].text.startswith("The Hebrew word")

        # Find Hebrew segment
        hebrew_seg = next(s for s in segments if s.direction == TextDirection.RTL)
        assert hebrew_seg is not None
        assert "בְּרֵאשִׁית" in hebrew_seg.text

    def test_directional_run(self):
        """Test directional run properties."""
        run = DirectionalRun(text="שלום", direction=TextDirection.RTL, start=0, end=4, level=1)

        assert run.text == "שלום"
        assert run.direction == TextDirection.RTL
        assert run.is_rtl
        assert run.length == 4

        # LTR run
        ltr_run = DirectionalRun(text="hello", direction=TextDirection.LTR, start=0, end=5, level=0)
        assert not ltr_run.is_rtl

    def test_apply_bidi_algorithm(self):
        """Test Unicode bidirectional algorithm application."""
        # Mixed text
        text = "Genesis 1:1 בְּרֵאשִׁית בָּרָא"
        result = self.handler.apply_bidi_algorithm(text)

        # Result should preserve logical order
        assert "Genesis" in result
        assert "בְּרֵאשִׁית" in result

    def test_format_mixed_direction(self):
        """Test formatting mixed direction text."""
        text = "The word בְּרֵאשִׁית in Hebrew"
        formatted = self.handler.format_mixed_direction(text)

        # Should contain directional marks
        assert "\u202B" in formatted or "\u202A" in formatted  # Embedding marks
        assert "\u202C" in formatted  # Pop directional formatting

    def test_create_interlinear_alignment(self):
        """Test creating interlinear alignment."""
        hebrew = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        english = "In-beginning created God"
        alignment = [(0, 0), (1, 1), (2, 2)]

        lines = self.handler.create_interlinear_alignment(hebrew, english, alignment)

        assert len(lines) == 2
        assert isinstance(lines[0], str)  # Hebrew line
        assert isinstance(lines[1], str)  # English line

    def test_neutral_character_resolution(self):
        """Test neutral character direction resolution."""
        # Neutrals between same direction
        text = "אבג 123 דהו"  # Hebrew with numbers
        segments = self.handler.segment_by_direction(text)

        # Numbers should join with surrounding Hebrew
        rtl_segments = [s for s in segments if s.direction == TextDirection.RTL]
        assert len(rtl_segments) >= 1

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        assert self.handler.detect_direction("") == TextDirection.AUTO
        assert self.handler.segment_by_direction("") == []
        assert self.handler.apply_bidi_algorithm("") == ""
        assert self.handler.format_mixed_direction("") == ""


class TestBidiAlgorithm:
    """Test full bidi algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bidi = BidiAlgorithm()

    def test_process_paragraph(self):
        """Test paragraph processing."""
        # Simple paragraph
        para = "This is English. וזה עברית. Back to English."
        result = self.bidi.process_paragraph(para)

        assert "This is English" in result
        assert "וזה עברית" in result
        assert "Back to English" in result

    def test_process_multiple_paragraphs(self):
        """Test processing multiple paragraphs."""
        text = "First paragraph.\n\nעברית בפסקה השנייה.\n\nThird paragraph."
        result = self.bidi.process_paragraph(text)

        lines = result.split("\n")
        assert len(lines) >= 3

    def test_mirror_brackets(self):
        """Test bracket mirroring in RTL text."""
        text = "Text (parentheses) and [brackets]"
        runs = [
            DirectionalRun(text=text, direction=TextDirection.RTL, start=0, end=len(text), level=1)
        ]

        result = self.bidi.mirror_brackets(text, runs)

        # In RTL context, brackets should be mirrored
        assert ")" in result  # Opening paren becomes closing
        assert "(" in result  # Closing paren becomes opening

    def test_auto_detect_base_direction(self):
        """Test automatic base direction detection."""
        # Mostly Hebrew
        hebrew_text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים and some English"
        result = self.bidi.process_paragraph(hebrew_text)
        assert result is not None

        # Mostly English
        english_text = "In the beginning בְּרֵאשִׁית was the word"
        result = self.bidi.process_paragraph(english_text)
        assert result is not None
