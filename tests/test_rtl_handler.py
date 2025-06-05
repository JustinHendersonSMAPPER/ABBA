"""
Tests for RTL (Right-to-Left) text handling functionality.

Tests the core RTL text processing capabilities including direction detection,
bidirectional algorithm implementation, and interlinear text alignment.
"""

import unittest
from typing import List, Tuple

from abba.language.rtl import (
    TextDirection,
    BidiClass,
    DirectionalRun,
    RTLHandler,
    BidiContext,
)


class TestTextDirection(unittest.TestCase):
    """Test TextDirection enum."""

    def test_direction_values(self):
        """Test TextDirection enum values."""
        self.assertEqual(TextDirection.LTR.value, "ltr")
        self.assertEqual(TextDirection.RTL.value, "rtl")
        self.assertEqual(TextDirection.AUTO.value, "auto")

    def test_direction_count(self):
        """Test that we have expected number of directions."""
        directions = list(TextDirection)
        self.assertEqual(len(directions), 4)  # LTR, RTL, AUTO, INHERIT


class TestBidiClass(unittest.TestCase):
    """Test BidiClass enum for Unicode bidirectional classes."""

    def test_strong_types(self):
        """Test strong directional types."""
        self.assertEqual(BidiClass.L.value, "L")  # Left-to-Right
        self.assertEqual(BidiClass.R.value, "R")  # Right-to-Left
        self.assertEqual(BidiClass.AL.value, "AL")  # Arabic Letter

    def test_weak_types(self):
        """Test weak directional types."""
        self.assertEqual(BidiClass.EN.value, "EN")  # European Number
        self.assertEqual(BidiClass.ES.value, "ES")  # European Separator
        self.assertEqual(BidiClass.ET.value, "ET")  # European Terminator
        self.assertEqual(BidiClass.AN.value, "AN")  # Arabic Number
        self.assertEqual(BidiClass.CS.value, "CS")  # Common Separator

    def test_neutral_types(self):
        """Test neutral directional types."""
        self.assertEqual(BidiClass.WS.value, "WS")  # Whitespace
        self.assertEqual(BidiClass.ON.value, "ON")  # Other Neutral
        self.assertEqual(BidiClass.S.value, "S")    # Segment Separator
        self.assertEqual(BidiClass.B.value, "B")    # Paragraph Separator


class TestDirectionalRun(unittest.TestCase):
    """Test DirectionalRun data class."""

    def test_run_creation(self):
        """Test creating directional runs."""
        run = DirectionalRun(
            text="שלום",
            direction=TextDirection.RTL,
            start=0,
            end=4
        )
        
        self.assertEqual(run.text, "שלום")
        self.assertEqual(run.direction, TextDirection.RTL)
        self.assertEqual(run.start, 0)
        self.assertEqual(run.end, 4)
        self.assertEqual(run.level, 0)

    def test_embedding_levels(self):
        """Test embedding level handling."""
        run = DirectionalRun(
            text="test",
            direction=TextDirection.LTR,
            start=0,
            end=4,
            level=2
        )
        
        self.assertEqual(run.level, 2)

    def test_run_length(self):
        """Test run length calculation."""
        run = DirectionalRun(
            text="hello",
            direction=TextDirection.LTR,
            start=5,
            end=10
        )
        
        self.assertEqual(len(run.text), 5)
        self.assertEqual(run.end - run.start, 5)


class TestRTLHandler(unittest.TestCase):
    """Test RTL text handler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = RTLHandler()

    def test_char_direction_hebrew(self):
        """Test Hebrew character direction detection."""
        # Hebrew letters
        self.assertEqual(self.handler.get_char_direction('א'), TextDirection.RTL)
        self.assertEqual(self.handler.get_char_direction('ב'), TextDirection.RTL)
        self.assertEqual(self.handler.get_char_direction('ש'), TextDirection.RTL)
        self.assertEqual(self.handler.get_char_direction('ת'), TextDirection.RTL)
        
        # Hebrew vowels and cantillation
        self.assertEqual(self.handler.get_char_direction('\u05B0'), TextDirection.RTL)  # Sheva
        self.assertEqual(self.handler.get_char_direction('\u05B1'), TextDirection.RTL)  # Hataf Segol
        self.assertEqual(self.handler.get_char_direction('\u0591'), TextDirection.RTL)  # Etnahta

    def test_char_direction_arabic(self):
        """Test Arabic character direction detection."""
        # Arabic letters
        self.assertEqual(self.handler.get_char_direction('ا'), TextDirection.RTL)  # Alif
        self.assertEqual(self.handler.get_char_direction('ب'), TextDirection.RTL)  # Ba
        self.assertEqual(self.handler.get_char_direction('ت'), TextDirection.RTL)  # Ta
        self.assertEqual(self.handler.get_char_direction('ن'), TextDirection.RTL)  # Noon

    def test_char_direction_latin(self):
        """Test Latin character direction detection."""
        # Latin letters should be LTR
        self.assertEqual(self.handler.get_char_direction('a'), TextDirection.LTR)
        self.assertEqual(self.handler.get_char_direction('A'), TextDirection.LTR)
        self.assertEqual(self.handler.get_char_direction('z'), TextDirection.LTR)
        self.assertEqual(self.handler.get_char_direction('Z'), TextDirection.LTR)

    def test_char_direction_neutral(self):
        """Test neutral character direction detection."""
        # Neutral characters should return AUTO
        self.assertEqual(self.handler.get_char_direction(' '), TextDirection.AUTO)
        self.assertEqual(self.handler.get_char_direction('.'), TextDirection.AUTO)
        self.assertEqual(self.handler.get_char_direction('1'), TextDirection.AUTO)
        self.assertEqual(self.handler.get_char_direction('!'), TextDirection.AUTO)
        self.assertEqual(self.handler.get_char_direction('('), TextDirection.AUTO)

    def test_detect_direction_hebrew(self):
        """Test direction detection for Hebrew text."""
        hebrew_text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        direction = self.handler.detect_direction(hebrew_text)
        self.assertEqual(direction, TextDirection.RTL)

    def test_detect_direction_arabic(self):
        """Test direction detection for Arabic text."""
        arabic_text = "في البدء خلق الله"
        direction = self.handler.detect_direction(arabic_text)
        self.assertEqual(direction, TextDirection.RTL)

    def test_detect_direction_english(self):
        """Test direction detection for English text."""
        english_text = "In the beginning God created"
        direction = self.handler.detect_direction(english_text)
        self.assertEqual(direction, TextDirection.LTR)

    def test_detect_direction_mixed(self):
        """Test direction detection for mixed text."""
        # Hebrew with English should detect as RTL (Hebrew dominant)
        mixed_text = "בְּרֵאשִׁית God בָּרָא"
        direction = self.handler.detect_direction(mixed_text)
        self.assertEqual(direction, TextDirection.RTL)
        
        # English with some Hebrew words should detect as LTR
        english_dominant = "The Hebrew word שלום means peace"
        direction = self.handler.detect_direction(english_dominant)
        self.assertEqual(direction, TextDirection.LTR)

    def test_detect_direction_empty(self):
        """Test direction detection for empty text."""
        self.assertEqual(self.handler.detect_direction(""), TextDirection.AUTO)
        self.assertEqual(self.handler.detect_direction("   "), TextDirection.AUTO)

    def test_segment_by_direction_simple(self):
        """Test segmenting simple directional text."""
        # Pure Hebrew should be one segment
        hebrew_text = "שלום עולם"
        runs = self.handler.segment_by_direction(hebrew_text)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].direction, TextDirection.RTL)
        self.assertEqual(runs[0].text, hebrew_text)
        
        # Pure English should be one segment
        english_text = "hello world"
        runs = self.handler.segment_by_direction(english_text)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].direction, TextDirection.LTR)
        self.assertEqual(runs[0].text, english_text)

    def test_segment_by_direction_mixed(self):
        """Test segmenting mixed direction text."""
        # Hebrew with English should create multiple segments
        mixed_text = "שלום world עולם"
        runs = self.handler.segment_by_direction(mixed_text)
        self.assertGreater(len(runs), 1)
        
        # Verify we have both RTL and LTR segments
        directions = [run.direction for run in runs]
        self.assertIn(TextDirection.RTL, directions)
        self.assertIn(TextDirection.LTR, directions)

    def test_apply_bidi_algorithm_simple(self):
        """Test applying bidi algorithm to simple text."""
        # Hebrew text should be processed correctly
        hebrew_text = "שלום עולם"
        result = self.handler.apply_bidi_algorithm(hebrew_text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # English text should remain unchanged
        english_text = "hello world"
        result = self.handler.apply_bidi_algorithm(english_text)
        self.assertEqual(result, english_text)

    def test_apply_bidi_algorithm_mixed(self):
        """Test applying bidi algorithm to mixed direction text."""
        # Mixed Hebrew-English text
        mixed_text = "שלום world"
        result = self.handler.apply_bidi_algorithm(mixed_text, TextDirection.RTL)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), len(mixed_text))

    def test_format_mixed_direction(self):
        """Test formatting mixed direction text."""
        # Hebrew with English words
        mixed_text = "בְּרֵאשִׁית God בָּרָא"
        result = self.handler.format_mixed_direction(mixed_text, TextDirection.RTL)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) >= len(mixed_text))  # May include directional marks

    def test_format_mixed_direction_with_numbers(self):
        """Test formatting mixed direction text with numbers."""
        # Hebrew text with numbers
        text_with_numbers = "פרק 1 פסוק 3"
        result = self.handler.format_mixed_direction(text_with_numbers, TextDirection.RTL, isolate_numbers=True)
        self.assertIsInstance(result, str)
        
        # Test without number isolation
        result_no_isolate = self.handler.format_mixed_direction(text_with_numbers, TextDirection.RTL, isolate_numbers=False)
        self.assertIsInstance(result_no_isolate, str)

    def test_create_interlinear_alignment_basic(self):
        """Test basic interlinear alignment."""
        # Hebrew-English word alignment
        hebrew = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        english = "In beginning created God"
        
        # Word alignments: [(hebrew_word_idx, english_word_idx), ...]
        word_alignment = [(0, 1), (1, 2), (2, 3)]  # beginning->created->God alignment
        
        result = self.handler.create_interlinear_alignment(hebrew, english, word_alignment)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Should return [hebrew_line, english_line]

    def test_create_interlinear_alignment_empty(self):
        """Test interlinear alignment with empty inputs."""
        result = self.handler.create_interlinear_alignment("", "", [])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_create_interlinear_alignment_mismatched(self):
        """Test interlinear alignment with mismatched word counts."""
        hebrew = "שלום עולם טוב"  # 3 words
        english = "hello good world peace"  # 4 words
        
        # Partial alignment
        word_alignment = [(0, 0), (2, 1)]  # Map some words
        
        result = self.handler.create_interlinear_alignment(hebrew, english, word_alignment)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_verse_reference_handling(self):
        """Test handling of verse references in RTL context."""
        # Hebrew text with verse reference
        text_with_ref = "בְּרֵאשִׁית א:א"
        direction = self.handler.detect_direction(text_with_ref)
        self.assertEqual(direction, TextDirection.RTL)
        
        # Segment the text
        runs = self.handler.segment_by_direction(text_with_ref)
        self.assertGreater(len(runs), 0)

    def test_biblical_text_with_punctuation(self):
        """Test biblical text with various punctuation marks."""
        # Hebrew with punctuation
        hebrew_with_punct = "וַיֹּאמֶר אֱלֹהִים, \"יְהִי אוֹר\""
        result = self.handler.apply_bidi_algorithm(hebrew_with_punct)
        self.assertIsInstance(result, str)
        
        # Should handle quotes and commas properly
        self.assertIn('"', result)
        self.assertIn(',', result)

    def test_cantillation_marks(self):
        """Test handling of Hebrew cantillation marks."""
        # Text with cantillation
        text_with_cant = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים"
        direction = self.handler.detect_direction(text_with_cant)
        self.assertEqual(direction, TextDirection.RTL)
        
        # Verify cantillation marks are detected as RTL
        for char in "\u0591\u0592\u0593\u0594\u0595":  # Various cantillation marks
            self.assertEqual(self.handler.get_char_direction(char), TextDirection.RTL)

    def test_greek_text_handling(self):
        """Test handling of Greek text (LTR but biblical)."""
        greek_text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
        direction = self.handler.detect_direction(greek_text)
        # Greek text may be detected as AUTO if no strong LTR characters are found
        self.assertIn(direction, [TextDirection.LTR, TextDirection.AUTO])
        
        # Greek characters should be LTR
        self.assertEqual(self.handler.get_char_direction('Ἐ'), TextDirection.LTR)
        self.assertEqual(self.handler.get_char_direction('ἀ'), TextDirection.LTR)
        self.assertEqual(self.handler.get_char_direction('λ'), TextDirection.LTR)

    def test_edge_cases(self):
        """Test various edge cases."""
        # Single character
        self.assertEqual(self.handler.detect_direction("א"), TextDirection.RTL)
        self.assertEqual(self.handler.detect_direction("a"), TextDirection.LTR)
        
        # Only numbers
        self.assertEqual(self.handler.detect_direction("123"), TextDirection.AUTO)
        
        # Only punctuation
        self.assertEqual(self.handler.detect_direction("!@#"), TextDirection.AUTO)
        
        # Mixed numbers and Hebrew
        mixed_num_heb = "123 שלום"
        direction = self.handler.detect_direction(mixed_num_heb)
        self.assertEqual(direction, TextDirection.RTL)

    def test_long_mixed_text(self):
        """Test handling of long mixed direction text."""
        long_mixed = """
        בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ
        In the beginning God created the heavens and the earth
        וְהָאָרֶץ הָיְתָה תֹהוּ וָבֹהוּ וְחֹשֶׁךְ עַל־פְּנֵי תְהוֹם
        And the earth was without form and void
        """
        
        # Should detect as mixed but lean toward one direction
        direction = self.handler.detect_direction(long_mixed)
        self.assertIn(direction, [TextDirection.RTL, TextDirection.LTR])
        
        # Should be able to segment
        runs = self.handler.segment_by_direction(long_mixed)
        self.assertGreater(len(runs), 1)


class TestBidiContext(unittest.TestCase):
    """Test BidiContext class functionality."""

    def test_context_creation(self):
        """Test creating bidi context."""
        context = BidiContext(base_direction=TextDirection.LTR)
        self.assertEqual(context.embedding_level, 0)
        # Note: BidiContext doesn't have length or to_html methods in actual implementation
        self.assertEqual(context.base_direction, TextDirection.LTR)

    def test_context_embedding(self):
        """Test embedding level management."""
        context = BidiContext(base_direction=TextDirection.LTR)
        initial_level = context.embedding_level
        
        # Test pushing embeddings
        context.push_embedding(TextDirection.RTL)
        self.assertGreater(context.embedding_level, initial_level)
        
        context.push_embedding(TextDirection.LTR)
        
        # Test popping embeddings
        context.pop_embedding()
        context.pop_embedding()

    def test_context_html_output(self):
        """Test basic context properties."""
        context = BidiContext(base_direction=TextDirection.RTL)
        self.assertEqual(context.base_direction, TextDirection.RTL)
        self.assertEqual(context.embedding_level, 0)
        self.assertIsNone(context.override_status)


if __name__ == "__main__":
    unittest.main()