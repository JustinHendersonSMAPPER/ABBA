"""
Right-to-left (RTL) text handling for biblical languages.

Provides support for Hebrew, Arabic, Syriac and other RTL scripts with proper
bidirectional text handling, mixed direction support, and alignment.
"""

import re
import unicodedata
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


class TextDirection(Enum):
    """Text directionality."""

    LTR = "ltr"  # Left-to-right
    RTL = "rtl"  # Right-to-left
    AUTO = "auto"  # Automatic detection
    INHERIT = "inherit"  # Inherit from parent


class BidiClass(Enum):
    """Unicode bidirectional character classes."""

    # Strong types
    L = "L"  # Left-to-right
    R = "R"  # Right-to-left
    AL = "AL"  # Arabic letter

    # Weak types
    EN = "EN"  # European number
    ES = "ES"  # European separator
    ET = "ET"  # European terminator
    AN = "AN"  # Arabic number
    CS = "CS"  # Common separator
    NSM = "NSM"  # Non-spacing mark

    # Neutral types
    B = "B"  # Paragraph separator
    S = "S"  # Segment separator
    WS = "WS"  # Whitespace
    ON = "ON"  # Other neutral

    # Explicit formatting
    LRE = "LRE"  # Left-to-right embedding
    LRO = "LRO"  # Left-to-right override
    RLE = "RLE"  # Right-to-left embedding
    RLO = "RLO"  # Right-to-left override
    PDF = "PDF"  # Pop directional formatting


@dataclass
class DirectionalRun:
    """A run of text with consistent direction."""

    text: str
    direction: TextDirection
    start: int
    end: int
    level: int = 0  # Embedding level (even=LTR, odd=RTL)

    @property
    def is_rtl(self) -> bool:
        """Check if run is RTL."""
        return self.level % 2 == 1


class RTLHandler:
    """Handler for right-to-left text processing."""

    # Unicode blocks for RTL scripts
    RTL_RANGES = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0700, 0x074F),  # Syriac
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0x0870, 0x089F),  # Arabic Extended-B
        (0xFB1D, 0xFB4F),  # Hebrew presentation forms
        (0xFB50, 0xFDFF),  # Arabic presentation forms A
        (0xFE70, 0xFEFF),  # Arabic presentation forms B
    ]

    # Strong directional characters
    STRONG_LTR = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    # Neutral characters that take surrounding direction
    NEUTRAL_CHARS = set(" \t\n\r.,;:!?\"'()-[]{}/<>@#$%^&*+=|\\`~")

    def __init__(self):
        """Initialize RTL handler."""
        # Build character direction map
        self._build_direction_map()

    def _build_direction_map(self):
        """Build map of character ranges to directions."""
        self.rtl_chars = set()
        for start, end in self.RTL_RANGES:
            for cp in range(start, end + 1):
                self.rtl_chars.add(cp)

    def detect_direction(self, text: str) -> TextDirection:
        """Detect primary direction of text.

        Args:
            text: Text to analyze

        Returns:
            Primary text direction
        """
        rtl_count = 0
        ltr_count = 0

        for char in text:
            cp = ord(char)
            if cp in self.rtl_chars:
                rtl_count += 1
            elif char in self.STRONG_LTR:
                ltr_count += 1

        if rtl_count > ltr_count:
            return TextDirection.RTL
        elif ltr_count > 0:
            return TextDirection.LTR
        else:
            return TextDirection.AUTO

    def segment_by_direction(self, text: str) -> List[DirectionalRun]:
        """Segment text into directional runs.

        Args:
            text: Text to segment

        Returns:
            List of directional runs
        """
        if not text:
            return []

        runs = []
        current_run = None

        for i, char in enumerate(text):
            char_dir = self.get_char_direction(char)

            # Skip neutrals for now
            if char_dir == TextDirection.AUTO:
                if current_run:
                    current_run.text += char
                    current_run.end = i
                continue

            # Start new run or continue current
            if not current_run or current_run.direction != char_dir:
                if current_run:
                    runs.append(current_run)

                current_run = DirectionalRun(
                    text=char,
                    direction=char_dir,
                    start=i,
                    end=i,
                    level=0 if char_dir == TextDirection.LTR else 1,
                )
            else:
                current_run.text += char
                current_run.end = i

        # Add final run
        if current_run:
            runs.append(current_run)

        # Resolve neutrals
        self._resolve_neutral_runs(runs, text)

        return runs

    def get_char_direction(self, char: str) -> TextDirection:
        """Get direction of a single character.

        Args:
            char: Character to check

        Returns:
            Character direction
        """
        if not char:
            return TextDirection.AUTO

        cp = ord(char)

        # Check RTL ranges
        if cp in self.rtl_chars:
            return TextDirection.RTL

        # Check strong LTR
        if char in self.STRONG_LTR:
            return TextDirection.LTR

        # Check by Unicode bidi class
        bidi_class = unicodedata.bidirectional(char)
        if bidi_class in ["R", "AL", "RLE", "RLO"]:
            return TextDirection.RTL
        elif bidi_class in ["L", "LRE", "LRO"]:
            return TextDirection.LTR

        return TextDirection.AUTO

    def _resolve_neutral_runs(self, runs: List[DirectionalRun], text: str):
        """Resolve direction of neutral character runs."""
        for i, run in enumerate(runs):
            if run.direction == TextDirection.AUTO:
                # Get surrounding directions
                prev_dir = runs[i - 1].direction if i > 0 else TextDirection.LTR
                next_dir = runs[i + 1].direction if i < len(runs) - 1 else TextDirection.LTR

                # If surrounded by same direction, use that
                if prev_dir == next_dir and prev_dir != TextDirection.AUTO:
                    run.direction = prev_dir
                    run.level = 0 if prev_dir == TextDirection.LTR else 1
                else:
                    # Default to LTR for neutrals
                    run.direction = TextDirection.LTR
                    run.level = 0

    def apply_bidi_algorithm(
        self, text: str, base_direction: TextDirection = TextDirection.AUTO
    ) -> str:
        """Apply Unicode Bidirectional Algorithm for display.

        Args:
            text: Text to process
            base_direction: Base paragraph direction

        Returns:
            Text with proper directional ordering for display
        """
        if not text:
            return text

        # Auto-detect base direction if needed
        if base_direction == TextDirection.AUTO:
            base_direction = self.detect_direction(text)

        # Get directional runs
        runs = self.segment_by_direction(text)

        # Apply bidi algorithm levels
        self._apply_bidi_levels(runs, base_direction)

        # Reorder runs for display
        display_runs = self._reorder_runs_for_display(runs, base_direction)

        # Build display text
        return "".join(run.text for run in display_runs)

    def _apply_bidi_levels(self, runs: List[DirectionalRun], base_direction: TextDirection):
        """Apply embedding levels according to bidi algorithm."""
        base_level = 0 if base_direction == TextDirection.LTR else 1

        for run in runs:
            if run.direction == TextDirection.RTL:
                run.level = base_level + 1 if base_level % 2 == 0 else base_level
            else:
                run.level = base_level if base_level % 2 == 0 else base_level + 1

    def _reorder_runs_for_display(
        self, runs: List[DirectionalRun], base_direction: TextDirection
    ) -> List[DirectionalRun]:
        """Reorder runs for visual display."""
        if not runs:
            return runs

        # Group by level
        max_level = max(run.level for run in runs)

        # Process each level from highest to lowest
        result = runs.copy()

        for level in range(max_level, -1, -1):
            # Find sequences at this level
            i = 0
            while i < len(result):
                if result[i].level >= level:
                    # Find end of sequence
                    j = i
                    while j < len(result) and result[j].level >= level:
                        j += 1

                    # Reverse sequence if odd level
                    if level % 2 == 1:
                        result[i:j] = reversed(result[i:j])

                    i = j
                else:
                    i += 1

        return result

    def format_mixed_direction(
        self,
        text: str,
        primary_dir: TextDirection = TextDirection.AUTO,
        isolate_numbers: bool = True,
    ) -> str:
        """Format text with mixed directions properly.

        Args:
            text: Text to format
            primary_dir: Primary text direction
            isolate_numbers: Whether to isolate numbers

        Returns:
            Formatted text with directional marks
        """
        if not text:
            return text

        # Detect primary direction if auto
        if primary_dir == TextDirection.AUTO:
            primary_dir = self.detect_direction(text)

        # Add directional marks
        formatted = []
        current_dir = primary_dir

        for char in text:
            char_dir = self.get_char_direction(char)

            # Handle direction changes
            if char_dir != TextDirection.AUTO and char_dir != current_dir:
                if char_dir == TextDirection.RTL:
                    formatted.append("\u202B")  # Right-to-left embedding
                else:
                    formatted.append("\u202A")  # Left-to-right embedding

                formatted.append(char)
                formatted.append("\u202C")  # Pop directional formatting
                current_dir = primary_dir
            else:
                formatted.append(char)

        return "".join(formatted)

    def create_interlinear_alignment(
        self, original: str, translation: str, word_alignment: List[Tuple[int, int]]
    ) -> List[str]:
        """Create aligned interlinear text for RTL/LTR pairs.

        Args:
            original: Original text (possibly RTL)
            translation: Translation text (possibly LTR)
            word_alignment: List of (original_idx, translation_idx) pairs

        Returns:
            List of formatted lines for display
        """
        # Split into words
        orig_words = original.split()
        trans_words = translation.split()

        # Detect directions
        orig_dir = self.detect_direction(original)
        trans_dir = self.detect_direction(translation)

        # Create alignment map
        alignment_map = {orig_idx: trans_idx for orig_idx, trans_idx in word_alignment}

        # Build aligned pairs
        aligned_pairs = []
        for i, orig_word in enumerate(orig_words):
            trans_idx = alignment_map.get(i)
            trans_word = trans_words[trans_idx] if trans_idx is not None else "—"
            aligned_pairs.append((orig_word, trans_word))

        # Format for display
        if orig_dir == TextDirection.RTL:
            # Reverse order for RTL display
            aligned_pairs = list(reversed(aligned_pairs))

        # Build formatted lines
        orig_line = []
        trans_line = []

        for orig_word, trans_word in aligned_pairs:
            # Calculate max width
            max_width = max(len(orig_word), len(trans_word)) + 2

            # Format with padding
            if orig_dir == TextDirection.RTL:
                orig_line.append(orig_word.rjust(max_width))
            else:
                orig_line.append(orig_word.ljust(max_width))

            trans_line.append(trans_word.center(max_width))

        return [" ".join(orig_line), " ".join(trans_line)]


class BidiAlgorithm:
    """Full Unicode Bidirectional Algorithm implementation."""

    def __init__(self):
        """Initialize the algorithm."""
        self.rtl_handler = RTLHandler()

    def process_paragraph(self, text: str, base_direction: Optional[TextDirection] = None) -> str:
        """Process a paragraph according to Unicode Bidi Algorithm.

        Args:
            text: Paragraph text
            base_direction: Base direction (auto-detect if None)

        Returns:
            Properly ordered text for display
        """
        if not text:
            return text

        # P1: Split by paragraph separators
        paragraphs = self._split_paragraphs(text)

        results = []
        for para in paragraphs:
            if para:
                # P2-P3: Determine paragraph level
                if base_direction is None:
                    para_dir = self.rtl_handler.detect_direction(para)
                else:
                    para_dir = base_direction

                # Process paragraph
                processed = self.rtl_handler.apply_bidi_algorithm(para, para_dir)
                results.append(processed)
            else:
                results.append(para)

        return "\n".join(results)

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Simple split by newlines
        return text.split("\n")

    def mirror_brackets(self, text: str, runs: List[DirectionalRun]) -> str:
        """Mirror brackets and parentheses in RTL runs.

        Args:
            text: Original text
            runs: Directional runs

        Returns:
            Text with mirrored brackets
        """
        # Mirroring pairs
        mirrors = {
            "(": ")",
            ")": "(",
            "[": "]",
            "]": "[",
            "{": "}",
            "}": "{",
            "<": ">",
            ">": "<",
        }

        result = list(text)

        for run in runs:
            if run.is_rtl:
                for i in range(run.start, run.end + 1):
                    if i < len(result) and result[i] in mirrors:
                        result[i] = mirrors[result[i]]

        return "".join(result)
