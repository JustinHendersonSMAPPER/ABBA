"""
Script detection for biblical texts.

Provides automatic detection of scripts (Hebrew, Greek, Latin, etc.) in mixed
text, supporting proper rendering and processing of multilingual content.
"""

import re
import unicodedata
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import defaultdict


class Script(Enum):
    """Supported script types."""

    # Biblical languages
    HEBREW = "hebrew"
    GREEK = "greek"
    LATIN = "latin"
    ARABIC = "arabic"
    SYRIAC = "syriac"
    COPTIC = "coptic"
    ETHIOPIC = "ethiopic"
    ARMENIAN = "armenian"
    GEORGIAN = "georgian"

    # Other scripts
    CYRILLIC = "cyrillic"
    DEVANAGARI = "devanagari"
    CHINESE = "chinese"

    # Generic
    COMMON = "common"  # Punctuation, numbers
    UNKNOWN = "unknown"


@dataclass
class ScriptRange:
    """A range of text in a specific script."""

    script: Script
    start: int
    end: int
    text: str
    confidence: float = 1.0

    @property
    def length(self) -> int:
        """Get length of range."""
        return self.end - self.start


class ScriptDetector:
    """Detect scripts in text."""

    # Unicode script ranges
    SCRIPT_RANGES = {
        Script.HEBREW: [
            (0x0590, 0x05FF),  # Hebrew
            (0xFB1D, 0xFB4F),  # Hebrew Presentation Forms
        ],
        Script.GREEK: [
            (0x0370, 0x03FF),  # Greek and Coptic
            (0x1F00, 0x1FFF),  # Greek Extended
        ],
        Script.LATIN: [
            (0x0041, 0x005A),  # Basic Latin uppercase
            (0x0061, 0x007A),  # Basic Latin lowercase
            (0x00C0, 0x00FF),  # Latin-1 Supplement
            (0x0100, 0x017F),  # Latin Extended-A
            (0x0180, 0x024F),  # Latin Extended-B
        ],
        Script.ARABIC: [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ],
        Script.SYRIAC: [
            (0x0700, 0x074F),  # Syriac
            (0x0860, 0x086F),  # Syriac Supplement
        ],
        Script.COPTIC: [
            (0x2C80, 0x2CFF),  # Coptic
            (0x0370, 0x03FF),  # Greek and Coptic (shared)
        ],
        Script.ETHIOPIC: [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
        ],
        Script.ARMENIAN: [
            (0x0530, 0x058F),  # Armenian
            (0xFB00, 0xFB17),  # Armenian ligatures
        ],
        Script.GEORGIAN: [
            (0x10A0, 0x10FF),  # Georgian
            (0x2D00, 0x2D2F),  # Georgian Supplement
        ],
        Script.CYRILLIC: [
            (0x0400, 0x04FF),  # Cyrillic
            (0x0500, 0x052F),  # Cyrillic Supplement
        ],
    }

    # Common characters (punctuation, spaces, etc.)
    COMMON_RANGES = [
        (0x0000, 0x0040),  # Control and basic punctuation
        (0x005B, 0x0060),  # More punctuation
        (0x007B, 0x00BF),  # More punctuation and symbols
        (0x2000, 0x206F),  # General punctuation
        (0x3000, 0x303F),  # CJK punctuation
    ]

    # Script-specific punctuation
    SCRIPT_PUNCTUATION = {
        Script.HEBREW: set("\u05BE\u05C0\u05C3\u05C6\u05F3\u05F4"),
        Script.GREEK: set("\u0387\u00B7"),
        Script.ARABIC: set("\u060C\u061B\u061E\u061F\u066A-\u066D"),
        Script.SYRIAC: set("\u0700-\u070D"),
    }

    def __init__(self):
        """Initialize the detector."""
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build efficient lookup tables for script detection."""
        self.char_to_script: Dict[int, Script] = {}

        # Build character -> script mapping
        for script, ranges in self.SCRIPT_RANGES.items():
            for start, end in ranges:
                for cp in range(start, end + 1):
                    # Coptic overlaps with Greek
                    if script == Script.COPTIC and cp < 0x03E2:
                        continue  # Skip Greek letters
                    self.char_to_script[cp] = script

        # Build common character set
        self.common_chars = set()
        for start, end in self.COMMON_RANGES:
            for cp in range(start, end + 1):
                self.common_chars.add(cp)

    def detect_script(self, text: str) -> Script:
        """Detect primary script of text.

        Args:
            text: Text to analyze

        Returns:
            Primary script
        """
        script_counts = self.count_scripts(text)

        if not script_counts or all(v == 0 for v in script_counts.values()):
            return Script.UNKNOWN

        # Remove common script from consideration
        script_counts.pop(Script.COMMON, None)

        # Return most frequent script
        return max(script_counts, key=script_counts.get)

    def detect_scripts(self, text: str, min_confidence: float = 0.8) -> List[Script]:
        """Detect all scripts present in text.

        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected scripts
        """
        script_counts = self.count_scripts(text)
        total_chars = sum(script_counts.values())

        if total_chars == 0:
            return []

        # Calculate confidence for each script
        scripts = []
        for script, count in script_counts.items():
            if script == Script.COMMON:
                continue

            confidence = count / total_chars
            if confidence >= min_confidence:
                scripts.append(script)

        return sorted(scripts, key=lambda s: script_counts[s], reverse=True)

    def segment_by_script(self, text: str, merge_common: bool = True) -> List[ScriptRange]:
        """Segment text by script changes.

        Args:
            text: Text to segment
            merge_common: Whether to merge common chars with adjacent scripts

        Returns:
            List of script ranges
        """
        if not text:
            return []

        ranges = []
        current_script = None
        current_start = 0
        current_text = []

        for i, char in enumerate(text):
            char_script = self.get_char_script(char)

            # Handle common characters
            if merge_common and char_script == Script.COMMON:
                if current_script:
                    # Add to current range
                    current_text.append(char)
                    continue
                else:
                    # Look ahead to next non-common character
                    next_script = self._find_next_script(text, i + 1)
                    if next_script and next_script != Script.COMMON:
                        char_script = next_script

            # Check for script change
            if char_script != current_script:
                # Save current range
                if current_script and current_text:
                    ranges.append(
                        ScriptRange(
                            script=current_script,
                            start=current_start,
                            end=i,
                            text="".join(current_text),
                        )
                    )

                # Start new range
                current_script = char_script
                current_start = i
                current_text = [char]
            else:
                current_text.append(char)

        # Save final range
        if current_script and current_text:
            ranges.append(
                ScriptRange(
                    script=current_script,
                    start=current_start,
                    end=len(text),
                    text="".join(current_text),
                )
            )

        return ranges

    def get_char_script(self, char: str) -> Script:
        """Get script of a single character.

        Args:
            char: Character to check

        Returns:
            Character's script
        """
        if not char:
            return Script.UNKNOWN

        cp = ord(char)

        # Check lookup table
        if cp in self.char_to_script:
            return self.char_to_script[cp]

        # Check if common
        if cp in self.common_chars:
            return Script.COMMON

        # Check Unicode script property
        try:
            script_name = unicodedata.name(char, "").split()[0].lower()
            script_map = {
                "hebrew": Script.HEBREW,
                "greek": Script.GREEK,
                "latin": Script.LATIN,
                "arabic": Script.ARABIC,
                "syriac": Script.SYRIAC,
                "coptic": Script.COPTIC,
                "ethiopic": Script.ETHIOPIC,
                "armenian": Script.ARMENIAN,
                "georgian": Script.GEORGIAN,
                "cyrillic": Script.CYRILLIC,
            }

            for key, script in script_map.items():
                if key in script_name:
                    return script
        except:
            pass

        return Script.UNKNOWN

    def count_scripts(self, text: str) -> Dict[Script, int]:
        """Count characters by script.

        Args:
            text: Text to analyze

        Returns:
            Map of script to character count
        """
        counts = defaultdict(int)

        for char in text:
            script = self.get_char_script(char)
            counts[script] += 1

        return dict(counts)

    def _find_next_script(self, text: str, start: int) -> Optional[Script]:
        """Find next non-common script after position."""
        for i in range(start, len(text)):
            script = self.get_char_script(text[i])
            if script != Script.COMMON:
                return script
        return None

    def is_mixed_script(self, text: str) -> bool:
        """Check if text contains multiple scripts.

        Args:
            text: Text to check

        Returns:
            True if multiple scripts present
        """
        scripts = self.detect_scripts(text, min_confidence=0.05)
        return len(scripts) > 1

    def get_script_info(self, script: Script) -> Dict[str, any]:
        """Get information about a script.

        Args:
            script: Script to query

        Returns:
            Script information
        """
        rtl_scripts = {Script.HEBREW, Script.ARABIC, Script.SYRIAC}

        info = {
            "name": script.value,
            "direction": "rtl" if script in rtl_scripts else "ltr",
            "requires_shaping": script in {Script.ARABIC, Script.SYRIAC},
            "combining_marks": script
            in {Script.HEBREW, Script.ARABIC, Script.SYRIAC, Script.GREEK},
        }

        # Add Unicode ranges
        if script in self.SCRIPT_RANGES:
            info["unicode_ranges"] = self.SCRIPT_RANGES[script]

        return info

    def detect_dominant_direction(self, text: str) -> str:
        """Detect dominant text direction.

        Args:
            text: Text to analyze

        Returns:
            'ltr', 'rtl', or 'mixed'
        """
        script_counts = self.count_scripts(text)

        rtl_count = sum(
            count
            for script, count in script_counts.items()
            if script in {Script.HEBREW, Script.ARABIC, Script.SYRIAC}
        )

        ltr_count = sum(
            count
            for script, count in script_counts.items()
            if script in {Script.LATIN, Script.GREEK, Script.CYRILLIC}
        )

        total = rtl_count + ltr_count
        if total == 0:
            return "ltr"  # Default

        rtl_ratio = rtl_count / total

        if rtl_ratio > 0.8:
            return "rtl"
        elif rtl_ratio < 0.2:
            return "ltr"
        else:
            return "mixed"

    def normalize_for_script(self, text: str, script: Script) -> str:
        """Apply script-specific normalization.

        Args:
            text: Text to normalize
            script: Target script

        Returns:
            Normalized text
        """
        if script == Script.HEBREW:
            # Remove Hebrew points if needed
            # This is a simplified version
            return text

        elif script == Script.GREEK:
            # Normalize Greek diacritics
            return unicodedata.normalize("NFC", text)

        elif script == Script.ARABIC:
            # Arabic normalization
            return unicodedata.normalize("NFC", text)

        else:
            # Default normalization
            return unicodedata.normalize("NFC", text)
