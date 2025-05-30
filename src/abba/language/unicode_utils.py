"""
Unicode normalization and handling utilities.

Provides comprehensive Unicode support for biblical texts including normalization,
combining character handling, and script-specific processing.
"""

import unicodedata
import re
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


class NormalizationForm(Enum):
    """Unicode normalization forms."""

    NFC = "NFC"  # Canonical Decomposition, followed by Canonical Composition
    NFD = "NFD"  # Canonical Decomposition
    NFKC = "NFKC"  # Compatibility Decomposition, followed by Canonical Composition
    NFKD = "NFKD"  # Compatibility Decomposition

    # Custom forms for biblical texts
    HEBREW_CANONICAL = "HEBREW_CANONICAL"
    GREEK_CANONICAL = "GREEK_CANONICAL"


@dataclass
class CombiningSequence:
    """A base character with its combining marks."""

    base: str
    combiners: List[str]
    start_pos: int
    end_pos: int

    def to_string(self) -> str:
        """Convert to string."""
        return self.base + "".join(self.combiners)

    def normalize_order(self) -> "CombiningSequence":
        """Normalize combining mark order."""
        # Sort by combining class
        sorted_combiners = sorted(self.combiners, key=lambda c: unicodedata.combining(c))
        return CombiningSequence(self.base, sorted_combiners, self.start_pos, self.end_pos)


class UnicodeNormalizer:
    """Base Unicode normalizer for biblical texts."""

    def __init__(self):
        """Initialize the normalizer."""
        self.setup_mappings()

    def setup_mappings(self):
        """Setup character mappings and equivalences."""
        # Common problematic character mappings
        self.char_mappings = {
            # Various apostrophes to standard
            "\u2019": "'",  # Right single quotation mark
            "\u02BC": "'",  # Modifier letter apostrophe
            "\u02B9": "'",  # Modifier letter prime
            # Various dashes to standard
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u2012": "-",  # Figure dash
            # Spaces
            "\u00A0": " ",  # Non-breaking space
            "\u2009": " ",  # Thin space
            "\u200A": " ",  # Hair space
        }

    def normalize(self, text: str, form: NormalizationForm = NormalizationForm.NFC) -> str:
        """Normalize text to specified form.

        Args:
            text: Text to normalize
            form: Normalization form

        Returns:
            Normalized text
        """
        if not text:
            return text

        # Apply custom normalizations first
        if form == NormalizationForm.HEBREW_CANONICAL:
            return self.normalize_hebrew(text)
        elif form == NormalizationForm.GREEK_CANONICAL:
            return self.normalize_greek(text)

        # Standard Unicode normalization
        return unicodedata.normalize(form.value, text)

    def normalize_hebrew(self, text: str) -> str:
        """Apply Hebrew-specific normalization."""
        # Implemented in HebrewNormalizer
        return text

    def normalize_greek(self, text: str) -> str:
        """Apply Greek-specific normalization."""
        # Implemented in GreekNormalizer
        return text

    def clean_text(self, text: str) -> str:
        """Clean text of problematic characters.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Apply character mappings
        for old, new in self.char_mappings.items():
            text = text.replace(old, new)

        # Remove zero-width characters
        text = self.remove_zero_width(text)

        # Normalize whitespace
        text = self.normalize_whitespace(text)

        return text

    def remove_zero_width(self, text: str) -> str:
        """Remove zero-width characters."""
        # Zero-width characters to remove
        zero_width = [
            "\u200B",  # Zero-width space
            "\u200C",  # Zero-width non-joiner
            "\u200D",  # Zero-width joiner
            "\uFEFF",  # Zero-width no-break space (BOM)
        ]

        for char in zero_width:
            text = text.replace(char, "")

        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple spaces with single
        text = re.sub(r"\s+", " ", text)

        # Trim
        text = text.strip()

        return text

    def decompose_to_sequences(self, text: str) -> List[CombiningSequence]:
        """Decompose text to base + combining sequences.

        Args:
            text: Text to decompose

        Returns:
            List of combining sequences
        """
        # First normalize to NFD
        nfd_text = unicodedata.normalize("NFD", text)

        sequences = []
        current_seq = None
        pos = 0

        for i, char in enumerate(nfd_text):
            combining_class = unicodedata.combining(char)

            if combining_class == 0:  # Base character
                if current_seq:
                    sequences.append(current_seq)

                current_seq = CombiningSequence(base=char, combiners=[], start_pos=pos, end_pos=pos)
                pos += 1
            else:  # Combining character
                if current_seq:
                    current_seq.combiners.append(char)
                    current_seq.end_pos = pos
                # Ignore combining mark without base

        # Add final sequence
        if current_seq:
            sequences.append(current_seq)

        return sequences


class HebrewNormalizer(UnicodeNormalizer):
    """Hebrew-specific Unicode normalizer."""

    def __init__(self):
        """Initialize Hebrew normalizer."""
        super().__init__()
        self.setup_hebrew_specific()

    def setup_hebrew_specific(self):
        """Setup Hebrew-specific mappings."""
        # Hebrew Unicode blocks
        self.HEBREW_BLOCK = range(0x0590, 0x0600)
        self.HEBREW_PRESENTATION = range(0xFB1D, 0xFB50)

        # Hebrew vowel points (nikud)
        self.HEBREW_VOWELS = {
            0x05B0,  # Sheva
            0x05B1,  # Hataf Segol
            0x05B2,  # Hataf Patah
            0x05B3,  # Hataf Qamats
            0x05B4,  # Hiriq
            0x05B5,  # Tsere
            0x05B6,  # Segol
            0x05B7,  # Patah
            0x05B8,  # Qamats
            0x05B9,  # Holam
            0x05BB,  # Qubuts
            0x05C7,  # Qamats Qatan
        }

        # Hebrew accents (teamim)
        self.HEBREW_ACCENTS = set(range(0x0591, 0x05AF))

        # Special Hebrew characters
        self.MAQAF = "\u05BE"  # Hebrew hyphen
        self.SOF_PASUQ = "\u05C3"  # End of verse

    def normalize_hebrew(self, text: str) -> str:
        """Normalize Hebrew text."""
        # Decompose
        text = unicodedata.normalize("NFD", text)

        # Handle final forms
        text = self.normalize_final_forms(text)

        # Normalize vowel order
        text = self.normalize_vowel_order(text)

        # Handle special cases
        text = self.handle_holam_vav(text)
        text = self.handle_shin_sin_dot(text)

        # Recompose
        text = unicodedata.normalize("NFC", text)

        return text

    def normalize_final_forms(self, text: str) -> str:
        """Normalize Hebrew final letter forms."""
        # Final form mappings
        finals = {
            "\u05DA": "\u05DB",  # Final Kaf → Kaf
            "\u05DD": "\u05DE",  # Final Mem → Mem
            "\u05DF": "\u05E0",  # Final Nun → Nun
            "\u05E3": "\u05E4",  # Final Pe → Pe
            "\u05E5": "\u05E6",  # Final Tsadi → Tsadi
        }

        # Keep finals only at word end
        words = text.split()
        result = []

        for word in words:
            if word and word[-1] in finals.values():
                # Convert last letter to final form if applicable
                reverse_finals = {v: k for k, v in finals.items()}
                if word[-1] in reverse_finals:
                    word = word[:-1] + reverse_finals[word[-1]]
            result.append(word)

        return " ".join(result)

    def normalize_vowel_order(self, text: str) -> str:
        """Normalize order of Hebrew vowels and marks."""
        sequences = self.decompose_to_sequences(text)

        normalized = []
        for seq in sequences:
            # Separate vowels and accents
            vowels = []
            accents = []
            other = []

            for mark in seq.combiners:
                cp = ord(mark)
                if cp in self.HEBREW_VOWELS:
                    vowels.append(mark)
                elif cp in self.HEBREW_ACCENTS:
                    accents.append(mark)
                else:
                    other.append(mark)

            # Canonical order: base + vowels + accents + other
            normalized_seq = seq.base + "".join(vowels + accents + other)
            normalized.append(normalized_seq)

        return "".join(normalized)

    def handle_holam_vav(self, text: str) -> str:
        """Handle holam vav combinations."""
        # Holam (0x05B9) + Vav (0x05D5) special cases
        # Sometimes written as Vav + Holam dot above

        # Pattern: Vav with holam following
        text = re.sub(r"\u05D5\u05B9", "\uFB4B", text)  # Vav with holam

        return text

    def handle_shin_sin_dot(self, text: str) -> str:
        """Handle shin/sin dot normalization."""
        # Shin dot (0x05C1) and Sin dot (0x05C2)
        # Ensure they're in correct position relative to Shin (0x05E9)

        # Pattern: Shin followed by dot
        text = re.sub(r"(\u05E9)([\u05C1\u05C2])", r"\1\2", text)

        return text

    def strip_hebrew_points(
        self, text: str, keep_vowels: bool = False, keep_accents: bool = False
    ) -> str:
        """Strip Hebrew pointing from text.

        Args:
            text: Hebrew text
            keep_vowels: Whether to keep vowel points
            keep_accents: Whether to keep accent marks

        Returns:
            Text with specified points removed
        """
        result = []

        for char in text:
            cp = ord(char)

            # Skip unwanted marks
            if cp in self.HEBREW_VOWELS and not keep_vowels:
                continue
            if cp in self.HEBREW_ACCENTS and not keep_accents:
                continue

            result.append(char)

        return "".join(result)


class GreekNormalizer(UnicodeNormalizer):
    """Greek-specific Unicode normalizer."""

    def __init__(self):
        """Initialize Greek normalizer."""
        super().__init__()
        self.setup_greek_specific()

    def setup_greek_specific(self):
        """Setup Greek-specific mappings."""
        # Greek Unicode blocks
        self.GREEK_BLOCK = range(0x0370, 0x0400)
        self.GREEK_EXTENDED = range(0x1F00, 0x2000)

        # Greek diacritics
        self.TONOS = "\u0301"  # Acute accent
        self.OXIA = "\u0301"  # Same as tonos
        self.VARIA = "\u0300"  # Grave accent
        self.PERISPOMENI = "\u0342"  # Circumflex
        self.PSILI = "\u0313"  # Smooth breathing
        self.DASIA = "\u0314"  # Rough breathing
        self.YPOGEGRAMMENI = "\u0345"  # Iota subscript
        self.DIALYTIKA = "\u0308"  # Diaeresis

        # Final sigma
        self.SIGMA = "\u03C3"
        self.FINAL_SIGMA = "\u03C2"

    def normalize_greek(self, text: str) -> str:
        """Normalize Greek text."""
        # Decompose
        text = unicodedata.normalize("NFD", text)

        # Normalize breathing marks
        text = self.normalize_breathing_marks(text)

        # Normalize accents
        text = self.normalize_greek_accents(text)

        # Handle final sigma
        text = self.normalize_final_sigma(text)

        # Recompose
        text = unicodedata.normalize("NFC", text)

        return text

    def normalize_breathing_marks(self, text: str) -> str:
        """Normalize Greek breathing marks."""
        # Ensure breathing marks come before accents
        sequences = self.decompose_to_sequences(text)

        normalized = []
        for seq in sequences:
            breathing = []
            accents = []
            other = []

            for mark in seq.combiners:
                if mark in [self.PSILI, self.DASIA]:
                    breathing.append(mark)
                elif mark in [self.TONOS, self.VARIA, self.PERISPOMENI]:
                    accents.append(mark)
                else:
                    other.append(mark)

            # Canonical order: base + breathing + accent + other
            normalized_seq = seq.base + "".join(breathing + accents + other)
            normalized.append(normalized_seq)

        return "".join(normalized)

    def normalize_greek_accents(self, text: str) -> str:
        """Normalize Greek accent marks."""
        # Convert polytonic to monotonic if needed
        # Modern Greek uses only acute accent

        # For ancient Greek, preserve all accents
        return text

    def normalize_final_sigma(self, text: str) -> str:
        """Normalize final sigma usage."""
        # Use final sigma at word boundaries
        words = text.split()
        result = []

        for word in words:
            if word and word[-1] == self.SIGMA:
                word = word[:-1] + self.FINAL_SIGMA

            # Also handle sigma before punctuation
            word = re.sub(r'\u03C3(?=[\s.,;:!?\'")\]])', self.FINAL_SIGMA, word)

            result.append(word)

        return " ".join(result)

    def strip_greek_accents(self, text: str, keep_breathing: bool = False) -> str:
        """Strip Greek accents from text.

        Args:
            text: Greek text
            keep_breathing: Whether to keep breathing marks

        Returns:
            Text with accents removed
        """
        # Decompose first
        text = unicodedata.normalize("NFD", text)

        # Remove diacritics
        result = []
        for char in text:
            if char in [self.TONOS, self.VARIA, self.PERISPOMENI]:
                continue
            if not keep_breathing and char in [self.PSILI, self.DASIA]:
                continue
            result.append(char)

        # Recompose
        return unicodedata.normalize("NFC", "".join(result))


class DiacriticHandler:
    """Handler for diacritical marks across scripts."""

    def __init__(self):
        """Initialize the handler."""
        self.setup_diacritic_data()

    def setup_diacritic_data(self):
        """Setup diacritic categories."""
        # Categories of diacritics
        self.COMBINING_ABOVE = set(range(0x0300, 0x0315))
        self.COMBINING_BELOW = set(range(0x0316, 0x0333))
        self.COMBINING_OVERLAY = set(range(0x0334, 0x0338))

    def count_diacritics(self, text: str) -> Dict[str, int]:
        """Count diacritics in text by category."""
        counts = {"above": 0, "below": 0, "overlay": 0, "other": 0}

        for char in text:
            if unicodedata.combining(char) > 0:
                cp = ord(char)
                if cp in self.COMBINING_ABOVE:
                    counts["above"] += 1
                elif cp in self.COMBINING_BELOW:
                    counts["below"] += 1
                elif cp in self.COMBINING_OVERLAY:
                    counts["overlay"] += 1
                else:
                    counts["other"] += 1

        return counts

    def has_diacritics(self, text: str) -> bool:
        """Check if text contains any diacritics."""
        return any(unicodedata.combining(char) > 0 for char in text)

    def strip_all_diacritics(self, text: str) -> str:
        """Remove all diacritical marks."""
        # Decompose
        text = unicodedata.normalize("NFD", text)

        # Filter out combining characters
        result = "".join(char for char in text if unicodedata.combining(char) == 0)

        # Recompose
        return unicodedata.normalize("NFC", result)


class CombiningCharacterHandler:
    """Handler for complex combining character sequences."""

    def __init__(self):
        """Initialize the handler."""
        pass

    def validate_combining_sequences(self, text: str) -> List[str]:
        """Validate combining character sequences.

        Returns:
            List of validation errors
        """
        errors = []
        normalizer = UnicodeNormalizer()
        sequences = normalizer.decompose_to_sequences(text)

        for seq in sequences:
            # Check for too many combiners
            if len(seq.combiners) > 4:
                errors.append(
                    f"Too many combining marks ({len(seq.combiners)}) "
                    f"at position {seq.start_pos}"
                )

            # Check for duplicate combiners
            if len(seq.combiners) != len(set(seq.combiners)):
                errors.append(f"Duplicate combining marks at position {seq.start_pos}")

            # Check combining class order
            classes = [unicodedata.combining(c) for c in seq.combiners]
            if classes != sorted(classes):
                errors.append(f"Incorrect combining mark order at position {seq.start_pos}")

        return errors

    def fix_combining_order(self, text: str) -> str:
        """Fix order of combining marks."""
        normalizer = UnicodeNormalizer()
        sequences = normalizer.decompose_to_sequences(text)

        # Fix order in each sequence
        fixed = []
        for seq in sequences:
            normalized = seq.normalize_order()
            fixed.append(normalized.to_string())

        return "".join(fixed)
