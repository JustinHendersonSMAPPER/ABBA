"""
Token extraction from Hebrew and Greek texts.

This module provides functionality to extract individual tokens (words)
from parsed Hebrew and Greek XML data, preserving morphological and
lexical information.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from ..morphology import (
    Language,
    UnifiedMorphology,
    UnifiedMorphologyParser,
)
from ..parsers.greek_parser import GreekVerse, GreekWord
from ..parsers.hebrew_parser import HebrewVerse, HebrewWord
from ..verse_id import VerseID


@dataclass
class ExtractedToken:
    """Represents an extracted token with all linguistic data."""

    text: str  # The actual word text
    lemma: Optional[str] = None
    strong_number: Optional[str] = None
    morphology: Optional[UnifiedMorphology] = None
    position: int = 0  # Position in verse (0-based)
    language: Language = Language.HEBREW
    gloss: Optional[str] = None  # Basic English gloss
    transliteration: Optional[str] = None

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format."""
        result = {
            "text": self.text,
            "position": self.position,
            "language": self.language.value,
        }

        if self.lemma:
            result["lemma"] = self.lemma
        if self.strong_number:
            result["strong_number"] = self.strong_number
        if self.morphology:
            result["morphology"] = self.morphology.to_dict()
        if self.gloss:
            result["gloss"] = self.gloss
        if self.transliteration:
            result["transliteration"] = self.transliteration

        return result


class TokenExtractor(Protocol):
    """Protocol for token extractors."""

    def extract_tokens(self, verse: any) -> List[ExtractedToken]:
        """Extract tokens from a verse."""
        ...

    def extract_tokens_from_text(self, text: str) -> List[ExtractedToken]:
        """Extract tokens from plain text."""
        ...


class HebrewTokenExtractor:
    """Extract tokens from Hebrew verse data."""

    def __init__(self) -> None:
        """Initialize the Hebrew token extractor."""
        self.morph_parser = UnifiedMorphologyParser()
        self._transliteration_map = self._build_transliteration_map()

    def extract_tokens(self, verse: HebrewVerse) -> List[ExtractedToken]:
        """
        Extract tokens from a Hebrew verse.

        Args:
            verse: HebrewVerse object with parsed data

        Returns:
            List of ExtractedToken objects
        """
        tokens = []

        for position, word in enumerate(verse.words):
            # Parse morphology if available
            morphology = None
            if word.morph:
                morphology = self.morph_parser.parse(word.morph, Language.HEBREW)

            # Generate transliteration
            transliteration = self._transliterate(word.text)

            token = ExtractedToken(
                text=word.text,
                lemma=word.lemma,
                strong_number=word.strong_number,
                morphology=morphology,
                position=position,
                language=Language.HEBREW,
                gloss=word.gloss,
                transliteration=transliteration,
            )

            tokens.append(token)

        return tokens

    def extract_tokens_from_text(self, text: str) -> List[ExtractedToken]:
        """
        Extract tokens from Hebrew plain text.

        Args:
            text: Hebrew text string

        Returns:
            List of ExtractedToken objects
        """
        # Split on whitespace and common punctuation
        import re

        words = re.findall(r"[\u0590-\u05FF]+", text)

        tokens = []
        for position, word in enumerate(words):
            token = ExtractedToken(
                text=word,
                position=position,
                language=Language.HEBREW,
                transliteration=self._transliterate(word),
            )
            tokens.append(token)

        return tokens

    def _build_transliteration_map(self) -> Dict[str, str]:
        """Build Hebrew to English transliteration mapping."""
        # Simplified transliteration map
        return {
            "א": "'",
            "ב": "b",
            "ג": "g",
            "ד": "d",
            "ה": "h",
            "ו": "v",
            "ז": "z",
            "ח": "ch",
            "ט": "t",
            "י": "y",
            "כ": "k",
            "ך": "k",
            "ל": "l",
            "מ": "m",
            "ם": "m",
            "נ": "n",
            "ן": "n",
            "ס": "s",
            "ע": "'",
            "פ": "p",
            "ף": "p",
            "צ": "ts",
            "ץ": "ts",
            "ק": "q",
            "ר": "r",
            "ש": "sh",
            "ת": "t",
            # Vowel points (simplified)
            "\u05B0": "e",
            "\u05B1": "e",
            "\u05B2": "a",
            "\u05B3": "o",
            "\u05B4": "i",
            "\u05B5": "e",
            "\u05B6": "e",
            "\u05B7": "a",
            "\u05B8": "a",
            "\u05B9": "o",
            "\u05BA": "o",
            "\u05BB": "u",
        }

    def _transliterate(self, hebrew_text: str) -> str:
        """Generate simple transliteration of Hebrew text."""
        if not hebrew_text:
            return ""

        result = []
        for char in hebrew_text:
            result.append(self._transliteration_map.get(char, char))

        return "".join(result)


class GreekTokenExtractor:
    """Extract tokens from Greek verse data."""

    def __init__(self) -> None:
        """Initialize the Greek token extractor."""
        self.morph_parser = UnifiedMorphologyParser()
        self._transliteration_map = self._build_transliteration_map()

    def extract_tokens(self, verse: GreekVerse) -> List[ExtractedToken]:
        """
        Extract tokens from a Greek verse.

        Args:
            verse: GreekVerse object with parsed data

        Returns:
            List of ExtractedToken objects
        """
        tokens = []

        for position, word in enumerate(verse.words):
            # Parse morphology if available
            morphology = None
            if word.morph:
                morphology = self.morph_parser.parse(word.morph, Language.GREEK)

            # Generate transliteration
            transliteration = self._transliterate(word.text)

            # Extract Strong's number from lemma if needed
            strong_number = word.strong_number
            if not strong_number and word.lemma:
                # Try to extract from lemma format
                import re

                match = re.search(r"G(\d+)", word.lemma)
                if match:
                    strong_number = f"G{match.group(1)}"

            token = ExtractedToken(
                text=word.text,
                lemma=word.lemma,
                strong_number=strong_number,
                morphology=morphology,
                position=position,
                language=Language.GREEK,
                transliteration=transliteration,
            )

            tokens.append(token)

        return tokens

    def extract_tokens_from_text(self, text: str) -> List[ExtractedToken]:
        """
        Extract tokens from Greek plain text.

        Args:
            text: Greek text string

        Returns:
            List of ExtractedToken objects
        """
        # Split on whitespace and punctuation
        import re

        words = re.findall(r"[\u0370-\u03FF\u1F00-\u1FFF]+", text)

        tokens = []
        for position, word in enumerate(words):
            token = ExtractedToken(
                text=word,
                position=position,
                language=Language.GREEK,
                transliteration=self._transliterate(word),
            )
            tokens.append(token)

        return tokens

    def _build_transliteration_map(self) -> Dict[str, str]:
        """Build Greek to English transliteration mapping."""
        # Simplified transliteration map
        return {
            # Lowercase
            "α": "a",
            "β": "b",
            "γ": "g",
            "δ": "d",
            "ε": "e",
            "ζ": "z",
            "η": "ē",
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
            "ς": "s",
            "τ": "t",
            "υ": "u",
            "φ": "ph",
            "χ": "ch",
            "ψ": "ps",
            "ω": "ō",
            # Uppercase
            "Α": "A",
            "Β": "B",
            "Γ": "G",
            "Δ": "D",
            "Ε": "E",
            "Ζ": "Z",
            "Η": "Ē",
            "Θ": "Th",
            "Ι": "I",
            "Κ": "K",
            "Λ": "L",
            "Μ": "M",
            "Ν": "N",
            "Ξ": "X",
            "Ο": "O",
            "Π": "P",
            "Ρ": "R",
            "Σ": "S",
            "Τ": "T",
            "Υ": "U",
            "Φ": "Ph",
            "Χ": "Ch",
            "Ψ": "Ps",
            "Ω": "Ō",
            # Common diacritics (simplified)
            "ά": "a",
            "έ": "e",
            "ή": "ē",
            "ί": "i",
            "ό": "o",
            "ύ": "u",
            "ώ": "ō",
            "ΐ": "i",
            "ΰ": "u",
        }

    def _transliterate(self, greek_text: str) -> str:
        """Generate simple transliteration of Greek text."""
        if not greek_text:
            return ""

        # Remove diacritics for simplicity
        import unicodedata

        normalized = unicodedata.normalize("NFD", greek_text)

        result = []
        skip_next = False

        for i, char in enumerate(normalized):
            if skip_next:
                skip_next = False
                continue

            # Handle special combinations
            if i < len(normalized) - 1:
                two_char = normalized[i : i + 2]
                if two_char in ["ου", "ΟΥ"]:
                    result.append("ou")
                    skip_next = True
                    continue
                elif two_char in ["ει", "ΕΙ"]:
                    result.append("ei")
                    skip_next = True
                    continue
                elif two_char in ["αι", "ΑΙ"]:
                    result.append("ai")
                    skip_next = True
                    continue

            # Single character mapping
            if char in self._transliteration_map:
                result.append(self._transliteration_map[char])
            elif not unicodedata.combining(char):
                result.append(char)

        return "".join(result)
