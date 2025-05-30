"""
Interlinear data generation for biblical texts.

This module generates verse-by-verse interlinear data with word-by-word
alignment, pronunciation guides, and morphological glosses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..morphology import Language, UnifiedMorphology
from ..parsers.greek_parser import GreekVerse
from ..parsers.hebrew_parser import HebrewVerse
from ..verse_id import VerseID
from .token_alignment import AlignedToken, TokenAlignment
from .token_extractor import (
    ExtractedToken,
    GreekTokenExtractor,
    HebrewTokenExtractor,
)


@dataclass
class InterlinearWord:
    """Represents a single word in interlinear format."""

    # Original language data
    original_text: str
    transliteration: str
    lemma: Optional[str] = None
    strong_number: Optional[str] = None

    # Morphological data
    morphology: Optional[UnifiedMorphology] = None
    morphology_gloss: Optional[str] = None  # Human-readable morphology

    # Translation data
    gloss: Optional[str] = None  # Basic word gloss
    translation_words: List[str] = field(default_factory=list)  # Aligned translation

    # Metadata
    position: int = 0
    language: Language = Language.HEBREW

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format for serialization."""
        result = {
            "original": self.original_text,
            "transliteration": self.transliteration,
            "position": self.position,
            "language": self.language.value,
        }

        if self.lemma:
            result["lemma"] = self.lemma
        if self.strong_number:
            result["strong_number"] = self.strong_number
        if self.gloss:
            result["gloss"] = self.gloss
        if self.translation_words:
            result["translation"] = " ".join(self.translation_words)
        if self.morphology:
            result["morphology"] = self.morphology.to_dict()
        if self.morphology_gloss:
            result["morphology_gloss"] = self.morphology_gloss

        return result

    def get_display_lines(self) -> Dict[str, str]:
        """Get display lines for interlinear presentation."""
        return {
            "original": self.original_text,
            "transliteration": self.transliteration,
            "morphology": self.morphology_gloss or "",
            "gloss": self.gloss or "",
            "translation": " ".join(self.translation_words) if self.translation_words else "",
        }


@dataclass
class InterlinearVerse:
    """Complete interlinear data for a verse."""

    verse_id: VerseID
    language: Language
    words: List[InterlinearWord]

    # Full verse text
    original_text: str = ""
    transliteration: str = ""
    translation_text: str = ""

    # Metadata
    alignment_score: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format."""
        return {
            "verse_id": str(self.verse_id),
            "language": self.language.value,
            "original_text": self.original_text,
            "transliteration": self.transliteration,
            "translation_text": self.translation_text,
            "words": [word.to_dict() for word in self.words],
            "alignment_score": self.alignment_score,
            "notes": self.notes,
        }

    def get_interlinear_display(self, line_format: str = "aligned") -> str:
        """
        Generate formatted interlinear display.

        Args:
            line_format: Display format - "aligned" or "sequential"

        Returns:
            Formatted interlinear text
        """
        if line_format == "aligned":
            return self._get_aligned_display()
        else:
            return self._get_sequential_display()

    def _get_aligned_display(self) -> str:
        """Generate aligned interlinear display."""
        if not self.words:
            return ""

        # Calculate column widths
        widths = {}
        for word in self.words:
            lines = word.get_display_lines()
            for key, value in lines.items():
                current_width = widths.get(key, 0)
                widths[key] = max(current_width, len(value))

        # Build display lines
        lines_dict = {
            "original": [],
            "transliteration": [],
            "morphology": [],
            "gloss": [],
            "translation": [],
        }

        for word in self.words:
            word_lines = word.get_display_lines()
            for key in lines_dict:
                value = word_lines.get(key, "")
                padded = value.ljust(widths.get(key, 0) + 2)
                lines_dict[key].append(padded)

        # Combine lines
        result = []
        result.append("".join(lines_dict["original"]))
        result.append("".join(lines_dict["transliteration"]))
        result.append("".join(lines_dict["morphology"]))
        result.append("".join(lines_dict["gloss"]))
        result.append("".join(lines_dict["translation"]))

        return "\n".join(result)

    def _get_sequential_display(self) -> str:
        """Generate sequential interlinear display."""
        result = []

        for word in self.words:
            result.append(f"Word {word.position + 1}:")
            result.append(f"  Original: {word.original_text}")
            result.append(f"  Transliteration: {word.transliteration}")

            if word.lemma:
                result.append(f"  Lemma: {word.lemma}")
            if word.strong_number:
                result.append(f"  Strong's: {word.strong_number}")
            if word.morphology_gloss:
                result.append(f"  Morphology: {word.morphology_gloss}")
            if word.gloss:
                result.append(f"  Gloss: {word.gloss}")
            if word.translation_words:
                result.append(f"  Translation: {' '.join(word.translation_words)}")

            result.append("")  # Empty line between words

        return "\n".join(result)


class InterlinearGenerator:
    """Generate interlinear data for biblical verses."""

    def __init__(self, lexicon_data: Optional[Dict[str, Dict]] = None) -> None:
        """
        Initialize the interlinear generator.

        Args:
            lexicon_data: Optional lexicon data for glosses
        """
        self.hebrew_extractor = HebrewTokenExtractor()
        self.greek_extractor = GreekTokenExtractor()
        self.lexicon_data = lexicon_data or {}

    def generate_hebrew_interlinear(
        self,
        verse: HebrewVerse,
        translation_text: Optional[str] = None,
        alignment: Optional[TokenAlignment] = None,
    ) -> InterlinearVerse:
        """
        Generate interlinear data for a Hebrew verse.

        Args:
            verse: Parsed Hebrew verse
            translation_text: Optional translation text
            alignment: Optional pre-computed token alignment

        Returns:
            InterlinearVerse object
        """
        # Extract tokens
        tokens = self.hebrew_extractor.extract_tokens(verse)

        # Create interlinear words
        interlinear_words = []

        for token in tokens:
            word = self._create_interlinear_word(token, Language.HEBREW)

            # Add aligned translation if available
            if alignment:
                aligned_token = self._find_aligned_token(token, alignment.alignments)
                if aligned_token:
                    word.translation_words = aligned_token.target_words

            interlinear_words.append(word)

        # Create verse object
        result = InterlinearVerse(
            verse_id=verse.verse_id,
            language=Language.HEBREW,
            words=interlinear_words,
            original_text=" ".join(w.text for w in verse.words),
            transliteration=" ".join(w.transliteration for w in interlinear_words),
            translation_text=translation_text or "",
            alignment_score=alignment.alignment_score if alignment else 0.0,
        )

        return result

    def generate_greek_interlinear(
        self,
        verse: GreekVerse,
        translation_text: Optional[str] = None,
        alignment: Optional[TokenAlignment] = None,
    ) -> InterlinearVerse:
        """
        Generate interlinear data for a Greek verse.

        Args:
            verse: Parsed Greek verse
            translation_text: Optional translation text
            alignment: Optional pre-computed token alignment

        Returns:
            InterlinearVerse object
        """
        # Extract tokens
        tokens = self.greek_extractor.extract_tokens(verse)

        # Create interlinear words
        interlinear_words = []

        for token in tokens:
            word = self._create_interlinear_word(token, Language.GREEK)

            # Add aligned translation if available
            if alignment:
                aligned_token = self._find_aligned_token(token, alignment.alignments)
                if aligned_token:
                    word.translation_words = aligned_token.target_words

            interlinear_words.append(word)

        # Create verse object
        result = InterlinearVerse(
            verse_id=verse.verse_id,
            language=Language.GREEK,
            words=interlinear_words,
            original_text=" ".join(w.text for w in verse.words),
            transliteration=" ".join(w.transliteration for w in interlinear_words),
            translation_text=translation_text or "",
            alignment_score=alignment.alignment_score if alignment else 0.0,
        )

        return result

    def _create_interlinear_word(
        self, token: ExtractedToken, language: Language
    ) -> InterlinearWord:
        """Create an InterlinearWord from an ExtractedToken."""
        # Get morphology gloss
        morph_gloss = None
        if token.morphology:
            morph_gloss = token.morphology.get_summary()

        # Get lexicon gloss if available
        lexicon_gloss = None
        if token.strong_number and token.strong_number in self.lexicon_data:
            lexicon_entry = self.lexicon_data[token.strong_number]
            lexicon_gloss = lexicon_entry.get("gloss", lexicon_entry.get("definition"))

        word = InterlinearWord(
            original_text=token.text,
            transliteration=token.transliteration or "",
            lemma=token.lemma,
            strong_number=token.strong_number,
            morphology=token.morphology,
            morphology_gloss=morph_gloss,
            gloss=lexicon_gloss or token.gloss,
            position=token.position,
            language=language,
        )

        return word

    def _find_aligned_token(
        self, token: ExtractedToken, alignments: List[AlignedToken]
    ) -> Optional[AlignedToken]:
        """Find the alignment for a specific token."""
        for alignment in alignments:
            for source_token in alignment.source_tokens:
                if source_token.position == token.position and source_token.text == token.text:
                    return alignment
        return None

    def generate_parallel_interlinear(
        self, verses: Dict[str, Union[HebrewVerse, GreekVerse]], translations: Dict[str, str]
    ) -> Dict[str, InterlinearVerse]:
        """
        Generate interlinear data for multiple parallel versions.

        Args:
            verses: Dictionary of version_code to parsed verse
            translations: Dictionary of version_code to translation text

        Returns:
            Dictionary of version_code to InterlinearVerse
        """
        results = {}

        for version_code, verse in verses.items():
            translation_text = translations.get(version_code, "")

            if isinstance(verse, HebrewVerse):
                interlinear = self.generate_hebrew_interlinear(verse, translation_text)
            elif isinstance(verse, GreekVerse):
                interlinear = self.generate_greek_interlinear(verse, translation_text)
            else:
                continue

            results[version_code] = interlinear

        return results

    def compare_interlinear_verses(
        self, verse1: InterlinearVerse, verse2: InterlinearVerse
    ) -> Dict[str, any]:
        """
        Compare two interlinear verses (e.g., Hebrew and Greek).

        Args:
            verse1: First interlinear verse
            verse2: Second interlinear verse

        Returns:
            Comparison results
        """
        comparison = {
            "verse_id": str(verse1.verse_id),
            "languages": [verse1.language.value, verse2.language.value],
            "word_count_diff": len(verse1.words) - len(verse2.words),
            "common_strongs": [],
            "unique_to_first": [],
            "unique_to_second": [],
            "morphology_comparison": [],
        }

        # Collect Strong's numbers
        strongs1 = {w.strong_number for w in verse1.words if w.strong_number}
        strongs2 = {w.strong_number for w in verse2.words if w.strong_number}

        comparison["common_strongs"] = list(strongs1 & strongs2)
        comparison["unique_to_first"] = list(strongs1 - strongs2)
        comparison["unique_to_second"] = list(strongs2 - strongs1)

        # Compare morphology for words with same position
        for i in range(min(len(verse1.words), len(verse2.words))):
            word1 = verse1.words[i]
            word2 = verse2.words[i]

            if word1.morphology and word2.morphology:
                morph_comp = {
                    "position": i,
                    "words": [word1.original_text, word2.original_text],
                    "pos_match": (
                        word1.morphology.features.part_of_speech
                        == word2.morphology.features.part_of_speech
                    ),
                    "morphology": [word1.morphology_gloss, word2.morphology_gloss],
                }
                comparison["morphology_comparison"].append(morph_comp)

        return comparison
