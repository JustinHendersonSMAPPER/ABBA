"""
Unified morphology system for Hebrew and Greek.

This module provides a unified interface for working with morphological
data from both Hebrew and Greek, enabling consistent analysis across
both biblical languages.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from .base import MorphologyFeatures
from .greek_morphology import GreekMorphology, GreekMorphologyParser
from .hebrew_morphology import HebrewMorphology, HebrewMorphologyParser


class Language(Enum):
    """Biblical language."""

    HEBREW = "hebrew"
    GREEK = "greek"
    ARAMAIC = "aramaic"


@dataclass
class UnifiedMorphology:
    """Unified morphological representation for any biblical language."""

    language: Language
    features: MorphologyFeatures
    original_code: str

    def is_verb(self) -> bool:
        """Check if word is a verb."""
        return self.features.part_of_speech == "verb"

    def is_noun(self) -> bool:
        """Check if word is a noun."""
        return self.features.part_of_speech == "noun"

    def is_participle(self) -> bool:
        """Check if word is a participle."""
        return (
            self.features.part_of_speech == "verb"
            and self.features.mood
            and self.features.mood.value == "participle"
        )

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if isinstance(self.features, HebrewMorphology):
            return self.features.get_hebrew_summary()
        elif isinstance(self.features, GreekMorphology):
            return self.features.get_greek_summary()
        else:
            return self.features.get_summary()

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format."""
        return {
            "language": self.language.value,
            "original_code": self.original_code,
            "features": self.features.to_dict(),
            "summary": self.get_summary(),
        }


class UnifiedMorphologyParser:
    """Parser that handles both Hebrew and Greek morphology."""

    def __init__(self) -> None:
        """Initialize the unified parser."""
        self.hebrew_parser = HebrewMorphologyParser()
        self.greek_parser = GreekMorphologyParser()
        self._unified_cache: Dict[str, UnifiedMorphology] = {}

    def parse(self, morph_code: str, language: Language) -> UnifiedMorphology:
        """
        Parse morphology code for specified language.

        Args:
            morph_code: Morphology code to parse
            language: Language of the morphology code

        Returns:
            UnifiedMorphology object
        """
        # Check cache
        cache_key = f"{language.value}:{morph_code}"
        if cache_key in self._unified_cache:
            return self._unified_cache[cache_key]

        # Parse based on language
        if language == Language.HEBREW:
            features = self.hebrew_parser.parse(morph_code)
        elif language == Language.GREEK:
            features = self.greek_parser.parse(morph_code)
        elif language == Language.ARAMAIC:
            # Use Hebrew parser for Aramaic (similar morphology)
            features = self.hebrew_parser.parse(morph_code)
        else:
            features = MorphologyFeatures()

        # Create unified morphology
        unified = UnifiedMorphology(language=language, features=features, original_code=morph_code)

        # Cache result
        self._unified_cache[cache_key] = unified

        return unified

    def parse_auto_detect(self, morph_code: str) -> UnifiedMorphology:
        """
        Parse morphology code with automatic language detection.

        Args:
            morph_code: Morphology code to parse

        Returns:
            UnifiedMorphology object
        """
        # Try to detect language based on morphology code format
        language = self._detect_language(morph_code)
        return self.parse(morph_code, language)

    def _detect_language(self, morph_code: str) -> Language:
        """Detect language from morphology code format."""
        if not morph_code:
            return Language.HEBREW  # Default

        # Greek codes often have dashes or start with certain letters
        if "-" in morph_code:
            # Dash format is typically Greek (e.g., V-PAI-3S)
            return Language.GREEK

        # Check for Hebrew prefixes
        if morph_code.startswith(("H", "C", "R")) and len(morph_code) > 2:
            # Likely Hebrew with article/conjunction/preposition
            return Language.HEBREW

        # Check first character against known patterns
        first_char = morph_code[0]

        # Hebrew part of speech codes
        if first_char in ["A", "C", "D", "N", "P", "R", "S", "T", "V"]:
            # Could be either, need more context
            # Check for Greek-specific patterns
            if len(morph_code) >= 2:
                second_char = morph_code[1]
                # Greek tense codes after V
                if first_char == "V" and second_char in ["P", "I", "F", "A", "X", "Y"]:
                    return Language.GREEK
                # Hebrew stem codes after V
                elif first_char == "V" and second_char in ["q", "N", "p", "P", "h", "H", "t"]:
                    return Language.HEBREW

        # Default to Hebrew if uncertain
        return Language.HEBREW

    def get_part_of_speech(
        self, morph_code: str, language: Optional[Language] = None
    ) -> Optional[str]:
        """
        Get part of speech from morphology code.

        Args:
            morph_code: Morphology code
            language: Language (auto-detected if not provided)

        Returns:
            Part of speech string or None
        """
        if language is None:
            language = self._detect_language(morph_code)

        if language == Language.HEBREW:
            return self.hebrew_parser.get_part_of_speech(morph_code)
        elif language == Language.GREEK:
            return self.greek_parser.get_part_of_speech(morph_code)

        return None

    def compare_morphologies(
        self, morph1: UnifiedMorphology, morph2: UnifiedMorphology
    ) -> Dict[str, any]:
        """
        Compare two morphologies and identify differences.

        Args:
            morph1: First morphology
            morph2: Second morphology

        Returns:
            Dictionary with comparison results
        """
        differences = {}
        agreements = {}

        # Compare basic features
        features1 = morph1.features.to_dict()
        features2 = morph2.features.to_dict()

        all_keys = set(features1.keys()) | set(features2.keys())

        for key in all_keys:
            val1 = features1.get(key)
            val2 = features2.get(key)

            if val1 == val2 and val1 is not None:
                agreements[key] = val1
            elif val1 != val2:
                differences[key] = {"first": val1, "second": val2}

        return {
            "languages": [morph1.language.value, morph2.language.value],
            "agreements": agreements,
            "differences": differences,
            "agreement_score": len(agreements) / len(all_keys) if all_keys else 0,
        }

    def get_morphology_statistics(self, morphologies: List[UnifiedMorphology]) -> Dict[str, any]:
        """
        Calculate statistics for a collection of morphologies.

        Args:
            morphologies: List of morphology objects

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total": len(morphologies),
            "by_language": {},
            "by_part_of_speech": {},
            "verb_statistics": {
                "by_tense": {},
                "by_voice": {},
                "by_mood": {},
                "by_stem": {},  # Hebrew only
            },
            "noun_statistics": {
                "by_case": {},
                "by_gender": {},
                "by_number": {},
                "by_state": {},  # Hebrew only
            },
        }

        for morph in morphologies:
            # Language statistics
            lang = morph.language.value
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1

            # Part of speech statistics
            pos = morph.features.part_of_speech
            if pos:
                stats["by_part_of_speech"][pos] = stats["by_part_of_speech"].get(pos, 0) + 1

            # Verb statistics
            if morph.is_verb():
                if morph.features.tense:
                    tense = morph.features.tense.value
                    stats["verb_statistics"]["by_tense"][tense] = (
                        stats["verb_statistics"]["by_tense"].get(tense, 0) + 1
                    )

                if morph.features.voice:
                    voice = morph.features.voice.value
                    stats["verb_statistics"]["by_voice"][voice] = (
                        stats["verb_statistics"]["by_voice"].get(voice, 0) + 1
                    )

                if morph.features.mood:
                    mood = morph.features.mood.value
                    stats["verb_statistics"]["by_mood"][mood] = (
                        stats["verb_statistics"]["by_mood"].get(mood, 0) + 1
                    )

                if hasattr(morph.features, "stem") and morph.features.stem:
                    stem = morph.features.stem.value
                    stats["verb_statistics"]["by_stem"][stem] = (
                        stats["verb_statistics"]["by_stem"].get(stem, 0) + 1
                    )

            # Noun statistics
            elif morph.is_noun() or morph.features.part_of_speech in ["adjective", "pronoun"]:
                if morph.features.case:
                    case = morph.features.case.value
                    stats["noun_statistics"]["by_case"][case] = (
                        stats["noun_statistics"]["by_case"].get(case, 0) + 1
                    )

                if morph.features.gender:
                    gender = morph.features.gender.value
                    stats["noun_statistics"]["by_gender"][gender] = (
                        stats["noun_statistics"]["by_gender"].get(gender, 0) + 1
                    )

                if morph.features.number:
                    number = morph.features.number.value
                    stats["noun_statistics"]["by_number"][number] = (
                        stats["noun_statistics"]["by_number"].get(number, 0) + 1
                    )

                if hasattr(morph.features, "state") and morph.features.state:
                    state = morph.features.state.value
                    stats["noun_statistics"]["by_state"][state] = (
                        stats["noun_statistics"]["by_state"].get(state, 0) + 1
                    )

        return stats
