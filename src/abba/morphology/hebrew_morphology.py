"""
Hebrew morphological parsing system.

This module parses Hebrew morphology codes from various tagging systems
and converts them to standardized grammatical categories.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .base import (
    Case,
    Gender,
    Mood,
    MorphologyFeatures,
    Number,
    Person,
    State,
    Stem,
    Tense,
)


@dataclass
class HebrewMorphology(MorphologyFeatures):
    """Hebrew-specific morphological features."""

    # Additional Hebrew features
    has_article: bool = False
    has_suffix: bool = False
    suffix_person: Optional[Person] = None
    suffix_gender: Optional[Gender] = None
    suffix_number: Optional[Number] = None

    # Prefix information
    has_preposition: bool = False
    preposition: Optional[str] = None
    has_conjunction: bool = False
    conjunction: Optional[str] = None

    def get_hebrew_summary(self) -> str:
        """Get Hebrew-specific summary."""
        parts = []

        # Add prefixes
        if self.has_conjunction:
            parts.append(f"conj({self.conjunction or 'ו'})")
        if self.has_preposition:
            parts.append(f"prep({self.preposition or ''})")
        if self.has_article:
            parts.append("art")

        # Add base summary
        parts.append(self.get_summary())

        # Add suffix info
        if self.has_suffix:
            suffix_parts = []
            if self.suffix_person:
                suffix_parts.append(self.suffix_person.value)
            if self.suffix_gender:
                suffix_parts.append(self.suffix_gender.value)
            if self.suffix_number:
                suffix_parts.append(self.suffix_number.value)
            if suffix_parts:
                parts.append(f"suff({' '.join(suffix_parts)})")

        return " ".join(parts)


class HebrewMorphologyParser:
    """Parser for Hebrew morphology codes."""

    # OSHB morphology code mappings
    PART_OF_SPEECH_MAP = {
        "A": "adjective",
        "C": "conjunction",
        "D": "adverb",
        "N": "noun",
        "P": "pronoun",
        "R": "preposition",
        "S": "suffix",
        "T": "particle",
        "V": "verb",
    }

    STEM_MAP = {
        "q": Stem.QAL,
        "Q": Stem.QAL,
        "N": Stem.NIPHAL,
        "p": Stem.PIEL,
        "P": Stem.PUAL,
        "h": Stem.HIPHIL,
        "H": Stem.HOPHAL,
        "t": Stem.HITHPAEL,
    }

    TENSE_MAP = {
        "p": Tense.PERFECT,
        "P": Tense.PERFECT,  # Also handle uppercase
        "q": Tense.PERFECT,  # Qatal
        "i": Tense.IMPERFECT,
        "I": Tense.IMPERFECT,  # Also handle uppercase
        "w": Tense.WAYYIQTOL,
        "c": Tense.IMPERFECT,  # Cohortative
        "j": Tense.IMPERFECT,  # Jussive
        "a": Tense.PERFECT,  # Active participle
        "s": Tense.PERFECT,  # Passive participle
    }

    MOOD_MAP = {
        "v": Mood.IMPERATIVE,
        "r": Mood.INFINITIVE,  # Infinitive construct
        "o": Mood.INFINITIVE,  # Infinitive absolute
        "c": Mood.COHORTATIVE,
        "j": Mood.JUSSIVE,
    }

    PERSON_MAP = {
        "1": Person.FIRST,
        "2": Person.SECOND,
        "3": Person.THIRD,
    }

    GENDER_MAP = {
        "m": Gender.MASCULINE,
        "M": Gender.MASCULINE,
        "f": Gender.FEMININE,
        "F": Gender.FEMININE,
        "b": Gender.COMMON,  # Both
        "c": Gender.COMMON,
    }

    NUMBER_MAP = {
        "s": Number.SINGULAR,
        "S": Number.SINGULAR,
        "d": Number.DUAL,
        "p": Number.PLURAL,
        "P": Number.PLURAL,
    }

    STATE_MAP = {
        "a": State.ABSOLUTE,
        "c": State.CONSTRUCT,
        "d": State.DETERMINED,
    }

    def __init__(self) -> None:
        """Initialize the Hebrew morphology parser."""
        self._morph_cache: Dict[str, HebrewMorphology] = {}

    def parse(self, morph_code: str) -> HebrewMorphology:
        """
        Parse Hebrew morphology code.

        Args:
            morph_code: Morphology code (e.g., "HNcmsa" for noun, common, masc, sing, abs)

        Returns:
            HebrewMorphology object with parsed features
        """
        if not morph_code:
            return HebrewMorphology()

        # Check cache
        if morph_code in self._morph_cache:
            return self._morph_cache[morph_code]

        morph = HebrewMorphology()

        # Handle prefixes (prepositions, conjunctions, article)
        code, prefixes = self._extract_prefixes(morph_code)
        if prefixes:
            self._parse_prefixes(prefixes, morph)

        # Parse main morphology code
        if code:
            self._parse_main_code(code, morph)

        # Handle suffixes
        if "/" in morph_code:
            suffix_code = morph_code.split("/")[-1]
            self._parse_suffix(suffix_code, morph)

        # Cache the result
        self._morph_cache[morph_code] = morph

        return morph

    def _extract_prefixes(self, morph_code: str) -> Tuple[str, str]:
        """Extract prefix codes from morphology string."""
        # Common Hebrew prefixes: H (article), C (conjunction), R (preposition)
        prefix_pattern = r"^([HCR/]+)(.+)$"
        match = re.match(prefix_pattern, morph_code)

        if match:
            return match.group(2), match.group(1)
        return morph_code, ""

    def _parse_prefixes(self, prefixes: str, morph: HebrewMorphology) -> None:
        """Parse Hebrew prefixes."""
        if "H" in prefixes:
            morph.has_article = True

        if "C" in prefixes:
            morph.has_conjunction = True
            morph.conjunction = "ו"  # Default vav

        if "R" in prefixes:
            morph.has_preposition = True
            # Could extract specific preposition if encoded

    def _parse_main_code(self, code: str, morph: HebrewMorphology) -> None:
        """Parse main morphology code."""
        if not code:
            return

        # First character is usually part of speech
        if code[0] in self.PART_OF_SPEECH_MAP:
            morph.part_of_speech = self.PART_OF_SPEECH_MAP[code[0]]
            code = code[1:]

        # For verbs, parse stem and conjugation
        if morph.part_of_speech == "verb" and code:
            # First character after V is stem
            if code[0] in self.STEM_MAP:
                morph.stem = self.STEM_MAP[code[0]]
                code = code[1:]

            # Next character could be tense or mood
            if code:
                if code[0] in self.TENSE_MAP:
                    morph.tense = self.TENSE_MAP[code[0]]
                    code = code[1:]
                elif code[0] in self.MOOD_MAP:
                    morph.mood = self.MOOD_MAP[code[0]]
                    code = code[1:]

            # Then person, gender, number
            if len(code) >= 3:
                if code[0] in self.PERSON_MAP:
                    morph.person = self.PERSON_MAP[code[0]]
                if code[1] in self.GENDER_MAP:
                    morph.gender = self.GENDER_MAP[code[1]]
                if code[2] in self.NUMBER_MAP:
                    morph.number = self.NUMBER_MAP[code[2]]

        # For nouns and adjectives
        elif morph.part_of_speech in ["noun", "adjective"] and code:
            # Parse in position order: type, gender, number, state
            # Format: Ncmsa = Noun common masculine singular absolute
            # Position 0 = noun type (c = common)
            # Position 1 = gender (m = masculine)
            # Position 2 = number (s = singular)
            # Position 3 = state (a = absolute)
            if len(code) >= 1:
                # Skip noun type indicator (first character)
                code = code[1:] if code[0] in ["c", "p"] else code

            # Now parse gender, number, state in order
            if len(code) >= 1 and code[0] in self.GENDER_MAP:
                morph.gender = self.GENDER_MAP[code[0]]
            if len(code) >= 2 and code[1] in self.NUMBER_MAP:
                morph.number = self.NUMBER_MAP[code[1]]
            if len(code) >= 3 and code[2] in self.STATE_MAP:
                morph.state = self.STATE_MAP[code[2]]

        # For pronouns
        elif morph.part_of_speech == "pronoun" and len(code) >= 3:
            if code[0] in self.PERSON_MAP:
                morph.person = self.PERSON_MAP[code[0]]
            if code[1] in self.GENDER_MAP:
                morph.gender = self.GENDER_MAP[code[1]]
            if code[2] in self.NUMBER_MAP:
                morph.number = self.NUMBER_MAP[code[2]]

    def _parse_suffix(self, suffix_code: str, morph: HebrewMorphology) -> None:
        """Parse pronominal suffix."""
        morph.has_suffix = True

        # Suffix format is usually Sp3ms (suffix pronoun 3rd masc sing)
        if suffix_code.startswith("S"):
            suffix_code = suffix_code[1:]

        if suffix_code.startswith("p") and len(suffix_code) >= 4:
            # Parse person, gender, number
            if suffix_code[1] in self.PERSON_MAP:
                morph.suffix_person = self.PERSON_MAP[suffix_code[1]]
            if suffix_code[2] in self.GENDER_MAP:
                morph.suffix_gender = self.GENDER_MAP[suffix_code[2]]
            if suffix_code[3] in self.NUMBER_MAP:
                morph.suffix_number = self.NUMBER_MAP[suffix_code[3]]

    def parse_oshb_morph(self, morph_attr: str) -> HebrewMorphology:
        """
        Parse Open Scriptures Hebrew Bible morphology attribute.

        Args:
            morph_attr: OSHB morph attribute (e.g., "HNcmsa")

        Returns:
            HebrewMorphology object
        """
        return self.parse(morph_attr)

    def get_part_of_speech(self, morph_code: str) -> Optional[str]:
        """Get just the part of speech from morphology code."""
        if not morph_code:
            return None

        # Skip prefixes
        code = re.sub(r"^[HCR/]+", "", morph_code)

        if code and code[0] in self.PART_OF_SPEECH_MAP:
            return self.PART_OF_SPEECH_MAP[code[0]]

        return None
