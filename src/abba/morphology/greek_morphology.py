"""
Greek morphological parsing system.

This module parses Greek morphology codes from various tagging systems
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
    Tense,
    Voice,
)


@dataclass
class GreekMorphology(MorphologyFeatures):
    """Greek-specific morphological features."""

    # Additional Greek features
    degree: Optional[str] = None  # Comparative, superlative for adjectives
    definite: Optional[bool] = None  # For articles

    def get_greek_summary(self) -> str:
        """Get Greek-specific summary."""
        parts = []

        # Add part of speech
        if self.part_of_speech:
            parts.append(self.part_of_speech)

        # For verbs
        if self.part_of_speech == "verb":
            if self.tense:
                parts.append(self.tense.value)
            if self.voice:
                parts.append(self.voice.value)
            if self.mood:
                parts.append(self.mood.value)
            if self.person:
                parts.append(self.person.value)
            if self.number:
                parts.append(self.number.value)

        # For nouns, adjectives, participles
        elif self.part_of_speech in ["noun", "adjective", "participle", "article"]:
            if self.case:
                parts.append(self.case.value)
            if self.gender:
                parts.append(self.gender.value)
            if self.number:
                parts.append(self.number.value)
            if self.degree:
                parts.append(self.degree)

        return " ".join(parts)


class GreekMorphologyParser:
    """Parser for Greek morphology codes."""

    # Common Greek morphology code mappings (Robinson/Pierpont style)
    PART_OF_SPEECH_MAP = {
        "N": "noun",
        "V": "verb",
        "A": "adjective",
        "D": "adverb",
        "C": "conjunction",
        "P": "preposition",
        "X": "particle",
        "I": "interjection",
        "R": "pronoun",
        "T": "article",
    }

    # Tense codes
    TENSE_MAP = {
        "P": Tense.PRESENT,
        "I": Tense.IMPERFECT_GREEK,
        "F": Tense.FUTURE,
        "A": Tense.AORIST,
        "X": Tense.PERFECT_GREEK,
        "Y": Tense.PLUPERFECT,
    }

    # Voice codes
    VOICE_MAP = {
        "A": Voice.ACTIVE,
        "M": Voice.MIDDLE,
        "P": Voice.PASSIVE,
        "E": Voice.MIDDLE_PASSIVE,  # Either middle or passive
        "D": Voice.MIDDLE_PASSIVE,  # Deponent
    }

    # Mood codes
    MOOD_MAP = {
        "I": Mood.INDICATIVE,
        "S": Mood.SUBJUNCTIVE,
        "O": Mood.OPTATIVE,
        "M": Mood.IMPERATIVE,
        "N": Mood.INFINITIVE,
        "P": Mood.PARTICIPLE,
    }

    # Person codes
    PERSON_MAP = {
        "1": Person.FIRST,
        "2": Person.SECOND,
        "3": Person.THIRD,
    }

    # Case codes
    CASE_MAP = {
        "N": Case.NOMINATIVE,
        "G": Case.GENITIVE,
        "D": Case.DATIVE,
        "A": Case.ACCUSATIVE,
        "V": Case.VOCATIVE,
    }

    # Number codes
    NUMBER_MAP = {
        "S": Number.SINGULAR,
        "P": Number.PLURAL,
    }

    # Gender codes
    GENDER_MAP = {
        "M": Gender.MASCULINE,
        "F": Gender.FEMININE,
        "N": Gender.NEUTER,
    }

    # Degree codes (for adjectives)
    DEGREE_MAP = {
        "C": "comparative",
        "S": "superlative",
    }

    def __init__(self) -> None:
        """Initialize the Greek morphology parser."""
        self._morph_cache: Dict[str, GreekMorphology] = {}

    def parse(self, morph_code: str) -> GreekMorphology:
        """
        Parse Greek morphology code.

        Args:
            morph_code: Morphology code (e.g., "V-PAI-3S" or "N-NSM")

        Returns:
            GreekMorphology object with parsed features
        """
        if not morph_code:
            return GreekMorphology()

        # Check cache
        if morph_code in self._morph_cache:
            return self._morph_cache[morph_code]

        morph = GreekMorphology()

        # Handle different morphology code formats
        if "-" in morph_code:
            # Format: V-PAI-3S (Part-TenseVoiceMood-PersonNumber)
            parts = morph_code.split("-")
            if parts:
                self._parse_dash_format(parts, morph)
        else:
            # Compact format: NPAI3S or similar
            self._parse_compact_format(morph_code, morph)

        # Cache the result
        self._morph_cache[morph_code] = morph

        return morph

    def _parse_dash_format(self, parts: list, morph: GreekMorphology) -> None:
        """Parse dash-separated morphology format."""
        if not parts:
            return

        # First part is part of speech
        if parts[0] in self.PART_OF_SPEECH_MAP:
            morph.part_of_speech = self.PART_OF_SPEECH_MAP[parts[0]]

        # For verbs
        if morph.part_of_speech == "verb" and len(parts) >= 2:
            # Second part is tense-voice-mood
            tvm = parts[1]
            if len(tvm) >= 3:
                if tvm[0] in self.TENSE_MAP:
                    morph.tense = self.TENSE_MAP[tvm[0]]
                if tvm[1] in self.VOICE_MAP:
                    morph.voice = self.VOICE_MAP[tvm[1]]
                if tvm[2] in self.MOOD_MAP:
                    morph.mood = self.MOOD_MAP[tvm[2]]

            # Third part depends on mood
            if len(parts) >= 3:
                if morph.mood == Mood.PARTICIPLE:
                    # For participles, third part is case-number-gender
                    cng = parts[2]
                    if len(cng) >= 3:
                        if cng[0] in self.CASE_MAP:
                            morph.case = self.CASE_MAP[cng[0]]
                        if cng[1] in self.NUMBER_MAP:
                            morph.number = self.NUMBER_MAP[cng[1]]
                        if cng[2] in self.GENDER_MAP:
                            morph.gender = self.GENDER_MAP[cng[2]]
                elif morph.mood not in [Mood.INFINITIVE]:
                    # For other verb forms, third part is person-number
                    pn = parts[2]
                    if len(pn) >= 2:
                        if pn[0] in self.PERSON_MAP:
                            morph.person = self.PERSON_MAP[pn[0]]
                        if pn[1] in self.NUMBER_MAP:
                            morph.number = self.NUMBER_MAP[pn[1]]

        # For nouns, adjectives, articles
        elif (
            morph.part_of_speech in ["noun", "adjective", "article", "pronoun"] and len(parts) >= 2
        ):
            # Second part is case-number-gender
            cng = parts[1]
            if len(cng) >= 3:
                if cng[0] in self.CASE_MAP:
                    morph.case = self.CASE_MAP[cng[0]]
                if cng[1] in self.NUMBER_MAP:
                    morph.number = self.NUMBER_MAP[cng[1]]
                if cng[2] in self.GENDER_MAP:
                    morph.gender = self.GENDER_MAP[cng[2]]

            # For adjectives, check for degree
            if morph.part_of_speech == "adjective" and len(parts) >= 3:
                if parts[2] in self.DEGREE_MAP:
                    morph.degree = self.DEGREE_MAP[parts[2]]

    def _parse_compact_format(self, code: str, morph: GreekMorphology) -> None:
        """Parse compact morphology format."""
        if not code:
            return

        pos = 0

        # Part of speech
        if pos < len(code) and code[pos] in self.PART_OF_SPEECH_MAP:
            morph.part_of_speech = self.PART_OF_SPEECH_MAP[code[pos]]
            pos += 1

        # Parse based on part of speech
        if morph.part_of_speech == "verb":
            # Tense
            if pos < len(code) and code[pos] in self.TENSE_MAP:
                morph.tense = self.TENSE_MAP[code[pos]]
                pos += 1

            # Voice
            if pos < len(code) and code[pos] in self.VOICE_MAP:
                morph.voice = self.VOICE_MAP[code[pos]]
                pos += 1

            # Mood
            if pos < len(code) and code[pos] in self.MOOD_MAP:
                morph.mood = self.MOOD_MAP[code[pos]]
                pos += 1

            # Person (not for infinitives/participles)
            if morph.mood not in [Mood.INFINITIVE, Mood.PARTICIPLE]:
                if pos < len(code) and code[pos] in self.PERSON_MAP:
                    morph.person = self.PERSON_MAP[code[pos]]
                    pos += 1

            # Number
            if pos < len(code) and code[pos] in self.NUMBER_MAP:
                morph.number = self.NUMBER_MAP[code[pos]]
                pos += 1

            # For participles, also parse case and gender
            if morph.mood == Mood.PARTICIPLE:
                if pos < len(code) and code[pos] in self.CASE_MAP:
                    morph.case = self.CASE_MAP[code[pos]]
                    pos += 1
                if pos < len(code) and code[pos] in self.GENDER_MAP:
                    morph.gender = self.GENDER_MAP[code[pos]]
                    pos += 1

        elif morph.part_of_speech in ["noun", "adjective", "article", "pronoun"]:
            # Case
            if pos < len(code) and code[pos] in self.CASE_MAP:
                morph.case = self.CASE_MAP[code[pos]]
                pos += 1

            # Number
            if pos < len(code) and code[pos] in self.NUMBER_MAP:
                morph.number = self.NUMBER_MAP[code[pos]]
                pos += 1

            # Gender
            if pos < len(code) and code[pos] in self.GENDER_MAP:
                morph.gender = self.GENDER_MAP[code[pos]]
                pos += 1

            # Degree (for adjectives)
            if morph.part_of_speech == "adjective" and pos < len(code):
                if code[pos] in self.DEGREE_MAP:
                    morph.degree = self.DEGREE_MAP[code[pos]]

    def parse_byzantine_morph(self, morph_attr: str) -> GreekMorphology:
        """
        Parse Byzantine/Robinson-Pierpont morphology attribute.

        Args:
            morph_attr: Byzantine morph attribute (e.g., "V-PAI-3S")

        Returns:
            GreekMorphology object
        """
        return self.parse(morph_attr)

    def get_part_of_speech(self, morph_code: str) -> Optional[str]:
        """Get just the part of speech from morphology code."""
        if not morph_code:
            return None

        # Handle dash format
        if "-" in morph_code:
            parts = morph_code.split("-")
            if parts and parts[0] in self.PART_OF_SPEECH_MAP:
                return self.PART_OF_SPEECH_MAP[parts[0]]
        # Handle compact format
        elif morph_code and morph_code[0] in self.PART_OF_SPEECH_MAP:
            return self.PART_OF_SPEECH_MAP[morph_code[0]]

        return None
