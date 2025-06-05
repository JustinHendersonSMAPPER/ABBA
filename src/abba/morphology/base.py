"""
Base classes and enumerations for morphological analysis.

This module defines the common grammatical categories and features
used across Hebrew and Greek morphological parsing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any


class Language(Enum):
    """Biblical language."""
    
    HEBREW = "hebrew"
    GREEK = "greek"
    ARAMAIC = "aramaic"


class Gender(Enum):
    """Grammatical gender."""

    MASCULINE = "masculine"
    FEMININE = "feminine"
    NEUTER = "neuter"  # Greek only
    COMMON = "common"  # Both masculine and feminine


class Number(Enum):
    """Grammatical number."""

    SINGULAR = "singular"
    DUAL = "dual"  # Hebrew only
    PLURAL = "plural"


class Person(Enum):
    """Grammatical person."""

    FIRST = "1st"
    SECOND = "2nd"
    THIRD = "3rd"


class Tense(Enum):
    """Verb tense."""

    # Hebrew tenses (aspect-based)
    PERFECT = "perfect"  # Completed action
    IMPERFECT = "imperfect"  # Incomplete action
    WAYYIQTOL = "wayyiqtol"  # Sequential imperfect

    # Greek tenses
    PRESENT = "present"
    AORIST = "aorist"
    FUTURE = "future"
    IMPERFECT_GREEK = "imperfect"  # Greek imperfect (different from Hebrew)
    PLUPERFECT = "pluperfect"
    PERFECT_GREEK = "perfect"  # Greek perfect


class Voice(Enum):
    """Verb voice."""

    ACTIVE = "active"
    PASSIVE = "passive"
    MIDDLE = "middle"  # Greek
    MIDDLE_PASSIVE = "middle/passive"  # Greek deponent


class Mood(Enum):
    """Verb mood."""

    INDICATIVE = "indicative"
    SUBJUNCTIVE = "subjunctive"
    OPTATIVE = "optative"  # Greek
    IMPERATIVE = "imperative"
    INFINITIVE = "infinitive"
    PARTICIPLE = "participle"

    # Hebrew specific
    JUSSIVE = "jussive"
    COHORTATIVE = "cohortative"


class Case(Enum):
    """Noun case (primarily Greek)."""

    NOMINATIVE = "nominative"
    GENITIVE = "genitive"
    DATIVE = "dative"
    ACCUSATIVE = "accusative"
    VOCATIVE = "vocative"

    # Hebrew construct state
    CONSTRUCT = "construct"
    ABSOLUTE = "absolute"


class State(Enum):
    """Hebrew noun state."""

    ABSOLUTE = "absolute"
    CONSTRUCT = "construct"
    DETERMINED = "determined"  # With article


class Stem(Enum):
    """Hebrew verb stem (binyan)."""

    QAL = "qal"
    NIPHAL = "niphal"
    PIEL = "piel"
    PUAL = "pual"
    HIPHIL = "hiphil"
    HOPHAL = "hophal"
    HITHPAEL = "hithpael"

    # Aramaic stems
    PEAL = "peal"
    PAEL = "pael"
    APHEL = "aphel"
    ITHPEEL = "ithpeel"


@dataclass
class MorphologyFeatures:
    """Base class for morphological features."""

    # Common features
    part_of_speech: Optional[str] = None
    gender: Optional[Gender] = None
    number: Optional[Number] = None
    person: Optional[Person] = None

    # Verb features
    tense: Optional[Tense] = None
    voice: Optional[Voice] = None
    mood: Optional[Mood] = None

    # Noun features
    case: Optional[Case] = None
    state: Optional[State] = None

    # Hebrew specific
    stem: Optional[Stem] = None

    # Additional features
    suffix: Optional[str] = None  # Pronominal suffixes
    prefix: Optional[str] = None  # Prepositions, conjunctions

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary format."""
        result = {}

        for field, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, Enum):
                    result[field] = value.value
                else:
                    result[field] = value

        return result

    def get_summary(self) -> str:
        """Get human-readable summary of features."""
        parts = []

        if self.part_of_speech:
            parts.append(self.part_of_speech)

        if self.gender:
            parts.append(self.gender.value)

        if self.number:
            parts.append(self.number.value)

        if self.person and self.part_of_speech in ["verb", "pronoun"]:
            parts.append(self.person.value)

        if self.tense:
            parts.append(self.tense.value)

        if self.voice and self.voice != Voice.ACTIVE:
            parts.append(self.voice.value)

        if self.mood:
            parts.append(self.mood.value)

        if self.case:
            parts.append(self.case.value)

        if self.stem:
            parts.append(self.stem.value)

        return " ".join(parts)
