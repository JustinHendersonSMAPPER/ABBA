"""
Morphological analysis system for biblical languages.

This package provides comprehensive morphological parsing and analysis
for Hebrew and Greek biblical texts.
"""

from .base import (
    Language,
    Gender,
    MorphologyFeatures,
    Number,
    Person,
    Tense,
    Voice,
    Mood,
    Case,
    State,
    Stem,
)
from .hebrew_morphology import HebrewMorphologyParser, HebrewMorphology
from .greek_morphology import GreekMorphologyParser, GreekMorphology
from .unified_morphology import UnifiedMorphologyParser, UnifiedMorphology

__all__ = [
    # Base classes and enums
    "Language",
    "Gender",
    "MorphologyFeatures",
    "Number",
    "Person",
    "Tense",
    "Voice",
    "Mood",
    "Case",
    "State",
    "Stem",
    # Hebrew morphology
    "HebrewMorphologyParser",
    "HebrewMorphology",
    # Greek morphology
    "GreekMorphologyParser",
    "GreekMorphology",
    # Unified morphology
    "UnifiedMorphologyParser",
    "UnifiedMorphology",
]
