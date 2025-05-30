"""
ABBA manuscript variants support.

This module provides support for textual criticism by tracking manuscript
variants, critical apparatus data, and witness attestation.
"""

from .models import (
    Manuscript,
    ManuscriptFamily,
    ManuscriptType,
    TextualVariant,
    VariantType,
    VariantUnit,
    Witness,
    WitnessType,
    Attestation,
    CriticalApparatus,
    ApparatusEntry,
    VariantReading,
    ReadingSupport,
)
from .parser import CriticalApparatusParser, VariantNotation
from .analyzer import VariantAnalyzer, VariantImpact
from .scorer import ConfidenceScorer, ManuscriptWeight

__all__ = [
    # Models
    "Manuscript",
    "ManuscriptFamily",
    "ManuscriptType",
    "TextualVariant",
    "VariantType",
    "VariantUnit",
    "Witness",
    "WitnessType",
    "Attestation",
    "CriticalApparatus",
    "ApparatusEntry",
    "VariantReading",
    "ReadingSupport",
    # Parser
    "CriticalApparatusParser",
    "VariantNotation",
    # Analyzer
    "VariantAnalyzer",
    "VariantImpact",
    # Scorer
    "ConfidenceScorer",
    "ManuscriptWeight",
]
