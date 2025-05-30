"""
Cross-reference system for biblical texts.

This package provides comprehensive cross-reference functionality including:
- Reference type classification (quotations, allusions, parallels)
- Citation tracking (OT quotes in NT)
- Bidirectional reference indices
- Confidence scoring algorithms
"""

from .models import (
    CrossReference,
    ReferenceType,
    ReferenceRelationship,
    CitationMatch,
    ReferenceConfidence,
    ReferenceCollection,
)
from .parser import CrossReferenceParser
from .classifier import ReferenceTypeClassifier
from .citation_tracker import CitationTracker
from .confidence_scorer import ConfidenceScorer

__all__ = [
    # Core models
    "CrossReference",
    "ReferenceType",
    "ReferenceRelationship",
    "CitationMatch",
    "ReferenceConfidence",
    "ReferenceCollection",
    # Processing components
    "CrossReferenceParser",
    "ReferenceTypeClassifier",
    "CitationTracker",
    "ConfidenceScorer",
]
