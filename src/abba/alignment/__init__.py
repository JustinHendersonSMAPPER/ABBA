"""
Verse alignment and mapping system.

This package provides sophisticated verse alignment capabilities for handling
different versification systems, canon traditions, and complex verse mappings.
"""

from .unified_reference import UnifiedReferenceSystem
from .verse_mapper import EnhancedVerseMapper, MappingConfidence, VerseMapping
from .bridge_tables import VersificationBridge, MappingData
from .canon_support import CanonManager, Canon
from .validation import AlignmentValidator

__all__ = [
    "UnifiedReferenceSystem",
    "EnhancedVerseMapper",
    "MappingConfidence",
    "VerseMapping",
    "VersificationBridge",
    "MappingData",
    "CanonManager",
    "Canon",
    "AlignmentValidator",
]
