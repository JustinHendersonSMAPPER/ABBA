"""
ABBA canon support for multi-tradition biblical texts.

This module provides support for different biblical canons including Protestant,
Catholic, Orthodox, and other traditions, with versification mapping and
translation management.
"""

from .models import (
    Canon,
    CanonBook,
    BookClassification,
    BookSection,
    VersificationScheme,
    VerseMapping,
    MappingType,
    Translation,
    TranslationPhilosophy,
    LicenseType,
    CanonTradition,
)
from .registry import CanonRegistry, canon_registry
from .versification import VersificationEngine
from .translation import TranslationRepository
from .comparison import CanonComparator
from .service import CanonService

__all__ = [
    # Models
    "Canon",
    "CanonBook",
    "BookClassification",
    "BookSection",
    "VersificationScheme",
    "VerseMapping",
    "MappingType",
    "Translation",
    "TranslationPhilosophy",
    "LicenseType",
    "CanonTradition",
    # Core components
    "CanonRegistry",
    "canon_registry",
    "VersificationEngine",
    "TranslationRepository",
    "CanonComparator",
    "CanonService",
]
