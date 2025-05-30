"""
Interlinear text processing and alignment system.

This package provides token-level alignment between original languages
and translations, including morphological analysis and lexicon integration.
"""

from .token_alignment import (
    AlignedToken,
    TokenAlignment,
    TokenAligner,
    AlignmentType,
)
from .token_extractor import (
    TokenExtractor,
    HebrewTokenExtractor,
    GreekTokenExtractor,
    ExtractedToken,
)
from .interlinear_generator import (
    InterlinearGenerator,
    InterlinearVerse,
    InterlinearWord,
)
from .lexicon_integration import (
    LexiconIntegrator,
    LexicalEntry,
    SemanticDomain,
)

__all__ = [
    # Token alignment
    "AlignedToken",
    "TokenAlignment",
    "TokenAligner",
    "AlignmentType",
    # Token extraction
    "TokenExtractor",
    "HebrewTokenExtractor",
    "GreekTokenExtractor",
    "ExtractedToken",
    # Interlinear generation
    "InterlinearGenerator",
    "InterlinearVerse",
    "InterlinearWord",
    # Lexicon integration
    "LexiconIntegrator",
    "LexicalEntry",
    "SemanticDomain",
]
