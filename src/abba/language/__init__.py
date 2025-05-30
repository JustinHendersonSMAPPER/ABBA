"""
ABBA extended language support.

This module provides comprehensive language support for biblical texts including
RTL rendering, Unicode normalization, transliteration, and special character
handling.
"""

from .rtl import RTLHandler, TextDirection, BidiAlgorithm
from .unicode_utils import (
    UnicodeNormalizer,
    NormalizationForm,
    HebrewNormalizer,
    GreekNormalizer,
    DiacriticHandler,
    CombiningCharacterHandler,
)
from .transliteration import (
    TransliterationEngine,
    TransliterationScheme,
    HebrewTransliterator,
    GreekTransliterator,
    ArabicTransliterator,
    SyriacTransliterator,
)
from .script_detector import ScriptDetector, Script, ScriptRange
from .font_support import FontManager, FontRequirements, FontFallback

__all__ = [
    # RTL Support
    "RTLHandler",
    "TextDirection",
    "BidiAlgorithm",
    # Unicode
    "UnicodeNormalizer",
    "NormalizationForm",
    "HebrewNormalizer",
    "GreekNormalizer",
    "DiacriticHandler",
    "CombiningCharacterHandler",
    # Transliteration
    "TransliterationEngine",
    "TransliterationScheme",
    "HebrewTransliterator",
    "GreekTransliterator",
    "ArabicTransliterator",
    "SyriacTransliterator",
    # Script Detection
    "ScriptDetector",
    "Script",
    "ScriptRange",
    # Font Support
    "FontManager",
    "FontRequirements",
    "FontFallback",
]
