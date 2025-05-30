"""
ABBA - Annotated Bible and Background Analysis

A comprehensive data format and processing library for biblical texts
with support for multiple versions, languages, and scholarly annotations.
"""

__version__ = "0.1.0"

# Core components
from .book_codes import (
    BOOK_INFO,
    BookCode,
    Canon,
    Testament,
    get_book_info,
    get_book_name,
    get_book_order,
    get_books_by_canon,
    get_books_by_testament,
    get_chapter_count,
    is_valid_book_code,
    normalize_book_name,
)
from .parsers import (
    GreekParser,
    GreekVerse,
    GreekWord,
    HebrewParser,
    HebrewVerse,
    HebrewWord,
    LexiconEntry,
    LexiconParser,
    Translation,
    TranslationParser,
    TranslationVerse,
)
from .verse_id import (
    VerseID,
    VerseRange,
    compare_verse_ids,
    create_verse_id,
    get_verse_parts,
    is_valid_verse_id,
    normalize_verse_id,
    parse_verse_id,
    parse_verse_range,
)
from .versification import (
    VersificationDifference,
    VersificationMapper,
    VersificationRules,
    VersificationSystem,
    get_versification_documentation,
)

__all__ = [
    # Version
    "__version__",
    # Book codes
    "BookCode",
    "Testament",
    "Canon",
    "BOOK_INFO",
    "normalize_book_name",
    "get_book_info",
    "get_book_name",
    "get_books_by_testament",
    "get_books_by_canon",
    "get_book_order",
    "is_valid_book_code",
    "get_chapter_count",
    # Verse IDs
    "VerseID",
    "VerseRange",
    "parse_verse_id",
    "create_verse_id",
    "parse_verse_range",
    "normalize_verse_id",
    "is_valid_verse_id",
    "compare_verse_ids",
    "get_verse_parts",
    # Versification
    "VersificationSystem",
    "VersificationMapper",
    "VersificationDifference",
    "VersificationRules",
    "get_versification_documentation",
    # Parsers
    "HebrewParser",
    "HebrewVerse",
    "HebrewWord",
    "GreekParser",
    "GreekVerse",
    "GreekWord",
    "TranslationParser",
    "Translation",
    "TranslationVerse",
    "LexiconParser",
    "LexiconEntry",
]
