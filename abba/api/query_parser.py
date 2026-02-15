"""Search query parser for structured and natural language queries.

Supports query syntax:
  - Simple text: ``love`` → full-text search
  - Book filter: ``love in:john`` or ``love book:john``
  - Testament filter: ``grace testament:new`` or ``grace ot``
  - Strong's lookup: ``strongs:H0430``
  - Exact phrase: ``"exact phrase"``
  - Language filter: ``language:hebrew``
  - Combined: ``"living water" in:john testament:new``
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Book name to ID mapping (abbreviated + full names)
BOOK_NAME_TO_ID: Dict[str, int] = {
    "gen": 1,
    "genesis": 1,
    "exo": 2,
    "exodus": 2,
    "lev": 3,
    "leviticus": 3,
    "num": 4,
    "numbers": 4,
    "deu": 5,
    "deuteronomy": 5,
    "jos": 6,
    "joshua": 6,
    "jdg": 7,
    "judges": 7,
    "rut": 8,
    "ruth": 8,
    "1sa": 9,
    "1samuel": 9,
    "2sa": 10,
    "2samuel": 10,
    "1ki": 11,
    "1kings": 11,
    "2ki": 12,
    "2kings": 12,
    "1ch": 13,
    "1chronicles": 13,
    "2ch": 14,
    "2chronicles": 14,
    "ezr": 15,
    "ezra": 15,
    "neh": 16,
    "nehemiah": 16,
    "est": 17,
    "esther": 17,
    "job": 18,
    "psa": 19,
    "psalms": 19,
    "psalm": 19,
    "pro": 20,
    "proverbs": 20,
    "ecc": 21,
    "ecclesiastes": 21,
    "sng": 22,
    "songofsolomon": 22,
    "song": 22,
    "isa": 23,
    "isaiah": 23,
    "jer": 24,
    "jeremiah": 24,
    "lam": 25,
    "lamentations": 25,
    "ezk": 26,
    "ezekiel": 26,
    "dan": 27,
    "daniel": 27,
    "hos": 28,
    "hosea": 28,
    "jol": 29,
    "joel": 29,
    "amo": 30,
    "amos": 30,
    "oba": 31,
    "obadiah": 31,
    "jon": 32,
    "jonah": 32,
    "mic": 33,
    "micah": 33,
    "nam": 34,
    "nahum": 34,
    "hab": 35,
    "habakkuk": 35,
    "zep": 36,
    "zephaniah": 36,
    "hag": 37,
    "haggai": 37,
    "zec": 38,
    "zechariah": 38,
    "mal": 39,
    "malachi": 39,
    "mat": 40,
    "matthew": 40,
    "mrk": 41,
    "mark": 41,
    "luk": 42,
    "luke": 42,
    "jhn": 43,
    "john": 43,
    "act": 44,
    "acts": 44,
    "rom": 45,
    "romans": 45,
    "1co": 46,
    "1corinthians": 46,
    "2co": 47,
    "2corinthians": 47,
    "gal": 48,
    "galatians": 48,
    "eph": 49,
    "ephesians": 49,
    "php": 50,
    "philippians": 50,
    "col": 51,
    "colossians": 51,
    "1th": 52,
    "1thessalonians": 52,
    "2th": 53,
    "2thessalonians": 53,
    "1ti": 54,
    "1timothy": 54,
    "2ti": 55,
    "2timothy": 55,
    "tit": 56,
    "titus": 56,
    "phm": 57,
    "philemon": 57,
    "heb": 58,
    "hebrews": 58,
    "jas": 59,
    "james": 59,
    "1pe": 60,
    "1peter": 60,
    "2pe": 61,
    "2peter": 61,
    "1jn": 62,
    "1john": 62,
    "2jn": 63,
    "2john": 63,
    "3jn": 64,
    "3john": 64,
    "jud": 65,
    "jude": 65,
    "rev": 66,
    "revelation": 66,
}


@dataclass
class ParsedQuery:
    """Parsed search query with extracted filters and modifiers."""

    text: str = ""
    is_exact_phrase: bool = False
    book_filter: Optional[int] = None
    book_name: Optional[str] = None
    testament_filter: Optional[str] = None  # "old" or "new"
    strongs_number: Optional[str] = None
    language_filter: Optional[str] = None
    search_type: str = "text"  # "text", "semantic", "strongs", "hybrid"
    filters_applied: List[str] = field(default_factory=list)

    @property
    def has_filters(self) -> bool:
        """Return True if any filters are applied."""
        return bool(self.book_filter or self.testament_filter or self.language_filter)


# Pattern for ``key:value`` modifiers
_MODIFIER_PATTERN = re.compile(r"(\w+):(\S+)")
# Pattern for quoted phrases
_QUOTED_PATTERN = re.compile(r'"([^"]+)"')


_MODIFIER_HANDLERS = {}


def _handle_book(result: ParsedQuery, value: str) -> bool:
    resolved = BOOK_NAME_TO_ID.get(value.replace(" ", ""))
    if resolved:
        result.book_filter = resolved
        result.book_name = value
        result.filters_applied.append(f"book={value}")
        return True
    return False


def _handle_testament(result: ParsedQuery, value: str) -> bool:
    if value in ("old", "ot"):
        result.testament_filter = "old"
    elif value in ("new", "nt"):
        result.testament_filter = "new"
    result.filters_applied.append(f"testament={result.testament_filter}")
    return True


def _handle_strongs(result: ParsedQuery, value: str) -> bool:
    result.strongs_number = value.upper()
    result.search_type = "strongs"
    result.filters_applied.append(f"strongs={value.upper()}")
    return True


def _handle_language(result: ParsedQuery, value: str) -> bool:
    if value in ("hebrew", "greek", "aramaic"):
        result.language_filter = value
        result.filters_applied.append(f"language={value}")
        return True
    return False


def _handle_type(result: ParsedQuery, value: str) -> bool:
    if value in ("text", "semantic", "hybrid"):
        result.search_type = value
        return True
    return False


_MODIFIER_HANDLERS = {
    "in": _handle_book,
    "book": _handle_book,
    "testament": _handle_testament,
    "strongs": _handle_strongs,
    "language": _handle_language,
    "lang": _handle_language,
    "type": _handle_type,
}


def _apply_modifier(result: ParsedQuery, key: str, value: str) -> bool:
    """Apply a single key:value modifier to a ParsedQuery. Returns True if consumed."""
    handler = _MODIFIER_HANDLERS.get(key)
    if handler:
        return handler(result, value)
    return False


def _detect_shorthand(result: ParsedQuery) -> None:
    """Detect shorthand testament filters (ot/nt) and bare Strong's numbers."""
    text_lower = result.text.lower().strip()
    if text_lower.endswith(" ot"):
        result.testament_filter = "old"
        result.text = result.text[:-3].strip()
        result.filters_applied.append("testament=old")
    elif text_lower.endswith(" nt"):
        result.testament_filter = "new"
        result.text = result.text[:-3].strip()
        result.filters_applied.append("testament=new")

    if not result.strongs_number and re.match(r"^[HG]\d{3,5}$", result.text.strip(), re.IGNORECASE):
        result.strongs_number = result.text.strip().upper()
        result.search_type = "strongs"


def parse_query(raw_query: str) -> ParsedQuery:
    """Parse a raw search string into a structured ParsedQuery.

    Args:
        raw_query: User-entered search string.

    Returns:
        ParsedQuery with extracted text, filters, and search type.
    """
    result = ParsedQuery()

    if not raw_query or not raw_query.strip():
        return result

    query = raw_query.strip()

    # Extract quoted phrases
    quoted_match = _QUOTED_PATTERN.search(query)
    if quoted_match:
        result.is_exact_phrase = True
        result.text = quoted_match.group(1)
        query = _QUOTED_PATTERN.sub("", query).strip()
    else:
        result.text = query

    # Extract key:value modifiers
    modifiers_found: List[str] = []
    for match in _MODIFIER_PATTERN.finditer(query):
        key = match.group(1).lower()
        value = match.group(2).lower()
        if _apply_modifier(result, key, value):
            modifiers_found.append(match.group(0))

    # Remove extracted modifiers from text
    if not result.is_exact_phrase:
        cleaned = query
        for mod in modifiers_found:
            cleaned = cleaned.replace(mod, "")
        result.text = " ".join(cleaned.split())

    _detect_shorthand(result)
    return result
