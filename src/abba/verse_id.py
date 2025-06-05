"""
Verse ID normalization system for the ABBA project.

This module provides utilities for creating and parsing canonical verse IDs
in the format: BOOK.CHAPTER.VERSE[part] (e.g., "GEN.1.1", "ROM.3.23a")
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .book_codes import get_chapter_count, is_valid_book_code, normalize_book_name

# Regex pattern for parsing canonical verse IDs
VERSE_ID_PATTERN = re.compile(r"^([A-Z1-3]{3})\.(\d+)\.(\d+)([a-z])?$")

# Regex pattern for parsing verse ranges
VERSE_RANGE_PATTERN = re.compile(r"^([A-Z1-3]{3}\.\d+\.\d+[a-z]?)-([A-Z1-3]{3}\.\d+\.\d+[a-z]?)$")


@dataclass
class VerseID:
    """Represents a parsed canonical verse identifier."""

    book: str
    chapter: int
    verse: int
    part: Optional[str] = None

    def __str__(self) -> str:
        """Convert back to canonical string format."""
        base = f"{self.book}.{self.chapter}.{self.verse}"
        if self.part:
            return f"{base}{self.part}"
        return base

    def __repr__(self) -> str:
        return f"VerseID('{str(self)}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VerseID):
            return False
        return (
            self.book == other.book
            and self.chapter == other.chapter
            and self.verse == other.verse
            and self.part == other.part
        )

    def __lt__(self, other: object) -> bool:
        """Enable sorting of verse IDs."""
        if not isinstance(other, VerseID):
            return NotImplemented

        # Compare by book order first
        from .book_codes import get_book_order

        book_order_self = get_book_order(self.book) or 0
        book_order_other = get_book_order(other.book) or 0

        if book_order_self != book_order_other:
            return book_order_self < book_order_other

        # Then by chapter
        if self.chapter != other.chapter:
            return self.chapter < other.chapter

        # Then by verse
        if self.verse != other.verse:
            return self.verse < other.verse

        # Finally by part
        if self.part != other.part:
            # None comes before 'a', 'a' before 'b', etc.
            if self.part is None:
                return True
            if other.part is None:
                return False
            return self.part < other.part

        return False

    def __le__(self, other: object) -> bool:
        """Enable <= comparison."""
        if not isinstance(other, VerseID):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        """Enable > comparison."""
        if not isinstance(other, VerseID):
            return NotImplemented
        return not self <= other

    def __ge__(self, other: object) -> bool:
        """Enable >= comparison."""
        if not isinstance(other, VerseID):
            return NotImplemented
        return not self < other

    def __hash__(self) -> int:
        return hash((self.book, self.chapter, self.verse, self.part))

    def to_dict(self) -> Dict[str, Union[str, int, None]]:
        """Convert to dictionary representation."""
        return {
            "canonical_id": str(self),
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "verse_part": self.part,
        }

    def next_verse(self) -> Optional["VerseID"]:
        """Get the next verse in sequence."""
        # Handle verse parts
        if self.part:
            if self.part < "z":
                return VerseID(
                    book=self.book,
                    chapter=self.chapter,
                    verse=self.verse,
                    part=chr(ord(self.part) + 1),
                )
            else:
                # Move to next verse number
                return VerseID(book=self.book, chapter=self.chapter, verse=self.verse + 1)

        # Normal verse increment
        return VerseID(book=self.book, chapter=self.chapter, verse=self.verse + 1)

    def previous_verse(self) -> Optional["VerseID"]:
        """Get the previous verse in sequence."""
        # Handle verse parts
        if self.part and self.part > "a":
            return VerseID(
                book=self.book, chapter=self.chapter, verse=self.verse, part=chr(ord(self.part) - 1)
            )
        elif self.part == "a":
            return VerseID(book=self.book, chapter=self.chapter, verse=self.verse)

        # Handle verse 1
        if self.verse == 1:
            if self.chapter > 1:
                # Move to previous chapter
                # Note: We don't know the last verse of prev chapter without external data
                return VerseID(book=self.book, chapter=self.chapter - 1, verse=1)  # Placeholder
            else:
                # First verse of book
                return None

        return VerseID(book=self.book, chapter=self.chapter, verse=self.verse - 1)


@dataclass
class VerseRange:
    """Represents a range of verses."""

    start: VerseID
    end: VerseID

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"

    def __contains__(self, verse_id: VerseID) -> bool:
        """Check if a verse is within this range."""
        return self.start <= verse_id <= self.end

    def to_list(self) -> List[VerseID]:
        """
        Expand the range to a list of individual verses.
        Note: This is simplified and doesn't handle all edge cases.
        """
        verses = []
        current = self.start

        while current <= self.end:
            verses.append(current)
            next_verse = current.next_verse()
            if not next_verse or next_verse > self.end:
                break
            current = next_verse

        return verses


def parse_verse_id(verse_str: str) -> Optional[VerseID]:
    """
    Parse a verse string into a VerseID object.

    Args:
        verse_str: Verse string in various formats:
                   - Canonical: "GEN.1.1" or "ROM.3.23a"
                   - Common: "Genesis 1:1" or "Gen 1:1"
                   - Compact: "Gn1:1" or "Ge1.1"

    Returns:
        VerseID object or None if parsing fails
    """
    # First try canonical format (preserve case for part extraction)
    match = VERSE_ID_PATTERN.match(verse_str)
    if not match:
        # Try with case normalization but preserve part
        normalized = verse_str
        # Extract the part first
        part_match = re.search(r"([a-zA-Z])$", verse_str)
        if part_match:
            part_char = part_match.group(1)
            # Replace with lowercase part
            normalized = verse_str[:-1] + part_char.lower()
        # Convert book code to uppercase
        if "." in normalized:
            parts = normalized.split(".")
            parts[0] = parts[0].upper()
            normalized = ".".join(parts)
        match = VERSE_ID_PATTERN.match(normalized)

    if match:
        book, chapter, verse, part = match.groups()
        book_code = book.upper()
        # Validate that the book code is valid
        if is_valid_book_code(book_code):
            return VerseID(
                book=book_code, chapter=int(chapter), verse=int(verse), part=part if part else None
            )

    # Try common formats with various separators
    # Handle formats like "Genesis 1:1" or "Gen. 1:1" or "1 John 3:16"
    common_pattern = re.compile(r"^((?:[123]\s*)?[A-Za-z]+\.?)\s*(\d+)[:.](\d+)([a-z])?$")
    match = common_pattern.match(verse_str.strip())
    if match:
        book_str, chapter, verse, part = match.groups()

        # Normalize book name
        book_code = normalize_book_name(book_str.strip())
        if book_code:
            return VerseID(
                book=book_code,
                chapter=int(chapter),
                verse=int(verse),
                part=part.lower() if part else None,
            )

    return None


def create_verse_id(
    book: str, chapter: int, verse: int, part: Optional[str] = None
) -> Optional[VerseID]:
    """
    Create a VerseID with validation.

    Args:
        book: Book name or code
        chapter: Chapter number
        verse: Verse number
        part: Optional verse part (a, b, c, etc.)

    Returns:
        VerseID object or None if invalid
    """
    # Normalize book
    book_code: Optional[str]
    if len(book) == 3 and is_valid_book_code(book):
        book_code = book.upper()
    else:
        book_code = normalize_book_name(book)

    if not book_code:
        return None

    # Validate chapter
    max_chapters = get_chapter_count(book_code)
    if max_chapters and (chapter < 1 or chapter > max_chapters):
        return None

    # Validate verse (basic check)
    if verse < 1:
        return None

    # Validate part
    if part and (len(part) != 1 or not part.islower() or part < "a" or part > "z"):
        return None

    return VerseID(book=book_code, chapter=chapter, verse=verse, part=part)


def parse_verse_range(range_str: str) -> Optional[VerseRange]:
    """
    Parse a verse range string.

    Args:
        range_str: Range in format "GEN.1.1-GEN.1.5" or "Genesis 1:1-5"

    Returns:
        VerseRange object or None if parsing fails
    """
    # Try canonical format first
    # Handle case sensitivity like in parse_verse_id
    normalized_range = range_str
    if "." in range_str:
        parts = range_str.split("-")
        if len(parts) == 2:
            normalized_parts = []
            for part in parts:
                if "." in part:
                    verse_parts = part.split(".")
                    verse_parts[0] = verse_parts[0].upper()
                    normalized_parts.append(".".join(verse_parts))
                else:
                    normalized_parts.append(part)
            normalized_range = "-".join(normalized_parts)

    match = VERSE_RANGE_PATTERN.match(normalized_range)
    if match:
        start_str, end_str = match.groups()
        start = parse_verse_id(start_str)
        end = parse_verse_id(end_str)

        # Validate that both verses are valid and range is correct
        if start and end and start <= end:
            return VerseRange(start=start, end=end)
        else:
            # Return None if either verse is invalid
            return None

    # Try common format: "Genesis 1:1-5" (same chapter)
    simple_range = re.compile(r"^((?:[123]\s*)?[A-Za-z]+\.?)\s*(\d+):(\d+)-(\d+)$")
    match = simple_range.match(range_str.strip())
    if match:
        book_str, chapter, start_verse, end_verse = match.groups()
        book_code = normalize_book_name(book_str.strip())

        if book_code:
            start = create_verse_id(book_code, int(chapter), int(start_verse))
            end = create_verse_id(book_code, int(chapter), int(end_verse))

            if start and end and start <= end:
                return VerseRange(start=start, end=end)

    # Try format with hyphen between full references
    parts = range_str.split("-")
    if len(parts) == 2:
        start = parse_verse_id(parts[0].strip())
        end = parse_verse_id(parts[1].strip())

        if start and end and start <= end:
            return VerseRange(start=start, end=end)

    return None


def normalize_verse_id(verse_str: str) -> Optional[str]:
    """
    Normalize any verse reference to canonical format.

    Args:
        verse_str: Verse in any supported format

    Returns:
        Canonical verse ID string or None if invalid
    """
    verse_id = parse_verse_id(verse_str)
    return str(verse_id) if verse_id else None


def is_valid_verse_id(verse_str: str) -> bool:
    """Check if a string is a valid verse ID."""
    return parse_verse_id(verse_str) is not None


def compare_verse_ids(verse1: Union[str, VerseID], verse2: Union[str, VerseID]) -> int:
    """
    Compare two verse IDs.

    Returns:
        -1 if verse1 < verse2
         0 if verse1 == verse2
         1 if verse1 > verse2
    """
    # Convert strings to VerseID if needed
    verse1_id: Optional[VerseID] = verse1 if isinstance(verse1, VerseID) else parse_verse_id(verse1)
    verse2_id: Optional[VerseID] = verse2 if isinstance(verse2, VerseID) else parse_verse_id(verse2)

    if not verse1_id or not verse2_id:
        raise ValueError("Invalid verse ID")

    if verse1_id < verse2_id:
        return -1
    elif verse1_id > verse2_id:
        return 1
    else:
        return 0


def get_verse_parts(base_verse: Union[str, VerseID]) -> List[VerseID]:
    """
    Get all parts of a verse (e.g., for "ROM.3.23", return any ROM.3.23a, ROM.3.23b, etc.)

    This implementation includes common verse divisions found in biblical texts.
    """
    verse_id: Optional[VerseID] = (
        base_verse if isinstance(base_verse, VerseID) else parse_verse_id(base_verse)
    )

    if not verse_id:
        return []

    # Common verses that have parts in various manuscripts/translations
    # This is a simplified mapping - a full implementation would use a database
    VERSES_WITH_PARTS = {
        # New Testament examples
        "ROM.3.23": ["a", "b"],
        "JHN.5.3": ["a", "b"],
        "JHN.5.4": ["a", "b"],
        "ACT.8.37": ["a", "b"],
        "MRK.7.16": ["a", "b"],
        "MRK.9.44": ["a", "b"],
        "MRK.9.46": ["a", "b"],
        "MRK.11.26": ["a", "b"],
        "MRK.15.28": ["a", "b"],
        "LUK.17.36": ["a", "b"],
        "LUK.23.17": ["a", "b"],
        # Old Testament examples  
        "1SA.13.1": ["a", "b"],
        "2KI.8.16": ["a", "b"],
        "PSA.145.13": ["a", "b"],
        "DAN.3.23": ["a", "b", "c"],  # Addition of the Three Young Men
    }

    verse_key = f"{verse_id.book}.{verse_id.chapter}.{verse_id.verse}"
    
    # If verse already has a part, just return it
    if verse_id.part:
        return [verse_id]
    
    # Check if this verse has known parts
    if verse_key in VERSES_WITH_PARTS:
        parts = []
        # Add the base verse first
        parts.append(verse_id)
        # Add all part variations
        for part in VERSES_WITH_PARTS[verse_key]:
            parts.append(VerseID(
                book=verse_id.book,
                chapter=verse_id.chapter,
                verse=verse_id.verse,
                part=part
            ))
        return parts
    
    # Default: return just the base verse
    return [verse_id]
