"""
Canon support for different biblical traditions and their versification systems.

This module handles the differences between Protestant, Catholic, Orthodox,
and other biblical canons, including book inclusion and ordering.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from ..book_codes import BookCode
from ..verse_id import VerseID, parse_verse_id
from ..versification import VersificationSystem


class Canon(Enum):
    """Biblical canon traditions."""

    PROTESTANT = "protestant"  # 66 books (standard Protestant canon)
    CATHOLIC = "catholic"  # 73 books (includes deuterocanonical)
    ORTHODOX = "orthodox"  # 76 books (Eastern Orthodox)
    ETHIOPIAN = "ethiopian"  # 81 books (Ethiopian Orthodox)
    SYRIAC = "syriac"  # 61 books (Syriac Peshitta)
    SAMARITAN = "samaritan"  # 5 books (Samaritan Pentateuch only)
    HEBREW_BIBLE = "hebrew_bible"  # 24 books (Hebrew Tanakh ordering)


@dataclass
class CanonInfo:
    """Information about a biblical canon."""

    canon: Canon
    name: str
    description: str
    book_count: int
    includes_deuterocanonical: bool
    primary_versification: VersificationSystem
    book_codes: List[str]
    additional_books: List[str] = None  # Books beyond Protestant canon

    def __post_init__(self) -> None:
        """Initialize additional books list if not provided."""
        if self.additional_books is None:
            self.additional_books = []


class CanonManager:
    """Handle different biblical canons and their versification."""

    def __init__(self) -> None:
        """Initialize the canon manager."""
        self._canon_info = self._load_canon_definitions()
        self._book_canon_map = self._build_book_canon_map()

    @property
    def canon_definitions(self) -> Dict[Canon, CanonInfo]:
        """Get canon definitions."""
        return self._canon_info

    def _load_canon_definitions(self) -> Dict[Canon, CanonInfo]:
        """Load definitions for all supported canons."""
        canon_defs = {}

        # Protestant Canon (66 books)
        # Define the standard Protestant canon books explicitly
        protestant_books = [
            # Old Testament (39 books)
            "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
            "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
            "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
            "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
            "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL",
            # New Testament (27 books)
            "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO",
            "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI",
            "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
            "3JN", "JUD", "REV"
        ]
        canon_defs[Canon.PROTESTANT] = CanonInfo(
            canon=Canon.PROTESTANT,
            name="Protestant Canon",
            description="Standard Protestant Bible with 39 OT and 27 NT books",
            book_count=66,
            includes_deuterocanonical=False,
            primary_versification=VersificationSystem.MODERN,
            book_codes=protestant_books,
        )

        # Catholic Canon (73 books) - Protestant + 7 deuterocanonical
        # Only include deuterocanonical books that exist in BookCode enum
        potential_deutero = ["TOB", "JDT", "MA1", "MA2", "WIS", "SIR", "BAR"]
        deuterocanonical_books = [
            book for book in potential_deutero if any(bc.value == book for bc in BookCode)
        ]
        catholic_books = protestant_books + deuterocanonical_books
        canon_defs[Canon.CATHOLIC] = CanonInfo(
            canon=Canon.CATHOLIC,
            name="Catholic Canon",
            description="Catholic Bible including deuterocanonical books",
            book_count=len(catholic_books),
            includes_deuterocanonical=True,
            primary_versification=VersificationSystem.VULGATE,
            book_codes=catholic_books,
            additional_books=deuterocanonical_books,
        )

        # Orthodox Canon (varies, but typically 76 books)
        potential_orthodox_extra = ["3MC", "4MC", "1ES", "2ES", "PSS"]
        orthodox_extra = [
            book for book in potential_orthodox_extra if any(bc.value == book for bc in BookCode)
        ]
        orthodox_additional = deuterocanonical_books + orthodox_extra
        orthodox_books = protestant_books + orthodox_additional
        canon_defs[Canon.ORTHODOX] = CanonInfo(
            canon=Canon.ORTHODOX,
            name="Eastern Orthodox Canon",
            description="Eastern Orthodox Bible with additional books",
            book_count=len(orthodox_books),
            includes_deuterocanonical=True,
            primary_versification=VersificationSystem.LXX,
            book_codes=orthodox_books,
            additional_books=orthodox_additional,
        )

        # Ethiopian Orthodox Canon (81 books)
        potential_ethiopian_extra = ["JUB", "ENO", "4ES"]
        ethiopian_extra = [
            book for book in potential_ethiopian_extra if any(bc.value == book for bc in BookCode)
        ]
        ethiopian_additional = orthodox_additional + ethiopian_extra
        ethiopian_books = protestant_books + ethiopian_additional
        canon_defs[Canon.ETHIOPIAN] = CanonInfo(
            canon=Canon.ETHIOPIAN,
            name="Ethiopian Orthodox Canon",
            description="Ethiopian Orthodox Bible with extensive additional books",
            book_count=len(ethiopian_books),
            includes_deuterocanonical=True,
            primary_versification=VersificationSystem.LXX,
            book_codes=ethiopian_books,
            additional_books=ethiopian_additional,
        )

        # Syriac Canon (Peshitta) - missing some NT books
        syriac_excluded = ["2PE", "2JN", "3JN", "JUD", "REV"]  # Not in original Peshitta
        syriac_books = [book for book in protestant_books if book not in syriac_excluded]
        canon_defs[Canon.SYRIAC] = CanonInfo(
            canon=Canon.SYRIAC,
            name="Syriac Peshitta Canon",
            description="Syriac Peshitta without some disputed NT books",
            book_count=len(syriac_books),
            includes_deuterocanonical=False,
            primary_versification=VersificationSystem.MODERN,
            book_codes=syriac_books,
        )

        # Samaritan Canon (Torah only)
        samaritan_books = ["GEN", "EXO", "LEV", "NUM", "DEU"]
        canon_defs[Canon.SAMARITAN] = CanonInfo(
            canon=Canon.SAMARITAN,
            name="Samaritan Canon",
            description="Samaritan Pentateuch (Torah only)",
            book_count=len(samaritan_books),
            includes_deuterocanonical=False,
            primary_versification=VersificationSystem.MT,
            book_codes=samaritan_books,
        )

        # Hebrew Bible (Tanakh) - same books as Protestant but different ordering
        # Traditional Hebrew ordering: Torah, Nevi'im, Ketuvim
        hebrew_books = [
            # Torah (5 books)
            "GEN",
            "EXO",
            "LEV",
            "NUM",
            "DEU",
            # Nevi'im - Former Prophets (6 books)
            "JOS",
            "JDG",
            "1SA",
            "2SA",
            "1KI",
            "2KI",
            # Nevi'im - Latter Prophets (15 books)
            "ISA",
            "JER",
            "EZK",  # Major prophets
            "HOS",
            "JOL",
            "AMO",
            "OBA",
            "JON",
            "MIC",
            "NAM",
            "HAB",
            "ZEP",
            "HAG",
            "ZEC",
            "MAL",  # Minor prophets
            # Ketuvim (13 books)
            "PSA",
            "PRO",
            "JOB",  # Poetic books
            "SNG",
            "RUT",
            "LAM",
            "ECC",
            "EST",  # Five scrolls
            "DAN",
            "EZR",
            "NEH",
            "1CH",
            "2CH",  # Other writings
        ]
        canon_defs[Canon.HEBREW_BIBLE] = CanonInfo(
            canon=Canon.HEBREW_BIBLE,
            name="Hebrew Bible (Tanakh)",
            description="Hebrew Bible in traditional Jewish ordering",
            book_count=len(hebrew_books),  # Use actual count
            includes_deuterocanonical=False,
            primary_versification=VersificationSystem.MT,
            book_codes=hebrew_books,
        )

        return canon_defs

    def _build_book_canon_map(self) -> Dict[str, List[Canon]]:
        """Build mapping from book codes to canons that include them."""
        book_canon_map = {}

        for canon, info in self._canon_info.items():
            for book_code in info.book_codes:
                if book_code not in book_canon_map:
                    book_canon_map[book_code] = []
                book_canon_map[book_code].append(canon)

        return book_canon_map

    def get_canon_books(self, canon: Canon) -> List[str]:
        """
        Get complete book list for a canon tradition.

        Args:
            canon: Canon tradition

        Returns:
            List of book codes in canonical order
        """
        if canon in self._canon_info:
            return self._canon_info[canon].book_codes.copy()
        return []

    def get_canon_info(self, canon: Canon) -> Optional[CanonInfo]:
        """
        Get information about a canon.

        Args:
            canon: Canon to get information for

        Returns:
            CanonInfo object or None if canon not found
        """
        return self._canon_info.get(canon)

    def is_book_in_canon(self, book_code: str, canon: Canon) -> bool:
        """
        Check if a book is included in a specific canon.

        Args:
            book_code: Book code to check
            canon: Canon to check against

        Returns:
            True if book is in the canon
        """
        if canon in self._canon_info:
            return book_code in self._canon_info[canon].book_codes
        return False

    def get_book_canons(self, book_code: str) -> List[Canon]:
        """
        Get all canons that include a specific book.

        Args:
            book_code: Book code to look up

        Returns:
            List of canons that include the book
        """
        return self._book_canon_map.get(book_code, [])

    def map_across_canons(
        self, verse_id: VerseID, from_canon: Canon, to_canon: Canon
    ) -> Optional[VerseID]:
        """
        Handle books that exist in some canons but not others.

        Args:
            verse_id: Verse ID to map
            from_canon: Source canon
            to_canon: Target canon

        Returns:
            Mapped verse ID or None if book doesn't exist in target canon
        """
        # Check if book exists in target canon
        if not self.is_book_in_canon(verse_id.book, to_canon):
            return None

        # If book exists in both canons, verse ID is the same
        # (versification differences are handled by the verse mapper)
        if self.is_book_in_canon(verse_id.book, from_canon):
            return verse_id

        return None

    def validate_canon_coverage(self, verse_ids: List[VerseID], canon: Canon) -> Dict[str, any]:
        """
        Ensure complete coverage for a canon tradition.

        Args:
            verse_ids: List of verse IDs to validate
            canon: Canon to validate against

        Returns:
            Dictionary with validation results
        """
        canon_books = set(self.get_canon_books(canon))
        provided_books = set(verse.book for verse in verse_ids)

        missing_books = canon_books - provided_books
        extra_books = provided_books - canon_books

        return {
            "canon": canon.value,
            "expected_books": len(canon_books),
            "provided_books": len(provided_books),
            "missing_books": list(missing_books),
            "extra_books": list(extra_books),
            "coverage_complete": len(missing_books) == 0,
            "coverage_percentage": (
                len(provided_books & canon_books) / len(canon_books) if canon_books else 0
            ),
        }

    def get_deuterocanonical_books(self) -> List[str]:
        """
        Get list of deuterocanonical book codes.

        Returns:
            List of book codes for deuterocanonical books
        """
        catholic_info = self._canon_info.get(Canon.CATHOLIC)
        if catholic_info and catholic_info.additional_books:
            return catholic_info.additional_books.copy()
        return []

    def is_deuterocanonical(self, book_code: str) -> bool:
        """
        Check if a book is deuterocanonical.

        Args:
            book_code: Book code to check

        Returns:
            True if book is deuterocanonical
        """
        deutero_books = self.get_deuterocanonical_books()
        return book_code in deutero_books

    def get_canon_differences(self, canon1: Canon, canon2: Canon) -> Dict[str, List[str]]:
        """
        Get differences between two canons.

        Args:
            canon1: First canon
            canon2: Second canon

        Returns:
            Dictionary with books unique to each canon
        """
        books1 = set(self.get_canon_books(canon1))
        books2 = set(self.get_canon_books(canon2))

        return {
            f"only_in_{canon1.value}": list(books1 - books2),
            f"only_in_{canon2.value}": list(books2 - books1),
            "common_books": list(books1 & books2),
        }

    def get_supported_canons(self) -> List[Canon]:
        """
        Get list of all supported canons.

        Returns:
            List of supported Canon values
        """
        return list(self._canon_info.keys())

    def filter_verses_by_canon(self, verse_ids: List[VerseID], canon: Canon) -> List[VerseID]:
        """
        Filter a list of verses to only include those in a specific canon.

        Args:
            verse_ids: List of verse IDs to filter
            canon: Canon to filter by

        Returns:
            Filtered list of verse IDs
        """
        canon_books = set(self.get_canon_books(canon))
        return [verse for verse in verse_ids if verse.book in canon_books]

    def get_canon_statistics(self, canon: Canon) -> Dict[str, any]:
        """
        Get statistics about a canon.

        Args:
            canon: Canon to get statistics for

        Returns:
            Dictionary with canon statistics
        """
        if canon not in self._canon_info:
            return {}

        info = self._canon_info[canon]
        books = info.book_codes

        # Count OT and NT books
        ot_books = []
        nt_books = []

        # Define NT books explicitly
        nt_book_codes = [
            "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO",
            "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI",
            "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
            "3JN", "JUD", "REV"
        ]
        
        for book_code in books:
            if book_code in nt_book_codes:
                nt_books.append(book_code)
            else:
                # Everything else is OT (including deuterocanonical)
                ot_books.append(book_code)

        stats = {
            "total_books": len(books),
            "ot_books": len(ot_books),
            "nt_books": len(nt_books),
            "includes_deuterocanonical": info.includes_deuterocanonical,
            "primary_versification": info.primary_versification.value,
        }

        if info.includes_deuterocanonical and info.additional_books:
            stats["deuterocanonical_books"] = len(info.additional_books)

        return stats

    def compare_canons(self, canon1: Canon, canon2: Canon) -> Dict[str, any]:
        """
        Compare two canons and show differences.

        Args:
            canon1: First canon to compare
            canon2: Second canon to compare

        Returns:
            Dictionary with comparison results
        """
        books1 = set(self.get_canon_books(canon1))
        books2 = set(self.get_canon_books(canon2))

        common = books1 & books2
        unique1 = books1 - books2
        unique2 = books2 - books1

        return {
            "common_books": list(common),
            "unique_to_first": list(unique1),
            "unique_to_second": list(unique2),
            "total_first": len(books1),
            "total_second": len(books2),
            "overlap_percentage": (
                len(common) / max(len(books1), len(books2)) if books1 or books2 else 0
            ),
        }

    def get_books_by_testament(self, canon: Canon, testament: str) -> List[str]:
        """
        Get books by testament (OT or NT) for a canon.

        Args:
            canon: Canon to get books from
            testament: "ot" for Old Testament, "nt" for New Testament

        Returns:
            List of book codes for the specified testament
        """
        if canon not in self._canon_info:
            return []

        testament = testament.lower()
        if testament not in ["ot", "nt"]:
            return []

        books = self.get_canon_books(canon)

        ot_books = []
        nt_books = []

        # Define NT books explicitly
        nt_book_codes = [
            "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO",
            "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI",
            "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
            "3JN", "JUD", "REV"
        ]
        
        for book_code in books:
            if book_code in nt_book_codes:
                nt_books.append(book_code)
            else:
                # Everything else is OT (including deuterocanonical)
                ot_books.append(book_code)

        return ot_books if testament == "ot" else nt_books

    def validate_canon_coverage(self, canon: Canon, provided_books: List[str]) -> Dict[str, any]:
        """
        Validate coverage of books for a canon.

        Args:
            canon: Canon to validate against
            provided_books: List of book codes that were provided

        Returns:
            Dictionary with validation results
        """
        expected_books = set(self.get_canon_books(canon))
        provided_books_set = set(provided_books)

        missing = expected_books - provided_books_set
        extra = provided_books_set - expected_books

        return {
            "is_complete": len(missing) == 0,
            "missing_books": list(missing),
            "extra_books": list(extra),
            "coverage_percentage": (
                len(provided_books_set & expected_books) / len(expected_books)
                if expected_books
                else 0
            ),
            "expected_count": len(expected_books),
            "provided_count": len(provided_books_set),
        }
