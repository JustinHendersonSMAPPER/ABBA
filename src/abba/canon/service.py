"""
Canon service providing high-level operations for biblical canon management.

Integrates canon registry, versification engine, translation repository, and
comparison tools into a unified service interface.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import json

from .models import (
    Canon,
    CanonTradition,
    BookClassification,
    BookSection,
    Translation,
    TranslationPhilosophy,
    LicenseType,
    VersificationScheme,
    VerseMapping,
    MappingType,
)
from .registry import CanonRegistry, canon_registry
from .versification import VersificationEngine, MappingResult
from .translation import TranslationRepository
from .comparison import CanonComparator, ComparisonResult
from ..verse_id import VerseID


class CanonService:
    """High-level service for canon-related operations."""

    def __init__(self, registry: Optional[CanonRegistry] = None, data_path: Optional[str] = None):
        """Initialize the canon service.

        Args:
            registry: Canon registry to use (defaults to global)
            data_path: Path to canon data directory
        """
        self.logger = logging.getLogger(__name__)

        # Use provided or global registry
        self.registry = registry or canon_registry

        # Initialize components
        self.versification = VersificationEngine()
        self.translations = TranslationRepository(data_path)
        self.comparator = CanonComparator(self.registry)

        # Data path for persistence
        self.data_path = Path(data_path) if data_path else None

        # Cache for computed data
        self._book_support_cache: Dict[str, Dict[str, bool]] = {}
        self._verse_existence_cache: Dict[Tuple[str, str], bool] = {}

    # Canon Operations

    def get_canon(self, canon_id: str) -> Optional[Canon]:
        """Get a canon by ID."""
        return self.registry.get_canon(canon_id)

    def list_canons(self, tradition: Optional[CanonTradition] = None) -> List[Canon]:
        """List all canons, optionally filtered by tradition."""
        if tradition:
            return self.registry.get_canons_by_tradition(tradition)
        else:
            return [
                self.registry.get_canon(cid)
                for cid in self.registry.list_canons()
                if self.registry.get_canon(cid)
            ]

    def get_canon_for_translation(self, translation_id: str) -> Optional[Canon]:
        """Get the canon used by a specific translation."""
        translation = self.translations.get_translation(translation_id)
        if translation:
            return self.registry.get_canon(translation.canon_id)
        return None

    # Book Operations

    def get_book_support(self, book_id: str) -> Dict[str, bool]:
        """Get which canons include a specific book.

        Args:
            book_id: Book identifier

        Returns:
            Dictionary mapping canon IDs to inclusion status
        """
        if book_id in self._book_support_cache:
            return self._book_support_cache[book_id]

        support = {}
        for canon_id in self.registry.list_canons():
            canon = self.registry.get_canon(canon_id)
            if canon:
                support[canon_id] = canon.has_book(book_id)

        self._book_support_cache[book_id] = support
        return support

    def get_universal_books(self) -> Set[str]:
        """Get books present in all canons."""
        canon_ids = self.registry.list_canons()
        if not canon_ids:
            return set()

        return self.registry.get_common_books(canon_ids)

    def get_tradition_specific_books(self, tradition: CanonTradition) -> Set[str]:
        """Get books unique to a specific tradition."""
        tradition_canons = self.registry.get_canons_by_tradition(tradition)
        if not tradition_canons:
            return set()

        # Get all books in this tradition
        tradition_books = set()
        for canon in tradition_canons:
            tradition_books.update(canon.get_book_ids())

        # Get books from other traditions
        other_books = set()
        for other_tradition in CanonTradition:
            if other_tradition != tradition:
                for canon in self.registry.get_canons_by_tradition(other_tradition):
                    other_books.update(canon.get_book_ids())

        # Return books unique to this tradition
        return tradition_books - other_books

    # Translation Operations

    def get_translation(self, translation_id: str) -> Optional[Translation]:
        """Get a translation by ID."""
        return self.translations.get_translation(translation_id)

    def find_translations(
        self,
        language: Optional[str] = None,
        canon: Optional[str] = None,
        license_type: Optional[LicenseType] = None,
        digital_only: bool = False,
    ) -> List[Translation]:
        """Find translations matching criteria."""
        return self.translations.search_translations(
            language=language, canon=canon, digital_only=digital_only
        )

    def get_translation_options(
        self, verse_id: VerseID, target_language: Optional[str] = None
    ) -> List[Translation]:
        """Get available translations for a specific verse.

        Args:
            verse_id: Verse reference
            target_language: Optional language filter

        Returns:
            List of translations that include this verse
        """
        options = []

        for translation in self.translations.list_translations():
            trans = self.translations.get_translation(translation)
            if not trans:
                continue

            # Check language
            if target_language and trans.language_code != target_language:
                continue

            # Check if canon includes this book
            canon = self.registry.get_canon(trans.canon_id)
            if canon and canon.has_book(verse_id.book):
                # TODO: Could also check verse existence via versification
                options.append(trans)

        return options

    # Versification Operations

    def map_verse(
        self, verse_id: VerseID, from_translation: str, to_translation: str
    ) -> MappingResult:
        """Map a verse between two translations.

        Args:
            verse_id: Source verse reference
            from_translation: Source translation ID
            to_translation: Target translation ID

        Returns:
            Mapping result with target verse(s)
        """
        # Get translations
        source_trans = self.translations.get_translation(from_translation)
        target_trans = self.translations.get_translation(to_translation)

        if not source_trans or not target_trans:
            return MappingResult(
                success=False,
                source_verses=[verse_id],
                target_verses=[],
                mapping_type=MappingType.NULL_MAPPING,
                notes="Translation not found",
            )

        # Map between versification schemes
        return self.versification.map_verse(
            verse_id, source_trans.versification_scheme_id, target_trans.versification_scheme_id
        )

    def check_verse_existence(self, verse_id: VerseID, canon_id: str) -> bool:
        """Check if a verse exists in a specific canon.

        Args:
            verse_id: Verse reference
            canon_id: Canon to check

        Returns:
            True if verse exists in canon
        """
        cache_key = (str(verse_id), canon_id)
        if cache_key in self._verse_existence_cache:
            return self._verse_existence_cache[cache_key]

        canon = self.registry.get_canon(canon_id)
        if not canon:
            return False

        # First check if book exists
        if not canon.has_book(verse_id.book):
            self._verse_existence_cache[cache_key] = False
            return False

        # TODO: Could check specific verse bounds based on canon data
        # For now, assume verse exists if book exists
        exists = True

        self._verse_existence_cache[cache_key] = exists
        return exists

    # Comparison Operations

    def compare_canons(self, canon1_id: str, canon2_id: str) -> Optional[ComparisonResult]:
        """Compare two canons."""
        return self.comparator.compare_canons(canon1_id, canon2_id)

    def find_optimal_canon(self, book_ids: List[str]) -> Optional[Canon]:
        """Find the most suitable canon for a set of books.

        Args:
            book_ids: List of required books

        Returns:
            Canon that includes all books, or None
        """
        for canon_id in self.registry.list_canons():
            canon = self.registry.get_canon(canon_id)
            if canon and all(canon.has_book(bid) for bid in book_ids):
                return canon

        return None

    # Analysis Operations

    def analyze_book_coverage(self, book_id: str) -> Dict[str, Any]:
        """Analyze a book's coverage across canons and translations.

        Args:
            book_id: Book to analyze

        Returns:
            Comprehensive coverage analysis
        """
        analysis = {
            "book_id": book_id,
            "canon_coverage": {},
            "tradition_coverage": {},
            "translation_coverage": {"total": 0, "by_language": {}, "by_philosophy": {}},
            "classifications": {},
        }

        # Canon coverage
        for canon_id in self.registry.list_canons():
            canon = self.registry.get_canon(canon_id)
            if canon and canon.has_book(book_id):
                analysis["canon_coverage"][canon_id] = {
                    "name": canon.name,
                    "tradition": canon.tradition.value,
                    "order": canon.get_book_order(book_id),
                }

                # Track classification
                for cb in canon.books:
                    if cb.book_id == book_id:
                        classification = cb.classification.value
                        if classification not in analysis["classifications"]:
                            analysis["classifications"][classification] = []
                        analysis["classifications"][classification].append(canon_id)
                        break

        # Tradition coverage
        for tradition in CanonTradition:
            canons = self.registry.get_canons_by_tradition(tradition)
            tradition_has_book = any(c.has_book(book_id) for c in canons)
            analysis["tradition_coverage"][tradition.value] = tradition_has_book

        # Translation coverage
        for trans_id in self.translations.list_translations():
            trans = self.translations.get_translation(trans_id)
            if trans:
                canon = self.registry.get_canon(trans.canon_id)
                if canon and canon.has_book(book_id):
                    analysis["translation_coverage"]["total"] += 1

                    # By language
                    if trans.language_code not in analysis["translation_coverage"]["by_language"]:
                        analysis["translation_coverage"]["by_language"][trans.language_code] = 0
                    analysis["translation_coverage"]["by_language"][trans.language_code] += 1

                    # By philosophy
                    philosophy = trans.philosophy.value
                    if philosophy not in analysis["translation_coverage"]["by_philosophy"]:
                        analysis["translation_coverage"]["by_philosophy"][philosophy] = 0
                    analysis["translation_coverage"]["by_philosophy"][philosophy] += 1

        return analysis

    def get_canon_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all canons."""
        stats = {
            "total_canons": len(self.registry.list_canons()),
            "total_unique_books": len(self._get_all_books()),
            "by_tradition": {},
            "by_book_count": {},
            "versification_schemes": len(self.registry.list_versification_schemes()),
            "total_translations": len(self.translations.list_translations()),
            "most_inclusive": None,
            "most_restrictive": None,
        }

        # Tradition breakdown
        for tradition in CanonTradition:
            canons = self.registry.get_canons_by_tradition(tradition)
            stats["by_tradition"][tradition.value] = len(canons)

        # Book count distribution
        min_books = float("inf")
        max_books = 0

        for canon_id in self.registry.list_canons():
            canon = self.registry.get_canon(canon_id)
            if canon:
                count = canon.book_count
                if str(count) not in stats["by_book_count"]:
                    stats["by_book_count"][str(count)] = []
                stats["by_book_count"][str(count)].append(canon.id)

                if count < min_books:
                    min_books = count
                    stats["most_restrictive"] = canon.name

                if count > max_books:
                    max_books = count
                    stats["most_inclusive"] = canon.name

        return stats

    def _get_all_books(self) -> Set[str]:
        """Get set of all books across all canons."""
        all_books = set()
        for canon_id in self.registry.list_canons():
            canon = self.registry.get_canon(canon_id)
            if canon:
                all_books.update(canon.get_book_ids())
        return all_books

    # Export/Import Operations

    def export_canon_data(self, output_path: str, format: str = "json") -> None:
        """Export all canon data to file.

        Args:
            output_path: Output file path
            format: Export format (currently only 'json')
        """
        data = {
            "canons": {},
            "versification_schemes": {},
            "statistics": self.get_canon_statistics(),
        }

        # Export canons
        for canon_id in self.registry.list_canons():
            canon = self.registry.get_canon(canon_id)
            if canon:
                data["canons"][canon_id] = {
                    "name": canon.name,
                    "tradition": canon.tradition.value,
                    "book_count": canon.book_count,
                    "books": [
                        {
                            "id": cb.book_id,
                            "order": cb.order,
                            "section": cb.section.value,
                            "classification": cb.classification.value,
                            "name": cb.canonical_name,
                        }
                        for cb in sorted(canon.books, key=lambda x: x.order)
                    ],
                }

        # Export versification schemes
        for scheme_id in self.registry.list_versification_schemes():
            scheme = self.registry.get_versification_scheme(scheme_id)
            if scheme:
                data["versification_schemes"][scheme_id] = {
                    "name": scheme.name,
                    "description": scheme.description,
                    "base_text": scheme.base_text,
                    "includes_apocrypha": scheme.includes_apocrypha,
                }

        # Write to file
        output_file = Path(output_path)
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported canon data to {output_path}")

    def generate_canon_report(self, canon_id: str) -> str:
        """Generate a detailed report for a specific canon.

        Args:
            canon_id: Canon to report on

        Returns:
            Formatted report string
        """
        canon = self.registry.get_canon(canon_id)
        if not canon:
            return f"Canon '{canon_id}' not found"

        lines = [
            f"# Canon Report: {canon.name}",
            f"\nTradition: {canon.tradition.value}",
            f"Total Books: {canon.book_count}",
            f"Established: {canon.established_date or 'Unknown'}",
            f"Authority: {canon.authority or 'Unknown'}",
            f"Versification: {canon.versification_scheme_id}",
            f"\n## Book List\n",
        ]

        # Group books by section
        by_section = {}
        for cb in canon.books:
            if cb.section not in by_section:
                by_section[cb.section] = []
            by_section[cb.section].append(cb)

        # Output by section
        for section in BookSection:
            if section in by_section:
                lines.append(f"\n### {section.value.replace('_', ' ').title()}")
                for cb in sorted(by_section[section], key=lambda x: x.order):
                    classification = ""
                    if cb.classification != BookClassification.PROTOCANONICAL:
                        classification = f" [{cb.classification.value}]"
                    lines.append(f"{cb.order}. {cb.canonical_name} ({cb.book_id}){classification}")

        # Add comparison with Protestant canon
        if canon_id != "protestant":
            protestant = self.registry.get_canon("protestant")
            if protestant:
                lines.append(f"\n## Comparison with Protestant Canon")

                result = self.comparator.compare_canons("protestant", canon_id)
                if result:
                    lines.append(f"\nBooks unique to {canon.name}:")
                    for book_id in sorted(result.second_only_books):
                        lines.append(f"- {book_id}")

                    if result.first_only_books:
                        lines.append(f"\nBooks in Protestant but not {canon.name}:")
                        for book_id in sorted(result.first_only_books):
                            lines.append(f"- {book_id}")

        # Add available translations
        lines.append(f"\n## Available Translations")
        translations = self.translations.get_translations_by_canon(canon_id)
        if translations:
            by_language = {}
            for trans in translations:
                if trans.language_code not in by_language:
                    by_language[trans.language_code] = []
                by_language[trans.language_code].append(trans)

            for lang in sorted(by_language.keys()):
                lines.append(f"\n{lang}:")
                for trans in by_language[lang]:
                    license_info = " (Public Domain)" if trans.is_public_domain() else ""
                    lines.append(f"- {trans.name} ({trans.abbreviation}){license_info}")
        else:
            lines.append("No translations registered for this canon")

        return "\n".join(lines)
