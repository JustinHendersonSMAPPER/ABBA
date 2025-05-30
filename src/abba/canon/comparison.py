"""
Canon comparison tools for analyzing differences between traditions.

Provides utilities for comparing biblical canons, identifying unique books,
common content, and structural differences between traditions.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .models import (
    Canon,
    CanonBook,
    BookSection,
    BookClassification,
    CanonDifference,
    CanonTradition,
)
from .registry import CanonRegistry


@dataclass
class ComparisonResult:
    """Result of comparing two canons."""

    first_canon: Canon
    second_canon: Canon

    # Book presence
    common_books: Set[str]
    first_only_books: Set[str]
    second_only_books: Set[str]

    # Structural differences
    order_differences: List[CanonDifference]
    section_differences: List[CanonDifference]
    name_differences: List[CanonDifference]

    # Statistics
    similarity_score: float  # 0.0 to 1.0

    def get_summary(self) -> str:
        """Get human-readable summary of comparison."""
        lines = [
            f"Comparison: {self.first_canon.name} vs {self.second_canon.name}",
            f"Similarity: {self.similarity_score:.1%}",
            f"Common books: {len(self.common_books)}",
            f"Unique to {self.first_canon.name}: {len(self.first_only_books)}",
            f"Unique to {self.second_canon.name}: {len(self.second_only_books)}",
            f"Order differences: {len(self.order_differences)}",
            f"Section differences: {len(self.section_differences)}",
            f"Name differences: {len(self.name_differences)}",
        ]
        return "\n".join(lines)


class CanonComparator:
    """Compare and analyze differences between biblical canons."""

    def __init__(self, registry: Optional[CanonRegistry] = None):
        """Initialize the canon comparator.

        Args:
            registry: Canon registry to use (defaults to global registry)
        """
        self.logger = logging.getLogger(__name__)
        self.registry = registry

        # Cache for comparison results
        self._comparison_cache: Dict[Tuple[str, str], ComparisonResult] = {}

    def compare_canons(
        self, first_canon_id: str, second_canon_id: str
    ) -> Optional[ComparisonResult]:
        """Compare two canons and identify all differences.

        Args:
            first_canon_id: ID of first canon
            second_canon_id: ID of second canon

        Returns:
            ComparisonResult with detailed differences
        """
        # Check cache
        cache_key = (first_canon_id, second_canon_id)
        if cache_key in self._comparison_cache:
            return self._comparison_cache[cache_key]

        # Get canons
        if self.registry:
            first_canon = self.registry.get_canon(first_canon_id)
            second_canon = self.registry.get_canon(second_canon_id)
        else:
            # Import here to avoid circular dependency
            from .registry import canon_registry

            first_canon = canon_registry.get_canon(first_canon_id)
            second_canon = canon_registry.get_canon(second_canon_id)

        if not first_canon or not second_canon:
            self.logger.error(
                f"Canon not found: {first_canon_id if not first_canon else second_canon_id}"
            )
            return None

        # Get book sets
        first_books = set(first_canon.get_book_ids())
        second_books = set(second_canon.get_book_ids())

        # Find common and unique books
        common_books = first_books & second_books
        first_only_books = first_books - second_books
        second_only_books = second_books - first_books

        # Analyze differences
        order_differences = self._find_order_differences(first_canon, second_canon, common_books)
        section_differences = self._find_section_differences(
            first_canon, second_canon, common_books
        )
        name_differences = self._find_name_differences(first_canon, second_canon, common_books)

        # Calculate similarity score
        similarity_score = self._calculate_similarity(
            first_canon,
            second_canon,
            common_books,
            first_only_books,
            second_only_books,
            order_differences,
            section_differences,
        )

        # Create result
        result = ComparisonResult(
            first_canon=first_canon,
            second_canon=second_canon,
            common_books=common_books,
            first_only_books=first_only_books,
            second_only_books=second_only_books,
            order_differences=order_differences,
            section_differences=section_differences,
            name_differences=name_differences,
            similarity_score=similarity_score,
        )

        # Cache result
        self._comparison_cache[cache_key] = result

        return result

    def _find_order_differences(
        self, first: Canon, second: Canon, common_books: Set[str]
    ) -> List[CanonDifference]:
        """Find books that appear in different order."""
        differences = []

        # Create position maps
        first_positions = {cb.book_id: cb.order for cb in first.books}
        second_positions = {cb.book_id: cb.order for cb in second.books}

        # Check each common book
        for book_id in common_books:
            first_pos = first_positions.get(book_id)
            second_pos = second_positions.get(book_id)

            if first_pos and second_pos:
                # Calculate relative position (0.0 to 1.0)
                first_rel = first_pos / len(first.books)
                second_rel = second_pos / len(second.books)

                # If relative positions differ significantly
                if abs(first_rel - second_rel) > 0.05:  # 5% threshold
                    differences.append(
                        CanonDifference(
                            difference_type="book_order",
                            book_id=book_id,
                            first_canon_position=first_pos,
                            second_canon_position=second_pos,
                            description=f"{book_id}: position {first_pos} vs {second_pos}",
                        )
                    )

        return differences

    def _find_section_differences(
        self, first: Canon, second: Canon, common_books: Set[str]
    ) -> List[CanonDifference]:
        """Find books that appear in different sections."""
        differences = []

        # Create section maps
        first_sections = {cb.book_id: cb.section for cb in first.books}
        second_sections = {cb.book_id: cb.section for cb in second.books}

        for book_id in common_books:
            first_section = first_sections.get(book_id)
            second_section = second_sections.get(book_id)

            if first_section != second_section:
                differences.append(
                    CanonDifference(
                        difference_type="book_section",
                        book_id=book_id,
                        first_canon_section=first_section,
                        second_canon_section=second_section,
                        description=f"{book_id}: {first_section.value} vs {second_section.value}",
                    )
                )

        return differences

    def _find_name_differences(
        self, first: Canon, second: Canon, common_books: Set[str]
    ) -> List[CanonDifference]:
        """Find books with different canonical names."""
        differences = []

        # Create name maps
        first_names = {cb.book_id: cb.canonical_name for cb in first.books}
        second_names = {cb.book_id: cb.canonical_name for cb in second.books}

        for book_id in common_books:
            first_name = first_names.get(book_id)
            second_name = second_names.get(book_id)

            if first_name != second_name and first_name and second_name:
                differences.append(
                    CanonDifference(
                        difference_type="book_name",
                        book_id=book_id,
                        first_canon_name=first_name,
                        second_canon_name=second_name,
                        description=f"{book_id}: '{first_name}' vs '{second_name}'",
                    )
                )

        return differences

    def _calculate_similarity(
        self,
        first: Canon,
        second: Canon,
        common_books: Set[str],
        first_only: Set[str],
        second_only: Set[str],
        order_diffs: List[CanonDifference],
        section_diffs: List[CanonDifference],
    ) -> float:
        """Calculate overall similarity score between canons."""
        # Weights for different factors
        BOOK_PRESENCE_WEIGHT = 0.5
        ORDER_WEIGHT = 0.25
        SECTION_WEIGHT = 0.25

        # Book presence similarity
        total_books = len(set(first.get_book_ids()) | set(second.get_book_ids()))
        if total_books > 0:
            book_similarity = len(common_books) / total_books
        else:
            book_similarity = 1.0

        # Order similarity (for common books)
        if common_books:
            order_similarity = 1.0 - (len(order_diffs) / len(common_books))
        else:
            order_similarity = 1.0

        # Section similarity (for common books)
        if common_books:
            section_similarity = 1.0 - (len(section_diffs) / len(common_books))
        else:
            section_similarity = 1.0

        # Weighted average
        similarity = (
            BOOK_PRESENCE_WEIGHT * book_similarity
            + ORDER_WEIGHT * order_similarity
            + SECTION_WEIGHT * section_similarity
        )

        return similarity

    def find_book_journey(self, book_id: str) -> Dict[str, Dict[str, any]]:
        """Trace a book's presence across all canons.

        Args:
            book_id: Book identifier

        Returns:
            Dictionary mapping canon IDs to book information
        """
        journey = {}

        if self.registry:
            registry = self.registry
        else:
            from .registry import canon_registry

            registry = canon_registry

        for canon_id in registry.list_canons():
            canon = registry.get_canon(canon_id)
            if canon and canon.has_book(book_id):
                # Find the book in this canon
                for cb in canon.books:
                    if cb.book_id == book_id:
                        journey[canon_id] = {
                            "canon_name": canon.name,
                            "tradition": canon.tradition.value,
                            "order": cb.order,
                            "section": cb.section.value,
                            "classification": cb.classification.value,
                            "canonical_name": cb.canonical_name,
                            "alternate_names": cb.alternate_names,
                        }
                        break

        return journey

    def compare_traditions(
        self, tradition1: CanonTradition, tradition2: CanonTradition
    ) -> List[ComparisonResult]:
        """Compare all canons between two traditions.

        Args:
            tradition1: First tradition
            tradition2: Second tradition

        Returns:
            List of comparison results
        """
        results = []

        if self.registry:
            registry = self.registry
        else:
            from .registry import canon_registry

            registry = canon_registry

        # Get canons for each tradition
        tradition1_canons = registry.get_canons_by_tradition(tradition1)
        tradition2_canons = registry.get_canons_by_tradition(tradition2)

        # Compare each pair
        for canon1 in tradition1_canons:
            for canon2 in tradition2_canons:
                result = self.compare_canons(canon1.id, canon2.id)
                if result:
                    results.append(result)

        return results

    def find_deuterocanonical_differences(self) -> Dict[str, List[str]]:
        """Identify deuterocanonical books across traditions.

        Returns:
            Dictionary mapping canon IDs to their deuterocanonical books
        """
        deuterocanonical_by_canon = {}

        if self.registry:
            registry = self.registry
        else:
            from .registry import canon_registry

            registry = canon_registry

        for canon_id in registry.list_canons():
            canon = registry.get_canon(canon_id)
            if canon:
                deutero_books = [
                    cb.book_id
                    for cb in canon.books
                    if cb.classification == BookClassification.DEUTEROCANONICAL
                ]
                if deutero_books:
                    deuterocanonical_by_canon[canon_id] = deutero_books

        return deuterocanonical_by_canon

    def generate_compatibility_matrix(
        self, canon_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Generate compatibility matrix between canons.

        Args:
            canon_ids: Specific canons to compare (None for all)

        Returns:
            Matrix of similarity scores between canons
        """
        if self.registry:
            registry = self.registry
        else:
            from .registry import canon_registry

            registry = canon_registry

        if not canon_ids:
            canon_ids = registry.list_canons()

        matrix = {}

        for id1 in canon_ids:
            matrix[id1] = {}
            for id2 in canon_ids:
                if id1 == id2:
                    matrix[id1][id2] = 1.0
                else:
                    result = self.compare_canons(id1, id2)
                    if result:
                        matrix[id1][id2] = result.similarity_score
                    else:
                        matrix[id1][id2] = 0.0

        return matrix

    def find_most_inclusive_canon(self) -> Optional[Canon]:
        """Find the canon with the most books.

        Returns:
            The most inclusive canon
        """
        if self.registry:
            registry = self.registry
        else:
            from .registry import canon_registry

            registry = canon_registry

        most_inclusive = None
        max_books = 0

        for canon_id in registry.list_canons():
            canon = registry.get_canon(canon_id)
            if canon and canon.book_count > max_books:
                max_books = canon.book_count
                most_inclusive = canon

        return most_inclusive

    def get_book_classification_summary(self) -> Dict[str, Dict[BookClassification, int]]:
        """Get summary of book classifications across all canons.

        Returns:
            Dictionary mapping canon IDs to classification counts
        """
        summary = {}

        if self.registry:
            registry = self.registry
        else:
            from .registry import canon_registry

            registry = canon_registry

        for canon_id in registry.list_canons():
            canon = registry.get_canon(canon_id)
            if canon:
                classification_counts = defaultdict(int)
                for cb in canon.books:
                    classification_counts[cb.classification] += 1
                summary[canon_id] = dict(classification_counts)

        return summary
