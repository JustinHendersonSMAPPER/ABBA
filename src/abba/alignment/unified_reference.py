"""
Unified Reference System (URS) for canonical verse ID generation and mapping.

This module provides the central system for creating canonical verse references
that work across all versification systems and translation traditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

from ..book_codes import BookCode, get_book_info, normalize_book_name
from ..verse_id import VerseID, create_verse_id, parse_verse_id
from ..versification import VersificationSystem, VersificationMapper
from .verse_mapper import EnhancedVerseMapper, MappingType, VerseMapping


@dataclass
class CanonicalMapping:
    """Maps a source reference to its canonical representation."""

    source_ref: str
    canonical_id: VerseID
    source_system: VersificationSystem
    confidence: float
    notes: Optional[str] = None


@dataclass
class VersionMapping:
    """Complete mapping data for a specific translation/version."""

    version_code: str
    versification_system: VersificationSystem
    canonical_mappings: Dict[str, CanonicalMapping]
    coverage_stats: Dict[str, int]  # Statistics about mapping coverage

    def get_canonical_id(self, source_ref: str) -> Optional[VerseID]:
        """Get canonical ID for a source reference."""
        if source_ref in self.canonical_mappings:
            return self.canonical_mappings[source_ref].canonical_id
        return None


class UnifiedReferenceSystem:
    """Central system for canonical verse ID generation and mapping."""

    def __init__(self) -> None:
        """Initialize the Unified Reference System."""
        self.verse_mapper = EnhancedVerseMapper()
        self.versification_mapper = VersificationMapper()
        self.version_mappings: Dict[str, VersionMapping] = {}
        self._canonical_system = VersificationSystem.MODERN

        # Load known translation mappings
        self._load_translation_mappings()

    def _load_translation_mappings(self) -> None:
        """Load known mappings for common Bible translations."""
        # ESV, NIV, NASB, CSB use Modern versification
        modern_translations = ["ESV", "NIV", "NASB", "CSB", "NLT", "HCSB", "NET"]
        for trans in modern_translations:
            self.version_mappings[trans] = VersionMapping(
                version_code=trans,
                versification_system=VersificationSystem.MODERN,
                canonical_mappings={},
                coverage_stats={},
            )

        # KJV and NKJV use KJV versification
        kjv_translations = ["KJV", "NKJV"]
        for trans in kjv_translations:
            self.version_mappings[trans] = VersionMapping(
                version_code=trans,
                versification_system=VersificationSystem.KJV,
                canonical_mappings={},
                coverage_stats={},
            )

        # Catholic translations often use Vulgate-based versification
        catholic_translations = ["NAB", "NABRE", "RSV-CE", "DRA"]
        for trans in catholic_translations:
            self.version_mappings[trans] = VersionMapping(
                version_code=trans,
                versification_system=VersificationSystem.VULGATE,
                canonical_mappings={},
                coverage_stats={},
            )

    def generate_canonical_id(
        self,
        source_ref: str,
        source_system: Optional[VersificationSystem] = None,
        translation_code: Optional[str] = None,
    ) -> Optional[VerseID]:
        """
        Convert any verse reference to canonical form.

        Args:
            source_ref: Source verse reference in any format
            source_system: Versification system of the source (if known)
            translation_code: Translation code (if known)

        Returns:
            Canonical VerseID or None if conversion fails
        """
        # First, try to parse the reference directly
        parsed_verse = parse_verse_id(source_ref)
        if not parsed_verse:
            return None

        # Determine source versification system
        if source_system is None:
            if translation_code and translation_code in self.version_mappings:
                source_system = self.version_mappings[translation_code].versification_system
            else:
                # Default to modern for unknown sources
                source_system = VersificationSystem.MODERN

        # If already in canonical system, return as-is
        if source_system == self._canonical_system:
            return parsed_verse

        # Map to canonical system
        mapping = self.verse_mapper.map_verse(parsed_verse, source_system, self._canonical_system)

        if len(mapping.target_verses) == 1:
            return mapping.target_verses[0]
        elif len(mapping.target_verses) > 1:
            # Return first target for complex mappings
            return mapping.target_verses[0]
        else:
            # No mapping available
            return None

    def create_version_mapping(
        self,
        translation_code: str,
        versification_system: VersificationSystem,
        sample_references: Optional[List[str]] = None,
    ) -> VersionMapping:
        """
        Create complete mapping for a translation.

        Args:
            translation_code: Code for the translation (e.g., "ESV", "NIV")
            versification_system: Versification system used by the translation
            sample_references: Sample references to test mapping

        Returns:
            Complete VersionMapping for the translation
        """
        mapping = VersionMapping(
            version_code=translation_code,
            versification_system=versification_system,
            canonical_mappings={},
            coverage_stats={"total_refs": 0, "mapped_refs": 0, "failed_refs": 0},
        )

        # If sample references provided, create mappings for them
        if sample_references:
            for ref in sample_references:
                canonical_id = self.generate_canonical_id(
                    ref, versification_system, translation_code
                )
                mapping.coverage_stats["total_refs"] += 1

                if canonical_id:
                    canonical_mapping = CanonicalMapping(
                        source_ref=ref,
                        canonical_id=canonical_id,
                        source_system=versification_system,
                        confidence=0.95,  # High confidence for direct mapping
                    )
                    mapping.canonical_mappings[ref] = canonical_mapping
                    mapping.coverage_stats["mapped_refs"] += 1
                else:
                    mapping.coverage_stats["failed_refs"] += 1

        self.version_mappings[translation_code] = mapping
        return mapping

    def resolve_verse_variants(self, verse_id: Union[str, VerseID]) -> List[VerseID]:
        """
        Handle verses with multiple valid representations.

        Args:
            verse_id: Verse ID that may have variants

        Returns:
            List of all valid canonical representations
        """
        if isinstance(verse_id, str):
            parsed_verse = parse_verse_id(verse_id)
        else:
            parsed_verse = verse_id

        if not parsed_verse:
            return []

        variants = [parsed_verse]

        # Check for common variants
        verse_str = str(parsed_verse)

        # Handle verses that might have parts (e.g., "ROM.1.1" might also be "ROM.1.1a")
        if not parsed_verse.part:
            # Check if there's a variant with a part
            variant_with_part = create_verse_id(
                parsed_verse.book, parsed_verse.chapter, parsed_verse.verse, "a"
            )
            if variant_with_part:
                variants.append(variant_with_part)

        # Handle textual variants for specific verses
        textual_variants = self._get_textual_variants(parsed_verse)
        variants.extend(textual_variants)

        return list(set(variants))  # Remove duplicates

    def _get_textual_variants(self, verse_id: VerseID) -> List[VerseID]:
        """Get known textual variants for a verse."""
        variants = []
        verse_str = str(verse_id)

        # Define known textual variants
        textual_variant_map = {
            # Mark's longer ending
            "MRK.16.9": ["MRK.16.9a", "MRK.16.9b"],  # Some traditions split this
            "MRK.16.20": ["MRK.16.20a"],  # Alternative ending
            # John's Pericope Adulterae
            "JHN.7.53": ["JHN.8.1"],  # Sometimes moved
            "JHN.8.11": ["JHN.7.53"],  # Cross-references
            # Romans doxology variations
            "ROM.16.25": ["ROM.14.24", "ROM.16.24a"],  # Appears in different locations
        }

        if verse_str in textual_variant_map:
            for variant_str in textual_variant_map[verse_str]:
                variant_verse = parse_verse_id(variant_str)
                if variant_verse:
                    variants.append(variant_verse)

        return variants

    def validate_canonical_id(self, verse_id: Union[str, VerseID]) -> bool:
        """
        Validate that a verse ID is a valid canonical reference.

        Args:
            verse_id: Verse ID to validate

        Returns:
            True if valid canonical reference
        """
        if isinstance(verse_id, str):
            parsed_verse = parse_verse_id(verse_id)
        else:
            parsed_verse = verse_id

        if not parsed_verse:
            return False

        # Check that book code is valid
        try:
            book_code = BookCode(parsed_verse.book)
        except ValueError:
            return False

        # Check that chapter/verse numbers are reasonable
        book_info = get_book_info(book_code)
        if not book_info:
            return False

        # Validate chapter count
        if hasattr(book_info, "chapters"):
            chapter_count = book_info.chapters
        elif isinstance(book_info, dict):
            chapter_count = book_info.get("chapters", 50)  # Default fallback
        else:
            chapter_count = 50  # Default fallback

        if parsed_verse.chapter < 1 or parsed_verse.chapter > chapter_count:
            return False

        # Validate verse number (basic check - verse 1 should always exist)
        if parsed_verse.verse < 1:
            return False

        # More sophisticated verse count validation would require
        # chapter-specific data, which could be added later

        return True

    def get_mapping_statistics(
        self, translation_code: str
    ) -> Optional[Dict[str, Union[int, float]]]:
        """
        Get mapping statistics for a translation.

        Args:
            translation_code: Translation to get statistics for

        Returns:
            Dictionary with mapping statistics or None if translation not found
        """
        if translation_code not in self.version_mappings:
            return None

        mapping = self.version_mappings[translation_code]
        stats = mapping.coverage_stats.copy()

        # Calculate percentages
        total = stats.get("total_refs", 0)
        if total > 0:
            stats["mapping_rate"] = stats.get("mapped_refs", 0) / total
            stats["failure_rate"] = stats.get("failed_refs", 0) / total
        else:
            stats["mapping_rate"] = 0.0
            stats["failure_rate"] = 0.0

        return stats

    def get_supported_translations(self) -> List[str]:
        """
        Get list of supported translation codes.

        Returns:
            List of translation codes with mapping support
        """
        return list(self.version_mappings.keys())

    def bulk_generate_canonical_ids(
        self,
        references: List[str],
        source_system: Optional[VersificationSystem] = None,
        translation_code: Optional[str] = None,
    ) -> Dict[str, Optional[VerseID]]:
        """
        Generate canonical IDs for multiple references efficiently.

        Args:
            references: List of source references
            source_system: Source versification system
            translation_code: Translation code

        Returns:
            Dictionary mapping source references to canonical IDs
        """
        results = {}

        for ref in references:
            canonical_id = self.generate_canonical_id(ref, source_system, translation_code)
            results[ref] = canonical_id

        return results

    def find_cross_system_conflicts(self) -> List[Tuple[str, str, List[VerseMapping]]]:
        """
        Find verses that have conflicting mappings across versification systems.

        Returns:
            List of (system1, system2, conflicting_mappings) tuples
        """
        conflicts = []

        # Check all system pairs
        systems = self.verse_mapper.get_supported_systems()
        for sys1 in systems:
            for sys2 in systems:
                if sys1 != sys2:
                    conflicting_mappings = self.verse_mapper.find_conflicting_mappings(sys1, sys2)
                    if conflicting_mappings:
                        conflicts.append((sys1.value, sys2.value, conflicting_mappings))

        return conflicts
