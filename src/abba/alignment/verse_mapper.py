"""
Enhanced verse mapping system with comprehensive versification support.

This module provides sophisticated algorithms for mapping verses between
different versification systems, handling complex cases like verse splits,
merges, missing verses, and chapter boundary differences.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from ..verse_id import VerseID, VerseRange, parse_verse_id
from ..versification import VersificationSystem


class MappingType(Enum):
    """Types of verse mappings between versification systems."""

    EXACT = "exact"  # 1:1 mapping
    SPLIT = "split"  # 1:many mapping (verse split)
    MERGE = "merge"  # many:1 mapping (verses merged)
    SHIFT = "shift"  # verse number shifted
    MISSING = "missing"  # verse exists in source but not target
    ADDED = "added"  # verse exists in target but not source
    REORDERED = "reordered"  # verses appear in different order


class MappingConfidence(Enum):
    """Confidence levels for verse mappings."""

    CERTAIN = 1.0  # Universally agreed mapping
    HIGH = 0.9  # Strong manuscript/scholarly consensus
    MEDIUM = 0.7  # Some scholarly disagreement
    LOW = 0.5  # Significant uncertainty
    DISPUTED = 0.3  # Major disagreement among traditions


@dataclass
class VerseMapping:
    """Represents a mapping between verses in different versification systems."""

    source_verse: VerseID
    target_verses: List[VerseID]
    mapping_type: MappingType
    confidence: MappingConfidence
    notes: Optional[str] = None

    def is_one_to_one(self) -> bool:
        """Check if this is a simple 1:1 mapping."""
        return len(self.target_verses) == 1 and self.mapping_type == MappingType.EXACT

    def is_complex(self) -> bool:
        """Check if this mapping requires special handling."""
        return self.mapping_type in [
            MappingType.SPLIT,
            MappingType.MERGE,
            MappingType.REORDERED,
            MappingType.MISSING,
        ]


class EnhancedVerseMapper:
    """Advanced verse mapping with comprehensive versification support."""

    def __init__(self) -> None:
        """Initialize the enhanced verse mapper."""
        self._mapping_cache: Dict[
            Tuple[VersificationSystem, VersificationSystem], Dict[str, VerseMapping]
        ] = {}
        self._load_core_mappings()

    def _load_core_mappings(self) -> None:
        """Load essential verse mappings for common versification differences."""
        # Psalm title mappings (Hebrew MT vs Modern numbering)
        self._create_psalm_mappings()

        # Malachi chapter division differences
        self._create_malachi_mappings()

        # Joel chapter differences (Hebrew 3 chapters vs English 4)
        self._create_joel_mappings()

        # Common textual variant mappings
        self._create_textual_variant_mappings()

    def _create_psalm_mappings(self) -> None:
        """Create mappings for Psalm numbering differences."""
        mt_to_modern = {}

        # Psalms 10-113: MT has titles, Modern often doesn't (causes +1 offset)
        # For simplicity, documenting key differences

        # Psalm 3: MT has title, causes offset
        mt_to_modern["PSA.3.1"] = VerseMapping(
            source_verse=parse_verse_id("PSA.3.1"),
            target_verses=[parse_verse_id("PSA.3.1")],  # Title becomes verse 1
            mapping_type=MappingType.SHIFT,
            confidence=MappingConfidence.CERTAIN,
            notes="Hebrew title becomes verse 1 in many modern translations",
        )

        # Store in cache
        cache_key = (VersificationSystem.MT, VersificationSystem.MODERN)
        if cache_key not in self._mapping_cache:
            self._mapping_cache[cache_key] = {}
        self._mapping_cache[cache_key].update(mt_to_modern)

    def _create_malachi_mappings(self) -> None:
        """Create mappings for Malachi chapter differences."""
        mt_to_modern = {}

        # Hebrew Malachi 3:19-24 becomes English Malachi 4:1-6
        for verse in range(19, 25):  # 19-24
            english_verse = verse - 18  # Maps to 1-6
            mt_verse_id = f"MAL.3.{verse}"
            modern_verse_id = f"MAL.4.{english_verse}"

            mt_to_modern[mt_verse_id] = VerseMapping(
                source_verse=parse_verse_id(mt_verse_id),
                target_verses=[parse_verse_id(modern_verse_id)],
                mapping_type=MappingType.SHIFT,
                confidence=MappingConfidence.CERTAIN,
                notes="Hebrew chapter 3 verses 19-24 become chapter 4 verses 1-6",
            )

        # Store in cache
        cache_key = (VersificationSystem.MT, VersificationSystem.MODERN)
        if cache_key not in self._mapping_cache:
            self._mapping_cache[cache_key] = {}
        self._mapping_cache[cache_key].update(mt_to_modern)

    def _create_joel_mappings(self) -> None:
        """Create mappings for Joel chapter differences."""
        hebrew_to_english = {}

        # Hebrew Joel 3:1-5 becomes English Joel 2:28-32
        for verse in range(1, 6):  # Hebrew 3:1-5
            english_verse = verse + 27  # Maps to 2:28-32
            heb_verse_id = f"JOL.3.{verse}"
            eng_verse_id = f"JOL.2.{english_verse}"

            hebrew_to_english[heb_verse_id] = VerseMapping(
                source_verse=parse_verse_id(heb_verse_id),
                target_verses=[parse_verse_id(eng_verse_id)],
                mapping_type=MappingType.SHIFT,
                confidence=MappingConfidence.CERTAIN,
                notes="Hebrew chapter 3 becomes part of English chapter 2",
            )

        # Hebrew Joel 4 becomes English Joel 3
        # This is a systematic offset for the entire chapter

        # Store in cache
        cache_key = (VersificationSystem.MT, VersificationSystem.MODERN)
        if cache_key not in self._mapping_cache:
            self._mapping_cache[cache_key] = {}
        self._mapping_cache[cache_key].update(hebrew_to_english)

    def _create_textual_variant_mappings(self) -> None:
        """Create mappings for common textual variants."""
        kjv_to_modern = {}

        # Acts 8:37 - present in KJV but missing in modern critical texts
        kjv_to_modern["ACT.8.37"] = VerseMapping(
            source_verse=parse_verse_id("ACT.8.37"),
            target_verses=[],  # No target - verse missing in modern texts
            mapping_type=MappingType.MISSING,
            confidence=MappingConfidence.HIGH,
            notes="Verse present in Textus Receptus/KJV but absent in critical texts",
        )

        # 1 John 5:7 (Comma Johanneum) - heavily disputed
        kjv_to_modern["1JN.5.7"] = VerseMapping(
            source_verse=parse_verse_id("1JN.5.7"),
            target_verses=[parse_verse_id("1JN.5.7")],  # Verse exists but content differs
            mapping_type=MappingType.EXACT,
            confidence=MappingConfidence.DISPUTED,
            notes="Content significantly different due to Comma Johanneum textual variant",
        )

        # Store in cache
        cache_key = (VersificationSystem.KJV, VersificationSystem.MODERN)
        if cache_key not in self._mapping_cache:
            self._mapping_cache[cache_key] = {}
        self._mapping_cache[cache_key].update(kjv_to_modern)

    def map_verse(
        self,
        verse_id: Union[str, VerseID],
        from_system: VersificationSystem,
        to_system: VersificationSystem,
    ) -> VerseMapping:
        """
        Map a single verse between versification systems.

        Args:
            verse_id: Source verse ID to map
            from_system: Source versification system
            to_system: Target versification system

        Returns:
            VerseMapping with target verse(s) and metadata
        """
        if isinstance(verse_id, str):
            source_verse = parse_verse_id(verse_id)
        else:
            source_verse = verse_id

        if not source_verse:
            raise ValueError(f"Invalid verse ID: {verse_id}")

        # Check if systems are the same
        if from_system == to_system:
            return VerseMapping(
                source_verse=source_verse,
                target_verses=[source_verse],
                mapping_type=MappingType.EXACT,
                confidence=MappingConfidence.CERTAIN,
            )

        # Look up mapping in cache
        cache_key = (from_system, to_system)
        verse_key = str(source_verse)

        if cache_key in self._mapping_cache and verse_key in self._mapping_cache[cache_key]:
            return self._mapping_cache[cache_key][verse_key]

        # Default mapping (assume 1:1 for unmapped verses)
        return VerseMapping(
            source_verse=source_verse,
            target_verses=[source_verse],
            mapping_type=MappingType.EXACT,
            confidence=MappingConfidence.HIGH,
            notes="Default 1:1 mapping - no specific versification differences documented",
        )

    def map_verse_range(
        self,
        verse_range: Union[str, VerseRange],
        from_system: VersificationSystem,
        to_system: VersificationSystem,
    ) -> List[VerseMapping]:
        """
        Map a range of verses between versification systems.

        Args:
            verse_range: Range of verses to map
            from_system: Source versification system
            to_system: Target versification system

        Returns:
            List of VerseMapping objects for the range
        """
        if isinstance(verse_range, str):
            # Parse the range string
            if "-" in verse_range:
                start_str, end_str = verse_range.split("-", 1)
                start_verse = parse_verse_id(start_str.strip())
                end_verse = parse_verse_id(end_str.strip())

                if start_verse and end_verse:
                    range_obj = VerseRange(start_verse, end_verse)
                else:
                    raise ValueError(f"Invalid verse range: {verse_range}")
            else:
                # Single verse
                single_verse = parse_verse_id(verse_range)
                if single_verse:
                    return [self.map_verse(single_verse, from_system, to_system)]
                else:
                    raise ValueError(f"Invalid verse range: {verse_range}")
        else:
            range_obj = verse_range

        # Map each verse in the range
        mappings = []
        for verse in range_obj.to_list():
            mapping = self.map_verse(verse, from_system, to_system)
            mappings.append(mapping)

        return mappings

    def get_mapping_confidence(self, mapping: VerseMapping) -> float:
        """
        Get the confidence score for a verse mapping.

        Args:
            mapping: The verse mapping to evaluate

        Returns:
            Confidence score between 0.0 and 1.0
        """
        return mapping.confidence.value

    def find_conflicting_mappings(
        self, from_system: VersificationSystem, to_system: VersificationSystem
    ) -> List[VerseMapping]:
        """
        Find mappings that might have conflicts or ambiguities.

        Args:
            from_system: Source versification system
            to_system: Target versification system

        Returns:
            List of potentially problematic mappings
        """
        cache_key = (from_system, to_system)
        if cache_key not in self._mapping_cache:
            return []

        conflicts = []
        for mapping in self._mapping_cache[cache_key].values():
            if mapping.confidence in [
                MappingConfidence.LOW,
                MappingConfidence.DISPUTED,
            ] or mapping.mapping_type in [
                MappingType.MERGE,
                MappingType.SPLIT,
                MappingType.REORDERED,
            ]:
                conflicts.append(mapping)

        return conflicts

    def get_supported_systems(self) -> Set[VersificationSystem]:
        """
        Get the set of versification systems with mapping support.

        Returns:
            Set of supported VersificationSystem values
        """
        systems = set()
        for from_sys, to_sys in self._mapping_cache.keys():
            systems.add(from_sys)
            systems.add(to_sys)
        return systems

    def validate_mapping_consistency(
        self, system1: VersificationSystem, system2: VersificationSystem
    ) -> bool:
        """
        Validate that mappings between two systems are consistent.

        Args:
            system1: First versification system
            system2: Second versification system

        Returns:
            True if mappings are consistent, False otherwise
        """
        forward_key = (system1, system2)
        reverse_key = (system2, system1)

        if forward_key not in self._mapping_cache or reverse_key not in self._mapping_cache:
            return True  # No mappings to validate

        forward_mappings = self._mapping_cache[forward_key]
        reverse_mappings = self._mapping_cache[reverse_key]

        # Check for round-trip consistency
        for verse_id, forward_mapping in forward_mappings.items():
            if len(forward_mapping.target_verses) == 1:  # Only check 1:1 mappings
                target_verse = str(forward_mapping.target_verses[0])
                if target_verse in reverse_mappings:
                    reverse_mapping = reverse_mappings[target_verse]
                    if (
                        len(reverse_mapping.target_verses) == 1
                        and str(reverse_mapping.target_verses[0]) != verse_id
                    ):
                        return False

        return True
