"""
Versification engine for mapping verses between different numbering systems.

Handles the complex task of converting verse references between different
versification schemes (e.g., Hebrew vs. Greek Psalm numbering).
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from .models import VerseMapping, MappingType, VersificationScheme
from ..verse_id import VerseID


@dataclass
class MappingResult:
    """Result of a verse mapping operation."""

    success: bool
    source_verses: List[VerseID]
    target_verses: List[VerseID]
    mapping_type: MappingType
    confidence: float = 1.0
    notes: Optional[str] = None

    @property
    def is_split(self) -> bool:
        """Check if this is a verse split (one-to-many)."""
        return self.mapping_type == MappingType.ONE_TO_MANY

    @property
    def is_merge(self) -> bool:
        """Check if this is a verse merge (many-to-one)."""
        return self.mapping_type == MappingType.MANY_TO_ONE

    @property
    def is_null(self) -> bool:
        """Check if verse doesn't exist in target."""
        return self.mapping_type == MappingType.NULL_MAPPING


class VersificationEngine:
    """Engine for handling verse mapping between versification schemes."""

    def __init__(self):
        """Initialize the versification engine."""
        self.logger = logging.getLogger(__name__)

        # Mapping storage: {(source_scheme, target_scheme): {source_ref: VerseMapping}}
        self._mappings: Dict[Tuple[str, str], Dict[str, VerseMapping]] = defaultdict(dict)

        # Reverse mappings for bidirectional lookup
        self._reverse_mappings: Dict[Tuple[str, str], Dict[str, VerseMapping]] = defaultdict(dict)

        # Scheme metadata
        self._schemes: Dict[str, VersificationScheme] = {}

        # Initialize default mappings
        self._initialize_default_mappings()

    def _initialize_default_mappings(self):
        """Initialize default verse mappings between common schemes."""
        # Psalm numbering differences (Hebrew vs. LXX/Vulgate)
        self._add_psalm_mappings()

        # 3 John verse differences
        self._add_3john_mappings()

        # Daniel additions
        self._add_daniel_mappings()

        # Malachi chapter differences
        self._add_malachi_mappings()

        # Other common differences
        self._add_other_mappings()

    def _add_psalm_mappings(self):
        """Add Psalm numbering mappings between Hebrew and Greek."""
        # Psalms 9-10 in Hebrew = Psalm 9 in LXX
        # This causes all subsequent psalms to be off by one

        # Hebrew Psalm 9 = LXX Psalm 9:1-21
        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.ONE_TO_ONE,
                source_book="PSA",
                source_chapter=9,
                source_verses=list(range(1, 21)),  # All verses
                target_book="PSA",
                target_chapter=9,
                target_verses=list(range(1, 21)),
                notes="Hebrew Psalm 9 = first part of LXX Psalm 9",
            )
        )

        # Hebrew Psalm 10 = LXX Psalm 9:22-39
        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.MANY_TO_ONE,
                source_book="PSA",
                source_chapter=10,
                source_verses=list(range(1, 19)),
                target_book="PSA",
                target_chapter=9,
                target_verses=list(range(22, 40)),
                notes="Hebrew Psalm 10 = second part of LXX Psalm 9",
            )
        )

        # Psalms 11-113 are off by one
        for psalm in range(11, 114):
            self.add_mapping(
                VerseMapping(
                    source_scheme_id="masoretic",
                    target_scheme_id="septuagint",
                    mapping_type=MappingType.ONE_TO_ONE,
                    source_book="PSA",
                    source_chapter=psalm,
                    source_verses=[0],  # All verses indicator
                    target_book="PSA",
                    target_chapter=psalm - 1,
                    target_verses=[0],
                    notes=f"Hebrew Psalm {psalm} = LXX Psalm {psalm - 1}",
                )
            )

        # Psalms 114-115 in Hebrew = Psalm 113 in LXX
        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.MANY_TO_ONE,
                source_book="PSA",
                source_chapter=114,
                source_verses=[0],
                target_book="PSA",
                target_chapter=113,
                target_verses=list(range(1, 9)),
                notes="Hebrew Psalm 114 = first part of LXX Psalm 113",
            )
        )

        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.MANY_TO_ONE,
                source_book="PSA",
                source_chapter=115,
                source_verses=[0],
                target_book="PSA",
                target_chapter=113,
                target_verses=list(range(10, 27)),
                notes="Hebrew Psalm 115 = second part of LXX Psalm 113",
            )
        )

        # Psalm 116 in Hebrew = Psalms 114-115 in LXX
        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.ONE_TO_MANY,
                source_book="PSA",
                source_chapter=116,
                source_verses=list(range(1, 10)),
                target_book="PSA",
                target_chapter=114,
                target_verses=[0],
                notes="Hebrew Psalm 116:1-9 = LXX Psalm 114",
            )
        )

        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.ONE_TO_MANY,
                source_book="PSA",
                source_chapter=116,
                source_verses=list(range(10, 20)),
                target_book="PSA",
                target_chapter=115,
                target_verses=[0],
                notes="Hebrew Psalm 116:10-19 = LXX Psalm 115",
            )
        )

        # Psalms 117-146 are off by one
        for psalm in range(117, 147):
            self.add_mapping(
                VerseMapping(
                    source_scheme_id="masoretic",
                    target_scheme_id="septuagint",
                    mapping_type=MappingType.ONE_TO_ONE,
                    source_book="PSA",
                    source_chapter=psalm,
                    source_verses=[0],
                    target_book="PSA",
                    target_chapter=psalm - 1,
                    target_verses=[0],
                    notes=f"Hebrew Psalm {psalm} = LXX Psalm {psalm - 1}",
                )
            )

        # Psalm 147 in Hebrew = Psalms 146-147 in LXX
        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.ONE_TO_MANY,
                source_book="PSA",
                source_chapter=147,
                source_verses=list(range(1, 12)),
                target_book="PSA",
                target_chapter=146,
                target_verses=[0],
                notes="Hebrew Psalm 147:1-11 = LXX Psalm 146",
            )
        )

        self.add_mapping(
            VerseMapping(
                source_scheme_id="masoretic",
                target_scheme_id="septuagint",
                mapping_type=MappingType.ONE_TO_MANY,
                source_book="PSA",
                source_chapter=147,
                source_verses=list(range(12, 21)),
                target_book="PSA",
                target_chapter=147,
                target_verses=[0],
                notes="Hebrew Psalm 147:12-20 = LXX Psalm 147",
            )
        )

        # Psalms 148-150 are the same
        for psalm in range(148, 151):
            self.add_mapping(
                VerseMapping(
                    source_scheme_id="masoretic",
                    target_scheme_id="septuagint",
                    mapping_type=MappingType.ONE_TO_ONE,
                    source_book="PSA",
                    source_chapter=psalm,
                    source_verses=[0],
                    target_book="PSA",
                    target_chapter=psalm,
                    target_verses=[0],
                    notes=f"Psalm {psalm} is the same in both",
                )
            )

    def _add_3john_mappings(self):
        """Add 3 John verse division mappings."""
        # Some traditions have 3 John 1:14-15, others end at 1:14
        self.add_mapping(
            VerseMapping(
                source_scheme_id="standard",
                target_scheme_id="vulgate",
                mapping_type=MappingType.ONE_TO_MANY,
                source_book="3JN",
                source_chapter=1,
                source_verses=[14],
                target_book="3JN",
                target_chapter=1,
                target_verses=[14, 15],
                notes="3 John 1:14 in standard = 1:14-15 in Vulgate",
            )
        )

    def _add_daniel_mappings(self):
        """Add Daniel Greek additions mappings."""
        # Daniel 3:24-90 (Song of the Three Young Men) exists only in LXX
        self.add_mapping(
            VerseMapping(
                source_scheme_id="septuagint",
                target_scheme_id="masoretic",
                mapping_type=MappingType.NULL_MAPPING,
                source_book="DAN",
                source_chapter=3,
                source_verses=list(range(24, 91)),
                target_book="DAN",
                target_chapter=3,
                target_verses=[],
                notes="Song of the Three Young Men - not in Hebrew",
            )
        )

        # Daniel 13 (Susanna) - only in Greek
        self.add_mapping(
            VerseMapping(
                source_scheme_id="septuagint",
                target_scheme_id="masoretic",
                mapping_type=MappingType.NULL_MAPPING,
                source_book="DAN",
                source_chapter=13,
                source_verses=[0],  # Entire chapter
                target_book="DAN",
                target_chapter=0,
                target_verses=[],
                notes="Susanna - not in Hebrew Daniel",
            )
        )

        # Daniel 14 (Bel and the Dragon) - only in Greek
        self.add_mapping(
            VerseMapping(
                source_scheme_id="septuagint",
                target_scheme_id="masoretic",
                mapping_type=MappingType.NULL_MAPPING,
                source_book="DAN",
                source_chapter=14,
                source_verses=[0],  # Entire chapter
                target_book="DAN",
                target_chapter=0,
                target_verses=[],
                notes="Bel and the Dragon - not in Hebrew Daniel",
            )
        )

    def _add_malachi_mappings(self):
        """Add Malachi chapter division mappings."""
        # Malachi 3:19-24 in Hebrew = 4:1-6 in Greek/English
        for verse in range(19, 25):
            self.add_mapping(
                VerseMapping(
                    source_scheme_id="masoretic",
                    target_scheme_id="standard",
                    mapping_type=MappingType.ONE_TO_ONE,
                    source_book="MAL",
                    source_chapter=3,
                    source_verses=[verse],
                    target_book="MAL",
                    target_chapter=4,
                    target_verses=[verse - 18],
                    notes=f"Hebrew Mal 3:{verse} = English Mal 4:{verse-18}",
                )
            )

    def _add_other_mappings(self):
        """Add other common verse mappings."""
        # Add more mappings as needed
        pass

    def add_mapping(self, mapping: VerseMapping) -> None:
        """Add a verse mapping to the engine."""
        # Create mapping key
        scheme_key = (mapping.source_scheme_id, mapping.target_scheme_id)

        # Store forward mapping
        for source_verse in mapping.source_verses:
            source_ref = f"{mapping.source_book}.{mapping.source_chapter}.{source_verse}"
            self._mappings[scheme_key][source_ref] = mapping

        # Store reverse mapping if not null
        if mapping.mapping_type != MappingType.NULL_MAPPING:
            reverse_key = (mapping.target_scheme_id, mapping.source_scheme_id)
            for target_verse in mapping.target_verses:
                target_ref = f"{mapping.target_book}.{mapping.target_chapter}.{target_verse}"
                self._reverse_mappings[reverse_key][target_ref] = mapping

    def map_verse(self, verse_id: VerseID, source_scheme: str, target_scheme: str) -> MappingResult:
        """Map a verse from source to target versification scheme."""
        if source_scheme == target_scheme:
            # No mapping needed
            return MappingResult(
                success=True,
                source_verses=[verse_id],
                target_verses=[verse_id],
                mapping_type=MappingType.ONE_TO_ONE,
                confidence=1.0,
            )

        # Look for direct mapping
        scheme_key = (source_scheme, target_scheme)
        verse_ref = str(verse_id)

        # Check if we have a specific verse mapping
        if verse_ref in self._mappings[scheme_key]:
            mapping = self._mappings[scheme_key][verse_ref]
            return self._apply_mapping(verse_id, mapping)

        # Check reverse mappings
        if verse_ref in self._reverse_mappings[scheme_key]:
            mapping = self._reverse_mappings[scheme_key][verse_ref]
            # For reverse mapping, we need to invert the result
            return self._apply_reverse_mapping(verse_id, mapping)

        # Check for chapter-level mapping (verse 0 indicates all verses)
        chapter_ref = f"{verse_id.book}.{verse_id.chapter}.0"
        if chapter_ref in self._mappings[scheme_key]:
            mapping = self._mappings[scheme_key][chapter_ref]
            return self._apply_mapping(verse_id, mapping)

        # Check chapter-level reverse mapping
        if chapter_ref in self._reverse_mappings[scheme_key]:
            mapping = self._reverse_mappings[scheme_key][chapter_ref]
            return self._apply_reverse_mapping(verse_id, mapping)

        # No mapping found - assume direct correspondence
        return MappingResult(
            success=True,
            source_verses=[verse_id],
            target_verses=[verse_id],
            mapping_type=MappingType.ONE_TO_ONE,
            confidence=0.8,  # Lower confidence for assumed mapping
            notes="No specific mapping found, assuming direct correspondence",
        )

    def _apply_mapping(self, source_verse: VerseID, mapping: VerseMapping) -> MappingResult:
        """Apply a mapping to create result."""
        # Handle null mapping
        if mapping.mapping_type == MappingType.NULL_MAPPING:
            return MappingResult(
                success=True,
                source_verses=[source_verse],
                target_verses=[],
                mapping_type=MappingType.NULL_MAPPING,
                confidence=mapping.confidence,
                notes=mapping.notes,
            )

        # Create target verses
        target_verses = []

        if mapping.target_verses == [0]:  # Chapter-level mapping
            # Map to same verse number in target chapter
            target_verses.append(
                VerseID(mapping.target_book, mapping.target_chapter, source_verse.verse)
            )
        else:
            # Specific verse mapping
            for target_verse_num in mapping.target_verses:
                target_verses.append(
                    VerseID(mapping.target_book, mapping.target_chapter, target_verse_num)
                )

        return MappingResult(
            success=True,
            source_verses=[source_verse],
            target_verses=target_verses,
            mapping_type=mapping.mapping_type,
            confidence=mapping.confidence,
            notes=mapping.notes,
        )

    def _apply_reverse_mapping(self, target_verse: VerseID, mapping: VerseMapping) -> MappingResult:
        """Apply a reverse mapping (invert source and target)."""
        # For reverse mapping, target becomes source
        if mapping.mapping_type == MappingType.NULL_MAPPING:
            # Can't reverse a null mapping
            return MappingResult(
                success=False,
                source_verses=[target_verse],
                target_verses=[],
                mapping_type=MappingType.NULL_MAPPING,
                confidence=0.0,
                notes="Cannot reverse a null mapping",
            )

        # Invert the mapping type
        inverted_type = mapping.mapping_type
        if mapping.mapping_type == MappingType.ONE_TO_MANY:
            inverted_type = MappingType.MANY_TO_ONE
        elif mapping.mapping_type == MappingType.MANY_TO_ONE:
            inverted_type = MappingType.ONE_TO_MANY

        # Create source verses (from original mapping's source)
        source_verses = []
        if mapping.source_verses == [0]:  # Chapter-level mapping
            source_verses.append(
                VerseID(mapping.source_book, mapping.source_chapter, target_verse.verse)
            )
        else:
            # Find which target verse we're at
            target_idx = None
            for i, tv in enumerate(mapping.target_verses):
                if tv == target_verse.verse or (
                    tv == 0 and target_verse.chapter == mapping.target_chapter
                ):
                    target_idx = i
                    break

            if target_idx is not None and target_idx < len(mapping.source_verses):
                source_verses.append(
                    VerseID(
                        mapping.source_book,
                        mapping.source_chapter,
                        (
                            mapping.source_verses[target_idx]
                            if mapping.source_verses != [0]
                            else target_verse.verse
                        ),
                    )
                )
            else:
                # Default to same verse number
                source_verses.append(
                    VerseID(mapping.source_book, mapping.source_chapter, target_verse.verse)
                )

        return MappingResult(
            success=True,
            source_verses=[target_verse],
            target_verses=source_verses,
            mapping_type=inverted_type,
            confidence=mapping.confidence,
            notes=f"Reverse of: {mapping.notes}" if mapping.notes else "Reverse mapping",
        )

    def map_verse_range(
        self, start_verse: VerseID, end_verse: VerseID, source_scheme: str, target_scheme: str
    ) -> List[MappingResult]:
        """Map a range of verses between schemes."""
        results = []

        # Generate all verses in range
        current = start_verse
        while current <= end_verse:
            result = self.map_verse(current, source_scheme, target_scheme)
            results.append(result)

            # Move to next verse
            current = VerseID(current.book, current.chapter, current.verse + 1)

        return results

    def get_scheme_differences(
        self, scheme1: str, scheme2: str, book: Optional[str] = None
    ) -> List[VerseMapping]:
        """Get all mapping differences between two schemes."""
        differences = []

        # Forward mappings
        forward_key = (scheme1, scheme2)
        if forward_key in self._mappings:
            for mapping in self._mappings[forward_key].values():
                if not book or mapping.source_book == book:
                    if mapping not in differences:
                        differences.append(mapping)

        # Reverse mappings
        reverse_key = (scheme2, scheme1)
        if reverse_key in self._mappings:
            for mapping in self._mappings[reverse_key].values():
                if not book or mapping.source_book == book:
                    if mapping not in differences:
                        differences.append(mapping)

        return differences

    def register_scheme(self, scheme: VersificationScheme) -> None:
        """Register a versification scheme."""
        self._schemes[scheme.id] = scheme
        self.logger.info(f"Registered versification scheme: {scheme.name}")

    def get_scheme(self, scheme_id: str) -> Optional[VersificationScheme]:
        """Get a versification scheme by ID."""
        return self._schemes.get(scheme_id)

    def can_map_between(self, source_scheme: str, target_scheme: str) -> bool:
        """Check if mapping is possible between two schemes."""
        if source_scheme == target_scheme:
            return True

        # Check if we have any mappings between the schemes
        forward_key = (source_scheme, target_scheme)
        reverse_key = (target_scheme, source_scheme)

        return (forward_key in self._mappings and len(self._mappings[forward_key]) > 0) or (
            reverse_key in self._mappings and len(self._mappings[reverse_key]) > 0
        )
