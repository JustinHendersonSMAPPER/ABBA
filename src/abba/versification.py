"""
Versification mapping system for the ABBA project.

This module handles the complex mapping between different versification systems
used by various Bible translations. It provides utilities for converting verse
references between systems and handling edge cases like split/combined verses.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .verse_id import VerseID, parse_verse_id


class VersificationSystem(Enum):
    """Known versification systems."""

    MT = "MT"  # Masoretic Text (Hebrew Bible)
    LXX = "LXX"  # Septuagint (Greek OT)
    VULGATE = "Vulgate"  # Latin Vulgate
    KJV = "KJV"  # King James Version tradition
    MODERN = "Modern"  # Modern critical editions
    ORTHODOX = "Orthodox"  # Orthodox tradition
    CATHOLIC = "Catholic"  # Catholic tradition


@dataclass
class VersificationDifference:
    """Represents a difference between versification systems."""

    book: str
    chapter: int
    system1: VersificationSystem
    system2: VersificationSystem
    difference_type: str  # 'split', 'merge', 'offset', 'missing', 'added'
    description: str
    mapping: Dict[str, Any]  # Flexible mapping structure


# Known versification differences
# This is a subset of actual differences for demonstration
VERSIFICATION_DIFFERENCES: List[VersificationDifference] = [
    # Psalm titles counted as verse 1 in Hebrew
    VersificationDifference(
        book="PSA",
        chapter=3,
        system1=VersificationSystem.MT,
        system2=VersificationSystem.MODERN,
        difference_type="offset",
        description="Hebrew includes psalm title as verse 1",
        mapping={"offset": 1, "start_verse": 1},
    ),
    # 3 John has no verse numbers in some Greek manuscripts
    VersificationDifference(
        book="3JN",
        chapter=1,
        system1=VersificationSystem.KJV,
        system2=VersificationSystem.MODERN,
        difference_type="split",
        description="KJV has 14 verses, modern editions have 15",
        mapping={"14": ["14", "15"]},
    ),
    # Malachi 4 in Christian Bibles = Malachi 3:19-24 in Hebrew
    VersificationDifference(
        book="MAL",
        chapter=4,
        system1=VersificationSystem.KJV,
        system2=VersificationSystem.MT,
        difference_type="offset",
        description="Hebrew combines chapter 4 into chapter 3",
        mapping={"4:1-6": "3:19-24"},
    ),
    # Romans 16:25-27 placement varies
    VersificationDifference(
        book="ROM",
        chapter=16,
        system1=VersificationSystem.KJV,
        system2=VersificationSystem.MODERN,
        difference_type="reorder",
        description="Doxology placement varies in manuscripts",
        mapping={"16:25-27": ["14:24-26", "16:25-27"]},
    ),
]


# Translation to versification system mapping
TRANSLATION_SYSTEMS: Dict[str, VersificationSystem] = {
    # English translations
    "KJV": VersificationSystem.KJV,
    "NKJV": VersificationSystem.KJV,
    "ESV": VersificationSystem.MODERN,
    "NIV": VersificationSystem.MODERN,
    "NASB": VersificationSystem.MODERN,
    "RSV": VersificationSystem.MODERN,
    "NRSV": VersificationSystem.MODERN,
    # Other language translations
    "BHS": VersificationSystem.MT,  # Biblia Hebraica Stuttgartensia
    "LXX": VersificationSystem.LXX,
    "VUL": VersificationSystem.VULGATE,
    # Default for unknown
    "DEFAULT": VersificationSystem.MODERN,
}


class VersificationMapper:
    """Handles mapping between different versification systems."""

    def __init__(self) -> None:
        self.differences = self._build_difference_index()

    def _build_difference_index(self) -> Dict[Tuple[str, int], List[VersificationDifference]]:
        """Build an index of differences by book and chapter."""
        index: Dict[Tuple[str, int], List[VersificationDifference]] = {}
        for diff in VERSIFICATION_DIFFERENCES:
            key = (diff.book, diff.chapter)
            if key not in index:
                index[key] = []
            index[key].append(diff)
        return index

    def get_system_for_translation(self, translation_code: str) -> VersificationSystem:
        """Get the versification system used by a translation."""
        return TRANSLATION_SYSTEMS.get(translation_code.upper(), VersificationSystem.MODERN)

    def map_verse(
        self,
        verse_id: Union[str, VerseID],
        from_system: VersificationSystem,
        to_system: VersificationSystem,
    ) -> List[VerseID]:
        """
        Map a verse from one versification system to another.

        Args:
            verse_id: Source verse ID
            from_system: Source versification system
            to_system: Target versification system

        Returns:
            List of mapped verse IDs (may be multiple for split verses)
        """
        # Parse verse ID if string
        parsed_verse_id: Optional[VerseID]
        if isinstance(verse_id, str):
            parsed_verse_id = parse_verse_id(verse_id)
            if not parsed_verse_id:
                return []
        else:
            parsed_verse_id = verse_id

        if from_system == to_system:
            return [parsed_verse_id]

        # Check for known differences
        key = (parsed_verse_id.book, parsed_verse_id.chapter)
        differences = self.differences.get(key, [])

        for diff in differences:
            if self._applies_to_systems(diff, from_system, to_system):
                return self._apply_mapping(parsed_verse_id, diff, from_system, to_system)

        # No mapping needed
        return [parsed_verse_id]

    def _applies_to_systems(
        self,
        diff: VersificationDifference,
        from_system: VersificationSystem,
        to_system: VersificationSystem,
    ) -> bool:
        """Check if a difference applies to the given system pair."""
        return (diff.system1 == from_system and diff.system2 == to_system) or (
            diff.system1 == to_system and diff.system2 == from_system
        )

    def _apply_mapping(
        self,
        verse_id: VerseID,
        diff: VersificationDifference,
        from_system: VersificationSystem,
        to_system: VersificationSystem,
    ) -> List[VerseID]:
        """Apply a versification difference mapping."""
        # Handle offset differences (e.g., Psalm titles)
        if diff.difference_type == "offset":
            offset = int(diff.mapping.get("offset", 0))
            start_verse = int(diff.mapping.get("start_verse", 1))

            if verse_id.verse >= start_verse:
                # Apply offset
                if diff.system1 == from_system:
                    new_verse = verse_id.verse - offset
                else:
                    new_verse = verse_id.verse + offset

                if new_verse >= 1:
                    return [
                        VerseID(
                            book=verse_id.book,
                            chapter=verse_id.chapter,
                            verse=new_verse,
                            part=verse_id.part,
                        )
                    ]
                else:
                    # Invalid verse number after offset
                    return []

        # Handle other mapping types
        # This is simplified - real implementation would be more complex
        return [verse_id]

    def get_split_verses(
        self, book: str, chapter: int, system: VersificationSystem
    ) -> List[Tuple[int, List[str]]]:
        """
        Get verses that are split in the given system.

        Returns:
            List of (verse_number, [parts]) tuples
        """
        split_verses = []
        key = (book, chapter)

        for diff in self.differences.get(key, []):
            if diff.difference_type == "split" and diff.system1 == system:
                # Extract split information from mapping
                for verse_str, parts in diff.mapping.items():
                    if isinstance(parts, list) and len(parts) > 1:
                        verse_num = int(verse_str.split(":")[-1])
                        split_verses.append((verse_num, ["a", "b", "c"][: len(parts)]))

        return split_verses


class VersificationRules:
    """
    Documentation of versification mapping rules.

    This class provides detailed documentation of how different versification
    systems handle various edge cases and differences.
    """

    @staticmethod
    def get_psalm_title_rules() -> Dict[str, str]:
        """Rules for handling Psalm titles/superscriptions."""
        return {
            "MT": "Psalm titles are counted as verse 1",
            "LXX": "Psalm titles are counted as verse 1",
            "Modern": "Psalm titles are unnumbered (verse 1 begins with first content)",
            "Mapping": "MT/LXX verse n = Modern verse (n-1) for Psalms with titles",
        }

    @staticmethod
    def get_chapter_division_rules() -> Dict[str, List[Dict[str, Any]]]:
        """Rules for books with different chapter divisions."""
        return {
            "Malachi": [
                {
                    "MT": "3 chapters (3:1-24)",
                    "Christian": "4 chapters (3:1-18, 4:1-6)",
                    "Mapping": "MT 3:19-24 = Christian 4:1-6",
                }
            ],
            "Joel": [
                {
                    "MT": "4 chapters",
                    "Christian": "3 chapters",
                    "Mapping": "MT 3:1-5 = Christian 2:28-32; MT 4:1-21 = Christian 3:1-21",
                }
            ],
        }

    @staticmethod
    def get_verse_order_variants() -> List[Dict[str, Any]]:
        """Documentation of verses that appear in different orders."""
        return [
            {
                "reference": "ROM.16.25-27",
                "issue": "Doxology placement",
                "variants": [
                    "After 14:23 (some manuscripts)",
                    "After 15:33 (p46)",
                    "After 16:23 (most manuscripts)",
                    "Omitted (some Western texts)",
                ],
            },
            {
                "reference": "JHN.7.53-8.11",
                "issue": "Pericope Adulterae placement",
                "variants": [
                    "After John 7:52 (traditional)",
                    "After John 21:25 (family 1)",
                    "After Luke 21:38 (family 13)",
                    "Omitted (earliest manuscripts)",
                ],
            },
        ]

    @staticmethod
    def get_split_verse_rules() -> List[Dict[str, Any]]:
        """Rules for verses that are split differently across translations."""
        return [
            {
                "type": "Hebrew poetry",
                "rule": "Parallel lines may be split into separate verses",
                "examples": ["Many Psalms", "Proverbs"],
            },
            {
                "type": "Long verses",
                "rule": "Verses exceeding typical length may be subdivided",
                "examples": ["EST.8.9 (longest verse)"],
            },
            {
                "type": "Manuscript variants",
                "rule": "Different manuscript traditions may divide verses differently",
                "examples": ["3JN.1.14-15"],
            },
        ]

    @staticmethod
    def get_canonical_rules() -> Dict[str, str]:
        """Rules for establishing canonical verse IDs."""
        return {
            "Base System": "Protestant versification (66 books)",
            "Verse Parts": "Use lowercase letters (a, b, c) for split verses",
            "Missing Verses": "Maintain number sequence with empty content marker",
            "Added Verses": "Use previous verse number with part suffix",
            "Chapter Boundaries": "Follow traditional Protestant chapter divisions",
            "Book Order": "Follow Protestant canonical order",
            "Apocrypha": "Assign codes after REV (e.g., TOB, JDT, WIS)",
        }


def get_versification_documentation() -> str:
    """Generate comprehensive versification documentation."""
    doc = """
# ABBA Versification Mapping Rules

## Overview
The ABBA project uses a canonical versification system based on the modern
Protestant tradition, with mappings to handle variations in other traditions.

## Canonical Verse ID Format
- Format: `BOOK.CHAPTER.VERSE[part]`
- Example: `GEN.1.1`, `PSA.119.176`, `ROM.3.23a`
- Book codes: 3-letter standardized codes
- Verse parts: Lowercase letters for split verses

## Major Versification Systems

### Masoretic Text (MT)
- Used by: Hebrew Bible, Jewish translations
- Key differences:
  - Psalm superscriptions counted as verse 1
  - Different chapter divisions in some books
  - No New Testament

### Septuagint (LXX)
- Used by: Greek Orthodox tradition
- Key differences:
  - Includes deuterocanonical books
  - Some different verse divisions
  - Different book order

### Vulgate
- Used by: Catholic tradition
- Key differences:
  - Includes deuterocanonical books
  - Some unique verse divisions
  - Influenced many Western translations

### King James Version (KJV)
- Used by: KJV, NKJV, and derivatives
- Key differences:
  - Includes verses now considered spurious
  - Some unique verse divisions
  - Basis for much English tradition

### Modern Critical
- Used by: ESV, NIV, NRSV, etc.
- Key differences:
  - Omits some verses as non-original
  - Follows oldest manuscript evidence
  - More standardized across translations

## Handling Differences

### 1. Offset Differences
When verse numbering is consistently offset:
- Example: Psalms with titles
- Solution: Apply arithmetic offset

### 2. Split/Merge Differences
When verses are divided differently:
- Example: 3 John 14-15
- Solution: Use verse parts (a, b, c)

### 3. Missing/Added Verses
When verses exist in some traditions but not others:
- Example: Mark 16:9-20, John 7:53-8:11
- Solution: Include with confidence markers

### 4. Chapter Boundary Differences
When chapter divisions differ:
- Example: Malachi 3-4, Joel 2-3
- Solution: Map between systems explicitly

## Implementation Guidelines

1. **Always store in canonical format**
   - Use base Protestant versification
   - Apply mappings on input/output

2. **Preserve source information**
   - Track original reference
   - Note versification system used

3. **Handle edge cases gracefully**
   - Provide fallbacks for unknown mappings
   - Log unmapped references for review

4. **Support round-trip conversion**
   - Ensure mappings are reversible
   - Preserve original intent

## Common Pitfalls

1. **Assuming 1:1 mapping**
   - Some verses map to multiple verses
   - Some verses have no equivalent

2. **Ignoring manuscript variants**
   - Different texts have different content
   - Versification follows the text

3. **Hardcoding offsets**
   - Offsets vary by chapter and book
   - Use explicit mapping tables

4. **Forgetting verse parts**
   - Essential for split verses
   - Maintain reading order
"""
    return doc
