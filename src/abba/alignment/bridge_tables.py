"""
Versification bridge tables for comprehensive mapping between systems.

This module provides comprehensive mapping data and optimized lookup structures
for converting between different versification systems.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from ..verse_id import VerseID, parse_verse_id
from ..versification import VersificationSystem
from .verse_mapper import MappingType, MappingConfidence, VerseMapping


@dataclass
class MappingData:
    """Structured mapping data between versification systems."""

    from_system: VersificationSystem
    to_system: VersificationSystem
    mappings: Dict[str, List[str]]  # source_verse -> [target_verses]
    metadata: Dict[str, any]
    confidence_scores: Dict[str, float] = None
    mapping_types: Dict[str, str] = None

    def __post_init__(self) -> None:
        """Initialize optional fields."""
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.mapping_types is None:
            self.mapping_types = {}


class VersificationBridge:
    """Comprehensive mapping tables between versification systems."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the versification bridge.

        Args:
            data_dir: Directory containing versification mapping data files
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.mapping_data: Dict[Tuple[VersificationSystem, VersificationSystem], MappingData] = {}
        self.lookup_index: Dict[
            Tuple[VersificationSystem, VersificationSystem], Dict[str, List[VerseID]]
        ] = {}

        # Load built-in mapping data
        self._load_built_in_mappings()

        # Load external data files if directory provided
        if self.data_dir and self.data_dir.exists():
            self._load_external_mappings()

        # Build optimized lookup structures
        self._build_mapping_index()

    def _load_built_in_mappings(self) -> None:
        """Load built-in mapping data for common versification differences."""
        # Hebrew MT to Modern versification mappings
        self._load_mt_to_modern_mappings()

        # KJV to Modern mappings (textual variants)
        self._load_kjv_to_modern_mappings()

        # LXX differences (basic framework)
        self._load_lxx_mappings()

    def _load_mt_to_modern_mappings(self) -> None:
        """Load Hebrew Masoretic Text to Modern versification mappings."""
        mappings = {}
        confidence_scores = {}
        mapping_types = {}

        # Psalm title differences - comprehensive mapping
        psalm_mappings = self._generate_psalm_title_mappings()
        mappings.update(psalm_mappings["mappings"])
        confidence_scores.update(psalm_mappings["confidence"])
        mapping_types.update(psalm_mappings["types"])

        # Malachi chapter differences
        mal_mappings = self._generate_malachi_mappings()
        mappings.update(mal_mappings["mappings"])
        confidence_scores.update(mal_mappings["confidence"])
        mapping_types.update(mal_mappings["types"])

        # Joel chapter differences
        joel_mappings = self._generate_joel_mappings()
        mappings.update(joel_mappings["mappings"])
        confidence_scores.update(joel_mappings["confidence"])
        mapping_types.update(joel_mappings["types"])

        # Store the mapping data
        mapping_data = MappingData(
            from_system=VersificationSystem.MT,
            to_system=VersificationSystem.MODERN,
            mappings=mappings,
            confidence_scores=confidence_scores,
            mapping_types=mapping_types,
            metadata={
                "description": "Hebrew Masoretic Text to Modern versification mapping",
                "source": "Built-in ABBA mappings",
                "last_updated": "2024-01-01",
                "verse_count": len(mappings),
            },
        )

        key = (VersificationSystem.MT, VersificationSystem.MODERN)
        self.mapping_data[key] = mapping_data

    def _generate_psalm_title_mappings(self) -> Dict[str, Dict[str, any]]:
        """Generate comprehensive Psalm title mappings."""
        mappings = {}
        confidence = {}
        types = {}

        # Psalms with titles that affect verse numbering
        titled_psalms = [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            68,
            69,
            70,
            75,
            76,
            77,
            80,
            81,
            83,
            84,
            85,
            88,
            89,
            90,
            92,
            98,
            100,
            101,
            102,
            108,
            109,
            110,
            140,
            141,
            142,
            143,
        ]

        for psalm_num in titled_psalms:
            psalm_id = f"PSA.{psalm_num}"

            # For psalms with titles, Hebrew verse 1 (title) often becomes English superscription
            # Hebrew verse 2 becomes English verse 1, etc.

            # Get approximate verse count for this psalm (simplified)
            max_verses = self._get_approximate_psalm_verse_count(psalm_num)

            for verse_num in range(1, max_verses + 1):
                hebrew_verse = f"{psalm_id}.{verse_num}"

                if verse_num == 1:
                    # Hebrew verse 1 (title) often has no English equivalent or becomes superscription
                    mappings[hebrew_verse] = []  # No direct mapping
                    confidence[hebrew_verse] = 0.9
                    types[hebrew_verse] = MappingType.MISSING.value
                else:
                    # Hebrew verse N becomes English verse N-1
                    english_verse = f"{psalm_id}.{verse_num - 1}"
                    mappings[hebrew_verse] = [english_verse]
                    confidence[hebrew_verse] = 0.95
                    types[hebrew_verse] = MappingType.SHIFT.value

        return {"mappings": mappings, "confidence": confidence, "types": types}

    def _get_approximate_psalm_verse_count(self, psalm_num: int) -> int:
        """Get approximate verse count for a psalm (simplified lookup)."""
        # Simplified verse counts for major psalms
        verse_counts = {
            1: 6,
            2: 12,
            3: 8,
            4: 8,
            5: 12,
            6: 10,
            7: 17,
            8: 9,
            9: 20,
            10: 18,
            23: 6,
            51: 19,
            119: 176,
            150: 6,
        }
        return verse_counts.get(psalm_num, 20)  # Default to 20 verses

    def _generate_malachi_mappings(self) -> Dict[str, Dict[str, any]]:
        """Generate Malachi chapter boundary mappings."""
        mappings = {}
        confidence = {}
        types = {}

        # Hebrew Malachi 3:19-24 becomes English Malachi 4:1-6
        hebrew_verses = [(3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24)]
        english_verses = [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)]

        for (heb_ch, heb_v), (eng_ch, eng_v) in zip(hebrew_verses, english_verses):
            heb_ref = f"MAL.{heb_ch}.{heb_v}"
            eng_ref = f"MAL.{eng_ch}.{eng_v}"

            mappings[heb_ref] = [eng_ref]
            confidence[heb_ref] = 1.0  # Certain mapping
            types[heb_ref] = MappingType.SHIFT.value

        return {"mappings": mappings, "confidence": confidence, "types": types}

    def _generate_joel_mappings(self) -> Dict[str, Dict[str, any]]:
        """Generate Joel chapter boundary mappings."""
        mappings = {}
        confidence = {}
        types = {}

        # Hebrew Joel 3:1-5 becomes English Joel 2:28-32
        for verse in range(1, 6):  # Hebrew 3:1-5
            heb_ref = f"JOL.3.{verse}"
            eng_ref = f"JOL.2.{verse + 27}"  # Maps to 2:28-32

            mappings[heb_ref] = [eng_ref]
            confidence[heb_ref] = 1.0
            types[heb_ref] = MappingType.SHIFT.value

        # Hebrew Joel 4 becomes English Joel 3 (systematic offset)
        for verse in range(1, 22):  # Approximate Joel 4 length
            heb_ref = f"JOL.4.{verse}"
            eng_ref = f"JOL.3.{verse}"

            mappings[heb_ref] = [eng_ref]
            confidence[heb_ref] = 1.0
            types[heb_ref] = MappingType.SHIFT.value

        return {"mappings": mappings, "confidence": confidence, "types": types}

    def _load_kjv_to_modern_mappings(self) -> None:
        """Load KJV to Modern versification mappings (textual variants)."""
        mappings = {}
        confidence_scores = {}
        mapping_types = {}

        # Textual variants where KJV has verses that modern texts omit
        missing_in_modern = [
            "MAT.17.21",
            "MAT.18.11",
            "MAT.23.14",
            "MRK.7.16",
            "MRK.9.44",
            "MRK.9.46",
            "MRK.11.26",
            "MRK.15.28",
            "LUK.17.36",
            "LUK.23.17",
            "JHN.5.4",
            "ACT.8.37",
            "ACT.15.34",
            "ACT.24.7",
            "ACT.28.29",
            "ROM.16.24",
        ]

        for verse_ref in missing_in_modern:
            mappings[verse_ref] = []  # No modern equivalent
            confidence_scores[verse_ref] = 0.85  # High confidence that it's missing
            mapping_types[verse_ref] = MappingType.MISSING.value

        # Comma Johanneum (1 John 5:7) - present in KJV but disputed
        mappings["1JN.5.7"] = ["1JN.5.7"]  # Verse exists but content differs significantly
        confidence_scores["1JN.5.7"] = 0.3  # Disputed
        mapping_types["1JN.5.7"] = MappingType.EXACT.value  # Same verse number, different content

        # Store the mapping data
        mapping_data = MappingData(
            from_system=VersificationSystem.KJV,
            to_system=VersificationSystem.MODERN,
            mappings=mappings,
            confidence_scores=confidence_scores,
            mapping_types=mapping_types,
            metadata={
                "description": "KJV to Modern versification mapping (textual variants)",
                "source": "Textual criticism scholarship",
                "last_updated": "2024-01-01",
                "verse_count": len(mappings),
            },
        )

        key = (VersificationSystem.KJV, VersificationSystem.MODERN)
        self.mapping_data[key] = mapping_data

    def _load_lxx_mappings(self) -> None:
        """Load LXX (Septuagint) versification mappings."""
        mappings = {}
        confidence_scores = {}
        mapping_types = {}

        # LXX has different Psalm numbering
        # This is a simplified representation - full LXX mapping would be extensive

        # Psalms 10-113 in Hebrew are combined/split differently in LXX
        # For now, just mark known major differences

        # Hebrew Psalms 9-10 are combined as LXX Psalm 9
        mappings["PSA.10.1"] = ["PSA.9.22"]  # Approximate mapping
        confidence_scores["PSA.10.1"] = 0.7  # Medium confidence due to complexity
        mapping_types["PSA.10.1"] = MappingType.MERGE.value

        # Store basic LXX mapping data
        mapping_data = MappingData(
            from_system=VersificationSystem.LXX,
            to_system=VersificationSystem.MODERN,
            mappings=mappings,
            confidence_scores=confidence_scores,
            mapping_types=mapping_types,
            metadata={
                "description": "Septuagint to Modern versification mapping (basic)",
                "source": "LXX scholarship",
                "last_updated": "2024-01-01",
                "verse_count": len(mappings),
                "note": "Incomplete - full LXX mapping requires extensive research",
            },
        )

        key = (VersificationSystem.LXX, VersificationSystem.MODERN)
        self.mapping_data[key] = mapping_data

    def _load_external_mappings(self) -> None:
        """Load mapping data from external JSON files."""
        if not self.data_dir:
            return

        # Define expected mapping files
        mapping_files = [
            ("mt_to_modern.json", VersificationSystem.MT, VersificationSystem.MODERN),
            ("kjv_to_modern.json", VersificationSystem.KJV, VersificationSystem.MODERN),
            ("lxx_to_modern.json", VersificationSystem.LXX, VersificationSystem.MODERN),
            ("vulgate_to_modern.json", VersificationSystem.VULGATE, VersificationSystem.MODERN),
        ]

        for filename, from_sys, to_sys in mapping_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    mapping_data = MappingData(
                        from_system=from_sys,
                        to_system=to_sys,
                        mappings=data.get("mappings", {}),
                        confidence_scores=data.get("confidence_scores", {}),
                        mapping_types=data.get("mapping_types", {}),
                        metadata=data.get("metadata", {}),
                    )

                    key = (from_sys, to_sys)
                    self.mapping_data[key] = mapping_data

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to load mapping file {filename}: {e}")

    def _build_mapping_index(self) -> None:
        """Create optimized lookup structures for fast verse mapping."""
        for (from_sys, to_sys), mapping_data in self.mapping_data.items():
            index = {}

            for source_verse_str, target_verse_strs in mapping_data.mappings.items():
                target_verses = []
                for target_str in target_verse_strs:
                    target_verse = parse_verse_id(target_str)
                    if target_verse:
                        target_verses.append(target_verse)

                index[source_verse_str] = target_verses

            self.lookup_index[(from_sys, to_sys)] = index

    def get_mapping(
        self,
        source_verse: Union[str, VerseID],
        from_system: VersificationSystem,
        to_system: VersificationSystem,
    ) -> Optional[VerseMapping]:
        """
        Get mapping for a specific verse between systems.

        Args:
            source_verse: Source verse to map
            from_system: Source versification system
            to_system: Target versification system

        Returns:
            VerseMapping object or None if no mapping exists
        """
        if isinstance(source_verse, VerseID):
            source_str = str(source_verse)
            source_verse_obj = source_verse
        else:
            source_str = source_verse
            source_verse_obj = parse_verse_id(source_verse)

        if not source_verse_obj:
            return None

        # Look up in index
        key = (from_system, to_system)
        if key not in self.lookup_index:
            return None

        index = self.lookup_index[key]
        if source_str not in index:
            return None

        target_verses = index[source_str]

        # Get mapping metadata
        mapping_data = self.mapping_data[key]
        confidence = mapping_data.confidence_scores.get(source_str, 0.8)  # Default confidence
        mapping_type_str = mapping_data.mapping_types.get(source_str, MappingType.EXACT.value)

        try:
            mapping_type = MappingType(mapping_type_str)
        except ValueError:
            mapping_type = MappingType.EXACT

        try:
            confidence_enum = MappingConfidence(confidence)
        except ValueError:
            # Convert float to closest enum value
            if confidence >= 0.95:
                confidence_enum = MappingConfidence.CERTAIN
            elif confidence >= 0.85:
                confidence_enum = MappingConfidence.HIGH
            elif confidence >= 0.65:
                confidence_enum = MappingConfidence.MEDIUM
            elif confidence >= 0.45:
                confidence_enum = MappingConfidence.LOW
            else:
                confidence_enum = MappingConfidence.DISPUTED

        return VerseMapping(
            source_verse=source_verse_obj,
            target_verses=target_verses,
            mapping_type=mapping_type,
            confidence=confidence_enum,
        )

    def has_mapping(self, from_system: VersificationSystem, to_system: VersificationSystem) -> bool:
        """
        Check if mapping exists between two systems.

        Args:
            from_system: Source versification system
            to_system: Target versification system

        Returns:
            True if mapping data exists
        """
        return (from_system, to_system) in self.mapping_data

    def get_supported_systems(self) -> Set[VersificationSystem]:
        """
        Get all versification systems with mapping support.

        Returns:
            Set of supported systems
        """
        systems = set()
        for from_sys, to_sys in self.mapping_data.keys():
            systems.add(from_sys)
            systems.add(to_sys)
        return systems

    def get_mapping_statistics(
        self, from_system: VersificationSystem, to_system: VersificationSystem
    ) -> Optional[Dict[str, any]]:
        """
        Get statistics about a mapping between two systems.

        Args:
            from_system: Source versification system
            to_system: Target versification system

        Returns:
            Dictionary with mapping statistics or None if mapping doesn't exist
        """
        key = (from_system, to_system)
        if key not in self.mapping_data:
            return None

        mapping_data = self.mapping_data[key]

        # Calculate statistics
        total_mappings = len(mapping_data.mappings)
        exact_mappings = sum(
            1 for mt in mapping_data.mapping_types.values() if mt == MappingType.EXACT.value
        )
        missing_mappings = sum(
            1 for mt in mapping_data.mapping_types.values() if mt == MappingType.MISSING.value
        )

        avg_confidence = (
            sum(mapping_data.confidence_scores.values()) / len(mapping_data.confidence_scores)
            if mapping_data.confidence_scores
            else 0
        )

        return {
            "from_system": from_system.value,
            "to_system": to_system.value,
            "total_mappings": total_mappings,
            "exact_mappings": exact_mappings,
            "missing_mappings": missing_mappings,
            "complex_mappings": total_mappings - exact_mappings - missing_mappings,
            "average_confidence": round(avg_confidence, 3),
            "metadata": mapping_data.metadata,
        }

    def validate_mappings(self) -> Dict[str, any]:
        """
        Ensure bidirectional consistency in mappings.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_systems": len(self.get_supported_systems()),
            "total_mappings": len(self.mapping_data),
            "bidirectional_pairs": 0,
            "inconsistencies": [],
            "missing_reverse_mappings": [],
        }

        # Check for bidirectional consistency
        system_pairs = list(self.mapping_data.keys())

        for from_sys, to_sys in system_pairs:
            reverse_key = (to_sys, from_sys)

            if reverse_key in self.mapping_data:
                validation_results["bidirectional_pairs"] += 1

                # Check for inconsistencies (simplified check)
                forward_data = self.mapping_data[(from_sys, to_sys)]
                reverse_data = self.mapping_data[reverse_key]

                # Sample a few mappings to check consistency
                sample_mappings = list(forward_data.mappings.items())[:10]

                for source_verse, target_verses in sample_mappings:
                    if len(target_verses) == 1:  # Only check 1:1 mappings
                        target_verse = target_verses[0]
                        if target_verse in reverse_data.mappings:
                            reverse_targets = reverse_data.mappings[target_verse]
                            if len(reverse_targets) == 1 and reverse_targets[0] != source_verse:
                                validation_results["inconsistencies"].append(
                                    {
                                        "forward": f"{source_verse} -> {target_verse}",
                                        "reverse": f"{target_verse} -> {reverse_targets[0]}",
                                        "systems": f"{from_sys.value} <-> {to_sys.value}",
                                    }
                                )
            else:
                validation_results["missing_reverse_mappings"].append(
                    f"{to_sys.value} -> {from_sys.value}"
                )

        return validation_results
