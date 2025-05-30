"""
Cross-reference parser for biblical reference databases.

Parses various formats of cross-reference data and normalizes them
into the ABBA cross-reference format.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .models import (
    CrossReference,
    ReferenceType,
    ReferenceRelationship,
    ReferenceConfidence,
    ReferenceCollection,
)
from ..verse_id import VerseID, parse_verse_id


class CrossReferenceParser:
    """Parser for various cross-reference data formats."""

    def __init__(self):
        """Initialize the parser."""
        self.known_abbreviations = self._build_book_abbreviations()
        self.reference_patterns = self._build_reference_patterns()

    def _build_book_abbreviations(self) -> Dict[str, str]:
        """Build mapping of book abbreviations to canonical codes."""
        return {
            # Old Testament
            "Gen": "GEN",
            "Genesis": "GEN",
            "Exo": "EXO",
            "Exodus": "EXO",
            "Ex": "EXO",
            "Lev": "LEV",
            "Leviticus": "LEV",
            "Num": "NUM",
            "Numbers": "NUM",
            "Deu": "DEU",
            "Deuteronomy": "DEU",
            "Dt": "DEU",
            "Jos": "JOS",
            "Joshua": "JOS",
            "Jdg": "JDG",
            "Judges": "JDG",
            "Rut": "RUT",
            "Ruth": "RUT",
            "1Sa": "1SA",
            "1 Samuel": "1SA",
            "1 Sam": "1SA",
            "2Sa": "2SA",
            "2 Samuel": "2SA",
            "2 Sam": "2SA",
            "1Ki": "1KI",
            "1 Kings": "1KI",
            "2Ki": "2KI",
            "2 Kings": "2KI",
            "1Ch": "1CH",
            "1 Chronicles": "1CH",
            "1 Chr": "1CH",
            "2Ch": "2CH",
            "2 Chronicles": "2CH",
            "2 Chr": "2CH",
            "Ezr": "EZR",
            "Ezra": "EZR",
            "Neh": "NEH",
            "Nehemiah": "NEH",
            "Est": "EST",
            "Esther": "EST",
            "Job": "JOB",
            "Psa": "PSA",
            "Psalms": "PSA",
            "Ps": "PSA",
            "Pro": "PRO",
            "Proverbs": "PRO",
            "Prov": "PRO",
            "Ecc": "ECC",
            "Ecclesiastes": "ECC",
            "Sol": "SNG",
            "Song of Solomon": "SNG",
            "SOS": "SNG",
            "Isa": "ISA",
            "Isaiah": "ISA",
            "Is": "ISA",
            "Jer": "JER",
            "Jeremiah": "JER",
            "Lam": "LAM",
            "Lamentations": "LAM",
            "Eze": "EZK",
            "Ezekiel": "EZK",
            "Ezek": "EZK",
            "Dan": "DAN",
            "Daniel": "DAN",
            "Hos": "HOS",
            "Hosea": "HOS",
            "Joe": "JOL",
            "Joel": "JOL",
            "Amo": "AMO",
            "Amos": "AMO",
            "Oba": "OBA",
            "Obadiah": "OBA",
            "Jon": "JON",
            "Jonah": "JON",
            "Mic": "MIC",
            "Micah": "MIC",
            "Nah": "NAM",
            "Nahum": "NAM",
            "Hab": "HAB",
            "Habakkuk": "HAB",
            "Zep": "ZEP",
            "Zephaniah": "ZEP",
            "Hag": "HAG",
            "Haggai": "HAG",
            "Zec": "ZEC",
            "Zechariah": "ZEC",
            "Zech": "ZEC",
            "Mal": "MAL",
            "Malachi": "MAL",
            # New Testament
            "Mat": "MAT",
            "Matthew": "MAT",
            "Mt": "MAT",
            "Mar": "MRK",
            "Mark": "MRK",
            "Mk": "MRK",
            "Luk": "LUK",
            "Luke": "LUK",
            "Lk": "LUK",
            "Joh": "JHN",
            "John": "JHN",
            "Jn": "JHN",
            "Act": "ACT",
            "Acts": "ACT",
            "Rom": "ROM",
            "Romans": "ROM",
            "1Co": "1CO",
            "1 Corinthians": "1CO",
            "1 Cor": "1CO",
            "2Co": "2CO",
            "2 Corinthians": "2CO",
            "2 Cor": "2CO",
            "Gal": "GAL",
            "Galatians": "GAL",
            "Eph": "EPH",
            "Ephesians": "EPH",
            "Phi": "PHP",
            "Philippians": "PHP",
            "Phil": "PHP",
            "Col": "COL",
            "Colossians": "COL",
            "1Th": "1TH",
            "1 Thessalonians": "1TH",
            "1 Thess": "1TH",
            "2Th": "2TH",
            "2 Thessalonians": "2TH",
            "2 Thess": "2TH",
            "1Ti": "1TI",
            "1 Timothy": "1TI",
            "1 Tim": "1TI",
            "2Ti": "2TI",
            "2 Timothy": "2TI",
            "2 Tim": "2TI",
            "Tit": "TIT",
            "Titus": "TIT",
            "Phm": "PHM",
            "Philemon": "PHM",
            "Heb": "HEB",
            "Hebrews": "HEB",
            "Jas": "JAS",
            "James": "JAS",
            "1Pe": "1PE",
            "1 Peter": "1PE",
            "1 Pet": "1PE",
            "2Pe": "2PE",
            "2 Peter": "2PE",
            "2 Pet": "2PE",
            "1Jo": "1JN",
            "1 John": "1JN",
            "1 Jn": "1JN",
            "2Jo": "2JN",
            "2 John": "2JN",
            "2 Jn": "2JN",
            "3Jo": "3JN",
            "3 John": "3JN",
            "3 Jn": "3JN",
            "Jud": "JUD",
            "Jude": "JUD",
            "Rev": "REV",
            "Revelation": "REV",
        }

    def _build_reference_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for parsing biblical references."""
        patterns = [
            # Full format: "Genesis 1:1"
            re.compile(r"([123]?\s*[A-Za-z]+)\s+(\d+):(\d+)(?:-(\d+))?"),
            # Short format: "Gen 1:1"
            re.compile(r"([123]?[A-Za-z]+)\s+(\d+):(\d+)(?:-(\d+))?"),
            # Abbreviated: "Ge1:1"
            re.compile(r"([123]?[A-Za-z]+)(\d+):(\d+)(?:-(\d+))?"),
            # With periods: "Gen. 1:1"
            re.compile(r"([123]?\s*[A-Za-z]+)\.\s+(\d+):(\d+)(?:-(\d+))?"),
        ]
        return patterns

    def parse_reference_string(self, ref_string: str) -> Optional[VerseID]:
        """Parse a biblical reference string into a VerseID."""
        ref_string = ref_string.strip()

        for pattern in self.reference_patterns:
            match = pattern.match(ref_string)
            if match:
                book_str = match.group(1).strip()
                chapter = int(match.group(2))
                verse = int(match.group(3))

                # Normalize book name
                canonical_book = self.known_abbreviations.get(book_str)
                if canonical_book:
                    try:
                        return VerseID(book=canonical_book, chapter=chapter, verse=verse)
                    except ValueError:
                        continue

        return None

    def parse_reference_range(self, ref_string: str) -> List[VerseID]:
        """Parse a reference that might include verse ranges."""
        # Handle ranges like "Gen 1:1-3" or "Gen 1:1-2:5"
        if "-" in ref_string:
            parts = ref_string.split("-")
            if len(parts) == 2:
                start_ref = self.parse_reference_string(parts[0].strip())
                end_part = parts[1].strip()

                if start_ref:
                    # If end part is just a number, it's a verse range in same chapter
                    if end_part.isdigit():
                        end_verse = int(end_part)
                        verses = []
                        for v in range(start_ref.verse, end_verse + 1):
                            verses.append(
                                VerseID(book=start_ref.book, chapter=start_ref.chapter, verse=v)
                            )
                        return verses
                    else:
                        # Try to parse as full reference
                        end_ref = self.parse_reference_string(end_part)
                        if end_ref:
                            # Generate range between start and end
                            return self._generate_verse_range(start_ref, end_ref)

        # Single verse
        single_ref = self.parse_reference_string(ref_string)
        return [single_ref] if single_ref else []

    def _generate_verse_range(self, start: VerseID, end: VerseID) -> List[VerseID]:
        """Generate all verses in a range."""
        verses = []

        if start.book != end.book:
            # Cross-book ranges not supported for simplicity
            return [start, end]

        if start.chapter == end.chapter:
            # Same chapter
            for v in range(start.verse, end.verse + 1):
                verses.append(VerseID(book=start.book, chapter=start.chapter, verse=v))
        else:
            # Cross-chapter range - just return endpoints for now
            verses = [start, end]

        return verses

    def parse_reference_list(self, ref_list_string: str) -> List[VerseID]:
        """Parse a list of references separated by semicolons or commas."""
        # Split on common separators
        separators = [";", ",", "|"]
        ref_strings = [ref_list_string]

        for sep in separators:
            new_strings = []
            for s in ref_strings:
                new_strings.extend(s.split(sep))
            ref_strings = new_strings

        verses = []
        for ref_str in ref_strings:
            ref_str = ref_str.strip()
            if ref_str:
                verses.extend(self.parse_reference_range(ref_str))

        return [v for v in verses if v is not None]

    def parse_json_references(self, file_path: Path) -> ReferenceCollection:
        """Parse cross-references from JSON format."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        references = []

        for item in data.get("references", []):
            source_verse = parse_verse_id(item["source"])
            target_refs = self.parse_reference_list(item["target"])

            for target_verse in target_refs:
                # Create basic confidence
                confidence = ReferenceConfidence(
                    overall_score=item.get("confidence", 0.8),
                    textual_similarity=item.get("textual_similarity", 0.5),
                    thematic_similarity=item.get("thematic_similarity", 0.7),
                    structural_similarity=item.get("structural_similarity", 0.3),
                    scholarly_consensus=item.get("scholarly_consensus", 0.8),
                    uncertainty_factors=item.get("uncertainty_factors", []),
                )

                # Determine reference type
                ref_type = ReferenceType.THEMATIC_PARALLEL
                if "type" in item:
                    try:
                        ref_type = ReferenceType(item["type"])
                    except ValueError:
                        pass

                # Determine relationship
                relationship = ReferenceRelationship.PARALLELS
                if "relationship" in item:
                    try:
                        relationship = ReferenceRelationship(item["relationship"])
                    except ValueError:
                        pass

                ref = CrossReference(
                    source_verse=source_verse,
                    target_verse=target_verse,
                    reference_type=ref_type,
                    relationship=relationship,
                    confidence=confidence,
                    source="json_import",
                )
                references.append(ref)

        metadata = data.get("metadata", {})
        metadata["format"] = "json"
        metadata["source_file"] = str(file_path)

        return ReferenceCollection(references=references, metadata=metadata)

    def parse_xml_references(self, file_path: Path) -> ReferenceCollection:
        """Parse cross-references from XML format."""
        tree = ET.parse(file_path)
        root = tree.getroot()

        references = []

        # Look for different XML structures
        for ref_elem in root.findall(".//reference"):
            source_str = ref_elem.get("source") or ref_elem.find("source").text
            target_str = ref_elem.get("target") or ref_elem.find("target").text

            source_verse = self.parse_reference_string(source_str)
            target_verses = self.parse_reference_list(target_str)

            if source_verse:
                for target_verse in target_verses:
                    confidence = ReferenceConfidence(
                        overall_score=float(ref_elem.get("confidence", 0.8)),
                        textual_similarity=0.5,
                        thematic_similarity=0.7,
                        structural_similarity=0.3,
                        scholarly_consensus=0.8,
                        uncertainty_factors=[],
                    )

                    ref_type_str = ref_elem.get("type", "thematic_parallel")
                    try:
                        ref_type = ReferenceType(ref_type_str)
                    except ValueError:
                        ref_type = ReferenceType.THEMATIC_PARALLEL

                    ref = CrossReference(
                        source_verse=source_verse,
                        target_verse=target_verse,
                        reference_type=ref_type,
                        relationship=ReferenceRelationship.PARALLELS,
                        confidence=confidence,
                        source="xml_import",
                    )
                    references.append(ref)

        metadata = {"format": "xml", "source_file": str(file_path)}

        return ReferenceCollection(references=references, metadata=metadata)

    def parse_treasury_format(self, file_path: Path) -> ReferenceCollection:
        """Parse Treasury of Scripture Knowledge format."""
        references = []

        with open(file_path, "r", encoding="utf-8") as f:
            current_verse = None

            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Check if this is a verse header
                verse_match = re.match(r"^([A-Za-z0-9 ]+\s+\d+:\d+)", line)
                if verse_match:
                    current_verse = self.parse_reference_string(verse_match.group(1))
                    # Extract references from the rest of the line
                    ref_part = line[verse_match.end() :].strip()
                    if ref_part:
                        target_verses = self.parse_reference_list(ref_part)
                        self._add_treasury_references(references, current_verse, target_verses)
                elif current_verse:
                    # Continuation line with more references
                    target_verses = self.parse_reference_list(line)
                    self._add_treasury_references(references, current_verse, target_verses)

        metadata = {
            "format": "treasury",
            "source_file": str(file_path),
            "description": "Treasury of Scripture Knowledge cross-references",
        }

        return ReferenceCollection(references=references, metadata=metadata)

    def _add_treasury_references(
        self, references: List[CrossReference], source_verse: VerseID, target_verses: List[VerseID]
    ):
        """Add Treasury-style references to the collection."""
        for target_verse in target_verses:
            confidence = ReferenceConfidence(
                overall_score=0.75,  # Treasury has good historical reliability
                textual_similarity=0.4,
                thematic_similarity=0.8,
                structural_similarity=0.3,
                scholarly_consensus=0.9,  # Well-established references
                uncertainty_factors=[],
            )

            ref = CrossReference(
                source_verse=source_verse,
                target_verse=target_verse,
                reference_type=ReferenceType.THEMATIC_PARALLEL,
                relationship=ReferenceRelationship.PARALLELS,
                confidence=confidence,
                source="treasury",
            )
            references.append(ref)

    def create_sample_references(self) -> ReferenceCollection:
        """Create sample cross-references for testing and demonstration."""
        references = []

        # Genesis 1:1 references
        gen_1_1 = VerseID(book="GEN", chapter=1, verse=1)
        john_1_1 = VerseID(book="JHN", chapter=1, verse=1)

        # John 1:1 echoes Genesis 1:1 ("In the beginning")
        confidence = ReferenceConfidence(
            overall_score=0.95,
            textual_similarity=0.9,  # "In the beginning" phrase
            thematic_similarity=0.95,  # Creation theme
            structural_similarity=0.7,
            scholarly_consensus=0.98,
            uncertainty_factors=[],
            lexical_links=3,  # "in", "the", "beginning"
            semantic_links=2,  # creation, divine action
            contextual_support=5,
        )

        ref1 = CrossReference(
            source_verse=john_1_1,
            target_verse=gen_1_1,
            reference_type=ReferenceType.ALLUSION,
            relationship=ReferenceRelationship.ALLUDES_TO,
            confidence=confidence,
            source="sample_data",
            theological_theme="creation_parallel",
        )
        references.append(ref1)

        # Psalm 23:1 and John 10:11 (shepherd theme)
        ps_23_1 = VerseID(book="PSA", chapter=23, verse=1)
        john_10_11 = VerseID(book="JHN", chapter=10, verse=11)

        confidence = ReferenceConfidence(
            overall_score=0.88,
            textual_similarity=0.6,  # Different words, same concept
            thematic_similarity=0.98,  # Shepherd theme
            structural_similarity=0.4,
            scholarly_consensus=0.95,
            uncertainty_factors=[],
            lexical_links=1,  # "shepherd"
            semantic_links=4,  # care, protection, guidance, provision
            contextual_support=4,
        )

        ref2 = CrossReference(
            source_verse=john_10_11,
            target_verse=ps_23_1,
            reference_type=ReferenceType.THEMATIC_PARALLEL,
            relationship=ReferenceRelationship.FULFILLS,
            confidence=confidence,
            source="sample_data",
            theological_theme="shepherd_messiah",
        )
        references.append(ref2)

        # Isaiah 53:7 quoted in Acts 8:32
        isa_53_7 = VerseID(book="ISA", chapter=53, verse=7)
        acts_8_32 = VerseID(book="ACT", chapter=8, verse=32)

        confidence = ReferenceConfidence(
            overall_score=0.99,
            textual_similarity=0.95,  # Near exact quotation
            thematic_similarity=0.98,
            structural_similarity=0.9,
            scholarly_consensus=0.99,
            uncertainty_factors=[],
            lexical_links=8,  # Most words match
            semantic_links=3,
            contextual_support=5,
        )

        ref3 = CrossReference(
            source_verse=acts_8_32,
            target_verse=isa_53_7,
            reference_type=ReferenceType.DIRECT_QUOTE,
            relationship=ReferenceRelationship.QUOTES,
            confidence=confidence,
            source="sample_data",
            theological_theme="suffering_servant",
        )
        references.append(ref3)

        # Create bidirectional references
        all_refs = []
        for ref in references:
            all_refs.append(ref)
            # Add reverse reference if it makes sense
            if ref.relationship != ReferenceRelationship.PARALLELS:
                all_refs.append(ref.get_reverse_reference())

        metadata = {
            "format": "sample",
            "description": "Sample cross-references for testing",
            "version": "1.0",
            "created_by": "ABBA system",
        }

        return ReferenceCollection(references=all_refs, metadata=metadata)

    def merge_collections(self, collections: List[ReferenceCollection]) -> ReferenceCollection:
        """Merge multiple reference collections, removing duplicates."""
        all_references = []
        all_metadata = {}

        # Collect references and detect duplicates
        seen_pairs = set()

        for collection in collections:
            all_metadata.update(collection.metadata)

            for ref in collection.references:
                # Create a key for duplicate detection
                key = (str(ref.source_verse), str(ref.target_verse), ref.reference_type)

                if key not in seen_pairs:
                    seen_pairs.add(key)
                    all_references.append(ref)
                else:
                    # TODO: Could merge confidence scores here
                    pass

        merged_metadata = {
            "format": "merged",
            "source_collections": len(collections),
            "merged_data": all_metadata,
        }

        return ReferenceCollection(references=all_references, metadata=merged_metadata)
