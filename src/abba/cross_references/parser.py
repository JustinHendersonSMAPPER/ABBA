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
from .. import book_codes


class CrossReferenceParser:
    """Parser for various cross-reference data formats."""

    def __init__(self):
        """Initialize the parser."""
        self.known_abbreviations = self._build_book_abbreviations()
        self.reference_patterns = self._build_reference_patterns()

    def _build_book_abbreviations(self) -> Dict[str, str]:
        """Build mapping of book abbreviations to canonical codes."""
        abbrevs = {}
        
        # Add standard book names
        book_info = book_codes.BOOK_INFO
        for code, info in book_info.items():
            # Add full name
            abbrevs[info["name"]] = code
            # Add abbreviation
            abbrevs[info["abbr"]] = code
            # Add the code itself
            abbrevs[code] = code
        
        # Add common variations
        common_variations = {
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
            "Song": "SNG",
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
            "Mal": "MAL",
            "Malachi": "MAL",
            # New Testament
            "Mat": "MAT",
            "Matt": "MAT",
            "Matthew": "MAT",
            "Mar": "MRK",
            "Mark": "MRK",
            "Mrk": "MRK",
            "Luk": "LUK",
            "Luke": "LUK",
            "Joh": "JHN",
            "John": "JHN",
            "Jhn": "JHN",
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
            "Phil": "PHP",
            "Philippians": "PHP",
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
            "Jam": "JAS",
            "James": "JAS",
            "Jas": "JAS",
            "1Pe": "1PE",
            "1 Peter": "1PE",
            "1 Pet": "1PE",
            "2Pe": "2PE",
            "2 Peter": "2PE",
            "2 Pet": "2PE",
            "1Jo": "1JN",
            "1 John": "1JN",
            "1Jn": "1JN",
            "2Jo": "2JN",
            "2 John": "2JN",
            "2Jn": "2JN",
            "3Jo": "3JN",
            "3 John": "3JN",
            "3Jn": "3JN",
            "Jud": "JUD",
            "Jude": "JUD",
            "Rev": "REV",
            "Revelation": "REV",
        }
        
        abbrevs.update(common_variations)
        return abbrevs

    def _build_reference_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for parsing references."""
        patterns = []
        
        # Pattern: Book Chapter:Verse (e.g., "John 3:16")
        patterns.append(
            re.compile(r"([1-3]?\s*\w+)\s+(\d+):(\d+)(?:-(\d+))?")
        )
        
        # Pattern: Book Chapter.Verse (e.g., "John 3.16")
        patterns.append(
            re.compile(r"([1-3]?\s*\w+)\s+(\d+)\.(\d+)(?:-(\d+))?")
        )
        
        # Pattern: Book Chapter Verse (e.g., "John 3 16")
        patterns.append(
            re.compile(r"([1-3]?\s*\w+)\s+(\d+)\s+(\d+)(?:-(\d+))?")
        )
        
        return patterns

    def normalize_book_name(self, book_name: str) -> Optional[str]:
        """Normalize a book name to its canonical code."""
        # Clean the book name
        book_name = book_name.strip()
        
        # Try direct lookup
        if book_name in self.known_abbreviations:
            return self.known_abbreviations[book_name]
        
        # Try case-insensitive lookup
        for name, code in self.known_abbreviations.items():
            if name.lower() == book_name.lower():
                return code
        
        # Try using book_codes module
        normalized = book_codes.normalize_book_name(book_name)
        if normalized:
            return normalized
        
        return None

    def parse_reference_string(self, ref_string: str, expand_ranges: bool = False) -> List[VerseID]:
        """
        Parse a reference string into VerseID object(s).
        
        Args:
            ref_string: Reference string (e.g., "John 3:16" or "John 3:16; Matt 5:17-19")
            expand_ranges: If True, expand ranges to all verses. If False, only return start verse.
            
        Returns:
            List[VerseID] for all parsed references
        """
        verses = []
        
        # Split by common separators
        parts = re.split(r"[;,]", ref_string)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Try each pattern
            for pattern in self.reference_patterns:
                match = pattern.match(part)
                if match:
                    groups = match.groups()
                    book_name = groups[0]
                    chapter = int(groups[1])
                    start_verse = int(groups[2])
                    end_verse = int(groups[3]) if groups[3] else start_verse
                    
                    # Normalize book name
                    book_code = self.normalize_book_name(book_name)
                    if book_code:
                        if expand_ranges:
                            # Add all verses in range
                            for verse_num in range(start_verse, end_verse + 1):
                                try:
                                    verse_id = VerseID(book_code, chapter, verse_num)
                                    verses.append(verse_id)
                                except:
                                    # Skip invalid verses
                                    pass
                        else:
                            # Only add start verse
                            try:
                                verse_id = VerseID(book_code, chapter, start_verse)
                                verses.append(verse_id)
                            except:
                                # Skip invalid verses
                                pass
                    break
            else:
                # Try simple parsing
                try:
                    verse_id = parse_verse_id(part)
                    if verse_id:
                        verses.append(verse_id)
                except:
                    pass
        
        # Always return a list
        return verses
    
    def parse_reference_list(self, ref_string: str) -> List[VerseID]:
        """
        Parse a list of references (alias for parse_reference_string).
        
        Args:
            ref_string: Reference string with multiple references
            
        Returns:
            List of VerseID objects
        """
        # For lists, we don't expand ranges by default
        return self.parse_reference_string(ref_string, expand_ranges=False)
    
    def parse_reference_range(self, ref_string: str) -> List[VerseID]:
        """
        Parse a single reference that may contain a range.
        
        Args:
            ref_string: Single reference string (e.g., "Gen 1:1-3")
            
        Returns:
            List of VerseID objects for all verses in range
        """
        # For ranges, we want to expand them
        return self.parse_reference_string(ref_string, expand_ranges=True)

    def parse_json_references(self, json_path: str) -> ReferenceCollection:
        """
        Parse cross-references from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            ReferenceCollection object
        """
        references = []
        metadata = {}
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Extract metadata - check for explicit metadata field first
            if "metadata" in data:
                metadata = data["metadata"]
            else:
                # Otherwise extract all non-references keys as metadata
                metadata = {k: v for k, v in data.items() if k != "references"}
            items = data.get("references", [])
        else:
            items = []
        
        for item in items:
            try:
                # Parse source and target verses
                source_verse = self._parse_verse_from_json(item.get("source"))
                target_verse = self._parse_verse_from_json(item.get("target"))
                
                if not source_verse or not target_verse:
                    continue
                
                # Determine reference type
                ref_type = self._determine_reference_type(item)
                relationship = self._determine_relationship(item)
                
                # Create confidence
                confidence = ReferenceConfidence(
                    overall_score=item.get("confidence", 0.5),
                    textual_similarity=item.get("text_similarity", 0.5),
                    thematic_similarity=item.get("thematic_similarity", 0.5),
                    structural_similarity=item.get("structural_similarity", 0.5),
                    scholarly_consensus=item.get("scholarly_consensus", 0.5),
                    uncertainty_factors=item.get("uncertainty", [])
                )
                
                # Create reference
                ref = CrossReference(
                    source_verse=source_verse,
                    target_verse=target_verse,
                    reference_type=ref_type,
                    relationship=relationship,
                    confidence=confidence,
                    source=item.get("source_db", "json"),
                    verified=item.get("verified", False)
                )
                
                references.append(ref)
                
            except Exception as e:
                # Skip problematic entries
                continue
        
        # Create and return collection
        collection = ReferenceCollection(
            references=references,
            metadata=metadata
        )
        return collection

    def _parse_verse_from_json(self, verse_data: Any) -> Optional[VerseID]:
        """Parse verse from JSON data."""
        if not verse_data:
            return None
        
        if isinstance(verse_data, str):
            # Try parsing as string reference
            result = self.parse_reference_string(verse_data)
            if isinstance(result, VerseID):
                return result
            elif isinstance(result, list) and len(result) > 0:
                return result[0]
            return None
        
        elif isinstance(verse_data, dict):
            # Parse from dict
            book = verse_data.get("book")
            chapter = verse_data.get("chapter")
            verse = verse_data.get("verse")
            
            if book and chapter and verse:
                book_code = self.normalize_book_name(book)
                if book_code:
                    try:
                        return VerseID(book_code, int(chapter), int(verse))
                    except:
                        pass
        
        return None

    def _determine_reference_type(self, item: Dict) -> ReferenceType:
        """Determine reference type from JSON data."""
        ref_type = item.get("type", "").lower()
        
        type_mapping = {
            "quote": ReferenceType.DIRECT_QUOTE,
            "direct_quote": ReferenceType.DIRECT_QUOTE,
            "partial": ReferenceType.PARTIAL_QUOTE,
            "partial_quote": ReferenceType.PARTIAL_QUOTE,
            "paraphrase": ReferenceType.PARAPHRASE,
            "allusion": ReferenceType.ALLUSION,
            "parallel": ReferenceType.THEMATIC_PARALLEL,
            "thematic": ReferenceType.THEMATIC_PARALLEL,
            "prophecy": ReferenceType.PROPHECY_FULFILLMENT,
            "type": ReferenceType.TYPE_ANTITYPE,
        }
        
        return type_mapping.get(ref_type, ReferenceType.ALLUSION)

    def _determine_relationship(self, item: Dict) -> ReferenceRelationship:
        """Determine relationship from JSON data."""
        rel = item.get("relationship", "").lower()
        
        rel_mapping = {
            "quotes": ReferenceRelationship.QUOTES,
            "quoted_by": ReferenceRelationship.QUOTED_BY,
            "alludes": ReferenceRelationship.ALLUDES_TO,
            "alludes_to": ReferenceRelationship.ALLUDES_TO,
            "fulfills": ReferenceRelationship.FULFILLS,
            "explains": ReferenceRelationship.EXPLAINS,
            "parallels": ReferenceRelationship.PARALLELS,
        }
        
        return rel_mapping.get(rel, ReferenceRelationship.ALLUDES_TO)
    
    def create_sample_references(self) -> ReferenceCollection:
        """Create a sample collection of cross references for testing."""
        collection = ReferenceCollection(
            references=[],
            metadata={"format": "sample", "created": "2024-01-01"}
        )
        
        # Add some sample references
        sample_refs = [
            {
                "source_verse": "MAT.5.17",
                "target_verses": ["DEU.4.2", "DEU.12.32"],
                "type": ReferenceType.DIRECT_QUOTE,
                "relationship": ReferenceRelationship.QUOTES,
                "confidence": 0.9,
                "text_match": "Do not think that I have come to abolish the Law"
            },
            {
                "source_verse": "HEB.11.1",
                "target_verses": ["ROM.8.24", "2CO.5.7"],
                "type": ReferenceType.THEMATIC_PARALLEL,
                "relationship": ReferenceRelationship.PARALLELS,
                "confidence": 0.85,
                "notes": "Faith and hope theme"
            },
            {
                "source_verse": "REV.22.13",
                "target_verses": ["ISA.44.6", "ISA.48.12"],
                "type": ReferenceType.ALLUSION,
                "relationship": ReferenceRelationship.ALLUDES_TO,
                "confidence": 0.95,
                "text_match": "I am the Alpha and the Omega"
            }
        ]
        
        for ref_data in sample_refs:
            source = parse_verse_id(ref_data["source_verse"])
            if source:
                for target_str in ref_data["target_verses"]:
                    target = parse_verse_id(target_str)
                    if target:
                        # Create confidence object
                        conf_score = ref_data.get("confidence", 0.8)
                        confidence = ReferenceConfidence(
                            overall_score=conf_score,
                            textual_similarity=conf_score,
                            thematic_similarity=conf_score * 0.9,
                            structural_similarity=conf_score * 0.8,
                            scholarly_consensus=conf_score * 0.85,
                            uncertainty_factors=[]
                        )
                        
                        ref = CrossReference(
                            source_verse=source,
                            target_verse=target,
                            reference_type=ref_data["type"],
                            relationship=ref_data["relationship"],
                            confidence=confidence
                        )
                        collection.references.append(ref)
        
        return collection
    
    def merge_collections(self, collections: List[ReferenceCollection]) -> ReferenceCollection:
        """Merge multiple reference collections, removing duplicates."""
        merged = ReferenceCollection(
            references=[],
            metadata={"format": "merged", "source_collections": len(collections)}
        )
        
        # Track unique references to avoid duplicates
        seen_refs = set()
        
        for collection in collections:
            for ref in collection.references:
                # Create a unique key for the reference
                ref_key = (
                    str(ref.source_verse),
                    str(ref.target_verse),
                    ref.reference_type,
                    ref.relationship
                )
                
                if ref_key not in seen_refs:
                    seen_refs.add(ref_key)
                    merged.references.append(ref)
        
        return merged