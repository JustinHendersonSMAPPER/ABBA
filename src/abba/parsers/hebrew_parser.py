"""
Hebrew XML parser for morphological data.

This module parses OSIS format Hebrew Bible XML files with Strong's numbers
and morphological annotations from the Open Scriptures Hebrew Bible project.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..verse_id import VerseID, parse_verse_id


@dataclass
class HebrewWord:
    """Represents a Hebrew word with morphological data."""

    text: str
    lemma: Optional[str] = None
    strong_number: Optional[str] = None
    morph: Optional[str] = None
    gloss: Optional[str] = None
    id: Optional[str] = None

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> "HebrewWord":
        """Create HebrewWord from XML <w> element."""
        text = element.text or ""

        # Extract lemma and Strong's number
        lemma_attr = element.get("lemma", "")
        strong_number = None
        lemma = None

        if lemma_attr:
            # Parse lemma format: "1254 a" or "b/7225" or "430"
            lemma_parts = lemma_attr.split()
            if lemma_parts:
                # Extract Strong's number (first numeric part)
                strong_match = re.search(r"\d+", lemma_parts[0])
                if strong_match:
                    strong_number = f"H{strong_match.group()}"
                lemma = lemma_attr

        return cls(
            text=text,
            lemma=lemma,
            strong_number=strong_number,
            morph=element.get("morph"),
            id=element.get("id"),
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "lemma": self.lemma,
            "strong_number": self.strong_number,
            "morph": self.morph,
            "gloss": self.gloss,
            "id": self.id,
        }


@dataclass
class HebrewVerse:
    """Represents a Hebrew verse with morphological data."""

    verse_id: VerseID
    words: List[HebrewWord]
    osis_id: str

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> Optional["HebrewVerse"]:
        """Create HebrewVerse from XML <verse> element."""
        osis_id = element.get("osisID", "")
        if not osis_id:
            return None

        # Parse verse ID from OSIS format (e.g., "Gen.1.1")
        verse_id = parse_verse_id(osis_id)
        if not verse_id:
            return None

        # Extract words
        words = []
        # Look for <w> elements both with and without namespace
        word_elements = element.findall(".//w") + element.findall(
            ".//{http://www.bibletechnologies.net/2003/OSIS/namespace}w"
        )
        for word_elem in word_elements:
            word = HebrewWord.from_xml_element(word_elem)
            if word.text.strip():  # Only include non-empty words
                words.append(word)

        return cls(
            verse_id=verse_id,
            words=words,
            osis_id=osis_id,
        )

    def to_dict(self) -> Dict[str, Union[str, List[Dict[str, Optional[str]]]]]:
        """Convert to dictionary format."""
        return {
            "verse_id": str(self.verse_id),
            "osis_id": self.osis_id,
            "words": [word.to_dict() for word in self.words],
        }


class HebrewParser:
    """Parser for Hebrew OSIS XML files."""

    def __init__(self) -> None:
        """Initialize the Hebrew parser."""
        self.namespace = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

    def parse_file(self, file_path: Union[str, Path]) -> List[HebrewVerse]:
        """
        Parse a Hebrew XML file and extract verses with morphological data.

        Args:
            file_path: Path to the Hebrew XML file

        Returns:
            List of HebrewVerse objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ET.ParseError: If the XML is malformed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Hebrew XML file not found: {file_path}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find all verses using namespace
            verses = []
            # Look for <verse> elements both with and without namespace
            verse_elements = root.findall(".//verse") + root.findall(
                ".//osis:verse", self.namespace
            )
            for verse_elem in verse_elements:
                verse = HebrewVerse.from_xml_element(verse_elem)
                if verse:
                    verses.append(verse)

            return verses

        except ET.ParseError as e:
            raise ET.ParseError(f"Failed to parse Hebrew XML file {file_path}: {e}")

    def parse_book(self, book_code: str, data_dir: Union[str, Path]) -> List[HebrewVerse]:
        """
        Parse a Hebrew book by book code.

        Args:
            book_code: 3-letter book code (e.g., "GEN")
            data_dir: Directory containing Hebrew XML files

        Returns:
            List of HebrewVerse objects for the book
        """
        data_dir = Path(data_dir)

        # Map canonical book codes to Hebrew filenames
        filename_map = {
            "GEN": "Gen.xml",
            "EXO": "Exod.xml",
            "LEV": "Lev.xml",
            "NUM": "Num.xml",
            "DEU": "Deut.xml",
            "JOS": "Josh.xml",
            "JDG": "Judg.xml",
            "RUT": "Ruth.xml",
            "1SA": "1Sam.xml",
            "2SA": "2Sam.xml",
            "1KI": "1Kgs.xml",
            "2KI": "2Kgs.xml",
            "1CH": "1Chr.xml",
            "2CH": "2Chr.xml",
            "EZR": "Ezra.xml",
            "NEH": "Neh.xml",
            "EST": "Esth.xml",
            "JOB": "Job.xml",
            "PSA": "Ps.xml",
            "PRO": "Prov.xml",
            "ECC": "Eccl.xml",
            "SNG": "Song.xml",
            "ISA": "Isa.xml",
            "JER": "Jer.xml",
            "LAM": "Lam.xml",
            "EZK": "Ezek.xml",
            "DAN": "Dan.xml",
            "HOS": "Hos.xml",
            "JOL": "Joel.xml",
            "AMO": "Amos.xml",
            "OBA": "Obad.xml",
            "JON": "Jonah.xml",
            "MIC": "Mic.xml",
            "NAM": "Nah.xml",
            "HAB": "Hab.xml",
            "ZEP": "Zeph.xml",
            "HAG": "Hag.xml",
            "ZEC": "Zech.xml",
            "MAL": "Mal.xml",
        }

        filename = filename_map.get(book_code.upper())
        if not filename:
            raise ValueError(f"Unknown Hebrew book code: {book_code}")

        file_path = data_dir / filename
        return self.parse_file(file_path)

    def get_book_statistics(self, verses: List[HebrewVerse]) -> Dict[str, int]:
        """
        Get statistics for parsed Hebrew verses.

        Args:
            verses: List of HebrewVerse objects

        Returns:
            Dictionary with statistics
        """
        if not verses:
            return {"verse_count": 0, "word_count": 0, "morphed_word_count": 0}

        word_count = sum(len(verse.words) for verse in verses)
        morphed_word_count = sum(1 for verse in verses for word in verse.words if word.morph)

        return {
            "verse_count": len(verses),
            "word_count": word_count,
            "morphed_word_count": morphed_word_count,
            "book_code": verses[0].verse_id.book if verses else "unknown",
        }

    def extract_strong_numbers(self, verses: List[HebrewVerse]) -> List[str]:
        """
        Extract all unique Strong's numbers from verses.

        Args:
            verses: List of HebrewVerse objects

        Returns:
            Sorted list of unique Strong's numbers
        """
        strong_numbers = set()
        for verse in verses:
            for word in verse.words:
                if word.strong_number:
                    strong_numbers.add(word.strong_number)

        return sorted(strong_numbers)
