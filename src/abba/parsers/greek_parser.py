"""
Greek XML parser for morphological data.

This module parses TEI format Greek New Testament XML files from the
Byzantine Majority Text project.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..verse_id import VerseID, create_verse_id


@dataclass
class GreekWord:
    """Represents a Greek word."""

    text: str
    lemma: Optional[str] = None
    morph: Optional[str] = None
    strong_number: Optional[str] = None
    id: Optional[str] = None

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> "GreekWord":
        """Create GreekWord from XML <w> element."""
        text = element.text or ""

        # Try different ways to get the xml:id attribute
        word_id = element.get("xml:id") or element.get("{http://www.w3.org/XML/1998/namespace}id")

        return cls(
            text=text,
            lemma=element.get("lemma"),
            morph=element.get("ana"),  # TEI uses "ana" for morphological analysis
            id=word_id,
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "lemma": self.lemma,
            "morph": self.morph,
            "strong_number": self.strong_number,
            "id": self.id,
        }


@dataclass
class GreekVerse:
    """Represents a Greek verse."""

    verse_id: VerseID
    words: List[GreekWord]
    tei_id: str

    @classmethod
    def from_xml_element(cls, element: ET.Element, book_code: str) -> Optional["GreekVerse"]:
        """Create GreekVerse from XML <ab> element (TEI verse container)."""
        tei_id = element.get("n", "")
        if not tei_id:
            return None

        # Parse TEI verse ID format: "B01K1V1" -> Book 01, Chapter 1, Verse 1
        match = re.match(r"B(\d+)K(\d+)V(\d+)", tei_id)
        if not match:
            return None

        book_num, chapter_num, verse_num = match.groups()
        chapter = int(chapter_num)
        verse = int(verse_num)

        # Create verse ID
        verse_id = create_verse_id(book_code, chapter, verse)
        if not verse_id:
            return None

        # Extract words
        words = []
        # Look for <w> elements both with and without namespace
        word_elements = element.findall(".//w") + element.findall(
            ".//{http://www.tei-c.org/ns/1.0}w"
        )
        for word_elem in word_elements:
            word = GreekWord.from_xml_element(word_elem)
            if word.text.strip():  # Only include non-empty words
                words.append(word)

        return cls(
            verse_id=verse_id,
            words=words,
            tei_id=tei_id,
        )

    def to_dict(self) -> Dict[str, Union[str, List[Dict[str, Optional[str]]]]]:
        """Convert to dictionary format."""
        return {
            "verse_id": str(self.verse_id),
            "tei_id": self.tei_id,
            "words": [word.to_dict() for word in self.words],
        }


class GreekParser:
    """Parser for Greek TEI XML files."""

    def __init__(self) -> None:
        """Initialize the Greek parser."""
        self.namespace = {"tei": "http://www.tei-c.org/ns/1.0"}

    def parse_file(self, file_path: Union[str, Path], book_code: str) -> List[GreekVerse]:
        """
        Parse a Greek XML file and extract verses.

        Args:
            file_path: Path to the Greek XML file
            book_code: Canonical book code (e.g., "MAT")

        Returns:
            List of GreekVerse objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ET.ParseError: If the XML is malformed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Greek XML file not found: {file_path}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find all verse containers using namespace
            verses = []
            # Look for <ab> elements both with and without namespace
            ab_elements = root.findall(".//ab") + root.findall(".//tei:ab", self.namespace)
            for ab_elem in ab_elements:
                verse = GreekVerse.from_xml_element(ab_elem, book_code)
                if verse:
                    verses.append(verse)

            return verses

        except ET.ParseError as e:
            raise ET.ParseError(f"Failed to parse Greek XML file {file_path}: {e}")

    def parse_book(self, book_code: str, data_dir: Union[str, Path]) -> List[GreekVerse]:
        """
        Parse a Greek book by book code.

        Args:
            book_code: 3-letter book code (e.g., "MAT")
            data_dir: Directory containing Greek XML files

        Returns:
            List of GreekVerse objects for the book
        """
        data_dir = Path(data_dir)

        # Map canonical book codes to Greek filenames
        filename_map = {
            "MAT": "MAT.xml",
            "MRK": "MAR.xml",  # Mark uses MAR in the files
            "LUK": "LUK.xml",
            "JHN": "JOH.xml",  # John uses JOH in the files
            "ACT": "ACT.xml",
            "ROM": "ROM.xml",
            "1CO": "1CO.xml",
            "2CO": "2CO.xml",
            "GAL": "GAL.xml",
            "EPH": "EPH.xml",
            "PHP": "PHP.xml",
            "COL": "COL.xml",
            "1TH": "1TH.xml",
            "2TH": "2TH.xml",
            "1TI": "1TI.xml",
            "2TI": "2TI.xml",
            "TIT": "TIT.xml",
            "PHM": "PHM.xml",
            "HEB": "HEB.xml",
            "JAS": "JAM.xml",  # James uses JAM in the files
            "1PE": "1PE.xml",
            "2PE": "2PE.xml",
            "1JN": "1JO.xml",  # 1 John uses 1JO in the files
            "2JN": "2JO.xml",  # 2 John uses 2JO in the files
            "3JN": "3JO.xml",  # 3 John uses 3JO in the files
            "JUD": "JUD.xml",
            "REV": "REV.xml",
        }

        filename = filename_map.get(book_code.upper())
        if not filename:
            raise ValueError(f"Unknown Greek book code: {book_code}")

        file_path = data_dir / filename
        return self.parse_file(file_path, book_code)

    def get_book_statistics(self, verses: List[GreekVerse]) -> Dict[str, int]:
        """
        Get statistics for parsed Greek verses.

        Args:
            verses: List of GreekVerse objects

        Returns:
            Dictionary with statistics
        """
        if not verses:
            return {"verse_count": 0, "word_count": 0}

        word_count = sum(len(verse.words) for verse in verses)

        return {
            "verse_count": len(verses),
            "word_count": word_count,
            "book_code": verses[0].verse_id.book if verses else "unknown",
        }

    def extract_lemmas(self, verses: List[GreekVerse]) -> List[str]:
        """
        Extract all unique lemmas from verses.

        Args:
            verses: List of GreekVerse objects

        Returns:
            Sorted list of unique lemmas
        """
        lemmas = set()
        for verse in verses:
            for word in verse.words:
                if word.lemma:
                    lemmas.add(word.lemma)

        return sorted(lemmas)

    @staticmethod
    def get_all_nt_books() -> List[str]:
        """
        Get list of all New Testament book codes supported by the Greek parser.

        Returns:
            List of canonical book codes
        """
        return [
            "MAT",
            "MRK",
            "LUK",
            "JHN",
            "ACT",
            "ROM",
            "1CO",
            "2CO",
            "GAL",
            "EPH",
            "PHP",
            "COL",
            "1TH",
            "2TH",
            "1TI",
            "2TI",
            "TIT",
            "PHM",
            "HEB",
            "JAS",
            "1PE",
            "2PE",
            "1JN",
            "2JN",
            "3JN",
            "JUD",
            "REV",
        ]
