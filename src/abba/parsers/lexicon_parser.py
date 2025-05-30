"""
Strong's lexicon parser.

This module parses Strong's Hebrew and Greek lexicon XML files and extracts
word definitions, glosses, and morphological information.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class LexiconEntry:
    """Represents a Strong's lexicon entry."""

    strong_number: str
    word: str
    transliteration: Optional[str] = None
    pronunciation: Optional[str] = None
    definition: Optional[str] = None
    gloss: Optional[str] = None
    language: str = "hebrew"  # "hebrew" or "greek"
    morphology: Optional[str] = None
    etymology: Optional[str] = None
    usage_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary format."""
        return {
            "strong_number": self.strong_number,
            "word": self.word,
            "transliteration": self.transliteration,
            "pronunciation": self.pronunciation,
            "definition": self.definition,
            "gloss": self.gloss,
            "language": self.language,
            "morphology": self.morphology,
            "etymology": self.etymology,
            "usage_notes": self.usage_notes,
        }


class LexiconParser:
    """Parser for Strong's lexicon XML files."""

    def __init__(self) -> None:
        """Initialize the lexicon parser."""
        self.namespace = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

    def parse_file(
        self, file_path: Union[str, Path], language: str = "hebrew"
    ) -> List[LexiconEntry]:
        """
        Parse a Strong's lexicon XML file.

        Args:
            file_path: Path to the lexicon XML file
            language: Language of the lexicon ("hebrew" or "greek")

        Returns:
            List of LexiconEntry objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ET.ParseError: If the XML is malformed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Lexicon XML file not found: {file_path}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            entries = []
            for div_elem in root.findall(".//osis:div[@type='entry']", self.namespace):
                entry = self._parse_entry_element(div_elem, language)
                if entry:
                    entries.append(entry)

            return entries

        except ET.ParseError as e:
            raise ET.ParseError(f"Failed to parse lexicon XML file {file_path}: {e}")

    def _parse_entry_element(self, div_elem: ET.Element, language: str) -> Optional[LexiconEntry]:
        """Parse a lexicon entry from a div element."""
        # Get Strong's number from the div's "n" attribute
        strong_num = div_elem.get("n")
        if not strong_num:
            return None

        # Format Strong's number with prefix
        prefix = "H" if language == "hebrew" else "G"
        strong_number = f"{prefix}{strong_num}"

        # Find the main word element
        word_elem = div_elem.find(".//osis:w", self.namespace)
        if word_elem is None:
            return None

        word = word_elem.text or ""
        transliteration = word_elem.get("xlit")
        pronunciation = word_elem.get("POS")
        morphology = word_elem.get("morph")

        # Extract definition from list items
        definition_parts = []
        for item_elem in div_elem.findall(".//osis:item", self.namespace):
            if item_elem.text:
                definition_parts.append(item_elem.text.strip())

        definition = "; ".join(definition_parts) if definition_parts else None

        # Extract gloss from note elements
        gloss = None
        for note_elem in div_elem.findall(".//osis:note[@type='explanation']", self.namespace):
            if note_elem.text:
                # Remove curly braces from gloss
                gloss_text = note_elem.text.strip()
                if gloss_text.startswith("{") and gloss_text.endswith("}"):
                    gloss = gloss_text[1:-1]
                else:
                    gloss = gloss_text
                break

        # Extract etymology from exegesis notes
        etymology = None
        for note_elem in div_elem.findall(".//osis:note[@type='exegesis']", self.namespace):
            if note_elem.text:
                etymology = note_elem.text.strip()
                break

        # Extract usage notes from translation notes
        usage_notes = None
        for note_elem in div_elem.findall(".//osis:note[@type='translation']", self.namespace):
            if note_elem.text:
                usage_notes = note_elem.text.strip()
                break

        return LexiconEntry(
            strong_number=strong_number,
            word=word,
            transliteration=transliteration,
            pronunciation=pronunciation,
            definition=definition,
            gloss=gloss,
            language=language,
            morphology=morphology,
            etymology=etymology,
            usage_notes=usage_notes,
        )

    def parse_hebrew_lexicon(self, data_dir: Union[str, Path]) -> List[LexiconEntry]:
        """
        Parse the Hebrew Strong's lexicon.

        Args:
            data_dir: Directory containing lexicon files

        Returns:
            List of Hebrew LexiconEntry objects
        """
        data_dir = Path(data_dir)
        hebrew_file = data_dir / "strongs_hebrew.xml"
        return self.parse_file(hebrew_file, "hebrew")

    def parse_greek_lexicon(self, data_dir: Union[str, Path]) -> List[LexiconEntry]:
        """
        Parse the Greek Strong's lexicon.

        Args:
            data_dir: Directory containing lexicon files

        Returns:
            List of Greek LexiconEntry objects
        """
        data_dir = Path(data_dir)
        greek_file = data_dir / "strongs_greek.xml"
        return self.parse_file(greek_file, "greek")

    def parse_both_lexicons(self, data_dir: Union[str, Path]) -> Dict[str, List[LexiconEntry]]:
        """
        Parse both Hebrew and Greek lexicons.

        Args:
            data_dir: Directory containing lexicon files

        Returns:
            Dictionary with "hebrew" and "greek" keys containing lexicon entries
        """
        return {
            "hebrew": self.parse_hebrew_lexicon(data_dir),
            "greek": self.parse_greek_lexicon(data_dir),
        }

    def get_lexicon_statistics(self, entries: List[LexiconEntry]) -> Dict[str, int]:
        """
        Get statistics for lexicon entries.

        Args:
            entries: List of LexiconEntry objects

        Returns:
            Dictionary with statistics
        """
        if not entries:
            return {
                "total_entries": 0,
                "entries_with_definitions": 0,
                "entries_with_glosses": 0,
                "entries_with_etymology": 0,
            }

        entries_with_definitions = sum(1 for entry in entries if entry.definition)
        entries_with_glosses = sum(1 for entry in entries if entry.gloss)
        entries_with_etymology = sum(1 for entry in entries if entry.etymology)

        return {
            "total_entries": len(entries),
            "entries_with_definitions": entries_with_definitions,
            "entries_with_glosses": entries_with_glosses,
            "entries_with_etymology": entries_with_etymology,
            "language": entries[0].language if entries else "unknown",
        }

    def create_strong_lookup(self, entries: List[LexiconEntry]) -> Dict[str, LexiconEntry]:
        """
        Create a lookup dictionary by Strong's number.

        Args:
            entries: List of LexiconEntry objects

        Returns:
            Dictionary mapping Strong's numbers to LexiconEntry objects
        """
        return {entry.strong_number: entry for entry in entries}

    def search_by_word(self, entries: List[LexiconEntry], search_term: str) -> List[LexiconEntry]:
        """
        Search lexicon entries by word or transliteration.

        Args:
            entries: List of LexiconEntry objects
            search_term: Term to search for

        Returns:
            List of matching LexiconEntry objects
        """
        search_term = search_term.lower()
        matches = []

        for entry in entries:
            if (
                search_term in entry.word.lower()
                or (entry.transliteration and search_term in entry.transliteration.lower())
                or (entry.gloss and search_term in entry.gloss.lower())
            ):
                matches.append(entry)

        return matches


@dataclass
class StrongsLexicon:
    """Container for Strong's Hebrew and Greek lexicon data."""

    hebrew_entries: Dict[str, LexiconEntry]
    greek_entries: Dict[str, LexiconEntry]

    @classmethod
    def from_directory(cls, data_dir: Union[str, Path]) -> "StrongsLexicon":
        """
        Load Strong's lexicon from a directory.

        Args:
            data_dir: Directory containing lexicon files

        Returns:
            StrongsLexicon instance
        """
        parser = LexiconParser()
        lexicons = parser.parse_both_lexicons(data_dir)

        hebrew_dict = parser.create_strong_lookup(lexicons["hebrew"])
        greek_dict = parser.create_strong_lookup(lexicons["greek"])

        return cls(hebrew_entries=hebrew_dict, greek_entries=greek_dict)

    def get_entry(self, strong_number: str) -> Optional[LexiconEntry]:
        """Get entry by Strong's number."""
        if strong_number.startswith("H"):
            return self.hebrew_entries.get(strong_number)
        elif strong_number.startswith("G"):
            return self.greek_entries.get(strong_number)
        return None
