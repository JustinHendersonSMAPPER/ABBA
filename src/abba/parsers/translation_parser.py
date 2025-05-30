"""
Translation JSON parser and normalizer.

This module parses various Bible translation JSON files and normalizes them
to a standard format with canonical book codes and verse references.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..book_codes import normalize_book_name
from ..verse_id import VerseID, create_verse_id


@dataclass
class TranslationVerse:
    """Represents a verse in a translation."""

    verse_id: VerseID
    text: str
    original_book_name: str
    original_chapter: int
    original_verse: int

    def to_dict(self) -> Dict[str, Union[str, int]]:
        """Convert to dictionary format."""
        return {
            "verse_id": str(self.verse_id),
            "text": self.text,
            "original_book_name": self.original_book_name,
            "original_chapter": self.original_chapter,
            "original_verse": self.original_verse,
        }


@dataclass
class Translation:
    """Represents a complete Bible translation."""

    version: str
    name: str
    language: str
    copyright: Optional[str] = None
    source: Optional[str] = None
    website: Optional[str] = None
    license_url: Optional[str] = None
    verses: Optional[List[TranslationVerse]] = None

    def __post_init__(self) -> None:
        """Initialize verses list if not provided."""
        if self.verses is None:
            self.verses = []

    def to_dict(self) -> Dict[str, Union[str, int, List[Dict[str, Union[str, int]]], None]]:
        """Convert to dictionary format."""
        return {
            "version": self.version,
            "name": self.name,
            "language": self.language,
            "copyright": self.copyright,
            "source": self.source,
            "website": self.website,
            "license_url": self.license_url,
            "verse_count": len(self.verses),
            "verses": [verse.to_dict() for verse in self.verses],
        }


class TranslationParser:
    """Parser for Bible translation JSON files."""

    def __init__(self) -> None:
        """Initialize the translation parser."""
        pass

    def parse_file(self, file_path: Union[str, Path]) -> Translation:
        """
        Parse a translation JSON file.

        Args:
            file_path: Path to the translation JSON file

        Returns:
            Translation object

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the JSON is malformed
            ValueError: If required fields are missing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Translation JSON file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to parse JSON file {file_path}: {e}", e.doc, e.pos)

        # Extract metadata
        version = data.get("version", "")
        name = data.get("name", "")
        language = data.get("language", "")

        if not version or not name:
            raise ValueError(f"Missing required fields (version, name) in {file_path}")

        translation = Translation(
            version=version,
            name=name,
            language=language,
            copyright=data.get("copyright"),
            source=data.get("source"),
            website=data.get("website"),
            license_url=data.get("license_url"),
        )

        # Parse verses
        books_data = data.get("books", {})
        for original_book_name, book_data in books_data.items():
            # Normalize book name to canonical code
            canonical_book = normalize_book_name(original_book_name)
            if not canonical_book:
                # Skip unknown books (e.g., deuterocanonical books not in Protestant canon)
                continue

            chapters = book_data.get("chapters", [])
            for chapter_data in chapters:
                chapter_num = chapter_data.get("chapter", 0)
                verses = chapter_data.get("verses", [])

                for verse_data in verses:
                    verse_num = verse_data.get("verse", 0)
                    verse_text = verse_data.get("text", "")

                    if chapter_num > 0 and verse_num > 0 and verse_text:
                        # Create canonical verse ID
                        verse_id = create_verse_id(canonical_book, chapter_num, verse_num)
                        if verse_id:
                            translation_verse = TranslationVerse(
                                verse_id=verse_id,
                                text=verse_text,
                                original_book_name=original_book_name,
                                original_chapter=chapter_num,
                                original_verse=verse_num,
                            )
                            translation.verses.append(translation_verse)

        return translation

    def parse_directory(
        self, data_dir: Union[str, Path], limit: Optional[int] = None
    ) -> List[Translation]:
        """
        Parse all translation JSON files in a directory.

        Args:
            data_dir: Directory containing translation JSON files
            limit: Optional limit on number of files to parse

        Returns:
            List of Translation objects
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Translation directory not found: {data_dir}")

        json_files = list(data_dir.glob("*.json"))
        if limit:
            json_files = json_files[:limit]

        translations = []
        for json_file in json_files:
            try:
                translation = self.parse_file(json_file)
                translations.append(translation)
            except (json.JSONDecodeError, ValueError) as e:
                # Log error but continue with other files
                print(f"Warning: Skipping {json_file}: {e}")
                continue

        return translations

    def get_translation_statistics(
        self, translation: Translation
    ) -> Dict[str, Union[str, int, float]]:
        """
        Get statistics for a translation.

        Args:
            translation: Translation object

        Returns:
            Dictionary with statistics
        """
        if not translation.verses:
            return {
                "version": translation.version,
                "name": translation.name,
                "language": translation.language,
                "verse_count": 0,
                "book_count": 0,
                "avg_verse_length": 0,
            }

        # Count unique books
        books = set(verse.verse_id.book for verse in translation.verses)

        # Calculate average verse length
        total_length = sum(len(verse.text) for verse in translation.verses)
        avg_length = total_length / len(translation.verses) if translation.verses else 0

        return {
            "version": translation.version,
            "name": translation.name,
            "language": translation.language,
            "verse_count": len(translation.verses),
            "book_count": len(books),
            "avg_verse_length": round(avg_length, 1),
        }

    def normalize_book_names(self, translation_data: Dict) -> Dict[str, Optional[str]]:
        """
        Extract and normalize all book names from translation data.

        Args:
            translation_data: Raw translation JSON data

        Returns:
            Dictionary mapping original book names to canonical codes
        """
        book_mapping = {}
        books_data = translation_data.get("books", {})

        for original_book_name in books_data.keys():
            canonical_book = normalize_book_name(original_book_name)
            book_mapping[original_book_name] = canonical_book

        return book_mapping

    def extract_verse_variations(self, translations: List[Translation]) -> Dict[str, List[str]]:
        """
        Extract verse variations across translations for comparison.

        Args:
            translations: List of Translation objects

        Returns:
            Dictionary mapping verse IDs to list of texts across translations
        """
        verse_variations: Dict[str, List[str]] = {}

        for translation in translations:
            for verse in translation.verses:
                verse_key = str(verse.verse_id)
                if verse_key not in verse_variations:
                    verse_variations[verse_key] = []
                verse_variations[verse_key].append(verse.text)

        return verse_variations
