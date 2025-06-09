"""
Minimal JSON export for ABBA data.

A dependency-minimal implementation for creating JSON files
from biblical verse data, optimized for CDN distribution.
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set


@dataclass
class MinimalVerse:
    """Minimal verse representation."""

    book: str
    chapter: int
    verse: int
    text: str
    translation: str = "KJV"


def export_to_json_files(
    verses: List[MinimalVerse],
    output_dir: str,
    by_book: bool = True,
    by_chapter: bool = True,
    single_file: bool = False,
) -> Dict[str, int]:
    """
    Export verses to JSON files.

    Args:
        verses: List of verses to export
        output_dir: Directory to write JSON files
        by_book: Create separate files for each book
        by_chapter: Create separate files for each chapter
        single_file: Create a single JSON file with all verses

    Returns:
        Dictionary with export statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, any] = {
        "total_verses": len(verses),
        "files_created": 0,
        "books": set(),
        "chapters": set(),
    }

    if single_file:
        # Export all verses to a single file
        all_data = {
            "metadata": {
                "verse_count": len(verses),
                "translation": verses[0].translation if verses else "KJV",
            },
            "verses": [asdict(v) for v in verses],
        }

        output_file = output_path / "verses.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        stats["files_created"] = 1

    else:
        # Organize verses by book and chapter
        organized: DefaultDict[str, DefaultDict[int, List[MinimalVerse]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for verse in verses:
            organized[verse.book][verse.chapter].append(verse)
            stats["books"].add(verse.book)  # type: ignore
            stats["chapters"].add(f"{verse.book}_{verse.chapter}")  # type: ignore

        # Create index file
        index_data = {
            "books": sorted(list(stats["books"])),  # type: ignore
            "verse_count": len(verses),
            "structure": {},
        }

        # Export by book/chapter structure
        for book, chapters in organized.items():
            book_dir = output_path / book
            book_dir.mkdir(exist_ok=True)

            book_data = {
                "book": book,
                "chapters": sorted(chapters.keys()),
                "verse_count": sum(len(verses) for verses in chapters.values()),
            }

            index_data["structure"][book] = {
                "chapters": book_data["chapters"],
                "verse_count": book_data["verse_count"],
            }

            if by_book and not by_chapter:
                # Single file per book
                book_verses = []
                for chapter in sorted(chapters.keys()):
                    book_verses.extend(chapters[chapter])

                book_file = book_dir / f"{book}.json"
                with open(book_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {"book": book, "verses": [asdict(v) for v in book_verses]},
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                stats["files_created"] += 1

            elif by_chapter:
                # Separate file per chapter
                for chapter, chapter_verses in chapters.items():
                    chapter_file = book_dir / f"{book}_{chapter}.json"
                    with open(chapter_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "book": book,
                                "chapter": chapter,
                                "verses": [asdict(v) for v in chapter_verses],
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

                    stats["files_created"] += 1

        # Write index file
        index_file = output_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        stats["files_created"] += 1

    # Convert sets to counts for JSON serialization
    stats["book_count"] = len(stats["books"])  # type: ignore
    stats["chapter_count"] = len(stats["chapters"])  # type: ignore
    del stats["books"]
    del stats["chapters"]

    return {k: v for k, v in stats.items() if isinstance(v, int)}  # type: ignore


def load_json_verses(json_path: str) -> List[Dict]:
    """
    Load verses from a JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        List of verse dictionaries
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "verses" in data:
        return data["verses"]
    else:
        # Assume it's a list of verses
        return data


def create_verse_lookup(verses: List[MinimalVerse]) -> Dict[str, MinimalVerse]:
    """
    Create a lookup dictionary for quick verse access.

    Args:
        verses: List of verses

    Returns:
        Dictionary mapping verse ID to verse
    """
    lookup = {}
    for verse in verses:
        key = f"{verse.book}_{verse.chapter}_{verse.verse}"
        lookup[key] = verse
    return lookup


class MinimalJSONExporter:
    """Simple JSON exporter for verse data."""
    
    def __init__(self, output_path: str):
        """Initialize exporter with output path."""
        self.output_path = Path(output_path)
        self.verses: List[Dict[str, any]] = []
    
    def add_verse(self, verse_id: str, book: str, chapter: int, 
                  verse: int, text: str, translation: str = "KJV") -> None:
        """Add a verse to the export."""
        self.verses.append({
            "verse_id": verse_id,
            "book": book,
            "chapter": chapter,
            "verse": verse,
            "text": text,
            "translation": translation
        })
    
    def finalize(self) -> None:
        """Write verses to JSON file."""
        # Ensure output directory exists
        if self.output_path.suffix not in ['.json', '.JSON']:
            # If output_path is a directory, create verses.json inside it
            self.output_path.mkdir(parents=True, exist_ok=True)
            output_file = self.output_path / "verses.json"
        else:
            # If output_path is a file, ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = self.output_path
        
        # Write JSON data
        data = {
            "metadata": {
                "verse_count": len(self.verses),
                "format_version": "1.0"
            },
            "verses": self.verses
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()
