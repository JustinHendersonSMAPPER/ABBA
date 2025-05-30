#!/usr/bin/env python3
"""
Extract Bible translations from bible.helloao.org SQLite database.
Converts to ABBA canonical JSON format.
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


def calculate_sha256(data: Union[str, bytes]) -> str:
    """Calculate SHA256 hash of data."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# Standard Bible book ID mappings
BOOK_ID_MAPPINGS = {
    # Old Testament
    "GEN": {"names": {"en": {"full": "Genesis", "short": "Gen"}}},
    "EXO": {"names": {"en": {"full": "Exodus", "short": "Exod"}}},
    "LEV": {"names": {"en": {"full": "Leviticus", "short": "Lev"}}},
    "NUM": {"names": {"en": {"full": "Numbers", "short": "Num"}}},
    "DEU": {"names": {"en": {"full": "Deuteronomy", "short": "Deut"}}},
    "JOS": {"names": {"en": {"full": "Joshua", "short": "Josh"}}},
    "JDG": {"names": {"en": {"full": "Judges", "short": "Judg"}}},
    "RUT": {"names": {"en": {"full": "Ruth", "short": "Ruth"}}},
    "1SA": {"names": {"en": {"full": "1 Samuel", "short": "1Sam"}}},
    "2SA": {"names": {"en": {"full": "2 Samuel", "short": "2Sam"}}},
    "1KI": {"names": {"en": {"full": "1 Kings", "short": "1Kgs"}}},
    "2KI": {"names": {"en": {"full": "2 Kings", "short": "2Kgs"}}},
    "1CH": {"names": {"en": {"full": "1 Chronicles", "short": "1Chr"}}},
    "2CH": {"names": {"en": {"full": "2 Chronicles", "short": "2Chr"}}},
    "EZR": {"names": {"en": {"full": "Ezra", "short": "Ezra"}}},
    "NEH": {"names": {"en": {"full": "Nehemiah", "short": "Neh"}}},
    "EST": {"names": {"en": {"full": "Esther", "short": "Esth"}}},
    "JOB": {"names": {"en": {"full": "Job", "short": "Job"}}},
    "PSA": {"names": {"en": {"full": "Psalms", "short": "Ps"}}},
    "PRO": {"names": {"en": {"full": "Proverbs", "short": "Prov"}}},
    "ECC": {"names": {"en": {"full": "Ecclesiastes", "short": "Eccl"}}},
    "SNG": {"names": {"en": {"full": "Song of Solomon", "short": "Song"}}},
    "ISA": {"names": {"en": {"full": "Isaiah", "short": "Isa"}}},
    "JER": {"names": {"en": {"full": "Jeremiah", "short": "Jer"}}},
    "LAM": {"names": {"en": {"full": "Lamentations", "short": "Lam"}}},
    "EZK": {"names": {"en": {"full": "Ezekiel", "short": "Ezek"}}},
    "DAN": {"names": {"en": {"full": "Daniel", "short": "Dan"}}},
    "HOS": {"names": {"en": {"full": "Hosea", "short": "Hos"}}},
    "JOL": {"names": {"en": {"full": "Joel", "short": "Joel"}}},
    "AMO": {"names": {"en": {"full": "Amos", "short": "Amos"}}},
    "OBA": {"names": {"en": {"full": "Obadiah", "short": "Obad"}}},
    "JON": {"names": {"en": {"full": "Jonah", "short": "Jonah"}}},
    "MIC": {"names": {"en": {"full": "Micah", "short": "Mic"}}},
    "NAM": {"names": {"en": {"full": "Nahum", "short": "Nah"}}},
    "HAB": {"names": {"en": {"full": "Habakkuk", "short": "Hab"}}},
    "ZEP": {"names": {"en": {"full": "Zephaniah", "short": "Zeph"}}},
    "HAG": {"names": {"en": {"full": "Haggai", "short": "Hag"}}},
    "ZEC": {"names": {"en": {"full": "Zechariah", "short": "Zech"}}},
    "MAL": {"names": {"en": {"full": "Malachi", "short": "Mal"}}},
    # New Testament
    "MAT": {"names": {"en": {"full": "Matthew", "short": "Matt"}}},
    "MRK": {"names": {"en": {"full": "Mark", "short": "Mark"}}},
    "LUK": {"names": {"en": {"full": "Luke", "short": "Luke"}}},
    "JHN": {"names": {"en": {"full": "John", "short": "John"}}},
    "ACT": {"names": {"en": {"full": "Acts", "short": "Acts"}}},
    "ROM": {"names": {"en": {"full": "Romans", "short": "Rom"}}},
    "1CO": {"names": {"en": {"full": "1 Corinthians", "short": "1Cor"}}},
    "2CO": {"names": {"en": {"full": "2 Corinthians", "short": "2Cor"}}},
    "GAL": {"names": {"en": {"full": "Galatians", "short": "Gal"}}},
    "EPH": {"names": {"en": {"full": "Ephesians", "short": "Eph"}}},
    "PHP": {"names": {"en": {"full": "Philippians", "short": "Phil"}}},
    "COL": {"names": {"en": {"full": "Colossians", "short": "Col"}}},
    "1TH": {"names": {"en": {"full": "1 Thessalonians", "short": "1Thess"}}},
    "2TH": {"names": {"en": {"full": "2 Thessalonians", "short": "2Thess"}}},
    "1TI": {"names": {"en": {"full": "1 Timothy", "short": "1Tim"}}},
    "2TI": {"names": {"en": {"full": "2 Timothy", "short": "2Tim"}}},
    "TIT": {"names": {"en": {"full": "Titus", "short": "Titus"}}},
    "PHM": {"names": {"en": {"full": "Philemon", "short": "Phlm"}}},
    "HEB": {"names": {"en": {"full": "Hebrews", "short": "Heb"}}},
    "JAS": {"names": {"en": {"full": "James", "short": "Jas"}}},
    "1PE": {"names": {"en": {"full": "1 Peter", "short": "1Pet"}}},
    "2PE": {"names": {"en": {"full": "2 Peter", "short": "2Pet"}}},
    "1JN": {"names": {"en": {"full": "1 John", "short": "1John"}}},
    "2JN": {"names": {"en": {"full": "2 John", "short": "2John"}}},
    "3JN": {"names": {"en": {"full": "3 John", "short": "3John"}}},
    "JUD": {"names": {"en": {"full": "Jude", "short": "Jude"}}},
    "REV": {"names": {"en": {"full": "Revelation", "short": "Rev"}}},
}


def clean_verse_text(text: str) -> str:
    """Clean verse text by removing formatting artifacts."""
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def extract_translation(  # pylint: disable=too-many-locals
    conn: sqlite3.Connection, translation_id: str, output_dir: str
) -> Optional[Dict[str, Any]]:
    """Extract a single translation from the database."""
    cursor = conn.cursor()

    # Get translation metadata
    cursor.execute(
        """
        SELECT id, name, englishName, language, website, licenseUrl, licenseNotes
        FROM Translation
        WHERE id = ?
    """,
        (translation_id,),
    )

    trans_row = cursor.fetchone()
    if not trans_row:
        print(f"Translation {translation_id} not found")
        return None

    trans_id, _, english_name, language, website, license_url, license_notes = trans_row

    print(f"\nExtracting {english_name} ({trans_id})...")

    # Create Bible in expected format (from example_structure.json)
    bible_data: Dict[str, Any] = {
        "version": trans_id.upper(),
        "name": english_name,
        "language": language[:2] if len(language) >= 2 else language,  # Convert 'eng' to 'en'
        "copyright": license_notes or "See license URL",
        "source": "bible.helloao.org",
        "website": website,
        "license_url": license_url,
        "books": {},
    }

    # Get all books for this translation
    cursor.execute(
        """
        SELECT id, name, commonName, "order", numberOfChapters
        FROM Book
        WHERE translationId = ?
        ORDER BY "order"
    """,
        (translation_id,),
    )

    books = cursor.fetchall()

    for book_id, book_name, _, order, num_chapters in books:
        # Map to standard book ID if possible
        standard_book_id = book_id.upper()
        book_abbr = None

        if standard_book_id not in BOOK_ID_MAPPINGS:
            # Try to match by name
            for std_id, std_info in BOOK_ID_MAPPINGS.items():
                if book_name.lower() == std_info["names"]["en"]["full"].lower():
                    standard_book_id = std_id
                    book_abbr = std_info["names"]["en"]["short"]
                    break
        else:
            book_abbr = BOOK_ID_MAPPINGS[standard_book_id]["names"]["en"]["short"]

        if standard_book_id not in BOOK_ID_MAPPINGS:
            print(f"  Warning: Unknown book {book_id} ({book_name}) - skipping")
            continue

        # Determine testament
        testament = "OT" if order <= 39 else "NT"

        # Create book structure matching example_structure.json
        book_data: Dict[str, Any] = {
            "name": book_name,
            "abbr": book_abbr,
            "testament": testament,
            "chapters": []
        }

        # Get all chapters for this book
        for chapter_num in range(1, num_chapters + 1):
            chapter_data: Dict[str, Any] = {"chapter": chapter_num, "verses": []}

            # Get verses for this chapter
            cursor.execute(
                """
                SELECT number, text
                FROM ChapterVerse
                WHERE translationId = ? AND bookId = ? AND chapterNumber = ?
                ORDER BY number
            """,
                (translation_id, book_id, chapter_num),
            )

            verses = cursor.fetchall()

            for verse_num, verse_text in verses:
                verse_data = {"verse": verse_num, "text": clean_verse_text(verse_text)}
                chapter_data["verses"].append(verse_data)

            if chapter_data["verses"]:
                book_data["chapters"].append(chapter_data)

        if book_data["chapters"]:
            bible_data["books"][book_abbr] = book_data

    # Save the translation
    filename = f"{trans_id.lower()}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(bible_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {filepath}")
    print(f"  Extracted {len(bible_data['books'])} books")

    # Calculate file hash
    with open(filepath, "rb") as f:
        file_hash = calculate_sha256(f.read())

    return {
        "id": trans_id.upper(),
        "filename": filename,
        "name": english_name,
        "language": language[:2] if len(language) >= 2 else language,
        "sha256": file_hash,
        "size": os.path.getsize(filepath),
        "books_count": len(bible_data["books"]),
        "copyright": license_notes or "See license URL",
        "website": website,
        "license_url": license_url,
    }


def update_manifest(extracted_translations: List[Dict[str, Any]], output_dir: str) -> None:
    """Update the manifest with extracted translations."""
    manifest_path = os.path.join(output_dir, "manifest.json")

    manifest: Dict[str, Any] = {
        "description": "Bible translations extracted from bible.helloao.org",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "format_version": "1.0.0",
        "source": "https://bible.helloao.org/",
        "versions": {},
    }

    for trans_info in extracted_translations:
        manifest["versions"][trans_info["id"]] = {
            "filename": trans_info["filename"],
            "name": trans_info["name"],
            "language": trans_info["language"],
            "copyright": trans_info["copyright"],
            "website": trans_info.get("website", ""),
            "license_url": trans_info.get("license_url", ""),
            "sha256": trans_info["sha256"],
            "file_size": trans_info["size"],
            "books_count": trans_info["books_count"],
            "format": "bible_gateway",  # Changed from "canonical" to match example format
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nUpdated manifest with {len(manifest['versions'])} translations")


def main() -> None:
    """Main function to extract Bible translations from SQLite database."""
    parser = argparse.ArgumentParser(description="Extract Bible translations from SQLite database")
    parser.add_argument("--db", default="processing/bible.db", help="Path to SQLite database")
    parser.add_argument("--output", default="data/sources/translations", help="Output directory")
    parser.add_argument("--translations", nargs="+", help="Specific translation IDs to extract")
    parser.add_argument("--languages", nargs="+", help="Extract translations in these languages")
    parser.add_argument("--limit", type=int, help="Limit number of translations to extract")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    # Build query for translations
    query = "SELECT id, englishName, language FROM Translation WHERE 1=1"
    params = []

    if args.translations:
        placeholders = ",".join(["?" for _ in args.translations])
        query += f" AND id IN ({placeholders})"
        params.extend(args.translations)

    if args.languages:
        lang_placeholders = ",".join(["?" for _ in args.languages])
        query += f" AND language IN ({lang_placeholders})"
        params.extend(args.languages)

    query += " ORDER BY language, englishName"

    if args.limit:
        query += f" LIMIT {args.limit}"

    # Get translations to extract
    cursor.execute(query, params)
    translations = cursor.fetchall()

    print(f"Found {len(translations)} translations to extract")

    extracted = []

    # Extract each translation
    for trans_id, _, _ in translations:
        try:
            result = extract_translation(conn, trans_id, args.output)
            if result:
                extracted.append(result)
        except (sqlite3.Error, IOError, OSError) as e:
            print(f"Error extracting {trans_id}: {e}")

    # Update manifest
    if extracted:
        update_manifest(extracted, args.output)
        print(f"\nSuccessfully extracted {len(extracted)} translations")

    conn.close()


if __name__ == "__main__":
    main()
