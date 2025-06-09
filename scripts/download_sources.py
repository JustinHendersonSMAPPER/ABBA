#!/usr/bin/env python3
"""
Download and extract only the necessary source files for the ABBA project.
Calculates SHA256 hashes of the actual data files used.
"""
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# For Python < 3.11 compatibility
UTC = timezone.utc


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_and_extract_hebrew() -> None:
    """Download and extract Hebrew Bible XML files."""
    print("Downloading Open Scriptures Hebrew Bible...")
    url = "https://github.com/openscriptures/morphhb/archive/refs/heads/master.zip"

    # Download
    zip_path = "temp_hebrew.zip"
    urllib.request.urlretrieve(url, zip_path)

    # Extract only the XML files
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".xml") and "/wlc/" in file:
                # Extract to hebrew directory, removing the path prefix
                target = os.path.join("data/sources/hebrew", os.path.basename(file))
                with zip_ref.open(file) as source, open(target, "wb") as target_file:
                    shutil.copyfileobj(source, target_file)

    os.remove(zip_path)
    print(f"  Extracted {len(os.listdir('data/sources/hebrew'))} Hebrew Bible books")


def download_and_extract_greek() -> None:
    """Download and extract Greek NT files."""
    print("Downloading Byzantine Majority Text...")
    url = "https://github.com/byztxt/byzantine-majority-text/archive/refs/heads/master.zip"

    # Download
    zip_path = "temp_greek.zip"
    urllib.request.urlretrieve(url, zip_path)

    # Extract XML files from the no-accents directory (easier to work with)
    extracted_count = 0
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            # Extract individual book XML files (not the combined BYZ file)
            if (
                file.endswith(".xml")
                and "/no-accents/" in file
                and "/BYZ/" not in file
                and os.path.basename(file) != "byz.xml"
            ):
                # Extract to greek directory
                target = os.path.join("data/sources/greek", os.path.basename(file))
                with zip_ref.open(file) as source, open(target, "wb") as target_file:
                    shutil.copyfileobj(source, target_file)
                extracted_count += 1

    os.remove(zip_path)
    print(f"  Extracted {extracted_count} Greek NT books")






def generate_manifest() -> Dict[str, Any]:
    """Generate manifest with SHA256 hashes and metadata."""
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "sources": {}
    }

    # Hash Hebrew files
    hebrew_files = {}
    for file in sorted(os.listdir("data/sources/hebrew")):
        filepath = os.path.join("data/sources/hebrew", file)
        hebrew_files[file] = calculate_sha256(filepath)

    manifest["sources"]["hebrew"] = {
        "name": "Open Scriptures Hebrew Bible",
        "license": "CC BY 4.0",
        "url": "https://github.com/openscriptures/morphhb",
        "files": hebrew_files,
        "total_files": len(hebrew_files),
    }

    # Hash Greek files
    greek_files = {}
    for file in sorted(os.listdir("data/sources/greek")):
        filepath = os.path.join("data/sources/greek", file)
        greek_files[file] = calculate_sha256(filepath)

    manifest["sources"]["greek"] = {
        "name": "Byzantine Majority Text",
        "license": "Public Domain",
        "url": "https://github.com/byztxt/byzantine-majority-text",
        "files": greek_files,
        "total_files": len(greek_files),
    }


    # Hash translation files
    translation_files = {}
    if os.path.exists("data/sources/translations"):
        for file in sorted(os.listdir("data/sources/translations")):
            filepath = os.path.join("data/sources/translations", file)
            translation_files[file] = calculate_sha256(filepath)

    manifest["sources"]["translations"] = {
        "name": "Public Domain Bible Translations",
        "license": "Public Domain",
        "url": "https://github.com/scrollmapper/bible_databases",
        "files": translation_files,
        "total_files": len(translation_files),
        "versions": {
            "web.json": "World English Bible",
            "kjv.json": "King James Version (1769)",
            "asv.json": "American Standard Version (1901)",
            "ylt.json": "Young's Literal Translation (1898)",
        },
    }
    
    # Hash Hebrew morphology files
    hebrew_morph_files = {}
    hebrew_morph_dir = "data/sources/morphology/hebrew"
    if os.path.exists(hebrew_morph_dir):
        for file in sorted(os.listdir(hebrew_morph_dir)):
            filepath = os.path.join(hebrew_morph_dir, file)
            hebrew_morph_files[file] = calculate_sha256(filepath)

    manifest["sources"]["hebrew_morphology"] = {
        "name": "Open Scriptures Hebrew Bible Morphology",
        "license": "CC BY 4.0",
        "url": "https://github.com/openscriptures/morphhb",
        "files": hebrew_morph_files,
        "total_files": len(hebrew_morph_files),
    }
    
    # Hash Greek morphology files
    greek_morph_files = {}
    greek_morph_dir = "data/sources/morphology/greek"
    if os.path.exists(greek_morph_dir):
        for file in sorted(os.listdir(greek_morph_dir)):
            filepath = os.path.join(greek_morph_dir, file)
            greek_morph_files[file] = calculate_sha256(filepath)

    manifest["sources"]["greek_morphology"] = {
        "name": "MorphGNT - Morphological Greek New Testament",
        "license": "CC BY-SA 3.0",
        "url": "https://github.com/morphgnt/sblgnt",
        "files": greek_morph_files,
        "total_files": len(greek_morph_files),
    }

    # Save manifest
    with open("data/sources/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nManifest generated at data/sources/manifest.json")
    return manifest


def download_morphgnt() -> None:
    """Download MorphGNT morphological data for Greek New Testament."""
    print("Downloading MorphGNT morphological data...")
    
    # Create morphology directory
    morph_dir = Path("data/sources/morphology/greek")
    morph_dir.mkdir(parents=True, exist_ok=True)
    
    # MorphGNT provides data per book in the root directory
    base_url = "https://raw.githubusercontent.com/morphgnt/sblgnt/master/"
    
    # NT book codes and names used by MorphGNT
    books = [
        ("61-Mt-morphgnt.txt", "matthew"),
        ("62-Mk-morphgnt.txt", "mark"),
        ("63-Lk-morphgnt.txt", "luke"),
        ("64-Jn-morphgnt.txt", "john"),
        ("65-Ac-morphgnt.txt", "acts"),
        ("66-Ro-morphgnt.txt", "romans"),
        ("67-1Co-morphgnt.txt", "1corinthians"),
        ("68-2Co-morphgnt.txt", "2corinthians"),
        ("69-Ga-morphgnt.txt", "galatians"),
        ("70-Eph-morphgnt.txt", "ephesians"),
        ("71-Php-morphgnt.txt", "philippians"),
        ("72-Col-morphgnt.txt", "colossians"),
        ("73-1Th-morphgnt.txt", "1thessalonians"),
        ("74-2Th-morphgnt.txt", "2thessalonians"),
        ("75-1Ti-morphgnt.txt", "1timothy"),
        ("76-2Ti-morphgnt.txt", "2timothy"),
        ("77-Tit-morphgnt.txt", "titus"),
        ("78-Phm-morphgnt.txt", "philemon"),
        ("79-Heb-morphgnt.txt", "hebrews"),
        ("80-Jas-morphgnt.txt", "james"),
        ("81-1Pe-morphgnt.txt", "1peter"),
        ("82-2Pe-morphgnt.txt", "2peter"),
        ("83-1Jn-morphgnt.txt", "1john"),
        ("84-2Jn-morphgnt.txt", "2john"),
        ("85-3Jn-morphgnt.txt", "3john"),
        ("86-Jud-morphgnt.txt", "jude"),
        ("87-Re-morphgnt.txt", "revelation")
    ]
    
    downloaded = 0
    for filename, book_name in books:
        url = f"{base_url}{filename}"
        local_filename = morph_dir / f"{book_name}.txt"
        
        try:
            urllib.request.urlretrieve(url, local_filename)
            downloaded += 1
        except Exception as e:
            print(f"  Warning: Failed to download {book_name}: {e}")
    
    print(f"  Downloaded {downloaded} MorphGNT book files")


def convert_morphgnt_to_json() -> None:
    """Convert MorphGNT text format to JSON."""
    print("Converting MorphGNT data to JSON...")
    
    morph_dir = Path("data/sources/morphology/greek")
    if not morph_dir.exists():
        print("  Greek morphology directory not found")
        return
    
    converted_count = 0
    for txt_file in sorted(morph_dir.glob("*.txt")):
        try:
            book_name = txt_file.stem
            verses = []
            current_verse = None
            current_verse_id = None
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # MorphGNT format: bcv pos parse word norm lemma
                    # Example: 610101 N- ----NSF- Βίβλος Βίβλος βίβλος
                    parts = line.split()
                    if len(parts) >= 6:
                        bcv = parts[0]
                        # Parse book/chapter/verse (61 01 01)
                        book_num = bcv[:2]
                        chapter = int(bcv[2:4])
                        verse = int(bcv[4:6])
                        verse_id = f"{book_name}.{chapter}.{verse}"
                        
                        # Create new verse if needed
                        if verse_id != current_verse_id:
                            if current_verse and current_verse['words']:
                                verses.append(current_verse)
                            current_verse = {
                                'verse_id': verse_id,
                                'chapter': chapter,
                                'verse': verse,
                                'words': []
                            }
                            current_verse_id = verse_id
                        
                        # Parse morphology
                        word_data = {
                            'text': parts[3],  # surface form
                            'normalized': parts[4],  # normalized form
                            'lemma': parts[5],
                            'pos': parts[1],  # part of speech
                            'parse': parts[2],  # full morphological parse
                        }
                        
                        # Parse the morphology code (e.g., ----NSF-)
                        parse_code = parts[2]
                        if len(parse_code) >= 8:
                            morph_features = {
                                'person': parse_code[0] if parse_code[0] != '-' else None,
                                'tense': parse_code[1] if parse_code[1] != '-' else None,
                                'voice': parse_code[2] if parse_code[2] != '-' else None,
                                'mood': parse_code[3] if parse_code[3] != '-' else None,
                                'case': parse_code[4] if parse_code[4] != '-' else None,
                                'number': parse_code[5] if parse_code[5] != '-' else None,
                                'gender': parse_code[6] if parse_code[6] != '-' else None,
                                'degree': parse_code[7] if parse_code[7] != '-' else None,
                            }
                            # Remove None values
                            word_data['morph_features'] = {k: v for k, v in morph_features.items() if v}
                        
                        if current_verse:
                            current_verse['words'].append(word_data)
            
            # Add last verse
            if current_verse and current_verse['words']:
                verses.append(current_verse)
            
            # Save as JSON
            json_path = morph_dir / f"{book_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'book': book_name,
                    'language': 'greek',
                    'verses': verses
                }, f, ensure_ascii=False, indent=2)
            
            # Remove the original txt file
            txt_file.unlink()
            converted_count += 1
            
        except Exception as e:
            print(f"  Error converting {txt_file.name}: {e}")
    
    print(f"  Converted {converted_count} Greek morphology files to JSON")


def convert_oshb_to_json() -> None:
    """Convert OSHB XML morphological data to JSON format."""
    print("Converting OSHB morphological data to JSON...")
    
    hebrew_dir = Path("data/sources/hebrew")
    morph_dir = Path("data/sources/morphology/hebrew")
    morph_dir.mkdir(parents=True, exist_ok=True)
    
    if not hebrew_dir.exists():
        print("  Hebrew source directory not found")
        return
    
    # Define namespaces used in OSHB
    namespaces = {
        'osis': 'http://www.bibletechnologies.net/2003/OSIS/namespace',
        'xml': 'http://www.w3.org/XML/1998/namespace'
    }
    
    converted_count = 0
    for xml_file in sorted(hebrew_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract book name from file
            book_name = xml_file.stem
            
            # Parse verses and morphology
            verses = []
            
            # Find all verses
            for verse in root.findall('.//osis:verse', namespaces):
                verse_data = {
                    'osisID': verse.get('osisID', ''),
                    'words': []
                }
                
                # Extract words with morphological data
                for w in verse.findall('.//osis:w', namespaces):
                    word_data = {
                        'text': w.text or '',
                        'lemma': w.get('lemma', ''),
                        'morph': w.get('morph', ''),
                        'id': w.get('ID', ''),
                    }
                    
                    # Add language if present
                    lang = w.get('{http://www.w3.org/XML/1998/namespace}lang')
                    if lang:
                        word_data['lang'] = lang
                    
                    # Add any other relevant attributes
                    for attr in ['POS', 'person', 'gender', 'number', 'state', 'type']:
                        if w.get(attr):
                            word_data[attr] = w.get(attr)
                    
                    if word_data['text']:  # Only add if has text
                        verse_data['words'].append(word_data)
                
                if verse_data['words']:  # Only add if has words
                    verses.append(verse_data)
            
            # Save as JSON
            json_path = morph_dir / f"{book_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'book': book_name,
                    'language': 'hebrew',
                    'verses': verses
                }, f, ensure_ascii=False, indent=2)
            
            converted_count += 1
            
        except Exception as e:
            print(f"  Error converting {xml_file.name}: {e}")
    
    print(f"  Converted {converted_count} Hebrew morphology files to JSON")


def main() -> None:
    """Main download process."""
    os.makedirs("data/sources/hebrew", exist_ok=True)
    os.makedirs("data/sources/greek", exist_ok=True)
    os.makedirs("data/sources/translations", exist_ok=True)
    os.makedirs("data/sources/morphology/hebrew", exist_ok=True)
    os.makedirs("data/sources/morphology/greek", exist_ok=True)

    download_and_extract_hebrew()
    download_and_extract_greek()
    
    # Download morphological data
    download_morphgnt()
    
    # Convert morphological data to JSON
    convert_oshb_to_json()
    convert_morphgnt_to_json()

    # Import and use abba_data_downloader for translations
    try:
        # Add parent directory to path to import from src
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.abba_data_downloader import ABBADataDownloader
        
        print("Downloading Bible translations...")
        downloader = ABBADataDownloader()
        if not downloader.check_data_exists():
            downloader.download_all()
        else:
            print("  Translations already exist")
    except ImportError as e:
        print(f"Warning: Could not import ABBADataDownloader: {e}")
        print("  Skipping translation downloads")

    manifest = generate_manifest()

    print("\nDownload complete!")
    print(f"Total files: {sum(m['total_files'] for m in manifest['sources'].values())}")


if __name__ == "__main__":
    main()
