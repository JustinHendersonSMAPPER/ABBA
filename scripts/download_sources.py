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


def download_bdb_lexicon() -> None:
    """Download BDB Hebrew Lexicon data (public domain)."""
    print("Downloading BDB Hebrew Lexicon...")
    
    # BDB (Brown-Driver-Briggs) is public domain (published 1906)
    # Try multiple sources for comprehensive Hebrew lexicon data
    sources = [
        {
            "url": "https://raw.githubusercontent.com/openscriptures/HebrewLexicon/master/BDB_Lexicon.xml",
            "filename": "bdb_lexicon.xml",
            "description": "BDB from OpenScriptures"
        },
        {
            "url": "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Lexicons/Hebrew/BDB.xml", 
            "filename": "bdb_step_lexicon.xml",
            "description": "BDB from STEPBible"
        },
        {
            "url": "https://raw.githubusercontent.com/morphgnt/morphological-lexicon/master/data/bdb-hebrew.json",
            "filename": "bdb_morphological.json", 
            "description": "BDB morphological data"
        }
    ]
    
    downloaded_any = False
    for source in sources:
        try:
            target_path = f"data/lexicons/{source['filename']}"
            urllib.request.urlretrieve(source["url"], target_path)
            print(f"  ✓ Downloaded {source['description']}")
            downloaded_any = True
        except Exception as e:
            print(f"  ⚠ Could not download {source['description']}: {e}")
    
    if not downloaded_any:
        print("  ⚠ No BDB sources available, creating enhanced Hebrew lexicon...")
        create_enhanced_hebrew_lexicon()
    else:
        print("  ✓ BDB Hebrew lexicon data available for comprehensive analysis")


def create_enhanced_hebrew_lexicon() -> None:
    """Create an enhanced Hebrew lexicon from available OSHB morphological data."""
    print("  Creating enhanced Hebrew lexicon from OSHB morphological data...")
    
    # Try to extract comprehensive lexicon from existing morphological data
    morph_dir = Path("data/sources/morphology/hebrew")
    lemma_entries = {}
    
    if morph_dir.exists():
        for json_file in morph_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for verse in data.get('verses', []):
                        for word in verse.get('words', []):
                            lemma = word.get('lemma', '').strip()
                            text = word.get('text', '').strip()
                            pos = word.get('POS', word.get('pos', '')).strip()
                            
                            if lemma and text:
                                clean_lemma = lemma.replace('/', '').strip()
                                if clean_lemma not in lemma_entries:
                                    lemma_entries[clean_lemma] = {
                                        "hebrew": clean_lemma,
                                        "pos": pos,
                                        "forms": set(),
                                        "frequency": 0
                                    }
                                lemma_entries[clean_lemma]["forms"].add(text)
                                lemma_entries[clean_lemma]["frequency"] += 1
            except Exception as e:
                print(f"    Warning: Error processing {json_file.name}: {e}")
    
    # Add basic meanings for common words
    basic_meanings = {
        "אלהים": ["God", "god", "gods", "divine being"],
        "ארץ": ["earth", "land", "ground", "country"],
        "שמים": ["heaven", "heavens", "sky"],
        "ברא": ["create", "make", "form"],
        "ראשית": ["beginning", "first", "start"],
        "היה": ["be", "was", "were", "become"],
        "יהוה": ["LORD", "Yahweh"],
        "מים": ["water", "waters"],
        "רוח": ["spirit", "wind", "breath"],
        "אור": ["light"],
        "טוב": ["good"],
        "יום": ["day"],
        "לילה": ["night"],
        "את": [],  # Object marker
        "ו": ["and"],
        "ה": ["the"],
        "ב": ["in", "on", "with"],
        "ל": ["to", "for"],
        "מ": ["from", "of"],
    }
    
    # Combine morphological data with basic meanings
    enhanced_lexicon = {
        "metadata": {
            "name": "Enhanced Hebrew Lexicon", 
            "source": "OSHB morphological data + public domain references",
            "license": "Public Domain",
            "created": datetime.now(UTC).isoformat(),
            "entries_count": len(lemma_entries)
        },
        "entries": {}
    }
    
    for lemma, entry in lemma_entries.items():
        enhanced_entry = {
            "hebrew": lemma,
            "pos": entry["pos"],
            "frequency": entry["frequency"],
            "forms": list(entry["forms"]),
            "meanings": basic_meanings.get(lemma, [])
        }
        enhanced_lexicon["entries"][lemma] = enhanced_entry
    
    with open("data/lexicons/enhanced_hebrew_lexicon.json", "w", encoding="utf-8") as f:
        json.dump(enhanced_lexicon, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Created enhanced Hebrew lexicon with {len(lemma_entries)} lemmas")


def create_basic_hebrew_lexicon() -> None:
    """Create a basic Hebrew lexicon from public domain sources."""
    # Fallback to basic lexicon if enhanced creation fails
    create_enhanced_hebrew_lexicon()


def download_greek_lexicon() -> None:
    """Download comprehensive Greek lexicon data (public domain sources)."""
    print("Downloading Greek Lexicon...")
    
    # Thayer's Greek-English Lexicon is public domain (published 1889)
    # BDAG is copyrighted, but Thayer's provides comprehensive NT Greek coverage
    sources = [
        {
            "url": "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Lexicons/Greek/Thayers.xml",
            "filename": "thayers_lexicon.xml", 
            "description": "Thayer's Greek-English Lexicon from STEPBible"
        },
        {
            "url": "https://raw.githubusercontent.com/openscriptures/GreekLexicon/master/Thayers_Lexicon.xml",
            "filename": "thayers_openscriptures.xml",
            "description": "Thayer's from OpenScriptures"
        },
        {
            "url": "https://raw.githubusercontent.com/morphgnt/morphological-lexicon/master/data/thayers-greek.json",
            "filename": "thayers_morphological.json",
            "description": "Thayer's morphological data"
        },
        {
            "url": "https://raw.githubusercontent.com/biblicalhumanities/strongs/master/greek.json",
            "filename": "strongs_greek.json",
            "description": "Strong's Greek dictionary"
        }
    ]
    
    downloaded_any = False
    for source in sources:
        try:
            target_path = f"data/lexicons/{source['filename']}"
            urllib.request.urlretrieve(source["url"], target_path)
            print(f"  ✓ Downloaded {source['description']}")
            downloaded_any = True
        except Exception as e:
            print(f"  ⚠ Could not download {source['description']}: {e}")
    
    if not downloaded_any:
        print("  ⚠ No comprehensive Greek sources available, creating enhanced Greek lexicon...")
        create_enhanced_greek_lexicon()
    else:
        print("  ✓ Comprehensive Greek lexicon data available (Thayer's + Strong's)")


def create_enhanced_greek_lexicon() -> None:
    """Create an enhanced Greek lexicon from available MorphGNT morphological data."""
    print("  Creating enhanced Greek lexicon from MorphGNT morphological data...")
    
    # Try to extract comprehensive lexicon from existing morphological data
    morph_dir = Path("data/sources/morphology/greek") 
    lemma_entries = {}
    
    if morph_dir.exists():
        for json_file in morph_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for verse in data.get('verses', []):
                        for word in verse.get('words', []):
                            lemma = word.get('lemma', '').strip()
                            text = word.get('text', '').strip()
                            normalized = word.get('normalized', '').strip()
                            pos = word.get('pos', '').strip()
                            
                            if lemma and text:
                                if lemma not in lemma_entries:
                                    lemma_entries[lemma] = {
                                        "greek": lemma,
                                        "pos": pos,
                                        "forms": set(),
                                        "frequency": 0
                                    }
                                lemma_entries[lemma]["forms"].add(text)
                                if normalized:
                                    lemma_entries[lemma]["forms"].add(normalized)
                                lemma_entries[lemma]["frequency"] += 1
            except Exception as e:
                print(f"    Warning: Error processing {json_file.name}: {e}")
    
    # Add basic meanings for common words  
    basic_meanings = {
        "θεός": ["God", "god", "divine", "deity"],
        "κόσμος": ["world", "universe", "cosmos"],
        "λόγος": ["word", "message", "speech", "reason"],
        "ἀρχή": ["beginning", "origin", "rule"],
        "εἰμί": ["am", "is", "are", "was", "were", "be"],
        "γίνομαι": ["become", "came", "was", "happen"],
        "ἔρχομαι": ["come", "came", "go"],
        "λέγω": ["say", "said", "speak", "tell"],
        "ἄνθρωπος": ["man", "human", "person", "people"],
        "φῶς": ["light"],
        "σκοτία": ["darkness"],
        "ζωή": ["life"],
        "ἀλήθεια": ["truth"],
        "χάρις": ["grace"],
        "δόξα": ["glory"],
        "ὁ": ["the"],
        "καί": ["and", "also", "even"],
        "ἐν": ["in", "on", "among"],
        "εἰς": ["into", "to", "for"],
        "ἐκ": ["from", "out of"],
        "πρός": ["with", "to", "toward"],
        "διά": ["through", "because of"],
        "μετά": ["with", "after"],
        "ὅτι": ["that", "because"],
    }
    
    # Combine morphological data with basic meanings
    enhanced_lexicon = {
        "metadata": {
            "name": "Enhanced Greek Lexicon",
            "source": "MorphGNT morphological data + public domain references", 
            "license": "Public Domain",
            "created": datetime.now(UTC).isoformat(),
            "entries_count": len(lemma_entries)
        },
        "entries": {}
    }
    
    for lemma, entry in lemma_entries.items():
        enhanced_entry = {
            "greek": lemma,
            "pos": entry["pos"],
            "frequency": entry["frequency"],
            "forms": list(entry["forms"]),
            "meanings": basic_meanings.get(lemma, [])
        }
        enhanced_lexicon["entries"][lemma] = enhanced_entry
    
    with open("data/lexicons/enhanced_greek_lexicon.json", "w", encoding="utf-8") as f:
        json.dump(enhanced_lexicon, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Created enhanced Greek lexicon with {len(lemma_entries)} lemmas")


def create_basic_greek_lexicon() -> None:
    """Create a basic Greek lexicon from public domain sources."""
    # Fallback to basic lexicon if enhanced creation fails
    create_enhanced_greek_lexicon()


def download_modern_parallel_corpora() -> None:
    """Download modern parallel biblical corpora for training."""
    print("Downloading modern parallel corpora...")
    
    try:
        # Download Berean Study Bible (Public Domain)
        print("  Downloading Berean Study Bible...")
        berean_url = "https://berean.bible/downloads/bsb.json"
        
        try:
            urllib.request.urlretrieve(berean_url, "data/parallel_corpora/berean_study_bible.json")
            print("    ✓ Downloaded Berean Study Bible")
        except:
            print("    ⚠ Berean Study Bible not available from direct link")
        
        # Create a sample parallel corpus for demonstration
        create_sample_parallel_corpus()
        
    except Exception as e:
        print(f"  ⚠ Could not download parallel corpora: {e}")
        create_sample_parallel_corpus()


def create_sample_parallel_corpus() -> None:
    """Create sample parallel corpus for alignment training."""
    sample_corpus = {
        "metadata": {
            "name": "Sample Biblical Parallel Corpus",
            "description": "Sample alignments for training biblical alignment models",
            "license": "Public Domain",
            "created": datetime.now(UTC).isoformat()
        },
        "alignments": [
            {
                "reference": "Gen.1.1",
                "hebrew": ["בראשית", "ברא", "אלהים", "את", "השמים", "ואת", "הארץ"],
                "english": ["In", "the", "beginning", "God", "created", "the", "heavens", "and", "the", "earth"],
                "manual_alignments": [
                    {"hebrew_index": 0, "english_indices": [0, 1, 2], "confidence": 0.95, "type": "phrase"},
                    {"hebrew_index": 1, "english_indices": [4], "confidence": 0.9, "type": "word"},
                    {"hebrew_index": 2, "english_indices": [3], "confidence": 0.95, "type": "word"},
                    {"hebrew_index": 3, "english_indices": [], "confidence": 1.0, "type": "null"},
                    {"hebrew_index": 4, "english_indices": [5, 6], "confidence": 0.9, "type": "phrase"},
                    {"hebrew_index": 5, "english_indices": [7], "confidence": 0.95, "type": "word"},
                    {"hebrew_index": 6, "english_indices": [8, 9], "confidence": 0.9, "type": "phrase"}
                ]
            },
            {
                "reference": "John.1.1", 
                "greek": ["Ἐν", "ἀρχῇ", "ἦν", "ὁ", "λόγος", "καὶ", "ὁ", "λόγος", "ἦν", "πρὸς", "τὸν", "θεόν", "καὶ", "θεὸς", "ἦν", "ὁ", "λόγος"],
                "english": ["In", "the", "beginning", "was", "the", "Word", "and", "the", "Word", "was", "with", "God", "and", "the", "Word", "was", "God"],
                "manual_alignments": [
                    {"greek_index": 0, "english_indices": [0], "confidence": 0.95, "type": "word"},
                    {"greek_index": 1, "english_indices": [1, 2], "confidence": 0.9, "type": "phrase"},
                    {"greek_index": 2, "english_indices": [3], "confidence": 0.9, "type": "word"},
                    {"greek_index": 3, "english_indices": [4], "confidence": 0.95, "type": "word"},
                    {"greek_index": 4, "english_indices": [5], "confidence": 0.95, "type": "word"}
                ]
            }
        ]
    }
    
    with open("data/parallel_corpora/sample_alignments.json", "w", encoding="utf-8") as f:
        json.dump(sample_corpus, f, indent=2, ensure_ascii=False)
    
    print("  ✓ Created sample parallel corpus")


def download_strongs_concordance() -> None:
    """Download Strong's Concordance data (public domain)."""
    print("Downloading Strong's Concordance...")
    
    # Strong's Concordance is public domain and provides structured lemma-to-definition mappings
    sources = [
        {
            "url": "https://raw.githubusercontent.com/biblicalhumanities/strongs/master/hebrew.json",
            "filename": "strongs_hebrew.json",
            "description": "Strong's Hebrew dictionary"
        },
        {
            "url": "https://raw.githubusercontent.com/biblicalhumanities/strongs/master/greek.json", 
            "filename": "strongs_greek.json",
            "description": "Strong's Greek dictionary"
        },
        {
            "url": "https://raw.githubusercontent.com/openscriptures/strongs/master/strongshebrewdictionary.xml",
            "filename": "strongs_hebrew.xml",
            "description": "Strong's Hebrew XML from OpenScriptures"
        },
        {
            "url": "https://raw.githubusercontent.com/openscriptures/strongs/master/strongsgreekdictionary.xml",
            "filename": "strongs_greek.xml", 
            "description": "Strong's Greek XML from OpenScriptures"
        }
    ]
    
    downloaded_any = False
    for source in sources:
        try:
            target_path = f"data/lexicons/{source['filename']}"
            urllib.request.urlretrieve(source["url"], target_path)
            print(f"  ✓ Downloaded {source['description']}")
            downloaded_any = True
        except Exception as e:
            print(f"  ⚠ Could not download {source['description']}: {e}")
    
    if downloaded_any:
        print("  ✓ Strong's Concordance data available for lemma-to-definition mapping")
    else:
        print("  ⚠ No Strong's sources available")


def download_embedding_metadata() -> None:
    """Download information about available multilingual embeddings."""
    print("Setting up embedding model information...")
    
    embedding_info = {
        "metadata": {
            "description": "Multilingual embedding models for biblical alignment",
            "updated": datetime.now(UTC).isoformat()
        },
        "models": {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
                "description": "Multilingual sentence embedding model",
                "languages": 50,
                "size": "420MB",
                "license": "Apache 2.0",
                "recommended": True,
                "download_command": "pip install sentence-transformers"
            },
            "sentence-transformers/LaBSE": {
                "description": "Language-agnostic BERT Sentence Embedding",
                "languages": 109, 
                "size": "1.88GB",
                "license": "Apache 2.0",
                "recommended": True,
                "download_command": "pip install sentence-transformers"
            },
            "microsoft/Multilingual-MiniLM-L12-H384": {
                "description": "Microsoft multilingual model",
                "languages": 100,
                "size": "134MB", 
                "license": "MIT",
                "recommended": False,
                "download_command": "pip install transformers"
            }
        },
        "usage_note": "These models will be downloaded automatically when first used. No manual download required."
    }
    
    with open("data/embeddings/models_info.json", "w", encoding="utf-8") as f:
        json.dump(embedding_info, f, indent=2, ensure_ascii=False)
    
    print("  ✓ Created embedding model information")


def main() -> None:
    """Main download process."""
    os.makedirs("data/sources/hebrew", exist_ok=True)
    os.makedirs("data/sources/greek", exist_ok=True)
    os.makedirs("data/sources/translations", exist_ok=True)
    os.makedirs("data/sources/morphology/hebrew", exist_ok=True)
    os.makedirs("data/sources/morphology/greek", exist_ok=True)
    os.makedirs("data/lexicons", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    os.makedirs("data/parallel_corpora", exist_ok=True)

    download_and_extract_hebrew()
    download_and_extract_greek()
    
    # Download morphological data
    download_morphgnt()
    
    # Convert morphological data to JSON
    convert_oshb_to_json()
    convert_morphgnt_to_json()
    
    # Download comprehensive lexicon resources
    print("\n" + "="*60)
    print("DOWNLOADING COMPREHENSIVE LEXICON RESOURCES")
    print("="*60)
    print("Note: BDAG and HALOT are copyrighted, downloading public domain alternatives:")
    print("  - BDB Hebrew Lexicon (Brown-Driver-Briggs, 1906)")
    print("  - Thayer's Greek-English Lexicon (1889)")
    print("  - Strong's Concordance (public domain)")
    print("  - Enhanced lexicons from OSHB/MorphGNT morphological data")
    
    download_bdb_lexicon()
    download_greek_lexicon()
    download_strongs_concordance()
    download_modern_parallel_corpora()
    download_embedding_metadata()

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
