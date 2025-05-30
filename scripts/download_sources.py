#!/usr/bin/env python3
"""
Download and extract only the necessary source files for the ABBA project.
Calculates SHA256 hashes of the actual data files used.
"""
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
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


def download_and_extract_strongs() -> None:
    """Download and extract Strong's lexicon files."""
    print("Downloading Strong's Concordance...")
    url = "https://github.com/openscriptures/strongs/archive/refs/heads/master.zip"

    # Download
    zip_path = "temp_strongs.zip"
    urllib.request.urlretrieve(url, zip_path)

    # Extract XML files
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith("strongsgreek.xml"):
                target = os.path.join("data/sources/lexicons", "strongs_greek.xml")
                with zip_ref.open(file) as source, open(target, "wb") as target_file:
                    shutil.copyfileobj(source, target_file)
            elif file.endswith("StrongHebrewG.xml"):
                target = os.path.join("data/sources/lexicons", "strongs_hebrew.xml")
                with zip_ref.open(file) as source, open(target, "wb") as target_file:
                    shutil.copyfileobj(source, target_file)

    os.remove(zip_path)
    print("  Extracted Strong's Hebrew and Greek lexicons")


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

    # Hash lexicon files
    lexicon_files = {}
    for file in sorted(os.listdir("data/sources/lexicons")):
        filepath = os.path.join("data/sources/lexicons", file)
        lexicon_files[file] = calculate_sha256(filepath)

    manifest["sources"]["lexicons"] = {
        "name": "Strong's Concordance",
        "license": "Public Domain",
        "url": "https://github.com/openscriptures/strongs",
        "files": lexicon_files,
        "total_files": len(lexicon_files),
    }

    # Hash translation files
    translation_files = {}
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

    # Save manifest
    with open("data/sources/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nManifest generated at data/sources/manifest.json")
    return manifest


def main() -> None:
    """Main download process."""
    os.makedirs("data/sources/hebrew", exist_ok=True)
    os.makedirs("data/sources/greek", exist_ok=True)
    os.makedirs("data/sources/lexicons", exist_ok=True)
    os.makedirs("data/sources/translations", exist_ok=True)

    download_and_extract_hebrew()
    download_and_extract_greek()
    download_and_extract_strongs()

    manifest = generate_manifest()

    print("\nDownload complete!")
    print(f"Total files: {sum(m['total_files'] for m in manifest['sources'].values())}")


if __name__ == "__main__":
    main()
