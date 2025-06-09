#!/usr/bin/env python3
"""Debug Greek alignment coverage issue."""

import json
import os
from pathlib import Path

# Mapping from export book codes to Greek morphology filenames
BOOK_CODE_TO_GREEK_FILE = {
    'Matt': 'matthew',
    'Mark': 'mark', 
    'Luke': 'luke',
    'John': 'john',
    'Acts': 'acts',
    'Rom': 'romans',
    '1Cor': '1corinthians',
    '2Cor': '2corinthians',
    'Gal': 'galatians',
    'Eph': 'ephesians',
    'Phil': 'philippians',
    'Col': 'colossians',
    '1Thess': '1thessalonians',
    '2Thess': '2thessalonians',
    '1Tim': '1timothy',
    '2Tim': '2timothy',
    'Titus': 'titus',
    'Phlm': 'philemon',
    'Heb': 'hebrews',
    'Jas': 'james',
    '1Pet': '1peter',
    '2Pet': '2peter',
    '1John': '1john',
    '2John': '2john',
    '3John': '3john',
    'Jude': 'jude',
    'Rev': 'revelation'
}

# Also map canonical 3-letter codes
CANONICAL_TO_GREEK_FILE = {
    'MAT': 'matthew',
    'MRK': 'mark',
    'LUK': 'luke', 
    'JHN': 'john',
    'ACT': 'acts',
    'ROM': 'romans',
    '1CO': '1corinthians',
    '2CO': '2corinthians',
    'GAL': 'galatians',
    'EPH': 'ephesians',
    'PHP': 'philippians',
    'COL': 'colossians',
    '1TH': '1thessalonians',
    '2TH': '2thessalonians',
    '1TI': '1timothy',
    '2TI': '2timothy',
    'TIT': 'titus',
    'PHM': 'philemon',
    'HEB': 'hebrews',
    'JAS': 'james',
    '1PE': '1peter',
    '2PE': '2peter',
    '1JN': '1john',
    '2JN': '2john',
    '3JN': '3john',
    'JUD': 'jude',
    'REV': 'revelation'
}

def check_greek_morphology_loading():
    """Check if Greek morphology files can be loaded."""
    
    greek_dir = Path("data/sources/morphology/greek")
    
    print("Greek Morphology Debug Report")
    print("=" * 50)
    
    print(f"\n1. Greek morphology directory: {greek_dir}")
    print(f"   Exists: {greek_dir.exists()}")
    
    if greek_dir.exists():
        files = sorted(greek_dir.glob("*.json"))
        print(f"   Files found: {len(files)}")
        print("\n   Available files:")
        for f in files:
            print(f"   - {f.name}")
    
    print("\n2. Book Code Mappings:")
    print("\n   Export codes -> Greek filenames:")
    for code, filename in sorted(BOOK_CODE_TO_GREEK_FILE.items()):
        filepath = greek_dir / f"{filename}.json"
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {code:10} -> {filename}.json")
    
    print("\n3. Sample Greek Morphology Data:")
    # Load Matthew as example
    matt_file = greek_dir / "matthew.json"
    if matt_file.exists():
        with open(matt_file) as f:
            data = json.load(f)
        
        print(f"\n   Book: {data.get('book')}")
        print(f"   Language: {data.get('language')}")
        print(f"   Total verses: {len(data.get('verses', []))}")
        
        if data.get('verses'):
            verse = data['verses'][0]
            print(f"\n   First verse: {verse.get('verse_id')}")
            print(f"   Words in verse: {len(verse.get('words', []))}")
            if verse.get('words'):
                word = verse['words'][0]
                print(f"\n   First word example:")
                print(f"     Text: {word.get('text')}")
                print(f"     Lemma: {word.get('lemma')}")
                print(f"     Parse: {word.get('parse')}")
                print(f"     Morph features: {word.get('morph_features')}")
    
    print("\n4. Export Coverage Check:")
    # Check what directories have exports
    export_dirs = [
        'aligned_export_improved',
        'aligned_full_test', 
        'full_enhanced_export',
        'full_fixed_export'
    ]
    
    for dir_name in export_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"\n   {dir_name}:")
            # Check a sample NT book
            matt_export = dir_path / "Matt.json"
            if matt_export.exists():
                with open(matt_export) as f:
                    data = json.load(f)
                
                # Check first verse for Greek data
                has_greek = False
                if 'chapters' in data and data['chapters']:
                    ch1 = data['chapters'][0]
                    if 'verses' in ch1 and ch1['verses']:
                        v1 = ch1['verses'][0]
                        if 'greek_text' in v1 or 'greek_words' in v1:
                            has_greek = True
                            print(f"     ✓ Has Greek data")
                        else:
                            print(f"     ✗ No Greek data found")
                            print(f"     Keys in verse: {list(v1.keys())}")
            else:
                print(f"     - Matt.json not found")

def suggest_fix():
    """Suggest how to fix the issue."""
    
    print("\n" + "=" * 50)
    print("SUGGESTED FIX:")
    print("=" * 50)
    
    print("""
The issue appears to be that the export process is not loading Greek morphology data.

Possible causes:
1. The export code may not have the correct mapping from book codes (Matt, Mark, etc.) 
   to Greek morphology filenames (matthew.json, mark.json, etc.)

2. The export code may not be configured to load Greek morphology at all.

3. The verse ID format mismatch: morphology files use "matthew.1.1" but the system
   may expect "MAT.1.1" or "Matt.1.1"

To fix:
1. Update the export code to include the book code to filename mapping shown above
2. Ensure the Greek morphology loader is called during export
3. Handle the verse ID format conversion when matching verses

The mapping dictionaries in this script can be used in the export code.
""")

if __name__ == '__main__':
    check_greek_morphology_loading()
    suggest_fix()