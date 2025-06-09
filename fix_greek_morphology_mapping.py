#!/usr/bin/env python3
"""Fix for Greek morphology filename mapping issue."""

import json
import shutil
from pathlib import Path

# Mapping from book codes to Greek morphology filenames
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

def create_symlinks():
    """Create symbolic links from book codes to actual Greek morphology files."""
    
    greek_dir = Path("data/sources/morphology/greek")
    if not greek_dir.exists():
        print(f"Greek morphology directory not found: {greek_dir}")
        return
    
    print("Creating symbolic links for Greek morphology files...")
    print("=" * 50)
    
    created = 0
    for book_code, filename in BOOK_CODE_TO_GREEK_FILE.items():
        source_file = greek_dir / f"{filename}.json"
        target_link = greek_dir / f"{book_code}.json"
        
        if source_file.exists():
            if target_link.exists():
                if target_link.is_symlink():
                    print(f"  Link already exists: {book_code}.json -> {filename}.json")
                else:
                    print(f"  File already exists (not a link): {book_code}.json")
            else:
                try:
                    target_link.symlink_to(source_file.name)
                    print(f"  ✓ Created: {book_code}.json -> {filename}.json")
                    created += 1
                except Exception as e:
                    print(f"  ✗ Failed to create link for {book_code}: {e}")
        else:
            print(f"  ✗ Source file not found: {filename}.json")
    
    print(f"\nCreated {created} new symbolic links")

def copy_files_instead():
    """Alternative: Copy files with book code names instead of symlinks."""
    
    greek_dir = Path("data/sources/morphology/greek")
    if not greek_dir.exists():
        print(f"Greek morphology directory not found: {greek_dir}")
        return
    
    print("Copying Greek morphology files with book code names...")
    print("=" * 50)
    
    copied = 0
    for book_code, filename in BOOK_CODE_TO_GREEK_FILE.items():
        source_file = greek_dir / f"{filename}.json"
        target_file = greek_dir / f"{book_code}.json"
        
        if source_file.exists():
            if target_file.exists():
                print(f"  File already exists: {book_code}.json")
            else:
                try:
                    shutil.copy2(source_file, target_file)
                    print(f"  ✓ Copied: {filename}.json -> {book_code}.json")
                    copied += 1
                except Exception as e:
                    print(f"  ✗ Failed to copy {book_code}: {e}")
        else:
            print(f"  ✗ Source file not found: {filename}.json")
    
    print(f"\nCopied {copied} files")

def patch_coverage_analyzer():
    """Create a patched version of the coverage analyzer with proper mapping."""
    
    patch_content = '''"""
Patched Coverage Analyzer with Greek morphology filename mapping.
"""

from pathlib import Path
from typing import Dict

# Mapping from book codes to Greek morphology filenames
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

def get_morphology_path(morphology_dir: Path, language: str, book_code: str) -> Path:
    """Get the correct morphology file path for a book."""
    
    if language == 'greek' and book_code in BOOK_CODE_TO_GREEK_FILE:
        # Use the mapping for Greek files
        filename = BOOK_CODE_TO_GREEK_FILE[book_code]
        return morphology_dir / language / f"{filename}.json"
    else:
        # Default behavior for Hebrew and others
        return morphology_dir / language / f"{book_code}.json"

# To use this in the coverage analyzer, replace line 137:
# morph_path = self.morphology_dir / language / f"{book_code}.json"
# with:
# morph_path = get_morphology_path(self.morphology_dir, language, book_code)
'''
    
    with open('morphology_mapping_patch.py', 'w') as f:
        f.write(patch_content)
    
    print("\nCreated morphology_mapping_patch.py with the mapping function")
    print("To fix the coverage analyzer, update line 137 to use get_morphology_path()")

def main():
    """Main function to fix the Greek morphology mapping issue."""
    
    print("Greek Morphology Mapping Fix")
    print("=" * 50)
    print("\nThis script provides three solutions:")
    print("1. Create symbolic links (requires Unix-like OS)")
    print("2. Copy files with book code names")
    print("3. Generate a patch for the coverage analyzer")
    
    choice = input("\nWhich solution would you like to apply? (1/2/3): ").strip()
    
    if choice == '1':
        create_symlinks()
    elif choice == '2':
        copy_files_instead()
    elif choice == '3':
        patch_coverage_analyzer()
    else:
        print("Invalid choice. Showing all options:")
        print("\n1. Symbolic links:")
        create_symlinks()
        print("\n2. File copying:")
        # Don't actually copy, just show what would happen
        print("(Run with choice 2 to actually copy files)")
        print("\n3. Patch generation:")
        patch_coverage_analyzer()

if __name__ == '__main__':
    main()