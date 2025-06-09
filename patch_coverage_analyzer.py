#!/usr/bin/env python3
"""Patch the coverage analyzer to fix Greek morphology loading."""

import re
from pathlib import Path

def patch_coverage_analyzer():
    """Patch the ABBACoverageAnalyzer to handle Greek morphology filenames correctly."""
    
    # Path to the coverage analyzer
    analyzer_path = Path("src/abba_coverage_analyzer.py")
    
    if not analyzer_path.exists():
        print(f"Coverage analyzer not found at {analyzer_path}")
        return False
    
    # Read the file
    with open(analyzer_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "BOOK_CODE_TO_GREEK_FILE" in content:
        print("File appears to be already patched")
        return True
    
    # Add the mapping dictionary after imports
    mapping_code = '''
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

'''
    
    # Insert after the logger line
    logger_match = re.search(r"(logger = logging\.getLogger\('ABBA\.CoverageAnalyzer'\))", content)
    if logger_match:
        insert_pos = logger_match.end()
        content = content[:insert_pos] + "\n" + mapping_code + content[insert_pos:]
    else:
        print("Could not find logger definition to insert mapping after")
        return False
    
    # Replace the morph_path line
    old_line = r"morph_path = self\.morphology_dir / language / f\"{book_code}\.json\""
    new_code = '''# Get correct morphology file path
        if language == 'greek' and book_code in BOOK_CODE_TO_GREEK_FILE:
            filename = BOOK_CODE_TO_GREEK_FILE[book_code]
            morph_path = self.morphology_dir / language / f"{filename}.json"
        else:
            morph_path = self.morphology_dir / language / f"{book_code}.json"'''
    
    content = re.sub(old_line, new_code, content)
    
    # Save backup
    backup_path = analyzer_path.with_suffix('.py.bak')
    with open(backup_path, 'w') as f:
        f.write(open(analyzer_path, 'r').read())
    print(f"Created backup at {backup_path}")
    
    # Write patched file
    with open(analyzer_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {analyzer_path}")
    return True

def main():
    """Apply the patch."""
    print("Patching Coverage Analyzer for Greek Morphology")
    print("=" * 50)
    
    if patch_coverage_analyzer():
        print("\nPatch applied successfully!")
        print("The coverage analyzer will now correctly map book codes to Greek morphology filenames.")
    else:
        print("\nPatch failed. Please check the error messages above.")

if __name__ == '__main__':
    main()