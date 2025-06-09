#!/usr/bin/env python3
"""Debug cross-reference loading."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abba.verse_id import parse_verse_id, normalize_book_name
from abba.book_codes import is_valid_book_code

def main():
    # Load cross-references
    xref_file = Path(__file__).parent.parent / "data" / "cross_references.json"
    
    with open(xref_file, 'r') as f:
        data = json.load(f)
    
    print("Testing verse ID parsing:")
    print("-" * 60)
    
    # Test each reference
    for ref in data['references'][:5]:
        source = ref['source']
        target = ref['target']
        
        print(f"\nSource: {source}")
        source_parsed = parse_verse_id(source)
        if source_parsed:
            print(f"  ✓ Parsed: {source_parsed}")
        else:
            print(f"  ✗ Failed to parse")
            # Try to understand why
            parts = source.split('.')
            if len(parts) >= 1:
                book = parts[0]
                print(f"  Book code: {book}")
                print(f"  Is valid code: {is_valid_book_code(book)}")
                normalized = normalize_book_name(book)
                print(f"  Normalized: {normalized}")
        
        print(f"\nTarget: {target}")
        target_parsed = parse_verse_id(target)
        if target_parsed:
            print(f"  ✓ Parsed: {target_parsed}")
        else:
            print(f"  ✗ Failed to parse")
            parts = target.split('.')
            if len(parts) >= 1:
                book = parts[0]
                print(f"  Book code: {book}")
                print(f"  Is valid code: {is_valid_book_code(book)}")
                normalized = normalize_book_name(book)
                print(f"  Normalized: {normalized}")

if __name__ == "__main__":
    main()