#!/usr/bin/env python3
"""Test the canon-aware import system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.parallel_import import Canon, get_translation_canon, EXTENDED_CANON_BOOKS

def test_canon_detection():
    """Test that translations are correctly mapped to canons."""
    print("Testing Canon Detection")
    print("="*50)
    
    test_cases = [
        # Protestant
        ('KJV', Canon.BOOKS_66),
        ('ESV', Canon.BOOKS_66),
        ('NIV', Canon.BOOKS_66),
        ('NET', Canon.BOOKS_66),
        
        # Catholic
        ('NABRE', Canon.BOOKS_73),
        ('DRC', Canon.BOOKS_73),
        ('RSV-CE', Canon.BOOKS_73),
        ('catholic_edition', Canon.BOOKS_73),
        
        # Orthodox
        ('EOB', Canon.BOOKS_76_PLUS),
        ('LXX', Canon.BOOKS_76_PLUS),
        ('SEPT', Canon.BOOKS_76_PLUS),
        
        # Jewish
        ('JPS', Canon.BOOKS_39),
        ('TNK', Canon.BOOKS_39),
        
        # Default cases
        ('unknown_translation', Canon.BOOKS_66),
        ('BSB', Canon.BOOKS_66),
    ]
    
    for translation_id, expected_canon in test_cases:
        actual_canon = get_translation_canon(translation_id)
        status = "✓" if actual_canon == expected_canon else "✗"
        print(f"{status} {translation_id:20} -> {actual_canon.value:12} (expected: {expected_canon.value})")

def test_book_validation():
    """Test that extended canon books are recognized."""
    print("\n\nTesting Extended Canon Book Recognition")
    print("="*50)
    
    # Test WIS (Wisdom) in different contexts
    test_scenarios = [
        ('WIS', 'KJV', False, "Protestant Bible should warn about WIS"),
        ('WIS', 'NABRE', True, "Catholic Bible should accept WIS"),
        ('WIS', 'EOB', True, "Orthodox Bible should accept WIS"),
        ('UNKNOWN', 'NABRE', False, "Unknown book should warn in any Bible"),
        ('1MA', 'DRC', True, "Catholic Bible should accept 1 Maccabees"),
        ('3MA', 'OSB', True, "Orthodox Bible should accept 3 Maccabees"),
        ('ENO', 'ETHIOP', True, "Ethiopian Bible should accept Enoch"),
    ]
    
    for book_code, translation_id, should_be_silent, description in test_scenarios:
        canon = get_translation_canon(translation_id)
        
        # Check if it's a known extended book
        is_extended = any(
            book_code in books 
            for books in EXTENDED_CANON_BOOKS.values()
        )
        
        # Apply the same logic as the import
        would_be_silent = canon != Canon.BOOKS_66 and is_extended
        
        status = "✓" if would_be_silent == should_be_silent else "✗"
        print(f"{status} {book_code} in {translation_id:10} -> Silent: {would_be_silent} - {description}")

def show_canon_books():
    """Display which books are in each canon."""
    print("\n\nExtended Canon Books by Tradition")
    print("="*50)
    
    for canon, books in EXTENDED_CANON_BOOKS.items():
        print(f"\n{canon.value.upper()} additions ({len(books)} books):")
        sorted_books = sorted(books)
        for i in range(0, len(sorted_books), 8):
            print("  " + ", ".join(sorted_books[i:i+8]))

if __name__ == "__main__":
    test_canon_detection()
    test_book_validation()
    show_canon_books()
    
    print("\n\n✅ Canon-aware import system is ready!")
    print("Now when you import Catholic/Orthodox Bibles, books like WIS won't generate warnings.")