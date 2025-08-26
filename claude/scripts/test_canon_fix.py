#!/usr/bin/env python3
"""Test that the canon fix properly handles deuterocanonical books."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.parallel_import import ALL_KNOWN_EXTENDED_BOOKS, BOOK_ID_MAP

def test_known_books():
    """Test that known deuterocanonical books are recognized."""
    print("Testing Known Deuterocanonical Books")
    print("="*60)
    
    # Books that should be silently skipped (known deuterocanonical)
    test_books = [
        'WIS',  # Wisdom - the book causing issues for spa_blm
        'SIR',  # Sirach
        'TOB',  # Tobit
        'JDT',  # Judith
        '1MA',  # 1 Maccabees
        '2MA',  # 2 Maccabees
        'BAR',  # Baruch
        'ESG',  # Esther Greek
        'ENO',  # 1 Enoch (Ethiopian)
        'JUB',  # Jubilees (Ethiopian)
    ]
    
    print("\nChecking deuterocanonical books (should be recognized):")
    for book in test_books:
        in_66 = book in BOOK_ID_MAP
        in_extended = book in ALL_KNOWN_EXTENDED_BOOKS
        status = "✓" if in_extended and not in_66 else "✗"
        print(f"{status} {book:4} - In 66-book map: {in_66}, In extended list: {in_extended}")
    
    print("\n\nChecking truly unknown books (should generate warnings):")
    unknown_books = ['XYZ', 'ABC', 'TEST', 'FAKE']
    for book in unknown_books:
        in_66 = book in BOOK_ID_MAP
        in_extended = book in ALL_KNOWN_EXTENDED_BOOKS
        status = "✓" if not in_extended and not in_66 else "✗"
        print(f"{status} {book:4} - In 66-book map: {in_66}, In extended list: {in_extended}")

def test_import_logic():
    """Simulate the import logic to show what would happen."""
    print("\n\nSimulating Import Logic")
    print("="*60)
    
    # Simulate various book IDs being imported
    test_cases = [
        ('GEN', 'KJV'),      # Standard book
        ('WIS', 'spa_blm'),  # Deuterocanonical in Spanish Bible
        ('ENO', 'ethiopic'), # Ethiopian canon book
        ('XYZ', 'unknown'),  # Truly unknown book
    ]
    
    for book_str, translation_id in test_cases:
        book_id = BOOK_ID_MAP.get(book_str, 0)
        
        if book_id == 0:
            # This is the logic in parallel_import.py
            if book_str in ALL_KNOWN_EXTENDED_BOOKS:
                action = "SKIP SILENTLY (known deuterocanonical)"
            else:
                action = "WARNING (unknown book)"
        else:
            action = "IMPORT (standard 66-book canon)"
        
        print(f"{book_str} in {translation_id:10} -> {action}")

def show_extended_books():
    """Display all recognized extended canon books."""
    print("\n\nAll Recognized Extended Canon Books")
    print("="*60)
    
    sorted_books = sorted(ALL_KNOWN_EXTENDED_BOOKS)
    print(f"Total: {len(sorted_books)} books\n")
    
    # Group by category
    categories = {
        'Catholic Core': ['TOB', 'JDT', 'WIS', 'SIR', 'BAR', '1MA', '2MA'],
        'Daniel Additions': ['LJE', 'S3Y', 'SUS', 'BEL', 'ESG'],
        'Orthodox': ['1ES', '3ES', '3MA', '4MA', 'MAN', 'PS2', 'PSS', 'LAO', 'ODE'],
        'Esdras Books': ['2ES', '4ES', '5ES', '6ES', 'EZA'],
        'Ethiopian': ['ENO', 'JUB'],
        'Other': ['DAG', 'PS3', 'POL', 'EEP', 'ADE']
    }
    
    for category, books in categories.items():
        present = [b for b in books if b in ALL_KNOWN_EXTENDED_BOOKS]
        if present:
            print(f"{category}: {', '.join(present)}")

if __name__ == "__main__":
    test_known_books()
    test_import_logic()
    show_extended_books()
    
    print("\n\n✅ The canon fix should now:")
    print("  - Silently skip WIS and other deuterocanonical books")
    print("  - Only warn about truly unknown books")
    print("  - Import exactly what's in bible.db for each translation")