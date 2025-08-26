#!/usr/bin/env python3
"""
Translation to Canon Mapping for ABBA.

This module defines which biblical canon each translation follows,
allowing the system to expect the correct books and avoid false warnings.
"""

from enum import Enum
from typing import Dict, Set

class Canon(Enum):
    """Biblical canon types."""
    PROTESTANT = "protestant"     # 66 books
    CATHOLIC = "catholic"         # 73 books  
    ORTHODOX = "orthodox"         # 76-81 books
    ETHIOPIAN = "ethiopian"       # 81 books
    JEWISH = "jewish"            # 39 books (Tanakh only)

# Books by canon (using STEPBible book codes)
CANON_BOOKS = {
    Canon.PROTESTANT: {
        # Old Testament (39 books)
        'GEN', 'EXO', 'LEV', 'NUM', 'DEU', 'JOS', 'JDG', 'RUT',
        '1SA', '2SA', '1KI', '2KI', '1CH', '2CH', 'EZR', 'NEH',
        'EST', 'JOB', 'PSA', 'PRO', 'ECC', 'SNG', 'ISA', 'JER',
        'LAM', 'EZK', 'DAN', 'HOS', 'JOL', 'AMO', 'OBA', 'JON',
        'MIC', 'NAM', 'HAB', 'ZEP', 'HAG', 'ZEC', 'MAL',
        # New Testament (27 books)
        'MAT', 'MRK', 'LUK', 'JHN', 'ACT', 'ROM', '1CO', '2CO',
        'GAL', 'EPH', 'PHP', 'COL', '1TH', '2TH', '1TI', '2TI',
        'TIT', 'PHM', 'HEB', 'JAS', '1PE', '2PE', '1JN', '2JN',
        '3JN', 'JUD', 'REV'
    },
    
    Canon.CATHOLIC: {
        # All Protestant books plus:
        'TOB', 'JDT', 'ESG', 'WIS', 'SIR', 'BAR', '1MA', '2MA',
        # Plus additions to DAN and EST (handled separately)
    },
    
    Canon.ORTHODOX: {
        # All Catholic books plus:
        '1ES', '3MA', 'MAN', 'PS2',  # Psalm 151
        # Some traditions include: '4MA', '2ES'
    },
    
    Canon.ETHIOPIAN: {
        # All Orthodox books plus:
        'ENO',  # 1 Enoch
        'JUB',  # Jubilees
        '4ES',  # 4 Ezra
        # And several others
    },
    
    Canon.JEWISH: {
        # Only Hebrew Bible/Old Testament
        'GEN', 'EXO', 'LEV', 'NUM', 'DEU', 'JOS', 'JDG', 'RUT',
        '1SA', '2SA', '1KI', '2KI', '1CH', '2CH', 'EZR', 'NEH',
        'EST', 'JOB', 'PSA', 'PRO', 'ECC', 'SNG', 'ISA', 'JER',
        'LAM', 'EZK', 'DAN', 'HOS', 'JOL', 'AMO', 'OBA', 'JON',
        'MIC', 'NAM', 'HAB', 'ZEP', 'HAG', 'ZEC', 'MAL'
    }
}

# Translation to Canon mapping
TRANSLATION_CANON_MAP: Dict[str, Canon] = {
    # Protestant Translations
    'KJV': Canon.PROTESTANT,
    'NKJV': Canon.PROTESTANT,
    'NIV': Canon.PROTESTANT,
    'ESV': Canon.PROTESTANT,
    'NASB': Canon.PROTESTANT,
    'NRSV': Canon.PROTESTANT,
    'NLT': Canon.PROTESTANT,
    'CSB': Canon.PROTESTANT,
    'NET': Canon.PROTESTANT,
    'BSB': Canon.PROTESTANT,
    'LSB': Canon.PROTESTANT,
    'WEB': Canon.PROTESTANT,
    'YLT': Canon.PROTESTANT,
    'DARBY': Canon.PROTESTANT,
    'ASV': Canon.PROTESTANT,
    'ERV': Canon.PROTESTANT,
    'ISV': Canon.PROTESTANT,
    
    # Catholic Translations
    'NABRE': Canon.CATHOLIC,     # New American Bible Revised Edition
    'DRC': Canon.CATHOLIC,        # Douay-Rheims Challoner
    'CPDV': Canon.CATHOLIC,       # Catholic Public Domain Version
    'RSV-CE': Canon.CATHOLIC,     # RSV Catholic Edition
    'NRSV-CE': Canon.CATHOLIC,    # NRSV Catholic Edition
    'GNT-CE': Canon.CATHOLIC,     # Good News Catholic Edition
    'NJB': Canon.CATHOLIC,        # New Jerusalem Bible
    'CCB': Canon.CATHOLIC,        # Christian Community Bible
    
    # Orthodox Translations
    'EOB': Canon.ORTHODOX,        # Eastern Orthodox Bible
    'OSB': Canon.ORTHODOX,        # Orthodox Study Bible
    'SAAS': Canon.ORTHODOX,       # St. Athanasius Academy Septuagint
    
    # Jewish Translations (OT only)
    'JPS': Canon.JEWISH,          # Jewish Publication Society
    'TNK': Canon.JEWISH,          # JPS Tanakh
    'OJB': Canon.JEWISH,          # Orthodox Jewish Bible
    
    # Ethiopian
    'AMHARIC': Canon.ETHIOPIAN,
    
    # Ancient versions often include deuterocanonical
    'LXX': Canon.ORTHODOX,        # Septuagint
    'VUL': Canon.CATHOLIC,        # Latin Vulgate
    'SYRP': Canon.ORTHODOX,       # Syriac Peshitta
}

def get_translation_canon(translation_id: str) -> Canon:
    """
    Get the canon for a translation ID.
    
    Args:
        translation_id: The translation identifier
        
    Returns:
        The Canon enum for this translation, defaults to PROTESTANT
    """
    # Normalize to uppercase for lookup
    translation_id = translation_id.upper()
    
    # Check exact match first
    if translation_id in TRANSLATION_CANON_MAP:
        return TRANSLATION_CANON_MAP[translation_id]
    
    # Check for Catholic editions (common pattern)
    if 'CE' in translation_id or 'CATHOLIC' in translation_id:
        return Canon.CATHOLIC
    
    # Default to Protestant canon (most common)
    return Canon.PROTESTANT

def is_book_in_translation_canon(book_code: str, translation_id: str) -> bool:
    """
    Check if a book should be in a translation's canon.
    
    Args:
        book_code: The book code (e.g., 'WIS', 'GEN')
        translation_id: The translation identifier
        
    Returns:
        True if the book is expected in this translation's canon
    """
    canon = get_translation_canon(translation_id)
    
    # Build the complete book set for this canon
    expected_books = CANON_BOOKS[Canon.PROTESTANT].copy()
    
    if canon in [Canon.CATHOLIC, Canon.ORTHODOX, Canon.ETHIOPIAN]:
        expected_books.update(CANON_BOOKS[Canon.CATHOLIC])
    
    if canon in [Canon.ORTHODOX, Canon.ETHIOPIAN]:
        expected_books.update(CANON_BOOKS[Canon.ORTHODOX])
        
    if canon == Canon.ETHIOPIAN:
        expected_books.update(CANON_BOOKS[Canon.ETHIOPIAN])
        
    if canon == Canon.JEWISH:
        expected_books = CANON_BOOKS[Canon.JEWISH]
    
    return book_code in expected_books

# Example usage
if __name__ == "__main__":
    # Test some translations
    test_cases = [
        ('ESV', 'WIS'),      # Protestant bible, Wisdom - should be False
        ('NABRE', 'WIS'),    # Catholic bible, Wisdom - should be True  
        ('KJV', 'MAT'),      # Protestant bible, Matthew - should be True
        ('JPS', 'MAT'),      # Jewish bible, Matthew - should be False
        ('EOB', '3MA'),      # Orthodox bible, 3 Maccabees - should be True
    ]
    
    for translation, book in test_cases:
        is_valid = is_book_in_translation_canon(book, translation)
        canon = get_translation_canon(translation)
        print(f"{translation} ({canon.value}) + {book} = {is_valid}")