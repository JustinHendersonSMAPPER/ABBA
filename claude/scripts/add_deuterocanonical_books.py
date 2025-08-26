#!/usr/bin/env python3
"""Add Deuterocanonical books to BOOK_ID_MAP if desired."""

# Extended book mapping including Deuterocanonical books
EXTENDED_BOOK_ID_MAP = {
    # Old Testament (Protestant Canon)
    'GEN': 1, 'EXO': 2, 'LEV': 3, 'NUM': 4, 'DEU': 5,
    'JOS': 6, 'JDG': 7, 'RUT': 8, '1SA': 9, '2SA': 10,
    '1KI': 11, '2KI': 12, '1CH': 13, '2CH': 14, 'EZR': 15,
    'NEH': 16, 'EST': 17, 'JOB': 18, 'PSA': 19, 'PRO': 20,
    'ECC': 21, 'SNG': 22, 'ISA': 23, 'JER': 24, 'LAM': 25,
    'EZK': 26, 'DAN': 27, 'HOS': 28, 'JOL': 29, 'AMO': 30,
    'OBA': 31, 'JON': 32, 'MIC': 33, 'NAM': 34, 'HAB': 35,
    'ZEP': 36, 'HAG': 37, 'ZEC': 38, 'MAL': 39,
    # New Testament  
    'MAT': 40, 'MRK': 41, 'LUK': 42, 'JHN': 43, 'ACT': 44,
    'ROM': 45, '1CO': 46, '2CO': 47, 'GAL': 48, 'EPH': 49,
    'PHP': 50, 'COL': 51, '1TH': 52, '2TH': 53, '1TI': 54,
    '2TI': 55, 'TIT': 56, 'PHM': 57, 'HEB': 58, 'JAS': 59,
    '1PE': 60, '2PE': 61, '1JN': 62, '2JN': 63, '3JN': 64,
    'JUD': 65, 'REV': 66,
    # Deuterocanonical/Apocrypha (books 67-80)
    'TOB': 67,  # Tobit
    'JDT': 68,  # Judith
    'ESG': 69,  # Esther (Greek additions)
    'WIS': 70,  # Wisdom of Solomon
    'SIR': 71,  # Sirach/Ecclesiasticus  
    'BAR': 72,  # Baruch
    'LJE': 73,  # Letter of Jeremiah
    'S3Y': 74,  # Song of the Three Young Men
    'SUS': 75,  # Susanna
    'BEL': 76,  # Bel and the Dragon
    '1MA': 77,  # 1 Maccabees
    '2MA': 78,  # 2 Maccabees
    '3MA': 79,  # 3 Maccabees
    '4MA': 80,  # 4 Maccabees
    # Additional books in some traditions
    '1ES': 81,  # 1 Esdras
    '2ES': 82,  # 2 Esdras
    'MAN': 83,  # Prayer of Manasseh
    'PS2': 84,  # Psalm 151
}

print("To add Deuterocanonical books support:")
print("1. Edit /home/jhenderson/github/ABBA/abba/parallel_import.py")
print("2. Replace the BOOK_ID_MAP with the extended version above")
print("3. This will include Catholic/Orthodox biblical books")
print("\nNote: This requires careful consideration of your project's scope")