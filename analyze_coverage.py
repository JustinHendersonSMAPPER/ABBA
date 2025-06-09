#!/usr/bin/env python3
"""Analyze Hebrew and Greek text coverage in Bible export."""

import os
import json
from collections import defaultdict

# KJV books in canonical order
KJV_OT_BOOKS = [
    'Gen', 'Exod', 'Lev', 'Num', 'Deut', 'Josh', 'Judg', 'Ruth', '1Sam', '2Sam',
    '1Kgs', '2Kgs', '1Chr', '2Chr', 'Ezra', 'Neh', 'Esth', 'Job', 'Ps', 'Prov',
    'Eccl', 'Song', 'Isa', 'Jer', 'Lam', 'Ezek', 'Dan', 'Hos', 'Joel', 'Amos',
    'Obad', 'Jonah', 'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal'
]

KJV_NT_BOOKS = [
    'Matt', 'Mark', 'Luke', 'John', 'Acts', 'Rom', '1Cor', '2Cor', 'Gal', 'Eph',
    'Phil', 'Col', '1Thess', '2Thess', '1Tim', '2Tim', 'Titus', 'Phlm', 'Heb',
    'Jas', '1Pet', '2Pet', '1John', '2John', '3John', 'Jude', 'Rev'
]

# Greek book code mappings (from source files)
GREEK_BOOK_CODES = {
    'MAT': 'Matt', 'MAR': 'Mark', 'LUK': 'Luke', 'JOH': 'John', 'ACT': 'Acts',
    'ROM': 'Rom', '1CO': '1Cor', '2CO': '2Cor', 'GAL': 'Gal', 'EPH': 'Eph',
    'PHP': 'Phil', 'COL': 'Col', '1TH': '1Thess', '2TH': '2Thess', '1TI': '1Tim',
    '2TI': '2Tim', 'TIT': 'Titus', 'PHM': 'Phlm', 'HEB': 'Heb', 'JAM': 'Jas',
    '1PE': '1Pet', '2PE': '2Pet', '1JO': '1John', '2JO': '2John', '3JO': '3John',
    'JUD': 'Jude', 'REV': 'Rev'
}

def analyze_coverage():
    """Analyze coverage of Hebrew and Greek texts."""
    print("=== Bible Text Coverage Analysis ===\n")
    
    # Check Hebrew source files
    hebrew_dir = 'data/sources/hebrew'
    hebrew_files = [f for f in os.listdir(hebrew_dir) if f.endswith('.xml') and f != 'VerseMap.xml']
    hebrew_books = [f.replace('.xml', '') for f in hebrew_files]
    
    print(f"Hebrew source files found: {len(hebrew_files)}")
    print(f"Expected OT books in KJV: {len(KJV_OT_BOOKS)}")
    
    # Find missing Hebrew books
    missing_hebrew = set(KJV_OT_BOOKS) - set(hebrew_books)
    extra_hebrew = set(hebrew_books) - set(KJV_OT_BOOKS)
    
    if missing_hebrew:
        print(f"\nMissing Hebrew books: {sorted(missing_hebrew)}")
    if extra_hebrew:
        print(f"Extra Hebrew books: {sorted(extra_hebrew)}")
    
    # Check Greek source files
    greek_dir = 'data/sources/greek'
    greek_files = [f for f in os.listdir(greek_dir) if f.endswith('.xml')]
    greek_codes = [f.replace('.xml', '') for f in greek_files]
    
    print(f"\n\nGreek source files found: {len(greek_files)}")
    print(f"Expected NT books in KJV: {len(KJV_NT_BOOKS)}")
    
    # Map Greek codes to standard book names
    greek_books = []
    unmapped_greek = []
    for code in greek_codes:
        if code in GREEK_BOOK_CODES:
            greek_books.append(GREEK_BOOK_CODES[code])
        else:
            unmapped_greek.append(code)
    
    # Find missing Greek books
    missing_greek = set(KJV_NT_BOOKS) - set(greek_books)
    extra_greek = set(greek_books) - set(KJV_NT_BOOKS)
    
    if missing_greek:
        print(f"\nMissing Greek books: {sorted(missing_greek)}")
    if extra_greek:
        print(f"Extra Greek books: {sorted(extra_greek)}")
    if unmapped_greek:
        print(f"Unmapped Greek codes: {sorted(unmapped_greek)}")
    
    # Check actual coverage in export
    export_dir = 'full_enhanced_export'
    coverage_stats = defaultdict(lambda: {'total_verses': 0, 'hebrew_verses': 0, 'greek_verses': 0})
    
    print("\n\n=== Checking Actual Coverage in Export ===")
    
    for book_file in os.listdir(export_dir):
        if not book_file.endswith('.json'):
            continue
            
        book_name = book_file.replace('.json', '')
        with open(os.path.join(export_dir, book_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for chapter in data.get('chapters', []):
            for verse in chapter.get('verses', []):
                coverage_stats[book_name]['total_verses'] += 1
                
                # Check for Hebrew text
                if verse.get('hebrew_text') or verse.get('hebrew_words'):
                    coverage_stats[book_name]['hebrew_verses'] += 1
                
                # Check for Greek text
                if verse.get('greek_text') or verse.get('greek_words'):
                    coverage_stats[book_name]['greek_verses'] += 1
    
    # Print coverage report
    print("\nBook Coverage Report:")
    print("-" * 80)
    print(f"{'Book':<10} {'Total':<10} {'Hebrew':<15} {'Greek':<15} {'Coverage':<20}")
    print("-" * 80)
    
    total_all = 0
    total_hebrew = 0
    total_greek = 0
    
    # OT Books
    print("\nOld Testament:")
    for book in KJV_OT_BOOKS:
        if book in coverage_stats:
            stats = coverage_stats[book]
            total = stats['total_verses']
            hebrew = stats['hebrew_verses']
            greek = stats['greek_verses']
            
            total_all += total
            total_hebrew += hebrew
            total_greek += greek
            
            hebrew_pct = (hebrew / total * 100) if total > 0 else 0
            greek_pct = (greek / total * 100) if total > 0 else 0
            
            coverage = []
            if hebrew_pct > 0:
                coverage.append(f"Heb: {hebrew_pct:.1f}%")
            if greek_pct > 0:
                coverage.append(f"Grk: {greek_pct:.1f}%")
            
            print(f"{book:<10} {total:<10} {hebrew:<15} {greek:<15} {', '.join(coverage):<20}")
    
    # NT Books
    print("\nNew Testament:")
    for book in KJV_NT_BOOKS:
        if book in coverage_stats:
            stats = coverage_stats[book]
            total = stats['total_verses']
            hebrew = stats['hebrew_verses']
            greek = stats['greek_verses']
            
            total_all += total
            total_hebrew += hebrew
            total_greek += greek
            
            hebrew_pct = (hebrew / total * 100) if total > 0 else 0
            greek_pct = (greek / total * 100) if total > 0 else 0
            
            coverage = []
            if hebrew_pct > 0:
                coverage.append(f"Heb: {hebrew_pct:.1f}%")
            if greek_pct > 0:
                coverage.append(f"Grk: {greek_pct:.1f}%")
            
            print(f"{book:<10} {total:<10} {hebrew:<15} {greek:<15} {', '.join(coverage):<20}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total verses: {total_all}")
    print(f"Verses with Hebrew text: {total_hebrew} ({total_hebrew/total_all*100:.1f}%)")
    print(f"Verses with Greek text: {total_greek} ({total_greek/total_all*100:.1f}%)")
    
    # Expected coverage
    ot_verses = sum(coverage_stats[book]['total_verses'] for book in KJV_OT_BOOKS if book in coverage_stats)
    nt_verses = sum(coverage_stats[book]['total_verses'] for book in KJV_NT_BOOKS if book in coverage_stats)
    
    print(f"\nExpected Hebrew coverage (OT): {ot_verses} verses")
    print(f"Expected Greek coverage (NT): {nt_verses} verses")
    
    if total_hebrew < ot_verses:
        print(f"\nMissing Hebrew coverage: {ot_verses - total_hebrew} verses ({(ot_verses - total_hebrew)/ot_verses*100:.1f}%)")
    if total_greek < nt_verses:
        print(f"Missing Greek coverage: {nt_verses - total_greek} verses ({(nt_verses - total_greek)/nt_verses*100:.1f}%)")

if __name__ == '__main__':
    analyze_coverage()