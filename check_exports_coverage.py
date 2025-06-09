#!/usr/bin/env python3
"""Check export directories and analyze coverage."""

import os
import json
from pathlib import Path
from collections import defaultdict

def find_export_dirs():
    """Find all directories that might contain Bible exports."""
    
    possible_dirs = [
        'full_enhanced_export',
        'aligned_export_improved',
        'aligned_full_test',
        'full_fixed_export',
        'enhanced_export',
        'fixed_export',
        'test_export',
        'verify_export',
        'enriched_bible',
        'full_bible_test'
    ]
    
    found_dirs = []
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            # Check if it contains JSON files
            json_files = list(Path(dir_name).glob('*.json'))
            if json_files:
                found_dirs.append((dir_name, len(json_files)))
    
    return found_dirs

def analyze_export_coverage(export_dir):
    """Analyze coverage in a specific export directory."""
    
    coverage_stats = defaultdict(lambda: {'total_verses': 0, 'hebrew_verses': 0, 'greek_verses': 0})
    
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
    
    return coverage_stats

def main():
    """Main function."""
    
    print("Bible Export Coverage Check")
    print("=" * 50)
    
    # Find export directories
    found_dirs = find_export_dirs()
    
    if not found_dirs:
        print("\nNo export directories found with JSON files!")
        print("\nSearching for any JSON files that look like Bible data...")
        
        # Search more broadly
        for root, dirs, files in os.walk('.'):
            # Skip hidden and cache directories
            if any(part.startswith('.') for part in Path(root).parts):
                continue
            if '__pycache__' in root or 'node_modules' in root:
                continue
                
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                # Check if these look like Bible files
                bible_files = []
                for f in json_files[:3]:  # Check first 3 files
                    try:
                        with open(os.path.join(root, f), 'r') as file:
                            data = json.load(file)
                            if isinstance(data, dict) and ('chapters' in data or 'verses' in data):
                                bible_files.append(f)
                    except:
                        pass
                
                if bible_files:
                    print(f"\nFound potential Bible data in: {root}")
                    print(f"  Files: {', '.join(bible_files[:5])}")
                    if len(bible_files) > 5:
                        print(f"  ... and {len(json_files) - 5} more files")
        
        return
    
    print(f"\nFound {len(found_dirs)} export directories:")
    for dir_name, file_count in found_dirs:
        print(f"  - {dir_name}: {file_count} JSON files")
    
    # Analyze the first directory with files
    if found_dirs:
        export_dir = found_dirs[0][0]
        print(f"\nAnalyzing coverage in: {export_dir}")
        print("-" * 50)
        
        coverage_stats = analyze_export_coverage(export_dir)
        
        # NT books for Greek checking
        nt_books = [
            'Matt', 'Mark', 'Luke', 'John', 'Acts', 'Rom', '1Cor', '2Cor', 'Gal', 'Eph',
            'Phil', 'Col', '1Thess', '2Thess', '1Tim', '2Tim', 'Titus', 'Phlm', 'Heb',
            'Jas', '1Pet', '2Pet', '1John', '2John', '3John', 'Jude', 'Rev'
        ]
        
        # Check Greek coverage for NT books
        greek_coverage_found = False
        for book in nt_books:
            if book in coverage_stats:
                stats = coverage_stats[book]
                if stats['greek_verses'] > 0:
                    greek_coverage_found = True
                    pct = (stats['greek_verses'] / stats['total_verses'] * 100) if stats['total_verses'] > 0 else 0
                    print(f"{book}: {stats['greek_verses']}/{stats['total_verses']} verses with Greek ({pct:.1f}%)")
        
        if not greek_coverage_found:
            print("\nNo Greek coverage found in any NT books!")
            print("\nChecking first verse of Matthew for available fields...")
            
            matt_file = Path(export_dir) / "Matt.json"
            if matt_file.exists():
                with open(matt_file, 'r') as f:
                    data = json.load(f)
                    if data.get('chapters') and data['chapters'][0].get('verses'):
                        first_verse = data['chapters'][0]['verses'][0]
                        print(f"Fields in Matt 1:1: {list(first_verse.keys())}")
                        
                        # Check if morphology is embedded differently
                        if 'words' in first_verse:
                            print(f"  'words' field found with {len(first_verse['words'])} items")
                            if first_verse['words']:
                                print(f"  First word: {first_verse['words'][0]}")

if __name__ == '__main__':
    main()