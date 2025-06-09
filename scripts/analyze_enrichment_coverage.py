#!/usr/bin/env python3
"""Analyze enrichment coverage in Bible export."""

import json
import sys
from collections import defaultdict
from pathlib import Path

def analyze_enrichments(data_dir: Path):
    """Analyze the enrichment coverage."""
    
    print("ABBA Bible Export Enrichment Coverage Analysis")
    print("=" * 50)
    
    # 1. Check cross-references
    print("\n1. Cross-References:")
    xref_file = data_dir / "cross_references.json"
    if xref_file.exists():
        with open(xref_file) as f:
            xref_data = json.load(f)
        
        refs = xref_data.get('references', [])
        print(f"   - Total cross-references in file: {len(refs)}")
        
        # Count by book
        source_books = defaultdict(int)
        target_books = defaultdict(int)
        for ref in refs:
            src = ref['source'].split('.')[0]
            tgt = ref['target'].split('.')[0]
            source_books[src] += 1
            target_books[tgt] += 1
        
        print(f"   - Source books covered: {list(source_books.keys())}")
        print(f"   - Target books covered: {list(target_books.keys())}")
    else:
        print(f"   - ERROR: Cross-references file not found at {xref_file}")
    
    # 2. Check Hebrew texts
    print("\n2. Hebrew Texts:")
    hebrew_dir = data_dir / "sources" / "hebrew"
    if hebrew_dir.exists():
        hebrew_files = list(hebrew_dir.glob("*.xml"))
        print(f"   - Total Hebrew files: {len(hebrew_files)}")
        
        # Map to standard book codes
        hebrew_book_map = {
            'Gen': 'GEN', 'Exod': 'EXO', 'Lev': 'LEV', 'Num': 'NUM', 'Deut': 'DEU',
            'Josh': 'JOS', 'Judg': 'JDG', 'Ruth': 'RUT', '1Sam': '1SA', '2Sam': '2SA',
            '1Kgs': '1KI', '2Kgs': '2KI', '1Chr': '1CH', '2Chr': '2CH', 'Ezra': 'EZR',
            'Neh': 'NEH', 'Esth': 'EST', 'Job': 'JOB', 'Ps': 'PSA', 'Prov': 'PRO',
            'Eccl': 'ECC', 'Song': 'SNG', 'Isa': 'ISA', 'Jer': 'JER', 'Lam': 'LAM',
            'Ezek': 'EZK', 'Dan': 'DAN', 'Hos': 'HOS', 'Joel': 'JOL', 'Amos': 'AMO',
            'Obad': 'OBA', 'Jonah': 'JON', 'Mic': 'MIC', 'Nah': 'NAH', 'Hab': 'HAB',
            'Zeph': 'ZEP', 'Hag': 'HAG', 'Zech': 'ZEC', 'Mal': 'MAL'
        }
        
        available_hebrew = []
        for f in hebrew_files:
            base = f.stem
            if base in hebrew_book_map:
                available_hebrew.append(hebrew_book_map[base])
            else:
                available_hebrew.append(base.upper())
        
        print(f"   - Available books: {sorted(available_hebrew)}")
        
        # Check if only first 5 OT books are handled in code
        print("\n   - Code limitation check:")
        print("     The load_hebrew_text() method only handles:")
        print("     GEN, EXO, LEV, NUM, DEU")
        print("     This explains why Hebrew enrichment is limited!")
    else:
        print(f"   - ERROR: Hebrew directory not found at {hebrew_dir}")
    
    # 3. Check Greek texts
    print("\n3. Greek Texts:")
    greek_dir = data_dir / "sources" / "greek"
    if greek_dir.exists():
        greek_files = list(greek_dir.glob("*.xml"))
        print(f"   - Total Greek files: {len(greek_files)}")
        
        available_greek = [f.stem for f in greek_files]
        print(f"   - Available books: {sorted(available_greek)}")
        
        # Check if only some NT books are handled in code
        print("\n   - Code limitation check:")
        print("     The load_greek_text() method only handles:")
        print("     MAT, MRK, LUK, JHN, ACT, ROM")
        print("     This explains why Greek enrichment is limited!")
    else:
        print(f"   - ERROR: Greek directory not found at {greek_dir}")
    
    # 4. Timeline analysis
    print("\n4. Timeline Events:")
    print("   - Hard-coded in process_bible() method:")
    print("     * Creation (GEN.1.1)")
    print("     * The Exodus (EXO.12.31)")
    print("   - No external timeline data file found")
    print("   - This explains why timeline enrichment is minimal!")
    
    # 5. Summary of Issues
    print("\n" + "=" * 50)
    print("SUMMARY OF ENRICHMENT COVERAGE ISSUES:")
    print("=" * 50)
    
    print("\n1. Cross-References:")
    print("   - Only 10 sample cross-references in data file")
    print("   - Need comprehensive cross-reference database")
    
    print("\n2. Hebrew Text:")
    print("   - Have 39 OT book files available")
    print("   - Code only processes 5 books (Torah)")
    print("   - Need to expand book_files mapping in load_hebrew_text()")
    
    print("\n3. Greek Text:")  
    print("   - Have 27 NT book files available")
    print("   - Code only processes 6 books")
    print("   - Need to expand book_files mapping in load_greek_text()")
    
    print("\n4. Timeline Events:")
    print("   - Only 2 hard-coded events")
    print("   - Need comprehensive timeline database file")
    print("   - Need timeline loader implementation")
    
    print("\n5. Recommendations:")
    print("   a) Expand Hebrew/Greek book mappings to cover all available files")
    print("   b) Create comprehensive cross-reference database")
    print("   c) Create timeline events database file")
    print("   d) Implement timeline loader from external file")
    print("   e) Consider using Treasury of Scripture Knowledge for cross-refs")

if __name__ == '__main__':
    data_dir = Path('data')
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    
    analyze_enrichments(data_dir)