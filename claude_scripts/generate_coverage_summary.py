#!/usr/bin/env python3
"""
Generate a summary of alignment coverage across translations.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def main():
    # Check models
    model_dir = Path('models/biblical_alignment')
    hebrew_model = model_dir / 'hebrew_english_biblical.json'
    greek_model = model_dir / 'greek_english_biblical.json'
    
    print("# Translation Coverage Analysis Report\n")
    print("## Model Summary\n")
    
    # Hebrew model
    if hebrew_model.exists():
        with open(hebrew_model, 'r') as f:
            data = json.load(f)
            mappings = len(data.get('strongs_mappings', {}))
            print(f"### Hebrew-English Model")
            print(f"- Strong's mappings: {mappings:,}")
            print(f"- Features: {', '.join(k for k,v in data.get('features', {}).items() if v)}")
            print()
    
    # Greek model  
    if greek_model.exists():
        with open(greek_model, 'r') as f:
            data = json.load(f)
            mappings = len(data.get('strongs_mappings', {}))
            print(f"### Greek-English Model")
            print(f"- Strong's mappings: {mappings:,}")
            print(f"- Features: {', '.join(k for k,v in data.get('features', {}).items() if v)}")
            print()
    
    # Sample translations
    trans_dir = Path('data/sources/translations')
    translations = list(trans_dir.glob('*.json'))
    
    print(f"## Translation Summary\n")
    print(f"Total translation files found: {len(translations)}\n")
    
    # Sample a few
    samples = ['eng_kjv.json', 'eng_web.json', 'eng_asv.json', 'eng_bbe.json']
    
    print("### Sample Analysis\n")
    print("| Translation | Language | Books | Verses | Words |")
    print("|-------------|----------|-------|--------|-------|")
    
    for sample in samples:
        trans_path = trans_dir / sample
        if trans_path.exists():
            with open(trans_path, 'r') as f:
                data = json.load(f)
                
            name = data.get('name', 'Unknown')
            lang = data.get('language', 'Unknown')
            books = data.get('books', {})
            
            verse_count = 0
            word_count = 0
            
            for book_data in books.values():
                for chapter in book_data.get('chapters', []):
                    for verse in chapter.get('verses', []):
                        verse_count += 1
                        word_count += len(verse.get('text', '').split())
                        
            print(f"| {name[:30]}... | {lang} | {len(books)} | {verse_count:,} | {word_count:,} |")
    
    print("\n## Current Limitations\n")
    print("1. **Limited Strong's Mappings**: Models currently have limited word mappings")
    print("2. **Training Data**: Models need more parallel text data for comprehensive coverage")
    print("3. **Translation Format**: Working with standardized JSON format from eBible.org")
    print()
    
    print("## Recommendations\n")
    print("1. **Expand Training Corpus**: Add more parallel texts with Strong's numbers")
    print("2. **Import Full Strong's**: Load complete Strong's concordance mappings")
    print("3. **Manual Alignments**: Add high-frequency word alignments manually")
    print("4. **Iterative Training**: Train models on aligned output to improve coverage")
    print()
    
    print("## Next Steps\n")
    print("1. Load full Strong's concordance into models")
    print("2. Create manual alignment files for common words")
    print("3. Implement progressive alignment training")
    print("4. Generate per-translation coverage reports")

if __name__ == '__main__':
    main()