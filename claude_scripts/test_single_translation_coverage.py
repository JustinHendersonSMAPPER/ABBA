#!/usr/bin/env python3
"""
Test coverage analysis on a single translation.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from abba_align.coverage_analyzer import AlignmentCoverageAnalyzer
from abba_align.model_info import ModelDiscovery

def main():
    # Test with KJV
    trans_path = Path('data/sources/translations/eng_kjv.json')
    
    if not trans_path.exists():
        print(f"Translation not found: {trans_path}")
        return
        
    # Load translation
    with open(trans_path, 'r') as f:
        data = json.load(f)
        
    print(f"Translation: {data.get('name', 'Unknown')}")
    print(f"Language: {data.get('language', 'Unknown')}")
    
    # Get books
    books = data.get('books', {})
    book_names = list(books.keys())
    print(f"Books: {len(book_names)}")
    
    # Check first book
    if book_names:
        first_book = books[book_names[0]].get('name', '')
        print(f"First book: {first_book}")
        
        # Determine source language
        if first_book in ['Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans']:
            source_lang = 'greek'
            testament = 'NT'
        else:
            source_lang = 'hebrew' 
            testament = 'OT'
            
        print(f"Testament: {testament}")
        print(f"Source language: {source_lang}")
        
        # Find model
        discovery = ModelDiscovery()
        model = discovery.find_model(source_lang, 'english')
        
        if model:
            print(f"Model found: {model.name}")
            print(f"Model path: {model.path}")
            
            # Initialize analyzer
            analyzer = AlignmentCoverageAnalyzer(source_lang=source_lang)
            
            # Load Strong's mappings from model
            if model.path.exists():
                with open(model.path, 'r') as f:
                    model_data = json.load(f)
                    strongs_count = len(model_data.get('strongs_mappings', {}))
                    print(f"Strong's mappings in model: {strongs_count}")
                    
                    # Load mappings
                    if 'strongs_mappings' in model_data:
                        for strongs_num, word_dict in model_data['strongs_mappings'].items():
                            analyzer.strongs_integration.strongs_to_english[strongs_num] = Counter(word_dict)
                            
            # Process just first chapter of Genesis
            if 'Gen' in books:
                gen = books['Gen']
                if gen.get('chapters'):
                    chapter1 = gen['chapters'][0]
                    print(f"\nAnalyzing Genesis 1 ({len(chapter1.get('verses', []))} verses)...")
                    
                    word_counts = Counter()
                    aligned_words = set()
                    pos_counts = defaultdict(Counter)
                    aligned_by_pos = defaultdict(set)
                    verse_stats = []
                    
                    for verse in chapter1.get('verses', []):
                        verse_data = {
                            'book': 'Gen',
                            'chapter': 1,
                            'verse': verse['verse'],
                            'text': verse['text']
                        }
                        
                        result = analyzer._analyze_verse(
                            verse_data, word_counts, pos_counts,
                            aligned_words, aligned_by_pos
                        )
                        verse_stats.append(result)
                        
                        if verse['verse'] <= 3:
                            print(f"  Verse {verse['verse']}: {result['coverage']:.1f}% coverage ({result['covered']}/{result['total_words']} words)")
                            if result['uncovered']:
                                print(f"    Uncovered: {', '.join(result['uncovered'][:5])}")
                    
                    # Calculate stats
                    stats = analyzer._calculate_statistics(
                        word_counts, pos_counts, aligned_words,
                        aligned_by_pos, verse_stats
                    )
                    
                    print(f"\nGenesis 1 Summary:")
                    print(f"  Token coverage: {stats['summary']['token_coverage']:.1f}%")
                    print(f"  Type coverage: {stats['summary']['type_coverage']:.1f}%")
                    print(f"  Total tokens: {stats['summary']['total_tokens']}")
                    print(f"  Total types: {stats['summary']['total_types']}")
        else:
            print(f"No model found for {source_lang}")

if __name__ == '__main__':
    main()