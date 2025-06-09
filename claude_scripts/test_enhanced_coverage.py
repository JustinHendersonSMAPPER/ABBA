#!/usr/bin/env python3
"""
Test coverage with enhanced models.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_coverage():
    """Test coverage with enhanced models."""
    
    # Load enhanced Hebrew model
    hebrew_model_path = Path('models/biblical_alignment/hebrew_english_enhanced.json')
    if not hebrew_model_path.exists():
        print("Enhanced Hebrew model not found")
        return
        
    with open(hebrew_model_path, 'r') as f:
        hebrew_model = json.load(f)
        
    print(f"Hebrew model loaded:")
    print(f"  Strong's entries: {len(hebrew_model['strongs_mappings']):,}")
    print(f"  Manual entries: {len(hebrew_model['manual_mappings'])}")
    print(f"  High-frequency words: {len(hebrew_model['high_frequency_words'])}")
    
    # Test on Genesis 1:1
    test_verse = "In the beginning God created the heaven and the earth"
    words = test_verse.lower().split()
    
    print(f"\nTesting Genesis 1:1: '{test_verse}'")
    print(f"Total words: {len(words)}")
    
    # Check coverage using high-frequency words
    covered = []
    uncovered = []
    
    for word in words:
        if word in hebrew_model['high_frequency_words']:
            covered.append(word)
            strongs = hebrew_model['high_frequency_words'][word]['strongs']
            print(f"  ✓ '{word}' -> {strongs}")
        else:
            # Check if it's a common English word
            if word in ['in', 'the', 'and']:
                covered.append(word)
                print(f"  ✓ '{word}' (common word)")
            else:
                uncovered.append(word)
                print(f"  ✗ '{word}'")
                
    coverage = len(covered) / len(words) * 100
    print(f"\nCoverage: {coverage:.1f}% ({len(covered)}/{len(words)} words)")
    
    # Show some manual alignments
    print("\nSample manual alignments:")
    for i, (strongs, data) in enumerate(hebrew_model['manual_mappings'].items()):
        if i >= 5:
            break
        print(f"  {strongs}: {data['lemma']} ({data['translit']}) -> {', '.join(data['primary_translations'])}")
        
    # Test full translation
    print("\n" + "="*60)
    test_full_translation()
    

def test_full_translation():
    """Test on a full translation sample."""
    
    # Load KJV sample
    kjv_path = Path('data/sources/translations/eng_kjv.json')
    if not kjv_path.exists():
        print("KJV translation not found")
        return
        
    # Load enhanced models
    hebrew_model_path = Path('models/biblical_alignment/hebrew_english_enhanced.json')
    greek_model_path = Path('models/biblical_alignment/greek_english_enhanced.json')
    
    with open(hebrew_model_path, 'r') as f:
        hebrew_model = json.load(f)
        
    with open(greek_model_path, 'r') as f:
        greek_model = json.load(f)
        
    # Create combined high-frequency word list
    high_freq_words = set()
    high_freq_words.update(hebrew_model['high_frequency_words'].keys())
    high_freq_words.update(greek_model['high_frequency_words'].keys())
    
    # Common English words that don't need alignment
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'that', 'this', 'these', 'those', 'as', 'if', 'when', 'where', 'why',
        'how', 'what', 'which', 'who', 'whom', 'whose', 'shall', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'cannot',
        'unto', 'upon', 'into', 'within', 'without', 'through', 'between',
        'there', 'their', 'them', 'they', 'thou', 'thee', 'thy', 'thine',
        'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your',
        'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its',
        'also', 'even', 'so', 'then', 'therefore', 'thus', 'hence', 'yet',
        'nor', 'neither', 'either', 'both', 'all', 'any', 'some', 'such',
        'no', 'not', 'none', 'nothing', 'never', 'ever', 'very', 'more',
        'most', 'less', 'least', 'much', 'many', 'few', 'little', 'great'
    }
    
    # Test on first chapter of Genesis
    with open(kjv_path, 'r') as f:
        kjv_data = json.load(f)
        
    genesis = kjv_data['books']['Gen']
    chapter1 = genesis['chapters'][0]
    
    total_words = 0
    covered_words = 0
    word_freq = Counter()
    uncovered_freq = Counter()
    
    print("\nAnalyzing Genesis Chapter 1...")
    
    for verse in chapter1['verses']:
        words = verse['text'].lower().split()
        words = [w.strip('.,;:!?') for w in words]  # Remove punctuation
        
        for word in words:
            if not word:
                continue
                
            total_words += 1
            word_freq[word] += 1
            
            if word in high_freq_words or word in common_words:
                covered_words += 1
            else:
                uncovered_freq[word] += 1
                
    coverage = covered_words / total_words * 100 if total_words > 0 else 0
    
    print(f"\nGenesis 1 Statistics:")
    print(f"  Total words: {total_words:,}")
    print(f"  Unique words: {len(word_freq)}")
    print(f"  Covered words: {covered_words:,}")
    print(f"  Coverage: {coverage:.1f}%")
    
    print(f"\nMost frequent uncovered words:")
    for word, count in uncovered_freq.most_common(10):
        print(f"  {word}: {count} occurrences")
        
    # Calculate weighted coverage
    total_occurrences = sum(word_freq.values())
    covered_occurrences = sum(count for word, count in word_freq.items() 
                             if word in high_freq_words or word in common_words)
    weighted_coverage = covered_occurrences / total_occurrences * 100
    
    print(f"\nWeighted coverage: {weighted_coverage:.1f}%")
    print("(This accounts for word frequency)")


if __name__ == '__main__':
    test_coverage()