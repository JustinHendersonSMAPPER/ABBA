#!/usr/bin/env python3
"""
Test alignment coverage with the newly trained representative models.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.word_alignment import IBMModel1

# Load models
print("Loading trained models...")
hebrew_model = IBMModel1()
hebrew_model.load_model(Path("models/alignment/hebrew_alignment_representative.json"))

greek_model = IBMModel1()
greek_model.load_model(Path("models/alignment/greek_alignment_representative.json"))

# Test on Genesis 1:1
gen_file = Path("full_fixed_export/Gen.json")
with open(gen_file, 'r') as f:
    gen_data = json.load(f)

verse = gen_data['chapters'][0]['verses'][0]
print(f"\n=== Testing {verse['verse_id']} ===")
print(f"English: {verse['translations']['eng_kjv']}")
print(f"Hebrew: {verse['hebrew_text']}")

# Get alignment
alignment = hebrew_model.align_verse(verse['hebrew_words'], verse['translations']['eng_kjv'])
print(f"\nAlignment score: {alignment.alignment_score:.3f}")
print(f"Mapped {len(alignment.alignments)} of {len(alignment.target_words)} English words")

print("\nAlignments:")
for align in alignment.alignments:
    print(f"  {align.source_word} -> {align.target_word} (conf: {align.confidence:.3f})")

# Test on Matthew 1:1
matt_file = Path("full_fixed_export/Matt.json")
with open(matt_file, 'r') as f:
    matt_data = json.load(f)

verse = matt_data['chapters'][0]['verses'][0]
print(f"\n=== Testing {verse['verse_id']} ===")
print(f"English: {verse['translations']['eng_kjv']}")
print(f"Greek: {verse['greek_text']}")

# Get alignment
alignment = greek_model.align_verse(verse['greek_words'], verse['translations']['eng_kjv'])
print(f"\nAlignment score: {alignment.alignment_score:.3f}")
print(f"Mapped {len(alignment.alignments)} of {len(alignment.target_words)} English words")

print("\nAlignments:")
for align in alignment.alignments:
    print(f"  {align.source_word} -> {align.target_word} (conf: {align.confidence:.3f})")

# Quick coverage test on full Bible
print("\n=== Quick Coverage Analysis ===")
total_words = 0
mapped_words = 0
unmapped_counter = Counter()

for book_file in Path("full_fixed_export").glob("*.json"):
    if book_file.name.startswith("_"):
        continue
        
    with open(book_file, 'r') as f:
        book_data = json.load(f)
    
    # Just check first 10 verses of each book
    verse_count = 0
    for chapter in book_data.get('chapters', []):
        for verse in chapter.get('verses', []):
            eng_text = verse.get('translations', {}).get('eng_kjv', '')
            if not eng_text:
                continue
                
            if verse.get('hebrew_words'):
                alignment = hebrew_model.align_verse(verse['hebrew_words'], eng_text)
            elif verse.get('greek_words'):
                alignment = greek_model.align_verse(verse['greek_words'], eng_text)
            else:
                continue
            
            total_words += len(alignment.target_words)
            mapped_words += len(alignment.alignments)
            
            # Track unmapped words
            mapped_indices = {a.target_idx for a in alignment.alignments}
            for i, word in enumerate(alignment.target_words):
                if i not in mapped_indices:
                    unmapped_counter[word] += 1
            
            verse_count += 1
            if verse_count >= 10:
                break
        if verse_count >= 10:
            break

coverage = mapped_words / total_words * 100 if total_words > 0 else 0
print(f"\nSample Coverage: {coverage:.1f}% ({mapped_words}/{total_words} words)")
print(f"\nMost common unmapped words:")
for word, count in unmapped_counter.most_common(20):
    print(f"  {word}: {count}")