#!/usr/bin/env python3
"""Quick test of alignment models."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.abba.alignment.word_alignment import IBMModel1

# Load models
print("Loading models...")
hebrew_model = IBMModel1()
hebrew_model.load_model(Path("models/alignment/hebrew_alignment.json"))
print(f"Hebrew model vocabulary: {len(hebrew_model.source_vocab)} source, {len(hebrew_model.target_vocab)} target")

greek_model = IBMModel1()  
greek_model.load_model(Path("models/alignment/greek_alignment.json"))
print(f"Greek model vocabulary: {len(greek_model.source_vocab)} source, {len(greek_model.target_vocab)} target")

# Test on a verse
gen_file = Path("full_fixed_export/Gen.json")
with open(gen_file, 'r') as f:
    gen_data = json.load(f)

verse = gen_data['chapters'][0]['verses'][0]
print(f"\nTesting on {verse['verse_id']}")
print(f"Hebrew words: {len(verse['hebrew_words'])}")
print(f"English: {verse['translations']['eng_kjv']}")

alignment = hebrew_model.align_verse(verse['hebrew_words'], verse['translations']['eng_kjv'])
print(f"\nAlignment results:")
print(f"  Source words: {len(alignment.source_words)}")
print(f"  Target words: {len(alignment.target_words)}")
print(f"  Alignments: {len(alignment.alignments)}")
print(f"  Score: {alignment.alignment_score:.3f}")

if alignment.alignments:
    print("\nSample alignments:")
    for align in alignment.alignments[:5]:
        print(f"  {align.source_word} -> {align.target_word} (conf: {align.confidence:.3f})")