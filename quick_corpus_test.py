#!/usr/bin/env python3
"""
Quick test with just Genesis and Matthew to verify statistical improvement.
"""

import subprocess
import sys
import json
from pathlib import Path

# Temporarily modify the corpus builder
corpus_script = Path('scripts/prepare_parallel_corpus.py')
content = corpus_script.read_text()

# Replace book lists
new_content = content.replace(
    "ot_books = [\n            'Gen', 'Exod', 'Lev', 'Num', 'Deut',  # Torah\n            'Ps',   # Psalms\n            'Isa'   # Isaiah\n        ]",
    "ot_books = ['Gen']  # Just Genesis for quick test"
).replace(
    "nt_books = [\n            'Matt', 'Mark', 'Luke', 'John',  # Gospels\n            'Acts',  # Acts\n            'Rom',   # Romans\n            'Rev'    # Revelation\n        ]",
    "nt_books = ['Matt']  # Just Matthew for quick test"
)

# Write temporary version
temp_script = Path('scripts/prepare_parallel_corpus_temp.py')
temp_script.write_text(new_content)

try:
    # Run corpus preparation
    print("Preparing corpus with Genesis and Matthew (full chapters)...")
    result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Run training
    print("\nTraining statistical models...")
    result = subprocess.run([sys.executable, "scripts/train_statistical_alignment.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check corpus size
    corpus_dir = Path('data/parallel_corpus')
    hebrew_lines = len((corpus_dir / 'hebrew_english.source').read_text().strip().split('\n'))
    greek_lines = len((corpus_dir / 'greek_english.source').read_text().strip().split('\n'))
    
    print(f"\nCorpus size: {hebrew_lines} Hebrew verses, {greek_lines} Greek verses")
    
    # Check model quality
    hebrew_model = json.loads((Path('models/statistical/hebrew_english_alignment.json')).read_text())
    greek_model = json.loads((Path('models/statistical/greek_english_alignment.json')).read_text())
    
    print(f"Hebrew vocabulary: {hebrew_model['source_vocab_size']} words")
    print(f"Greek vocabulary: {greek_model['source_vocab_size']} words")
    
    print("\nNow run 'python src/main.py' to test the improved alignment.")
    
finally:
    # Clean up
    temp_script.unlink()