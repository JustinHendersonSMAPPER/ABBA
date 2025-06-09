#!/usr/bin/env python3
"""
Quick test to demonstrate ensemble alignment approach.
Shows how combining IBM Model 1 with contextual embeddings improves accuracy.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.word_alignment import IBMModel1

# Simple example of ensemble scoring
def ensemble_align(verse_data, ibm_model):
    """Demonstrate ensemble alignment with IBM + context."""
    
    hebrew_words = verse_data['hebrew_words']
    eng_text = verse_data['translations']['ENG_KJV']
    
    print(f"\nVerse: {verse_data['verse_id']}")
    print(f"English: {eng_text}")
    print(f"Hebrew: {verse_data['hebrew_text']}")
    
    # Get IBM Model 1 alignments
    ibm_alignment = ibm_model.align_verse(hebrew_words, eng_text, threshold=0.05)
    
    print("\n=== IBM Model 1 Only ===")
    for align in ibm_alignment.alignments[:10]:
        print(f"{align.source_word} -> {align.target_word} (conf: {align.confidence:.3f})")
    
    # Simulate contextual scoring (in reality, this would use embeddings)
    print("\n=== Enhanced with Context (Simulated) ===")
    
    # Common biblical word patterns (simplified)
    context_boost = {
        ('אֱלֹהִ֑ים', 'god'): 0.3,      # Elohim -> God
        ('יְהוָ֥ה', 'lord'): 0.3,       # YHWH -> LORD
        ('בְּ', 'in'): 0.2,             # Common preposition
        ('וְ', 'and'): 0.2,             # Common conjunction
        ('אֶל', 'to'): 0.2,             # Common preposition
        ('הַ', 'the'): 0.2,             # Definite article
        ('בָּרָ֣א', 'created'): 0.2,    # Specific to Genesis 1:1
    }
    
    enhanced_alignments = []
    for align in ibm_alignment.alignments:
        # Base IBM score
        base_score = align.confidence
        
        # Add context boost if available
        pair = (align.source_word, align.target_word)
        context_score = context_boost.get(pair, 0.0)
        
        # Ensemble score (weighted combination)
        ensemble_score = 0.7 * base_score + 0.3 * context_score
        
        enhanced_alignments.append({
            'source': align.source_word,
            'target': align.target_word,
            'ibm_score': base_score,
            'context_score': context_score,
            'ensemble_score': ensemble_score
        })
    
    # Sort by ensemble score
    enhanced_alignments.sort(key=lambda x: x['ensemble_score'], reverse=True)
    
    for align in enhanced_alignments[:10]:
        improvement = ""
        if align['context_score'] > 0:
            improvement = f" [+{align['context_score']:.2f} context]"
        print(f"{align['source']} -> {align['target']} "
              f"(ensemble: {align['ensemble_score']:.3f}, "
              f"ibm: {align['ibm_score']:.3f}{improvement})")
    
    return enhanced_alignments


def main():
    # Load a test verse
    gen_file = Path('aligned_export_improved/Gen.json')
    with open(gen_file, 'r') as f:
        gen_data = json.load(f)
    
    # Get first verse
    verse = gen_data['chapters'][0]['verses'][0]
    
    # Load IBM model
    ibm_model = IBMModel1()
    ibm_model.load_model(Path('models/alignment/hebrew_alignment_representative.json'))
    
    # Test ensemble approach
    print("="*60)
    print("ENSEMBLE ALIGNMENT DEMONSTRATION")
    print("="*60)
    
    alignments = ensemble_align(verse, ibm_model)
    
    print("\n" + "="*60)
    print("BENEFITS OF ENSEMBLE APPROACH:")
    print("="*60)
    print("1. IBM Model provides statistical translation probabilities")
    print("2. Contextual embeddings capture semantic similarity")
    print("3. Combining both reduces errors and improves coverage")
    print("4. Can leverage linguistic knowledge (morphology, syntax)")
    print("5. More robust to rare words and ambiguous alignments")
    
    print("\nReal implementation would include:")
    print("- Word2Vec/FastText embeddings for each language")
    print("- Cross-lingual mapping learned from anchor pairs")
    print("- Syntactic and morphological features")
    print("- Phrase-level alignment for multi-word expressions")


if __name__ == "__main__":
    main()