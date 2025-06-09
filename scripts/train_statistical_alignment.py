#!/usr/bin/env python3
"""
Train statistical word alignment models using simple co-occurrence statistics.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleStatisticalAligner:
    """Simple statistical word aligner based on co-occurrence."""
    
    def __init__(self):
        self.translation_probs = defaultdict(lambda: defaultdict(float))
        self.source_counts = Counter()
        self.target_counts = Counter()
        self.cooccurrence_counts = defaultdict(lambda: defaultdict(int))
    
    def train(self, source_lines: List[str], target_lines: List[str]) -> None:
        """Train alignment model from parallel corpus."""
        logger.info(f"Training on {len(source_lines)} sentence pairs...")
        
        # Count co-occurrences
        for src_line, tgt_line in zip(source_lines, target_lines):
            src_words = src_line.strip().split()
            tgt_words = tgt_line.strip().split()
            
            # Count words
            for word in src_words:
                self.source_counts[word] += 1
            for word in tgt_words:
                self.target_counts[word] += 1
            
            # Count co-occurrences
            for src_word in set(src_words):  # Use set to count each word once per sentence
                for tgt_word in set(tgt_words):
                    self.cooccurrence_counts[src_word][tgt_word] += 1
        
        # Calculate translation probabilities using PMI-like score
        logger.info("Calculating translation probabilities...")
        total_pairs = len(source_lines)
        
        for src_word, tgt_words in self.cooccurrence_counts.items():
            src_freq = self.source_counts[src_word] / total_pairs
            
            for tgt_word, cooccur_count in tgt_words.items():
                tgt_freq = self.target_counts[tgt_word] / total_pairs
                joint_freq = cooccur_count / total_pairs
                
                # PMI score (Pointwise Mutual Information)
                if joint_freq > 0 and src_freq > 0 and tgt_freq > 0:
                    pmi = np.log(joint_freq / (src_freq * tgt_freq))
                    # Normalize to [0, 1]
                    score = 1 / (1 + np.exp(-pmi))
                    self.translation_probs[src_word][tgt_word] = score
        
        logger.info(f"Learned translations for {len(self.translation_probs)} source words")
    
    def get_translation_prob(self, source_word: str, target_word: str) -> float:
        """Get translation probability for a word pair."""
        return self.translation_probs.get(source_word, {}).get(target_word, 0.0)
    
    def save(self, output_path: Path) -> None:
        """Save model to JSON file."""
        # Convert to regular dict for JSON serialization
        model_data = {
            'translation_probs': {
                src: dict(tgts) for src, tgts in self.translation_probs.items()
            },
            'source_vocab_size': len(self.source_counts),
            'target_vocab_size': len(self.target_counts),
            'source_top_words': self.source_counts.most_common(100),
            'target_top_words': self.target_counts.most_common(100)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved model to {output_path}")
    
    def load(self, model_path: Path) -> None:
        """Load model from JSON file."""
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.translation_probs = defaultdict(lambda: defaultdict(float))
        for src, tgts in model_data['translation_probs'].items():
            for tgt, prob in tgts.items():
                self.translation_probs[src][tgt] = prob
        
        logger.info(f"Loaded model with {len(self.translation_probs)} source words")


def train_models():
    """Train alignment models for Hebrew and Greek."""
    corpus_dir = Path('data/parallel_corpus')
    models_dir = Path('models/statistical')
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Train Hebrew-English model
    hebrew_source = corpus_dir / 'hebrew_english.source'
    hebrew_target = corpus_dir / 'hebrew_english.target'
    
    if hebrew_source.exists() and hebrew_target.exists():
        logger.info("Training Hebrew-English alignment model...")
        
        with open(hebrew_source, 'r', encoding='utf-8') as f:
            hebrew_lines = f.readlines()
        with open(hebrew_target, 'r', encoding='utf-8') as f:
            english_lines = f.readlines()
        
        hebrew_aligner = SimpleStatisticalAligner()
        hebrew_aligner.train(hebrew_lines, english_lines)
        hebrew_aligner.save(models_dir / 'hebrew_english_alignment.json')
        
        # Show sample translations
        logger.info("\nSample Hebrew-English translations:")
        sample_words = ['אֱלֹהִ֑ים', 'בָּרָ֣א', 'הָ/אָֽרֶץ', 'הַ/שָּׁמַ֖יִם']
        for word in sample_words:
            if word in hebrew_aligner.translation_probs:
                top_trans = sorted(
                    hebrew_aligner.translation_probs[word].items(),
                    key=lambda x: x[1], reverse=True
                )[:3]
                logger.info(f"  {word}: {[f'{t[0]} ({t[1]:.3f})' for t in top_trans]}")
    
    # Train Greek-English model
    greek_source = corpus_dir / 'greek_english.source'
    greek_target = corpus_dir / 'greek_english.target'
    
    if greek_source.exists() and greek_target.exists():
        logger.info("\nTraining Greek-English alignment model...")
        
        with open(greek_source, 'r', encoding='utf-8') as f:
            greek_lines = f.readlines()
        with open(greek_target, 'r', encoding='utf-8') as f:
            english_lines = f.readlines()
        
        greek_aligner = SimpleStatisticalAligner()
        greek_aligner.train(greek_lines, english_lines)
        greek_aligner.save(models_dir / 'greek_english_alignment.json')
        
        # Show sample translations
        logger.info("\nSample Greek-English translations:")
        sample_words = ['Ἰησοῦ', 'Χριστοῦ', 'θεοῦ', 'ἐν']
        for word in sample_words:
            if word in greek_aligner.translation_probs:
                top_trans = sorted(
                    greek_aligner.translation_probs[word].items(),
                    key=lambda x: x[1], reverse=True
                )[:3]
                logger.info(f"  {word}: {[f'{t[0]} ({t[1]:.3f})' for t in top_trans]}")


def main():
    """Main function."""
    train_models()


if __name__ == "__main__":
    main()