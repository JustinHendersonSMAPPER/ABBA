#!/usr/bin/env python3
"""
Train word alignment models on full corpus files.

This script trains separate alignment models for each corpus (e.g., Hebrew, Greek, 
or translation) found as JSON files in the input directory. Each corpus gets its 
own trained model saved as {filename}_model.json.

Usage:
    python train_full_corpus.py --input-dir data/corpora --output-dir models/alignment
    
This will process each JSON file in data/corpora/ and create corresponding models
like hebrew_model.json, greek_model.json, eng_kjv_model.json, etc.
"""

import argparse
import json
import logging
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.word_alignment import IBMModel1

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_corpus_data(corpus_file: Path) -> List[Tuple[str, str]]:
    """
    Load text data from a corpus JSON file in translation format.
    
    Expected format (same as translations):
    {
        "version": "HEB_BHS",
        "name": "Biblia Hebraica Stuttgartensia",
        "language": "he",
        "books": {
            "Gen": {
                "chapters": [
                    {
                        "chapter": 1,
                        "verses": [
                            {"verse": 1, "text": "בְּרֵאשִׁית..."},
                            ...
                        ]
                    }
                ]
            }
        }
    }
    
    Returns list of (verse_id, text) tuples for training.
    """
    logger.info(f"Loading corpus from {corpus_file.name}...")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text_pairs = []
    
    # Extract version info
    version = data.get('version', corpus_file.stem)
    language = data.get('language', 'unknown')
    
    # Process books
    for book_name, book_data in data.get('books', {}).items():
        for chapter in book_data.get('chapters', []):
            chapter_num = chapter['chapter']
            
            for verse in chapter.get('verses', []):
                verse_num = verse['verse']
                text = verse.get('text', '')
                
                if text:
                    # Create verse ID
                    verse_id = f"{book_name}.{chapter_num}.{verse_num}"
                    text_pairs.append((verse_id, text))
    
    logger.info(f"Loaded {len(text_pairs)} verses from {corpus_file.name} ({language})")
    return text_pairs


def train_model_for_corpus(corpus_file: Path, output_dir: Path, 
                          iterations: int = 10, batch_size: int = 5000):
    """
    Train an alignment model for a single corpus file.
    
    Args:
        corpus_file: Path to corpus JSON file
        output_dir: Directory to save trained model
        iterations: Number of EM iterations
        batch_size: Size of training batches for memory efficiency
    """
    # Load corpus data
    parallel_pairs = load_corpus_data(corpus_file)
    
    if not parallel_pairs:
        logger.warning(f"No parallel data found in {corpus_file.name}, skipping...")
        return
    
    # Initialize model
    logger.info(f"Initializing model for {corpus_file.stem}...")
    model = IBMModel1(
        use_morphology=True,
        use_strongs=True,
        use_lexicon=True
    )
    model.num_iterations = iterations
    
    # Train in batches for memory efficiency
    if len(parallel_pairs) > batch_size:
        logger.info(f"Training in batches of {batch_size} due to large corpus size...")
        
        for i in range(0, len(parallel_pairs), batch_size):
            batch = parallel_pairs[i:i + batch_size]
            batch_end = min(i + batch_size, len(parallel_pairs))
            
            logger.info(f"Training batch {i//batch_size + 1} "
                       f"(verses {i+1}-{batch_end} of {len(parallel_pairs)})...")
            
            # Train on this batch
            model.train(batch, verbose=True)
            
            # Clear memory
            del batch
            gc.collect()
    else:
        # Train on full dataset if small enough
        logger.info(f"Training on full dataset ({len(parallel_pairs)} pairs)...")
        model.train(parallel_pairs, verbose=True)
    
    # Save model
    output_file = output_dir / f"{corpus_file.stem}_model.json"
    logger.info(f"Saving model to {output_file}...")
    model.save_model(output_file)
    
    # Log statistics
    logger.info(f"Model statistics for {corpus_file.stem}:")
    logger.info(f"  Source vocabulary: {len(model.source_vocab)} words")
    logger.info(f"  Target vocabulary: {len(model.target_vocab)} words")
    logger.info(f"  Translation pairs: {sum(len(model.trans_probs[s]) for s in model.source_vocab)}")
    
    # Clear memory
    del parallel_pairs
    del model
    gc.collect()


def evaluate_model(model_file: Path, test_pairs: List[Tuple[List[Dict], str]], 
                  num_samples: int = 5):
    """Evaluate a trained model on test samples."""
    logger.info(f"\nEvaluating {model_file.name}...")
    
    model = IBMModel1()
    model.load_model(model_file)
    
    total_score = 0.0
    for i, (source_words, target_text) in enumerate(test_pairs[:num_samples]):
        alignment = model.align_verse(source_words, target_text)
        total_score += alignment.alignment_score
        
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  Target: {target_text[:80]}...")
        logger.info(f"  Alignment score: {alignment.alignment_score:.3f}")
        logger.info(f"  Mapped {len(alignment.alignments)} of {len(alignment.target_words)} words")
        
        if alignment.alignments:
            logger.info("  Top alignments:")
            for align in alignment.alignments[:3]:
                logger.info(f"    {align.source_word} -> {align.target_word} "
                           f"(conf: {align.confidence:.3f})")
    
    avg_score = total_score / min(num_samples, len(test_pairs))
    logger.info(f"\nAverage alignment score: {avg_score:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train word alignment models for each corpus file"
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing corpus JSON files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/alignment'),
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of EM iterations for training (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5000,
        help='Batch size for memory-efficient training (default: 5000)'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate models after training'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in input directory
    corpus_files = list(args.input_dir.glob("*.json"))
    
    if not corpus_files:
        logger.error(f"No JSON files found in {args.input_dir}")
        return 1
    
    logger.info(f"Found {len(corpus_files)} corpus files to process")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training parameters: {args.iterations} iterations, batch size {args.batch_size}")
    logger.info("="*60)
    
    # Train model for each corpus
    for corpus_file in sorted(corpus_files):
        logger.info(f"\nProcessing {corpus_file.name}")
        logger.info("-"*40)
        
        try:
            train_model_for_corpus(
                corpus_file, 
                args.output_dir,
                iterations=args.iterations,
                batch_size=args.batch_size
            )
        except Exception as e:
            logger.error(f"Error processing {corpus_file.name}: {e}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info(f"Models saved to {args.output_dir}")
    
    # Optional evaluation
    if args.evaluate:
        logger.info("\n" + "="*60)
        logger.info("Evaluating trained models...")
        
        for model_file in sorted(args.output_dir.glob("*_model.json")):
            # Load corresponding corpus for test data
            corpus_name = model_file.stem.replace("_model", "")
            corpus_file = args.input_dir / f"{corpus_name}.json"
            
            if corpus_file.exists():
                test_pairs = load_corpus_data(corpus_file)
                if test_pairs:
                    evaluate_model(model_file, test_pairs[-100:])  # Use last 100 pairs
    
    return 0


if __name__ == "__main__":
    sys.exit(main())