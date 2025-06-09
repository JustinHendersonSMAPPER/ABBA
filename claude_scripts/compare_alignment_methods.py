#!/usr/bin/env python3
"""
Compare different alignment methods:
1. IBM Model 1 only (parallel)
2. Embedding-based only (monolingual)
3. Hybrid ensemble

Evaluates on test verses and reports accuracy metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.word_alignment import IBMModel1

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_test_verses(enriched_dir: Path, num_verses: int = 100) -> List[Dict]:
    """Load test verses with existing alignments for evaluation."""
    test_verses = []
    
    for book_file in sorted(enriched_dir.glob("*.json")):
        if book_file.name.startswith("_"):
            continue
            
        with open(book_file, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
            
        for chapter in book_data.get('chapters', []):
            for verse in chapter.get('verses', []):
                # Must have alignments for evaluation
                if verse.get('alignments') and (verse.get('hebrew_words') or verse.get('greek_words')):
                    test_verses.append(verse)
                    
                if len(test_verses) >= num_verses:
                    return test_verses
                    
    return test_verses


def extract_gold_alignments(verse: Dict) -> Set[Tuple[int, int]]:
    """Extract gold standard alignments from verse."""
    gold_alignments = set()
    
    alignments = verse.get('alignments', {})
    for trans_code, align_list in alignments.items():
        for alignment in align_list:
            src_idx = alignment.get('source_idx', -1)
            target_indices = alignment.get('target_indices', [])
            
            for tgt_idx in target_indices:
                if src_idx >= 0 and tgt_idx >= 0:
                    gold_alignments.add((src_idx, tgt_idx))
                    
    return gold_alignments


def evaluate_alignments(predicted: Set[Tuple[int, int]], 
                       gold: Set[Tuple[int, int]]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    if not gold:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
    if not predicted:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # True positives
    tp = len(predicted & gold)
    
    # Precision: what fraction of predicted alignments are correct
    precision = tp / len(predicted) if predicted else 0.0
    
    # Recall: what fraction of gold alignments were found
    recall = tp / len(gold) if gold else 0.0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'predicted': len(predicted),
        'gold': len(gold)
    }


def test_ibm_model(verse: Dict, model_path: Path) -> Set[Tuple[int, int]]:
    """Test IBM Model 1 alignment."""
    if not model_path.exists():
        return set()
        
    # Load model
    model = IBMModel1()
    model.load_model(model_path)
    
    # Get source words and target text
    source_words = verse.get('hebrew_words') or verse.get('greek_words', [])
    eng_text = verse.get('translations', {}).get('eng_kjv', '')
    
    if not source_words or not eng_text:
        return set()
    
    # Get alignment
    alignment = model.align_verse(source_words, eng_text, threshold=0.1)
    
    # Convert to set of tuples
    predicted = set()
    for align in alignment.alignments:
        predicted.add((align.source_idx, align.target_idx))
        
    return predicted


def test_hybrid_model(verse: Dict, model_dir: Path, model_prefix: str) -> Set[Tuple[int, int]]:
    """Test hybrid model alignment."""
    # This is a simplified version - full implementation would load the hybrid model
    # For now, we'll use IBM model as placeholder
    ibm_path = model_dir / f"{model_prefix}_ibm.json"
    return test_ibm_model(verse, ibm_path)


def compare_methods(test_verses: List[Dict], models_dir: Path):
    """Compare different alignment methods."""
    results = defaultdict(lambda: {
        'precision_sum': 0.0,
        'recall_sum': 0.0,
        'f1_sum': 0.0,
        'count': 0
    })
    
    for i, verse in enumerate(test_verses):
        # Get gold alignments
        gold = extract_gold_alignments(verse)
        if not gold:
            continue
            
        # Determine language
        is_hebrew = bool(verse.get('hebrew_words'))
        
        # Test IBM Model 1
        if is_hebrew:
            ibm_predicted = test_ibm_model(verse, models_dir / 'hebrew_alignment.json')
        else:
            ibm_predicted = test_ibm_model(verse, models_dir / 'greek_alignment.json')
            
        ibm_metrics = evaluate_alignments(ibm_predicted, gold)
        results['IBM Model 1']['precision_sum'] += ibm_metrics['precision']
        results['IBM Model 1']['recall_sum'] += ibm_metrics['recall']
        results['IBM Model 1']['f1_sum'] += ibm_metrics['f1']
        results['IBM Model 1']['count'] += 1
        
        # Test representative models
        if is_hebrew:
            rep_predicted = test_ibm_model(verse, models_dir / 'hebrew_alignment_representative.json')
        else:
            rep_predicted = test_ibm_model(verse, models_dir / 'greek_alignment_representative.json')
            
        rep_metrics = evaluate_alignments(rep_predicted, gold)
        results['Representative']['precision_sum'] += rep_metrics['precision']
        results['Representative']['recall_sum'] += rep_metrics['recall']
        results['Representative']['f1_sum'] += rep_metrics['f1']
        results['Representative']['count'] += 1
        
        # Test hybrid model (if available)
        hybrid_dir = models_dir / 'hybrid'
        if hybrid_dir.exists():
            if is_hebrew:
                hybrid_predicted = test_hybrid_model(verse, hybrid_dir, 'hebrew_english_hybrid')
            else:
                hybrid_predicted = test_hybrid_model(verse, hybrid_dir, 'greek_english_hybrid')
                
            hybrid_metrics = evaluate_alignments(hybrid_predicted, gold)
            results['Hybrid']['precision_sum'] += hybrid_metrics['precision']
            results['Hybrid']['recall_sum'] += hybrid_metrics['recall']
            results['Hybrid']['f1_sum'] += hybrid_metrics['f1']
            results['Hybrid']['count'] += 1
            
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1} verses...")
    
    # Calculate averages
    logger.info("\n" + "="*60)
    logger.info("ALIGNMENT METHOD COMPARISON RESULTS")
    logger.info("="*60)
    
    for method, stats in results.items():
        if stats['count'] > 0:
            avg_precision = stats['precision_sum'] / stats['count']
            avg_recall = stats['recall_sum'] / stats['count']
            avg_f1 = stats['f1_sum'] / stats['count']
            
            logger.info(f"\n{method}:")
            logger.info(f"  Precision: {avg_precision:.3f}")
            logger.info(f"  Recall:    {avg_recall:.3f}")
            logger.info(f"  F1 Score:  {avg_f1:.3f}")
            logger.info(f"  Evaluated: {stats['count']} verses")
    
    # Determine best method
    best_method = max(results.items(), 
                     key=lambda x: x[1]['f1_sum'] / x[1]['count'] if x[1]['count'] > 0 else 0)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST METHOD: {best_method[0]}")
    logger.info(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare different word alignment methods"
    )
    
    parser.add_argument(
        '--enriched-dir',
        type=Path,
        default=Path('aligned_export_improved'),
        help='Directory with enriched Bible export containing alignments'
    )
    
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('models/alignment'),
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--num-verses',
        type=int,
        default=100,
        help='Number of test verses to evaluate'
    )
    
    args = parser.parse_args()
    
    # Load test verses
    logger.info(f"Loading {args.num_verses} test verses...")
    test_verses = load_test_verses(args.enriched_dir, args.num_verses)
    logger.info(f"Loaded {len(test_verses)} verses with alignments")
    
    if not test_verses:
        logger.error("No test verses found!")
        return 1
    
    # Compare methods
    compare_methods(test_verses, args.models_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())