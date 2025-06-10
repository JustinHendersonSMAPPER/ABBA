#!/usr/bin/env python3
"""
Evaluate alignment quality against gold standard.

This script compares system-generated alignments to manually curated gold 
standard alignments and calculates precision, recall, and F1 scores.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from abba.alignment.ensemble_aligner import EnsembleAligner
from abba.alignment.position_aligner import PositionAligner
from abba.alignment.morphological_aligner import MorphologicalAligner
from abba.alignment.statistical_aligner import StatisticalAligner


class AlignmentEvaluator:
    """Evaluates alignment quality against gold standard."""
    
    def __init__(self, gold_alignments_path: str):
        """Initialize evaluator with gold standard data."""
        self.gold_path = Path(gold_alignments_path)
        self.gold_data = self._load_gold_standard()
    
    def _load_gold_standard(self) -> Dict:
        """Load gold standard alignments."""
        if not self.gold_path.exists():
            raise FileNotFoundError(f"Gold standard file not found: {self.gold_path}")
        
        with open(self.gold_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_aligner(self, aligner, confidence_threshold: float = 0.5) -> Dict:
        """
        Evaluate an aligner against all gold standard verses.
        
        Args:
            aligner: Alignment system to evaluate
            confidence_threshold: Minimum confidence to consider a prediction
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'total_verses': 0,
            'total_gold_alignments': 0,
            'total_predicted_alignments': 0,
            'correct_alignments': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'per_verse_results': []
        }
        
        for verse_data in self.gold_data['verses']:
            verse_result = self._evaluate_verse(verse_data, aligner, confidence_threshold)
            results['per_verse_results'].append(verse_result)
            
            results['total_verses'] += 1
            results['total_gold_alignments'] += verse_result['gold_count']
            results['total_predicted_alignments'] += verse_result['predicted_count']
            results['correct_alignments'] += verse_result['correct_count']
        
        # Calculate overall metrics
        if results['total_predicted_alignments'] > 0:
            results['precision'] = results['correct_alignments'] / results['total_predicted_alignments']
        
        if results['total_gold_alignments'] > 0:
            results['recall'] = results['correct_alignments'] / results['total_gold_alignments']
        
        if results['precision'] + results['recall'] > 0:
            results['f1'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
        
        return results
    
    def _evaluate_verse(self, verse_data: Dict, aligner, confidence_threshold: float) -> Dict:
        """Evaluate alignment for a single verse."""
        reference = verse_data['reference']
        source_words = verse_data.get('hebrew_words') or verse_data.get('greek_words')
        target_words = verse_data['english_words']
        gold_alignments = verse_data['gold_alignments']
        
        # Determine source language
        source_lang = 'hebrew' if 'hebrew_words' in verse_data else 'greek'
        
        # Get system predictions
        try:
            predicted_alignments = aligner.align_verse(
                source_words, target_words, source_lang, 'english'
            )
        except Exception as e:
            print(f"Warning: Alignment failed for {reference}: {e}")
            predicted_alignments = []
        
        # Filter by confidence threshold
        high_conf_predictions = [
            align for align in predicted_alignments 
            if align['confidence'] >= confidence_threshold
        ]
        
        # Convert to (source_idx, target_idx) tuples for comparison
        gold_pairs = {
            (align['source_index'], align['target_index']) 
            for align in gold_alignments
        }
        
        predicted_pairs = {
            (align['source_index'], align['target_index']) 
            for align in high_conf_predictions
        }
        
        # Calculate matches
        correct_pairs = gold_pairs.intersection(predicted_pairs)
        
        return {
            'reference': reference,
            'source_language': source_lang,
            'gold_count': len(gold_pairs),
            'predicted_count': len(predicted_pairs),
            'correct_count': len(correct_pairs),
            'precision': len(correct_pairs) / len(predicted_pairs) if predicted_pairs else 0.0,
            'recall': len(correct_pairs) / len(gold_pairs) if gold_pairs else 0.0,
            'gold_alignments': gold_pairs,
            'predicted_alignments': predicted_pairs,
            'correct_alignments': correct_pairs
        }
    
    def print_evaluation_report(self, results: Dict, aligner_name: str):
        """Print detailed evaluation report."""
        print(f"\n{'=' * 60}")
        print(f"Alignment Evaluation Report - {aligner_name}")
        print(f"{'=' * 60}")
        
        print(f"\nOverall Metrics:")
        print(f"  Verses evaluated: {results['total_verses']}")
        print(f"  Gold alignments: {results['total_gold_alignments']}")
        print(f"  Predicted alignments: {results['total_predicted_alignments']}")
        print(f"  Correct alignments: {results['correct_alignments']}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall: {results['recall']:.3f}")
        print(f"  F1-Score: {results['f1']:.3f}")
        
        print(f"\nPer-Verse Results:")
        print(f"{'Reference':<15} {'Lang':<6} {'Gold':<4} {'Pred':<4} {'Corr':<4} {'Prec':<5} {'Rec':<5}")
        print("-" * 50)
        
        for verse in results['per_verse_results']:
            print(f"{verse['reference']:<15} "
                  f"{verse['source_language']:<6} "
                  f"{verse['gold_count']:<4} "
                  f"{verse['predicted_count']:<4} "
                  f"{verse['correct_count']:<4} "
                  f"{verse['precision']:<5.2f} "
                  f"{verse['recall']:<5.2f}")


def create_test_aligners() -> Dict:
    """Create different aligners for comparison."""
    aligners = {}
    
    # Position aligner
    aligners['Position'] = PositionAligner(base_confidence=0.4)
    
    # Morphological aligner
    aligners['Morphological'] = MorphologicalAligner(base_confidence=0.6)
    
    # Statistical aligner
    try:
        aligners['Statistical'] = StatisticalAligner(base_confidence=0.8)
    except Exception as e:
        print(f"Warning: Could not load statistical aligner: {e}")
    
    # Ensemble aligner
    if 'Statistical' in aligners:
        aligners['Ensemble'] = EnsembleAligner([
            (aligners['Statistical'], 0.5),
            (aligners['Morphological'], 0.3),
            (aligners['Position'], 0.2)
        ])
    else:
        aligners['Ensemble'] = EnsembleAligner([
            (aligners['Morphological'], 0.6),
            (aligners['Position'], 0.4)
        ])
    
    return aligners


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate alignment quality')
    parser.add_argument('--gold-file', 
                       default='data/test/gold_alignments.json',
                       help='Path to gold standard alignments')
    parser.add_argument('--confidence-threshold', 
                       type=float, default=0.3,
                       help='Minimum confidence threshold for predictions')
    parser.add_argument('--aligner',
                       choices=['Position', 'Morphological', 'Statistical', 'Ensemble', 'all'],
                       default='all',
                       help='Which aligner(s) to evaluate')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AlignmentEvaluator(args.gold_file)
    
    # Create aligners
    aligners = create_test_aligners()
    
    # Select aligners to evaluate
    if args.aligner == 'all':
        aligners_to_test = aligners
    else:
        aligners_to_test = {args.aligner: aligners[args.aligner]}
    
    # Evaluate each aligner
    print(f"Evaluating alignments with confidence threshold: {args.confidence_threshold}")
    print(f"Gold standard: {args.gold_file}")
    
    all_results = {}
    for aligner_name, aligner in aligners_to_test.items():
        print(f"\nEvaluating {aligner_name} aligner...")
        results = evaluator.evaluate_aligner(aligner, args.confidence_threshold)
        all_results[aligner_name] = results
        evaluator.print_evaluation_report(results, aligner_name)
    
    # Summary comparison if multiple aligners
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Comparison Summary")
        print(f"{'=' * 60}")
        print(f"{'Aligner':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        for name, results in all_results.items():
            print(f"{name:<15} "
                  f"{results['precision']:<10.3f} "
                  f"{results['recall']:<10.3f} "
                  f"{results['f1']:<10.3f}")


if __name__ == "__main__":
    main()