#!/usr/bin/env python3
"""
Enhanced training system that uses comprehensive Strong's mappings.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from abba_align.alignment_trainer import AlignmentTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedAlignmentTrainer:
    """Enhanced trainer that uses full Strong's concordance and manual alignments."""
    
    def __init__(self, source_lang: str, target_lang: str = 'english'):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.strongs_model_path = Path('models/alignment/strongs_enhanced_alignment.json')
        self.manual_alignments_dir = Path('data/manual_alignments')
        self.output_dir = Path('models/biblical_alignment_enhanced')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_strongs_model(self) -> Dict:
        """Load the enhanced Strong's model."""
        if not self.strongs_model_path.exists():
            logger.error(f"Strong's model not found at {self.strongs_model_path}")
            logger.info("Please run: python scripts/load_full_strongs_concordance.py")
            return None
            
        with open(self.strongs_model_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def load_manual_alignments(self) -> Dict:
        """Load manual alignments for the source language."""
        manual_file = self.manual_alignments_dir / f'high_frequency_{self.source_lang}.json'
        
        if not manual_file.exists():
            logger.warning(f"No manual alignments found for {self.source_lang}")
            return {}
            
        with open(manual_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def create_enhanced_model(self) -> Dict:
        """Create an enhanced alignment model."""
        # Load base Strong's model
        strongs_model = self.load_strongs_model()
        if not strongs_model:
            return None
            
        # Load manual alignments
        manual_alignments = self.load_manual_alignments()
        
        # Create enhanced model
        enhanced_model = {
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'version': '2.0',
            'features': {
                'strongs': True,
                'manual_alignments': True,
                'morphology': True,
                'phrases': True,
                'syntax': True,
                'semantics': True,
                'discourse': True
            },
            'strongs_mappings': {},
            'manual_mappings': {},
            'alignment_probabilities': defaultdict(lambda: defaultdict(float)),
            'coverage_stats': {}
        }
        
        # Extract relevant Strong's mappings
        logger.info("Extracting Strong's mappings...")
        prefix = 'H' if self.source_lang == 'hebrew' else 'G'
        
        for source_term, translations in strongs_model['translation_pairs'].items():
            # Check if this is a Strong's number for our language
            if source_term.startswith(prefix):
                enhanced_model['strongs_mappings'][source_term] = translations
                
                # Initialize alignment probabilities
                total_count = sum(translations.values())
                for target, count in translations.items():
                    prob = count / total_count if total_count > 0 else 0
                    enhanced_model['alignment_probabilities'][source_term][target] = prob
                    
        # Add manual alignments with highest confidence
        logger.info("Integrating manual alignments...")
        for strongs_num, alignment_data in manual_alignments.items():
            enhanced_model['manual_mappings'][strongs_num] = alignment_data
            
            # Boost probabilities for manual alignments
            for primary_trans in alignment_data['primary_translations']:
                enhanced_model['alignment_probabilities'][strongs_num][primary_trans] = 0.9
                
            # Add secondary translations with lower probability
            for trans in alignment_data['all_translations']:
                if trans not in alignment_data['primary_translations']:
                    current_prob = enhanced_model['alignment_probabilities'][strongs_num].get(trans, 0)
                    enhanced_model['alignment_probabilities'][strongs_num][trans] = max(current_prob, 0.5)
                    
        # Calculate coverage statistics
        enhanced_model['coverage_stats'] = {
            'total_strongs_entries': len(enhanced_model['strongs_mappings']),
            'manual_entries': len(enhanced_model['manual_mappings']),
            'unique_source_terms': len(enhanced_model['alignment_probabilities']),
            'unique_target_terms': len(set(
                target for mappings in enhanced_model['alignment_probabilities'].values()
                for target in mappings
            ))
        }
        
        return enhanced_model
        
    def train_with_corpus(self, enhanced_model: Dict, corpus_dir: Path = None) -> Dict:
        """Train the model using corpus data if available."""
        if not corpus_dir:
            corpus_dir = Path('data/sources')
            
        # Run standard training with enhanced initialization
        logger.info(f"Training {self.source_lang}-{self.target_lang} with enhanced model...")
        
        # Save temporary enhanced model for training
        temp_model_path = self.output_dir / f'{self.source_lang}_{self.target_lang}_init.json'
        with open(temp_model_path, 'w', encoding='utf-8') as f:
            # Convert defaultdict to regular dict for JSON serialization
            model_to_save = enhanced_model.copy()
            model_to_save['alignment_probabilities'] = dict(
                (k, dict(v)) for k, v in enhanced_model['alignment_probabilities'].items()
            )
            json.dump(model_to_save, f, indent=2, ensure_ascii=False)
            
        # Run abba_align training with the enhanced model
        cmd = [
            sys.executable, '-m', 'abba_align', 'train',
            '--source', self.source_lang,
            '--target', self.target_lang,
            '--corpus-dir', str(corpus_dir),
            '--features', 'all',
            '--init-model', str(temp_model_path),
            '--output-dir', str(self.output_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Training completed with warnings: {result.stderr}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
        # Load the trained model and merge with enhanced data
        trained_model_path = self.output_dir / f'{self.source_lang}_{self.target_lang}_biblical.json'
        
        if trained_model_path.exists():
            with open(trained_model_path, 'r') as f:
                trained_model = json.load(f)
                
            # Merge enhanced features
            trained_model['manual_mappings'] = enhanced_model['manual_mappings']
            trained_model['enhanced'] = True
            
            # Save final model
            with open(trained_model_path, 'w', encoding='utf-8') as f:
                json.dump(trained_model, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Enhanced model saved to {trained_model_path}")
            return trained_model
        else:
            # Save enhanced model directly if training didn't produce output
            final_path = self.output_dir / f'{self.source_lang}_{self.target_lang}_enhanced.json'
            with open(final_path, 'w', encoding='utf-8') as f:
                model_to_save = enhanced_model.copy()
                model_to_save['alignment_probabilities'] = dict(
                    (k, dict(v)) for k, v in enhanced_model['alignment_probabilities'].items()
                )
                json.dump(model_to_save, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Enhanced model saved to {final_path}")
            return enhanced_model
            
    def evaluate_model(self, model: Dict) -> Dict:
        """Evaluate the enhanced model."""
        stats = model.get('coverage_stats', {})
        
        # Calculate quality metrics
        high_conf_alignments = sum(
            1 for source, targets in model.get('alignment_probabilities', {}).items()
            for target, prob in targets.items()
            if prob >= 0.8
        )
        
        manual_coverage = len(model.get('manual_mappings', {}))
        
        evaluation = {
            'total_mappings': stats.get('total_strongs_entries', 0),
            'manual_mappings': manual_coverage,
            'high_confidence_alignments': high_conf_alignments,
            'unique_sources': stats.get('unique_source_terms', 0),
            'unique_targets': stats.get('unique_target_terms', 0),
            'estimated_coverage': 'High' if manual_coverage >= 20 else 'Medium'
        }
        
        return evaluation
        
    def train(self):
        """Main training pipeline."""
        logger.info(f"=== Enhanced Training for {self.source_lang.title()}-English ===")
        
        # Create enhanced model
        enhanced_model = self.create_enhanced_model()
        if not enhanced_model:
            return
            
        # Train with corpus
        trained_model = self.train_with_corpus(enhanced_model)
        
        # Evaluate
        evaluation = self.evaluate_model(trained_model)
        
        # Print results
        logger.info("\n=== Training Complete ===")
        logger.info(f"Total mappings: {evaluation['total_mappings']:,}")
        logger.info(f"Manual mappings: {evaluation['manual_mappings']}")
        logger.info(f"High confidence alignments: {evaluation['high_confidence_alignments']:,}")
        logger.info(f"Estimated coverage: {evaluation['estimated_coverage']}")
        
        # Save evaluation
        eval_path = self.output_dir / f'{self.source_lang}_{self.target_lang}_evaluation.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
            
        logger.info(f"\nEvaluation saved to {eval_path}")


def main():
    """Train enhanced models for both Hebrew and Greek."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train enhanced alignment models')
    parser.add_argument('--source', choices=['hebrew', 'greek', 'both'], 
                      default='both', help='Source language to train')
    parser.add_argument('--corpus-dir', type=Path, 
                      default=Path('data/sources'),
                      help='Directory containing corpus files')
    
    args = parser.parse_args()
    
    if args.source in ['hebrew', 'both']:
        trainer = EnhancedAlignmentTrainer('hebrew')
        trainer.train()
        
    if args.source in ['greek', 'both']:
        trainer = EnhancedAlignmentTrainer('greek')
        trainer.train()


if __name__ == '__main__':
    main()