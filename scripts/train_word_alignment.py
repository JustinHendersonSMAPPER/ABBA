#!/usr/bin/env python3
"""
Train word alignment models for Hebrew-English and Greek-English.

This script loads the enriched Bible data and trains IBM Model 1
alignment models for accurate word-level mappings.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.word_alignment import IBMModel1, PhrasalAligner
from src.abba.parsers.lexicon_parser import LexiconParser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_enriched_verses(export_dir: Path) -> Tuple[List[Tuple[List[Dict], str]], 
                                                   List[Tuple[List[Dict], str]]]:
    """Load Hebrew and Greek parallel verses from exported data."""
    hebrew_pairs = []
    greek_pairs = []
    
    # Load all book files
    for book_file in sorted(export_dir.glob("*.json")):
        if book_file.name.startswith("_"):
            continue
            
        with open(book_file, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
            
        logger.info(f"Loading {book_file.name}")
        
        for chapter in book_data.get('chapters', []):
            for verse in chapter.get('verses', []):
                # Get English translation (prefer KJV)
                eng_text = verse.get('translations', {}).get('eng_kjv', '')
                
                # Hebrew verses
                if verse.get('hebrew_words') and eng_text:
                    hebrew_pairs.append((verse['hebrew_words'], eng_text))
                    
                # Greek verses  
                if verse.get('greek_words') and eng_text:
                    greek_pairs.append((verse['greek_words'], eng_text))
                    
    logger.info(f"Loaded {len(hebrew_pairs)} Hebrew-English pairs")
    logger.info(f"Loaded {len(greek_pairs)} Greek-English pairs")
    
    return hebrew_pairs, greek_pairs


def load_lexicons(data_dir: Path) -> Tuple[Dict, Dict]:
    """Load Strong's lexicons for initialization."""
    hebrew_lex = {}
    greek_lex = {}
    
    parser = LexiconParser()
    
    # Load Hebrew lexicon
    hebrew_lex_path = data_dir / "sources" / "lexicons" / "strongs_hebrew.xml"
    if hebrew_lex_path.exists():
        logger.info("Loading Hebrew lexicon...")
        try:
            # Parse the XML file directly
            import xml.etree.ElementTree as ET
            tree = ET.parse(hebrew_lex_path)
            root = tree.getroot()
            
            for entry in root.findall('.//entry'):
                strongs = entry.get('strongs', '')
                if strongs:
                    # Get the first definition as gloss
                    gloss = ""
                    definition = entry.find('.//def')
                    if definition is not None and definition.text:
                        gloss = definition.text.strip()
                    
                    hebrew_lex[strongs] = {'gloss': gloss}
                    
            logger.info(f"Loaded {len(hebrew_lex)} Hebrew lexicon entries")
        except Exception as e:
            logger.error(f"Error loading Hebrew lexicon: {e}")
    
    # Load Greek lexicon
    greek_lex_path = data_dir / "sources" / "lexicons" / "strongs_greek.xml"
    if greek_lex_path.exists():
        logger.info("Loading Greek lexicon...")
        try:
            tree = ET.parse(greek_lex_path)
            root = tree.getroot()
            
            for entry in root.findall('.//entry'):
                strongs = entry.get('strongs', '')
                if strongs:
                    gloss = ""
                    definition = entry.find('.//def')
                    if definition is not None and definition.text:
                        gloss = definition.text.strip()
                        
                    greek_lex[strongs] = {'gloss': gloss}
                    
            logger.info(f"Loaded {len(greek_lex)} Greek lexicon entries")
        except Exception as e:
            logger.error(f"Error loading Greek lexicon: {e}")
    
    return hebrew_lex, greek_lex


def train_hebrew_model(hebrew_pairs: List[Tuple[List[Dict], str]], 
                      hebrew_lex: Dict) -> IBMModel1:
    """Train Hebrew-English alignment model."""
    logger.info("Training Hebrew-English alignment model...")
    
    model = IBMModel1(use_morphology=True, use_strongs=True, use_lexicon=True)
    
    # Initialize from lexicon
    if hebrew_lex:
        model.initialize_from_lexicon(hebrew_lex)
    
    # Train on parallel corpus
    model.train(hebrew_pairs, verbose=True)
    
    return model


def train_greek_model(greek_pairs: List[Tuple[List[Dict], str]], 
                     greek_lex: Dict) -> IBMModel1:
    """Train Greek-English alignment model."""
    logger.info("Training Greek-English alignment model...")
    
    model = IBMModel1(use_morphology=True, use_strongs=True, use_lexicon=True)
    
    # Initialize from lexicon
    if greek_lex:
        model.initialize_from_lexicon(greek_lex)
    
    # Train on parallel corpus
    model.train(greek_pairs, verbose=True)
    
    return model


def evaluate_model(model: IBMModel1, test_pairs: List[Tuple[List[Dict], str]], 
                  name: str, sample_size: int = 5):
    """Evaluate alignment model on test pairs."""
    logger.info(f"\nEvaluating {name} model...")
    
    total_score = 0.0
    
    for i, (source_words, target_text) in enumerate(test_pairs[:sample_size]):
        alignment = model.align_verse(source_words, target_text)
        total_score += alignment.alignment_score
        
        # Print sample alignment
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Target: {target_text}")
        logger.info("Alignments:")
        
        for align in alignment.alignments[:5]:  # Show first 5 alignments
            logger.info(f"  {align.source_word} -> {align.target_word} "
                       f"(conf: {align.confidence:.3f})")
                       
    avg_score = total_score / min(sample_size, len(test_pairs))
    logger.info(f"\nAverage alignment score: {avg_score:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train word alignment models for biblical texts"
    )
    
    parser.add_argument(
        '--export-dir',
        type=Path,
        default=Path('full_fixed_export'),
        help='Directory containing enriched Bible export'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Directory containing source data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/alignment'),
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='Train on sample data only (faster)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load enriched verses
    hebrew_pairs, greek_pairs = load_enriched_verses(args.export_dir)
    
    if not hebrew_pairs and not greek_pairs:
        logger.error("No parallel data found. Please run the export first.")
        return 1
        
    # Use sample for faster training if requested
    if args.sample_only:
        hebrew_pairs = hebrew_pairs[:1000]
        greek_pairs = greek_pairs[:1000]
        logger.info("Using sample data for training")
    
    # Load lexicons
    hebrew_lex, greek_lex = load_lexicons(args.data_dir)
    
    # Train Hebrew model
    if hebrew_pairs:
        hebrew_model = train_hebrew_model(hebrew_pairs, hebrew_lex)
        
        # Save model
        hebrew_model.save_model(args.output_dir / "hebrew_alignment.json")
        
        # Evaluate
        evaluate_model(hebrew_model, hebrew_pairs[-100:], "Hebrew-English")
    
    # Train Greek model  
    if greek_pairs:
        greek_model = train_greek_model(greek_pairs, greek_lex)
        
        # Save model
        greek_model.save_model(args.output_dir / "greek_alignment.json")
        
        # Evaluate
        evaluate_model(greek_model, greek_pairs[-100:], "Greek-English")
        
    logger.info("\nTraining complete!")
    logger.info(f"Models saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())