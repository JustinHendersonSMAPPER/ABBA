#!/usr/bin/env python3
"""
Demonstrate the Strong's Concordance Alignment System.

This script shows how the enhanced alignment system works with
Strong's concordance data and manual mappings.
"""

import json
import logging
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from abba.alignment.word_alignment import IBMModel1, WordAlignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_strongs_lookup():
    """Demonstrate Strong's concordance lookup."""
    logger.info("\n=== Strong's Concordance Lookup Demo ===")
    
    # Load Strong's enhanced model
    model_path = Path(__file__).parent.parent / "models" / "alignment" / "strongs_enhanced_alignment.json"
    
    if not model_path.exists():
        logger.error("Strong's enhanced model not found. Run load_full_strongs_concordance.py first.")
        return
    
    # Load sample of the model
    sample_path = Path(__file__).parent.parent / "models" / "alignment" / "sample_strongs_enhanced_alignment.json"
    
    with open(sample_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    # Demonstrate lookups
    demo_strongs = ["H3068", "H430", "G2316", "G2424"]
    
    for strongs_num in demo_strongs:
        if strongs_num in model_data['trans_probs']:
            metadata = model_data['strongs_metadata'].get(
                'hebrew' if strongs_num.startswith('H') else 'greek', {}
            ).get(strongs_num, {})
            
            logger.info(f"\n{strongs_num}:")
            logger.info(f"  Original: {metadata.get('original', 'N/A')}")
            logger.info(f"  Transliteration: {metadata.get('translit', 'N/A')}")
            logger.info(f"  Gloss: {metadata.get('gloss', 'N/A')}")
            logger.info(f"  Top translations:")
            
            # Sort translations by probability
            translations = model_data['trans_probs'][strongs_num]
            sorted_trans = sorted(translations.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for trans, prob in sorted_trans:
                logger.info(f"    {trans}: {prob:.3f}")


def demonstrate_manual_mappings():
    """Demonstrate manual alignment mappings."""
    logger.info("\n=== Manual Alignment Mappings Demo ===")
    
    # Load manual mappings
    hebrew_manual = Path(__file__).parent.parent / "data" / "manual_alignments" / "high_frequency_hebrew.json"
    
    with open(hebrew_manual, 'r', encoding='utf-8') as f:
        hebrew_data = json.load(f)
    
    # Show some examples
    examples = ["H3068", "H430", "H559"]
    
    for strongs in examples:
        if strongs in hebrew_data['mappings']:
            mapping = hebrew_data['mappings'][strongs]
            logger.info(f"\n{strongs} - {mapping['hebrew']} ({mapping['transliteration']}):")
            logger.info(f"  Primary: {', '.join(mapping['primary_translations'])}")
            logger.info(f"  Confidence: {mapping['confidence']}")
            logger.info(f"  Notes: {mapping['notes']}")


def demonstrate_alignment():
    """Demonstrate word alignment with Strong's data."""
    logger.info("\n=== Word Alignment Demo ===")
    
    # Create test verse data
    test_verse = {
        'source_words': [
            {'text': 'בְּרֵאשִׁית', 'strongs': 'H7225', 'morph': 'Ncfsa', 'lang': 'hebrew'},
            {'text': 'בָּרָא', 'strongs': 'H1254', 'morph': 'Vqp3ms', 'lang': 'hebrew'},
            {'text': 'אֱלֹהִים', 'strongs': 'H430', 'morph': 'Ncmpa', 'lang': 'hebrew'},
            {'text': 'אֵת', 'strongs': 'H853', 'morph': 'To', 'lang': 'hebrew'},
            {'text': 'הַשָּׁמַיִם', 'strongs': 'H8064', 'morph': 'Td/Ncmpa', 'lang': 'hebrew'},
            {'text': 'וְאֵת', 'strongs': 'H853', 'morph': 'C/To', 'lang': 'hebrew'},
            {'text': 'הָאָרֶץ', 'strongs': 'H776', 'morph': 'Td/Ncfsa', 'lang': 'hebrew'}
        ],
        'target_text': "In the beginning God created the heaven and the earth"
    }
    
    # Load enhanced model (use sample for demo)
    model_path = Path(__file__).parent.parent / "models" / "alignment" / "sample_strongs_enhanced_alignment.json"
    
    if not model_path.exists():
        logger.error("Model not found. Run the alignment pipeline first.")
        return
    
    # Initialize model
    model = IBMModel1(use_strongs=True, use_lexicon=True)
    model.load_model(model_path)
    
    # Perform alignment
    alignment = model.align_verse(test_verse['source_words'], test_verse['target_text'])
    
    logger.info(f"\nSource: {' '.join(w['text'] for w in test_verse['source_words'])}")
    logger.info(f"Target: {test_verse['target_text']}")
    logger.info("\nAlignments:")
    
    for align in alignment.alignments:
        source_word = test_verse['source_words'][align.source_idx]
        logger.info(f"  {source_word['text']} ({source_word['strongs']}) -> {align.target_word} "
                   f"(confidence: {align.confidence:.3f})")


def demonstrate_coverage_stats():
    """Show coverage statistics."""
    logger.info("\n=== Coverage Statistics ===")
    
    # Load pipeline state if available
    state_path = Path(__file__).parent.parent / "models" / "alignment" / "pipeline_state.json"
    
    if state_path.exists():
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        if 'metrics' in state and 'evaluation' in state['metrics']:
            eval_metrics = state['metrics']['evaluation']
            
            for lang, metrics in eval_metrics.items():
                logger.info(f"\n{lang.capitalize()}:")
                logger.info(f"  Manual mappings covered: {metrics['covered_mappings']}/{metrics['total_manual_mappings']}")
                logger.info(f"  Coverage ratio: {metrics['coverage_ratio']:.2%}")
                logger.info(f"  High confidence ratio: {metrics['high_confidence_ratio']:.2%}")


def main():
    """Run all demonstrations."""
    logger.info("=== Strong's Concordance Alignment System Demonstration ===")
    
    # Check if models exist
    model_dir = Path(__file__).parent.parent / "models" / "alignment"
    if not model_dir.exists():
        logger.error("Models directory not found. Run the alignment scripts first.")
        return
    
    # Run demonstrations
    demonstrate_strongs_lookup()
    demonstrate_manual_mappings()
    demonstrate_alignment()
    demonstrate_coverage_stats()
    
    logger.info("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()