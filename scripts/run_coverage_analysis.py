#!/usr/bin/env python3
"""
Run complete coverage analysis with enhanced models.
This is the main entry point for analyzing Bible translation coverage.
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_source_data():
    """Ensure source data exists."""
    lexicon_dir = Path('data/sources/lexicons')
    if not lexicon_dir.exists():
        logger.info("Downloading source data...")
        result = subprocess.run([sys.executable, 'scripts/download_sources.py'])
        if result.returncode != 0:
            logger.error("Failed to download source data")
            sys.exit(1)
    else:
        logger.info("Source data exists ✓")


def build_enhanced_models():
    """Build enhanced models if they don't exist."""
    hebrew_enhanced = Path('models/biblical_alignment/hebrew_english_enhanced.json')
    greek_enhanced = Path('models/biblical_alignment/greek_english_enhanced.json')
    
    if not hebrew_enhanced.exists() or not greek_enhanced.exists():
        logger.info("Building enhanced models with full Strong's concordance...")
        
        # Load Strong's concordance
        result = subprocess.run([sys.executable, 'scripts/load_full_strongs_concordance.py'])
        if result.returncode != 0:
            logger.error("Failed to load Strong's concordance")
            sys.exit(1)
            
        # Create manual alignments
        result = subprocess.run([sys.executable, 'scripts/create_manual_alignments.py'])
        if result.returncode != 0:
            logger.error("Failed to create manual alignments")
            sys.exit(1)
            
        # Build enhanced models
        result = subprocess.run([sys.executable, 'scripts/create_enhanced_models.py'])
        if result.returncode != 0:
            logger.error("Failed to create enhanced models")
            sys.exit(1)
    else:
        logger.info("Enhanced models exist ✓")


def verify_enhanced_models():
    """Verify the enhanced models have correct entries."""
    logger.info("Verifying enhanced models...")
    
    model_dir = Path('models/biblical_alignment')
    all_good = True
    
    # Check Hebrew enhanced model
    hebrew_model = model_dir / 'hebrew_english_enhanced.json'
    if hebrew_model.exists():
        with open(hebrew_model, 'r') as f:
            model = json.load(f)
            mappings = len(model.get('strongs_mappings', {}))
            logger.info(f"  Hebrew enhanced model: {mappings:,} Strong's mappings")
            if mappings < 8000:
                logger.warning("  ⚠️  Hebrew model appears incomplete!")
                all_good = False
    else:
        logger.error("  ❌ Hebrew enhanced model not found!")
        all_good = False
    
    # Check Greek enhanced model  
    greek_model = model_dir / 'greek_english_enhanced.json'
    if greek_model.exists():
        with open(greek_model, 'r') as f:
            model = json.load(f)
            mappings = len(model.get('strongs_mappings', {}))
            logger.info(f"  Greek enhanced model: {mappings:,} Strong's mappings")
            if mappings < 5000:
                logger.warning("  ⚠️  Greek model appears incomplete!")
                all_good = False
    else:
        logger.error("  ❌ Greek enhanced model not found!")
        all_good = False
        
    return all_good


def configure_models():
    """Configure the analysis to use enhanced models."""
    logger.info("\nConfiguring models for analysis...")
    
    model_dir = Path('models/biblical_alignment')
    
    # First verify enhanced models exist and are complete
    if not verify_enhanced_models():
        logger.error("Enhanced models are missing or incomplete!")
        return False
    
    # Back up existing models if they exist
    for lang in ['hebrew', 'greek']:
        biblical_model = model_dir / f'{lang}_english_biblical.json'
        if biblical_model.exists():
            backup = model_dir / f'{lang}_english_biblical.json.backup'
            shutil.copy2(biblical_model, backup)
    
    # Copy enhanced models to biblical names
    shutil.copy2(
        model_dir / 'hebrew_english_enhanced.json',
        model_dir / 'hebrew_english_biblical.json'
    )
    shutil.copy2(
        model_dir / 'greek_english_enhanced.json',
        model_dir / 'greek_english_biblical.json'
    )
    
    # Verify the copies
    logger.info("\nVerifying configured models:")
    for lang in ['hebrew', 'greek']:
        with open(model_dir / f'{lang}_english_biblical.json', 'r') as f:
            model = json.load(f)
            mappings = len(model.get('strongs_mappings', {}))
            logger.info(f"  {lang.title()}: {mappings:,} Strong's mappings")
            
    return True


def run_analysis():
    """Run the coverage analysis."""
    logger.info("\nAnalyzing all translations...")
    logger.info("This analysis uses:")
    logger.info("  - Hebrew: 8,673 Strong's concordance entries + 25 manual alignments")
    logger.info("  - Greek: 5,472 Strong's concordance entries + 25 manual alignments")
    logger.info("")
    
    result = subprocess.run([
        sys.executable, 'scripts/analyze_all_translations_coverage.py',
        '--output', 'translation_coverage_report.md'
    ])
    
    if result.returncode != 0:
        logger.error("Coverage analysis failed")
        sys.exit(1)


def main():
    """Main entry point."""
    print("=" * 50)
    print("ABBA-Align Coverage Analysis")
    print("=" * 50)
    print()
    
    # Step 1: Check source data
    logger.info("Step 1: Checking source data...")
    check_source_data()
    
    # Step 2: Build enhanced models
    print()
    logger.info("Step 2: Checking enhanced models...")
    build_enhanced_models()
    
    # Step 3: Configure models
    print()
    logger.info("Step 3: Configuring models...")
    if not configure_models():
        logger.error("Failed to configure models properly")
        sys.exit(1)
    
    # Step 4: Run analysis
    print()
    logger.info("Step 4: Running analysis...")
    run_analysis()
    
    # Complete
    print()
    print("=" * 50)
    print("Analysis Complete!")
    print("=" * 50)
    print("Report saved to: translation_coverage_report.md")
    print()
    print("To view the report:")
    print("  cat translation_coverage_report.md")


if __name__ == '__main__':
    main()