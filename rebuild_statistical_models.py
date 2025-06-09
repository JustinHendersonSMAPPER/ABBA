#!/usr/bin/env python3
"""
Rebuild statistical models with full Bible corpus.
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Rebuild statistical models."""
    logger.info("Rebuilding statistical models with full Bible corpus...")
    
    # Step 1: Prepare full parallel corpus
    logger.info("\nStep 1: Preparing full parallel corpus...")
    result = subprocess.run(
        [sys.executable, "scripts/prepare_parallel_corpus.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to prepare corpus: {result.stderr}")
        return 1
    
    logger.info(result.stdout)
    
    # Step 2: Train statistical models
    logger.info("\nStep 2: Training statistical models...")
    result = subprocess.run(
        [sys.executable, "scripts/train_statistical_alignment.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to train models: {result.stderr}")
        return 1
    
    logger.info(result.stdout)
    
    # Check model sizes
    models_dir = Path('models/statistical')
    hebrew_model = models_dir / 'hebrew_english_alignment.json'
    greek_model = models_dir / 'greek_english_alignment.json'
    
    if hebrew_model.exists():
        size_mb = hebrew_model.stat().st_size / (1024 * 1024)
        logger.info(f"\nHebrew model size: {size_mb:.2f} MB")
    
    if greek_model.exists():
        size_mb = greek_model.stat().st_size / (1024 * 1024)
        logger.info(f"Greek model size: {size_mb:.2f} MB")
    
    logger.info("\nStatistical models rebuilt successfully!")
    logger.info("Run 'python src/main.py' to test the improved alignment.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())