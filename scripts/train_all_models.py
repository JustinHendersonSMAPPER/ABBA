#!/usr/bin/env python3
"""
Train all language models (Hebrew and Greek) with all features enabled.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(source_lang: str, target_lang: str = 'english'):
    """Train a single alignment model with all features."""
    logger.info(f"Training {source_lang} → {target_lang} model...")
    
    cmd = [
        sys.executable, '-m', 'abba_align', 'train',
        '--source', source_lang,
        '--target', target_lang,
        '--corpus-dir', 'data/sources',
        '--features', 'all',
        '--parallel-passages'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully trained {source_lang} → {target_lang} model")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to train {source_lang} → {target_lang} model")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Train all biblical language models."""
    print("=" * 60)
    print("TRAINING ALL BIBLICAL ALIGNMENT MODELS")
    print("=" * 60)
    print()
    
    # Ensure data directory exists
    data_dir = Path('data/sources')
    if not data_dir.exists():
        logger.error("Data directory not found. Please run: python scripts/download_sources.py")
        sys.exit(1)
    
    # Train models for both Hebrew and Greek
    languages = ['hebrew', 'greek']
    success_count = 0
    
    for lang in languages:
        if train_model(lang):
            success_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"Training complete: {success_count}/{len(languages)} models trained successfully")
    
    # List available models
    print("\nAvailable models:")
    model_dir = Path('models/biblical_alignment')
    if model_dir.exists():
        for model_file in sorted(model_dir.glob('*_biblical.json')):
            print(f"  - {model_file.name}")
    
    return success_count == len(languages)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)