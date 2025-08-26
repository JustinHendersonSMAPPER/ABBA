#!/usr/bin/env python3
"""Test resuming embedding generation after crash."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager, ContextBuilder, EmbeddingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize components
    db_path = Path("bible_data/abba.db")
    vector_path = Path("bible_data/vectors")
    models_path = Path("bible_data/models")
    
    db_manager = SQLiteManager(db_path)
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    model_manager = EmbeddingModelManager(cache_dir=str(models_path))
    context_builder = ContextBuilder(db_manager)
    
    pipeline = EmbeddingPipeline(
        db_manager=db_manager,
        chroma_manager=chroma_manager,
        model_manager=model_manager,
        context_builder=context_builder
    )
    
    # Check current progress
    progress = pipeline.progress
    print("Current embedding progress:")
    
    for trans_id, info in progress.get('verses', {}).items():
        if info.get('complete'):
            print(f"  {trans_id}: Complete ({info.get('count', 0)} verses)")
        else:
            print(f"  {trans_id}: In progress (last count: {info.get('last_count', 0)})")
    
    # Try to resume HBOMAS with smaller batch size
    print("\nResuming HBOMAS embedding with smaller batch size...")
    
    results = pipeline.embed_verses(
        translation_ids=["HBOMAS"],
        batch_size=50,  # Smaller batch size
        force_reembed=False  # Allow resume
    )
    
    print(f"\nResults:")
    print(f"  Translations processed: {results['translations_processed']}")
    print(f"  Verses embedded: {results['verses_embedded']:,}")
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for error in results['errors'][:5]:
            print(f"    - {error}")

if __name__ == "__main__":
    main()