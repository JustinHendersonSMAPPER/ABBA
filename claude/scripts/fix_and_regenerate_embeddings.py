#!/usr/bin/env python3
"""Fix and regenerate original language embeddings."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager, ContextBuilder
from abba.embeddings.original_language_pipeline import OriginalLanguageEmbeddingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize components
    db_path = Path("bible_data/abba.db")
    vector_path = Path("bible_data/vectors")
    models_path = Path("bible_data/models")
    progress_path = Path("bible_data/.embedding_progress.json")
    
    # Clear progress for original verses
    if progress_path.exists():
        import json
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        if 'original_verses' in progress:
            del progress['original_verses']
            
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        print("✓ Cleared original verse progress")
    
    db_manager = SQLiteManager(db_path)
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    
    # 1. Clean up bad original embeddings
    print("\n" + "="*60)
    print("Step 1: Cleaning up bad original embeddings")
    print("="*60)
    
    try:
        chroma_manager.delete_collection("original_verses")
        print("✓ Deleted original_verses collection")
    except:
        print("No original_verses collection to delete")
    
    # 2. Clean up legacy translation embeddings
    print("\n" + "="*60)
    print("Step 2: Cleaning up legacy translation embeddings")
    print("="*60)
    
    try:
        chroma_manager.delete_collection("verses")
        print("✓ Deleted legacy verses collection")
    except:
        print("No legacy verses collection to delete")
    
    # 3. Generate original language embeddings
    print("\n" + "="*60)
    print("Step 3: Generating original language embeddings")
    print("="*60)
    
    model_manager = EmbeddingModelManager(cache_dir=str(models_path))
    context_builder = ContextBuilder(db_manager)
    
    pipeline = OriginalLanguageEmbeddingPipeline(
        db_manager=db_manager,
        chroma_manager=chroma_manager,
        model_manager=model_manager,
        context_builder=context_builder
    )
    
    print("\nGenerating embeddings for ~29,126 canonical verses...")
    print("This will create ONE embedding per verse using Hebrew/Greek text")
    
    results = pipeline.embed_original_verses(
        batch_size=100,
        force_reembed=True  # Force since we cleaned up
    )
    
    if results.get('status') == 'already_embedded':
        print("\nOriginal verses already embedded")
    else:
        print(f"\n✓ Successfully embedded {results['verses_embedded']:,} canonical verses")
        if results.get('errors'):
            print(f"⚠️  Encountered {len(results['errors'])} errors:")
            for error in results['errors'][:5]:
                print(f"  - {error}")
    
    # 4. Verify the results
    print("\n" + "="*60)
    print("Step 4: Verification")
    print("="*60)
    
    stats = chroma_manager.get_database_stats()
    for collection_name, info in stats['collections'].items():
        print(f"\n{collection_name}:")
        print(f"  Count: {info.get('count', 0):,}")
        print(f"  Dimensions: {info.get('dimensions', 0)}")
        if 'metadata' in info:
            print(f"  Model: {info['metadata'].get('model', 'N/A')}")
            if collection_name == 'original_verses':
                print(f"  Type: {info['metadata'].get('type', 'N/A')}")
                print(f"  Languages: {info['metadata'].get('languages', 'N/A')}")

if __name__ == "__main__":
    main()