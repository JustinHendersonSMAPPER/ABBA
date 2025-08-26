#!/usr/bin/env python3
"""Clean up legacy embeddings and generate original language embeddings."""

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
    
    db_manager = SQLiteManager(db_path)
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    
    # 1. Clean up legacy translation-specific embeddings
    print("\n" + "="*60)
    print("Step 1: Cleaning up legacy translation embeddings")
    print("="*60)
    
    try:
        verses_collection = chroma_manager.get_collection("verses")
        if verses_collection:
            count = verses_collection.count()
            print(f"Found {count:,} legacy translation-specific embeddings")
            
            response = input("\nDelete these legacy embeddings? (y/N): ")
            if response.lower() == 'y':
                chroma_manager.delete_collection("verses")
                print("✓ Legacy embeddings deleted")
            else:
                print("Skipping deletion")
    except:
        print("No legacy embeddings found")
    
    # 2. Generate original language embeddings
    print("\n" + "="*60)
    print("Step 2: Generating original language embeddings")
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
        force_reembed=False
    )
    
    if results.get('status') == 'already_embedded':
        print("\nOriginal verses already embedded")
    else:
        print(f"\n✓ Successfully embedded {results['verses_embedded']:,} canonical verses")
        if results.get('errors'):
            print(f"⚠️  Encountered {len(results['errors'])} errors:")
            for error in results['errors'][:5]:
                print(f"  - {error}")
    
    # 3. Verify the results
    print("\n" + "="*60)
    print("Step 3: Verification")
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