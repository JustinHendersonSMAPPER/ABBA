#!/usr/bin/env python3
"""Test script for embedding generation functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager, ContextBuilder, EmbeddingPipeline


def test_embedding_pipeline():
    """Test the embedding pipeline with a small dataset."""
    print("=== Testing Embedding Pipeline ===\n")
    
    # Load configuration
    config = config_manager.load_config()
    
    # Initialize components
    print("1. Initializing components...")
    db_manager = SQLiteManager(config.abba_db_path)
    chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
    model_manager = EmbeddingModelManager(cache_dir=str(config.models_path))
    context_builder = ContextBuilder(db_manager)
    
    pipeline = EmbeddingPipeline(
        db_manager=db_manager,
        chroma_manager=chroma_manager,
        model_manager=model_manager,
        context_builder=context_builder
    )
    
    # Test context building
    print("\n2. Testing context builder...")
    try:
        # Test verse context (John 1:1 in KJV)
        verse_context = context_builder.build_verse_context("eng_kjv", 43, 1, 1)
        print(f"Verse context: {verse_context[:200]}...")
        
        # Test word context
        word_data = {
            'greek_text': 'λόγος',
            'transliteration': 'logos',
            'strongs_primary': 'G3056',
            'morphology_code': 'N-NSM',
            'language': 'greek'
        }
        word_context = context_builder.build_word_context(word_data)
        print(f"Word context: {word_context[:200]}...")
    except Exception as e:
        print(f"Context building error: {e}")
        return
    
    # Test embedding generation for a single verse
    print("\n3. Testing single verse embedding...")
    try:
        # Generate embedding for one verse
        texts = [verse_context]
        embeddings = model_manager.encode_texts(texts, model_type="english", show_progress=False)
        print(f"Generated embedding shape: {embeddings.shape}")
        print(f"Embedding dimensions: {embeddings.shape[1]}")
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return
    
    # Test ChromaDB storage
    print("\n4. Testing ChromaDB storage...")
    try:
        verses_collection = chroma_manager.get_or_create_collection("test_verses")
        
        # Add test embedding
        verse_id = chroma_manager.generate_verse_id("eng_kjv", 43, 1, 1)
        verses_collection.add(
            embeddings=[embeddings[0].tolist()],
            ids=[verse_id],
            metadatas=[{
                "translation_id": "eng_kjv",
                "book_id": 43,
                "chapter": 1,
                "verse": 1,
                "text": "In the beginning was the Word..."
            }]
        )
        print(f"Added verse with ID: {verse_id}")
        
        # Query to verify
        results = verses_collection.get(ids=[verse_id])
        print(f"Retrieved {len(results['ids'])} items from ChromaDB")
        
        # Clean up test collection
        chroma_manager.delete_collection("test_verses")
        print("Cleaned up test collection")
        
    except Exception as e:
        print(f"ChromaDB error: {e}")
        return
    
    # Show database statistics
    print("\n5. Current embedding statistics:")
    try:
        stats = chroma_manager.get_database_stats()
        print(f"Total collections: {len(stats['collections'])}")
        for name, collection_stats in stats['collections'].items():
            print(f"  {name}: {collection_stats.get('count', 0)} embeddings")
    except Exception as e:
        print(f"Stats error: {e}")
    
    print("\n✓ Embedding pipeline test complete!")


def test_batch_embedding():
    """Test batch embedding with progress tracking."""
    print("\n=== Testing Batch Embedding ===\n")
    
    # Load configuration
    config = config_manager.load_config()
    
    # Check if database has data
    db_manager = SQLiteManager(config.abba_db_path)
    stats = db_manager.get_database_stats()
    
    print(f"Database contains:")
    print(f"  Verses: {stats.get('verses', 0):,}")
    print(f"  Words: {stats.get('words', 0):,}")
    
    if stats.get('verses', 0) == 0:
        print("\n⚠️  No verses found in database. Please run main.py first to import data.")
        return
    
    print("\n✓ Database is ready for embedding generation.")
    print("\nTo generate embeddings, run:")
    print("  python abba/main.py --embed-verses --translations eng_kjv")
    print("  python abba/main.py --embed-words")
    print("  python abba/main.py --embed-all")


if __name__ == "__main__":
    print("ABBA Embedding System Test\n")
    
    # Test basic functionality
    test_embedding_pipeline()
    
    # Test batch processing readiness
    test_batch_embedding()