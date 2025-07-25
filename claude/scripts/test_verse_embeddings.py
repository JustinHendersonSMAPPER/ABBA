#!/usr/bin/env python3
"""Test verse embedding generation and semantic search."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager, ContextBuilder, EmbeddingPipeline


def test_verse_embeddings():
    """Test verse embedding generation with a small sample."""
    print("=== Testing Verse Embeddings ===\n")
    
    # Load configuration
    config = config_manager.load_config()
    
    # Initialize components
    db_manager = SQLiteManager(config.abba_db_path)
    chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
    model_manager = EmbeddingModelManager(cache_dir=str(config.models_path))
    context_builder = ContextBuilder(db_manager)
    
    # Check what translations we have
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT translation_id FROM verses LIMIT 5")
        translations = [row[0] for row in cursor.fetchall()]
    
    print(f"Available translations (first 5): {translations}")
    
    if not translations:
        print("❌ No translations found in database")
        return False
    
    # Pick first translation and get some sample verses
    test_translation = translations[0]
    print(f"\nTesting with translation: {test_translation}")
    
    # Get sample verses from John 1 (book_id 43 typically)
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT translation_id, book_id, chapter, verse, text
            FROM verses 
            WHERE translation_id = ? 
            AND book_id = 43 
            AND chapter = 1 
            AND verse <= 5
            ORDER BY verse
        """
        cursor.execute(query, (test_translation,))
        sample_verses = cursor.fetchall()
    
    print(f"Found {len(sample_verses)} sample verses from John 1:1-5")
    
    if not sample_verses:
        print("❌ No sample verses found")
        return False
    
    # Show sample verses
    for verse in sample_verses:
        translation_id, book_id, chapter, verse_num, text = verse
        print(f"  {translation_id} {book_id}:{chapter}:{verse_num} - {text[:60]}...")
    
    # Test context building
    print(f"\nTesting context building for first verse...")
    first_verse = sample_verses[0]
    translation_id, book_id, chapter, verse_num, text = first_verse
    
    try:
        context = context_builder.build_verse_context(
            translation_id, book_id, chapter, verse_num
        )
        print(f"Context built: {len(context)} characters")
        print(f"Context preview: {context[:150]}...")
        
        if not context:
            print("❌ Context builder returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Error building context: {e}")
        return False
    
    # Test embedding generation
    print(f"\nTesting embedding generation...")
    try:
        embedding = model_manager.encode_single(
            context,
            model_type="english",
            normalize=True
        )
        print(f"Embedding generated: shape {embedding.shape}, type {type(embedding)}")
        
        if embedding.shape[0] != 1024:  # E5-large-v2 dimensions
            print(f"❌ Unexpected embedding dimensions: {embedding.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False
    
    # Test adding to ChromaDB
    print(f"\nTesting ChromaDB storage...")
    try:
        verses_collection = chroma_manager.get_or_create_collection(
            "test_verses",
            metadata={
                "dimensions": 1024,
                "model": "intfloat/e5-large-v2",
                "type": "test_biblical_verses"
            }
        )
        
        # Add the test embedding
        verse_id = f"{translation_id}:{book_id:03d}:{chapter:03d}:{verse_num:03d}"
        
        verses_collection.add(
            embeddings=[embedding.tolist()],
            ids=[verse_id],
            metadatas=[{
                "translation_id": translation_id,
                "book_id": book_id,
                "chapter": chapter,
                "verse": verse_num,
                "text": text[:500],  # Truncate for storage
                "testament": "new",
                "book_name": "John"
            }]
        )
        
        print(f"✓ Added verse to ChromaDB with ID: {verse_id}")
        
        # Test retrieval
        results = verses_collection.get(ids=[verse_id])
        if results['ids']:
            print(f"✓ Retrieved verse successfully")
        else:
            print(f"❌ Failed to retrieve verse")
            return False
            
    except Exception as e:
        print(f"❌ Error with ChromaDB: {e}")
        return False
    
    # Test semantic search
    print(f"\nTesting semantic search...")
    try:
        # Search for similar verse with a related query
        test_query = "In the beginning was the Word"
        query_embedding = model_manager.encode_single(
            test_query,
            model_type="english",
            normalize=True
        )
        
        search_results = verses_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        if search_results['ids'] and search_results['ids'][0]:
            print(f"✓ Semantic search returned {len(search_results['ids'][0])} results")
            
            # Show top result
            top_result = search_results['ids'][0][0]
            top_distance = search_results['distances'][0][0]
            top_metadata = search_results['metadatas'][0][0]
            similarity = 1 - top_distance
            
            print(f"  Top result: {top_result}")
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Text: {top_metadata.get('text', '')[:100]}...")
            
        else:
            print(f"❌ Semantic search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Error in semantic search: {e}")
        return False
    
    # Clean up test collection
    try:
        chroma_manager.delete_collection("test_verses")
        print(f"✓ Cleaned up test collection")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up test collection: {e}")
    
    print(f"\n🎉 All verse embedding tests passed!")
    print(f"\nThis confirms that:")
    print(f"  ✓ Context building works")
    print(f"  ✓ Embedding generation works") 
    print(f"  ✓ ChromaDB storage works")
    print(f"  ✓ Semantic search works")
    print(f"\nThe issue might be with the full verse embedding pipeline.")
    print(f"Try running: python abba/main.py --embed-verses --force-reembed")
    
    return True


if __name__ == "__main__":
    success = test_verse_embeddings()
    sys.exit(0 if success else 1)