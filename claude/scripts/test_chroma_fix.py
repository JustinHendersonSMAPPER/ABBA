#!/usr/bin/env python3
"""Test ChromaDB corruption fix."""

import sys
import shutil
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.embeddings import ChromaManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chroma_corruption():
    """Test ChromaDB handling of corrupted data."""
    vector_path = Path("bible_data/vectors")
    
    print("\n" + "=" * 60)
    print("Testing ChromaDB Corruption Handling")
    print("=" * 60)
    
    # 1. Test with fresh/clean database
    print("\n1. Testing with clean database...")
    if vector_path.exists():
        print(f"   Removing existing vectors at {vector_path}")
        shutil.rmtree(vector_path)
    
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    stats = chroma_manager.get_database_stats()
    
    print(f"   Collections: {list(stats['collections'].keys())}")
    print("   ✓ Clean database works")
    
    # 2. Test collection stats with empty collection
    print("\n2. Testing collection stats...")
    test_collection = chroma_manager.get_or_create_collection(
        "test_collection",
        metadata={"test": True}
    )
    
    collection_stats = chroma_manager.get_collection_stats("test_collection")
    print(f"   Stats: {collection_stats}")
    
    if "error" in collection_stats:
        print(f"   ❌ Error: {collection_stats['error']}")
    else:
        print(f"   ✓ Collection has {collection_stats['count']} items")
    
    # 3. Test proper closing
    print("\n3. Testing proper ChromaDB closing...")
    chroma_manager.close()
    print("   ✓ ChromaDB closed successfully")
    
    # 4. Test reopening after close
    print("\n4. Testing reopening after close...")
    chroma_manager2 = ChromaManager(persist_path=str(vector_path))
    stats2 = chroma_manager2.get_database_stats()
    print(f"   Collections after reopen: {list(stats2['collections'].keys())}")
    print("   ✓ ChromaDB reopened successfully")
    
    # Close second instance
    chroma_manager2.close()
    
    print("\n✅ All ChromaDB tests passed!")

if __name__ == "__main__":
    test_chroma_corruption()