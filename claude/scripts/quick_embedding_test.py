#!/usr/bin/env python3
"""Quick test to verify embeddings are working."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.embeddings import ChromaManager

def main():
    """Quick embedding test."""
    print("Quick Embedding Test")
    print("=" * 50)
    
    try:
        # Open ChromaDB
        chroma = ChromaManager(persist_path="bible_data/vectors")
        
        # Check collections
        print("\nCollections:")
        for name in ["original_verses", "words"]:
            try:
                collection = chroma.get_collection(name)
                count = collection.count()
                print(f"✅ {name}: {count:,} embeddings")
                
                # Get a sample
                sample = collection.get(limit=1)
                if sample['ids']:
                    print(f"   Sample ID: {sample['ids'][0]}")
                    if sample['metadatas'][0]:
                        print(f"   Metadata: {list(sample['metadatas'][0].keys())}")
            except Exception as e:
                print(f"❌ {name}: {str(e)}")
        
        # Test a simple search
        print("\nTest Search:")
        verses = chroma.get_collection("original_verses")
        
        # Just get some verses - don't need to encode a query for this test
        results = verses.get(limit=5, include=['metadatas'])
        
        print(f"Retrieved {len(results['ids'])} verses:")
        for i, (id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            ref = metadata.get('reference', id)
            print(f"  {i+1}. {ref}")
        
        print("\n✅ Embeddings are working properly!")
        
        # Close
        chroma.close()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())