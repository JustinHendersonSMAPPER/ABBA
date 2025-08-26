#!/usr/bin/env python3
"""Test original language embeddings are working correctly."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database import SQLiteManager
from abba.embeddings import ChromaManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test that original embeddings are created and searchable."""
    print("\nTesting original language embeddings...")
    print("=" * 60)
    
    # Initialize components
    db_path = Path("bible_data/abba.db")
    vector_path = Path("bible_data/vectors")
    
    db_manager = SQLiteManager(db_path)
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    
    # Get statistics
    stats = chroma_manager.get_database_stats()
    print(f"\nDatabase statistics:")
    for collection_name, info in stats['collections'].items():
        print(f"\n{collection_name}:")
        print(f"  Count: {info.get('count', 0):,}")
        print(f"  Dimensions: {info.get('dimensions', 0)}")
        if 'metadata' in info:
            print(f"  Model: {info['metadata'].get('model', 'N/A')}")
            if collection_name == 'original_verses':
                print(f"  Type: {info['metadata'].get('type', 'N/A')}")
                print(f"  Languages: {info['metadata'].get('languages', 'N/A')}")
    
    # Test search functionality
    print("\n" + "=" * 60)
    print("Testing semantic search...")
    print("=" * 60)
    
    # Get the original verses collection
    try:
        original_verses = chroma_manager.get_collection("original_verses")
        
        # Test queries
        test_queries = [
            "love God with all your heart",
            "faith without works is dead",
            "in the beginning God created",
            "the Lord is my shepherd",
            "do not be anxious about tomorrow"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            # Search using the collection directly
            results = original_verses.query(
                query_texts=[query],
                n_results=3
            )
            
            if results['ids'][0]:
                for i, (id_, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    metadata = results['metadatas'][0][i]
                    
                    # Parse the ID to get book, chapter, verse
                    book_id, chapter, verse = id_.split(':')
                    book_id = int(book_id)
                    
                    # Map book_id to name (simplified)
                    book_names = {
                        1: "Genesis", 19: "Psalms", 40: "Matthew", 
                        42: "Luke", 43: "John", 45: "Romans", 
                        46: "1 Corinthians", 58: "Hebrews", 59: "James"
                    }
                    book_name = book_names.get(book_id, f"Book {book_id}")
                    
                    print(f"  {i+1}. {book_name} {int(chapter)}:{int(verse)} (distance: {distance:.3f})")
                    print(f"     Testament: {metadata.get('testament', 'N/A')}")
                    print(f"     Language: {metadata.get('language', 'N/A')}")
            else:
                print("  No results found")
        
        # Verify count matches expected
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT book || ':' || chapter || ':' || verse) 
                FROM stepbible_verses 
                WHERE original_word IS NOT NULL AND original_word != ''
            """)
            expected_count = cursor.fetchone()[0]
            actual_count = original_verses.count()
            
            print(f"\n" + "=" * 60)
            print("Verification Results")
            print("=" * 60)
            print(f"Expected verses: {expected_count:,}")
            print(f"Actual embeddings: {actual_count:,}")
            
            if actual_count == expected_count:
                print("\n✅ All canonical verses successfully embedded!")
            else:
                print(f"\n⚠️  Missing {expected_count - actual_count:,} verses")
        
        print("\n✅ Original language embeddings are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error testing embeddings: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()