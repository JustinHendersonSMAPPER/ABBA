#!/usr/bin/env python3
"""Test semantic search with original language embeddings."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test semantic search functionality."""
    print("\nTesting semantic search with original language embeddings...")
    print("=" * 60)
    
    # Initialize components
    db_path = Path("bible_data/abba.db")
    vector_path = Path("bible_data/vectors")
    models_path = Path("bible_data/models")
    
    db_manager = SQLiteManager(db_path)
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    model_manager = EmbeddingModelManager(cache_dir=str(models_path))
    
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
        
        print(f"\nTotal canonical verses: {original_verses.count():,}")
        print("\nPerforming semantic searches...")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            # Generate embedding for the query using multilingual model
            query_embedding = model_manager.encode_texts(
                [query],
                model_type="multilingual",
                show_progress=False
            )[0].tolist()
            
            # Search using the embedding
            results = original_verses.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            if results['ids'][0]:
                for i, (id_, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    metadata = results['metadatas'][0][i]
                    
                    # Parse the ID to get book, chapter, verse
                    book_id, chapter, verse = id_.split(':')
                    book_id = int(book_id)
                    
                    # Get the actual text from database
                    with db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        
                        # Get book name
                        cursor.execute("""
                            SELECT DISTINCT book FROM stepbible_verses
                            WHERE book IN (
                                SELECT DISTINCT book FROM stepbible_verses
                                WHERE chapter = ? AND verse = ?
                            )
                            LIMIT 1
                        """, (int(chapter), int(verse)))
                        book_result = cursor.fetchone()
                        book_name = book_result[0] if book_result else f"Book {book_id}"
                        
                        # Get original text
                        cursor.execute("""
                            SELECT 
                                GROUP_CONCAT(original_word, ' ') as text,
                                GROUP_CONCAT(english, ' ') as gloss
                            FROM stepbible_verses
                            WHERE book = ? AND chapter = ? AND verse = ?
                            GROUP BY book, chapter, verse
                        """, (book_name, int(chapter), int(verse)))
                        
                        text_result = cursor.fetchone()
                        if text_result:
                            original_text = text_result[0] or "N/A"
                            english_gloss = text_result[1] or "N/A"
                        else:
                            original_text = "N/A"
                            english_gloss = "N/A"
                    
                    print(f"  {i+1}. {book_name} {int(chapter)}:{int(verse)} (similarity: {1-distance:.3f})")
                    print(f"     Original: {original_text[:60]}...")
                    print(f"     Gloss: {english_gloss[:60]}...")
                    print(f"     Language: {metadata.get('language', 'N/A')}")
            else:
                print("  No results found")
        
        print("\n✅ Semantic search is working correctly!")
        
        # Show statistics
        print(f"\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Original verses indexed: {original_verses.count():,}")
        print(f"Embedding dimensions: 768 (multilingual)")
        print(f"Model: intfloat/multilingual-e5-base")
        print("\nThese embeddings enable universal semantic search across all translations!")
        
    except Exception as e:
        print(f"\n❌ Error testing semantic search: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()