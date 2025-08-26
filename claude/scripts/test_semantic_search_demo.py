#!/usr/bin/env python3
"""
Simple demonstration of semantic search functionality.

Shows how the original language embeddings enable universal search
across all Bible translations with a single embedding per verse.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.embeddings import ChromaManager, EmbeddingModelManager
import sqlite3


def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    # Initialize
    chroma_manager = ChromaManager(persist_path="bible_data/vectors")
    model_manager = EmbeddingModelManager()
    multilingual_model = model_manager.get_model("multilingual")
    verses_collection = chroma_manager.get_collection("original_verses")
    
    print("SEMANTIC SEARCH DEMONSTRATION")
    print("=" * 70)
    print(f"Using {verses_collection.count():,} original language embeddings")
    print()
    
    # Example searches
    searches = [
        ("love your enemies", "Finding verses about loving enemies"),
        ("faith moves mountains", "Finding verses about faith's power"),
        ("light in darkness", "Finding verses about light overcoming darkness"),
    ]
    
    for query, description in searches:
        print(f"\n{description}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        # Encode and search
        query_embedding = multilingual_model.encode(query)
        results = verses_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            include=['metadatas', 'distances']
        )
        
        # Display results
        for i, (verse_id, metadata, distance) in enumerate(zip(
            results['ids'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance
            print(f"\n{i+1}. {metadata.get('reference', verse_id)} (similarity: {similarity:.3f})")
            
            # Show original text if available
            if 'original_text' in metadata:
                print(f"   Original: {metadata['original_text'][:80]}...")
            
            # Show sample translation
            book_id, chapter, verse = verse_id.split(':')
            verse_text = get_verse_text(int(book_id), int(chapter), int(verse))
            if verse_text:
                print(f"   KJV: {verse_text[:80]}...")
    
    print("\n" + "=" * 70)
    print("Key Benefits:")
    print("- One embedding per verse (not per translation)")
    print("- Searches work across all 1,204 translations")
    print("- Based on original Hebrew/Greek/Aramaic text")
    print("- ~475x storage reduction vs translation-specific embeddings")
    
    chroma_manager.close()


def get_verse_text(book_id: int, chapter: int, verse: int, translation: str = 'KJV') -> str:
    """Get verse text from database."""
    try:
        with sqlite3.connect('bible_data/abba.db') as conn:
            cursor = conn.cursor()
            
            # Get translation ID
            cursor.execute("SELECT id FROM translations WHERE abbreviation = ?", (translation,))
            trans_row = cursor.fetchone()
            if not trans_row:
                return None
            
            # Get verse
            cursor.execute("""
                SELECT text FROM verses 
                WHERE translation_id = ? AND book_id = ? AND chapter = ? AND verse = ?
            """, (trans_row[0], book_id, chapter, verse))
            
            row = cursor.fetchone()
            return row[0] if row else None
            
    except Exception:
        return None


if __name__ == "__main__":
    demo_semantic_search()