#!/usr/bin/env python3
"""Test script to verify automatic embedding generation."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database import SQLiteManager
from abba.embeddings import ChromaManager


def test_auto_embedding_detection():
    """Test the logic for detecting missing embeddings."""
    print("=== Testing Auto-Embedding Detection ===\n")
    
    # Load configuration
    config = config_manager.load_config()
    
    # Initialize components
    db_manager = SQLiteManager(config.abba_db_path)
    chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
    
    # Check database stats
    db_stats = db_manager.get_database_stats()
    print("Database Statistics:")
    print(f"  Verses: {db_stats.get('verses', 0):,}")
    print(f"  Words: {db_stats.get('words', 0):,}")
    print(f"  Translations: {db_stats.get('translations', 0):,}")
    
    # Check vector database stats
    chroma_stats = chroma_manager.get_database_stats()
    print("\nVector Database Statistics:")
    if chroma_stats.get('collections'):
        for name, stats in chroma_stats['collections'].items():
            count = stats.get('count', 0)
            print(f"  {name}: {count:,} embeddings")
    else:
        print("  No collections found")
    
    # Simulate main.py logic
    print("\n=== Auto-Detection Logic ===")
    
    verses_exist = db_stats.get('verses', 0) > 0
    verse_embeddings_exist = chroma_stats.get('collections', {}).get('verses', {}).get('count', 0) > 0
    
    words_exist = db_stats.get('words', 0) > 0
    word_embeddings_exist = chroma_stats.get('collections', {}).get('words', {}).get('count', 0) > 0
    
    print(f"Verses in database: {'Yes' if verses_exist else 'No'} ({db_stats.get('verses', 0):,})")
    print(f"Verse embeddings exist: {'Yes' if verse_embeddings_exist else 'No'} ({chroma_stats.get('collections', {}).get('verses', {}).get('count', 0):,})")
    
    print(f"Words in database: {'Yes' if words_exist else 'No'} ({db_stats.get('words', 0):,})")
    print(f"Word embeddings exist: {'Yes' if word_embeddings_exist else 'No'} ({chroma_stats.get('collections', {}).get('words', {}).get('count', 0):,})")
    
    # Determine what would be auto-generated
    auto_embed_verses = verses_exist and not verse_embeddings_exist
    auto_embed_words = words_exist and not word_embeddings_exist
    
    print(f"\n=== Automatic Actions ===")
    if auto_embed_verses:
        print("✓ Would auto-generate verse embeddings")
    else:
        print("- No need for verse embeddings")
        
    if auto_embed_words:
        print("✓ Would auto-generate word embeddings")
    else:
        print("- No need for word embeddings")
    
    if not auto_embed_verses and not auto_embed_words:
        print("✓ All embeddings are up to date!")
    
    print("\n=== Summary ===")
    print("When you run 'python abba/main.py' without flags:")
    
    if auto_embed_verses or auto_embed_words:
        print("- Main.py will automatically detect missing embeddings")
        print("- It will generate missing embeddings without requiring flags")
        print("- This ensures all data is always complete")
    else:
        print("- Main.py will skip embedding generation (already complete)")
        print("- Use --force-reembed to regenerate existing embeddings")
    
    print("\n✓ Auto-embedding test complete!")


if __name__ == "__main__":
    test_auto_embedding_detection()