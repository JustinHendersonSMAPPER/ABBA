#!/usr/bin/env python3
"""Reset embedding progress and collections to test fixes."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.embeddings import ChromaManager


def reset_embeddings():
    """Reset embedding progress and collections."""
    print("=== Resetting Embeddings ===\n")
    
    # Load configuration
    config = config_manager.load_config()
    
    # Initialize ChromaDB manager
    chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
    
    # Check current status
    stats = chroma_manager.get_database_stats()
    print("Current ChromaDB status:")
    if stats.get('collections'):
        for name, collection_stats in stats['collections'].items():
            count = collection_stats.get('count', 0)
            print(f"  {name}: {count:,} embeddings")
    else:
        print("  No collections found")
    
    # Reset collections
    print("\nResetting collections...")
    
    try:
        # Delete existing collections
        if 'verses' in stats.get('collections', {}):
            chroma_manager.delete_collection('verses')
            print("✓ Deleted verses collection")
        
        if 'words' in stats.get('collections', {}):
            chroma_manager.delete_collection('words')
            print("✓ Deleted words collection")
        
        # Reset progress file
        progress_file = Path("bible_data") / ".embedding_progress.json"
        if progress_file.exists():
            progress_file.unlink()
            print("✓ Deleted embedding progress file")
        
        print("\n✓ All embeddings reset successfully!")
        print("You can now run 'python abba/main.py' to regenerate embeddings with the fix.")
        
    except Exception as e:
        print(f"Error resetting embeddings: {e}")


if __name__ == "__main__":
    reset_embeddings()