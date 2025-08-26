#!/usr/bin/env python3
"""Test the fixed validator."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database.original_embedding_validator import OriginalEmbeddingValidator
from abba.embeddings import ChromaManager

def main():
    db_path = Path('bible_data/abba.db')
    vector_path = Path('bible_data/vectors')
    
    # Create ChromaManager
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    
    # Create validator with existing ChromaManager
    validator = OriginalEmbeddingValidator(
        db_path=db_path,
        vector_path=vector_path,
        chroma_manager=chroma_manager
    )
    
    # Run validation
    results, success = validator.validate_all()
    validator.print_summary(results, success)
    
    # Close ChromaDB
    chroma_manager.close()
    
    if success:
        print('\n✅ All validation checks passed!')
        return 0
    else:
        print('\n❌ Some validation checks failed')
        return 1

if __name__ == "__main__":
    sys.exit(main())