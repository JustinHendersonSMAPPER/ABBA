#!/usr/bin/env python3
"""
Debug why embeddings aren't being found for prototype building
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.semantic.strongs_concordance import StrongsConcordance
from abba.semantic.semantic_concordance import ConceptDefinition
from abba.embeddings.chroma_manager import ChromaManager
from abba.logging_setup import logger

def debug_embeddings():
    """Debug embedding lookup issues."""
    
    # Load configuration
    config = config_manager.load_config()
    
    # Setup paths
    db_path = config.data_dir / "abba.db"
    chroma_path = config.vectors_path
    
    logger.info(f"Debugging embedding lookup...")
    logger.info(f"Database: {db_path}")
    logger.info(f"ChromaDB: {chroma_path}")
    
    # Initialize components
    strongs_concordance = StrongsConcordance(db_path)
    chroma_manager = ChromaManager(str(chroma_path))
    
    # Get collection
    collection = chroma_manager.get_collection("original_verses")
    
    # Test concept
    concept = ConceptDefinition(
        name='love',
        description='Divine and human love',
        primary_strongs=['H0160', 'G0025', 'G0026'],
        extended_strongs=[],
        validation_source="test"
    )
    
    try:
        logger.info(f"Finding lexical matches for concept: {concept.name}")
        
        # Get lexical matches
        matches = strongs_concordance.build_concordance(concept)
        logger.info(f"Found {len(matches)} lexical matches")
        
        if matches:
            # Sample first few matches
            logger.info(f"\nSample matches:")
            for i, match in enumerate(matches[:5], 1):
                logger.info(f"   {i}. {match.verse_id} - {match.strongs_matched}")
                
            # Try to get embeddings for these verse IDs
            logger.info(f"\n🔍 Checking embeddings for sample verse IDs...")
            sample_verse_ids = [m.verse_id for m in matches[:5]]
            
            for verse_id in sample_verse_ids:
                try:
                    # Query ChromaDB directly
                    results = collection.get(ids=[verse_id])
                    if results['ids']:
                        logger.info(f"   ✓ {verse_id}: Found embedding")
                    else:
                        logger.warning(f"   ✗ {verse_id}: No embedding found")
                except Exception as e:
                    logger.error(f"   ✗ {verse_id}: Error - {e}")
            
            # Check what IDs actually exist in ChromaDB
            logger.info(f"\n📊 Checking ChromaDB contents...")
            
            # Get a few random IDs from ChromaDB
            sample_results = collection.get(limit=10)
            if sample_results['ids']:
                logger.info(f"Sample ChromaDB IDs:")
                for i, chroma_id in enumerate(sample_results['ids'][:5], 1):
                    logger.info(f"   {i}. {chroma_id}")
                
                # Compare formats
                logger.info(f"\nFormat comparison:")
                logger.info(f"   Lexical match ID: '{sample_verse_ids[0]}'")
                logger.info(f"   ChromaDB ID:      '{sample_results['ids'][0]}'")
            else:
                logger.error("No embeddings found in ChromaDB!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_embeddings()
    sys.exit(0 if success else 1)