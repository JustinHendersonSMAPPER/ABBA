#!/usr/bin/env python3
"""
Test semantic mapping for a single concept
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.semantic.concept_mapper import ConceptMapper
from abba.logging_setup import logger

def test_single_concept():
    """Test mapping a single concept."""
    
    # Load configuration
    config = config_manager.load_config()
    
    # Setup paths
    db_path = config.data_dir / "abba.db"
    chroma_path = config.vectors_path
    
    # Ollama configuration
    ollama_config = {
        'host': config.ollama_host,
        'models': config.ollama_semantic_models,
        'consensus_threshold': config.ollama_consensus_threshold,
        'timeout': config.ollama_timeout
    }
    
    logger.info(f"Testing single concept mapping...")
    logger.info(f"Database: {db_path}")
    logger.info(f"ChromaDB: {chroma_path}")
    logger.info(f"Ollama: {ollama_config}")
    
    # Initialize mapper
    mapper = ConceptMapper(db_path, chroma_path, ollama_config)
    
    # Test concept data
    concept_data = {
        'name': 'test_love',
        'description': 'Divine and human love',
        'strongs_numbers': ['H0160', 'G0025', 'G0026']
    }
    
    try:
        logger.info(f"Processing test concept: {concept_data['name']}")
        
        stats = mapper.process_concept(
            concept_data,
            max_semantic_results=20,
            validate_semantic=False  # Skip Ollama for this test
        )
        
        logger.info(f"✅ Success!")
        logger.info(f"   Total matches: {stats.total_matches}")
        logger.info(f"   Lexical: {stats.lexical_matches}")
        logger.info(f"   Semantic: {stats.semantic_matches}")
        logger.info(f"   Processing time: {stats.processing_time:.2f}s")
        
        # Test retrieval
        matches = mapper.search_concept('test_love')
        logger.info(f"✅ Retrieved {len(matches)} matches from database")
        
        if matches:
            for i, match in enumerate(matches[:3], 1):
                logger.info(f"   {i}. {match.verse_id} ({match.confidence:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_concept()
    sys.exit(0 if success else 1)