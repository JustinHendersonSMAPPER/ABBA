#!/usr/bin/env python3
"""
Test semantic search for love concept directly to debug Ollama validation
"""

import sys
from pathlib import Path
import logging

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

from abba.config import config_manager
from abba.semantic.semantic_concordance import SemanticConcordance, ConceptDefinition
from abba.logging_setup import logger

def test_love_semantic():
    """Test semantic search for love with debug output."""
    
    # Load configuration
    config = config_manager.load_config()
    
    # Setup paths
    db_path = config.data_dir / "abba.db"
    chroma_path = config.vectors_path
    
    # Ollama configuration
    ollama_config = {
        'host': config.ollama_host,
        'models': config.ollama_semantic_models,
        'consensus_threshold': 0.5,  # Lower threshold for testing
        'timeout': 60  # Longer timeout
    }
    
    logger.info(f"Testing semantic search for 'love' concept")
    logger.info(f"Database: {db_path}")
    logger.info(f"ChromaDB: {chroma_path}")
    logger.info(f"Ollama: {ollama_config}")
    
    # Initialize semantic concordance
    concordance = SemanticConcordance(db_path, chroma_path, ollama_config)
    
    # Define love concept
    concept = ConceptDefinition(
        name='love',
        description='Divine and human love, affection, charity, benevolence',
        primary_strongs=['H0157', 'H0160', 'H1730', 'G0025', 'G0026', 'G5368'],
        extended_strongs=['H2617', 'H7355', 'G5360', 'G5361', 'G5362'],
        validation_source="Biblical lexicons"
    )
    
    try:
        logger.info(f"Building semantic concordance for: {concept.name}")
        
        # Build concordance with only 10 semantic candidates for faster testing
        matches = concordance.build_semantic_concordance(
            concept,
            max_semantic_results=10,  # Just test with 10 candidates
            validate_semantic=True    # Enable Ollama validation
        )
        
        logger.info(f"✅ Found {len(matches)} total matches")
        
        # Analyze results
        lexical = [m for m in matches if not m.is_semantic_only]
        semantic = [m for m in matches if m.is_semantic_only]
        
        logger.info(f"   Lexical matches: {len(lexical)}")
        logger.info(f"   Semantic matches: {len(semantic)}")
        
        if semantic:
            logger.info(f"\n🎉 SUCCESS! Found {len(semantic)} semantic matches:")
            for i, match in enumerate(semantic[:5], 1):
                logger.info(f"   {i}. {match.verse_id}")
                logger.info(f"      Confidence: {match.confidence:.3f}")
                logger.info(f"      Semantic score: {match.semantic_score:.3f}")
                logger.info(f"      Ollama validation: {match.ollama_validation}")
                logger.info(f"      Evidence: {match.evidence}")
        else:
            logger.warning(f"\n⚠️ No semantic matches found")
            logger.warning(f"This indicates Ollama validation is still filtering everything")
            
            # Let's test the Ollama validation directly
            logger.info(f"\n🔍 Testing Ollama validation directly...")
            
            # Get a sample verse that should match
            test_verse_data = {
                'book': 'John',
                'chapter': 3,
                'verse': 16,
                'original_text': 'οὕτως γὰρ ἠγάπησεν ὁ θεὸς τὸν κόσμον',
                'strongs_in_verse': ['G0025']  # agapao - to love
            }
            
            logger.info(f"Testing with John 3:16 (contains ἠγάπησεν - 'loved')")
            
            try:
                result = concordance._validate_with_ollama(concept, test_verse_data)
                logger.info(f"Ollama validation result:")
                logger.info(f"   Answer: {result.answer}")
                logger.info(f"   Confidence: {result.confidence}")
                logger.info(f"   Explanation: {result.explanation}")
                
                if result.answer == 'NO':
                    logger.error(f"❌ Ollama incorrectly rejected John 3:16 for 'love' concept!")
                    logger.error(f"This is a clear bug - John 3:16 is THE classic verse about God's love")
            except Exception as e:
                logger.error(f"❌ Ollama validation error: {e}")
                import traceback
                traceback.print_exc()
        
        return len(semantic) > 0  # Success if we got semantic matches
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_love_semantic()
    sys.exit(0 if success else 1)