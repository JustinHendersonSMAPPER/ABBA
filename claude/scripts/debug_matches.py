#!/usr/bin/env python3
"""
Debug duplicate matches issue
"""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.semantic.semantic_concordance import SemanticConcordance, ConceptDefinition
from abba.logging_setup import logger

def debug_matches():
    """Debug duplicate matches."""
    
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
    
    logger.info(f"Debugging duplicate matches...")
    
    # Initialize semantic concordance
    concordance = SemanticConcordance(db_path, chroma_path, ollama_config)
    
    # Test concept
    concept = ConceptDefinition(
        name='test_love',
        description='Divine and human love',
        primary_strongs=['H0160', 'G0025', 'G0026'],
        extended_strongs=[],
        validation_source="test"
    )
    
    try:
        logger.info(f"Building concordance for {concept.name}")
        
        matches = concordance.build_semantic_concordance(
            concept,
            max_semantic_results=20,
            validate_semantic=False
        )
        
        logger.info(f"Got {len(matches)} total matches")
        
        # Check for duplicates
        verse_ids = [m.verse_id for m in matches]
        verse_counts = Counter(verse_ids)
        
        duplicates = {vid: count for vid, count in verse_counts.items() if count > 1}
        
        if duplicates:
            logger.warning(f"Found {len(duplicates)} verses with duplicate matches:")
            for vid, count in sorted(duplicates.items()):
                logger.warning(f"  {vid}: {count} matches")
                
                # Show the duplicates
                verse_matches = [m for m in matches if m.verse_id == vid]
                for i, match in enumerate(verse_matches):
                    logger.warning(f"    {i+1}. {match.match_type}, conf={match.confidence:.3f}, evidence={match.evidence[:50]}")
        else:
            logger.info("✅ No duplicate verse IDs found")
        
        # Check concept-verse pairs (what the constraint actually checks)
        pairs = [(concept.name, m.verse_id) for m in matches]
        pair_counts = Counter(pairs)
        
        duplicate_pairs = {pair: count for pair, count in pair_counts.items() if count > 1}
        
        if duplicate_pairs:
            logger.error(f"Found {len(duplicate_pairs)} concept-verse pairs with duplicates:")
            for (cname, vid), count in sorted(duplicate_pairs.items()):
                logger.error(f"  ({cname}, {vid}): {count} matches")
        else:
            logger.info("✅ No duplicate concept-verse pairs found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_matches()
    sys.exit(0 if success else 1)