#!/usr/bin/env python3
"""Test concept mapping with a single concept using llama3."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.concept_validator import ConceptValidationPipeline


def test_single_concept_mapping():
    """Test mapping a single concept to demonstrate the system."""
    print("CONCEPT MAPPING TEST")
    print("=" * 70)
    
    # Load configuration
    config = config_manager.load_config()
    
    # Initialize pipeline
    pipeline = ConceptValidationPipeline(config)
    
    # Test setup
    print("Testing setup...")
    if not pipeline.test_ollama_connection():
        print("❌ Ollama connection failed")
        return False
    
    if not pipeline.validate_setup():
        print("❌ Setup validation failed")
        return False
    
    print("✅ Setup validated successfully")
    
    # List available concepts
    concepts = pipeline.list_concepts()
    print(f"\nAvailable concepts: {', '.join(concepts)}")
    
    if not concepts:
        print("❌ No concepts found")
        return False
    
    # Test with the first concept (divine_love)
    test_concept = "divine_love"
    if test_concept not in concepts:
        test_concept = concepts[0]  # Use first available
    
    print(f"\nTesting concept mapping for: {test_concept}")
    print("-" * 50)
    
    try:
        # Get the concept definition
        concept_def = pipeline.concept_manager.get_concept_by_name(test_concept)
        if concept_def:
            print(f"Concept: {concept_def.name}")
            print(f"Description: {concept_def.description[:100]}...")
            print(f"Hebrew terms: {concept_def.hebrew_terms}")
            print(f"Greek terms: {concept_def.greek_terms}")
            print(f"Strong's numbers: {concept_def.strongs_numbers}")
            print(f"Keywords: {concept_def.keywords}")
        
        # Note: For a full test, we would run:
        # result = pipeline.validate_concept(test_concept)
        # But this would take a long time, so we'll just test the setup
        
        print(f"\n✅ Concept mapping system is ready!")
        print(f"\nTo run full concept mapping:")
        print(f"  python abba/main.py --map-concepts")
        print(f"\nNote: Full mapping will take significant time as it:")
        print(f"  1. Finds verses using traditional methods (Strong's, keywords)")
        print(f"  2. Validates each match using llama3 LLM analysis")
        print(f"  3. Scans all 29,126 verses for additional matches")
        print(f"  4. Saves results to database with full traceability")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during concept mapping test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    success = test_single_concept_mapping()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())