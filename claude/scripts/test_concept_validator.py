#!/usr/bin/env python3
"""
Test script for the concept validator.

This demonstrates how to use the concept validator to check if
Hebrew/Greek terms and Strong's numbers in concepts.yaml exist
in the SQLite and embeddings databases.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.logging_setup import setup_logging, configure_standard_logging
from claude.scripts.concept_validator import ConceptValidator


def test_concept_validator():
    """Test the concept validator with current concepts."""
    print("🔍 Testing ABBA Concept Validator")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    configure_standard_logging()
    
    # Load configuration
    config = config_manager.load_config()
    
    # Create validator
    validator = ConceptValidator(config)
    
    print(f"📁 Concepts file: {config.concepts_path}")
    print(f"🗄️  Database: {config.data_dir / 'abba.db'}")
    print(f"🔍 Embeddings: {config.vector_db_path}")
    print()
    
    # Test single concept validation
    print("Testing single concept validation...")
    result = validator.validate_concept("trinity")
    
    if result.validation_passed:
        print(f"✅ Trinity concept validation: PASSED")
    else:
        print(f"❌ Trinity concept validation: FAILED")
        print(f"   Missing Hebrew: {result.hebrew_terms_missing}")
        print(f"   Missing Greek: {result.greek_terms_missing}")
        print(f"   Missing Strong's: {result.strongs_missing}")
    
    print()
    
    # Test all concepts
    print("Running full validation...")
    results = validator.validate_all_concepts()
    
    # Quick summary
    passed = sum(1 for r in results if r.validation_passed)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} concepts")
    print(f"❌ Failed: {total - passed}/{total} concepts")
    
    if total - passed > 0:
        print("\nFailed concepts:")
        for result in results:
            if not result.validation_passed:
                issues = []
                if result.hebrew_terms_missing:
                    issues.append(f"Hebrew: {len(result.hebrew_terms_missing)} missing")
                if result.greek_terms_missing:
                    issues.append(f"Greek: {len(result.greek_terms_missing)} missing")
                if result.strongs_missing:
                    issues.append(f"Strong's: {len(result.strongs_missing)} missing")
                if not result.has_embeddings:
                    issues.append("No embeddings")
                
                print(f"  • {result.concept_name}: {', '.join(issues)}")
    
    # Clean up
    validator.close()
    
    print(f"\n✨ Test completed. Run full validation with:")
    print(f"   python abba/main.py --validate-concept-data")
    
    return passed == total


if __name__ == "__main__":
    success = test_concept_validator()
    sys.exit(0 if success else 1)