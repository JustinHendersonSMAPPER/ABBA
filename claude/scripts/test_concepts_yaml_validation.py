#!/usr/bin/env python3
"""
Test Concepts YAML with Strong's-Centric Approach

Tests how well the existing concepts.yaml works with our Strong's-based
semantic search methodology.
"""

import sys
import yaml
from pathlib import Path
from collections import defaultdict

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.semantic.strongs_concordance import StrongsConcordance, ConceptDefinition
from abba.logging_setup import logger


def load_concepts_yaml(concepts_path: Path):
    """Load concepts from YAML file."""
    with open(concepts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def convert_to_strongs_definition(concept_data: dict) -> ConceptDefinition:
    """Convert YAML concept to Strong's-based definition."""
    # Extract Strong's numbers from the concept
    primary_strongs = []
    extended_strongs = []
    
    # Collect all Strong's numbers
    if 'strongs_numbers' in concept_data:
        primary_strongs.extend(concept_data['strongs_numbers'])
    
    # For now, treat all as primary since we don't have the distinction
    # in the current YAML format
    
    return ConceptDefinition(
        name=concept_data.get('name', 'unnamed'),
        description=concept_data.get('description', ''),
        primary_strongs=primary_strongs,
        extended_strongs=[],  # Current YAML doesn't distinguish
        validation_source="concepts.yaml"
    )


def test_concepts_yaml():
    """Test all concepts from concepts.yaml."""
    # Load configuration
    config = config_manager.load_config()
    db_path = config.data_dir / "abba.db"
    concepts_path = config.concepts_path
    
    # Initialize concordance
    concordance = StrongsConcordance(db_path)
    
    # Load concepts
    logger.info(f"Loading concepts from {concepts_path}")
    concepts_data = load_concepts_yaml(concepts_path)
    
    if 'concepts' not in concepts_data:
        logger.error("No 'concepts' key found in YAML")
        return False
    
    print("\n🔍 Testing Concepts from concepts.yaml")
    print("=" * 80)
    
    # Track statistics
    total_concepts = 0
    concepts_with_matches = 0
    strongs_coverage = defaultdict(int)
    missing_strongs = []
    
    # Test each concept
    for concept_data in concepts_data['concepts']:
        total_concepts += 1
        name = concept_data.get('name', 'unnamed')
        
        print(f"\n📖 Testing concept: {name}")
        print(f"   Description: {concept_data.get('description', 'N/A')[:80]}...")
        
        # Get Strong's numbers
        strongs_numbers = concept_data.get('strongs_numbers', [])
        hebrew_terms = concept_data.get('hebrew_terms', [])
        greek_terms = concept_data.get('greek_terms', [])
        
        print(f"   Strong's numbers: {len(strongs_numbers)}")
        print(f"   Hebrew terms: {len(hebrew_terms)}")
        print(f"   Greek terms: {len(greek_terms)}")
        
        if not strongs_numbers:
            print("   ⚠️  No Strong's numbers - skipping concordance search")
            continue
        
        # Convert to Strong's definition
        try:
            strongs_def = convert_to_strongs_definition(concept_data)
            
            # Build concordance
            matches = concordance.build_concordance(strongs_def)
            
            if matches:
                concepts_with_matches += 1
                print(f"   ✅ Found {len(matches)} matches")
                
                # Count by Strong's number
                for strongs in strongs_numbers:
                    strongs_matches = [m for m in matches if strongs in m.strongs_matched]
                    if strongs_matches:
                        strongs_coverage[strongs] = len(strongs_matches)
                        print(f"      • {strongs}: {len(strongs_matches)} verses")
                    else:
                        missing_strongs.append((name, strongs))
                        print(f"      • {strongs}: ❌ NOT FOUND")
                
                # Show sample matches
                print(f"   📝 Sample matches:")
                for match in matches[:3]:
                    print(f"      • {match.verse_id} - {match.original_text[:30]}...")
            else:
                print(f"   ❌ No matches found!")
                for strongs in strongs_numbers:
                    missing_strongs.append((name, strongs))
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total concepts tested: {total_concepts}")
    print(f"Concepts with matches: {concepts_with_matches}")
    print(f"Success rate: {concepts_with_matches/total_concepts*100:.1f}%")
    
    # Strong's coverage
    print(f"\n📈 Strong's Number Coverage:")
    print(f"Total unique Strong's numbers: {len(strongs_coverage)}")
    print(f"Average verses per Strong's: {sum(strongs_coverage.values())/len(strongs_coverage):.1f}" if strongs_coverage else "N/A")
    
    # Most common Strong's
    if strongs_coverage:
        print(f"\n🔝 Most frequent Strong's numbers:")
        for strongs, count in sorted(strongs_coverage.items(), key=lambda x: -x[1])[:10]:
            print(f"   • {strongs}: {count} verses")
    
    # Missing Strong's
    if missing_strongs:
        print(f"\n❌ Missing Strong's numbers ({len(missing_strongs)} total):")
        for concept, strongs in missing_strongs[:10]:
            print(f"   • {concept}: {strongs}")
        if len(missing_strongs) > 10:
            print(f"   ... and {len(missing_strongs) - 10} more")
    
    # Concepts without Strong's
    concepts_without_strongs = []
    for concept_data in concepts_data['concepts']:
        if not concept_data.get('strongs_numbers'):
            concepts_without_strongs.append(concept_data.get('name', 'unnamed'))
    
    if concepts_without_strongs:
        print(f"\n⚠️  Concepts without Strong's numbers ({len(concepts_without_strongs)}):")
        for name in concepts_without_strongs:
            print(f"   • {name}")
    
    print("\n" + "=" * 80)
    print("🔍 ANALYSIS")
    print("=" * 80)
    
    if concepts_with_matches == 0:
        print("❌ CRITICAL: No concepts produced matches!")
        print("   This suggests a fundamental issue with either:")
        print("   1. The Strong's numbers in concepts.yaml")
        print("   2. The database content")
        print("   3. The concordance search logic")
    elif concepts_with_matches < total_concepts * 0.5:
        print("⚠️  WARNING: Less than 50% of concepts produced matches")
        print("   Consider:")
        print("   1. Reviewing Strong's numbers for accuracy")
        print("   2. Adding missing Strong's numbers to concepts")
        print("   3. Checking database completeness")
    else:
        print("✅ Good coverage! Most concepts are producing matches.")
    
    return concepts_with_matches > 0


def main():
    """Run the test."""
    print("🚀 Testing Concepts YAML with Strong's-Centric Approach")
    print("=" * 80)
    
    try:
        success = test_concepts_yaml()
        
        if success:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test revealed issues that need attention")
            
        return success
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)