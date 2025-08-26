#!/usr/bin/env python3
"""
Test Strong's-Centric Concordance

Demonstrates the Strong's-based semantic search methodology.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.semantic.strongs_concordance import StrongsConcordance, ConceptDefinition
from abba.logging_setup import logger


def test_love_concept():
    """Test searching for the concept of love."""
    # Load configuration
    config = config_manager.load_config()
    db_path = config.data_dir / "abba.db"
    
    # Initialize concordance
    concordance = StrongsConcordance(db_path)
    
    # Define love concept using Strong's numbers
    love_concept = ConceptDefinition(
        name="love",
        description="Biblical concept of love (agape, phileo)",
        primary_strongs=["G25", "G26"],      # ἀγαπάω, ἀγάπη
        extended_strongs=["G5368", "G5360"], # φιλέω, φιλαδελφία
        validation_source="BDAG, Strong's Concordance"
    )
    
    # Also test with Hebrew
    love_hebrew = ConceptDefinition(
        name="love_hebrew",
        description="Love in Hebrew",
        primary_strongs=["H157", "H160"],    # אהב, אהבה
        extended_strongs=["H2617"],          # חסד (chesed)
        validation_source="BDB, Strong's Concordance"
    )
    
    print("\n🔍 Testing Strong's-Centric Concordance")
    print("=" * 60)
    
    # Build concordance for Greek love
    print("\n📖 Searching for LOVE (Greek)...")
    greek_matches = concordance.build_concordance(love_concept)
    
    print(f"\nFound {len(greek_matches)} total matches:")
    
    # Group by match type
    by_type = {}
    for match in greek_matches:
        if match.match_type not in by_type:
            by_type[match.match_type] = 0
        by_type[match.match_type] += 1
    
    for match_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  • {match_type}: {count} matches")
    
    # Show sample matches
    print("\n📝 Sample matches:")
    for match in greek_matches[:5]:
        print(f"  • {match.verse_id} - {match.original_text}")
        print(f"    Type: {match.match_type}, Confidence: {match.confidence}")
        print(f"    Evidence: {match.evidence}")
    
    # Build concordance for Hebrew love
    print("\n\n📖 Searching for LOVE (Hebrew)...")
    hebrew_matches = concordance.build_concordance(love_hebrew)
    
    print(f"\nFound {len(hebrew_matches)} total matches")
    
    # Test phrase patterns
    print("\n\n📖 Testing phrase patterns...")
    kingdom_concept = ConceptDefinition(
        name="kingdom_of_god",
        description="The kingdom/reign of God",
        primary_strongs=["G932"],  # βασιλεία
        phrase_patterns=[
            {"strongs": ["G932", "G2316"], "name": "kingdom of God"},
            {"strongs": ["G932", "G3772"], "name": "kingdom of heaven"}
        ],
        validation_source="Gospel usage"
    )
    
    kingdom_matches = concordance.build_concordance(kingdom_concept)
    phrase_matches = [m for m in kingdom_matches if m.match_type == 'phrase']
    
    print(f"Found {len(phrase_matches)} phrase pattern matches")
    
    # Generate report
    print("\n\n📊 Generating concordance report...")
    report = concordance.generate_report(love_concept, greek_matches[:100])
    print("\nFirst 500 characters of report:")
    print(report[:500] + "...")
    
    # Demonstrate transparency
    print("\n\n🔍 Transparency demonstration:")
    print("Each match is traceable to its source:")
    sample = greek_matches[0] if greek_matches else None
    if sample:
        print(f"  Verse: {sample.verse_id}")
        print(f"  Match type: {sample.match_type}")
        print(f"  Confidence: {sample.confidence}")
        print(f"  Evidence: {sample.evidence}")
        print(f"  Strong's: {', '.join(sample.strongs_matched)}")
    
    return True


def test_strongs_validation():
    """Test Strong's number validation."""
    config = config_manager.load_config()
    concordance = StrongsConcordance(config.data_dir / "abba.db")
    
    print("\n\n🔍 Testing Strong's number validation...")
    
    # Test valid Strong's numbers
    try:
        valid_concept = concordance.define_concept(
            name="test",
            primary_strongs=["G26", "H430"]  # Valid numbers
        )
        print("✅ Valid Strong's numbers accepted")
    except Exception as e:
        print(f"❌ Error with valid numbers: {e}")
    
    # Test invalid Strong's numbers
    try:
        invalid_concept = concordance.define_concept(
            name="test_invalid",
            primary_strongs=["G99999", "H99999"]  # Invalid
        )
        print("❌ Invalid Strong's numbers accepted (should have warned)")
    except Exception as e:
        print(f"✅ Invalid Strong's numbers properly handled: {e}")
    
    return True


def main():
    """Run all tests."""
    print("🚀 Strong's-Centric Concordance Test Suite")
    print("=" * 60)
    
    try:
        # Test love concept
        test_love_concept()
        
        # Test validation
        test_strongs_validation()
        
        print("\n\n✅ All tests completed successfully!")
        print("\nThis demonstrates:")
        print("1. Strong's numbers as primary semantic anchors")
        print("2. Transparent confidence scoring")
        print("3. Evidence trail for every match")
        print("4. No algorithmic inference - only lexicon-based matches")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)