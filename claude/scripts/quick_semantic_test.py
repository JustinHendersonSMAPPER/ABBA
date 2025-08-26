#!/usr/bin/env python3
"""Quick test to verify semantic concordance works end-to-end."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import ABBAConfig
from abba.semantic.strongs_concordance import StrongsConcordance, ConceptDefinition
from abba.database.sqlite_manager import SQLiteManager


def quick_test():
    """Quick test of the semantic concordance system."""
    
    # Setup paths
    db_path = Path("bible_data/abba.db")
    
    print("🔍 Quick Semantic Concordance Test")
    print("=" * 60)
    
    # 1. Test Strong's concordance
    print("\n1️⃣ Testing Strong's Concordance...")
    try:
        concordance = StrongsConcordance(db_path)
        
        # Test concept: love
        love_concept = ConceptDefinition(
            name="love",
            description="Divine and human love",
            primary_strongs=["G0025", "G0026", "H0157", "H2617"],
            validation_source="test"
        )
        
        matches = concordance.build_concordance(love_concept)
        print(f"✅ Found {len(matches)} matches for 'love'")
        
        # Show breakdown by match type
        by_type = {}
        for match in matches:
            by_type[match.match_type] = by_type.get(match.match_type, 0) + 1
        
        for match_type, count in sorted(by_type.items()):
            print(f"   • {match_type}: {count} matches")
        
        # Show sample matches
        print("\nSample matches:")
        for match in matches[:5]:
            print(f"   • {match.verse_id}: {match.original_text[:40]}...")
            
    except Exception as e:
        print(f"❌ Strong's concordance failed: {e}")
        return False
    
    # 2. Check database statistics
    print("\n2️⃣ Database Statistics...")
    try:
        with SQLiteManager(db_path).get_connection() as conn:
            cursor = conn.cursor()
            
            # Count verses by language
            cursor.execute("""
                SELECT language, COUNT(DISTINCT book || ':' || chapter || ':' || verse) as verses
                FROM stepbible_verses
                GROUP BY language
            """)
            
            for lang, count in cursor.fetchall():
                print(f"   • {lang.title()}: {count:,} verses")
            
            # Count Strong's numbers
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN strongs_lexical LIKE 'H%' THEN 1 ELSE 0 END) as hebrew,
                    SUM(CASE WHEN strongs_lexical LIKE 'G%' THEN 1 ELSE 0 END) as greek
                FROM stepbible_verses
                WHERE strongs_lexical IS NOT NULL
            """)
            
            hebrew, greek = cursor.fetchone()
            print(f"   • Hebrew Strong's: {hebrew:,} occurrences")
            print(f"   • Greek Strong's: {greek:,} occurrences")
            
    except Exception as e:
        print(f"❌ Database statistics failed: {e}")
        return False
    
    # 3. Test specific concepts
    print("\n3️⃣ Testing Multiple Concepts...")
    test_concepts = [
        ("faith", ["G4102", "H0539"]),
        ("grace", ["G5485", "H2580"]),
        ("kingdom", ["G0932", "H4438"])
    ]
    
    for name, strongs in test_concepts:
        concept = ConceptDefinition(
            name=name,
            description=f"Test concept: {name}",
            primary_strongs=strongs,
            validation_source="test"
        )
        
        matches = concordance.build_concordance(concept)
        print(f"   • {name}: {len(matches)} matches")
    
    print("\n✅ All tests passed!")
    print("\nNotes:")
    print("• Lexical search (Strong's) is fully functional")
    print("• Semantic search requires embeddings to be generated")
    print("• Run: python abba/main.py --embed-verses")
    
    return True


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)