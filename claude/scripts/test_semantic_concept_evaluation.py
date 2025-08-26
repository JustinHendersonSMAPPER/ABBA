#!/usr/bin/env python3
"""
Test Semantic Concept Evaluation

This script tests the complete semantic concordance pipeline including:
1. Strong's-based lexical matching
2. Embedding-based semantic search
3. Ollama validation for false positive reduction
4. Combined ranking and reporting
"""

import sys
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
import time

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.semantic.semantic_concordance import SemanticConcordance, ConceptDefinition
from abba.logging_setup import logger


def load_concept_from_yaml(concepts_path: Path, concept_name: str) -> ConceptDefinition:
    """Load a specific concept from concepts.yaml."""
    with open(concepts_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    for concept_data in data['concepts']:
        if concept_data['name'] == concept_name:
            return ConceptDefinition(
                name=concept_data['name'],
                description=concept_data.get('description', ''),
                primary_strongs=concept_data.get('strongs_numbers', []),
                extended_strongs=[],  # Could be enhanced later
                validation_source="concepts.yaml"
            )
    
    raise ValueError(f"Concept '{concept_name}' not found in {concepts_path}")


def test_semantic_concept(concept_name: str, 
                         max_semantic_results: int = 50,
                         validate_semantic: bool = True):
    """Test semantic concordance for a specific concept."""
    
    # Load configuration
    config = config_manager.load_config()
    db_path = config.data_dir / "abba.db"
    chroma_path = config.data_dir / "chroma"
    concepts_path = config.concepts_path
    
    # Ollama configuration
    ollama_config = {
        'host': config.ollama_host,
        'models': config.ollama_semantic_models,
        'consensus_threshold': config.ollama_consensus_threshold,
        'timeout': config.ollama_timeout
    }
    
    print(f"\n🔍 Testing Semantic Concordance for: {concept_name}")
    print("=" * 80)
    
    try:
        # Load concept
        concept = load_concept_from_yaml(concepts_path, concept_name)
        print(f"\n📖 Concept: {concept.name}")
        print(f"Description: {concept.description[:100]}...")
        print(f"Primary Strong's: {', '.join(concept.primary_strongs)}")
        
        # Initialize semantic concordance
        print(f"\n🚀 Initializing semantic concordance...")
        print(f"   Database: {db_path}")
        print(f"   Embeddings: {chroma_path}")
        print(f"   Ollama: {ollama_config['host']} with {ollama_config['models']}")
        
        concordance = SemanticConcordance(db_path, chroma_path, ollama_config)
        
        # Build semantic concordance
        print(f"\n⚙️  Building semantic concordance...")
        print(f"   Max semantic results: {max_semantic_results}")
        print(f"   Validate with Ollama: {validate_semantic}")
        
        start_time = time.time()
        matches = concordance.build_semantic_concordance(
            concept,
            max_semantic_results=max_semantic_results,
            validate_semantic=validate_semantic
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Concordance built in {elapsed_time:.1f} seconds")
        
        # Analyze results
        lexical_matches = [m for m in matches if not m.is_semantic_only]
        semantic_matches = [m for m in matches if m.is_semantic_only]
        
        print(f"\n📊 Results Summary:")
        print(f"   Total matches: {len(matches)}")
        print(f"   Lexical matches: {len(lexical_matches)}")
        print(f"   Semantic matches: {len(semantic_matches)}")
        
        # Lexical match breakdown
        if lexical_matches:
            print(f"\n📚 Lexical Match Types:")
            type_counts = defaultdict(int)
            for match in lexical_matches:
                type_counts[match.match_type] += 1
            for match_type, count in sorted(type_counts.items()):
                print(f"   • {match_type}: {count} matches")
        
        # Semantic validation breakdown
        if semantic_matches:
            print(f"\n🤖 Ollama Validation Results:")
            validation_counts = defaultdict(int)
            for match in semantic_matches:
                validation_counts[match.ollama_validation] += 1
            for validation, count in sorted(validation_counts.items()):
                print(f"   • {validation}: {count} matches")
            
            # Average scores
            avg_semantic_score = sum(m.semantic_score for m in semantic_matches) / len(semantic_matches)
            avg_ollama_confidence = sum(m.ollama_confidence for m in semantic_matches) / len(semantic_matches)
            print(f"\n   Average semantic score: {avg_semantic_score:.3f}")
            print(f"   Average Ollama confidence: {avg_ollama_confidence:.3f}")
        
        # Show top matches
        print(f"\n🔝 Top 10 Matches (by confidence):")
        for i, match in enumerate(matches[:10], 1):
            match_type = "Semantic" if match.is_semantic_only else "Lexical"
            print(f"\n{i}. {match.verse_id} ({match_type})")
            print(f"   Original: {match.original_text[:60]}...")
            print(f"   Confidence: {match.confidence:.3f}")
            print(f"   Evidence: {match.evidence}")
            if match.is_semantic_only:
                print(f"   Semantic Score: {match.semantic_score:.3f}")
                print(f"   Ollama: {match.ollama_validation} (confidence: {match.ollama_confidence:.2f})")
        
        # Generate report
        print(f"\n📄 Generating full report...")
        report = concordance.generate_semantic_report(concept, matches)
        
        # Save report
        report_path = Path(f"semantic_report_{concept_name}_{int(time.time())}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   Report saved to: {report_path}")
        
        # Performance metrics
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Total processing time: {elapsed_time:.1f} seconds")
        if semantic_matches:
            print(f"   Semantic validation time: ~{elapsed_time * len(semantic_matches) / len(matches):.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_concepts(concepts_path: Path):
    """List all available concepts in concepts.yaml."""
    with open(concepts_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    print("\n📚 Available Concepts:")
    print("=" * 80)
    
    for i, concept in enumerate(data['concepts'], 1):
        name = concept['name']
        description = concept.get('description', 'No description')[:60]
        strongs_count = len(concept.get('strongs_numbers', []))
        print(f"{i:2}. {name:<20} - {description}... ({strongs_count} Strong's)")


def main():
    """Run the semantic concept evaluation test."""
    parser = argparse.ArgumentParser(description='Test semantic concept evaluation')
    parser.add_argument('concept', nargs='?', help='Concept name to test')
    parser.add_argument('--list', action='store_true', help='List available concepts')
    parser.add_argument('--max-semantic', type=int, default=50, 
                       help='Maximum semantic results to evaluate (default: 50)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip Ollama validation (faster but less accurate)')
    
    args = parser.parse_args()
    
    # Load config for concepts path
    config = config_manager.load_config()
    concepts_path = config.concepts_path
    
    if args.list or not args.concept:
        list_available_concepts(concepts_path)
        if not args.concept:
            print("\nUsage: python test_semantic_concept_evaluation.py <concept_name>")
            print("Example: python test_semantic_concept_evaluation.py love")
        return 0
    
    # Test the concept
    success = test_semantic_concept(
        args.concept,
        max_semantic_results=args.max_semantic,
        validate_semantic=not args.no_validation
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())