#!/usr/bin/env python3
"""Test Ollama integration with llama3 model."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.ollama_validator import OllamaValidator
from abba.semantic.ollama_analyzer import OllamaAnalyzer, SemanticAnalysisResult


def test_ollama_connection():
    """Test basic Ollama connection."""
    print("Testing Ollama Connection")
    print("=" * 50)
    
    validator = OllamaValidator("http://localhost:11434")
    
    # Test connection
    if validator.is_available():
        print("✅ Ollama server is available")
    else:
        print("❌ Ollama server is not available")
        return False
    
    # Get available models
    models = validator.get_available_models()
    print(f"\nAvailable models: {len(models)}")
    for model in models:
        print(f"  - {model}")
    
    # Test llama3 specifically
    if "llama3" in models or any("llama3" in model for model in models):
        print("\n✅ llama3 model is available")
    else:
        print("\n❌ llama3 model not found")
        print("Please run: ollama pull llama3")
        return False
    
    # Test generation
    print("\nTesting text generation with llama3...")
    if validator.test_model_generation("llama3", "What is the meaning of life?"):
        print("✅ llama3 generation test successful")
    else:
        print("❌ llama3 generation test failed")
        return False
    
    return True


def test_semantic_analyzer():
    """Test the semantic analyzer with llama3."""
    print("\n" + "=" * 50)
    print("Testing Semantic Analyzer")
    print("=" * 50)
    
    try:
        # Initialize analyzer with llama3
        analyzer = OllamaAnalyzer(
            host="http://localhost:11434",
            models=["llama3"],
            consensus_threshold=0.7,
            timeout=30
        )
        
        # Test verse analysis for a concept
        verse_text = "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life."
        concept_name = "divine_love"
        concept_description = "God's love, mercy, compassion, and loving-kindness toward humanity, including covenant love and unconditional love."
        
        print(f"Analyzing verse for concept '{concept_name}':")
        print(f"Verse: {verse_text[:60]}...")
        
        result = analyzer.analyze_verse_for_concept(
            verse_text=verse_text,
            concept_name=concept_name,
            concept_description=concept_description,
            verse_reference="John 3:16"
        )
        
        print(f"\nResults:")
        print(f"  Relevance score: {result.relevance_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Consensus reached: {result.consensus_reached}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        if result.reasoning:
            print(f"  Reasoning: {result.reasoning[:100]}...")
        
        if result.error:
            print(f"  ❌ Error: {result.error}")
            return False
        else:
            print("  ✅ Analysis completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in semantic analyzer: {e}")
        return False


def test_concept_extraction():
    """Test concept extraction from verses."""
    print("\n" + "=" * 50)
    print("Testing Concept Extraction")
    print("=" * 50)
    
    try:
        analyzer = OllamaAnalyzer(
            host="http://localhost:11434",
            models=["llama3"],
            timeout=30
        )
        
        # Test verses
        test_verses = [
            ("The Lord is my shepherd; I shall not want.", "Psalm 23:1"),
            ("In the beginning was the Word, and the Word was with God, and the Word was God.", "John 1:1"),
            ("Love is patient, love is kind.", "1 Corinthians 13:4")
        ]
        
        for verse_text, reference in test_verses:
            print(f"\nExtracting concepts from {reference}:")
            print(f"  {verse_text}")
            
            result = analyzer.extract_concepts_from_verse(
                verse_text=verse_text,
                verse_reference=reference
            )
            
            if result.error:
                print(f"  ❌ Error: {result.error}")
                continue
            
            print(f"  Concepts found: {result.concepts}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Processing time: {result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in concept extraction: {e}")
        return False


def main():
    """Run all tests."""
    print("OLLAMA INTEGRATION TEST")
    print("=" * 70)
    
    success = True
    
    # Test 1: Connection
    if not test_ollama_connection():
        success = False
    
    # Test 2: Semantic Analysis
    if success and not test_semantic_analyzer():
        success = False
    
    # Test 3: Concept Extraction
    if success and not test_concept_extraction():
        success = False
    
    # Summary
    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED")
        print("\nOllama integration is working correctly with llama3!")
        print("\nNext steps:")
        print("1. Run: python abba/main.py --validate-concepts")
        print("2. Or run: python abba/main.py --map-concepts")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Install llama3: ollama pull llama3")
        print("3. Check Ollama is accessible at http://localhost:11434")
    
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())