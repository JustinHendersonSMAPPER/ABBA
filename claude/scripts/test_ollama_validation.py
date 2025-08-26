#!/usr/bin/env python3
"""
Test Ollama validation to see why it's filtering everything out
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.semantic.ollama_analyzer import OllamaAnalyzer
from abba.logging_setup import logger

def test_ollama_validation():
    """Test Ollama validation directly."""
    
    # Load configuration
    config = config_manager.load_config()
    
    # Ollama configuration
    ollama_config = {
        'host': config.ollama_host,
        'models': config.ollama_semantic_models,
        'consensus_threshold': config.ollama_consensus_threshold,
        'timeout': config.ollama_timeout
    }
    
    logger.info(f"Testing Ollama validation...")
    logger.info(f"Models: {ollama_config['models']}")
    
    # Initialize Ollama analyzer
    analyzer = OllamaAnalyzer(
        host=ollama_config['host'],
        models=ollama_config['models'],
        consensus_threshold=ollama_config['consensus_threshold'],
        timeout=ollama_config['timeout']
    )
    
    # Test with a simple example that should definitely match
    test_cases = [
        {
            'concept': 'love',
            'description': 'Divine and human love, compassion, and affection',
            'verse_ref': 'John 3:16',
            'original_text': 'ἠγάπησεν',  # Greek: "loved"
            'strongs': ['G25'],  # agapao - to love
            'expected': 'YES'
        },
        {
            'concept': 'faith',
            'description': 'Trust and belief in God',
            'verse_ref': 'Hebrews 11:1',
            'original_text': 'πίστις',  # Greek: "faith"
            'strongs': ['G4102'],  # pistis - faith
            'expected': 'YES'
        },
        {
            'concept': 'sin',
            'description': 'Transgression against God',
            'verse_ref': 'Romans 3:23',
            'original_text': 'ἥμαρτον',  # Greek: "sinned"
            'strongs': ['G264'],  # hamartano - to sin
            'expected': 'YES'
        }
    ]
    
    for test in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {test['concept']} - {test['verse_ref']}")
        logger.info(f"Expected: {test['expected']}")
        
        prompt = f"""You are a biblical scholar validating semantic matches.

Biblical Concept: {test['concept']}
Description: {test['description']}
Verse Reference: {test['verse_ref']}
Original Text: {test['original_text']}
Strong's Numbers in verse: {', '.join(test['strongs'])}

Question: Does this verse genuinely relate to the concept of "{test['concept']}"?

Analyze the original language text and its grammatical forms. Consider whether this verse:
1. Directly expresses the concept
2. Contains related theological themes
3. Uses the concept metaphorically
4. Or is a false positive with no real connection

Respond in this exact format:
ANSWER: [YES/MAYBE/NO]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [One sentence explanation]"""
        
        try:
            # Test with each model
            for model in ollama_config['models']:
                logger.info(f"\nModel: {model}")
                response = analyzer.generate_completion(prompt, model=model)
                logger.info(f"Response:\n{response}")
                
                # Try to parse the response
                lines = response.strip().split('\n')
                answer = None
                confidence = 0.0
                explanation = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('ANSWER:'):
                        answer = line.split(':', 1)[1].strip()
                    elif line.startswith('CONFIDENCE:'):
                        try:
                            confidence = float(line.split(':', 1)[1].strip())
                        except:
                            confidence = 0.0
                    elif line.startswith('EXPLANATION:'):
                        explanation = line.split(':', 1)[1].strip()
                
                logger.info(f"\nParsed:")
                logger.info(f"  Answer: {answer}")
                logger.info(f"  Confidence: {confidence}")
                logger.info(f"  Explanation: {explanation}")
                
                if answer != test['expected']:
                    logger.warning(f"  ⚠️ Unexpected answer! Expected {test['expected']}, got {answer}")
                else:
                    logger.info(f"  ✅ Correct answer!")
                    
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    return True

if __name__ == "__main__":
    success = test_ollama_validation()
    sys.exit(0 if success else 1)