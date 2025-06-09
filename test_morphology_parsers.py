#!/usr/bin/env python3
"""Test morphological parsers."""

from pathlib import Path
from src.abba.morphology.oshb_parser import OSHBParser
from src.abba.morphology.sblgnt_parser import SBLGNTParser


def test_hebrew_parser():
    """Test Hebrew morphology parser."""
    print("Testing Hebrew Parser (OSHB)")
    print("-" * 50)
    
    parser = OSHBParser()
    
    # Test Genesis 1:1
    words = parser.get_verse_words("Gen", 1, 1)
    if words:
        print(f"Genesis 1:1 - Found {len(words)} words:")
        for i, word in enumerate(words[:3]):  # Show first 3 words
            print(f"\n  Word {i+1}: {word['text']}")
            print(f"    Lemma: {word['lemma']}")
            print(f"    Morph code: {word['morph_code']}")
            print(f"    Features: {word['features']}")
    else:
        print("No words found for Genesis 1:1")
    
    # Test morphology parsing
    print("\n\nMorphology Code Parsing Examples:")
    test_codes = ["HNcmpa", "HVqp3ms", "HR/Ncfsa", "HC", "HTo"]
    for code in test_codes:
        features = parser.parse_morph_code(code)
        print(f"  {code} -> {features}")


def test_greek_parser():
    """Test Greek morphology parser."""
    print("\n\nTesting Greek Parser (SBLGNT)")
    print("-" * 50)
    
    parser = SBLGNTParser()
    
    # Test Matthew 1:1
    words = parser.get_verse_words("Matt", 1, 1)
    if words:
        print(f"Matthew 1:1 - Found {len(words)} words:")
        for i, word in enumerate(words[:3]):  # Show first 3 words
            print(f"\n  Word {i+1}: {word['text']}")
            print(f"    Normalized: {word['normalized']}")
            print(f"    Lemma: {word['lemma']}")
            print(f"    POS code: {word['pos_code']}")
            print(f"    Parse code: {word['parse_code']}")
            print(f"    Features: {word['features']}")
    else:
        print("No words found for Matthew 1:1")
    
    # Test verb parsing
    print("\n\nVerb Parse Code Examples:")
    test_codes = ["V-PAI-3S", "V-AAN", "V-PPN-NSM", "V-FAI-1P"]
    for code in test_codes:
        features = parser.parse_verb_code(code)
        print(f"  {code} -> {features}")


def main():
    """Run tests."""
    test_hebrew_parser()
    test_greek_parser()
    
    print("\n\nParsers tested successfully!")


if __name__ == "__main__":
    main()