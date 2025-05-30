#!/usr/bin/env python3
"""
Demonstration of the enhanced alignment system for Phase 2.2 and 2.3.

This script shows how the multi-layered alignment approach would work
to provide complete word-level alignment with semantic loss detection.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.parsers.hebrew_parser import HebrewParser
from src.abba.parsers.greek_parser import GreekParser
from src.abba.parsers.translation_parser import TranslationParser
from src.abba.alignment.statistical_aligner import AlignmentPipeline, EnhancedAlignment


def demonstrate_alignment_pipeline(data_dir: Path):
    """Demonstrate the complete alignment pipeline."""
    
    print("=" * 80)
    print("ABBA Enhanced Alignment System Demonstration")
    print("=" * 80)
    
    # Initialize parsers
    hebrew_parser = HebrewParser()
    greek_parser = GreekParser()
    translation_parser = TranslationParser()
    
    # Parse sample data
    print("\n1. Loading sample data...")
    
    # Load Hebrew Genesis 1:1
    hebrew_verses = []
    gen_file = data_dir / "sources" / "hebrew" / "Gen.xml"
    if gen_file.exists():
        hebrew_verses = hebrew_parser.parse_file(gen_file)[:1]  # Just first verse
        print(f"   Loaded {len(hebrew_verses)} Hebrew verses")
    
    # Load Greek John 1:1
    greek_verses = []
    john_file = data_dir / "sources" / "greek" / "JOH.xml"
    if john_file.exists():
        greek_verses = greek_parser.parse_file(john_file, "JHN")[:1]  # Just first verse
        print(f"   Loaded {len(greek_verses)} Greek verses")
    
    # Load sample translations (simulated since we don't have aligned data yet)
    translation_verses = create_sample_translations()
    print(f"   Created {len(translation_verses)} sample translation verses")
    
    # Initialize alignment pipeline
    print("\n2. Building alignment model...")
    pipeline = AlignmentPipeline()
    
    # Process corpus
    print("   Training statistical models...")
    all_alignments = pipeline.process_corpus(hebrew_verses, greek_verses, translation_verses)
    
    print(f"   Generated alignments for {len(all_alignments)} verses")
    
    # Demonstrate results
    print("\n3. Alignment Results:")
    demonstrate_alignment_results(all_alignments)
    
    # Demonstrate search capabilities
    print("\n4. Cross-Language Search Capabilities:")
    demonstrate_search_capabilities(pipeline)
    
    # Show semantic loss detection
    print("\n5. Semantic Loss Detection:")
    demonstrate_semantic_loss_detection(all_alignments)
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)


def create_sample_translations():
    """Create sample translation data for demonstration."""
    from src.abba.verse_id import create_verse_id
    from src.abba.parsers.translation_parser import TranslationVerse
    
    # Sample translations for Genesis 1:1
    gen_translations = [
        ("In the beginning God created the heavens and the earth.", "KJV"),
        ("In the beginning God created the heaven and the earth.", "ASV"),
        ("In the beginning, God created the heavens and the earth.", "NIV"),
    ]
    
    # Sample translations for John 1:1
    john_translations = [
        ("In the beginning was the Word, and the Word was with God, and the Word was God.", "KJV"),
        ("In the beginning was the Word, and the Word was with God, and the Word was God.", "ASV"),
        ("In the beginning was the Word, and the Word was with God, and the Word was God.", "NIV"),
    ]
    
    verses = []
    
    # Create Genesis verses
    gen_id = create_verse_id("GEN", 1, 1)
    for text, version in gen_translations:
        verse = TranslationVerse(
            verse_id=gen_id,
            text=text,
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        verses.append(verse)
    
    # Create John verses
    john_id = create_verse_id("JHN", 1, 1)
    for text, version in john_translations:
        verse = TranslationVerse(
            verse_id=john_id,
            text=text,
            original_book_name="John",
            original_chapter=1,
            original_verse=1
        )
        verses.append(verse)
    
    return verses


def demonstrate_alignment_results(all_alignments: Dict[str, List[EnhancedAlignment]]):
    """Show detailed alignment results."""
    
    for verse_id, alignments in all_alignments.items():
        print(f"\n   Verse: {verse_id}")
        print("   " + "-" * 50)
        
        for i, alignment in enumerate(alignments[:5]):  # Show first 5 alignments
            source_text = " ".join(token.text for token in alignment.source_tokens)
            target_text = " ".join(alignment.target_words)
            
            print(f"   [{i+1}] {source_text} → {target_text}")
            print(f"       Confidence: {alignment.confidence.value} ({alignment.confidence_score:.2f})")
            print(f"       Method: {alignment.alignment_method}")
            
            if alignment.strong_numbers:
                print(f"       Strong's: {', '.join(alignment.strong_numbers)}")
            
            if alignment.semantic_losses:
                loss = alignment.semantic_losses[0]
                print(f"       ⚠️  Semantic Loss: {loss.description}")
            
            if alignment.alternative_translations:
                alts = ', '.join(alignment.alternative_translations[:3])
                print(f"       Alternatives: {alts}")
            
            print()


def demonstrate_search_capabilities(pipeline: AlignmentPipeline):
    """Demonstrate cross-language search capabilities."""
    
    search_tests = [
        ("beginning", "english"),
        ("God", "english"),
        ("H430", "strongs"),  # Elohim
        ("G3056", "strongs"),  # Logos
    ]
    
    for query, search_type in search_tests:
        results = pipeline.search_cross_language(query, search_type)
        print(f"   Search '{query}' ({search_type}): {len(results)} verses found")
        if results:
            print(f"     Verses: {', '.join(results)}")


def demonstrate_semantic_loss_detection(all_alignments: Dict[str, List[EnhancedAlignment]]):
    """Show semantic loss detection in action."""
    
    total_losses = 0
    loss_types = {}
    
    for verse_id, alignments in all_alignments.items():
        for alignment in alignments:
            for loss in alignment.semantic_losses:
                total_losses += 1
                loss_type = loss.loss_type.value
                if loss_type not in loss_types:
                    loss_types[loss_type] = 0
                loss_types[loss_type] += 1
                
                print(f"   🔍 {verse_id}: {loss.original_concept} → {loss.translation_concept}")
                print(f"      Loss: {loss.description}")
                print(f"      Severity: {loss.severity:.1f}/1.0")
                print(f"      Explanation: {loss.explanation}")
                print()
    
    print(f"   Total semantic losses detected: {total_losses}")
    print(f"   Loss types: {loss_types}")


def generate_ui_highlighting_data(all_alignments: Dict[str, List[EnhancedAlignment]]) -> Dict:
    """
    Generate data structure for UI highlighting functionality.
    
    This shows how the alignment data would be used to support
    the hover functionality the user requested.
    """
    
    ui_data = {}
    
    for verse_id, alignments in all_alignments.items():
        verse_ui_data = {
            "verse_id": verse_id,
            "word_highlights": [],
            "hover_data": {}
        }
        
        for alignment in alignments:
            # Mark words for highlighting based on semantic loss
            if alignment.semantic_losses:
                for word in alignment.target_words:
                    word_data = {
                        "word": word,
                        "highlight_type": "semantic_loss",
                        "severity": max(loss.severity for loss in alignment.semantic_losses),
                        "tooltip_id": f"{verse_id}_{word}"
                    }
                    verse_ui_data["word_highlights"].append(word_data)
                    
                    # Create hover tooltip data
                    tooltip_data = {
                        "original_text": " ".join(token.text for token in alignment.source_tokens),
                        "translation": word,
                        "strong_numbers": alignment.strong_numbers,
                        "confidence": alignment.confidence_score,
                        "semantic_losses": [
                            {
                                "type": loss.loss_type.value,
                                "description": loss.description,
                                "explanation": loss.explanation,
                                "severity": loss.severity
                            }
                            for loss in alignment.semantic_losses
                        ],
                        "alternatives": alignment.alternative_translations,
                        "morphology": alignment.morphological_notes
                    }
                    verse_ui_data["hover_data"][f"{verse_id}_{word}"] = tooltip_data
        
        ui_data[verse_id] = verse_ui_data
    
    return ui_data


def main():
    """Main demonstration function."""
    data_dir = Path(__file__).parent.parent / "data"
    
    if not data_dir.exists():
        print("Data directory not found. Please ensure data is available.")
        return
    
    # Run demonstration
    demonstrate_alignment_pipeline(data_dir)
    
    # Show how UI data would be generated
    print("\n6. UI Highlighting Data Structure:")
    print("   This shows how alignment data supports hover functionality...")
    
    # This would be called with real alignment data
    sample_ui_data = {
        "GEN.1.1": {
            "word_highlights": [
                {
                    "word": "God",
                    "highlight_type": "semantic_loss",
                    "severity": 0.7,
                    "tooltip_id": "GEN.1.1_God"
                }
            ],
            "hover_data": {
                "GEN.1.1_God": {
                    "original_text": "אֱלֹהִים",
                    "translation": "God",
                    "strong_numbers": ["H430"],
                    "confidence": 0.95,
                    "semantic_losses": [
                        {
                            "type": "lexical_richness",
                            "description": "Hebrew 'Elohim' is plural form suggesting majesty",
                            "explanation": "The plural form implies the fullness and majesty of God",
                            "severity": 0.4
                        }
                    ],
                    "alternatives": ["God", "gods", "divine beings"],
                    "morphology": ["masculine plural noun"]
                }
            }
        }
    }
    
    print("   Sample UI data structure:")
    print(json.dumps(sample_ui_data, indent=4))


if __name__ == "__main__":
    main()