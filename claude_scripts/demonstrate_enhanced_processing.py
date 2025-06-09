#!/usr/bin/env python3
"""
Demonstrate enhanced Bible processing with real data integration.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abba.cli_enhanced import EnhancedBibleProcessor
from abba.verse_id import VerseID
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demonstrate_verse_processing():
    """Demonstrate processing of specific verses with all enrichments."""
    # Initialize processor
    data_dir = Path(__file__).parent.parent / "data"
    processor = EnhancedBibleProcessor(data_dir)
    
    print("\n=== Enhanced Bible Processing Demonstration ===\n")
    
    # Process just a few books for demonstration
    verses = processor.process_bible(
        translation_codes=['eng_web', 'eng_kjv'],
        limit_books=['Gen', 'Mat', 'Jhn']  # Genesis, Matthew, John
    )
    
    if not verses:
        print("No verses processed. Check data directory.")
        return
    
    # Show some interesting verses
    demo_verses = [
        'Gen.1.1',  # Creation
        'Mat.1.1',  # Genealogy of Jesus
        'Jhn.1.1',  # In the beginning was the Word
    ]
    
    for verse_key in demo_verses:
        if verse_key in verses:
            enriched = verses[verse_key]
            print(f"\n{'='*60}")
            print(f"Verse: {verse_key}")
            print(f"Book: {enriched.metadata.get('book_name')}")
            print(f"Testament: {enriched.metadata.get('testament')}")
            print(f"{'='*60}")
            
            # Show translations
            print("\nTranslations:")
            for trans_code, text in enriched.translations.items():
                print(f"  {trans_code}: {text[:100]}...")
            
            # Show Hebrew data if available
            if enriched.hebrew_data:
                print(f"\nHebrew Text ({len(enriched.hebrew_data)} words):")
                for i, word in enumerate(enriched.hebrew_data[:5]):  # First 5 words
                    print(f"  {i+1}. {word.text}")
                    if word.transliteration:
                        print(f"     Transliteration: {word.transliteration}")
                    if word.lemma:
                        print(f"     Lemma: {word.lemma}")
                    if word.morph_code:
                        print(f"     Morphology: {word.morph_code}")
                    if word.lexicon_entry:
                        gloss = word.lexicon_entry.get('kjv_def', '')[:50]
                        if gloss:
                            print(f"     Gloss: {gloss}...")
            
            # Show Greek data if available
            if enriched.greek_data:
                print(f"\nGreek Text ({len(enriched.greek_data)} words):")
                for i, word in enumerate(enriched.greek_data[:5]):  # First 5 words
                    print(f"  {i+1}. {word.text}")
                    if word.transliteration:
                        print(f"     Transliteration: {word.transliteration}")
            
            # Show annotations
            if enriched.annotations:
                print("\nAnnotations:")
                for ann in enriched.annotations:
                    print(f"  - {ann['type']}: {ann['value']} (confidence: {ann['confidence']:.2f})")
            
            # Show cross-references
            if enriched.cross_references:
                print("\nCross-references:")
                for xref in enriched.cross_references:
                    print(f"  - {xref['target']}: {xref['theological_theme']}")
            
            # Show timeline events
            if enriched.timeline_events:
                print("\nTimeline Events:")
                for event in enriched.timeline_events:
                    print(f"  - {event['name']}: {event['description']}")

def demonstrate_interlinear():
    """Demonstrate interlinear display generation."""
    data_dir = Path(__file__).parent.parent / "data"
    processor = EnhancedBibleProcessor(data_dir)
    
    print("\n\n=== Interlinear Display Demonstration ===\n")
    
    # Process Genesis 1:1
    verses = processor.process_bible(
        translation_codes=['eng_web'],
        limit_books=['Gen']
    )
    
    verse_key = 'Gen.1.1'
    if verse_key in verses:
        enriched = verses[verse_key]
        
        if enriched.interlinear_data:
            print(f"Interlinear for {verse_key}:")
            print("-" * 80)
            
            tokens = enriched.interlinear_data['tokens']
            
            # Print Hebrew text
            hebrew_line = " ".join(t['text'] for t in tokens)
            print(f"Hebrew:  {hebrew_line}")
            
            # Print transliteration
            translit_line = " ".join(t.get('transliteration', '???') for t in tokens)
            print(f"Translit: {translit_line}")
            
            # Print glosses
            gloss_line = " ".join(t.get('gloss', '???')[:15] for t in tokens)
            print(f"Gloss:   {gloss_line}")
            
            # Print English
            english = enriched.translations.get('ENG_WEB', '')
            print(f"\nEnglish: {english}")

def export_sample_data():
    """Export sample enriched data for inspection."""
    data_dir = Path(__file__).parent.parent / "data"
    processor = EnhancedBibleProcessor(data_dir)
    
    print("\n\n=== Exporting Sample Data ===\n")
    
    # Process a small sample
    verses = processor.process_bible(
        translation_codes=['eng_web'],
        limit_books=['Gen']  # Just Genesis
    )
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "output" / "enhanced_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export first chapter of Genesis
    genesis_1 = {}
    for verse_key, enriched in verses.items():
        if verse_key.startswith('Gen.1.'):
            genesis_1[verse_key] = {
                'text': enriched.translations.get('ENG_WEB', ''),
                'hebrew': [
                    {
                        'text': w.text,
                        'transliteration': w.transliteration,
                        'lemma': w.lemma,
                        'morphology': w.morph_analysis,
                        'gloss': w.lexicon_entry.get('kjv_def', '') if w.lexicon_entry else ''
                    }
                    for w in (enriched.hebrew_data or [])
                ],
                'annotations': enriched.annotations,
                'metadata': enriched.metadata
            }
    
    # Write to file
    output_file = output_dir / "genesis_1_enriched.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(genesis_1, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(genesis_1)} verses to {output_file}")
    
    # Also create a summary
    summary = {
        'total_verses': len(verses),
        'verses_with_hebrew': sum(1 for v in verses.values() if v.hebrew_data),
        'verses_with_annotations': sum(1 for v in verses.values() if v.annotations),
        'sample_verse': verse_key if verse_key in verses else None
    }
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_verse_processing()
    demonstrate_interlinear()
    export_sample_data()
    
    print("\n\nDemonstration complete!")