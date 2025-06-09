#!/usr/bin/env python3
"""
Demonstrate word alignment functionality.

This script shows how word-level alignments work between
Hebrew/Greek and English translations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.cli_with_alignment import AlignedBibleProcessor, AlignedBibleExporter


def format_alignment_display(verse_data: Dict) -> str:
    """Format alignment data for display."""
    output = []
    
    # Display verse reference and translation
    output.append(f"\n{'='*80}")
    output.append(f"Verse: {verse_data['verse_id']}")
    output.append(f"{'='*80}")
    
    # Display English text
    eng_text = verse_data['translations'].get('eng_kjv', '')
    output.append(f"\nEnglish (KJV):")
    output.append(f"  {eng_text}")
    
    # Display original language text
    if verse_data.get('hebrew_text'):
        output.append(f"\nHebrew Text:")
        output.append(f"  {verse_data['hebrew_text']}")
        output.append(f"\nHebrew Words:")
        for i, word in enumerate(verse_data['hebrew_words']):
            output.append(f"  [{i}] {word['text']} - {word.get('transliteration', '')} "
                         f"({word.get('lemma', '')}) [{word.get('morph', '')}]")
                         
    elif verse_data.get('greek_text'):
        output.append(f"\nGreek Text:")
        output.append(f"  {verse_data['greek_text']}")
        output.append(f"\nGreek Words:")
        for i, word in enumerate(verse_data['greek_words']):
            output.append(f"  [{i}] {word['text']} - {word.get('transliteration', '')} "
                         f"({word.get('lemma', '')}) [{word.get('morph', '')}]")
    
    # Display alignments
    if 'alignments' in verse_data and 'eng_kjv' in verse_data['alignments']:
        output.append(f"\nWord Alignments (confidence scores):")
        output.append("-" * 60)
        
        for alignment in verse_data['alignments']['eng_kjv']:
            source_word = alignment['source_word']
            target_phrase = alignment['target_phrase']
            confidence = alignment['confidence']
            source_idx = alignment['source_idx']
            
            # Get morphology info
            morph_info = ""
            if verse_data.get('hebrew_words'):
                word_data = verse_data['hebrew_words'][source_idx]
                if 'morphology' in word_data:
                    morph = word_data['morphology']
                    morph_parts = []
                    if 'part_of_speech' in morph:
                        morph_parts.append(morph['part_of_speech'])
                    if 'gender' in morph:
                        morph_parts.append(morph['gender'])
                    if 'number' in morph:
                        morph_parts.append(morph['number'])
                    morph_info = f" [{', '.join(morph_parts)}]"
                        
            output.append(f"  {source_word}{morph_info} → {target_phrase} ({confidence:.3f})")
    
    # Display cross-references if any
    if verse_data.get('cross_references'):
        output.append(f"\nCross References:")
        for xref in verse_data['cross_references']:
            output.append(f"  → {xref['target']} ({xref['type']}, {xref['confidence']:.2f})")
            
    # Display timeline events if any
    if verse_data.get('timeline_events'):
        output.append(f"\nTimeline Events:")
        for event in verse_data['timeline_events']:
            output.append(f"  • {event['name']} ({event['date']}, confidence: {event['confidence']})")
    
    return '\n'.join(output)


def demonstrate_sample_verses():
    """Demonstrate alignment on a few sample verses."""
    # Process just a few key verses
    processor = AlignedBibleProcessor()
    
    # Process Genesis 1:1 and John 1:1
    verses = processor.process_bible(
        translation_codes=["ENG_KJV"],
        book_filter=["Gen", "John", "Matt"]
    )
    
    # Key verses to demonstrate
    demo_verses = [
        "Gen.1.1",    # Creation - Hebrew
        "Gen.1.2",    # Spirit of God - Hebrew
        "John.1.1",   # In the beginning was the Word - Greek
        "John.1.14",  # Word became flesh - Greek
        "Matt.5.3",   # Blessed are the poor in spirit - Greek
    ]
    
    # Display each verse
    for verse_id in demo_verses:
        if verse_id in verses:
            verse = verses[verse_id]
            
            # Convert to dict format for display
            verse_data = {
                'verse_id': verse_id,
                'translations': verse.translations,
                'hebrew_text': verse.hebrew_text,
                'hebrew_words': verse.hebrew_words,
                'greek_text': verse.greek_text,
                'greek_words': verse.greek_words,
                'alignments': verse.alignments,
                'cross_references': verse.cross_references,
                'timeline_events': verse.timeline_events
            }
            
            print(format_alignment_display(verse_data))
        else:
            print(f"\nVerse {verse_id} not found in processed data")


def demonstrate_from_export(export_dir: Path):
    """Demonstrate alignment from exported JSON files."""
    # Load a few sample verses from export
    sample_files = [
        (export_dir / "Gen.json", ["1.1", "1.2", "1.3"]),
        (export_dir / "Matt.json", ["1.1", "5.3", "22.37"]),
        (export_dir / "John.json", ["1.1", "1.14", "3.16"])
    ]
    
    for file_path, verse_refs in sample_files:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
                
            for chapter in book_data['chapters']:
                for verse in chapter['verses']:
                    verse_ref = f"{chapter['chapter']}.{verse['verse']}"
                    if verse_ref in verse_refs:
                        print(format_alignment_display(verse))


def main():
    """Main demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate word alignment functionality"
    )
    
    parser.add_argument(
        '--export-dir',
        type=Path,
        help='Use existing export directory instead of processing'
    )
    
    parser.add_argument(
        '--process-sample',
        action='store_true',
        help='Process sample verses (requires alignment models)'
    )
    
    args = parser.parse_args()
    
    if args.export_dir and args.export_dir.exists():
        print("Demonstrating alignment from exported data...")
        demonstrate_from_export(args.export_dir)
    elif args.process_sample:
        print("Processing sample verses with alignment...")
        demonstrate_sample_verses()
    else:
        print("Please specify either --export-dir or --process-sample")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())