#!/usr/bin/env python3
"""
Test enhanced Bible processing without ML components.
"""

import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abba.verse_id import VerseID, parse_verse_id
from abba.cross_references.loader import CrossReferenceLoader
from abba.language.transliteration import create_transliterator
from abba.morphology.hebrew_morphology import HebrewMorphologyParser
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_cross_references():
    """Test loading and using cross-references."""
    print("\n=== Testing Cross-Reference Loading ===\n")
    
    data_dir = Path(__file__).parent.parent / "data"
    loader = CrossReferenceLoader(data_dir)
    
    # Load cross-references
    refs = loader.load_from_json()
    print(f"Loaded {len(refs)} cross-references")
    
    # Get statistics
    stats = loader.get_reference_statistics()
    print("\nCross-reference statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test finding references for Genesis 1:1
    verse_id = parse_verse_id("GEN.1.1")
    if verse_id:
        refs_from = loader.find_references_for_verse(verse_id)
        print(f"\nReferences from Genesis 1:1: {len(refs_from)}")
        for ref in refs_from:
            print(f"  → {ref.target_verse}: {ref.theological_theme}")


def test_hebrew_parsing():
    """Test parsing Hebrew XML files."""
    print("\n\n=== Testing Hebrew XML Parsing ===\n")
    
    data_dir = Path(__file__).parent.parent / "data" / "sources"
    hebrew_file = data_dir / "hebrew" / "Gen.xml"
    
    if not hebrew_file.exists():
        print(f"Hebrew file not found: {hebrew_file}")
        return
    
    # Parse the file
    transliterator = create_transliterator('hebrew')
    hebrew_morph = HebrewMorphologyParser()
    
    try:
        tree = ET.parse(hebrew_file)
        root = tree.getroot()
        
        # Define namespaces
        namespaces = {
            'osis': 'http://www.bibletechnologies.net/2003/OSIS/namespace'
        }
        
        # Find first verse
        verse_elem = root.find('.//osis:verse[@osisID="Gen.1.1"]', namespaces)
        if verse_elem is not None:
            print("Genesis 1:1 Hebrew text:")
            print("-" * 60)
            
            words = []
            for w_elem in verse_elem.findall('.//osis:w', namespaces):
                text = ''.join(w_elem.itertext()).strip()
                lemma = w_elem.get('lemma', '')
                morph = w_elem.get('morph', '')
                
                if text:
                    translit = transliterator.transliterate(text, 'hebrew')
                    words.append({
                        'text': text,
                        'transliteration': translit,
                        'lemma': lemma,
                        'morph': morph
                    })
            
            # Display words
            for i, word in enumerate(words):
                print(f"\nWord {i+1}:")
                print(f"  Hebrew: {word['text']}")
                print(f"  Translit: {word['transliteration']}")
                print(f"  Lemma: {word['lemma']}")
                print(f"  Morph: {word['morph']}")
                
                # Parse morphology
                if word['morph']:
                    try:
                        morph_analysis = hebrew_morph.parse(word['morph'])
                        print(f"  Analysis: {morph_analysis}")
                    except Exception as e:
                        print(f"  Could not parse morphology: {e}")
            
    except Exception as e:
        logger.error(f"Error parsing Hebrew XML: {e}")


def test_translation_loading():
    """Test loading translation files."""
    print("\n\n=== Testing Translation Loading ===\n")
    
    trans_dir = Path(__file__).parent.parent / "data" / "sources" / "translations"
    
    # Find English translations
    eng_translations = list(trans_dir.glob("eng_*.json"))
    print(f"Found {len(eng_translations)} English translations")
    
    # Load first translation
    if eng_translations:
        trans_file = eng_translations[0]
        print(f"\nLoading: {trans_file.name}")
        
        with open(trans_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Version: {data.get('version')}")
        print(f"Name: {data.get('name')}")
        print(f"Language: {data.get('language')}")
        
        # Get Genesis 1:1
        if 'books' in data and 'Gen' in data['books']:
            gen = data['books']['Gen']
            if 'chapters' in gen and len(gen['chapters']) > 0:
                chapter1 = gen['chapters'][0]
                if 'verses' in chapter1 and len(chapter1['verses']) > 0:
                    verse1 = chapter1['verses'][0]
                    print(f"\nGenesis 1:1: {verse1['text']}")


def test_lexicon_parsing():
    """Test parsing Strong's lexicon."""
    print("\n\n=== Testing Lexicon Parsing ===\n")
    
    lexicon_file = Path(__file__).parent.parent / "data" / "sources" / "lexicons" / "strongs_hebrew.xml"
    
    if not lexicon_file.exists():
        print(f"Lexicon file not found: {lexicon_file}")
        return
    
    try:
        tree = ET.parse(lexicon_file)
        root = tree.getroot()
        
        # Find first few entries
        entries = root.findall('.//entry')[:5]
        print(f"Total entries found: {len(root.findall('.//entry'))}")
        print(f"Showing first {len(entries)} lexicon entries:")
        
        for entry in entries:
            strongs = entry.get('strongs')
            if strongs:
                print(f"\nStrong's {strongs}:")
                
                # Get definition
                strongs_def = entry.find('.//strongs_def')
                if strongs_def is not None:
                    definition = ''.join(strongs_def.itertext()).strip()
                    print(f"  Definition: {definition[:100]}...")
                
                # Get KJV usage
                kjv_def = entry.find('.//kjv_def')
                if kjv_def is not None:
                    kjv = ''.join(kjv_def.itertext()).strip()
                    print(f"  KJV usage: {kjv[:100]}...")
                    
    except Exception as e:
        logger.error(f"Error parsing lexicon: {e}")


def main():
    """Run all tests."""
    print("=== Enhanced Bible Processing Tests ===")
    print("Testing individual components without ML...")
    
    test_cross_references()
    test_hebrew_parsing()
    test_translation_loading()
    test_lexicon_parsing()
    
    print("\n\nAll tests completed!")


if __name__ == "__main__":
    main()