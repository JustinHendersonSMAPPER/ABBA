"""
Enhanced CLI for ABBA Bible processing with real data integration.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime

from abba.book_codes import normalize_book_name, get_book_name, is_valid_book_code
from abba.canon.registry import CanonRegistry
from abba.cross_references.models import CrossReference, ReferenceType
from abba.cross_references.parser import CrossReferenceParser
from abba.cross_references.loader import CrossReferenceLoader
from abba.export.minimal_sqlite import MinimalSQLiteExporter
from abba.export.minimal_json import MinimalJSONExporter
from abba.language.transliteration import create_transliterator
from abba.morphology.hebrew_morphology import HebrewMorphology, HebrewMorphologyParser
from abba.morphology.greek_morphology import GreekMorphology, GreekMorphologyParser
from abba.parsers.hebrew_parser import HebrewParser
from abba.parsers.greek_parser import GreekParser
from abba.parsers.lexicon_parser import LexiconParser
from abba.timeline.models import Event, TimePoint, EventType, create_bce_date
from abba.verse_id import VerseID, parse_verse_id
from abba.annotations.annotation_engine import AnnotationEngine
from abba.annotations.models import Annotation, AnnotationType
from abba.interlinear.token_extractor import TokenExtractor
from abba.interlinear.lexicon_integration import LexiconIntegrator

logger = logging.getLogger(__name__)


@dataclass
class OriginalLanguageWord:
    """Represents a word in the original language with morphology."""
    text: str
    transliteration: Optional[str] = None
    lemma: Optional[str] = None
    morph_code: Optional[str] = None
    morph_analysis: Optional[Dict] = None
    strong_number: Optional[str] = None
    lexicon_entry: Optional[Dict] = None
    position: int = 0


@dataclass
class EnrichedVerse:
    """Verse with all enrichment data."""
    verse_id: VerseID
    translations: Dict[str, str] = field(default_factory=dict)
    hebrew_data: Optional[List[OriginalLanguageWord]] = None
    greek_data: Optional[List[OriginalLanguageWord]] = None
    morphology: Optional[Dict] = None
    cross_references: List[Dict] = field(default_factory=list)
    annotations: List[Dict] = field(default_factory=list)
    timeline_events: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    interlinear_data: Optional[Dict] = None


class EnhancedBibleProcessor:
    """Process Bible data with real data integration."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.translations_dir = data_dir / "translations"
        self.hebrew_dir = data_dir / "hebrew"
        self.greek_dir = data_dir / "greek"
        self.lexicons_dir = data_dir / "lexicons"
        
        # Initialize components
        self.canon_registry = CanonRegistry()
        self.hebrew_morph = HebrewMorphologyParser()
        self.greek_morph = GreekMorphologyParser()
        self.hebrew_parser = HebrewParser()
        self.greek_parser = GreekParser()
        self.lexicon_parser = LexiconParser()
        # Try to initialize annotation engine, but make it optional
        try:
            self.annotation_engine = AnnotationEngine()
        except Exception as e:
            logger.warning(f"Could not initialize annotation engine: {e}")
            self.annotation_engine = None
        self.token_extractor = TokenExtractor()
        self.transliterator = create_transliterator()
        
        # Initialize lexicon integration
        self.lexicon_integration = LexiconIntegrator(str(self.lexicons_dir))
        
        # Initialize cross-reference loader
        self.cross_ref_loader = CrossReferenceLoader(data_dir)
        
        # Cache for processed data
        self.verses_cache: Dict[str, EnrichedVerse] = {}
        self.cross_refs_cache: List[CrossReference] = []
        self.timeline_cache: List[Event] = []
        self.lexicon_cache: Dict[str, Dict] = {}
        
        # Load lexicons
        self._load_lexicons()
        
        # Load cross-references
        self._load_cross_references()
    
    def _load_lexicons(self):
        """Load Strong's Hebrew and Greek lexicons."""
        hebrew_lexicon_path = self.lexicons_dir / "strongs_hebrew.xml"
        greek_lexicon_path = self.lexicons_dir / "strongs_greek.xml"
        
        if hebrew_lexicon_path.exists():
            logger.info("Loading Hebrew lexicon...")
            self.lexicon_cache['hebrew'] = self._parse_strongs_lexicon(hebrew_lexicon_path)
            logger.info(f"Loaded {len(self.lexicon_cache['hebrew'])} Hebrew entries")
        
        if greek_lexicon_path.exists():
            logger.info("Loading Greek lexicon...")
            self.lexicon_cache['greek'] = self._parse_strongs_lexicon(greek_lexicon_path)
            logger.info(f"Loaded {len(self.lexicon_cache['greek'])} Greek entries")
    
    def _load_cross_references(self):
        """Load cross-references from available sources."""
        # Try to load from JSON file
        json_path = self.data_dir / "cross_references.json"
        if json_path.exists():
            self.cross_refs_cache = self.cross_ref_loader.load_from_json(json_path)
            logger.info(f"Loaded {len(self.cross_refs_cache)} cross-references")
        else:
            logger.info("No cross-references file found")
    
    def _parse_strongs_lexicon(self, path: Path) -> Dict[str, Dict]:
        """Parse Strong's lexicon XML file."""
        lexicon = {}
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # Find all entry elements
            for entry in root.findall('.//entry'):
                strongs = entry.get('strongs')
                if not strongs:
                    continue
                
                # Extract entry data
                entry_data = {
                    'strongs': strongs,
                    'unicode': '',
                    'translit': '',
                    'pronunciation': '',
                    'definition': '',
                    'kjv_def': '',
                    'derivation': ''
                }
                
                # Get Greek/Hebrew text
                greek_elem = entry.find('.//greek')
                if greek_elem is not None:
                    entry_data['unicode'] = greek_elem.get('unicode', '')
                    entry_data['translit'] = greek_elem.get('translit', '')
                
                # Get pronunciation
                pron_elem = entry.find('.//pronunciation')
                if pron_elem is not None:
                    entry_data['pronunciation'] = pron_elem.get('strongs', '')
                
                # Get definitions
                strongs_def = entry.find('.//strongs_def')
                if strongs_def is not None:
                    entry_data['definition'] = ''.join(strongs_def.itertext()).strip()
                
                kjv_def = entry.find('.//kjv_def')
                if kjv_def is not None:
                    entry_data['kjv_def'] = ''.join(kjv_def.itertext()).strip()
                
                derivation = entry.find('.//strongs_derivation')
                if derivation is not None:
                    entry_data['derivation'] = ''.join(derivation.itertext()).strip()
                
                lexicon[strongs] = entry_data
                
        except Exception as e:
            logger.error(f"Error parsing lexicon {path}: {e}")
        
        return lexicon
    
    def parse_hebrew_xml(self, file_path: Path) -> Dict[str, List[OriginalLanguageWord]]:
        """Parse Hebrew XML file to extract words with morphology."""
        verses = {}
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Define namespaces
            namespaces = {
                'osis': 'http://www.bibletechnologies.net/2003/OSIS/namespace'
            }
            
            # Find all verses
            for verse_elem in root.findall('.//osis:verse', namespaces):
                verse_id = verse_elem.get('osisID')
                if not verse_id:
                    continue
                
                # Parse verse ID (e.g., "Gen.1.1")
                parts = verse_id.split('.')
                if len(parts) != 3:
                    continue
                
                book, chapter, verse = parts
                chapter = int(chapter)
                verse = int(verse)
                
                # Extract words
                words = []
                position = 0
                
                for w_elem in verse_elem.findall('.//osis:w', namespaces):
                    # Get text content
                    text = ''.join(w_elem.itertext()).strip()
                    if not text:
                        continue
                    
                    # Get attributes
                    lemma = w_elem.get('lemma', '')
                    morph = w_elem.get('morph', '')
                    
                    # Extract Strong's number from lemma
                    strong_num = None
                    if lemma:
                        # Handle compound lemmas (e.g., "b/7225")
                        parts = lemma.split('/')
                        for part in parts:
                            if part.strip().isdigit():
                                strong_num = f"H{part.strip()}"
                                break
                    
                    # Parse morphology
                    morph_analysis = None
                    if morph:
                        try:
                            morph_analysis = self.hebrew_morph.parse(morph)
                        except Exception as e:
                            logger.debug(f"Could not parse Hebrew morph code {morph}: {e}")
                    
                    # Get lexicon entry
                    lexicon_entry = None
                    if strong_num and 'hebrew' in self.lexicon_cache:
                        lexicon_entry = self.lexicon_cache['hebrew'].get(strong_num)
                    
                    # Create word object
                    word = OriginalLanguageWord(
                        text=text,
                        transliteration=self.transliterator.transliterate(text, 'hebrew'),
                        lemma=lemma,
                        morph_code=morph,
                        morph_analysis=morph_analysis,
                        strong_number=strong_num,
                        lexicon_entry=lexicon_entry,
                        position=position
                    )
                    
                    words.append(word)
                    position += 1
                
                # Store verses by normalized book code
                normalized_book = normalize_book_name(book) or book.upper()
                verse_key = f"{normalized_book}.{chapter}.{verse}"
                verses[verse_key] = words
                
        except Exception as e:
            logger.error(f"Error parsing Hebrew XML {file_path}: {e}")
        
        return verses
    
    def parse_greek_xml(self, file_path: Path) -> Dict[str, List[OriginalLanguageWord]]:
        """Parse Greek XML file to extract words with morphology."""
        verses = {}
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Define namespaces for TEI format
            namespaces = {
                'tei': 'http://www.tei-c.org/ns/1.0'
            }
            
            # Find all ab elements (verses)
            for ab_elem in root.findall('.//tei:ab', namespaces):
                verse_ref = ab_elem.get('n')
                if not verse_ref:
                    continue
                
                # Parse verse reference (e.g., "B01K1V1" = Book 01, Chapter 1, Verse 1)
                if len(verse_ref) >= 7 and verse_ref.startswith('B'):
                    book_num = int(verse_ref[1:3])
                    chapter_start = verse_ref.find('K') + 1
                    verse_start = verse_ref.find('V') + 1
                    
                    if chapter_start > 0 and verse_start > 0:
                        chapter = int(verse_ref[chapter_start:verse_start-1])
                        verse = int(verse_ref[verse_start:])
                        
                        # Map book number to code
                        book_code = self._map_book_number_to_code(book_num)
                        if not book_code:
                            continue
                        
                        # Extract words
                        words = []
                        position = 0
                        
                        for w_elem in ab_elem.findall('.//tei:w', namespaces):
                            # Get text content
                            text = ''.join(w_elem.itertext()).strip()
                            if not text:
                                continue
                            
                            # For Byzantine text, morphology might be in different format
                            # For now, create basic word object
                            word = OriginalLanguageWord(
                                text=text,
                                transliteration=self.transliterator.transliterate(text, 'greek'),
                                position=position
                            )
                            
                            words.append(word)
                            position += 1
                        
                        verse_key = f"{book_code}.{chapter}.{verse}"
                        verses[verse_key] = words
                
        except Exception as e:
            logger.error(f"Error parsing Greek XML {file_path}: {e}")
        
        return verses
    
    def _map_book_number_to_code(self, book_num: int) -> Optional[str]:
        """Map book number from Greek text to standard book code."""
        # New Testament book mapping
        nt_books = {
            1: 'MAT', 2: 'MRK', 3: 'LUK', 4: 'JHN', 5: 'ACT',
            6: 'ROM', 7: '1CO', 8: '2CO', 9: 'GAL', 10: 'EPH',
            11: 'PHP', 12: 'COL', 13: '1TH', 14: '2TH', 15: '1TI',
            16: '2TI', 17: 'TIT', 18: 'PHM', 19: 'HEB', 20: 'JAS',
            21: '1PE', 22: '2PE', 23: '1JN', 24: '2JN', 25: '3JN',
            26: 'JUD', 27: 'REV'
        }
        return nt_books.get(book_num)
    
    def load_hebrew_data(self) -> Dict[str, Dict[str, List[OriginalLanguageWord]]]:
        """Load and parse all Hebrew text files."""
        hebrew_data = {}
        
        if not self.hebrew_dir.exists():
            logger.warning(f"Hebrew data directory not found: {self.hebrew_dir}")
            return hebrew_data
        
        # Process all Hebrew XML files
        for xml_file in self.hebrew_dir.glob("*.xml"):
            logger.info(f"Processing Hebrew file: {xml_file.name}")
            verses = self.parse_hebrew_xml(xml_file)
            
            # Group by book
            for verse_key, words in verses.items():
                book = verse_key.split('.')[0]
                if book not in hebrew_data:
                    hebrew_data[book] = {}
                hebrew_data[book][verse_key] = words
        
        return hebrew_data
    
    def load_greek_data(self) -> Dict[str, Dict[str, List[OriginalLanguageWord]]]:
        """Load and parse all Greek text files."""
        greek_data = {}
        
        if not self.greek_dir.exists():
            logger.warning(f"Greek data directory not found: {self.greek_dir}")
            return greek_data
        
        # Process all Greek XML files
        for xml_file in self.greek_dir.glob("*.xml"):
            logger.info(f"Processing Greek file: {xml_file.name}")
            verses = self.parse_greek_xml(xml_file)
            
            # Group by book
            for verse_key, words in verses.items():
                book = verse_key.split('.')[0]
                if book not in greek_data:
                    greek_data[book] = {}
                greek_data[book][verse_key] = words
        
        return greek_data
    
    def generate_annotations(self, verse_text: str, verse_id: VerseID) -> List[Dict]:
        """Generate annotations using the ML annotation engine."""
        annotations = []
        
        try:
            # Generate annotations using the engine if available
            if self.annotation_engine is None:
                return annotations
                
            ann_results = self.annotation_engine.annotate(
                text=verse_text,
                verse_id=verse_id,
                annotation_types=[
                    AnnotationType.THEOLOGICAL_THEME,
                    AnnotationType.TOPIC,
                    AnnotationType.PERSON,
                    AnnotationType.PLACE,
                    AnnotationType.EVENT
                ]
            )
            
            # Convert to dict format
            for ann in ann_results:
                annotations.append({
                    'type': ann.annotation_type.value,
                    'value': ann.value,
                    'confidence': ann.confidence,
                    'method': ann.method,
                    'metadata': ann.metadata
                })
                
        except Exception as e:
            logger.debug(f"Could not generate annotations for {verse_id}: {e}")
        
        return annotations
    
    def enrich_verse(self, book: str, chapter: int, verse: int,
                     translations: Dict[str, Dict],
                     hebrew_data: Dict[str, Dict[str, List[OriginalLanguageWord]]],
                     greek_data: Dict[str, Dict[str, List[OriginalLanguageWord]]]) -> EnrichedVerse:
        """Create enriched verse with all available data."""
        verse_id = VerseID(book, chapter, verse)
        enriched = EnrichedVerse(verse_id=verse_id)
        
        # Normalize book code
        normalized_book = normalize_book_name(book) or book.upper()
        verse_key = f"{normalized_book}.{chapter}.{verse}"
        
        # Add translations
        for trans_code, trans_data in translations.items():
            if book in trans_data.get('books', {}):
                book_data = trans_data['books'][book]
                for ch in book_data.get('chapters', []):
                    if ch['chapter'] == chapter:
                        for v in ch['verses']:
                            if v['verse'] == verse:
                                enriched.translations[trans_code] = v['text']
                                break
        
        # Add Hebrew data if available
        if normalized_book in hebrew_data and verse_key in hebrew_data[normalized_book]:
            enriched.hebrew_data = hebrew_data[normalized_book][verse_key]
        
        # Add Greek data if available
        if normalized_book in greek_data and verse_key in greek_data[normalized_book]:
            enriched.greek_data = greek_data[normalized_book][verse_key]
        
        # Generate interlinear data if original language is available
        if enriched.hebrew_data or enriched.greek_data:
            interlinear = {
                'tokens': []
            }
            
            words = enriched.hebrew_data or enriched.greek_data
            for word in words:
                token_data = {
                    'text': word.text,
                    'transliteration': word.transliteration,
                    'position': word.position
                }
                
                if word.morph_analysis:
                    token_data['morphology'] = word.morph_analysis
                
                if word.lexicon_entry:
                    token_data['gloss'] = word.lexicon_entry.get('kjv_def', '')
                    token_data['definition'] = word.lexicon_entry.get('definition', '')
                
                interlinear['tokens'].append(token_data)
            
            enriched.interlinear_data = interlinear
        
        # Generate annotations from primary translation
        primary_text = enriched.translations.get('ENG_WEB') or next(iter(enriched.translations.values()), '')
        if primary_text:
            enriched.annotations = self.generate_annotations(primary_text, verse_id)
        
        # Add metadata
        ot_books = ['GEN', 'EXO', 'LEV', 'NUM', 'DEU', 'JOS', 'JDG', 'RUT', '1SA', '2SA', '1KI', '2KI', 
                    '1CH', '2CH', 'EZR', 'NEH', 'EST', 'JOB', 'PSA', 'PRO', 'ECC', 'SNG', 'ISA', 'JER', 
                    'LAM', 'EZK', 'DAN', 'HOS', 'JOL', 'AMO', 'OBA', 'JON', 'MIC', 'NAH', 'HAB', 'ZEP', 
                    'HAG', 'ZEC', 'MAL']
        
        is_ot = normalized_book in ot_books
        
        enriched.metadata = {
            'book_name': get_book_name(normalized_book) or book,
            'testament': 'OT' if is_ot else 'NT',
            'canonical_order': self.canon_registry.get_canon('protestant').get_book_order(normalized_book),
            'has_hebrew': bool(enriched.hebrew_data),
            'has_greek': bool(enriched.greek_data),
            'word_count': len(enriched.hebrew_data or enriched.greek_data or [])
        }
        
        # Add cross-references from loaded data
        normalized_verse_id = VerseID(normalized_book, chapter, verse)
        for xref in self.cross_refs_cache:
            if xref.source_verse == normalized_verse_id:
                enriched.cross_references.append({
                    'target': str(xref.target_verse),
                    'type': xref.reference_type.value,
                    'relationship': xref.relationship.value,
                    'confidence': xref.confidence.overall_score,
                    'topic_tags': xref.topic_tags,
                    'theological_theme': xref.theological_theme,
                    'notes': getattr(xref, 'notes', '')
                })
        
        # Add timeline events (would come from external source)
        if verse_key == "GEN.1.1":
            enriched.timeline_events.append({
                'id': 'creation',
                'name': 'Creation',
                'description': 'The creation of the world',
                'categories': ['theological', 'cosmological']
            })
        
        return enriched
    
    def load_translations(self, translation_codes: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Load specified translations or all available."""
        translations = {}
        
        # Check if we're in the sources directory structure
        sources_translations = self.data_dir / "sources" / "translations"
        if sources_translations.exists():
            self.translations_dir = sources_translations
            
        if not self.translations_dir.exists():
            logger.error(f"Translations directory not found at {self.translations_dir}")
            return translations
        
        # Look for translation JSON files directly
        translation_files = list(self.translations_dir.glob("*.json"))
        logger.info(f"Found {len(translation_files)} translation files")
        
        # Filter based on requested codes
        if translation_codes:
            # Filter files that match requested codes
            filtered_files = []
            for f in translation_files:
                for code in translation_codes:
                    if code.lower() in f.stem.lower():
                        filtered_files.append(f)
                        break
            translation_files = filtered_files
        else:
            # Default to common English translations
            default_patterns = ['eng_kjv', 'eng_asv', 'eng_web', 'eng_bbe', 'eng_ylt']
            filtered_files = []
            for f in translation_files:
                for pattern in default_patterns:
                    if pattern in f.stem.lower():
                        filtered_files.append(f)
                        break
            if filtered_files:
                translation_files = filtered_files
            else:
                # If no defaults found, take first 5 English translations
                translation_files = [f for f in translation_files if 'eng_' in f.stem.lower()][:5]
        
        # Load translation files
        for file_path in translation_files:
            try:
                logger.info(f"Loading translation: {file_path.name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                    # Use version code from the data or filename
                    version_code = trans_data.get('version', file_path.stem.upper())
                    translations[version_code] = trans_data
            except Exception as e:
                logger.warning(f"Failed to load translation {file_path}: {e}")
        
        logger.info(f"Loaded {len(translations)} translations")
        return translations
    
    def process_bible(self, translation_codes: Optional[List[str]] = None,
                     limit_books: Optional[List[str]] = None) -> Dict[str, EnrichedVerse]:
        """Process complete Bible with all enrichments."""
        logger.info("Starting enhanced Bible processing...")
        
        # Adjust paths if we're in sources directory
        sources_dir = self.data_dir / "sources"
        if sources_dir.exists():
            self.hebrew_dir = sources_dir / "hebrew"
            self.greek_dir = sources_dir / "greek"
            self.lexicons_dir = sources_dir / "lexicons"
            self.data_dir = sources_dir
        
        # Load all data sources
        translations = self.load_translations(translation_codes)
        if not translations:
            logger.error("No translations loaded")
            return {}
        
        hebrew_data = self.load_hebrew_data()
        greek_data = self.load_greek_data()
        
        # Also reload cross-references in case path changed
        self._load_cross_references()
        
        logger.info(f"Loaded Hebrew data for {len(hebrew_data)} books")
        logger.info(f"Loaded Greek data for {len(greek_data)} books")
        logger.info(f"Loaded {len(self.cross_refs_cache)} cross-references")
        
        # Process all verses
        processed_verses = {}
        
        # Get all books from loaded translations
        all_books = set()
        for trans_data in translations.values():
            all_books.update(trans_data.get('books', {}).keys())
        
        # Apply book limit if specified
        if limit_books:
            all_books = {b for b in all_books if any(
                lb.upper() in b.upper() for lb in limit_books
            )}
        
        logger.info(f"Processing {len(all_books)} books...")
        
        # Process each book
        for book in sorted(all_books):
            book_code = book
            
            # Get chapter/verse structure from first translation
            first_trans = next(iter(translations.values()))
            if book not in first_trans.get('books', {}):
                continue
            
            book_data = first_trans['books'][book]
            verse_count = 0
            
            for chapter_data in book_data.get('chapters', []):
                chapter_num = chapter_data['chapter']
                
                for verse_data in chapter_data['verses']:
                    verse_num = verse_data['verse']
                    
                    # Create enriched verse
                    enriched = self.enrich_verse(
                        book_code, chapter_num, verse_num,
                        translations, hebrew_data, greek_data
                    )
                    
                    # Store in cache
                    verse_key = f"{book_code}.{chapter_num}.{verse_num}"
                    processed_verses[verse_key] = enriched
                    verse_count += 1
            
            logger.info(f"Processed {verse_count} verses from {book_code}")
        
        logger.info(f"Total processed {len(processed_verses)} verses")
        return processed_verses


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ABBA Enhanced Bible Processor - Export enriched Bible data with real integration"
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing source data (default: data)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for enriched data'
    )
    
    parser.add_argument(
        '--translations',
        nargs='+',
        help='Specific translation codes to include (e.g., eng_kjv eng_web)'
    )
    
    parser.add_argument(
        '--books',
        nargs='+',
        help='Limit processing to specific books (e.g., GEN MAT JHN)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process Bible data
    processor = EnhancedBibleProcessor(Path(args.data_dir))
    verses = processor.process_bible(args.translations, args.books)
    
    if not verses:
        logger.error("No verses processed")
        sys.exit(1)
    
    # Export enriched data
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_verses': len(verses),
        'books_processed': len(set(v.verse_id.book for v in verses.values())),
        'has_hebrew': sum(1 for v in verses.values() if v.hebrew_data),
        'has_greek': sum(1 for v in verses.values() if v.greek_data),
        'has_annotations': sum(1 for v in verses.values() if v.annotations),
        'translations': list(set(
            trans_code 
            for v in verses.values() 
            for trans_code in v.translations.keys()
        ))
    }
    
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Export sample verses for inspection
    sample_verses = {}
    for i, (verse_key, enriched) in enumerate(verses.items()):
        if i >= 10:  # Just first 10 verses as sample
            break
        
        sample_verses[verse_key] = {
            'translations': enriched.translations,
            'hebrew_words': [
                {
                    'text': w.text,
                    'transliteration': w.transliteration,
                    'lemma': w.lemma,
                    'morph': w.morph_code,
                    'gloss': w.lexicon_entry.get('kjv_def', '') if w.lexicon_entry else ''
                }
                for w in (enriched.hebrew_data or [])
            ],
            'greek_words': [
                {
                    'text': w.text,
                    'transliteration': w.transliteration
                }
                for w in (enriched.greek_data or [])
            ],
            'annotations': enriched.annotations,
            'metadata': enriched.metadata
        }
    
    with open(output_dir / 'sample_verses.json', 'w', encoding='utf-8') as f:
        json.dump(sample_verses, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Export complete! Summary and samples written to {output_dir}")


if __name__ == '__main__':
    main()