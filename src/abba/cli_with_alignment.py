#!/usr/bin/env python3
"""
Enhanced CLI with word alignment support.

This version includes word-level alignment between original languages
and translations, building on the fixed enrichment implementation.
"""

import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Import alignment functionality
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.verse_id import VerseID
from abba.book_codes import normalize_book_name
from abba.morphology.hebrew_morphology import HebrewMorphology
from abba.morphology.greek_morphology import GreekMorphology
from abba.language.transliteration import create_transliterator
# Create a default transliterator
Transliterator = create_transliterator
from abba.parsers.lexicon_parser import LexiconParser
from abba.cross_references.models import CrossReference, ReferenceConfidence
from abba.cross_references.loader import CrossReferenceLoader
from abba.timeline.models import Event, EventType, TimePoint, create_bce_date
from abba.alignment.word_alignment import IBMModel1, format_alignment_output
from abba.export.minimal_json import MinimalJSONExporter
from abba.export.minimal_sqlite import MinimalSQLiteExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AlignedEnrichedVerse:
    """Enhanced verse with all enrichment data including alignments."""
    def __init__(self, verse_id: VerseID):
        self.verse_id = verse_id
        self.translations = {}
        self.hebrew_text = None
        self.hebrew_words = []
        self.greek_text = None
        self.greek_words = []
        self.cross_references = []
        self.timeline_events = []
        self.annotations = []
        self.alignments = {}  # translation_code -> alignment data
        self.metadata = {}


class AlignedBibleProcessor:
    """Process Bible with enhanced features including word alignment."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.translations_dir = self.data_dir / "sources" / "translations" 
        self.hebrew_dir = self.data_dir / "sources" / "hebrew"
        self.greek_dir = self.data_dir / "sources" / "greek"
        self.lexicons_dir = self.data_dir / "sources" / "lexicons"
        self.timeline_dir = self.data_dir / "timeline"
        self.models_dir = Path("models") / "alignment"
        
        # Initialize processors
        self.hebrew_morph = HebrewMorphology()
        self.greek_morph = GreekMorphology()
        # Create transliterators for both languages
        self.hebrew_transliterator = create_transliterator('hebrew')
        self.greek_transliterator = create_transliterator('greek')
        
        # Cross-reference loader
        cross_ref_path = self.data_dir / "cross_references.json"
        self.cross_ref_loader = CrossReferenceLoader(cross_ref_path)
        
        # Cache for loaded data
        self.lexicon_cache = {}
        self.cross_refs_cache = []
        self.timeline_cache = []
        
        # Alignment models
        self.hebrew_aligner = None
        self.greek_aligner = None
        
    def load_alignment_models(self):
        """Load pre-trained alignment models."""
        # Load Hebrew alignment model
        hebrew_model_path = self.models_dir / "hebrew_alignment.json"
        if hebrew_model_path.exists():
            logger.info("Loading Hebrew alignment model...")
            self.hebrew_aligner = IBMModel1()
            self.hebrew_aligner.load_model(hebrew_model_path)
        else:
            logger.warning(f"Hebrew alignment model not found at {hebrew_model_path}")
            
        # Load Greek alignment model
        greek_model_path = self.models_dir / "greek_alignment.json"
        if greek_model_path.exists():
            logger.info("Loading Greek alignment model...")
            self.greek_aligner = IBMModel1()
            self.greek_aligner.load_model(greek_model_path)
        else:
            logger.warning(f"Greek alignment model not found at {greek_model_path}")
    
    def load_translations(self, translation_codes: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Load Bible translations from JSON files."""
        translations = {}
        
        # Default to ENG_KJV if no codes specified
        if not translation_codes:
            translation_codes = ["ENG_KJV"]
        
        for trans_code in translation_codes:
            # Try lowercase first (like the fixed version)
            trans_path = self.translations_dir / f"{trans_code.lower()}.json"
            if trans_path.exists():
                logger.info(f"Loading translation: {trans_code}")
                with open(trans_path, 'r', encoding='utf-8') as f:
                    translations[trans_code] = json.load(f)
            else:
                logger.warning(f"Translation not found: {trans_path}")
        
        return translations
    
    def load_hebrew_text(self, book: str) -> Dict[str, Any]:
        """Load Hebrew text for a book with complete mappings."""
        hebrew_data = {}
        
        # Complete mapping of all Hebrew books
        book_files = {
            # Torah
            'GEN': 'Gen.xml', 'Gen': 'Gen.xml',
            'EXO': 'Exod.xml', 'Exod': 'Exod.xml',
            'LEV': 'Lev.xml', 'Lev': 'Lev.xml',
            'NUM': 'Num.xml', 'Num': 'Num.xml',
            'DEU': 'Deut.xml', 'Deut': 'Deut.xml',
            # Historical Books
            'JOS': 'Josh.xml', 'Josh': 'Josh.xml',
            'JDG': 'Judg.xml', 'Judg': 'Judg.xml',
            'RUT': 'Ruth.xml', 'Ruth': 'Ruth.xml',
            '1SA': '1Sam.xml', '1Sam': '1Sam.xml',
            '2SA': '2Sam.xml', '2Sam': '2Sam.xml',
            '1KI': '1Kgs.xml', '1Kgs': '1Kgs.xml',
            '2KI': '2Kgs.xml', '2Kgs': '2Kgs.xml',
            '1CH': '1Chr.xml', '1Chr': '1Chr.xml',
            '2CH': '2Chr.xml', '2Chr': '2Chr.xml',
            'EZR': 'Ezra.xml', 'Ezra': 'Ezra.xml',
            'NEH': 'Neh.xml', 'Neh': 'Neh.xml',
            'EST': 'Esth.xml', 'Esth': 'Esth.xml',
            # Wisdom Literature
            'JOB': 'Job.xml', 'Job': 'Job.xml',
            'PSA': 'Ps.xml', 'Ps': 'Ps.xml',
            'PRO': 'Prov.xml', 'Prov': 'Prov.xml',
            'ECC': 'Eccl.xml', 'Eccl': 'Eccl.xml',
            'SNG': 'Song.xml', 'Song': 'Song.xml',
            # Major Prophets
            'ISA': 'Isa.xml', 'Isa': 'Isa.xml',
            'JER': 'Jer.xml', 'Jer': 'Jer.xml',
            'LAM': 'Lam.xml', 'Lam': 'Lam.xml',
            'EZK': 'Ezek.xml', 'Ezek': 'Ezek.xml',
            'DAN': 'Dan.xml', 'Dan': 'Dan.xml',
            # Minor Prophets
            'HOS': 'Hos.xml', 'Hos': 'Hos.xml',
            'JOL': 'Joel.xml', 'Joel': 'Joel.xml',
            'AMO': 'Amos.xml', 'Amos': 'Amos.xml',
            'OBA': 'Obad.xml', 'Obad': 'Obad.xml',
            'JON': 'Jonah.xml', 'Jonah': 'Jonah.xml',
            'MIC': 'Mic.xml', 'Mic': 'Mic.xml',
            'NAM': 'Nah.xml', 'Nah': 'Nah.xml',
            'HAB': 'Hab.xml', 'Hab': 'Hab.xml',
            'ZEP': 'Zeph.xml', 'Zeph': 'Zeph.xml',
            'HAG': 'Hag.xml', 'Hag': 'Hag.xml',
            'ZEC': 'Zech.xml', 'Zech': 'Zech.xml',
            'MAL': 'Mal.xml', 'Mal': 'Mal.xml',
        }
        
        filename = book_files.get(book) or book_files.get(book.upper())
        if not filename:
            return hebrew_data
            
        file_path = self.hebrew_dir / filename
        if not file_path.exists():
            return hebrew_data
            
        logger.info(f"Loading Hebrew text for {book}")
        
        try:
            # Parse the OSIS XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Define namespaces
            ns = {'osis': 'http://www.bibletechnologies.net/2003/OSIS/namespace'}
            
            # Find all verses
            for verse_elem in root.findall('.//osis:verse', ns):
                osisID = verse_elem.get('osisID', '')
                if not osisID:
                    continue
                    
                # Parse verse reference
                parts = osisID.split('.')
                if len(parts) >= 3:
                    chapter = int(parts[1])
                    verse = int(parts[2])
                    
                    # Extract words with morphology
                    words = []
                    for w in verse_elem.findall('.//osis:w', ns):
                        word_data = {
                            'text': w.text or '',
                            'lemma': w.get('lemma', ''),
                            'morph': w.get('morph', ''),
                            'transliteration': self.hebrew_transliterator.transliterate(w.text or '')
                        }
                        
                        # Parse morphology if available
                        if word_data['morph']:
                            morph_code = word_data['morph'].replace('oshm:', '')
                            try:
                                morph_data = self.hebrew_morph.parse_morph_code(morph_code)
                                word_data['morphology'] = morph_data
                            except:
                                pass
                                
                        words.append(word_data)
                    
                    verse_key = f"{chapter}.{verse}"
                    hebrew_data[verse_key] = {
                        'text': ' '.join(w['text'] for w in words),
                        'words': words
                    }
                    
        except Exception as e:
            logger.error(f"Error parsing Hebrew XML for {book}: {e}")
            
        return hebrew_data
    
    def load_greek_text(self, book: str) -> Dict[str, Any]:
        """Load Greek text for a book with complete mappings."""
        greek_data = {}
        
        # Complete mapping of all Greek books
        book_files = {
            # Gospels
            'MAT': 'MAT.xml', 'Matt': 'MAT.xml',
            'MRK': 'MAR.xml', 'Mark': 'MAR.xml', 
            'LUK': 'LUK.xml', 'Luke': 'LUK.xml',
            'JHN': 'JOH.xml', 'John': 'JOH.xml',
            # History
            'ACT': 'ACT.xml', 'Acts': 'ACT.xml',
            # Pauline Epistles
            'ROM': 'ROM.xml', 'Rom': 'ROM.xml',
            '1CO': '1CO.xml', '1Cor': '1CO.xml',
            '2CO': '2CO.xml', '2Cor': '2CO.xml',
            'GAL': 'GAL.xml', 'Gal': 'GAL.xml',
            'EPH': 'EPH.xml', 'Eph': 'EPH.xml',
            'PHP': 'PHP.xml', 'Phil': 'PHP.xml',
            'COL': 'COL.xml', 'Col': 'COL.xml',
            '1TH': '1TH.xml', '1Thess': '1TH.xml',
            '2TH': '2TH.xml', '2Thess': '2TH.xml',
            '1TI': '1TI.xml', '1Tim': '1TI.xml',
            '2TI': '2TI.xml', '2Tim': '2TI.xml',
            'TIT': 'TIT.xml', 'Titus': 'TIT.xml',
            'PHM': 'PHM.xml', 'Phlm': 'PHM.xml',
            # General Epistles
            'HEB': 'HEB.xml', 'Heb': 'HEB.xml',
            'JAS': 'JAM.xml', 'Jas': 'JAM.xml',
            '1PE': '1PE.xml', '1Pet': '1PE.xml',
            '2PE': '2PE.xml', '2Pet': '2PE.xml',
            '1JN': '1JO.xml', '1John': '1JO.xml',
            '2JN': '2JO.xml', '2John': '2JO.xml',
            '3JN': '3JO.xml', '3John': '3JO.xml',
            'JUD': 'JUD.xml', 'Jude': 'JUD.xml',
            # Apocalyptic
            'REV': 'REV.xml', 'Rev': 'REV.xml',
        }
        
        filename = book_files.get(book) or book_files.get(book.upper())
        if not filename:
            return greek_data
            
        file_path = self.greek_dir / filename
        if not file_path.exists():
            return greek_data
            
        logger.info(f"Loading Greek text for {book}")
        
        try:
            # Parse the XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Handle TEI namespace
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Process chapters and verses in TEI format
            for chapter_elem in root.findall('.//tei:div[@type="chapter"]', ns):
                chapter_n = chapter_elem.get('n', '')
                if chapter_n and chapter_n.startswith('B'):
                    # Extract chapter number from format like "B01K1"
                    k_index = chapter_n.find('K')
                    if k_index > 0:
                        chapter_num = int(chapter_n[k_index+1:])
                        
                        # Process verses in this chapter
                        for verse_elem in chapter_elem.findall('.//tei:ab', ns):
                            verse_n = verse_elem.get('n', '')
                            if verse_n and verse_n.startswith('B'):
                                # Extract verse number from format like "B01K1V1"
                                v_index = verse_n.find('V')
                                if v_index > 0:
                                    verse_num = int(verse_n[v_index+1:])
                                    
                                    # Extract words
                                    words = []
                                    for w in verse_elem.findall('.//tei:w', ns):
                                        word_data = {
                                            'text': w.text or '',
                                            'lemma': w.get('lemma', ''),
                                            'morph': w.get('type', ''),
                                            'strongs': w.get('lemma', ''),
                                            'transliteration': self.greek_transliterator.transliterate(w.text or '')
                                        }
                                        words.append(word_data)
                                    
                                    verse_key = f"{chapter_num}.{verse_num}"
                                    greek_data[verse_key] = {
                                        'text': ' '.join(w['text'] for w in words),
                                        'words': words
                                    }
                    
        except Exception as e:
            logger.error(f"Error parsing Greek XML for {book}: {e}")
            
        return greek_data
    
    def load_timeline_events(self):
        """Load comprehensive timeline events."""
        # Load from file if available
        timeline_file = self.data_dir / "timeline_events.json"
        if timeline_file.exists():
            logger.info("Loading timeline events from file")
            try:
                with open(timeline_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for event_data in data.get('events', []):
                        # Parse verse references
                        verse_refs = []
                        for ref in event_data.get('verse_refs', []):
                            # Parse reference like "GEN.1.1"
                            parts = ref.split('.')
                            if len(parts) == 3:
                                verse_refs.append(VerseID(parts[0], int(parts[1]), int(parts[2])))
                        
                        # Create time point - handle negative dates as BCE
                        date_value = event_data.get('date', 0)
                        if date_value < 0:
                            date = create_bce_date(abs(date_value))
                        else:
                            date = datetime(date_value, 1, 1)
                        
                        if not date:
                            continue
                        
                        time_point = TimePoint(
                            exact_date=date,
                            confidence=event_data.get('confidence', 0.5)
                        )
                        
                        # Create event
                        event = Event(
                            id=event_data['id'],
                            name=event_data['name'],
                            description=event_data.get('description', ''),
                            event_type=EventType.POINT,
                            time_point=time_point,
                            verse_refs=verse_refs,
                            categories=event_data.get('categories', [])
                        )
                        
                        self.timeline_cache.append(event)
                        
                logger.info(f"Loaded {len(self.timeline_cache)} timeline events")
            except Exception as e:
                logger.error(f"Error loading timeline events: {e}")
    
    def align_verse_texts(self, verse: AlignedEnrichedVerse):
        """Add word alignments to verse."""
        # Align Hebrew text
        if verse.hebrew_words and self.hebrew_aligner:
            for trans_code, trans_text in verse.translations.items():
                if trans_text:
                    alignment = self.hebrew_aligner.align_verse(
                        verse.hebrew_words, trans_text
                    )
                    verse.alignments[trans_code] = format_alignment_output(
                        alignment, {trans_code: trans_text}
                    )['alignments'][trans_code]
                    
        # Align Greek text  
        elif verse.greek_words and self.greek_aligner:
            for trans_code, trans_text in verse.translations.items():
                if trans_text:
                    alignment = self.greek_aligner.align_verse(
                        verse.greek_words, trans_text
                    )
                    verse.alignments[trans_code] = format_alignment_output(
                        alignment, {trans_code: trans_text}
                    )['alignments'][trans_code]
    
    def enrich_verse(self, book: str, chapter: int, verse: int, 
                     translations: Dict[str, Dict],
                     hebrew_data: Dict[str, Any],
                     greek_data: Dict[str, Any]) -> AlignedEnrichedVerse:
        """Create enriched verse with all available data including alignments."""
        # Try to normalize book code
        normalized_book = normalize_book_name(book) or book.upper()
        verse_id = VerseID(normalized_book, chapter, verse)
        enriched = AlignedEnrichedVerse(verse_id=verse_id)
        
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
        
        # Add Hebrew text if available
        verse_key = f"{chapter}.{verse}"
        if verse_key in hebrew_data:
            heb_data = hebrew_data[verse_key]
            enriched.hebrew_text = heb_data['text']
            enriched.hebrew_words = heb_data['words']
        
        # Add Greek text if available
        if verse_key in greek_data:
            grk_data = greek_data[verse_key]
            enriched.greek_text = grk_data['text']
            enriched.greek_words = grk_data['words']
        
        # Add metadata
        # Determine testament
        if normalized_book in ['GEN', 'EXO', 'LEV', 'NUM', 'DEU', 'JOS', 'JDG', 'RUT', 
                               '1SA', '2SA', '1KI', '2KI', '1CH', '2CH', 'EZR', 'NEH', 
                               'EST', 'JOB', 'PSA', 'PRO', 'ECC', 'SNG', 'ISA', 'JER',
                               'LAM', 'EZK', 'DAN', 'HOS', 'JOL', 'AMO', 'OBA', 'JON',
                               'MIC', 'NAH', 'HAB', 'ZEP', 'HAG', 'ZEC', 'MAL']:
            enriched.metadata['testament'] = 'OT'
        else:
            enriched.metadata['testament'] = 'NT'
        
        # Add book metadata
        enriched.metadata['book_name'] = book
        enriched.metadata['canonical_order'] = self._get_canonical_order(normalized_book)
        
        # Add cross-references
        for xref in self.cross_refs_cache:
            if xref.source == verse_id:
                enriched.cross_references.append({
                    'target': str(xref.target),
                    'type': xref.reference_type.value if hasattr(xref.reference_type, 'value') else str(xref.reference_type),
                    'relationship': xref.relationship.value if hasattr(xref.relationship, 'value') else str(xref.relationship),
                    'confidence': xref.confidence.overall_score if hasattr(xref.confidence, 'overall_score') else xref.confidence,
                    'topic_tags': xref.topic_tags if xref.topic_tags else [],
                    'theological_theme': xref.theological_theme if hasattr(xref, 'theological_theme') else None
                })
        
        # Add timeline events
        for event in self.timeline_cache:
            if verse_id in event.verse_refs:
                enriched.timeline_events.append({
                    'id': event.id,
                    'name': event.name,
                    'description': event.description,
                    'date': event.time_point.exact_date.isoformat() if event.time_point.exact_date else None,
                    'confidence': event.time_point.confidence,
                    'categories': event.categories
                })
        
        # Add word alignments
        self.align_verse_texts(enriched)
        
        return enriched
    
    def _get_canonical_order(self, book: str) -> int:
        """Get canonical order for a book."""
        order_map = {
            'GEN': 1, 'EXO': 2, 'LEV': 3, 'NUM': 4, 'DEU': 5,
            'JOS': 6, 'JDG': 7, 'RUT': 8, '1SA': 9, '2SA': 10,
            '1KI': 11, '2KI': 12, '1CH': 13, '2CH': 14, 'EZR': 15,
            'NEH': 16, 'EST': 17, 'JOB': 18, 'PSA': 19, 'PRO': 20,
            'ECC': 21, 'SNG': 22, 'ISA': 23, 'JER': 24, 'LAM': 25,
            'EZK': 26, 'DAN': 27, 'HOS': 28, 'JOL': 29, 'AMO': 30,
            'OBA': 31, 'JON': 32, 'MIC': 33, 'NAH': 34, 'HAB': 35,
            'ZEP': 36, 'HAG': 37, 'ZEC': 38, 'MAL': 39,
            'MAT': 40, 'MRK': 41, 'LUK': 42, 'JHN': 43, 'ACT': 44,
            'ROM': 45, '1CO': 46, '2CO': 47, 'GAL': 48, 'EPH': 49,
            'PHP': 50, 'COL': 51, '1TH': 52, '2TH': 53, '1TI': 54,
            '2TI': 55, 'TIT': 56, 'PHM': 57, 'HEB': 58, 'JAS': 59,
            '1PE': 60, '2PE': 61, '1JN': 62, '2JN': 63, '3JN': 64,
            'JUD': 65, 'REV': 66
        }
        return order_map.get(book, 99)
    
    def process_bible(self, translation_codes: Optional[List[str]] = None,
                     book_filter: Optional[List[str]] = None) -> Dict[str, AlignedEnrichedVerse]:
        """Process Bible with all enrichments including alignments."""
        logger.info("Starting aligned Bible processing...")
        
        # Load translations
        translations = self.load_translations(translation_codes)
        if not translations:
            logger.error("No translations loaded")
            return {}
        
        # Load alignment models
        self.load_alignment_models()
        
        # Load cross-references
        logger.info("Loading cross-references...")
        try:
            self.cross_refs_cache = self.cross_ref_loader.load_from_json()
            logger.info(f"Loaded {len(self.cross_refs_cache)} cross-references")
        except Exception as e:
            logger.warning(f"Could not load cross-references: {e}")
            self.cross_refs_cache = []
        
        # Load timeline events
        self.load_timeline_events()
        
        # Process all verses
        processed_verses = {}
        
        # Get all books from loaded translations
        all_books = set()
        for trans_data in translations.values():
            all_books.update(trans_data.get('books', {}).keys())
        
        # Apply book filter if specified
        if book_filter:
            all_books = all_books.intersection(set(book_filter))
        
        logger.info(f"Processing {len(all_books)} books...")
        
        # Track enrichment statistics
        stats = {
            'total_verses': 0,
            'hebrew_enriched': 0,
            'greek_enriched': 0,
            'cross_refs_added': 0,
            'timeline_events_added': 0,
            'aligned_verses': 0
        }
        
        for book in sorted(all_books):
            # Load Hebrew/Greek data for this book
            hebrew_data = self.load_hebrew_text(book)
            greek_data = self.load_greek_text(book)
            
            # Get chapter/verse structure from first translation
            first_trans = next(iter(translations.values()))
            if book not in first_trans.get('books', {}):
                continue
                
            book_data = first_trans['books'][book]
            
            for chapter_data in book_data.get('chapters', []):
                chapter_num = chapter_data['chapter']
                
                for verse_data in chapter_data['verses']:
                    verse_num = verse_data['verse']
                    
                    # Create enriched verse
                    enriched = self.enrich_verse(
                        book, chapter_num, verse_num, 
                        translations, hebrew_data, greek_data
                    )
                    
                    # Update statistics
                    stats['total_verses'] += 1
                    if enriched.hebrew_text:
                        stats['hebrew_enriched'] += 1
                    if enriched.greek_text:
                        stats['greek_enriched'] += 1
                    if enriched.cross_references:
                        stats['cross_refs_added'] += len(enriched.cross_references)
                    if enriched.timeline_events:
                        stats['timeline_events_added'] += len(enriched.timeline_events)
                    if enriched.alignments:
                        stats['aligned_verses'] += 1
                    
                    # Store in cache
                    verse_key = f"{book}.{chapter_num}.{verse_num}"
                    processed_verses[verse_key] = enriched
        
        # Log enrichment statistics
        logger.info(f"Processed {stats['total_verses']} verses")
        logger.info(f"Hebrew enrichment: {stats['hebrew_enriched']}/{stats['total_verses']} ({stats['hebrew_enriched']/stats['total_verses']*100:.1f}%)")
        logger.info(f"Greek enrichment: {stats['greek_enriched']}/{stats['total_verses']} ({stats['greek_enriched']/stats['total_verses']*100:.1f}%)")
        logger.info(f"Cross-references added: {stats['cross_refs_added']}")
        logger.info(f"Timeline events added: {stats['timeline_events_added']}")
        logger.info(f"Verses with alignments: {stats['aligned_verses']}/{stats['total_verses']} ({stats['aligned_verses']/stats['total_verses']*100:.1f}%)")
        
        return processed_verses


class AlignedBibleExporter:
    """Export enriched Bible data with alignments to JSON format."""
    
    def export_to_json(self, verses: Dict[str, AlignedEnrichedVerse], output_dir: Path) -> None:
        """Export to JSON files organized by book."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group verses by book
        by_book = defaultdict(lambda: defaultdict(list))
        
        for verse_key, enriched in verses.items():
            parts = verse_key.split('.')
            book = parts[0]
            chapter = int(parts[1])
            
            verse_data = {
                'verse': enriched.verse_id.verse,
                'verse_id': verse_key,
                'translations': enriched.translations,
                'hebrew_text': enriched.hebrew_text,
                'hebrew_words': enriched.hebrew_words,
                'greek_text': enriched.greek_text,
                'greek_words': enriched.greek_words,
                'cross_references': enriched.cross_references,
                'timeline_events': enriched.timeline_events,
                'alignments': enriched.alignments,
                'metadata': enriched.metadata
            }
            
            by_book[book][chapter].append(verse_data)
        
        # Write each book
        for book, chapters in by_book.items():
            book_data = {
                'book': book,
                'chapters': []
            }
            
            for chapter_num in sorted(chapters.keys()):
                chapter_data = {
                    'chapter': chapter_num,
                    'verses': sorted(chapters[chapter_num], key=lambda v: v['verse'])
                }
                book_data['chapters'].append(chapter_data)
            
            # Write book file
            book_file = output_dir / f"{book}.json"
            with open(book_file, 'w', encoding='utf-8') as f:
                json.dump(book_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported {book}.json")
            
        # Write statistics with alignment info
        stats_file = output_dir / "_statistics.json"
        verses_with_hebrew = sum(1 for v in verses.values() if v.hebrew_text)
        verses_with_greek = sum(1 for v in verses.values() if v.greek_text)
        verses_with_xrefs = sum(1 for v in verses.values() if v.cross_references)
        verses_with_timeline = sum(1 for v in verses.values() if v.timeline_events)
        verses_with_alignment = sum(1 for v in verses.values() if v.alignments)
        
        stats = {
            'total_verses': len(verses),
            'verses_with_hebrew': verses_with_hebrew,
            'verses_with_greek': verses_with_greek,
            'verses_with_cross_references': verses_with_xrefs,
            'verses_with_timeline_events': verses_with_timeline,
            'verses_with_alignments': verses_with_alignment,
            'books_processed': len(by_book),
            'enrichment_coverage': {
                'hebrew_percentage': round(verses_with_hebrew / len(verses) * 100, 2) if verses else 0,
                'greek_percentage': round(verses_with_greek / len(verses) * 100, 2) if verses else 0,
                'cross_reference_percentage': round(verses_with_xrefs / len(verses) * 100, 2) if verses else 0,
                'timeline_percentage': round(verses_with_timeline / len(verses) * 100, 2) if verses else 0,
                'alignment_percentage': round(verses_with_alignment / len(verses) * 100, 2) if verses else 0,
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ABBA - Advanced Biblical Text Processing with Word Alignment"
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for enriched data'
    )
    
    parser.add_argument(
        '--translations',
        nargs="+",
        default=["ENG_KJV"],
        help='Translation codes to include (default: ENG_KJV)'
    )
    
    parser.add_argument(
        '--books',
        nargs="+",
        help='Specific books to export (default: all)'
    )
    
    parser.add_argument(
        '--train-alignments',
        action='store_true',
        help='Train alignment models before processing'
    )
    
    args = parser.parse_args()
    
    # Train alignment models if requested
    if args.train_alignments:
        logger.info("Training alignment models first...")
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/train_word_alignment.py",
            "--export-dir", "full_fixed_export"
        ])
        if result.returncode != 0:
            logger.error("Alignment training failed")
            return 1
    
    # Create processor
    processor = AlignedBibleProcessor()
    
    # Process Bible
    verses = processor.process_bible(
        translation_codes=args.translations,
        book_filter=args.books
    )
    
    if not verses:
        logger.error("No verses processed")
        return 1
    
    # Export with alignments
    exporter = AlignedBibleExporter()
    exporter.export_to_json(verses, args.output)
    
    logger.info("Export complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())