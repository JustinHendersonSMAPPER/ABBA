"""
Fixed version of Simple enhanced CLI with comprehensive enrichment coverage.
This version processes all available Hebrew/Greek texts and includes more timeline events.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import xml.etree.ElementTree as ET

from abba.book_codes import normalize_book_name, get_book_name
from abba.canon.registry import CanonRegistry
from abba.cross_references.models import CrossReference, ReferenceType, ReferenceRelationship, ReferenceConfidence
from abba.cross_references.loader import CrossReferenceLoader
from abba.language.transliteration import create_transliterator
from abba.morphology.hebrew_morphology import HebrewMorphology
from abba.morphology.greek_morphology import GreekMorphology
from abba.parsers.hebrew_parser import HebrewParser
from abba.parsers.greek_parser import GreekParser
from abba.parsers.lexicon_parser import LexiconParser
from abba.timeline.models import Event, TimePoint, EventType, create_bce_date
from abba.verse_id import VerseID


logger = logging.getLogger(__name__)


@dataclass
class EnrichedVerse:
    """Verse with all enrichment data."""
    verse_id: VerseID
    translations: Dict[str, str] = field(default_factory=dict)
    hebrew_text: Optional[str] = None
    hebrew_words: List[Dict] = field(default_factory=list)
    greek_text: Optional[str] = None
    greek_words: List[Dict] = field(default_factory=list)
    cross_references: List[Dict] = field(default_factory=list)
    timeline_events: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class SimpleEnhancedBibleProcessor:
    """Process Bible data with comprehensive data integration."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.translations_dir = data_dir / "sources" / "translations"
        self.hebrew_dir = data_dir / "sources" / "hebrew"
        self.greek_dir = data_dir / "sources" / "greek"
        self.lexicons_dir = data_dir / "sources" / "lexicons"
        
        # Initialize components
        self.canon_registry = CanonRegistry()
        self.hebrew_morph = HebrewMorphology()
        self.greek_morph = GreekMorphology()
        self.hebrew_parser = HebrewParser()
        self.greek_parser = GreekParser()
        self.transliterator = create_transliterator("hebrew")
        
        # Initialize cross-reference loader
        self.cross_ref_loader = CrossReferenceLoader(data_dir)
        
        # Cache for processed data
        self.verses_cache: Dict[str, EnrichedVerse] = {}
        self.cross_refs_cache: List[CrossReference] = []
        self.timeline_cache: List[Event] = []
        self.lexicon_cache: Dict[str, Dict] = {}
        
    def load_translations(self, translation_codes: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Load specified translations."""
        translations = {}
        
        # Load each requested translation
        for code in translation_codes or ["eng_kjv"]:
            file_path = self.translations_dir / f"{code.lower()}.json"
            if file_path.exists():
                logger.info(f"Loading translation: {code}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    translations[code] = json.load(f)
            else:
                logger.warning(f"Translation file not found: {file_path}")
                
        return translations
    
    def load_hebrew_text(self, book: str) -> Dict[str, Any]:
        """Load Hebrew text for a book with comprehensive coverage."""
        hebrew_data = {}
        
        # Comprehensive mapping for all Hebrew Bible books
        book_files = {
            'GEN': 'Gen.xml', 'Gen': 'Gen.xml',
            'EXO': 'Exod.xml', 'Exod': 'Exod.xml',
            'LEV': 'Lev.xml', 'Lev': 'Lev.xml',
            'NUM': 'Num.xml', 'Num': 'Num.xml',
            'DEU': 'Deut.xml', 'Deut': 'Deut.xml',
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
            'JOB': 'Job.xml', 'Job': 'Job.xml',
            'PSA': 'Ps.xml', 'Ps': 'Ps.xml', 'Psa': 'Ps.xml',
            'PRO': 'Prov.xml', 'Prov': 'Prov.xml',
            'ECC': 'Eccl.xml', 'Eccl': 'Eccl.xml',
            'SNG': 'Song.xml', 'Song': 'Song.xml',
            'ISA': 'Isa.xml', 'Isa': 'Isa.xml',
            'JER': 'Jer.xml', 'Jer': 'Jer.xml',
            'LAM': 'Lam.xml', 'Lam': 'Lam.xml',
            'EZK': 'Ezek.xml', 'Ezek': 'Ezek.xml',
            'DAN': 'Dan.xml', 'Dan': 'Dan.xml',
            'HOS': 'Hos.xml', 'Hos': 'Hos.xml',
            'JOL': 'Joel.xml', 'Joel': 'Joel.xml',
            'AMO': 'Amos.xml', 'Amos': 'Amos.xml',
            'OBA': 'Obad.xml', 'Obad': 'Obad.xml',
            'JON': 'Jonah.xml', 'Jonah': 'Jonah.xml',
            'MIC': 'Mic.xml', 'Mic': 'Mic.xml',
            'NAH': 'Nah.xml', 'Nah': 'Nah.xml',
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
                            'transliteration': self.transliterator.transliterate(w.text or '')
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
        """Load Greek text for a book with comprehensive coverage."""
        greek_data = {}
        
        # Comprehensive mapping for all New Testament books
        book_files = {
            'MAT': 'MAT.xml', 'Matt': 'MAT.xml', 'Matthew': 'MAT.xml',
            'MRK': 'MAR.xml', 'Mark': 'MAR.xml', 'MAR': 'MAR.xml',
            'LUK': 'LUK.xml', 'Luke': 'LUK.xml',
            'JHN': 'JOH.xml', 'John': 'JOH.xml', 'JOH': 'JOH.xml',
            'ACT': 'ACT.xml', 'Acts': 'ACT.xml',
            'ROM': 'ROM.xml', 'Rom': 'ROM.xml',
            '1CO': '1CO.xml', '1Cor': '1CO.xml',
            '2CO': '2CO.xml', '2Cor': '2CO.xml',
            'GAL': 'GAL.xml', 'Gal': 'GAL.xml',
            'EPH': 'EPH.xml', 'Eph': 'EPH.xml',
            'PHP': 'PHP.xml', 'Phil': 'PHP.xml', 'PHI': 'PHP.xml',
            'COL': 'COL.xml', 'Col': 'COL.xml',
            '1TH': '1TH.xml', '1Thess': '1TH.xml',
            '2TH': '2TH.xml', '2Thess': '2TH.xml',
            '1TI': '1TI.xml', '1Tim': '1TI.xml',
            '2TI': '2TI.xml', '2Tim': '2TI.xml',
            'TIT': 'TIT.xml', 'Titus': 'TIT.xml',
            'PHM': 'PHM.xml', 'Phlm': 'PHM.xml', 'Philem': 'PHM.xml',
            'HEB': 'HEB.xml', 'Heb': 'HEB.xml',
            'JAM': 'JAM.xml', 'Jas': 'JAM.xml', 'James': 'JAM.xml',
            '1PE': '1PE.xml', '1Pet': '1PE.xml',
            '2PE': '2PE.xml', '2Pet': '2PE.xml',
            '1JO': '1JO.xml', '1John': '1JO.xml',
            '2JO': '2JO.xml', '2John': '2JO.xml',
            '3JO': '3JO.xml', '3John': '3JO.xml',
            'JUD': 'JUD.xml', 'Jude': 'JUD.xml',
            'REV': 'REV.xml', 'Rev': 'REV.xml', 'Revelation': 'REV.xml',
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
                    # Find the K and get the number after it
                    k_index = chapter_n.find('K')
                    if k_index > 0:
                        chapter_num = int(chapter_n[k_index+1:])
                    
                    for verse_elem in chapter_elem.findall('.//tei:ab', ns):
                        try:
                            verse_n = verse_elem.get('n', '')
                            if verse_n and 'V' in verse_n:
                                # Extract verse number from format like "B01K1V1"
                                verse_num = int(verse_n.split('V')[1])
                                
                                # Extract words
                                words = []
                                for w in verse_elem.findall('.//tei:w', ns):
                                    word_data = {
                                        'text': (w.text or '').strip(),
                                        'lemma': w.get('lemma', ''),
                                        'morph': w.get('type', ''),
                                        'strongs': w.get('strongs', ''),
                                        'transliteration': self.transliterator.transliterate(w.text or '')
                                    }
                                    
                                    # Parse morphology
                                    if word_data['morph']:
                                        try:
                                            morph_data = self.greek_morph.parse_morph_code(word_data['morph'])
                                            word_data['morphology'] = morph_data
                                        except:
                                            pass
                                            
                                    words.append(word_data)
                                
                                verse_key = f"{chapter_num}.{verse_num}"
                                greek_data[verse_key] = {
                                    'text': ' '.join(w['text'] for w in words if w['text']),
                                    'words': words
                                }
                        except Exception as e:
                            logger.debug(f"Error processing verse: {e}")
                    
        except Exception as e:
            logger.error(f"Error parsing Greek XML for {book}: {e}")
            
        return greek_data
    
    def load_timeline_events(self):
        """Load timeline events from file or create comprehensive default set."""
        timeline_file = self.data_dir / "timeline_events.json"
        
        if timeline_file.exists():
            logger.info("Loading timeline events from file")
            try:
                with open(timeline_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                events = []
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
                        event_type=EventType(event_data.get('event_type', 'point')),
                        time_point=time_point,
                        verse_refs=verse_refs,
                        categories=event_data.get('categories', [])
                    )
                    
                    events.append(event)
                
                self.timeline_cache = events
                logger.info(f"Loaded {len(events)} timeline events")
                return
                
            except Exception as e:
                logger.warning(f"Error loading timeline events: {e}")
        
        # Create comprehensive default timeline
        logger.info("Creating default timeline events")
        self.timeline_cache = [
            Event(
                id="creation",
                name="Creation",
                description="The creation of the world",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(4004), confidence=0.3),
                verse_refs=[VerseID("GEN", 1, 1)],
                categories=["theological", "cosmological"]
            ),
            Event(
                id="flood",
                name="The Great Flood",
                description="The flood of Noah",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(2349), confidence=0.3),
                verse_refs=[VerseID("GEN", 7, 11), VerseID("GEN", 8, 4)],
                categories=["judgment", "covenant"]
            ),
            Event(
                id="abraham_call",
                name="Call of Abraham",
                description="God calls Abraham to leave Ur",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(2091), confidence=0.5),
                verse_refs=[VerseID("GEN", 12, 1)],
                categories=["covenant", "patriarchal"]
            ),
            Event(
                id="exodus",
                name="The Exodus",
                description="Israel leaves Egypt",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(1446), confidence=0.7),
                verse_refs=[VerseID("EXO", 12, 31), VerseID("EXO", 12, 41)],
                categories=["historical", "foundational", "redemption"]
            ),
            Event(
                id="david_king",
                name="David Becomes King",
                description="David becomes king over all Israel",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(1010), confidence=0.8),
                verse_refs=[VerseID("2SA", 5, 3)],
                categories=["monarchy", "davidic_covenant"]
            ),
            Event(
                id="temple_built",
                name="Solomon's Temple Built",
                description="Completion of the First Temple",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(959), confidence=0.8),
                verse_refs=[VerseID("1KI", 6, 38)],
                categories=["temple", "worship"]
            ),
            Event(
                id="exile_babylon",
                name="Babylonian Exile",
                description="Judah taken into captivity",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(586), confidence=0.9),
                verse_refs=[VerseID("2KI", 25, 8), VerseID("JER", 52, 12)],
                categories=["judgment", "exile"]
            ),
            Event(
                id="jesus_birth",
                name="Birth of Jesus",
                description="The birth of Jesus Christ",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(4), confidence=0.7),
                verse_refs=[VerseID("MAT", 2, 1), VerseID("LUK", 2, 7)],
                categories=["incarnation", "messianic"]
            ),
            Event(
                id="crucifixion",
                name="Crucifixion of Jesus",
                description="The death of Jesus on the cross",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=datetime(30, 4, 3), confidence=0.8),
                verse_refs=[VerseID("MAT", 27, 35), VerseID("MRK", 15, 24), 
                           VerseID("LUK", 23, 33), VerseID("JHN", 19, 18)],
                categories=["redemption", "atonement"]
            ),
            Event(
                id="resurrection",
                name="Resurrection of Jesus",
                description="Jesus rises from the dead",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=datetime(30, 4, 5), confidence=0.8),
                verse_refs=[VerseID("MAT", 28, 6), VerseID("MRK", 16, 6),
                           VerseID("LUK", 24, 6), VerseID("JHN", 20, 1)],
                categories=["resurrection", "victory"]
            ),
            Event(
                id="pentecost",
                name="Day of Pentecost",
                description="The Holy Spirit comes upon the disciples",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=datetime(30, 5, 24), confidence=0.8),
                verse_refs=[VerseID("ACT", 2, 1)],
                categories=["church", "holy_spirit"]
            ),
        ]
    
    def load_lexicon_data(self):
        """Load lexicon data from files."""
        if self.lexicon_cache:
            return
            
        # Load Greek lexicon
        greek_lex_path = self.lexicons_dir / "strongs_greek.xml"
        if greek_lex_path.exists():
            logger.info("Loading Greek lexicon")
            try:
                parser = LexiconParser()
                self.lexicon_cache['greek'] = parser.parse_strongs_greek(str(greek_lex_path))
                logger.info(f"Loaded {len(self.lexicon_cache['greek'])} Greek entries")
            except Exception as e:
                logger.error(f"Error loading Greek lexicon: {e}")
        
        # Load Hebrew lexicon  
        hebrew_lex_path = self.lexicons_dir / "strongs_hebrew.xml"
        if hebrew_lex_path.exists():
            logger.info("Loading Hebrew lexicon")
            try:
                parser = LexiconParser()
                self.lexicon_cache['hebrew'] = parser.parse_strongs_hebrew(str(hebrew_lex_path))
                logger.info(f"Loaded {len(self.lexicon_cache['hebrew'])} Hebrew entries")
            except Exception as e:
                logger.error(f"Error loading Hebrew lexicon: {e}")
    
    def enrich_verse(self, book: str, chapter: int, verse: int, 
                     translations: Dict[str, Dict],
                     hebrew_data: Dict[str, Any],
                     greek_data: Dict[str, Any]) -> EnrichedVerse:
        """Create enriched verse with all available data."""
        # Try to normalize book code
        normalized_book = normalize_book_name(book) or book.upper()
        verse_id = VerseID(normalized_book, chapter, verse)
        enriched = EnrichedVerse(verse_id=verse_id)
        
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
            
            # Add lexicon data for Hebrew words
            if 'hebrew' in self.lexicon_cache:
                for word in enriched.hebrew_words:
                    if word.get('lemma'):
                        # Extract Strong's number from lemma
                        strongs = word['lemma'].split('/')[-1] if '/' in word['lemma'] else word['lemma']
                        if strongs in self.lexicon_cache['hebrew']:
                            word['lexicon'] = self.lexicon_cache['hebrew'][strongs]
        
        # Add Greek text if available
        if verse_key in greek_data:
            grk_data = greek_data[verse_key]
            enriched.greek_text = grk_data['text']
            enriched.greek_words = grk_data['words']
            
            # Add lexicon data for Greek words
            if 'greek' in self.lexicon_cache:
                for word in enriched.greek_words:
                    if word.get('strongs'):
                        strongs = word['strongs'].lstrip('G')
                        if strongs in self.lexicon_cache['greek']:
                            word['lexicon'] = self.lexicon_cache['greek'][strongs]
        
        # Add metadata
        # Determine testament
        ot_books = ['GEN', 'EXO', 'LEV', 'NUM', 'DEU', 'JOS', 'JDG', 'RUT', '1SA', '2SA', '1KI', '2KI', 
                    '1CH', '2CH', 'EZR', 'NEH', 'EST', 'JOB', 'PSA', 'PRO', 'ECC', 'SNG', 'ISA', 'JER', 
                    'LAM', 'EZK', 'DAN', 'HOS', 'JOL', 'AMO', 'OBA', 'JON', 'MIC', 'NAH', 'HAB', 'ZEP', 
                    'HAG', 'ZEC', 'MAL']
        
        is_ot = normalized_book in ot_books
        
        enriched.metadata = {
            'book_name': get_book_name(normalized_book) or book,
            'testament': 'OT' if is_ot else 'NT',
            'canonical_order': self.canon_registry.get_canon('protestant').get_book_order(normalized_book)
        }
        
        # Add cross-references
        for xref in self.cross_refs_cache:
            if xref.source_verse == verse_id:
                enriched.cross_references.append({
                    'target': str(xref.target_verse),
                    'type': xref.reference_type.value,
                    'relationship': xref.relationship.value,
                    'confidence': xref.confidence.overall_score,
                    'topic_tags': xref.topic_tags,
                    'theological_theme': xref.theological_theme
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
        
        return enriched
    
    def process_bible(self, translation_codes: Optional[List[str]] = None,
                     book_filter: Optional[List[str]] = None) -> Dict[str, EnrichedVerse]:
        """Process Bible with all enrichments."""
        logger.info("Starting enhanced Bible processing...")
        
        # Load translations
        translations = self.load_translations(translation_codes)
        if not translations:
            logger.error("No translations loaded")
            return {}
        
        # Load lexicon data
        self.load_lexicon_data()
        
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
            filtered_books = set()
            for book in all_books:
                normalized = normalize_book_name(book) or book.upper()
                if any(normalized == normalize_book_name(f) or book.upper() == f.upper() for f in book_filter):
                    filtered_books.add(book)
            all_books = filtered_books
        
        logger.info(f"Processing {len(all_books)} books...")
        
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
                    
                    # Store in cache
                    verse_key = f"{book}.{chapter_num}.{verse_num}"
                    processed_verses[verse_key] = enriched
        
        logger.info(f"Processed {len(processed_verses)} verses")
        
        # Log enrichment statistics
        verses_with_hebrew = sum(1 for v in processed_verses.values() if v.hebrew_text)
        verses_with_greek = sum(1 for v in processed_verses.values() if v.greek_text)
        verses_with_xrefs = sum(1 for v in processed_verses.values() if v.cross_references)
        verses_with_timeline = sum(1 for v in processed_verses.values() if v.timeline_events)
        
        logger.info(f"Enrichment statistics:")
        logger.info(f"  - Verses with Hebrew text: {verses_with_hebrew}")
        logger.info(f"  - Verses with Greek text: {verses_with_greek}")
        logger.info(f"  - Verses with cross-references: {verses_with_xrefs}")
        logger.info(f"  - Verses with timeline events: {verses_with_timeline}")
        
        return processed_verses


class BibleExporter:
    """Export enriched Bible data to JSON format."""
    
    def export_to_json(self, verses: Dict[str, EnrichedVerse], output_dir: Path) -> None:
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
                'metadata': enriched.metadata
            }
            
            by_book[book][chapter].append(verse_data)
        
        # Write each book
        for book_code, chapters in by_book.items():
            book_data = {
                'book': book_code,
                'name': get_book_name(normalize_book_name(book_code) or book_code),
                'chapters': []
            }
            
            for chapter_num in sorted(chapters.keys()):
                chapter_data = {
                    'chapter': chapter_num,
                    'verses': sorted(chapters[chapter_num], key=lambda v: v['verse'])
                }
                book_data['chapters'].append(chapter_data)
            
            # Write book file
            book_file = output_dir / f"{book_code}.json"
            with open(book_file, 'w', encoding='utf-8') as f:
                json.dump(book_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported to {output_dir}")
        
        # Write summary statistics
        stats_file = output_dir / "_statistics.json"
        verses_with_hebrew = sum(1 for v in verses.values() if v.hebrew_text)
        verses_with_greek = sum(1 for v in verses.values() if v.greek_text)
        verses_with_xrefs = sum(1 for v in verses.values() if v.cross_references)
        verses_with_timeline = sum(1 for v in verses.values() if v.timeline_events)
        
        stats = {
            'total_verses': len(verses),
            'verses_with_hebrew': verses_with_hebrew,
            'verses_with_greek': verses_with_greek,
            'verses_with_cross_references': verses_with_xrefs,
            'verses_with_timeline_events': verses_with_timeline,
            'books_processed': len(by_book),
            'enrichment_coverage': {
                'hebrew_percentage': round(verses_with_hebrew / len(verses) * 100, 2) if verses else 0,
                'greek_percentage': round(verses_with_greek / len(verses) * 100, 2) if verses else 0,
                'cross_reference_percentage': round(verses_with_xrefs / len(verses) * 100, 2) if verses else 0,
                'timeline_percentage': round(verses_with_timeline / len(verses) * 100, 2) if verses else 0,
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ABBA Simple Enhanced Bible Processor - Export enriched Bible data with comprehensive coverage"
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
    processor = SimpleEnhancedBibleProcessor(Path(args.data_dir))
    verses = processor.process_bible(args.translations, args.books)
    
    if not verses:
        logger.error("No verses processed")
        sys.exit(1)
    
    # Export data
    exporter = BibleExporter()
    output_path = Path(args.output)
    exporter.export_to_json(verses, output_path)
    
    logger.info("Export complete!")


if __name__ == '__main__':
    main()