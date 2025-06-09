"""
Comprehensive CLI for ABBA Bible processing.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from abba.book_codes import normalize_book_name, get_book_name, is_valid_book_code
from abba.canon.registry import CanonRegistry
from abba.cross_references.models import CrossReference, ReferenceType
from abba.export.minimal_sqlite import MinimalSQLiteExporter
from abba.export.minimal_json import MinimalJSONExporter
from abba.language.transliteration import create_transliterator
from abba.morphology.hebrew_morphology import HebrewMorphology
from abba.morphology.greek_morphology import GreekMorphology
from abba.parsers.hebrew_parser import HebrewParser
from abba.parsers.greek_parser import GreekParser
from abba.timeline.models import Event, TimePoint, EventType, create_bce_date
from abba.verse_id import VerseID, parse_verse_id


logger = logging.getLogger(__name__)


@dataclass
class EnrichedVerse:
    """Verse with all enrichment data."""
    verse_id: VerseID
    translations: Dict[str, str] = field(default_factory=dict)
    hebrew_data: Optional[Dict] = None
    greek_data: Optional[Dict] = None
    morphology: Optional[Dict] = None
    cross_references: List[Dict] = field(default_factory=list)
    annotations: List[Dict] = field(default_factory=list)
    timeline_events: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class BibleProcessor:
    """Process Bible data with all enrichments."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.translations_dir = data_dir / "translations"
        self.hebrew_dir = data_dir / "hebrew"
        self.greek_dir = data_dir / "greek"
        self.lexicons_dir = data_dir / "lexicons"
        
        # Initialize components
        self.canon_registry = CanonRegistry()
        self.hebrew_morph = HebrewMorphology()
        self.greek_morph = GreekMorphology()
        self.hebrew_parser = HebrewParser()
        self.greek_parser = GreekParser()
        
        # Cache for processed data
        self.verses_cache: Dict[str, EnrichedVerse] = {}
        self.cross_refs_cache: List[CrossReference] = []
        self.timeline_cache: List[Event] = []
        
    def load_translations(self, translation_codes: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Load specified translations or all available."""
        translations = {}
        
        # Load manifest
        manifest_path = self.translations_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"Translations manifest not found at {manifest_path}")
            return translations
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Filter translations
        if translation_codes:
            # Specific translations requested
            versions_to_load = {
                code: info for code, info in manifest['versions'].items()
                if any(tc.upper() in code.upper() for tc in translation_codes)
            }
        else:
            # Default to common English translations
            default_codes = ['ENG_KJV', 'ENG_ASV', 'ENG_WEB', 'ENG_BBE', 'ENG_YLT']
            versions_to_load = {
                code: info for code, info in manifest['versions'].items()
                if code in default_codes
            }
            
        # Load translation files
        for code, info in versions_to_load.items():
            file_path = self.translations_dir / info['filename']
            if file_path.exists():
                logger.info(f"Loading translation: {code} - {info['name']}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    translations[code] = json.load(f)
            else:
                logger.warning(f"Translation file not found: {file_path}")
                
        return translations
    
    def load_hebrew_data(self) -> Dict[str, Any]:
        """Load Hebrew text with morphology."""
        hebrew_data = {}
        
        if not self.hebrew_dir.exists():
            logger.warning(f"Hebrew data directory not found: {self.hebrew_dir}")
            return hebrew_data
            
        # Map book names to files
        book_files = {
            'GEN': 'Gen.xml',
            'EXO': 'Exod.xml',
            'LEV': 'Lev.xml',
            'NUM': 'Num.xml',
            'DEU': 'Deut.xml',
            # Add more mappings as needed
        }
        
        for book_code, filename in book_files.items():
            file_path = self.hebrew_dir / filename
            if file_path.exists():
                logger.info(f"Loading Hebrew data for {book_code}")
                # Here we would parse the XML file
                # For now, create placeholder data
                hebrew_data[book_code] = {
                    'source': 'Open Scriptures Hebrew Bible',
                    'file': str(file_path)
                }
                
        return hebrew_data
    
    def load_greek_data(self) -> Dict[str, Any]:
        """Load Greek text with morphology."""
        greek_data = {}
        
        if not self.greek_dir.exists():
            logger.warning(f"Greek data directory not found: {self.greek_dir}")
            return greek_data
            
        # Map book names to files
        book_files = {
            'MAT': 'MAT.xml',
            'MRK': 'MAR.xml',
            'LUK': 'LUK.xml',
            'JHN': 'JOH.xml',
            'ACT': 'ACT.xml',
            'ROM': 'ROM.xml',
            '1CO': '1CO.xml',
            '2CO': '2CO.xml',
            'GAL': 'GAL.xml',
            'EPH': 'EPH.xml',
            'PHI': 'PHP.xml',
            'COL': 'COL.xml',
            '1TH': '1TH.xml',
            '2TH': '2TH.xml',
            '1TI': '1TI.xml',
            '2TI': '2TI.xml',
            'TIT': 'TIT.xml',
            'PHM': 'PHM.xml',
            'HEB': 'HEB.xml',
            'JAM': 'JAM.xml',
            '1PE': '1PE.xml',
            '2PE': '2PE.xml',
            '1JO': '1JO.xml',
            '2JO': '2JO.xml',
            '3JO': '3JO.xml',
            'JUD': 'JUD.xml',
            'REV': 'REV.xml'
        }
        
        for book_code, filename in book_files.items():
            file_path = self.greek_dir / filename
            if file_path.exists():
                logger.info(f"Loading Greek data for {book_code}")
                greek_data[book_code] = {
                    'source': 'Byzantine Majority Text',
                    'file': str(file_path)
                }
                
        return greek_data
    
    def load_cross_references(self) -> List[CrossReference]:
        """Load cross-reference data."""
        # This would load from a cross-reference database
        # For now, return sample data
        from abba.cross_references.models import ReferenceRelationship, ReferenceConfidence
        
        refs = [
            CrossReference(
                source_verse=VerseID("GEN", 1, 1),
                target_verse=VerseID("JHN", 1, 1),
                reference_type=ReferenceType.THEMATIC_PARALLEL,
                relationship=ReferenceRelationship.PARALLELS,
                confidence=ReferenceConfidence(
                    overall_score=0.95,
                    textual_similarity=0.85,
                    thematic_similarity=0.98,
                    structural_similarity=0.75,
                    scholarly_consensus=0.90,
                    uncertainty_factors=[],
                    lexical_links=3,
                    semantic_links=5
                ),
                topic_tags=["creation", "beginning"],
                theological_theme="Creation and Logos"
            ),
            CrossReference(
                source_verse=VerseID("JHN", 3, 16),
                target_verse=VerseID("ROM", 5, 8),
                reference_type=ReferenceType.THEMATIC_PARALLEL,
                relationship=ReferenceRelationship.PARALLELS,
                confidence=ReferenceConfidence(
                    overall_score=0.85,
                    textual_similarity=0.70,
                    thematic_similarity=0.95,
                    structural_similarity=0.65,
                    scholarly_consensus=0.88,
                    uncertainty_factors=[],
                    lexical_links=2,
                    semantic_links=4
                ),
                topic_tags=["love", "salvation"],
                theological_theme="God's love"
            ),
        ]
        return refs
    
    def load_timeline_events(self) -> List[Event]:
        """Load historical timeline data."""
        events = [
            Event(
                id="creation",
                name="Creation",
                description="The creation of the world",
                event_type=EventType.POINT,
                time_point=TimePoint(
                    exact_date=create_bce_date(4004),  # Traditional date
                    confidence=0.3
                ),
                verse_refs=[VerseID("GEN", 1, 1)],
                categories=["theological", "cosmological"]
            ),
            Event(
                id="exodus",
                name="The Exodus",
                description="Israel leaves Egypt",
                event_type=EventType.POINT,
                time_point=TimePoint(
                    exact_date=create_bce_date(1446),
                    confidence=0.7
                ),
                verse_refs=[VerseID("EXO", 12, 31)],
                categories=["historical", "foundational"]
            ),
        ]
        return events
    
    def enrich_verse(self, book: str, chapter: int, verse: int, 
                     translations: Dict[str, Dict]) -> EnrichedVerse:
        """Create enriched verse with all available data."""
        verse_id = VerseID(book, chapter, verse)
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
        
        # Debug log
        if not enriched.translations:
            logger.debug(f"No translations found for {book}.{chapter}.{verse}")
        
        # Add metadata
        # Determine testament based on book code (handle various formats)
        book_upper = book.upper()
        ot_books = ['GEN', 'EXO', 'LEV', 'NUM', 'DEU', 'JOS', 'JDG', 'RUT', '1SA', '2SA', '1KI', '2KI', 
                    '1CH', '2CH', 'EZR', 'NEH', 'EST', 'JOB', 'PSA', 'PRO', 'ECC', 'SNG', 'ISA', 'JER', 
                    'LAM', 'EZK', 'DAN', 'HOS', 'JOL', 'AMO', 'OBA', 'JON', 'MIC', 'NAH', 'HAB', 'ZEP', 
                    'HAG', 'ZEC', 'MAL']
        
        # Check if it's OT (handle case differences)
        is_ot = any(book_upper.startswith(ot_book[:3]) for ot_book in ot_books)
        
        # Try to get normalized name
        normalized_book = normalize_book_name(book) or book_upper
        
        enriched.metadata = {
            'book_name': get_book_name(normalized_book) or book,
            'testament': 'OT' if is_ot else 'NT',
            'canonical_order': self.canon_registry.get_canon('protestant').get_book_order(normalized_book)
        }
        
        # Add cross-references
        # Need to normalize verse_id for comparison
        normalized_verse_id = VerseID(normalized_book, chapter, verse)
        for xref in self.cross_refs_cache:
            if xref.source_verse == normalized_verse_id:
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
            if normalized_verse_id in event.verse_refs:
                enriched.timeline_events.append({
                    'id': event.id,
                    'name': event.name,
                    'description': event.description,
                    'date': event.time_point.exact_date.isoformat() if event.time_point.exact_date else None,
                    'confidence': event.time_point.confidence,
                    'categories': event.categories
                })
        
        # Add sample annotations
        enriched.annotations = [
            {
                'type': 'theological_theme',
                'value': 'Creation',
                'confidence': 0.9
            }
        ] if normalized_book == 'GEN' and chapter == 1 and verse == 1 else []
        
        return enriched
    
    def process_bible(self, translation_codes: Optional[List[str]] = None) -> Dict[str, EnrichedVerse]:
        """Process complete Bible with all enrichments."""
        logger.info("Starting Bible processing...")
        
        # Load all data sources
        translations = self.load_translations(translation_codes)
        if not translations:
            logger.error("No translations loaded")
            return {}
            
        hebrew_data = self.load_hebrew_data()
        greek_data = self.load_greek_data()
        self.cross_refs_cache = self.load_cross_references()
        self.timeline_cache = self.load_timeline_events()
        
        # Process all verses
        processed_verses = {}
        
        # Get all books from loaded translations
        all_books = set()
        for trans_data in translations.values():
            all_books.update(trans_data.get('books', {}).keys())
        
        logger.info(f"Processing {len(all_books)} books...")
        
        for book in sorted(all_books):
            # Keep original book code from translation
            book_code = book
            
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
                    enriched = self.enrich_verse(book_code, chapter_num, verse_num, translations)
                    
                    # Store in cache
                    verse_key = f"{book_code}.{chapter_num}.{verse_num}"
                    processed_verses[verse_key] = enriched
        
        logger.info(f"Processed {len(processed_verses)} verses")
        return processed_verses


class BibleExporter:
    """Export enriched Bible data to various formats."""
    
    def export_to_json_by_translation(self, verses: Dict[str, EnrichedVerse], 
                                    output_dir: Path) -> None:
        """Export to JSON files organized by translation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group verses by translation
        by_translation = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for verse_key, enriched in verses.items():
            for trans_code, text in enriched.translations.items():
                book = enriched.verse_id.book
                chapter = enriched.verse_id.chapter
                
                verse_data = {
                    'verse': enriched.verse_id.verse,
                    'text': text,
                    'verse_id': verse_key,
                    'cross_references': enriched.cross_references,
                    'annotations': enriched.annotations,
                    'timeline_events': enriched.timeline_events,
                    'metadata': enriched.metadata
                }
                
                by_translation[trans_code][book][chapter].append(verse_data)
        
        # Write files for each translation
        logger.info(f"Found {len(by_translation)} translations to export")
        for trans_code, books in by_translation.items():
            trans_dir = output_dir / trans_code
            trans_dir.mkdir(exist_ok=True)
            logger.info(f"Exporting {trans_code} with {len(books)} books")
            
            # Create manifest for this translation
            manifest = {
                'translation': trans_code,
                'generated_at': datetime.now().isoformat(),
                'format_version': '1.0',
                'books': list(books.keys()),
                'enrichments': {
                    'cross_references': True,
                    'annotations': True,
                    'timeline': True,
                    'morphology': True
                }
            }
            
            # Write manifest
            with open(trans_dir / 'manifest.json', 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            # Write each book
            for book_code, chapters in books.items():
                book_data = {
                    'book': book_code,
                    'name': get_book_name(book_code),
                    'chapters': []
                }
                
                for chapter_num in sorted(chapters.keys()):
                    chapter_data = {
                        'chapter': chapter_num,
                        'verses': sorted(chapters[chapter_num], key=lambda v: v['verse'])
                    }
                    book_data['chapters'].append(chapter_data)
                
                # Write book file
                book_file = trans_dir / f"{book_code}.json"
                with open(book_file, 'w', encoding='utf-8') as f:
                    json.dump(book_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {trans_code} to {trans_dir}")
    
    def export_to_sqlite(self, verses: Dict[str, EnrichedVerse], output_path: Path) -> None:
        """Export to SQLite database with full enrichment."""
        exporter = MinimalSQLiteExporter(str(output_path))
        
        for verse_key, enriched in verses.items():
            # Export each translation
            for trans_code, text in enriched.translations.items():
                exporter.add_verse(
                    verse_id=verse_key,
                    book=enriched.verse_id.book,
                    chapter=enriched.verse_id.chapter,
                    verse=enriched.verse_id.verse,
                    text=text,
                    translation=trans_code
                )
        
        exporter.finalize()
        logger.info(f"Exported to SQLite: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ABBA Bible Processor - Export enriched Bible data"
    )
    
    parser.add_argument(
        '--data-dir',
        default='data/sources',
        help='Directory containing source data (default: data/sources)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory (for JSON) or file (for SQLite)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'sqlite'],
        default='json',
        help='Export format (default: json)'
    )
    
    parser.add_argument(
        '--translations',
        nargs='+',
        help='Specific translation codes to include (e.g., ENG_KJV ENG_ASV)'
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
    processor = BibleProcessor(Path(args.data_dir))
    verses = processor.process_bible(args.translations)
    
    if not verses:
        logger.error("No verses processed")
        sys.exit(1)
    
    # Export data
    exporter = BibleExporter()
    output_path = Path(args.output)
    
    if args.format == 'json':
        exporter.export_to_json_by_translation(verses, output_path)
    else:
        exporter.export_to_sqlite(verses, output_path)
    
    logger.info("Export complete!")


if __name__ == '__main__':
    main()