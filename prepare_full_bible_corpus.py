#!/usr/bin/env python3
"""
Prepare FULL Bible corpus using ALL verses from ALL books.
This maximizes training data for statistical alignment models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.abba.morphology.oshb_parser import OSHBParser
from src.abba.morphology.sblgnt_parser import SBLGNTParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullBibleCorpusBuilder:
    """Build parallel corpus using ALL verses from ALL books."""
    
    def __init__(self):
        self.translations_dir = Path('data/sources/translations')
        self.morphology_dir = Path('data/sources/morphology')
        self.output_dir = Path('data/parallel_corpus')
        self.output_dir.mkdir(exist_ok=True)
        
        self.hebrew_parser = OSHBParser(self.morphology_dir / 'hebrew')
        self.greek_parser = SBLGNTParser(self.morphology_dir / 'greek')
        
        # Use KJV as primary target translation (most compatible)
        self.target_translations = ['eng_kjv']
    
    def _get_standard_book_code(self, morphology_book_code: str) -> str:
        """Map morphology book codes to standard translation book codes."""
        # Greek morphology to standard book code mapping
        greek_to_standard = {
            'matthew': 'Matt', 'mark': 'Mark', 'luke': 'Luke', 'john': 'John',
            'acts': 'Acts', 'romans': 'Rom', '1corinthians': '1Cor',
            '2corinthians': '2Cor', 'galatians': 'Gal', 'ephesians': 'Eph',
            'philippians': 'Phil', 'colossians': 'Col', '1thessalonians': '1Thess',
            '2thessalonians': '2Thess', '1timothy': '1Tim', '2timothy': '2Tim',
            'titus': 'Titus', 'philemon': 'Phlm', 'hebrews': 'Heb',
            'james': 'Jas', '1peter': '1Pet', '2peter': '2Pet',
            '1john': '1John', '2john': '2John', '3john': '3John',
            'jude': 'Jude', 'revelation': 'Rev'
        }
        return greek_to_standard.get(morphology_book_code, morphology_book_code)
    
    def extract_verse_text(self, translation_data: Dict, book_code: str, 
                          chapter: int, verse: int) -> str:
        """Extract verse text from translation data."""
        if 'books' in translation_data:
            if book_code in translation_data['books']:
                book_data = translation_data['books'][book_code]
                if 'chapters' in book_data:
                    for ch in book_data['chapters']:
                        if ch.get('chapter') == chapter:
                            for v in ch.get('verses', []):
                                if v.get('verse') == verse:
                                    return v.get('text', '').strip()
        return ""
    
    def prepare_hebrew_corpus(self) -> Tuple[List[str], List[str]]:
        """Prepare Hebrew-English parallel corpus using ALL OT verses."""
        logger.info("Preparing Hebrew-English parallel corpus (FULL Bible)...")
        
        source_lines = []
        target_lines = []
        
        # All Old Testament books
        ot_books = [
            'Gen', 'Exod', 'Lev', 'Num', 'Deut', 'Josh', 'Judg', 'Ruth',
            '1Sam', '2Sam', '1Kgs', '2Kgs', '1Chr', '2Chr', 'Ezra', 'Neh',
            'Esth', 'Job', 'Ps', 'Prov', 'Eccl', 'Song', 'Isa', 'Jer',
            'Lam', 'Ezek', 'Dan', 'Hos', 'Joel', 'Amos', 'Obad', 'Jonah',
            'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal'
        ]
        
        # Load target translation once
        target_translation = None
        for trans_id in self.target_translations:
            trans_path = self.translations_dir / f"{trans_id}.json"
            if trans_path.exists():
                try:
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        target_translation = json.load(f)
                    break
                except Exception as e:
                    logger.error(f"Error loading {trans_id}: {e}")
        
        if not target_translation:
            logger.error("No target translation found!")
            return [], []
        
        total_verses_processed = 0
        for book_code in ot_books:
            logger.info(f"Processing {book_code}...")
            book_data = self.hebrew_parser.load_book(book_code)
            if not book_data:
                logger.warning(f"No Hebrew data for {book_code}")
                continue
            
            verses = book_data.get('verses', [])
            logger.info(f"  Found {len(verses)} verses in {book_code}")
            
            for verse_data in verses:
                osis_id = verse_data.get('osisID', '')
                parts = osis_id.split('.')
                if len(parts) != 3:
                    continue
                
                chapter = int(parts[1])
                verse = int(parts[2])
                
                hebrew_words = [w['text'] for w in verse_data.get('words', [])]
                if not hebrew_words:
                    continue
                
                english_text = self.extract_verse_text(
                    target_translation, book_code, chapter, verse
                )
                
                if english_text:
                    # Clean English text
                    english_words = english_text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(';', '').replace(':', '').split()
                    
                    if english_words:
                        source_lines.append(' '.join(hebrew_words))
                        target_lines.append(' '.join(english_words))
                        total_verses_processed += 1
            
            logger.info(f"  Added {total_verses_processed} verses so far")
        
        logger.info(f"Collected {len(source_lines)} Hebrew-English verse pairs")
        return source_lines, target_lines
    
    def prepare_greek_corpus(self) -> Tuple[List[str], List[str]]:
        """Prepare Greek-English parallel corpus using ALL NT verses."""
        logger.info("Preparing Greek-English parallel corpus (FULL Bible)...")
        
        source_lines = []
        target_lines = []
        
        # All New Testament books (using lowercase names as they appear in Greek morphology files)
        nt_books = [
            'matthew', 'mark', 'luke', 'john', 'acts', 'romans', 
            '1corinthians', '2corinthians', 'galatians', 'ephesians', 
            'philippians', 'colossians', '1thessalonians', '2thessalonians',
            '1timothy', '2timothy', 'titus', 'philemon', 'hebrews',
            'james', '1peter', '2peter', '1john', '2john', '3john',
            'jude', 'revelation'
        ]
        
        # Load target translation once
        target_translation = None
        for trans_id in self.target_translations:
            trans_path = self.translations_dir / f"{trans_id}.json"
            if trans_path.exists():
                try:
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        target_translation = json.load(f)
                    break
                except Exception as e:
                    logger.error(f"Error loading {trans_id}: {e}")
        
        if not target_translation:
            logger.error("No target translation found!")
            return [], []
        
        total_verses_processed = 0
        for book_code in nt_books:
            logger.info(f"Processing {book_code}...")
            book_data = self.greek_parser.load_book(book_code)
            if not book_data:
                logger.warning(f"No Greek data for {book_code}")
                continue
            
            verses = book_data.get('verses', [])
            logger.info(f"  Found {len(verses)} verses in {book_code}")
            
            for verse_data in verses:
                chapter = verse_data.get('chapter')
                verse = verse_data.get('verse')
                
                greek_words = [w['text'] for w in verse_data.get('words', [])]
                if not greek_words:
                    continue
                
                # Map Greek morphology book name to standard book code for translation lookup
                standard_book_code = self._get_standard_book_code(book_code)
                english_text = self.extract_verse_text(
                    target_translation, standard_book_code, chapter, verse
                )
                
                if english_text:
                    # Clean English text
                    english_words = english_text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(';', '').replace(':', '').split()
                    
                    if english_words:
                        source_lines.append(' '.join(greek_words))
                        target_lines.append(' '.join(english_words))
                        total_verses_processed += 1
            
            logger.info(f"  Added {total_verses_processed} verses so far")
        
        logger.info(f"Collected {len(source_lines)} Greek-English verse pairs")
        return source_lines, target_lines
    
    def save_corpus(self, source_lines: List[str], target_lines: List[str], 
                   prefix: str) -> None:
        """Save parallel corpus to files."""
        source_file = self.output_dir / f"{prefix}.source"
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(source_lines))
        
        target_file = self.output_dir / f"{prefix}.target"
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(target_lines))
        
        combined_file = self.output_dir / f"{prefix}.parallel"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for src, tgt in zip(source_lines, target_lines):
                f.write(f"{src} ||| {tgt}\n")
        
        logger.info(f"Saved {len(source_lines)} pairs to {prefix}.*")
    
    def build(self) -> None:
        """Build FULL Bible parallel corpora."""
        logger.info("Building FULL Bible parallel corpora...")
        
        hebrew_src, hebrew_tgt = self.prepare_hebrew_corpus()
        if hebrew_src:
            self.save_corpus(hebrew_src, hebrew_tgt, "hebrew_english")
        
        greek_src, greek_tgt = self.prepare_greek_corpus()
        if greek_src:
            self.save_corpus(greek_src, greek_tgt, "greek_english")
        
        print("\nFULL BIBLE Corpus Statistics:")
        print(f"Hebrew-English pairs: {len(hebrew_src):,}")
        print(f"Greek-English pairs: {len(greek_src):,}")
        print(f"Total verse pairs: {len(hebrew_src) + len(greek_src):,}")
        print("\nThis represents the complete Bible corpus for statistical training!")


def main():
    builder = FullBibleCorpusBuilder()
    builder.build()


if __name__ == "__main__":
    main()