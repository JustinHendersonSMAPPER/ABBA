#!/usr/bin/env python3
"""
Prepare parallel corpus for statistical alignment training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.morphology.oshb_parser import OSHBParser
from src.abba.morphology.sblgnt_parser import SBLGNTParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelCorpusBuilder:
    """Build parallel corpus from Bible translations."""
    
    def __init__(self):
        self.translations_dir = Path('data/sources/translations')
        self.morphology_dir = Path('data/sources/morphology')
        self.output_dir = Path('data/parallel_corpus')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize parsers
        self.hebrew_parser = OSHBParser(self.morphology_dir / 'hebrew')
        self.greek_parser = SBLGNTParser(self.morphology_dir / 'greek')
        
        # High-quality translations to use (just KJV for speed)
        self.target_translations = [
            'eng_kjv',    # King James Version
        ]
    
    def extract_verse_text(self, translation_data: Dict, book_code: str, 
                          chapter: int, verse: int) -> str:
        """Extract verse text from translation data."""
        if 'books' in translation_data:
            # New format
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
        """Prepare Hebrew-English parallel corpus."""
        logger.info("Preparing Hebrew-English parallel corpus...")
        
        source_lines = []
        target_lines = []
        verse_count = 0
        
        # Process key OT books (Torah + Psalms + Isaiah)
        ot_books = [
            'Gen', 'Exod', 'Lev', 'Num', 'Deut',  # Torah
            'Ps',   # Psalms
            'Isa'   # Isaiah
        ]
        
        for book_code in tqdm(ot_books, desc="Processing OT books"):
            
            # Load Hebrew morphology
            book_data = self.hebrew_parser.load_book(book_code)
            if not book_data:
                continue
            
            # Process all verses
            for verse_data in book_data.get('verses', []):
                # Extract chapter and verse from osisID
                osis_id = verse_data.get('osisID', '')
                parts = osis_id.split('.')
                if len(parts) != 3:
                    continue
                
                chapter = int(parts[1])
                verse = int(parts[2])
                
                # Get Hebrew words
                hebrew_words = [w['text'] for w in verse_data.get('words', [])]
                if not hebrew_words:
                    continue
                
                # Get English translations
                for trans_id in self.target_translations:
                    trans_path = self.translations_dir / f"{trans_id}.json"
                    if not trans_path.exists():
                        continue
                    
                    try:
                        with open(trans_path, 'r', encoding='utf-8') as f:
                            trans_data = json.load(f)
                        
                        english_text = self.extract_verse_text(
                            trans_data, book_code, chapter, verse
                        )
                        
                        if english_text:
                            # Clean and tokenize
                            english_words = english_text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(';', '').replace(':', '').split()
                            
                            if english_words:
                                source_lines.append(' '.join(hebrew_words))
                                target_lines.append(' '.join(english_words))
                                verse_count += 1
                                
                    except Exception as e:
                        logger.error(f"Error processing {trans_id}: {e}")
        
        logger.info(f"Collected {verse_count} Hebrew-English verse pairs")
        return source_lines, target_lines
    
    def prepare_greek_corpus(self) -> Tuple[List[str], List[str]]:
        """Prepare Greek-English parallel corpus."""
        logger.info("Preparing Greek-English parallel corpus...")
        
        source_lines = []
        target_lines = []
        verse_count = 0
        
        # Process key NT books (Gospels + Acts + Romans + Revelation)
        nt_books = [
            'Matt', 'Mark', 'Luke', 'John',  # Gospels
            'Acts',  # Acts
            'Rom',   # Romans
            'Rev'    # Revelation
        ]
        
        for book_code in tqdm(nt_books, desc="Processing NT books"):
            
            # Load Greek morphology
            book_data = self.greek_parser.load_book(book_code)
            if not book_data:
                continue
            
            # Process all verses
            for verse_data in book_data.get('verses', []):
                chapter = verse_data.get('chapter')
                verse = verse_data.get('verse')
                
                # Get Greek words
                greek_words = [w['text'] for w in verse_data.get('words', [])]
                if not greek_words:
                    continue
                
                # Get English translations
                for trans_id in self.target_translations:
                    trans_path = self.translations_dir / f"{trans_id}.json"
                    if not trans_path.exists():
                        continue
                    
                    try:
                        with open(trans_path, 'r', encoding='utf-8') as f:
                            trans_data = json.load(f)
                        
                        english_text = self.extract_verse_text(
                            trans_data, book_code, chapter, verse
                        )
                        
                        if english_text:
                            # Clean and tokenize
                            english_words = english_text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(';', '').replace(':', '').split()
                            
                            if english_words:
                                source_lines.append(' '.join(greek_words))
                                target_lines.append(' '.join(english_words))
                                verse_count += 1
                                
                    except Exception as e:
                        logger.error(f"Error processing {trans_id}: {e}")
        
        logger.info(f"Collected {verse_count} Greek-English verse pairs")
        return source_lines, target_lines
    
    def save_corpus(self, source_lines: List[str], target_lines: List[str], 
                   prefix: str) -> None:
        """Save parallel corpus to files."""
        # Save source
        source_file = self.output_dir / f"{prefix}.source"
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(source_lines))
        logger.info(f"Saved {len(source_lines)} source lines to {source_file}")
        
        # Save target
        target_file = self.output_dir / f"{prefix}.target"
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(target_lines))
        logger.info(f"Saved {len(target_lines)} target lines to {target_file}")
        
        # Save combined for some tools
        combined_file = self.output_dir / f"{prefix}.parallel"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for src, tgt in zip(source_lines, target_lines):
                f.write(f"{src} ||| {tgt}\n")
        logger.info(f"Saved combined corpus to {combined_file}")
    
    def build(self) -> None:
        """Build all parallel corpora."""
        logger.info("Building parallel corpora...")
        
        # Hebrew-English
        hebrew_src, hebrew_tgt = self.prepare_hebrew_corpus()
        if hebrew_src:
            self.save_corpus(hebrew_src, hebrew_tgt, "hebrew_english")
        
        # Greek-English
        greek_src, greek_tgt = self.prepare_greek_corpus()
        if greek_src:
            self.save_corpus(greek_src, greek_tgt, "greek_english")
        
        logger.info("Parallel corpus preparation complete!")
        
        # Print statistics
        print("\nCorpus Statistics:")
        print(f"Hebrew-English pairs: {len(hebrew_src)}")
        print(f"Greek-English pairs: {len(greek_src)}")
        print(f"Total verse pairs: {len(hebrew_src) + len(greek_src)}")


def main():
    """Main function."""
    builder = ParallelCorpusBuilder()
    builder.build()


if __name__ == "__main__":
    main()