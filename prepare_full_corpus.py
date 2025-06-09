#!/usr/bin/env python3
"""
Prepare full corpus by sampling verses from all books.
This provides better coverage than using complete books from limited selection.
"""

import subprocess
import sys
import json
from pathlib import Path

# Create a modified corpus builder that samples from all books
corpus_content = '''#!/usr/bin/env python3
"""
Prepare parallel corpus with sampling from all books.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.morphology.oshb_parser import OSHBParser
from src.abba.morphology.sblgnt_parser import SBLGNTParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SamplingParallelCorpusBuilder:
    """Build parallel corpus by sampling from all books."""
    
    def __init__(self, verses_per_book=50):
        self.translations_dir = Path('data/sources/translations')
        self.morphology_dir = Path('data/sources/morphology')
        self.output_dir = Path('data/parallel_corpus')
        self.output_dir.mkdir(exist_ok=True)
        self.verses_per_book = verses_per_book
        
        self.hebrew_parser = OSHBParser(self.morphology_dir / 'hebrew')
        self.greek_parser = SBLGNTParser(self.morphology_dir / 'greek')
        
        self.target_translations = ['eng_kjv']
    
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
        """Prepare Hebrew-English parallel corpus by sampling."""
        logger.info("Preparing Hebrew-English parallel corpus (sampling)...")
        
        source_lines = []
        target_lines = []
        
        ot_books = [
            'Gen', 'Exod', 'Lev', 'Num', 'Deut', 'Josh', 'Judg', 'Ruth',
            '1Sam', '2Sam', '1Kgs', '2Kgs', '1Chr', '2Chr', 'Ezra', 'Neh',
            'Esth', 'Job', 'Ps', 'Prov', 'Eccl', 'Song', 'Isa', 'Jer',
            'Lam', 'Ezek', 'Dan', 'Hos', 'Joel', 'Amos', 'Obad', 'Jonah',
            'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal'
        ]
        
        for book_code in ot_books:
            book_data = self.hebrew_parser.load_book(book_code)
            if not book_data:
                continue
            
            verses = book_data.get('verses', [])
            if not verses:
                continue
            
            # Sample verses
            sample_size = min(self.verses_per_book, len(verses))
            sampled_verses = random.sample(verses, sample_size)
            
            for verse_data in sampled_verses:
                osis_id = verse_data.get('osisID', '')
                parts = osis_id.split('.')
                if len(parts) != 3:
                    continue
                
                chapter = int(parts[1])
                verse = int(parts[2])
                
                hebrew_words = [w['text'] for w in verse_data.get('words', [])]
                if not hebrew_words:
                    continue
                
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
                            english_words = english_text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(';', '').replace(':', '').split()
                            
                            if english_words:
                                source_lines.append(' '.join(hebrew_words))
                                target_lines.append(' '.join(english_words))
                                
                    except Exception as e:
                        logger.error(f"Error processing {trans_id}: {e}")
        
        logger.info(f"Collected {len(source_lines)} Hebrew-English verse pairs")
        return source_lines, target_lines
    
    def prepare_greek_corpus(self) -> Tuple[List[str], List[str]]:
        """Prepare Greek-English parallel corpus by sampling."""
        logger.info("Preparing Greek-English parallel corpus (sampling)...")
        
        source_lines = []
        target_lines = []
        
        nt_books = [
            'Matt', 'Mark', 'Luke', 'John', 'Acts', 'Rom', '1Cor', '2Cor',
            'Gal', 'Eph', 'Phil', 'Col', '1Thess', '2Thess', '1Tim', '2Tim',
            'Titus', 'Phlm', 'Heb', 'Jas', '1Pet', '2Pet', '1John', '2John',
            '3John', 'Jude', 'Rev'
        ]
        
        for book_code in nt_books:
            book_data = self.greek_parser.load_book(book_code)
            if not book_data:
                continue
            
            verses = book_data.get('verses', [])
            if not verses:
                continue
            
            # Sample verses
            sample_size = min(self.verses_per_book, len(verses))
            sampled_verses = random.sample(verses, sample_size)
            
            for verse_data in sampled_verses:
                chapter = verse_data.get('chapter')
                verse = verse_data.get('verse')
                
                greek_words = [w['text'] for w in verse_data.get('words', [])]
                if not greek_words:
                    continue
                
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
                            english_words = english_text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(';', '').replace(':', '').split()
                            
                            if english_words:
                                source_lines.append(' '.join(greek_words))
                                target_lines.append(' '.join(english_words))
                                
                    except Exception as e:
                        logger.error(f"Error processing {trans_id}: {e}")
        
        logger.info(f"Collected {len(source_lines)} Greek-English verse pairs")
        return source_lines, target_lines
    
    def save_corpus(self, source_lines: List[str], target_lines: List[str], 
                   prefix: str) -> None:
        """Save parallel corpus to files."""
        source_file = self.output_dir / f"{prefix}.source"
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(source_lines))
        
        target_file = self.output_dir / f"{prefix}.target"
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(target_lines))
        
        combined_file = self.output_dir / f"{prefix}.parallel"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for src, tgt in zip(source_lines, target_lines):
                f.write(f"{src} ||| {tgt}\\n")
        
        logger.info(f"Saved {len(source_lines)} pairs to {prefix}.*")
    
    def build(self) -> None:
        """Build all parallel corpora."""
        logger.info("Building sampled parallel corpora...")
        
        hebrew_src, hebrew_tgt = self.prepare_hebrew_corpus()
        if hebrew_src:
            self.save_corpus(hebrew_src, hebrew_tgt, "hebrew_english")
        
        greek_src, greek_tgt = self.prepare_greek_corpus()
        if greek_src:
            self.save_corpus(greek_src, greek_tgt, "greek_english")
        
        print("\\nCorpus Statistics:")
        print(f"Hebrew-English pairs: {len(hebrew_src)}")
        print(f"Greek-English pairs: {len(greek_src)}")
        print(f"Total verse pairs: {len(hebrew_src) + len(greek_src)}")


def main():
    builder = SamplingParallelCorpusBuilder(verses_per_book=30)
    builder.build()


if __name__ == "__main__":
    main()
'''

# Write temporary corpus builder
temp_script = Path('prepare_sampled_corpus.py')
temp_script.write_text(corpus_content)

try:
    print("Preparing corpus by sampling from all books...")
    result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("\nTraining statistical models...")
    result = subprocess.run([sys.executable, "scripts/train_statistical_alignment.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("\nCorpus prepared and models trained!")
    print("Run 'python src/main.py' to test the alignment with full Bible coverage.")
    
finally:
    temp_script.unlink()