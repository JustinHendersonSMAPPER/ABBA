#!/usr/bin/env python3
"""
Create corpus files from exported Bible data.

This script consolidates the individual book files into corpus files:
- hebrew_corpus.json: All OT books with Hebrew text
- greek_corpus.json: All NT books with Greek text
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define which books belong to which corpus
HEBREW_BOOKS = {
    'Gen', 'Exod', 'Lev', 'Num', 'Deut', 'Josh', 'Judg', 'Ruth',
    '1Sam', '2Sam', '1Kgs', '2Kgs', '1Chr', '2Chr', 'Ezra', 'Neh',
    'Esth', 'Job', 'Ps', 'Prov', 'Eccl', 'Song', 'Isa', 'Jer',
    'Lam', 'Ezek', 'Dan', 'Hos', 'Joel', 'Amos', 'Obad', 'Jonah',
    'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal'
}

GREEK_BOOKS = {
    'Matt', 'Mark', 'Luke', 'John', 'Acts', 'Rom', '1Cor', '2Cor',
    'Gal', 'Eph', 'Phil', 'Col', '1Thess', '2Thess', '1Tim', '2Tim',
    'Titus', 'Phlm', 'Heb', 'Jas', '1Pet', '2Pet', '1John', '2John',
    '3John', 'Jude', 'Rev'
}


def create_corpus(export_dir: Path, book_set: set, language: str, 
                 text_field: str, version_code: str, corpus_name: str) -> Dict[str, Any]:
    """Create a corpus from a set of books in translation format."""
    # Metadata for the corpus
    corpus = {
        'version': version_code,
        'name': corpus_name,
        'language': language[:2].lower(),  # 'he' for hebrew, 'gr' for greek
        'copyright': 'Public Domain',
        'source': 'OSIS/TEI XML files',
        'books': {}
    }
    
    verse_count = 0
    
    for book_name in sorted(book_set):
        book_file = export_dir / f"{book_name}.json"
        if not book_file.exists():
            logger.warning(f"Book file not found: {book_file}")
            continue
            
        with open(book_file, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        
        book_verses = 0
        chapters_data = []
        
        for chapter in book_data.get('chapters', []):
            chapter_verses = []
            
            for verse in chapter.get('verses', []):
                # Get the text content
                text_content = verse.get(text_field)
                if not text_content:
                    continue
                
                # Create verse entry in translation format
                verse_entry = {
                    'verse': verse['verse'],
                    'text': text_content
                }
                
                chapter_verses.append(verse_entry)
                book_verses += 1
                verse_count += 1
            
            if chapter_verses:
                chapters_data.append({
                    'chapter': chapter['chapter'],
                    'verses': chapter_verses
                })
        
        if chapters_data:
            # Determine testament
            testament = 'OT' if book_name in HEBREW_BOOKS else 'NT'
            
            corpus['books'][book_name] = {
                'name': book_name,  # Could be enhanced with full names
                'abbr': book_name,
                'testament': testament,
                'chapters': chapters_data
            }
            
            logger.info(f"  {book_name}: {book_verses} verses")
    
    logger.info(f"Total {language} verses: {verse_count}")
    return corpus


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create corpus files from Bible export"
    )
    
    parser.add_argument(
        '--export-dir',
        type=Path,
        default=Path('full_fixed_export'),
        help='Directory containing exported Bible books'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/corpora'),
        help='Directory to save corpus files'
    )
    
    args = parser.parse_args()
    
    # Validate export directory
    if not args.export_dir.exists():
        logger.error(f"Export directory not found: {args.export_dir}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Hebrew corpus
    logger.info("Creating Hebrew corpus...")
    hebrew_corpus = create_corpus(
        args.export_dir, 
        HEBREW_BOOKS, 
        'hebrew',
        'hebrew_text',  # Field containing the Hebrew text
        'HEB_BHS',       # Version code
        'Biblia Hebraica Stuttgartensia'  # Full name
    )
    
    # Save Hebrew corpus
    hebrew_file = args.output_dir / 'heb_bhs.json'  # Match translation naming convention
    with open(hebrew_file, 'w', encoding='utf-8') as f:
        json.dump(hebrew_corpus, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {hebrew_file}")
    
    # Create Greek corpus
    logger.info("\nCreating Greek corpus...")
    greek_corpus = create_corpus(
        args.export_dir,
        GREEK_BOOKS,
        'greek', 
        'greek_text',    # Field containing the Greek text
        'GRC_NA28',      # Version code
        'Nestle-Aland 28th Edition'  # Full name
    )
    
    # Save Greek corpus
    greek_file = args.output_dir / 'grc_na28.json'  # Match translation naming convention
    with open(greek_file, 'w', encoding='utf-8') as f:
        json.dump(greek_corpus, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {greek_file}")
    
    logger.info("\nCorpus creation complete!")
    logger.info(f"Files saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())