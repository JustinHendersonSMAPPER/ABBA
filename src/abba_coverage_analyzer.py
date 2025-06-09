"""
ABBA Coverage Analyzer Module

Analyzes translation coverage using alignment models and extracted JSON translations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger('ABBA.CoverageAnalyzer')


class ABBACoverageAnalyzer:
    """Analyzes biblical translation coverage using alignment models."""
    
    def __init__(self, aligner=None):
        self.translations_dir = Path('data/sources/translations')
        self.morphology_dir = Path('data/sources/morphology')
        self.aligner = aligner
        self.confidence_threshold = 0.0  # Accept all alignments for coverage analysis
        
    def get_available_translations(self) -> List[Path]:
        """Get list of available translation JSON files."""
        return list(self.translations_dir.glob('*.json'))
    
    def analyze_translation(self, translation_path: Path) -> Optional[Dict]:
        """Analyze coverage for a single translation."""
        if not self.aligner:
            logger.warning("No aligner provided - cannot analyze coverage")
            return None
            
        translation_id = translation_path.stem
        logger.info(f"Analyzing {translation_id}")
        
        # Load translation data
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                translation_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {translation_path}: {e}")
            return None
        
        # Extract books data
        books_data = []
        if 'books' in translation_data:
            # New format: {version, name, language, books: {bookName: bookData}}
            for book_code, book_data in translation_data['books'].items():
                book_data['book_code'] = book_code
                books_data.append(book_data)
        elif isinstance(translation_data, list):
            # Old format: array of books
            books_data = translation_data
        else:
            logger.warning(f"Unknown translation format for {translation_id}")
            return None
        
        # Count books and verses
        book_count = len(books_data)
        verse_count = 0
        
        # Separate OT and NT books
        ot_books = []
        nt_books = []
        
        # Book codes for testament determination
        ot_book_codes = {
            'Gen', 'Exod', 'Lev', 'Num', 'Deut', 'Josh', 'Judg', 'Ruth', '1Sam', '2Sam',
            '1Kgs', '2Kgs', '1Chr', '2Chr', 'Ezra', 'Neh', 'Esth', 'Job', 'Ps', 'Prov',
            'Eccl', 'Song', 'Isa', 'Jer', 'Lam', 'Ezek', 'Dan', 'Hos', 'Joel', 'Amos',
            'Obad', 'Jonah', 'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal'
        }
        
        for book in books_data:
            # Get book code from either field
            book_code = book.get('book_code') or book.get('abbr') or book.get('book', '')
            
            # Count verses in this book
            for chapter in book.get('chapters', []):
                verse_count += len(chapter.get('verses', []))
            
            # Categorize by testament using book code
            if book_code in ot_book_codes:
                ot_books.append(book)
            else:
                nt_books.append(book)
        
        # Analyze Hebrew (OT) coverage
        hebrew_stats = self._analyze_books_coverage(ot_books, 'hebrew')
        
        # Analyze Greek (NT) coverage  
        greek_stats = self._analyze_books_coverage(nt_books, 'greek')
        
        # Calculate overall coverage
        total_tokens = hebrew_stats['total_tokens'] + greek_stats['total_tokens']
        aligned_tokens = hebrew_stats['aligned_tokens'] + greek_stats['aligned_tokens']
        
        overall_coverage = (aligned_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        return {
            'translation_id': translation_id,
            'book_count': book_count,
            'verse_count': verse_count,
            'overall_coverage': overall_coverage,
            'hebrew_coverage': hebrew_stats['alignment_coverage'] if hebrew_stats['total_tokens'] > 0 else None,
            'greek_coverage': greek_stats['alignment_coverage'] if greek_stats['total_tokens'] > 0 else None,
            'hebrew_stats': hebrew_stats,
            'greek_stats': greek_stats,
            'has_hebrew': len(ot_books) > 0,
            'has_greek': len(nt_books) > 0
        }
    
    def _analyze_books_coverage(self, books: List[Dict], language: str) -> Dict:
        """Analyze coverage for a set of books using the aligner."""
        if not books:
            return {
                'total_tokens': 0,
                'aligned_tokens': 0,
                'total_verses': 0,
                'aligned_verses': 0,
                'alignment_coverage': 0,
                'verse_coverage': 0
            }
        
        total_tokens = 0
        aligned_tokens = 0
        total_verses = 0
        aligned_verses = 0
        
        # Sample first few verses from first book for efficiency
        sample_book = books[0]
        book_code = sample_book.get('book_code', '')
        
        # Try to load morphological data for this book
        morph_path = self.morphology_dir / language / f"{book_code}.json"
        if not morph_path.exists():
            logger.debug(f"No morphological data for {book_code} in {language}")
            return {
                'total_tokens': 0,
                'aligned_tokens': 0,
                'total_verses': 0,
                'aligned_verses': 0,
                'alignment_coverage': 0,
                'verse_coverage': 0
            }
        
        try:
            with open(morph_path, 'r', encoding='utf-8') as f:
                morph_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load morphological data: {e}")
            return {
                'total_tokens': 0,
                'aligned_tokens': 0,
                'total_verses': 0,
                'aligned_verses': 0,
                'alignment_coverage': 0,
                'verse_coverage': 0
            }
        
        # Sample first 10 verses
        sample_size = min(10, len(sample_book.get('chapters', [{}])[0].get('verses', [])))
        
        for i in range(sample_size):
            # Get translation verse
            if 'chapters' in sample_book and sample_book['chapters']:
                chapter = sample_book['chapters'][0]
                if 'verses' in chapter and i < len(chapter['verses']):
                    verse = chapter['verses'][i]
                    verse_text = verse.get('text', '')
                    
                    # Get source language verse
                    if 'verses' in morph_data and i < len(morph_data['verses']):
                        source_verse = morph_data['verses'][i]
                        source_words = [w['text'] for w in source_verse.get('words', [])]
                        
                        # Simple word tokenization for target
                        target_words = verse_text.strip().replace('.', '').replace(',', '').split()
                        
                        if source_words and target_words:
                            # Perform alignment
                            alignments = self.aligner.align_verse(
                                source_words, target_words, language, 'english'
                            )
                            
                            total_verses += 1
                            total_tokens += len(source_words)
                            
                            # Count aligned tokens (with any confidence)
                            aligned_in_verse = len([a for a in alignments if a['confidence'] > self.confidence_threshold])
                            aligned_tokens += aligned_in_verse
                            
                            if aligned_in_verse > 0:
                                aligned_verses += 1
        
        # Extrapolate from sample to full book set
        if sample_size > 0:
            # Count total verses across all books
            total_verses_all = sum(
                len(verse)
                for book in books
                for chapter in book.get('chapters', [])
                for verse in chapter.get('verses', [])
            )
            
            # Scale up from sample
            scale_factor = total_verses_all / sample_size
            total_tokens = int(total_tokens * scale_factor)
            aligned_tokens = int(aligned_tokens * scale_factor)
            total_verses = int(total_verses * scale_factor)
            aligned_verses = int(aligned_verses * scale_factor)
        
        alignment_coverage = (aligned_tokens / total_tokens * 100) if total_tokens > 0 else 0
        verse_coverage = (aligned_verses / total_verses * 100) if total_verses > 0 else 0
        
        return {
            'total_tokens': total_tokens,
            'aligned_tokens': aligned_tokens,
            'total_verses': total_verses,
            'aligned_verses': aligned_verses,
            'alignment_coverage': alignment_coverage,
            'verse_coverage': verse_coverage
        }
    
    def analyze_all_translations(self, sample_size: int = 10) -> List[Dict]:
        """Analyze coverage for all available translations."""
        results = []
        translations = self.get_available_translations()[:sample_size]
        
        for translation_path in translations:
            result = self.analyze_translation(translation_path)
            if result:
                results.append(result)
        
        return results
    
    def generate_coverage_report(self, results: List[Dict]) -> str:
        """Generate a coverage report from analysis results."""
        if not results:
            return "No results to report"
        
        report_lines = [
            "Translation Coverage Analysis Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Translations analyzed: {len(results)}",
            f"Aligner: {self.aligner.__class__.__name__ if self.aligner else 'None'}",
            "",
        ]
        
        # Sort by overall coverage
        sorted_results = sorted(results, key=lambda x: x['overall_coverage'], reverse=True)
        
        # Summary statistics
        avg_coverage = sum(r['overall_coverage'] for r in results) / len(results)
        
        # Calculate Hebrew average only for translations that have Hebrew
        hebrew_results = [r for r in results if r['hebrew_coverage'] is not None]
        avg_hebrew = sum(r['hebrew_coverage'] for r in hebrew_results) / len(hebrew_results) if hebrew_results else 0
        
        # Calculate Greek average only for translations that have Greek  
        greek_results = [r for r in results if r['greek_coverage'] is not None]
        avg_greek = sum(r['greek_coverage'] for r in greek_results) / len(greek_results) if greek_results else 0
        
        report_lines.extend([
            "Summary Statistics",
            "-" * 30,
            f"Average overall coverage: {avg_coverage:.1f}%",
            f"Average Hebrew coverage: {avg_hebrew:.1f}% ({len(hebrew_results)} translations with OT)",
            f"Average Greek coverage: {avg_greek:.1f}% ({len(greek_results)} translations with NT)",
            "",
            "Top 10 Translations by Coverage",
            "-" * 30,
        ])
        
        # Top translations
        for result in sorted_results[:10]:
            hebrew_str = f"{result['hebrew_coverage']:5.1f}%" if result['hebrew_coverage'] is not None else "  N/A"
            greek_str = f"{result['greek_coverage']:5.1f}%" if result['greek_coverage'] is not None else "  N/A"
            report_lines.append(
                f"{result['translation_id']:20s} - Overall: {result['overall_coverage']:5.1f}% "
                f"(Hebrew: {hebrew_str}, Greek: {greek_str})"
            )
        
        return "\n".join(report_lines)