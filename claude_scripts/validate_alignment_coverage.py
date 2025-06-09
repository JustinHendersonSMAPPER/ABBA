#!/usr/bin/env python3
"""
Validate alignment model coverage against a translation.

This script calculates what percentage of words in a translation
can be aligned back to the original language using the trained model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AlignmentCoverageValidator:
    """Validate alignment model coverage on translations."""
    
    def __init__(self, model_path: Path):
        """Initialize with trained alignment model."""
        self.model_path = model_path
        self.model = self._load_model()
        self.strongs_mappings = self._load_strongs_mappings()
        self.coverage_stats = {
            'total_words': 0,
            'unique_words': 0,
            'covered_words': 0,
            'covered_unique': 0,
            'uncovered_words': Counter(),
            'coverage_by_frequency': defaultdict(int),
            'coverage_by_type': defaultdict(lambda: {'total': 0, 'covered': 0})
        }
        
    def _load_model(self) -> Dict:
        """Load the trained alignment model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        with open(self.model_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _load_strongs_mappings(self) -> Dict[str, Set[str]]:
        """Extract Strong's to English mappings from model."""
        # This would load the actual mappings from the model
        # For now, return a placeholder
        return self.model.get('strongs_mappings', {})
        
    def validate_translation(self, translation_path: Path) -> Dict:
        """Validate coverage on a translation file."""
        logger.info(f"Validating coverage on {translation_path}")
        
        if not translation_path.exists():
            raise FileNotFoundError(f"Translation not found: {translation_path}")
            
        with open(translation_path, 'r', encoding='utf-8') as f:
            translation_data = json.load(f)
            
        # Process each verse
        word_counts = Counter()
        aligned_words = set()
        verse_coverage = []
        
        for book in translation_data.get('books', []):
            book_name = book.get('name', 'Unknown')
            
            for chapter in book.get('chapters', []):
                chapter_num = chapter.get('number', 0)
                
                for verse in chapter.get('verses', []):
                    verse_num = verse.get('number', 0)
                    verse_id = f"{book_name}.{chapter_num}.{verse_num}"
                    
                    # Get verse text and words
                    text = verse.get('text', '')
                    words = self._tokenize(text)
                    
                    # Check alignment coverage
                    covered = 0
                    uncovered = []
                    
                    for word in words:
                        word_lower = word.lower()
                        word_counts[word_lower] += 1
                        
                        if self._is_aligned(word_lower, verse):
                            covered += 1
                            aligned_words.add(word_lower)
                        else:
                            uncovered.append(word)
                            
                    # Calculate verse-level coverage
                    coverage = (covered / len(words) * 100) if words else 0
                    verse_coverage.append({
                        'verse_id': verse_id,
                        'total_words': len(words),
                        'covered': covered,
                        'coverage': coverage,
                        'uncovered_words': uncovered
                    })
                    
        # Calculate overall statistics
        self._calculate_statistics(word_counts, aligned_words, verse_coverage)
        
        return self.coverage_stats
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Remove punctuation and split
        import re
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) > 0]
        
    def _is_aligned(self, word: str, verse_data: Dict) -> bool:
        """Check if a word can be aligned to source language."""
        # Check if word appears in any Strong's mapping
        if hasattr(self, 'strongs_mappings'):
            for strongs_num, english_words in self.strongs_mappings.items():
                if word in english_words:
                    return True
                    
        # Check if word has direct alignment in verse
        alignments = verse_data.get('alignments', [])
        for alignment in alignments:
            if word in alignment.get('target', '').lower():
                return True
                
        return False
        
    def _calculate_statistics(self, word_counts: Counter, 
                            aligned_words: Set[str],
                            verse_coverage: List[Dict]):
        """Calculate comprehensive coverage statistics."""
        # Basic counts
        total_words = sum(word_counts.values())
        unique_words = len(word_counts)
        covered_unique = len(aligned_words)
        
        # Coverage by frequency
        covered_tokens = 0
        for word, count in word_counts.items():
            if word in aligned_words:
                covered_tokens += count
                self.coverage_stats['coverage_by_frequency'][count] += 1
            else:
                self.coverage_stats['uncovered_words'][word] = count
                
        # Overall coverage
        self.coverage_stats.update({
            'total_words': total_words,
            'unique_words': unique_words,
            'covered_words': covered_tokens,
            'covered_unique': covered_unique,
            'token_coverage': (covered_tokens / total_words * 100) if total_words > 0 else 0,
            'type_coverage': (covered_unique / unique_words * 100) if unique_words > 0 else 0,
            'verse_coverage': verse_coverage
        })
        
        # Coverage by word type (frequent vs rare)
        for word, count in word_counts.items():
            if count >= 100:
                word_type = 'high_frequency'
            elif count >= 10:
                word_type = 'medium_frequency'
            else:
                word_type = 'low_frequency'
                
            self.coverage_stats['coverage_by_type'][word_type]['total'] += count
            if word in aligned_words:
                self.coverage_stats['coverage_by_type'][word_type]['covered'] += count
                
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a detailed coverage report."""
        report = []
        report.append("=" * 60)
        report.append("ALIGNMENT COVERAGE VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall coverage
        report.append("OVERALL COVERAGE")
        report.append("-" * 30)
        report.append(f"Total words (tokens): {self.coverage_stats['total_words']:,}")
        report.append(f"Unique words (types): {self.coverage_stats['unique_words']:,}")
        report.append(f"Covered words (tokens): {self.coverage_stats['covered_words']:,}")
        report.append(f"Covered unique (types): {self.coverage_stats['covered_unique']:,}")
        report.append("")
        report.append(f"TOKEN COVERAGE: {self.coverage_stats['token_coverage']:.2f}%")
        report.append(f"TYPE COVERAGE: {self.coverage_stats['type_coverage']:.2f}%")
        report.append("")
        
        # Coverage by frequency
        report.append("COVERAGE BY WORD FREQUENCY")
        report.append("-" * 30)
        for word_type, stats in sorted(self.coverage_stats['coverage_by_type'].items()):
            total = stats['total']
            covered = stats['covered']
            percentage = (covered / total * 100) if total > 0 else 0
            report.append(f"{word_type}: {covered:,}/{total:,} ({percentage:.2f}%)")
        report.append("")
        
        # Most common uncovered words
        report.append("TOP 20 UNCOVERED WORDS")
        report.append("-" * 30)
        uncovered = self.coverage_stats['uncovered_words']
        for word, count in uncovered.most_common(20):
            report.append(f"  {word}: {count:,} occurrences")
        report.append("")
        
        # Verse-level statistics
        verse_coverages = [v['coverage'] for v in self.coverage_stats['verse_coverage']]
        if verse_coverages:
            report.append("VERSE-LEVEL STATISTICS")
            report.append("-" * 30)
            report.append(f"Average verse coverage: {sum(verse_coverages)/len(verse_coverages):.2f}%")
            report.append(f"Verses with 100% coverage: {sum(1 for c in verse_coverages if c == 100)}")
            report.append(f"Verses with >90% coverage: {sum(1 for c in verse_coverages if c >= 90)}")
            report.append(f"Verses with >80% coverage: {sum(1 for c in verse_coverages if c >= 80)}")
            report.append(f"Verses with <50% coverage: {sum(1 for c in verse_coverages if c < 50)}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save report if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
            
        return report_text
        

def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate alignment model coverage on translations"
    )
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained alignment model'
    )
    parser.add_argument(
        '--translation',
        type=Path,
        required=True,
        help='Path to translation JSON file'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Path to save detailed report'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=80.0,
        help='Minimum acceptable coverage percentage (default: 80%)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Validate coverage
    validator = AlignmentCoverageValidator(args.model)
    stats = validator.validate_translation(args.translation)
    
    # Generate report
    report = validator.generate_report(args.report)
    
    if args.verbose:
        print(report)
    else:
        print(f"\nToken Coverage: {stats['token_coverage']:.2f}%")
        print(f"Type Coverage: {stats['type_coverage']:.2f}%")
        
    # Check if coverage meets minimum
    if stats['token_coverage'] < args.min_coverage:
        logger.warning(f"Coverage {stats['token_coverage']:.2f}% is below minimum {args.min_coverage}%")
        exit(1)
    else:
        logger.info(f"Coverage {stats['token_coverage']:.2f}% meets minimum requirement")
        

if __name__ == '__main__':
    main()