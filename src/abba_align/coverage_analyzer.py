"""
Coverage analysis for alignment models.

Provides detailed analysis of how well an alignment model covers
the vocabulary of a target translation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
import numpy as np

from .lexicon_integration import StrongsLexiconIntegration

logger = logging.getLogger(__name__)


class AlignmentCoverageAnalyzer:
    """Analyze coverage of alignment models on translations."""
    
    def __init__(self, source_lang: str = 'hebrew'):
        """Initialize coverage analyzer."""
        self.source_lang = source_lang
        self.strongs_integration = StrongsLexiconIntegration()
        self._load_lexicon()
        
    def _load_lexicon(self):
        """Load Strong's lexicon for coverage analysis."""
        lexicon_path = Path(f'data/sources/lexicons/strongs_{self.source_lang}.json')
        if lexicon_path.exists():
            self.strongs_integration.load_strongs_lexicon(lexicon_path, self.source_lang)
            logger.debug(f"Loaded base lexicon with {len(self.strongs_integration.strongs_to_english)} extracted glosses")
        else:
            logger.warning(f"Strong's lexicon not found at {lexicon_path}")
            
    def analyze_translation_coverage(self, translation_path: Path, 
                                   alignment_model_path: Optional[Path] = None) -> Dict:
        """
        Analyze how well the alignment model covers a translation.
        
        Returns detailed statistics about coverage including:
        - Token coverage (% of word occurrences that can be aligned)
        - Type coverage (% of unique words that can be aligned)
        - Coverage by frequency band
        - Coverage by part of speech
        - List of uncovered high-frequency words
        """
        logger.info(f"Analyzing coverage for {translation_path}")
        
        # Load translation
        with open(translation_path, 'r', encoding='utf-8') as f:
            translation_data = json.load(f)
            
        # Initialize counters
        word_counts = Counter()
        pos_counts = defaultdict(Counter)
        aligned_words = set()
        aligned_by_pos = defaultdict(set)
        verse_stats = []
        
        # Process each verse
        for book in translation_data.get('books', []):
            for chapter in book.get('chapters', []):
                for verse in chapter.get('verses', []):
                    verse_result = self._analyze_verse(
                        verse, word_counts, pos_counts, 
                        aligned_words, aligned_by_pos
                    )
                    verse_stats.append(verse_result)
                    
        # Calculate overall statistics
        stats = self._calculate_statistics(
            word_counts, pos_counts, aligned_words, 
            aligned_by_pos, verse_stats
        )
        
        return stats
        
    def _analyze_verse(self, verse: Dict, word_counts: Counter,
                      pos_counts: Dict, aligned_words: Set,
                      aligned_by_pos: Dict) -> Dict:
        """Analyze coverage for a single verse."""
        text = verse.get('text', '')
        words = self._tokenize(text)
        
        covered = 0
        uncovered_words = []
        
        # Check if verse has source language data
        source_words = verse.get('source_words', [])
        alignments = verse.get('alignments', [])
        
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] += 1
            
            # Determine POS if available
            pos = self._guess_pos(word)
            pos_counts[pos][word_lower] += 1
            
            # Check if word can be aligned
            if self._can_align(word_lower, source_words, alignments):
                covered += 1
                aligned_words.add(word_lower)
                aligned_by_pos[pos].add(word_lower)
            else:
                uncovered_words.append(word)
                
        coverage = (covered / len(words) * 100) if words else 0
        
        return {
            'total_words': len(words),
            'covered': covered,
            'coverage': coverage,
            'uncovered': uncovered_words
        }
        
    def _can_align(self, word: str, source_words: List[Dict], 
                   alignments: List[Dict]) -> bool:
        """Check if a word can be aligned to source language."""
        # Method 1: Check Strong's mappings
        for strongs_num, english_words in self.strongs_integration.strongs_to_english.items():
            if word in english_words:
                # Verify this Strong's number appears in source
                for src_word in source_words:
                    if src_word.get('strongs') == strongs_num:
                        return True
                        
        # Method 2: Check explicit alignments
        for alignment in alignments:
            target_text = alignment.get('target', '').lower()
            if word in target_text.split():
                return True
                
        # Method 3: Check common words that don't need alignment
        if word in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                   'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}:
            return True  # These are often implied in Hebrew/Greek
            
        return False
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        import re
        # Simple word tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words
        
    def _guess_pos(self, word: str) -> str:
        """Simple POS guessing based on word patterns."""
        word_lower = word.lower()
        
        # Common patterns
        if word_lower.endswith('ing'):
            return 'verb'
        elif word_lower.endswith('ed'):
            return 'verb'
        elif word_lower.endswith('ly'):
            return 'adverb'
        elif word_lower.endswith('ness') or word_lower.endswith('tion'):
            return 'noun'
        elif word[0].isupper() and word not in {'I', 'The', 'A', 'An'}:
            return 'proper_noun'
        elif word_lower in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
            return 'determiner'
        elif word_lower in {'he', 'she', 'it', 'they', 'we', 'you', 'i'}:
            return 'pronoun'
        else:
            return 'other'
            
    def _calculate_statistics(self, word_counts: Counter, pos_counts: Dict,
                            aligned_words: Set, aligned_by_pos: Dict,
                            verse_stats: List[Dict]) -> Dict:
        """Calculate comprehensive coverage statistics."""
        # Overall token and type coverage
        total_tokens = sum(word_counts.values())
        total_types = len(word_counts)
        covered_tokens = sum(count for word, count in word_counts.items() 
                           if word in aligned_words)
        covered_types = len(aligned_words)
        
        # Coverage by frequency band
        freq_bands = {
            'high_freq': (100, float('inf')),  # 100+ occurrences
            'medium_freq': (10, 99),           # 10-99 occurrences
            'low_freq': (2, 9),                # 2-9 occurrences
            'hapax': (1, 1)                    # Single occurrence
        }
        
        coverage_by_freq = {}
        for band_name, (min_freq, max_freq) in freq_bands.items():
            band_words = [w for w, c in word_counts.items() 
                         if min_freq <= c <= max_freq]
            band_covered = [w for w in band_words if w in aligned_words]
            
            band_tokens = sum(word_counts[w] for w in band_words)
            covered_band_tokens = sum(word_counts[w] for w in band_covered)
            
            coverage_by_freq[band_name] = {
                'total_types': len(band_words),
                'covered_types': len(band_covered),
                'total_tokens': band_tokens,
                'covered_tokens': covered_band_tokens,
                'type_coverage': (len(band_covered) / len(band_words) * 100) if band_words else 0,
                'token_coverage': (covered_band_tokens / band_tokens * 100) if band_tokens else 0
            }
            
        # Coverage by POS
        coverage_by_pos = {}
        for pos, words in pos_counts.items():
            pos_total = sum(words.values())
            pos_covered = sum(count for word, count in words.items() 
                            if word in aligned_by_pos[pos])
            coverage_by_pos[pos] = {
                'total': pos_total,
                'covered': pos_covered,
                'coverage': (pos_covered / pos_total * 100) if pos_total else 0
            }
            
        # Verse-level statistics
        verse_coverages = [v['coverage'] for v in verse_stats]
        
        # Find most common uncovered words
        uncovered_words = Counter()
        for word, count in word_counts.items():
            if word not in aligned_words:
                uncovered_words[word] = count
                
        return {
            'summary': {
                'total_tokens': total_tokens,
                'total_types': total_types,
                'covered_tokens': covered_tokens,
                'covered_types': covered_types,
                'token_coverage': (covered_tokens / total_tokens * 100) if total_tokens else 0,
                'type_coverage': (covered_types / total_types * 100) if total_types else 0
            },
            'coverage_by_frequency': coverage_by_freq,
            'coverage_by_pos': coverage_by_pos,
            'verse_statistics': {
                'total_verses': len(verse_stats),
                'average_coverage': np.mean(verse_coverages) if verse_coverages else 0,
                'std_coverage': np.std(verse_coverages) if verse_coverages else 0,
                'perfect_coverage': sum(1 for c in verse_coverages if c == 100),
                'high_coverage': sum(1 for c in verse_coverages if c >= 90),
                'medium_coverage': sum(1 for c in verse_coverages if 70 <= c < 90),
                'low_coverage': sum(1 for c in verse_coverages if c < 70)
            },
            'uncovered_words': uncovered_words.most_common(50),
            'strongs_mapping_stats': {
                'total_mappings': len(self.strongs_integration.strongs_to_english),
                'average_translations_per_word': np.mean([
                    len(words) for words in self.strongs_integration.strongs_to_english.values()
                ]) if self.strongs_integration.strongs_to_english else 0
            }
        }
        
    def generate_coverage_report(self, stats: Dict, output_path: Optional[Path] = None) -> str:
        """Generate a formatted coverage report."""
        lines = []
        lines.append("=" * 70)
        lines.append("ALIGNMENT COVERAGE ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Summary
        summary = stats['summary']
        lines.append("OVERALL COVERAGE")
        lines.append("-" * 40)
        lines.append(f"Total word occurrences (tokens): {summary['total_tokens']:,}")
        lines.append(f"Unique words (types): {summary['total_types']:,}")
        lines.append(f"Aligned occurrences: {summary['covered_tokens']:,} ({summary['token_coverage']:.1f}%)")
        lines.append(f"Aligned unique words: {summary['covered_types']:,} ({summary['type_coverage']:.1f}%)")
        lines.append("")
        
        # Coverage by frequency
        lines.append("COVERAGE BY WORD FREQUENCY")
        lines.append("-" * 40)
        freq_order = ['high_freq', 'medium_freq', 'low_freq', 'hapax']
        freq_labels = {
            'high_freq': 'High frequency (100+)',
            'medium_freq': 'Medium frequency (10-99)',
            'low_freq': 'Low frequency (2-9)',
            'hapax': 'Hapax legomena (1)'
        }
        
        for band in freq_order:
            if band in stats['coverage_by_frequency']:
                data = stats['coverage_by_frequency'][band]
                lines.append(f"{freq_labels[band]}:")
                lines.append(f"  Types: {data['covered_types']:,}/{data['total_types']:,} ({data['type_coverage']:.1f}%)")
                lines.append(f"  Tokens: {data['covered_tokens']:,}/{data['total_tokens']:,} ({data['token_coverage']:.1f}%)")
        lines.append("")
        
        # Coverage by POS
        lines.append("COVERAGE BY PART OF SPEECH")
        lines.append("-" * 40)
        for pos, data in sorted(stats['coverage_by_pos'].items(), 
                              key=lambda x: x[1]['total'], reverse=True):
            if data['total'] > 0:
                lines.append(f"{pos}: {data['covered']:,}/{data['total']:,} ({data['coverage']:.1f}%)")
        lines.append("")
        
        # Verse statistics
        verse_stats = stats['verse_statistics']
        lines.append("VERSE-LEVEL STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Total verses analyzed: {verse_stats['total_verses']:,}")
        lines.append(f"Average verse coverage: {verse_stats['average_coverage']:.1f}%")
        lines.append(f"Standard deviation: {verse_stats['std_coverage']:.1f}%")
        lines.append(f"Verses with perfect coverage (100%): {verse_stats['perfect_coverage']:,}")
        lines.append(f"Verses with high coverage (≥90%): {verse_stats['high_coverage']:,}")
        lines.append(f"Verses with medium coverage (70-89%): {verse_stats['medium_coverage']:,}")
        lines.append(f"Verses with low coverage (<70%): {verse_stats['low_coverage']:,}")
        lines.append("")
        
        # Most common uncovered words
        lines.append("TOP 30 UNCOVERED WORDS")
        lines.append("-" * 40)
        for i, (word, count) in enumerate(stats['uncovered_words'][:30], 1):
            lines.append(f"{i:2d}. {word:<20} {count:,} occurrences")
        lines.append("")
        
        # Strong's statistics
        strongs_stats = stats['strongs_mapping_stats']
        lines.append("STRONG'S CONCORDANCE STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Total Strong's entries with mappings: {strongs_stats['total_mappings']:,}")
        lines.append(f"Average English words per Strong's entry: {strongs_stats['average_translations_per_word']:.1f}")
        lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Coverage report saved to {output_path}")
            
        return report