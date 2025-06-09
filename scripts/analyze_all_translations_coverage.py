#!/usr/bin/env python3
"""
Analyze coverage for all translations and generate a markdown report.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
from datetime import datetime
from collections import Counter, defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from abba_align.coverage_analyzer import AlignmentCoverageAnalyzer
from abba_align.model_info import ModelDiscovery

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TranslationCoverageAnalyzer:
    """Analyze coverage for multiple translations."""
    
    def __init__(self, translations_dir: Path = None):
        self.translations_dir = translations_dir or Path('translations')
        self.results = []
        self.model_discovery = ModelDiscovery()
        
    def analyze_all_translations(self) -> List[Dict]:
        """Analyze coverage for all translations in the directory."""
        if not self.translations_dir.exists():
            logger.error(f"Translations directory not found: {self.translations_dir}")
            return []
            
        translation_files = list(self.translations_dir.glob('*.json'))
        logger.info(f"Found {len(translation_files)} translation files")
        
        for trans_file in sorted(translation_files):
            logger.info(f"Analyzing {trans_file.name}...")
            result = self.analyze_single_translation(trans_file)
            if result:
                self.results.append(result)
                
        return self.results
        
    def analyze_single_translation(self, trans_path: Path) -> Dict:
        """Analyze coverage for a single translation."""
        try:
            # Detect if OT or NT based on content
            with open(trans_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both array and dict formats for books
            books = data.get('books', {})
            if isinstance(books, dict):
                # Get first book name from dict keys
                book_names = list(books.keys())
                if not book_names:
                    logger.warning(f"No books found in {trans_path}")
                    return None
                first_book_key = book_names[0]
                first_book = books[first_book_key].get('name', '')
            else:
                # Array format
                first_book = books[0].get('name', '') if books else ''
            
            # Determine source language
            if first_book in ['Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans']:
                source_lang = 'greek'
                testament = 'NT'
            else:
                source_lang = 'hebrew'
                testament = 'OT'
                
            # Find best model
            model = self.model_discovery.find_model(source_lang, 'english')
            if not model:
                logger.warning(f"No model found for {source_lang}")
                return None
                
            # Convert translation to expected format
            verses = []
            if isinstance(books, dict):
                for book_key, book_data in books.items():
                    for chapter in book_data.get('chapters', []):
                        for verse in chapter.get('verses', []):
                            verses.append({
                                'book': book_key,
                                'chapter': chapter['chapter'],
                                'verse': verse['verse'],
                                'text': verse['text']
                            })
            else:
                # Handle array format if needed
                for book in books:
                    for chapter in book.get('chapters', []):
                        for verse in chapter.get('verses', []):
                            verses.append({
                                'book': book['abbr'],
                                'chapter': chapter['chapter'],
                                'verse': verse['verse'],
                                'text': verse['text']
                            })
                            
            # Create temporary data structure
            temp_data = {'verses': verses}
            
            # Run coverage analysis
            analyzer = AlignmentCoverageAnalyzer(source_lang=source_lang)
            
            # Load model's Strong's mappings
            if model.path.exists():
                with open(model.path, 'r') as f:
                    model_data = json.load(f)
                    if 'strongs_mappings' in model_data:
                        # Override lexicon with model's mappings
                        analyzer.strongs_integration.strongs_to_english = {}
                        # Convert mappings to Counter format
                        for strongs_num, word_dict in model_data['strongs_mappings'].items():
                            analyzer.strongs_integration.strongs_to_english[strongs_num] = Counter(word_dict)
                        logger.info(f"Loaded {len(analyzer.strongs_integration.strongs_to_english):,} Strong's mappings from {model.name}")
            
            # Analyze verse by verse
            word_counts = Counter()
            aligned_words = set()
            pos_counts = defaultdict(Counter)
            aligned_by_pos = defaultdict(set)
            verse_stats = []
            
            for verse in verses:
                verse_result = analyzer._analyze_verse(
                    verse, word_counts, pos_counts, 
                    aligned_words, aligned_by_pos
                )
                verse_stats.append(verse_result)
                
            # Calculate statistics
            stats = analyzer._calculate_statistics(
                word_counts, pos_counts, aligned_words,
                aligned_by_pos, verse_stats
            )
            
            # Extract key metrics
            summary = stats['summary']
            freq_coverage = stats.get('coverage_by_frequency', {})
            
            return {
                'translation': trans_path.stem,
                'testament': testament,
                'source_language': source_lang,
                'model_used': model.name,
                'token_coverage': summary['token_coverage'],
                'type_coverage': summary['type_coverage'],
                'total_tokens': summary['total_tokens'],
                'total_types': summary['total_types'],
                'covered_tokens': summary['covered_tokens'],
                'covered_types': summary['covered_types'],
                'high_freq_coverage': freq_coverage.get('high_freq', {}).get('token_coverage', 0),
                'uncovered_top_words': stats.get('uncovered_words', [])[:5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {trans_path}: {e}")
            return None
            
    def generate_markdown_report(self, output_path: Path = None) -> str:
        """Generate a comprehensive markdown report."""
        lines = []
        
        # Header
        lines.append("# Translation Coverage Analysis Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary statistics
        if self.results:
            avg_token_coverage = sum(r['token_coverage'] for r in self.results) / len(self.results)
            avg_type_coverage = sum(r['type_coverage'] for r in self.results) / len(self.results)
            
            lines.append("## Summary Statistics")
            lines.append("")
            lines.append(f"- **Translations Analyzed**: {len(self.results)}")
            lines.append(f"- **Average Token Coverage**: {avg_token_coverage:.1f}%")
            lines.append(f"- **Average Type Coverage**: {avg_type_coverage:.1f}%")
            lines.append("")
        
        # Main coverage table
        lines.append("## Coverage by Translation")
        lines.append("")
        lines.append("| Translation | Testament | Source | Token Coverage | Type Coverage | Total Words | Covered Words | High-Freq Coverage |")
        lines.append("|-------------|-----------|--------|----------------|---------------|-------------|---------------|-------------------|")
        
        # Sort by token coverage descending
        sorted_results = sorted(self.results, key=lambda x: x['token_coverage'], reverse=True)
        
        for result in sorted_results:
            coverage_indicator = self._get_coverage_indicator(result['token_coverage'])
            
            lines.append(
                f"| {result['translation']} "
                f"| {result['testament']} "
                f"| {result['source_language'].title()} "
                f"| {coverage_indicator} {result['token_coverage']:.1f}% "
                f"| {result['type_coverage']:.1f}% "
                f"| {result['total_tokens']:,} "
                f"| {result['covered_tokens']:,} "
                f"| {result['high_freq_coverage']:.1f}% |"
            )
            
        lines.append("")
        
        # Coverage distribution
        lines.append("## Coverage Distribution")
        lines.append("")
        
        excellent = sum(1 for r in self.results if r['token_coverage'] >= 90)
        good = sum(1 for r in self.results if 80 <= r['token_coverage'] < 90)
        fair = sum(1 for r in self.results if 70 <= r['token_coverage'] < 80)
        poor = sum(1 for r in self.results if r['token_coverage'] < 70)
        
        lines.append("| Coverage Level | Range | Count | Percentage |")
        lines.append("|----------------|-------|-------|------------|")
        lines.append(f"| 🟢 Excellent | ≥90% | {excellent} | {excellent/len(self.results)*100:.1f}% |")
        lines.append(f"| 🟡 Good | 80-89% | {good} | {good/len(self.results)*100:.1f}% |")
        lines.append(f"| 🟠 Fair | 70-79% | {fair} | {fair/len(self.results)*100:.1f}% |")
        lines.append(f"| 🔴 Poor | <70% | {poor} | {poor/len(self.results)*100:.1f}% |")
        lines.append("")
        
        # Testament comparison
        ot_results = [r for r in self.results if r['testament'] == 'OT']
        nt_results = [r for r in self.results if r['testament'] == 'NT']
        
        if ot_results and nt_results:
            lines.append("## Testament Comparison")
            lines.append("")
            lines.append("| Testament | Avg Token Coverage | Avg Type Coverage | Translations |")
            lines.append("|-----------|-------------------|-------------------|--------------|")
            
            ot_token_avg = sum(r['token_coverage'] for r in ot_results) / len(ot_results)
            ot_type_avg = sum(r['type_coverage'] for r in ot_results) / len(ot_results)
            
            nt_token_avg = sum(r['token_coverage'] for r in nt_results) / len(nt_results)
            nt_type_avg = sum(r['type_coverage'] for r in nt_results) / len(nt_results)
            
            lines.append(f"| Old Testament | {ot_token_avg:.1f}% | {ot_type_avg:.1f}% | {len(ot_results)} |")
            lines.append(f"| New Testament | {nt_token_avg:.1f}% | {nt_type_avg:.1f}% | {len(nt_results)} |")
            lines.append("")
        
        # Most common uncovered words
        lines.append("## Common Uncovered Words")
        lines.append("")
        lines.append("Top uncovered words across translations:")
        lines.append("")
        
        # Aggregate uncovered words
        all_uncovered = {}
        for result in self.results:
            for word, count in result.get('uncovered_top_words', []):
                if word not in all_uncovered:
                    all_uncovered[word] = 0
                all_uncovered[word] += count
                
        # Show top 20
        sorted_uncovered = sorted(all_uncovered.items(), key=lambda x: x[1], reverse=True)[:20]
        
        lines.append("| Word | Total Occurrences |")
        lines.append("|------|-------------------|")
        for word, count in sorted_uncovered:
            lines.append(f"| {word} | {count:,} |")
        lines.append("")
        
        # Models used
        lines.append("## Models Used")
        lines.append("")
        models_used = set(r['model_used'] for r in self.results)
        for model_name in sorted(models_used):
            model = next((m for m in self.model_discovery.models if m.name == model_name), None)
            if model:
                lines.append(f"- **{model_name}**")
                features = [k for k, v in model.features.items() if v]
                lines.append(f"  - Features: {', '.join(features)}")
                lines.append(f"  - Mappings: {model.statistics.get('translation_mappings', 0):,}")
        lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        
        low_coverage = [r for r in self.results if r['token_coverage'] < 80]
        if low_coverage:
            lines.append("### Translations Needing Attention")
            lines.append("")
            for trans in sorted(low_coverage, key=lambda x: x['token_coverage']):
                lines.append(f"- **{trans['translation']}**: {trans['token_coverage']:.1f}% coverage")
                lines.append(f"  - Consider adding manual alignments for high-frequency words")
                lines.append(f"  - Top uncovered: {', '.join(w[0] for w in trans['uncovered_top_words'][:3])}")
            lines.append("")
            
        lines.append("### General Improvements")
        lines.append("")
        lines.append("1. Add more parallel texts to training corpus")
        lines.append("2. Include translation-specific vocabulary mappings")
        lines.append("3. Enhance Strong's concordance coverage")
        lines.append("4. Consider training translation-specific models")
        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Generated by ABBA-Align Coverage Analyzer*")
        
        report = "\n".join(lines)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
            
        return report
        
    def _get_coverage_indicator(self, coverage: float) -> str:
        """Get visual indicator for coverage level."""
        if coverage >= 90:
            return "🟢"
        elif coverage >= 80:
            return "🟡"
        elif coverage >= 70:
            return "🟠"
        else:
            return "🔴"


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze alignment coverage for all translations"
    )
    parser.add_argument(
        '--translations-dir',
        type=Path,
        default=Path('data/sources/translations'),
        help='Directory containing translation JSON files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('translation_coverage_report.md'),
        help='Output path for markdown report'
    )
    parser.add_argument(
        '--train-first',
        action='store_true',
        help='Train models before analysis'
    )
    
    args = parser.parse_args()
    
    # Train models if requested
    if args.train_first:
        logger.info("Training models first...")
        result = subprocess.run([sys.executable, 'scripts/train_all_models.py'])
        if result.returncode != 0:
            logger.error("Model training failed")
            sys.exit(1)
    
    # Check if models exist
    model_dir = Path('models/biblical_alignment')
    if not model_dir.exists() or not list(model_dir.glob('*_biblical.json')):
        logger.error("No models found. Please train models first:")
        logger.error("  python scripts/train_all_models.py")
        sys.exit(1)
    
    # Run analysis
    analyzer = TranslationCoverageAnalyzer(args.translations_dir)
    results = analyzer.analyze_all_translations()
    
    if not results:
        logger.error("No translations analyzed successfully")
        sys.exit(1)
        
    # Generate report
    report = analyzer.generate_markdown_report(args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Translations analyzed: {len(results)}")
    print(f"Report saved to: {args.output}")
    print("\nTop 5 by coverage:")
    for i, result in enumerate(sorted(results, key=lambda x: x['token_coverage'], reverse=True)[:5], 1):
        print(f"{i}. {result['translation']}: {result['token_coverage']:.1f}%")


if __name__ == '__main__':
    main()