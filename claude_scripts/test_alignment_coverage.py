#!/usr/bin/env python3
"""
Test word alignment coverage for the full KJV Bible.

This script analyzes whether every word in the KJV translation
maps back to an original language word.
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.word_alignment import IBMModel1


def tokenize_english(text: str) -> List[str]:
    """Tokenize English text consistently with alignment module."""
    text = text.lower()
    # Keep punctuation separate
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


def analyze_verse_coverage(verse_data: Dict) -> Dict:
    """Analyze alignment coverage for a single verse."""
    result = {
        'total_words': 0,
        'mapped_words': 0,
        'unmapped_words': [],
        'has_original': False,
        'alignments': []
    }
    
    # Get English text (try both cases)
    translations = verse_data.get('translations', {})
    eng_text = translations.get('eng_kjv', '') or translations.get('ENG_KJV', '')
    if not eng_text:
        return result
    
    # Tokenize English
    eng_tokens = tokenize_english(eng_text)
    result['total_words'] = len(eng_tokens)
    
    # Check if we have original language
    if verse_data.get('hebrew_words') or verse_data.get('greek_words'):
        result['has_original'] = True
    else:
        # No original language, so no alignment possible
        result['unmapped_words'] = eng_tokens
        return result
    
    # Get alignments (try both cases)
    alignments = verse_data.get('alignments', {})
    if 'eng_kjv' in alignments:
        alignments = alignments['eng_kjv']
    elif 'ENG_KJV' in alignments:
        alignments = alignments['ENG_KJV']
    else:
        alignments = []
    result['alignments'] = alignments
    
    # Track which English words are mapped
    mapped_indices = set()
    for alignment in alignments:
        if 'target_indices' in alignment:
            mapped_indices.update(alignment['target_indices'])
    
    # Find unmapped words
    for i, token in enumerate(eng_tokens):
        if i not in mapped_indices:
            result['unmapped_words'].append((i, token))
        else:
            result['mapped_words'] += 1
    
    return result


def analyze_book_coverage(book_file: Path) -> Dict:
    """Analyze alignment coverage for an entire book."""
    with open(book_file, 'r', encoding='utf-8') as f:
        book_data = json.load(f)
    
    book_stats = {
        'book': book_data.get('book', book_file.stem),
        'total_verses': 0,
        'verses_with_original': 0,
        'total_words': 0,
        'mapped_words': 0,
        'unmapped_words': defaultdict(int),
        'problem_verses': []
    }
    
    for chapter in book_data.get('chapters', []):
        for verse in chapter.get('verses', []):
            book_stats['total_verses'] += 1
            
            # Analyze verse coverage
            coverage = analyze_verse_coverage(verse)
            
            if coverage['has_original']:
                book_stats['verses_with_original'] += 1
                book_stats['total_words'] += coverage['total_words']
                book_stats['mapped_words'] += coverage['mapped_words']
                
                # Track unmapped words
                for idx, word in coverage['unmapped_words']:
                    book_stats['unmapped_words'][word] += 1
                
                # Track problematic verses (low coverage)
                if coverage['total_words'] > 0:
                    coverage_pct = coverage['mapped_words'] / coverage['total_words'] * 100
                    if coverage_pct < 50:  # Less than 50% mapped
                        book_stats['problem_verses'].append({
                            'verse_id': verse['verse_id'],
                            'coverage': coverage_pct,
                            'unmapped': [w for _, w in coverage['unmapped_words']]
                        })
    
    return book_stats


def analyze_full_bible(export_dir: Path) -> Dict:
    """Analyze alignment coverage for the full Bible."""
    overall_stats = {
        'total_books': 0,
        'ot_books': 0,
        'nt_books': 0,
        'total_verses': 0,
        'verses_with_original': 0,
        'total_words': 0,
        'mapped_words': 0,
        'unmapped_by_frequency': Counter(),
        'books_by_coverage': [],
        'overall_issues': []
    }
    
    # Process each book
    for book_file in sorted(export_dir.glob("*.json")):
        if book_file.name.startswith("_"):
            continue
        
        print(f"Analyzing {book_file.name}...")
        book_stats = analyze_book_coverage(book_file)
        
        overall_stats['total_books'] += 1
        overall_stats['total_verses'] += book_stats['total_verses']
        overall_stats['verses_with_original'] += book_stats['verses_with_original']
        overall_stats['total_words'] += book_stats['total_words']
        overall_stats['mapped_words'] += book_stats['mapped_words']
        
        # Aggregate unmapped words
        for word, count in book_stats['unmapped_words'].items():
            overall_stats['unmapped_by_frequency'][word] += count
        
        # Calculate book coverage
        if book_stats['total_words'] > 0:
            coverage_pct = book_stats['mapped_words'] / book_stats['total_words'] * 100
            overall_stats['books_by_coverage'].append({
                'book': book_stats['book'],
                'coverage': coverage_pct,
                'verses': book_stats['total_verses'],
                'words': book_stats['total_words'],
                'mapped': book_stats['mapped_words']
            })
            
            # Track OT/NT
            if book_stats['book'] in ['Gen', 'Exod', 'Lev', 'Num', 'Deut', 'Josh', 'Judg', 'Ruth',
                                     '1Sam', '2Sam', '1Kgs', '2Kgs', '1Chr', '2Chr', 'Ezra', 'Neh',
                                     'Esth', 'Job', 'Ps', 'Prov', 'Eccl', 'Song', 'Isa', 'Jer',
                                     'Lam', 'Ezek', 'Dan', 'Hos', 'Joel', 'Amos', 'Obad', 'Jonah',
                                     'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal']:
                overall_stats['ot_books'] += 1
            else:
                overall_stats['nt_books'] += 1
        
        # Add problem verses to overall issues
        if book_stats['problem_verses']:
            overall_stats['overall_issues'].extend([
                {**v, 'book': book_stats['book']} 
                for v in book_stats['problem_verses'][:3]  # Top 3 per book
            ])
    
    # Sort books by coverage
    overall_stats['books_by_coverage'].sort(key=lambda x: x['coverage'])
    
    return overall_stats


def categorize_unmapped_words(unmapped_words: Counter) -> Dict:
    """Categorize unmapped words by type."""
    categories = {
        'punctuation': [],
        'articles': [],
        'conjunctions': [],
        'prepositions': [],
        'pronouns': [],
        'auxiliary_verbs': [],
        'numbers': [],
        'proper_nouns': [],
        'other': []
    }
    
    # Common word lists
    articles = {'a', 'an', 'the'}
    conjunctions = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'as', 'if', 'when', 'while', 'because', 'although'}
    prepositions = {'in', 'on', 'at', 'by', 'for', 'from', 'to', 'with', 'of', 'about', 'over', 'under', 'between'}
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 
                'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those'}
    auxiliary = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 
                 'can', 'could', 'must'}
    
    for word, count in unmapped_words.most_common():
        if not word.isalpha():
            categories['punctuation'].append((word, count))
        elif word in articles:
            categories['articles'].append((word, count))
        elif word in conjunctions:
            categories['conjunctions'].append((word, count))
        elif word in prepositions:
            categories['prepositions'].append((word, count))
        elif word in pronouns:
            categories['pronouns'].append((word, count))
        elif word in auxiliary:
            categories['auxiliary_verbs'].append((word, count))
        elif word.isdigit():
            categories['numbers'].append((word, count))
        elif word[0].isupper():
            categories['proper_nouns'].append((word, count))
        else:
            categories['other'].append((word, count))
    
    return categories


def generate_coverage_report(stats: Dict, output_file: Path):
    """Generate a detailed coverage report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Word Alignment Coverage Report\n\n")
        
        # Overall statistics
        overall_coverage = (stats['mapped_words'] / stats['total_words'] * 100) if stats['total_words'] > 0 else 0
        
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Total Books**: {stats['total_books']} ({stats['ot_books']} OT, {stats['nt_books']} NT)\n")
        f.write(f"- **Total Verses**: {stats['total_verses']:,}\n")
        f.write(f"- **Verses with Original Language**: {stats['verses_with_original']:,} "
                f"({stats['verses_with_original']/stats['total_verses']*100:.1f}%)\n")
        f.write(f"- **Total English Words**: {stats['total_words']:,}\n")
        f.write(f"- **Mapped Words**: {stats['mapped_words']:,}\n")
        f.write(f"- **Unmapped Words**: {stats['total_words'] - stats['mapped_words']:,}\n")
        f.write(f"- **Overall Coverage**: {overall_coverage:.2f}%\n\n")
        
        # Books with lowest coverage
        f.write("## Books with Lowest Coverage\n\n")
        f.write("| Book | Coverage | Total Words | Mapped Words |\n")
        f.write("|------|----------|-------------|-------------|\n")
        for book in stats['books_by_coverage'][:10]:
            f.write(f"| {book['book']} | {book['coverage']:.1f}% | "
                   f"{book['words']:,} | {book['mapped']:,} |\n")
        f.write("\n")
        
        # Most common unmapped words
        f.write("## Most Common Unmapped Words\n\n")
        categories = categorize_unmapped_words(stats['unmapped_by_frequency'])
        
        for category, words in categories.items():
            if words:
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                f.write("| Word | Frequency |\n")
                f.write("|------|----------|\n")
                for word, count in words[:10]:
                    f.write(f"| {word} | {count:,} |\n")
                f.write("\n")
        
        # Problem verses
        if stats['overall_issues']:
            f.write("## Verses with Low Coverage\n\n")
            f.write("| Book | Verse | Coverage | Unmapped Words |\n")
            f.write("|------|-------|----------|----------------|\n")
            for issue in stats['overall_issues'][:20]:
                unmapped = ', '.join(issue['unmapped'][:5])
                if len(issue['unmapped']) > 5:
                    unmapped += f" (+{len(issue['unmapped'])-5} more)"
                f.write(f"| {issue['book']} | {issue['verse_id']} | "
                       f"{issue['coverage']:.1f}% | {unmapped} |\n")
        
        # Analysis summary
        f.write("\n## Analysis Summary\n\n")
        
        if overall_coverage >= 95:
            f.write("✅ **Excellent Coverage**: The alignment system achieves over 95% coverage.\n\n")
        elif overall_coverage >= 85:
            f.write("⚠️ **Good Coverage**: The alignment system achieves good coverage but has room for improvement.\n\n")
        else:
            f.write("❌ **Needs Improvement**: The alignment system needs significant improvement to achieve full coverage.\n\n")
        
        # Common issues
        f.write("### Common Issues Identified\n\n")
        
        # Analyze unmapped categories
        total_unmapped = stats['total_words'] - stats['mapped_words']
        if total_unmapped > 0:
            punct_count = sum(count for _, count in categories['punctuation'])
            article_count = sum(count for _, count in categories['articles'])
            conj_count = sum(count for _, count in categories['conjunctions'])
            
            f.write(f"1. **Punctuation**: {punct_count:,} unmapped ({punct_count/total_unmapped*100:.1f}%)\n")
            f.write(f"2. **Articles**: {article_count:,} unmapped ({article_count/total_unmapped*100:.1f}%)\n")
            f.write(f"3. **Conjunctions**: {conj_count:,} unmapped ({conj_count/total_unmapped*100:.1f}%)\n")
            
            # Function words explanation
            function_words = punct_count + article_count + conj_count
            f.write(f"\nFunction words and punctuation account for {function_words/total_unmapped*100:.1f}% "
                   f"of unmapped tokens. Many of these don't have direct equivalents in Hebrew/Greek.\n")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test word alignment coverage for the full KJV Bible"
    )
    
    parser.add_argument(
        '--export-dir',
        type=Path,
        default=Path('full_fixed_export'),
        help='Directory containing exported Bible with alignments'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('alignment_coverage_report.md'),
        help='Output file for coverage report'
    )
    
    parser.add_argument(
        '--train-first',
        action='store_true',
        help='Train alignment models before testing'
    )
    
    args = parser.parse_args()
    
    # Check if export exists
    if not args.export_dir.exists():
        print(f"Export directory not found: {args.export_dir}")
        print("Please run the export with alignments first:")
        print("  python -m abba.cli_with_alignment --output aligned_export")
        return 1
    
    # Train models if requested
    if args.train_first:
        print("Training alignment models...")
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/train_word_alignment.py",
            "--export-dir", str(args.export_dir)
        ])
        if result.returncode != 0:
            print("Training failed!")
            return 1
    
    # Analyze coverage
    print(f"Analyzing alignment coverage in {args.export_dir}...")
    stats = analyze_full_bible(args.export_dir)
    
    # Generate report
    print(f"Generating report: {args.output}")
    generate_coverage_report(stats, args.output)
    
    # Print summary
    overall_coverage = (stats['mapped_words'] / stats['total_words'] * 100) if stats['total_words'] > 0 else 0
    print(f"\n{'='*60}")
    print(f"Overall Word Alignment Coverage: {overall_coverage:.2f}%")
    print(f"Total Words: {stats['total_words']:,}")
    print(f"Mapped Words: {stats['mapped_words']:,}")
    print(f"Unmapped Words: {stats['total_words'] - stats['mapped_words']:,}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())