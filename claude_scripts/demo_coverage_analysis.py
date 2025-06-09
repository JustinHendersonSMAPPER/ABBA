#!/usr/bin/env python3
"""
Demonstrate alignment coverage analysis.

This shows how to validate that your alignment models cover
a sufficient percentage of a translation's vocabulary.
"""

import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from abba_align.coverage_analyzer import AlignmentCoverageAnalyzer


def create_sample_translation():
    """Create a sample translation for testing."""
    sample = {
        "books": [{
            "name": "Genesis",
            "chapters": [{
                "number": 1,
                "verses": [
                    {
                        "number": 1,
                        "text": "In the beginning God created the heavens and the earth",
                        "source_words": [
                            {"text": "בְּרֵאשִׁית", "strongs": "H7225"},
                            {"text": "בָּרָא", "strongs": "H1254"},
                            {"text": "אֱלֹהִים", "strongs": "H430"},
                            {"text": "אֵת", "strongs": "H853"},
                            {"text": "הַשָּׁמַיִם", "strongs": "H8064"},
                            {"text": "וְאֵת", "strongs": "H853"},
                            {"text": "הָאָרֶץ", "strongs": "H776"}
                        ]
                    },
                    {
                        "number": 2,
                        "text": "And the earth was without form and void and darkness was upon the face of the deep",
                        "source_words": [
                            {"text": "וְהָאָרֶץ", "strongs": "H776"},
                            {"text": "הָיְתָה", "strongs": "H1961"},
                            {"text": "תֹהוּ", "strongs": "H8414"},
                            {"text": "וָבֹהוּ", "strongs": "H922"}
                        ]
                    }
                ]
            }]
        }]
    }
    
    # Save sample
    sample_path = Path("test_translation.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2)
    
    return sample_path


def main():
    """Run coverage analysis demonstration."""
    print("=" * 60)
    print("ALIGNMENT COVERAGE ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Create sample translation
    print("Creating sample translation...")
    translation_path = create_sample_translation()
    
    # Initialize analyzer
    print("\nInitializing coverage analyzer...")
    analyzer = AlignmentCoverageAnalyzer(source_lang='hebrew')
    
    # Run analysis
    print("\nAnalyzing translation coverage...")
    stats = analyzer.analyze_translation_coverage(translation_path)
    
    # Generate report
    print("\nGenerating coverage report...")
    report = analyzer.generate_coverage_report(stats, Path("coverage_report.txt"))
    
    # Print report
    print("\n" + report)
    
    # Demonstrate different coverage scenarios
    print("\n" + "=" * 60)
    print("COVERAGE INTERPRETATION GUIDE")
    print("=" * 60)
    print()
    
    print("Token Coverage: Percentage of word occurrences that can be aligned")
    print("  - 90-100%: Excellent - Nearly all text can be traced to source")
    print("  - 80-90%:  Good - Most content is aligned, some gaps exist") 
    print("  - 70-80%:  Fair - Significant portions need manual alignment")
    print("  - <70%:    Poor - Major alignment gaps, needs improvement")
    print()
    
    print("Type Coverage: Percentage of unique words that can be aligned")
    print("  - Usually lower than token coverage")
    print("  - Shows vocabulary breadth of alignment model")
    print("  - Important for rare/technical terms")
    print()
    
    print("Coverage by Frequency:")
    print("  - High-frequency words should have >95% coverage")
    print("  - Medium-frequency words should have >85% coverage")
    print("  - Low-frequency words typically have lower coverage")
    print("  - Hapax legomena (single occurrence) often have lowest coverage")
    print()
    
    # Clean up
    translation_path.unlink()
    Path("coverage_report.txt").unlink(missing_ok=True)
    

if __name__ == '__main__':
    main()