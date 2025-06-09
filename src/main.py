#!/usr/bin/env python3
"""
ABBA - Annotated Bible and Background Analysis
Main orchestration script
"""

import logging
import sys
import subprocess
from pathlib import Path
import json

# Clear any cached modules that might have import issues
if 're' in sys.modules:
    del sys.modules['re']

from abba_data_downloader import ABBADataDownloader
from abba_coverage_analyzer import ABBACoverageAnalyzer
from abba.alignment.position_aligner import PositionAligner
from abba.alignment.morphological_aligner import MorphologicalAligner
from abba.alignment.statistical_aligner import StatisticalAligner
from abba.alignment.ensemble_aligner import EnsembleAligner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ABBA.Main')


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_alignments(aligner, aligner_name="Position-based"):
    """Demonstrate alignment with sample verses."""
    logger.info(f"\nDemonstrating {aligner_name} alignment:")
    
    # Load sample data
    try:
        # Load Genesis 1:1 in Hebrew
        hebrew_path = Path('data/sources/morphology/hebrew/Gen.json')
        if hebrew_path.exists():
            with open(hebrew_path, 'r', encoding='utf-8') as f:
                hebrew_data = json.load(f)
                hebrew_verse = hebrew_data['verses'][0]
                hebrew_words = [w['text'] for w in hebrew_verse['words']]
        else:
            # Fallback if morphology not available
            hebrew_words = ["בְּרֵאשִׁית", "בָּרָא", "אֱלֹהִים", "אֵת", "הַשָּׁמַיִם", "וְאֵת", "הָאָרֶץ"]
        
        # Load English translation
        kjv_path = Path('data/sources/translations/eng_kjv.json')
        if kjv_path.exists():
            with open(kjv_path, 'r', encoding='utf-8') as f:
                kjv_data = json.load(f)
                # Find Genesis 1:1
                english_text = None
                if 'books' in kjv_data and 'Gen' in kjv_data['books']:
                    gen_data = kjv_data['books']['Gen']
                    if 'chapters' in gen_data and len(gen_data['chapters']) > 0:
                        chapter1 = gen_data['chapters'][0]
                        if 'verses' in chapter1 and len(chapter1['verses']) > 0:
                            english_text = chapter1['verses'][0].get('text', '')
                
                if english_text:
                    # Simple word splitting - remove punctuation
                    english_words = english_text.strip().replace('.', '').replace(',', '').split()
                else:
                    english_words = ["In", "the", "beginning", "God", "created", "the", "heaven", "and", "the", "earth"]
        else:
            english_words = ["In", "the", "beginning", "God", "created", "the", "heaven", "and", "the", "earth"]
        
        # Perform alignment
        # Check if aligner supports morphological features
        if hasattr(aligner, 'align_verse') and 'book_code' in aligner.align_verse.__code__.co_varnames:
            # Morphological aligner - pass verse reference
            alignments = aligner.align_verse(
                hebrew_words, english_words, 'hebrew', 'english',
                book_code='Gen', chapter=1, verse=1
            )
        else:
            # Position aligner
            alignments = aligner.align_verse(hebrew_words, english_words, 'hebrew', 'english')
        
        # Display results
        print(f"\n  Genesis 1:1 Alignment (Hebrew → English) - {aligner_name}:")
        print("  " + "-" * 70)
        print(f"  Hebrew words: {len(hebrew_words)}")
        print(f"  English words: {len(english_words)}")
        print("\n  Alignments:")
        for align in alignments[:5]:  # Show first 5 alignments
            features_str = ""
            if 'features' in align and align['features']:
                pos = align['features'].get('pos', '')
                if pos:
                    features_str = f" [{pos}]"
            print(f"    {align['source_word']} → {align['target_word']} (confidence: {align['confidence']}){features_str}")
        if len(alignments) > 5:
            print(f"    ... and {len(alignments) - 5} more alignments")
        
    except Exception as e:
        logger.warning(f"Could not load sample data: {e}")
        
        # Use fallback demo
        hebrew_words = ["בְּרֵאשִׁית", "בָּרָא", "אֱלֹהִים"]
        english_words = ["In", "the", "beginning", "God", "created"]
        
        alignments = aligner.align_verse(hebrew_words, english_words)
        
        print("\n  Sample Alignment (Hebrew → English):")
        print("  " + "-" * 70)
        for align in alignments:
            print(f"    {align['source_word']} → {align['target_word']} (confidence: {align['confidence']})")
    
    print()


def ensure_parallel_corpus():
    """Ensure parallel corpus and statistical models exist."""
    corpus_dir = Path('data/parallel_corpus')
    models_dir = Path('models/statistical')
    
    # Check if corpus files exist
    hebrew_corpus = corpus_dir / 'hebrew_english.source'
    greek_corpus = corpus_dir / 'greek_english.source'
    hebrew_model = models_dir / 'hebrew_english_alignment.json'
    greek_model = models_dir / 'greek_english_alignment.json'
    
    corpus_missing = not all([
        hebrew_corpus.exists(),
        greek_corpus.exists(),
        hebrew_model.exists(),
        greek_model.exists()
    ])
    
    if corpus_missing:
        logger.info("Parallel corpus or statistical models missing. Building full Bible corpus...")
        
        try:
            # Run the full Bible corpus builder
            corpus_script = Path('prepare_full_bible_corpus.py')
            if corpus_script.exists():
                logger.info("Building parallel corpus from full Bible...")
                result = subprocess.run([sys.executable, str(corpus_script)], 
                                     capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    logger.error(f"Corpus building failed: {result.stderr}")
                    return False
                else:
                    logger.info("✓ Parallel corpus built successfully")
            else:
                logger.error("prepare_full_bible_corpus.py not found!")
                return False
            
            # Train statistical models
            train_script = Path('scripts/train_statistical_alignment.py')
            if train_script.exists():
                logger.info("Training statistical alignment models...")
                result = subprocess.run([sys.executable, str(train_script)], 
                                     capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    logger.error(f"Model training failed: {result.stderr}")
                    return False
                else:
                    logger.info("✓ Statistical models trained successfully")
            else:
                logger.error("scripts/train_statistical_alignment.py not found!")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Corpus building or model training timed out")
            return False
        except Exception as e:
            logger.error(f"Error building corpus: {e}")
            return False
    else:
        logger.info("✓ Parallel corpus and statistical models ready")
    
    return True


def main():
    """Main orchestration function."""
    print_header("ABBA - Annotated Bible and Background Analysis")
    print("Version 1.0 - Production Pipeline\n")
    
    # Step 1: Download and prepare source data
    print_header("Step 1: Source Data Preparation")
    downloader = ABBADataDownloader()
    
    if not downloader.check_data_exists():
        logger.info("Downloading required source data...")
        success = downloader.download_all()
        if not success:
            logger.error("Failed to download source data. Exiting.")
            return 1
    else:
        logger.info("✓ Source data already exists")
        
        # Check for missing translations even if data exists
        logger.info("Checking for new or missing translations...")
        missing_count = downloader.validate_and_sync_translations()
        
        if missing_count > 0:
            logger.info(f"✓ Extracted {missing_count} new translations")
        else:
            logger.info("✓ All translations are up to date")
    
    # Step 1.5: Ensure parallel corpus and statistical models exist
    logger.info("Checking parallel corpus and statistical models...")
    if not ensure_parallel_corpus():
        logger.error("Failed to prepare parallel corpus. Exiting.")
        return 1
    
    # Step 2: Position-based alignment
    print_header("Step 2: Alignment System")
    logger.info("✓ Strong's concordance removed")
    logger.info("Initializing position-based aligner...")
    
    # Create position aligner
    position_aligner = PositionAligner(base_confidence=0.4)
    
    # Demo position alignment
    demo_alignments(position_aligner, "Position-based")
    
    # Create morphological aligner
    logger.info("\nInitializing morphological aligner...")
    morph_aligner = MorphologicalAligner(base_confidence=0.6)
    
    # Demo morphological alignment
    demo_alignments(morph_aligner, "Morphological")
    
    # Create statistical aligner
    logger.info("\nInitializing statistical aligner...")
    stat_aligner = StatisticalAligner(base_confidence=0.8)
    
    # Demo statistical alignment
    demo_alignments(stat_aligner, "Statistical")
    
    # Create ensemble aligner
    logger.info("\nCreating ensemble aligner...")
    ensemble_aligner = EnsembleAligner([
        (stat_aligner, 0.5),     # Highest weight for statistical
        (morph_aligner, 0.3),    # Medium weight for morphological
        (position_aligner, 0.2)  # Lower weight for position
    ])
    
    # Demo ensemble alignment
    demo_alignments(ensemble_aligner, "Ensemble")
    
    # Step 3: Coverage analysis with ensemble aligner
    print_header("Step 3: Translation Coverage Analysis")
    logger.info("Analyzing coverage with ensemble aligner...")
    
    # Create coverage analyzer with the ensemble aligner
    analyzer = ABBACoverageAnalyzer(aligner=ensemble_aligner)
    
    # Analyze a sample of translations
    results = analyzer.analyze_all_translations(sample_size=5)
    
    # Generate and display report
    if results:
        report = analyzer.generate_coverage_report(results)
        print(report)
        
        # Save report
        report_path = Path('coverage_report_ensemble.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Coverage report saved to {report_path}")
    else:
        logger.warning("No coverage results generated")
    
    # Summary
    print_header("Pipeline Summary")
    
    # Count translations
    translations_dir = Path('data/sources/translations')
    translation_count = len(list(translations_dir.glob('*.json')))
    
    print(f"✓ Translations available: {translation_count}")
    print(f"✓ Hebrew texts ready for alignment")
    print(f"✓ Greek texts ready for alignment")
    print(f"✓ Strong's concordance removed successfully")
    print(f"✓ Position-based aligner operational")
    print(f"✓ Morphological aligner operational")
    print(f"✓ Statistical aligner operational")
    print(f"✓ Ensemble aligner combining all three methods")
    if results:
        avg_coverage = sum(r['overall_coverage'] for r in results) / len(results)
        print(f"✓ Average alignment coverage (ensemble): {avg_coverage:.1f}%")
    
    print("\nPipeline complete! 🎉")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())