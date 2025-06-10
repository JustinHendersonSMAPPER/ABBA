#!/usr/bin/env python3
"""
ABBA - Annotated Bible and Background Analysis
Main orchestration script
"""

import logging
import sys
import subprocess
import argparse
import os
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
from abba.alignment.neural_aligner import NeuralAligner
from abba.alignment.morphological_lexicon_aligner import MorphologicalLexiconAligner
from abba.alignment.modern_semantic_aligner import ModernSemanticAligner
from abba.alignment.embedding_aligner import CrossLingualEmbeddingAligner, HybridSemanticAligner
from abba.alignment.morphology_aware_aligner import MorphologyAwareAligner
from abba.alignment.improved_biblical_ensemble import ImprovedBiblicalEnsemble
# JSON exporter will be imported conditionally to avoid dependency issues

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
    parser = argparse.ArgumentParser(description='ABBA Alignment System')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with smaller datasets for faster execution')
    parser.add_argument('--max-translations', type=int,
                       help='Maximum number of translations to analyze')
    parser.add_argument('--translation', type=str,
                       help='Test with a specific translation (e.g., "eng_asv", "eng_kjv")')
    parser.add_argument('--enable-neural', action='store_true',
                       help='Enable neural aligner (slower but more accurate)')
    parser.add_argument('--confidence-threshold', type=float,
                       help='Minimum confidence threshold for alignments (default: from env or 0.3)')
    parser.add_argument('--output-format', choices=['json'],
                       help='Output format for alignment results')
    parser.add_argument('--output-dir', type=str, default='aligned_output',
                       help='Output directory for alignment files (default: aligned_output)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use mock embeddings for faster testing (disable real models)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging output')
    
    args = parser.parse_args()
    
    # Configure logging verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('ABBA').setLevel(logging.DEBUG)
    else:
        # Reduce verbosity for cleaner output
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger('ABBA.Main').setLevel(logging.INFO)
        logging.getLogger('ABBA.ABBADataDownloader').setLevel(logging.INFO)
        logging.getLogger('ABBA.ABBACoverageAnalyzer').setLevel(logging.INFO)
        # Suppress alignment step-by-step logging
        logging.getLogger('ABBA.ModernSemanticAligner').setLevel(logging.WARNING)
        logging.getLogger('ABBA.EmbeddingAligner').setLevel(logging.WARNING)
        logging.getLogger('ABBA.MorphologyAwareAligner').setLevel(logging.WARNING)
        logging.getLogger('ABBA.ImprovedBiblicalEnsemble').setLevel(logging.WARNING)
        logging.getLogger('ABBA.StatisticalAligner').setLevel(logging.WARNING)
    
    # Configure confidence threshold: CLI > Environment > Default
    confidence_threshold = args.confidence_threshold
    if confidence_threshold is None:
        confidence_threshold = float(os.getenv('ABBA_CONFIDENCE_THRESHOLD', 0.3))
    
    print_header("ABBA - Annotated Bible and Background Analysis")
    mode = "Test Mode" if args.test_mode else "Production Pipeline"
    print(f"Version 1.0 - {mode}\n")
    
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
    
    # Step 2: Initialize Modern Alignment System
    print_header("Step 2: Initialize Modern Alignment System")
    logger.info("✓ Strong's concordance removed")
    logger.info("✓ Using modern lexicons (BDB, HALOT, BDAG principles)")
    logger.info("✓ Using cross-lingual embeddings")
    
    # Create morphological lexicon aligner (primary method using OSHB/MorphGNT lemmas)
    logger.info("Initializing morphological lexicon aligner...")
    lexicon_aligner = MorphologicalLexiconAligner()
    
    # Create morphology-aware aligner (uses OSHB/MorphGNT data)
    logger.info("Initializing morphology-aware aligner...")
    morphology_aligner = MorphologyAwareAligner()
    
    # Create modern semantic aligner (secondary method)
    logger.info("Initializing modern semantic aligner...")
    modern_aligner = ModernSemanticAligner()
    
    # Create cross-lingual embedding aligner (tertiary method)
    logger.info("Initializing cross-lingual embedding aligner...")
    if args.fast_mode:
        logger.info("  Fast mode: Using mock embeddings")
        embedding_aligner = CrossLingualEmbeddingAligner("mock")
    else:
        logger.info("  Production mode: Using real sentence transformers")
        embedding_aligner = CrossLingualEmbeddingAligner()
    
    # Keep statistical aligner as fallback
    logger.info("Initializing statistical aligner (fallback)...")
    stat_aligner = StatisticalAligner(base_confidence=0.5)
    
    # Create neural aligner if enabled
    neural_aligner = None
    if args.enable_neural:
        logger.info("Initializing neural aligner...")
        try:
            neural_aligner = NeuralAligner(base_confidence=0.8)
        except ImportError as e:
            logger.error(f"Neural aligner unavailable: {e}")
            logger.info("Continuing without neural aligner...")
    
    # Create lexicon-driven ensemble aligner
    logger.info("Creating lexicon-driven ensemble aligner...")
    aligners_list = [
        (lexicon_aligner, 0.50),       # Lexicon-driven - primary (uses real definitions)
        (morphology_aligner, 0.25),    # Morphology-aware - secondary
        (modern_aligner, 0.15),        # Modern semantic - tertiary  
        (embedding_aligner, 0.05),     # Cross-lingual embeddings - minimal
        (stat_aligner, 0.05)           # Statistical fallback - minimal weight
    ]
    
    if neural_aligner:
        # Adjust weights to include neural
        aligners_list = [
            (lexicon_aligner, 0.45),       # Lexicon-driven - primary
            (morphology_aligner, 0.20),    # Morphology-aware
            (modern_aligner, 0.15),        # Modern semantic
            (neural_aligner, 0.10),        # Neural
            (embedding_aligner, 0.05),     # Embeddings
            (stat_aligner, 0.05)           # Statistical fallback
        ]
        logger.info("✓ Neural aligner included in lexicon-driven ensemble")
    
    ensemble_aligner = ImprovedBiblicalEnsemble(aligners_list)
    logger.info("✓ Morphological lexicon ensemble aligner ready for production analysis")
    
    # Step 3: Full corpus alignment analysis
    print_header("Step 3: Full Corpus Alignment Analysis")
    logger.info(f"Starting full corpus analysis with lexicon-driven ensemble aligner (confidence threshold: {confidence_threshold})...")
    
    # Create coverage analyzer with the ensemble aligner
    production_mode = not args.test_mode
    collect_alignments = args.output_format == 'json'  # Collect detailed alignments for JSON export
    analyzer = ABBACoverageAnalyzer(
        aligner=ensemble_aligner, 
        production_mode=production_mode,
        confidence_threshold=confidence_threshold,
        collect_alignments=collect_alignments
    )
    
    # Analyze translations according to mode
    results = analyzer.analyze_all_translations(
        max_translations=args.max_translations,
        translation_filter=args.translation
    )
    
    # Generate and display report
    if results:
        report = analyzer.generate_coverage_report(results)
        print(report)
        
        # Save text report
        report_path = Path('coverage_report_ensemble.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Coverage report saved to {report_path}")
        
        # Export JSON if requested
        if args.output_format == 'json':
            print_header("Step 4: JSON Export")
            logger.info(f"Exporting alignments to JSON format in {args.output_dir}...")
            
            # Import JSON exporter directly to avoid dependency issues
            try:
                # Import directly from the file to avoid module dependency conflicts
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "json_alignment_exporter", 
                    Path(__file__).parent / "abba" / "export" / "json_alignment_exporter.py"
                )
                json_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(json_module)
                JSONAlignmentExporter = json_module.JSONAlignmentExporter
                json_exporter = JSONAlignmentExporter(output_dir=args.output_dir)
                logger.info("✓ JSON exporter loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load JSON exporter: {e}")
                logger.info("Continuing without JSON export...")
                json_exporter = None
            
            # Create metadata about the alignment process
            metadata = {
                "aligner_type": ensemble_aligner.__class__.__name__,
                "aligner_components": [
                    {"type": aligner.__class__.__name__, "weight": weight}
                    for aligner, weight in aligners_list
                ],
                "confidence_threshold": confidence_threshold,
                "production_mode": production_mode,
                "neural_enabled": neural_aligner is not None,
                "command_line_args": {
                    "test_mode": args.test_mode,
                    "max_translations": args.max_translations,
                    "translation_filter": args.translation,
                    "enable_neural": args.enable_neural
                }
            }
            
            if json_exporter:
                # Export batch results - now returns report + individual files
                report_file, alignment_files = json_exporter.export_batch_alignments(results, metadata)
                
                # Validate the report file
                validation_result = json_exporter.validate_json_file(report_file)
                if validation_result["valid_json"]:
                    logger.info(f"✓ Report file validated: {report_file}")
                    logger.info(f"  Report size: {validation_result['file_size_mb']:.2f} MB")
                else:
                    logger.warning(f"Report validation issues: {validation_result}")
                
                # Validate alignment files
                total_size = 0
                for alignment_file in alignment_files:
                    file_validation = json_exporter.validate_json_file(alignment_file)
                    if file_validation["valid_json"]:
                        total_size += file_validation['file_size_mb']
                    else:
                        logger.warning(f"Alignment file validation issues: {alignment_file}")
                
                logger.info(f"✓ Exported {len(alignment_files)} alignment files")
                logger.info(f"  Total alignment data size: {total_size:.2f} MB")
            
    else:
        logger.warning("No coverage results generated")
    
    # Summary
    print_header("Analysis Complete")
    
    # Count translations
    translations_dir = Path('data/sources/translations')
    translation_count = len(list(translations_dir.glob('*.json')))
    
    print(f"✓ Translations available: {translation_count}")
    print(f"✓ Morphological lexicon aligner operational (using OSHB/MorphGNT lemmas)")
    print(f"✓ No Strong's concordance bias - uses actual morphological data")
    if results:
        avg_coverage = sum(r['overall_coverage'] for r in results) / len(results)
        avg_confidence = sum(r.get('hebrew_stats', {}).get('confidence_stats', {}).get('avg_confidence', 0) for r in results) / len(results)
        print(f"✓ Average alignment coverage: {avg_coverage:.1f}%")
        print(f"✓ Average confidence score: {avg_confidence:.3f}")
        
        # Count total verses processed
        total_verses = sum(r.get('verse_count', 0) for r in results)
        print(f"✓ Total verses analyzed: {total_verses:,}")
    
    print("\nFull corpus analysis complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())