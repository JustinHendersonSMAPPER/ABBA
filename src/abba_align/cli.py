#!/usr/bin/env python3
"""
ABBA-Align CLI: Advanced Biblical Text Alignment Tool

This is a specialized tool for biblical scholars and translators that provides:
- Training alignment models with biblical-specific features
- Evaluating alignment quality
- Producing richly annotated translations
- Analyzing parallel passages and intertextual references
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

from .morphological_analyzer import MorphologicalAnalyzer
from .phrase_detector import BiblicalPhraseDetector
from .parallel_passage_aligner import ParallelPassageAligner
from .alignment_trainer import AlignmentTrainer
from .alignment_annotator import AlignmentAnnotator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ABBAAlignCLI:
    """Main CLI interface for ABBA-Align."""
    
    def __init__(self):
        self.parser = self._create_parser()
        
    def _create_parser(self):
        """Create the argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            description="ABBA-Align: Biblical Text Alignment Toolkit",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Train alignment model with full features
  abba-align train --source hebrew --target english --features all
  
  # Annotate a translation with alignment confidence
  abba-align annotate --input mybible.json --output annotated.json
  
  # Analyze parallel passages
  abba-align parallels --books "Matt,Mark,Luke" --output synoptic.json
  
  # Extract biblical phrases
  abba-align phrases --language hebrew --min-frequency 5
  
  # Evaluate alignment quality
  abba-align evaluate --model mymodel.json --test-set test.json
            """
        )
        
        parser.add_argument(
            '--version', 
            action='version', 
            version='ABBA-Align 1.0.0'
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        train_parser = subparsers.add_parser(
            'train',
            help='Train alignment models with biblical-specific features'
        )
        train_parser.add_argument(
            '--source',
            choices=['hebrew', 'greek', 'aramaic'],
            required=True,
            help='Source language'
        )
        train_parser.add_argument(
            '--target',
            required=True,
            help='Target language code (e.g., english, spanish)'
        )
        train_parser.add_argument(
            '--corpus-dir',
            type=Path,
            required=True,
            help='Directory containing training corpora'
        )
        train_parser.add_argument(
            '--features',
            nargs='+',
            choices=['morphology', 'phrases', 'syntax', 'semantics', 'discourse', 'all'],
            default=['all'],
            help='Features to include in training (default: all)'
        )
        train_parser.add_argument(
            '--parallel-passages',
            action='store_true',
            help='Use parallel passages for improved training'
        )
        train_parser.add_argument(
            '--output-dir',
            type=Path,
            default=Path('models/biblical_alignment'),
            help='Directory to save trained models'
        )
        
        # Annotate command
        annotate_parser = subparsers.add_parser(
            'annotate',
            help='Annotate translations with alignment information'
        )
        annotate_parser.add_argument(
            '--input',
            type=Path,
            required=True,
            help='Input translation file or directory'
        )
        annotate_parser.add_argument(
            '--output',
            type=Path,
            required=True,
            help='Output file or directory for annotated translation'
        )
        annotate_parser.add_argument(
            '--model',
            type=Path,
            help='Alignment model to use (auto-detected if not specified)'
        )
        annotate_parser.add_argument(
            '--confidence-threshold',
            type=float,
            default=0.3,
            help='Minimum confidence for alignments'
        )
        annotate_parser.add_argument(
            '--include-morphology',
            action='store_true',
            help='Include morphological analysis in annotations'
        )
        annotate_parser.add_argument(
            '--include-phrases',
            action='store_true',
            help='Detect and annotate biblical phrases'
        )
        
        # Analyze parallel passages
        parallel_parser = subparsers.add_parser(
            'parallels',
            help='Analyze parallel passages across biblical texts'
        )
        parallel_parser.add_argument(
            '--books',
            nargs='+',
            help='Books to analyze for parallels'
        )
        parallel_parser.add_argument(
            '--type',
            choices=['synoptic', 'chronicles', 'ot-quotes', 'all'],
            default='all',
            help='Type of parallel analysis'
        )
        parallel_parser.add_argument(
            '--output',
            type=Path,
            required=True,
            help='Output file for parallel analysis'
        )
        
        # Extract phrases
        phrase_parser = subparsers.add_parser(
            'phrases',
            help='Extract and analyze biblical phrases'
        )
        phrase_parser.add_argument(
            '--language',
            choices=['hebrew', 'greek', 'english'],
            required=True,
            help='Language to analyze'
        )
        phrase_parser.add_argument(
            '--corpus',
            type=Path,
            required=True,
            help='Corpus file or directory'
        )
        phrase_parser.add_argument(
            '--min-frequency',
            type=int,
            default=3,
            help='Minimum frequency for phrase extraction'
        )
        phrase_parser.add_argument(
            '--output',
            type=Path,
            required=True,
            help='Output file for phrases'
        )
        
        # Evaluate command
        eval_parser = subparsers.add_parser(
            'evaluate',
            help='Evaluate alignment model quality'
        )
        eval_parser.add_argument(
            '--model',
            type=Path,
            required=True,
            help='Model to evaluate'
        )
        eval_parser.add_argument(
            '--test-set',
            type=Path,
            required=True,
            help='Test set with gold alignments'
        )
        eval_parser.add_argument(
            '--metrics',
            nargs='+',
            choices=['precision', 'recall', 'f1', 'aer', 'all'],
            default=['all'],
            help='Metrics to calculate'
        )
        
        # Morphology analysis
        morph_parser = subparsers.add_parser(
            'morphology',
            help='Analyze morphological patterns'
        )
        morph_parser.add_argument(
            '--language',
            choices=['hebrew', 'greek', 'aramaic'],
            required=True,
            help='Language to analyze'
        )
        morph_parser.add_argument(
            '--input',
            type=Path,
            required=True,
            help='Input text or corpus'
        )
        morph_parser.add_argument(
            '--decompose',
            action='store_true',
            help='Perform full morphological decomposition'
        )
        morph_parser.add_argument(
            '--output',
            type=Path,
            help='Output file for analysis'
        )
        
        # Coverage analysis
        coverage_parser = subparsers.add_parser(
            'coverage',
            help='Analyze alignment model coverage on translations'
        )
        coverage_parser.add_argument(
            '--translation',
            type=Path,
            required=True,
            help='Translation file to analyze'
        )
        coverage_parser.add_argument(
            '--source-language',
            choices=['hebrew', 'greek'],
            default='hebrew',
            help='Source language of alignment'
        )
        coverage_parser.add_argument(
            '--model',
            type=Path,
            help='Alignment model to use (optional)'
        )
        coverage_parser.add_argument(
            '--report',
            type=Path,
            help='Path to save coverage report'
        )
        coverage_parser.add_argument(
            '--min-coverage',
            type=float,
            default=80.0,
            help='Minimum acceptable coverage percentage'
        )
        
        # Model management
        model_parser = subparsers.add_parser(
            'models',
            help='List and manage alignment models'
        )
        model_parser.add_argument(
            '--list',
            action='store_true',
            help='List all available models'
        )
        model_parser.add_argument(
            '--find',
            nargs=2,
            metavar=('SOURCE', 'TARGET'),
            help='Find model for specific language pair'
        )
        model_parser.add_argument(
            '--info',
            type=Path,
            help='Show detailed info about a specific model'
        )
        model_parser.add_argument(
            '--validate',
            type=Path,
            help='Validate a model file'
        )
        model_parser.add_argument(
            '--json',
            action='store_true',
            help='Output in JSON format'
        )
        
        return parser
    
    def train(self, args):
        """Train alignment models."""
        logger.info(f"Training {args.source}-{args.target} alignment model...")
        
        # Validate corpus directory
        if not args.corpus_dir.exists():
            logger.error(f"Corpus directory not found: {args.corpus_dir}")
            sys.exit(1)
            
        # Check for required corpus files
        source_corpus_name = {
            'hebrew': 'heb_bhs.json',
            'greek': 'grc_na28.json',
            'aramaic': 'arc_targum.json'
        }.get(args.source)
        
        source_corpus = args.corpus_dir / source_corpus_name
        if not source_corpus.exists():
            logger.error(f"Source corpus not found: {source_corpus}")
            logger.error(f"Please ensure {source_corpus_name} exists in {args.corpus_dir}")
            sys.exit(1)
        
        try:
            # Initialize trainer with requested features
            trainer = AlignmentTrainer(
                source_lang=args.source,
                target_lang=args.target,
                enable_morphology='morphology' in args.features or 'all' in args.features,
                enable_phrases='phrases' in args.features or 'all' in args.features,
                enable_syntax='syntax' in args.features or 'all' in args.features,
                enable_semantics='semantics' in args.features or 'all' in args.features,
                enable_discourse='discourse' in args.features or 'all' in args.features,
                enable_strongs='all' in args.features  # Strong's is part of 'all'
            )
        except FileNotFoundError as e:
            logger.error(f"Required file not found: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            sys.exit(1)
        
        # Load corpora
        try:
            trainer.load_corpora(args.corpus_dir)
        except FileNotFoundError as e:
            logger.error(f"Failed to load corpora: {e}")
            sys.exit(1)
        
        # Use parallel passages if requested
        if args.parallel_passages:
            logger.info("Incorporating parallel passages...")
            try:
                parallel_aligner = ParallelPassageAligner()
                parallel_data = parallel_aligner.extract_training_data(args.corpus_dir)
                trainer.add_parallel_data(parallel_data)
            except Exception as e:
                logger.warning(f"Failed to load parallel passages: {e}")
                logger.warning("Continuing without parallel passage data...")
        
        # Train the model
        try:
            model = trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
        
        # Save the model
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.output_dir / f"{args.source}_{args.target}_biblical.json"
        try:
            trainer.save_model(model, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            sys.exit(1)
        
        # Generate training report
        report = trainer.generate_report()
        report_path = args.output_dir / f"{args.source}_{args.target}_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Training report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save training report: {e}")
    
    def annotate(self, args):
        """Annotate translations with alignment information."""
        logger.info(f"Annotating {args.input}...")
        
        # Initialize annotator
        annotator = AlignmentAnnotator(
            confidence_threshold=args.confidence_threshold,
            include_morphology=args.include_morphology,
            include_phrases=args.include_phrases
        )
        
        # Load model - automatic selection if not specified
        if args.model:
            annotator.load_model(args.model)
        else:
            # Auto-select best model based on input
            logger.info("Auto-selecting best alignment model...")
            from .model_info import ModelDiscovery, auto_select_model
            
            # First try auto-detection from file content
            auto_model = auto_select_model(args.input)
            
            if auto_model:
                logger.info(f"Auto-selected model: {auto_model}")
                annotator.load_model(auto_model)
            else:
                # Fallback to asking user
                logger.error("Could not auto-detect source language.")
                logger.error("Please specify model with --model or ensure input file contains Hebrew/Greek text")
                sys.exit(1)
        
        # Process input
        if args.input.is_dir():
            # Process directory
            annotator.annotate_directory(args.input, args.output)
        else:
            # Process single file
            annotated = annotator.annotate_file(args.input)
            
            # Save output
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(annotated, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Annotation complete. Output saved to {args.output}")
    
    def analyze_parallels(self, args):
        """Analyze parallel passages."""
        logger.info(f"Analyzing parallel passages...")
        
        analyzer = ParallelPassageAligner()
        
        # Load texts
        if args.books:
            analyzer.load_books(args.books)
        
        # Perform analysis
        if args.type == 'synoptic':
            results = analyzer.analyze_synoptic_gospels()
        elif args.type == 'chronicles':
            results = analyzer.analyze_chronicles_kings()
        elif args.type == 'ot-quotes':
            results = analyzer.analyze_ot_quotes_in_nt()
        else:
            results = analyzer.analyze_all_parallels()
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Parallel analysis saved to {args.output}")
    
    def extract_phrases(self, args):
        """Extract biblical phrases."""
        logger.info(f"Extracting {args.language} phrases...")
        
        detector = BiblicalPhraseDetector(
            language=args.language,
            min_frequency=args.min_frequency
        )
        
        # Load corpus
        detector.load_corpus(args.corpus)
        
        # Extract phrases
        phrases = detector.extract_phrases()
        
        # Analyze and rank phrases
        ranked_phrases = detector.rank_phrases(phrases)
        
        # Save results
        output_data = {
            'language': args.language,
            'total_phrases': len(ranked_phrases),
            'min_frequency': args.min_frequency,
            'phrases': ranked_phrases
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Extracted {len(ranked_phrases)} phrases")
        logger.info(f"Results saved to {args.output}")
    
    def evaluate(self, args):
        """Evaluate model quality."""
        logger.info(f"Evaluating model {args.model}...")
        
        from .evaluation import AlignmentEvaluator
        
        evaluator = AlignmentEvaluator()
        evaluator.load_model(args.model)
        evaluator.load_test_set(args.test_set)
        
        # Calculate metrics
        results = {}
        if 'all' in args.metrics or 'precision' in args.metrics:
            results['precision'] = evaluator.calculate_precision()
        if 'all' in args.metrics or 'recall' in args.metrics:
            results['recall'] = evaluator.calculate_recall()
        if 'all' in args.metrics or 'f1' in args.metrics:
            results['f1'] = evaluator.calculate_f1()
        if 'all' in args.metrics or 'aer' in args.metrics:
            results['aer'] = evaluator.calculate_aer()
        
        # Print results
        print("\nEvaluation Results:")
        print("=" * 40)
        for metric, value in results.items():
            print(f"{metric.upper()}: {value:.3f}")
        
        # Detailed analysis
        print("\nDetailed Analysis:")
        print("=" * 40)
        evaluator.print_error_analysis()
    
    def analyze_morphology(self, args):
        """Analyze morphological patterns."""
        logger.info(f"Analyzing {args.language} morphology...")
        
        analyzer = MorphologicalAnalyzer(language=args.language)
        
        # Load input
        if args.input.suffix == '.json':
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results = analyzer.analyze_corpus(data)
        else:
            # Assume plain text
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            results = analyzer.analyze_text(text, decompose=args.decompose)
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Analysis saved to {args.output}")
        else:
            # Print to console
            analyzer.print_analysis(results)
    
    def analyze_coverage(self, args):
        """Analyze alignment coverage on a translation."""
        logger.info(f"Analyzing coverage for {args.translation}")
        
        from .coverage_analyzer import AlignmentCoverageAnalyzer
        from .model_info import ModelDiscovery
        
        # Initialize analyzer
        analyzer = AlignmentCoverageAnalyzer(source_lang=args.source_language)
        
        # Auto-select model if not specified
        model_path = args.model
        if not model_path:
            logger.info(f"Auto-selecting best {args.source_language} model...")
            discovery = ModelDiscovery()
            model = discovery.find_model(args.source_language, 'english')
            if model:
                model_path = model.path
                logger.info(f"Using model: {model_path}")
            else:
                logger.warning(f"No model found for {args.source_language}")
        
        # Analyze translation
        stats = analyzer.analyze_translation_coverage(
            args.translation,
            model_path
        )
        
        # Generate report
        report = analyzer.generate_coverage_report(stats, args.report)
        
        # Print summary
        summary = stats['summary']
        print("\nCOVERAGE SUMMARY")
        print("=" * 40)
        print(f"Token Coverage: {summary['token_coverage']:.1f}%")
        print(f"Type Coverage: {summary['type_coverage']:.1f}%")
        
        # Check minimum coverage
        if summary['token_coverage'] < args.min_coverage:
            logger.warning(
                f"Token coverage {summary['token_coverage']:.1f}% "
                f"is below minimum {args.min_coverage}%"
            )
            return False
        else:
            logger.info(
                f"Token coverage {summary['token_coverage']:.1f}% "
                f"meets minimum requirement"
            )
            return True
    
    def manage_models(self, args):
        """Manage alignment models."""
        from .model_info import ModelDiscovery, ModelInfo, auto_select_model
        
        if args.info:
            # Show info about specific model
            try:
                model_info = ModelInfo(args.info)
                
                if args.json:
                    print(json.dumps(model_info.to_dict(), indent=2))
                else:
                    print(f"Model: {model_info.name}")
                    print(f"Source: {model_info.source_lang}")
                    print(f"Target: {model_info.target_lang}")
                    
                    if model_info.features:
                        enabled = [k for k, v in model_info.features.items() if v]
                        print(f"Features: {', '.join(enabled)}")
                        
                    if model_info.statistics:
                        stats = model_info.statistics
                        total_entries = stats.get('hebrew_entries', 0) + stats.get('greek_entries', 0)
                        print(f"Strong's entries: {total_entries:,}")
                        print(f"Translation mappings: {stats.get('translation_mappings', 0):,}")
                        
                    print(f"Estimated coverage: {model_info.get_coverage_estimate():.0f}%")
                    
            except Exception as e:
                logger.error(f"Could not load model info: {e}")
                
        elif args.validate:
            # Validate model file
            try:
                with open(args.validate, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                    
                # Basic validation
                is_valid = True
                issues = []
                
                if 'trained' not in model_data:
                    issues.append("Missing 'trained' flag")
                    is_valid = False
                    
                if is_valid:
                    print(f"✓ Model {args.validate} is valid")
                else:
                    print(f"✗ Model {args.validate} has issues:")
                    for issue in issues:
                        print(f"  - {issue}")
                        
            except json.JSONDecodeError:
                print(f"✗ Model {args.validate} is not valid JSON")
            except Exception as e:
                print(f"✗ Error validating model: {e}")
                
        else:
            # List or find models
            discovery = ModelDiscovery()
            
            if args.find:
                source, target = args.find
                model = discovery.find_model(source, target)
                
                if model:
                    if args.json:
                        print(json.dumps(model.to_dict(), indent=2))
                    else:
                        print(f"Found model: {model.name}")
                        print(f"Path: {model.path}")
                        enabled = [k for k, v in model.features.items() if v]
                        if enabled:
                            print(f"Features: {', '.join(enabled)}")
                        print(f"Estimated coverage: {model.get_coverage_estimate():.0f}%")
                else:
                    print(f"No model found for {source} → {target}")
                    print(f"\nTrain one with:")
                    print(f"  abba-align train --source {source} --target {target}")
                    
            else:  # List all
                if args.json:
                    models = discovery.list_all_models()
                    print(json.dumps(models, indent=2))
                else:
                    print(discovery.generate_report())
    
    def run(self):
        """Run the CLI."""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        # Route to appropriate handler
        if args.command == 'train':
            self.train(args)
        elif args.command == 'annotate':
            self.annotate(args)
        elif args.command == 'parallels':
            self.analyze_parallels(args)
        elif args.command == 'phrases':
            self.extract_phrases(args)
        elif args.command == 'evaluate':
            self.evaluate(args)
        elif args.command == 'morphology':
            self.analyze_morphology(args)
        elif args.command == 'coverage':
            self.analyze_coverage(args)
        elif args.command == 'models':
            self.manage_models(args)


def main():
    """Main entry point."""
    cli = ABBAAlignCLI()
    cli.run()


if __name__ == "__main__":
    main()