#!/usr/bin/env python3
"""
Training script for the modern alignment system.

This script trains the complete modern alignment pipeline on the available
biblical corpus data and generates alignment models for production use.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.complete_modern_aligner import CompleteModernAligner
from src.abba.parsers.hebrew_parser import HebrewParser
from src.abba.parsers.greek_parser import GreekParser
from src.abba.parsers.translation_parser import TranslationParser
from src.abba.parsers.lexicon_parser import LexiconParser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlignmentTrainer:
    """Comprehensive training system for modern alignment pipeline."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        """Initialize the training system."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize parsers
        self.hebrew_parser = HebrewParser()
        self.greek_parser = GreekParser()
        self.translation_parser = TranslationParser()
        self.lexicon_parser = LexiconParser()
        
        # Initialize aligner
        self.aligner = CompleteModernAligner()
        
        # Training statistics
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "hebrew_verses_processed": 0,
            "greek_verses_processed": 0,
            "translation_verses_processed": 0,
            "total_alignments_created": 0,
            "strongs_mappings_built": 0,
            "phrase_patterns_identified": 0,
            "quality_metrics": {}
        }
    
    def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting complete alignment training pipeline...")
        self.training_stats["start_time"] = time.time()
        
        try:
            # Step 1: Load and parse all source data
            logger.info("Step 1: Loading source data...")
            hebrew_verses, greek_verses, translation_verses = self._load_all_data()
            
            # Step 2: Train the alignment system
            logger.info("Step 2: Training alignment models...")
            training_results = self._train_alignment_models(
                hebrew_verses, greek_verses, translation_verses
            )
            
            # Step 3: Generate sample alignments for validation
            logger.info("Step 3: Generating sample alignments...")
            sample_alignments = self._generate_sample_alignments(
                hebrew_verses[:10], greek_verses[:10], translation_verses[:20]
            )
            
            # Step 4: Run quality analysis
            logger.info("Step 4: Running quality analysis...")
            quality_report = self._analyze_alignment_quality(sample_alignments)
            
            # Step 5: Save training results
            logger.info("Step 5: Saving training results...")
            self._save_training_results(training_results, quality_report, sample_alignments)
            
            # Update final statistics
            self.training_stats["end_time"] = time.time()
            self.training_stats["duration_seconds"] = (
                self.training_stats["end_time"] - self.training_stats["start_time"]
            )
            self.training_stats.update(training_results)
            self.training_stats["quality_metrics"] = quality_report["summary"]
            
            logger.info("Training completed successfully!")
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _load_all_data(self) -> tuple:
        """Load all source data for training."""
        hebrew_verses = []
        greek_verses = []
        translation_verses = []
        
        # Load Hebrew data
        hebrew_dir = self.data_dir / "sources" / "hebrew"
        if hebrew_dir.exists():
            logger.info("Loading Hebrew data...")
            
            hebrew_files = list(hebrew_dir.glob("*.xml"))
            for i, hebrew_file in enumerate(hebrew_files[:5]):  # Limit for demo
                try:
                    verses = self.hebrew_parser.parse_file(hebrew_file)
                    hebrew_verses.extend(verses[:10])  # First 10 verses per book
                    
                    if i % 10 == 0:
                        logger.info(f"Processed {i+1}/{len(hebrew_files)} Hebrew files")
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {hebrew_file}: {e}")
            
            logger.info(f"Loaded {len(hebrew_verses)} Hebrew verses")
        
        # Load Greek data
        greek_dir = self.data_dir / "sources" / "greek"
        if greek_dir.exists():
            logger.info("Loading Greek data...")
            
            greek_files = list(greek_dir.glob("*.xml"))
            for i, greek_file in enumerate(greek_files[:5]):  # Limit for demo
                try:
                    # Extract book code from filename
                    book_code = self._extract_book_code_from_filename(greek_file.name)
                    if book_code:
                        verses = self.greek_parser.parse_file(greek_file, book_code)
                        greek_verses.extend(verses[:10])  # First 10 verses per book
                        
                        if i % 10 == 0:
                            logger.info(f"Processed {i+1}/{len(greek_files)} Greek files")
                            
                except Exception as e:
                    logger.warning(f"Failed to parse {greek_file}: {e}")
            
            logger.info(f"Loaded {len(greek_verses)} Greek verses")
        
        # Create sample translation verses (since we don't have aligned translation data)
        translation_verses = self._create_sample_translation_verses(
            hebrew_verses, greek_verses
        )
        
        self.training_stats["hebrew_verses_processed"] = len(hebrew_verses)
        self.training_stats["greek_verses_processed"] = len(greek_verses)
        self.training_stats["translation_verses_processed"] = len(translation_verses)
        
        return hebrew_verses, greek_verses, translation_verses
    
    def _extract_book_code_from_filename(self, filename: str) -> str:
        """Extract book code from Greek filename."""
        # Map common Greek filenames to book codes
        filename_map = {
            "MAT.xml": "MAT",
            "MAR.xml": "MRK", 
            "LUK.xml": "LUK",
            "JOH.xml": "JHN",
            "ACT.xml": "ACT",
            "ROM.xml": "ROM",
            "1CO.xml": "1CO",
            "2CO.xml": "2CO",
            "GAL.xml": "GAL",
            "EPH.xml": "EPH",
            "PHP.xml": "PHP",
            "COL.xml": "COL",
            "1TH.xml": "1TH",
            "2TH.xml": "2TH",
            "1TI.xml": "1TI",
            "2TI.xml": "2TI",
            "TIT.xml": "TIT",
            "PHM.xml": "PHM",
            "HEB.xml": "HEB",
            "JAM.xml": "JAS",
            "1PE.xml": "1PE",
            "2PE.xml": "2PE",
            "1JO.xml": "1JN",
            "2JO.xml": "2JN",
            "3JO.xml": "3JN",
            "JUD.xml": "JUD",
            "REV.xml": "REV"
        }
        
        return filename_map.get(filename, filename.replace(".xml", "").upper())
    
    def _create_sample_translation_verses(self, hebrew_verses, greek_verses):
        """Create sample translation verses for training."""
        from src.abba.parsers.translation_parser import TranslationVerse
        
        translation_verses = []
        
        # Sample translations for common verses
        sample_translations = {
            "GEN.1.1": "In the beginning God created the heavens and the earth.",
            "GEN.1.2": "The earth was without form and void, and darkness was over the face of the deep.",
            "GEN.1.3": "And God said, Let there be light, and there was light.",
            "PSA.23.1": "The LORD is my shepherd; I shall not want.",
            "JHN.1.1": "In the beginning was the Word, and the Word was with God, and the Word was God.",
            "JHN.1.2": "He was in the beginning with God.",
            "JHN.1.3": "All things were made through him, and without him was not any thing made that was made.",
            "JHN.3.16": "For God so loved the world, that he gave his only begotten Son.",
            "ROM.3.23": "For all have sinned and fall short of the glory of God.",
            "EPH.2.8": "For by grace you have been saved through faith."
        }
        
        # Create translation verses for available original verses
        for verse in hebrew_verses + greek_verses:
            verse_key = str(verse.verse_id)
            if verse_key in sample_translations:
                trans_verse = TranslationVerse(
                    verse_id=verse.verse_id,
                    text=sample_translations[verse_key],
                    original_book_name=verse.verse_id.book,
                    original_chapter=verse.verse_id.chapter,
                    original_verse=verse.verse_id.verse
                )
                translation_verses.append(trans_verse)
        
        return translation_verses
    
    def _train_alignment_models(self, hebrew_verses, greek_verses, translation_verses):
        """Train the alignment models on the corpus."""
        logger.info("Training alignment models on corpus...")
        
        training_results = self.aligner.train_on_corpus(
            hebrew_verses=hebrew_verses,
            greek_verses=greek_verses,
            translation_verses=translation_verses
        )
        
        logger.info(f"Training results: {training_results}")
        return training_results
    
    def _generate_sample_alignments(self, hebrew_verses, greek_verses, translation_verses):
        """Generate sample alignments for quality analysis."""
        logger.info("Generating sample alignments...")
        
        sample_alignments = {}
        translation_map = {str(tv.verse_id): tv for tv in translation_verses}
        
        # Process Hebrew verses
        for verse in hebrew_verses:
            verse_key = str(verse.verse_id)
            if verse_key in translation_map:
                try:
                    alignments = self.aligner.align_verse_complete(
                        verse, translation_map[verse_key]
                    )
                    sample_alignments[verse_key] = alignments
                    
                    # Add to cross-validator for consistency analysis
                    self.aligner.cross_validator.add_translation_alignments(
                        "sample_translation", verse_key, alignments
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to align {verse_key}: {e}")
        
        # Process Greek verses
        for verse in greek_verses:
            verse_key = str(verse.verse_id)
            if verse_key in translation_map:
                try:
                    alignments = self.aligner.align_verse_complete(
                        verse, translation_map[verse_key]
                    )
                    sample_alignments[verse_key] = alignments
                    
                    # Add to cross-validator
                    self.aligner.cross_validator.add_translation_alignments(
                        "sample_translation", verse_key, alignments
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to align {verse_key}: {e}")
        
        self.training_stats["total_alignments_created"] = sum(
            len(alignments) for alignments in sample_alignments.values()
        )
        
        logger.info(f"Generated {len(sample_alignments)} verse alignments")
        return sample_alignments
    
    def _analyze_alignment_quality(self, sample_alignments):
        """Analyze the quality of generated alignments."""
        logger.info("Analyzing alignment quality...")
        
        # Generate comprehensive quality report
        quality_report = self.aligner.generate_comprehensive_report(sample_alignments)
        
        # Add additional analysis
        quality_report["training_specific"] = {
            "training_data_size": {
                "hebrew_verses": self.training_stats["hebrew_verses_processed"],
                "greek_verses": self.training_stats["greek_verses_processed"],
                "translation_verses": self.training_stats["translation_verses_processed"]
            },
            "model_coverage": {
                "strongs_mappings": len(self.aligner.strongs_mappings),
                "phrase_patterns": len(self.aligner.phrase_patterns)
            }
        }
        
        # Identify top improvement opportunities
        uncertain_alignments = []
        for verse_alignments in sample_alignments.values():
            uncertain_alignments.extend(
                self.aligner.active_learning.identify_uncertain_alignments(
                    verse_alignments, uncertainty_threshold=0.7
                )
            )
        
        quality_report["improvement_opportunities"] = {
            "total_uncertain_alignments": len(uncertain_alignments),
            "top_uncertain_cases": [
                {
                    "verse_id": alignment.verse_id,
                    "source_text": alignment.get_source_text(),
                    "target_text": alignment.get_target_text(),
                    "confidence": alignment.confidence.overall_score,
                    "uncertainty_factors": alignment.confidence.uncertainty_factors
                }
                for alignment in uncertain_alignments[:10]
            ]
        }
        
        logger.info("Quality analysis completed")
        return quality_report
    
    def _save_training_results(self, training_results, quality_report, sample_alignments):
        """Save all training results to files."""
        logger.info("Saving training results...")
        
        # Save training statistics
        training_file = self.output_dir / "training_statistics.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, indent=2, ensure_ascii=False)
        
        # Save quality report
        quality_file = self.output_dir / "quality_report.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        # Save sample alignments
        alignments_file = self.output_dir / "sample_alignments.json"
        serializable_alignments = {
            verse_id: [alignment.to_dict() for alignment in alignments]
            for verse_id, alignments in sample_alignments.items()
        }
        with open(alignments_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_alignments, f, indent=2, ensure_ascii=False)
        
        # Save Strong's mappings
        strongs_file = self.output_dir / "strongs_mappings.json"
        with open(strongs_file, 'w', encoding='utf-8') as f:
            json.dump(self.aligner.strongs_mappings, f, indent=2, ensure_ascii=False)
        
        # Generate human-readable summary
        self._generate_training_summary()
        
        logger.info(f"Training results saved to {self.output_dir}")
    
    def _generate_training_summary(self):
        """Generate human-readable training summary."""
        summary_file = self.output_dir / "training_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ABBA Modern Alignment Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Training Statistics:\n")
            f.write(f"  Duration: {self.training_stats['duration_seconds']:.1f} seconds\n")
            f.write(f"  Hebrew verses processed: {self.training_stats['hebrew_verses_processed']}\n")
            f.write(f"  Greek verses processed: {self.training_stats['greek_verses_processed']}\n")
            f.write(f"  Translation verses processed: {self.training_stats['translation_verses_processed']}\n")
            f.write(f"  Total alignments created: {self.training_stats['total_alignments_created']}\n")
            f.write(f"  Strong's mappings built: {self.training_stats.get('strongs_mappings_built', 0)}\n\n")
            
            if "quality_metrics" in self.training_stats:
                f.write("Quality Metrics:\n")
                quality = self.training_stats["quality_metrics"]
                f.write(f"  Average confidence: {quality.get('average_confidence', 0):.2f}\n")
                f.write(f"  Total verses processed: {quality.get('total_verses', 0)}\n")
                f.write(f"  Total alignments: {quality.get('total_alignments', 0)}\n\n")
            
            f.write("Training Status: COMPLETED SUCCESSFULLY\n")
            f.write("Next Steps:\n")
            f.write("  1. Review quality report for improvement opportunities\n")
            f.write("  2. Run validation script to test on new data\n")
            f.write("  3. Deploy trained models to production system\n")


def main():
    """Main training function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "data" / "training_output"
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize trainer
    trainer = AlignmentTrainer(data_dir, output_dir)
    
    try:
        # Run complete training
        results = trainer.run_complete_training()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Hebrew verses: {results['hebrew_verses_processed']}")
        print(f"Greek verses: {results['greek_verses_processed']}")
        print(f"Translation verses: {results['translation_verses_processed']}")
        print(f"Total alignments: {results['total_alignments_created']}")
        print(f"Strong's mappings: {results.get('strongs_mappings_built', 0)}")
        
        if "quality_metrics" in results:
            quality = results["quality_metrics"]
            print(f"Average confidence: {quality.get('average_confidence', 0):.2f}")
        
        print(f"\nResults saved to: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nTRAINING FAILED: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())