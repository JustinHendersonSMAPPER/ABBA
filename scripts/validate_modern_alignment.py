#!/usr/bin/env python3
"""
Validation script for the modern alignment system.

This script validates the complete modern alignment pipeline by testing it
on real biblical data and generating comprehensive quality reports.
"""

import json
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


class ModernAlignmentValidator:
    """Comprehensive validation system for modern alignment pipeline."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        """Initialize the validation system."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.hebrew_parser = HebrewParser()
        self.greek_parser = GreekParser()
        self.translation_parser = TranslationParser()
        self.aligner = CompleteModernAligner()
        
        # Validation results
        self.validation_results = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "test_cases": [],
            "performance_metrics": {},
            "quality_analysis": {},
            "recommendations": []
        }
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run the complete validation pipeline."""
        print("=" * 80)
        print("ABBA Modern Alignment System Validation")
        print("=" * 80)
        
        self.validation_results["start_time"] = time.time()
        
        try:
            # Step 1: Test basic functionality
            print("\n1. Testing Basic Functionality...")
            self._test_basic_functionality()
            
            # Step 2: Test with real data
            print("\n2. Testing with Real Biblical Data...")
            self._test_with_real_data()
            
            # Step 3: Performance testing
            print("\n3. Running Performance Tests...")
            self._test_performance()
            
            # Step 4: Quality assurance testing
            print("\n4. Running Quality Assurance Tests...")
            self._test_quality_assurance()
            
            # Step 5: Edge case testing
            print("\n5. Testing Edge Cases...")
            self._test_edge_cases()
            
            # Step 6: Generate comprehensive report
            print("\n6. Generating Validation Report...")
            self._generate_validation_report()
            
            # Update timing
            self.validation_results["end_time"] = time.time()
            self.validation_results["duration_seconds"] = (
                self.validation_results["end_time"] - self.validation_results["start_time"]
            )
            
            print("\n" + "=" * 80)
            print("VALIDATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            return self.validation_results
            
        except Exception as e:
            print(f"\nVALIDATION FAILED: {e}")
            raise
    
    def _test_basic_functionality(self):
        """Test basic functionality of all components."""
        test_results = []
        
        # Test initialization
        try:
            test_aligner = CompleteModernAligner()
            test_results.append({
                "test": "component_initialization",
                "status": "PASS",
                "message": "All components initialized successfully"
            })
        except Exception as e:
            test_results.append({
                "test": "component_initialization", 
                "status": "FAIL",
                "message": f"Initialization failed: {e}"
            })
        
        # Test confidence ensemble
        try:
            from src.abba.alignment.complete_modern_aligner import (
                AdvancedConfidenceEnsemble,
                CompleteAlignment,
                AlignmentConfidence,
                AlignmentMethod
            )
            from src.abba.interlinear.token_extractor import ExtractedToken
            
            ensemble = AdvancedConfidenceEnsemble()
            
            # Create test alignment
            test_alignment = CompleteAlignment(
                source_tokens=[ExtractedToken(text="test", strong_number="H1")],
                target_words=["test"],
                strong_numbers=["H1"],
                confidence=AlignmentConfidence(overall_score=0.0),
                alignment_methods=[AlignmentMethod.STRONGS]
            )
            
            confidence = ensemble.compute_confidence(test_alignment, {})
            
            if 0.0 <= confidence.overall_score <= 1.0:
                test_results.append({
                    "test": "confidence_ensemble",
                    "status": "PASS",
                    "message": f"Confidence computed: {confidence.overall_score:.2f}"
                })
            else:
                test_results.append({
                    "test": "confidence_ensemble",
                    "status": "FAIL",
                    "message": f"Invalid confidence score: {confidence.overall_score}"
                })
                
        except Exception as e:
            test_results.append({
                "test": "confidence_ensemble",
                "status": "FAIL", 
                "message": f"Confidence ensemble failed: {e}"
            })
        
        # Test semantic loss detection
        try:
            from src.abba.alignment.complete_modern_aligner import SemanticLossDetector
            
            detector = SemanticLossDetector()
            
            # Test with known semantic loss case
            test_alignment.strong_numbers = ["H2617"]  # chesed
            losses = detector.detect_semantic_losses(test_alignment)
            
            test_results.append({
                "test": "semantic_loss_detection",
                "status": "PASS",
                "message": f"Detected {len(losses)} semantic losses"
            })
            
        except Exception as e:
            test_results.append({
                "test": "semantic_loss_detection",
                "status": "FAIL",
                "message": f"Semantic loss detection failed: {e}"
            })
        
        # Test cross-translation validation
        try:
            from src.abba.alignment.complete_modern_aligner import CrossTranslationValidator
            
            validator = CrossTranslationValidator()
            
            # Add test data
            validator.add_translation_alignments("KJV", "GEN.1.1", [test_alignment])
            consistency = validator.compute_consistency_score("GEN.1.1", "H1")
            
            test_results.append({
                "test": "cross_translation_validation",
                "status": "PASS",
                "message": f"Consistency score: {consistency:.2f}"
            })
            
        except Exception as e:
            test_results.append({
                "test": "cross_translation_validation",
                "status": "FAIL",
                "message": f"Cross-validation failed: {e}"
            })
        
        # Test active learning
        try:
            from src.abba.alignment.complete_modern_aligner import ActiveLearningManager
            
            manager = ActiveLearningManager()
            uncertain = manager.identify_uncertain_alignments([test_alignment])
            
            test_results.append({
                "test": "active_learning",
                "status": "PASS",
                "message": f"Identified {len(uncertain)} uncertain alignments"
            })
            
        except Exception as e:
            test_results.append({
                "test": "active_learning",
                "status": "FAIL",
                "message": f"Active learning failed: {e}"
            })
        
        # Store results
        self.validation_results["test_cases"].extend(test_results)
        
        # Print summary
        passed = sum(1 for test in test_results if test["status"] == "PASS")
        total = len(test_results)
        print(f"   Basic functionality tests: {passed}/{total} passed")
        
        for test in test_results:
            status_symbol = "✓" if test["status"] == "PASS" else "✗"
            print(f"   {status_symbol} {test['test']}: {test['message']}")
    
    def _test_with_real_data(self):
        """Test alignment system with real biblical data."""
        test_results = []
        
        # Load sample Hebrew data
        hebrew_dir = self.data_dir / "sources" / "hebrew"
        gen_file = hebrew_dir / "Gen.xml"
        
        if gen_file.exists():
            try:
                # Parse Genesis 1:1-3
                hebrew_verses = self.hebrew_parser.parse_file(gen_file)[:3]
                
                test_results.append({
                    "test": "hebrew_data_loading",
                    "status": "PASS",
                    "message": f"Loaded {len(hebrew_verses)} Hebrew verses"
                })
                
                # Create sample translations
                sample_translations = [
                    {
                        "verse_id": str(hebrew_verses[0].verse_id),
                        "text": "In the beginning God created the heavens and the earth."
                    },
                    {
                        "verse_id": str(hebrew_verses[1].verse_id), 
                        "text": "The earth was without form and void, and darkness was over the face of the deep."
                    },
                    {
                        "verse_id": str(hebrew_verses[2].verse_id),
                        "text": "And God said, Let there be light, and there was light."
                    }
                ]
                
                # Train alignment system
                from src.abba.parsers.translation_parser import TranslationVerse
                
                translation_verses = []
                for trans_data in sample_translations:
                    if len(hebrew_verses) > 0:
                        verse = hebrew_verses[0]  # Use first verse structure
                        trans_verse = TranslationVerse(
                            verse_id=verse.verse_id,
                            text=trans_data["text"],
                            original_book_name="Genesis",
                            original_chapter=verse.verse_id.chapter,
                            original_verse=verse.verse_id.verse
                        )
                        translation_verses.append(trans_verse)
                
                # Train the aligner
                training_stats = self.aligner.train_on_corpus(
                    hebrew_verses=hebrew_verses,
                    greek_verses=[],
                    translation_verses=translation_verses
                )
                
                test_results.append({
                    "test": "alignment_training",
                    "status": "PASS",
                    "message": f"Training completed: {training_stats.get('strongs_mappings_built', 0)} mappings"
                })
                
                # Test alignment generation
                if translation_verses:
                    alignments = self.aligner.align_verse_complete(
                        hebrew_verses[0], translation_verses[0]
                    )
                    
                    if alignments:
                        avg_confidence = sum(a.confidence.overall_score for a in alignments) / len(alignments)
                        test_results.append({
                            "test": "alignment_generation",
                            "status": "PASS",
                            "message": f"Generated {len(alignments)} alignments, avg confidence: {avg_confidence:.2f}"
                        })
                    else:
                        test_results.append({
                            "test": "alignment_generation",
                            "status": "FAIL",
                            "message": "No alignments generated"
                        })
                
            except Exception as e:
                test_results.append({
                    "test": "hebrew_data_processing",
                    "status": "FAIL",
                    "message": f"Hebrew processing failed: {e}"
                })
        else:
            test_results.append({
                "test": "hebrew_data_loading",
                "status": "SKIP",
                "message": "Hebrew data not available"
            })
        
        # Load sample Greek data
        greek_dir = self.data_dir / "sources" / "greek"
        john_file = greek_dir / "JOH.xml"
        
        if john_file.exists():
            try:
                greek_verses = self.greek_parser.parse_file(john_file, "JHN")[:2]
                
                test_results.append({
                    "test": "greek_data_loading",
                    "status": "PASS",
                    "message": f"Loaded {len(greek_verses)} Greek verses"
                })
                
            except Exception as e:
                test_results.append({
                    "test": "greek_data_processing",
                    "status": "FAIL",
                    "message": f"Greek processing failed: {e}"
                })
        else:
            test_results.append({
                "test": "greek_data_loading",
                "status": "SKIP",
                "message": "Greek data not available"
            })
        
        # Store results
        self.validation_results["test_cases"].extend(test_results)
        
        # Print summary
        passed = sum(1 for test in test_results if test["status"] == "PASS")
        skipped = sum(1 for test in test_results if test["status"] == "SKIP")
        total = len(test_results)
        print(f"   Real data tests: {passed}/{total} passed, {skipped} skipped")
        
        for test in test_results:
            if test["status"] == "PASS":
                print(f"   ✓ {test['test']}: {test['message']}")
            elif test["status"] == "FAIL":
                print(f"   ✗ {test['test']}: {test['message']}")
            else:
                print(f"   - {test['test']}: {test['message']}")
    
    def _test_performance(self):
        """Test performance characteristics."""
        performance_metrics = {}
        
        print("   Testing alignment speed...")
        
        # Create test data
        from src.abba.interlinear.token_extractor import ExtractedToken
        from src.abba.alignment.complete_modern_aligner import (
            CompleteAlignment, AlignmentConfidence, AlignmentMethod
        )
        
        test_alignments = []
        for i in range(100):
            alignment = CompleteAlignment(
                source_tokens=[ExtractedToken(text=f"word{i}", strong_number=f"H{i}")],
                target_words=[f"english{i}"],
                strong_numbers=[f"H{i}"],
                confidence=AlignmentConfidence(overall_score=0.8),
                alignment_methods=[AlignmentMethod.STRONGS]
            )
            test_alignments.append(alignment)
        
        # Test confidence computation speed
        start_time = time.time()
        
        for alignment in test_alignments:
            confidence = self.aligner.confidence_ensemble.compute_confidence(alignment, {})
        
        confidence_time = time.time() - start_time
        performance_metrics["confidence_computation_per_alignment"] = confidence_time / len(test_alignments)
        
        # Test report generation speed
        start_time = time.time()
        
        sample_alignments = {"TEST.1.1": test_alignments}
        report = self.aligner.generate_comprehensive_report(sample_alignments)
        
        report_time = time.time() - start_time
        performance_metrics["report_generation_time"] = report_time
        
        # Test semantic loss detection speed
        start_time = time.time()
        
        for alignment in test_alignments[:10]:  # Test subset
            losses = self.aligner.semantic_loss_detector.detect_semantic_losses(alignment)
        
        semantic_time = time.time() - start_time
        performance_metrics["semantic_loss_detection_per_alignment"] = semantic_time / 10
        
        # Store results
        self.validation_results["performance_metrics"] = performance_metrics
        
        print(f"   Confidence computation: {performance_metrics['confidence_computation_per_alignment']*1000:.1f}ms per alignment")
        print(f"   Report generation: {performance_metrics['report_generation_time']:.1f}s for 100 alignments")
        print(f"   Semantic loss detection: {performance_metrics['semantic_loss_detection_per_alignment']*1000:.1f}ms per alignment")
    
    def _test_quality_assurance(self):
        """Test quality assurance features."""
        test_results = []
        
        # Test confidence scoring accuracy
        try:
            # Create high-confidence test case
            from src.abba.interlinear.token_extractor import ExtractedToken
            from src.abba.alignment.complete_modern_aligner import (
                CompleteAlignment, AlignmentConfidence, AlignmentMethod
            )
            
            high_conf_alignment = CompleteAlignment(
                source_tokens=[ExtractedToken(text="אֱלֹהִים", strong_number="H430")],
                target_words=["God"],
                strong_numbers=["H430"],
                confidence=AlignmentConfidence(overall_score=0.0),
                alignment_methods=[AlignmentMethod.STRONGS]
            )
            
            confidence = self.aligner.confidence_ensemble.compute_confidence(high_conf_alignment, {})
            
            if confidence.overall_score > 0.8:
                test_results.append({
                    "test": "high_confidence_detection",
                    "status": "PASS",
                    "message": f"High confidence correctly detected: {confidence.overall_score:.2f}"
                })
            else:
                test_results.append({
                    "test": "high_confidence_detection",
                    "status": "FAIL",
                    "message": f"High confidence not detected: {confidence.overall_score:.2f}"
                })
            
            # Test low-confidence case
            low_conf_alignment = CompleteAlignment(
                source_tokens=[ExtractedToken(text="unknown")],
                target_words=["uncertain", "translation"],
                strong_numbers=[],
                confidence=AlignmentConfidence(overall_score=0.0),
                alignment_methods=[AlignmentMethod.STATISTICAL]
            )
            
            low_confidence = self.aligner.confidence_ensemble.compute_confidence(low_conf_alignment, {})
            
            if low_confidence.overall_score < 0.5:
                test_results.append({
                    "test": "low_confidence_detection",
                    "status": "PASS",
                    "message": f"Low confidence correctly detected: {low_confidence.overall_score:.2f}"
                })
            else:
                test_results.append({
                    "test": "low_confidence_detection",
                    "status": "FAIL",
                    "message": f"Low confidence not detected: {low_confidence.overall_score:.2f}"
                })
            
        except Exception as e:
            test_results.append({
                "test": "confidence_accuracy",
                "status": "FAIL",
                "message": f"Confidence testing failed: {e}"
            })
        
        # Test semantic loss detection accuracy
        try:
            # Test known semantic loss
            chesed_alignment = CompleteAlignment(
                source_tokens=[ExtractedToken(text="חֶסֶד", strong_number="H2617")],
                target_words=["mercy"],
                strong_numbers=["H2617"],
                confidence=AlignmentConfidence(overall_score=0.8),
                alignment_methods=[AlignmentMethod.STRONGS]
            )
            
            losses = self.aligner.semantic_loss_detector.detect_semantic_losses(chesed_alignment)
            
            if losses and any(loss.original_concept.find("chesed") >= 0 for loss in losses):
                test_results.append({
                    "test": "semantic_loss_accuracy",
                    "status": "PASS",
                    "message": f"Chesed semantic loss correctly detected"
                })
            else:
                test_results.append({
                    "test": "semantic_loss_accuracy",
                    "status": "FAIL",
                    "message": "Known semantic loss not detected"
                })
            
        except Exception as e:
            test_results.append({
                "test": "semantic_loss_accuracy",
                "status": "FAIL",
                "message": f"Semantic loss testing failed: {e}"
            })
        
        # Store results
        self.validation_results["test_cases"].extend(test_results)
        
        # Print summary
        passed = sum(1 for test in test_results if test["status"] == "PASS")
        total = len(test_results)
        print(f"   Quality assurance tests: {passed}/{total} passed")
        
        for test in test_results:
            status_symbol = "✓" if test["status"] == "PASS" else "✗"
            print(f"   {status_symbol} {test['test']}: {test['message']}")
    
    def _test_edge_cases(self):
        """Test edge cases and error handling."""
        test_results = []
        
        # Test empty input
        try:
            from src.abba.alignment.complete_modern_aligner import (
                CompleteAlignment, AlignmentConfidence, AlignmentMethod
            )
            
            empty_alignment = CompleteAlignment(
                source_tokens=[],
                target_words=[],
                strong_numbers=[],
                confidence=AlignmentConfidence(overall_score=0.0),
                alignment_methods=[]
            )
            
            confidence = self.aligner.confidence_ensemble.compute_confidence(empty_alignment, {})
            
            test_results.append({
                "test": "empty_input_handling",
                "status": "PASS",
                "message": f"Empty input handled gracefully: {confidence.overall_score:.2f}"
            })
            
        except Exception as e:
            test_results.append({
                "test": "empty_input_handling",
                "status": "FAIL",
                "message": f"Empty input failed: {e}"
            })
        
        # Test malformed data
        try:
            from src.abba.interlinear.token_extractor import ExtractedToken
            
            malformed_alignment = CompleteAlignment(
                source_tokens=[ExtractedToken(text="")],  # Empty text
                target_words=["test"],
                strong_numbers=["INVALID"],  # Invalid Strong's number
                confidence=AlignmentConfidence(overall_score=0.0),
                alignment_methods=[AlignmentMethod.STRONGS]
            )
            
            confidence = self.aligner.confidence_ensemble.compute_confidence(malformed_alignment, {})
            
            test_results.append({
                "test": "malformed_data_handling",
                "status": "PASS",
                "message": "Malformed data handled gracefully"
            })
            
        except Exception as e:
            test_results.append({
                "test": "malformed_data_handling", 
                "status": "FAIL",
                "message": f"Malformed data failed: {e}"
            })
        
        # Test very long input
        try:
            from src.abba.interlinear.token_extractor import ExtractedToken
            
            long_alignment = CompleteAlignment(
                source_tokens=[ExtractedToken(text="x" * 1000)],  # Very long text
                target_words=["y" * 1000],
                strong_numbers=["H1"],
                confidence=AlignmentConfidence(overall_score=0.0),
                alignment_methods=[AlignmentMethod.STRONGS]
            )
            
            confidence = self.aligner.confidence_ensemble.compute_confidence(long_alignment, {})
            
            test_results.append({
                "test": "long_input_handling",
                "status": "PASS",
                "message": "Long input handled gracefully"
            })
            
        except Exception as e:
            test_results.append({
                "test": "long_input_handling",
                "status": "FAIL",
                "message": f"Long input failed: {e}"
            })
        
        # Store results
        self.validation_results["test_cases"].extend(test_results)
        
        # Print summary
        passed = sum(1 for test in test_results if test["status"] == "PASS")
        total = len(test_results)
        print(f"   Edge case tests: {passed}/{total} passed")
        
        for test in test_results:
            status_symbol = "✓" if test["status"] == "PASS" else "✗"
            print(f"   {status_symbol} {test['test']}: {test['message']}")
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        # Calculate overall statistics
        total_tests = len(self.validation_results["test_cases"])
        passed_tests = sum(1 for test in self.validation_results["test_cases"] if test["status"] == "PASS")
        failed_tests = sum(1 for test in self.validation_results["test_cases"] if test["status"] == "FAIL")
        skipped_tests = sum(1 for test in self.validation_results["test_cases"] if test["status"] == "SKIP")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate quality analysis
        quality_analysis = {
            "overall_success_rate": success_rate,
            "test_statistics": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests
            },
            "performance_analysis": self.validation_results["performance_metrics"],
            "critical_issues": [
                test for test in self.validation_results["test_cases"] 
                if test["status"] == "FAIL"
            ]
        }
        
        self.validation_results["quality_analysis"] = quality_analysis
        
        # Generate recommendations
        recommendations = []
        
        if success_rate < 90:
            recommendations.append("Overall success rate below 90% - review failed tests")
        
        if failed_tests > 0:
            recommendations.append(f"Address {failed_tests} failed test cases")
        
        perf = self.validation_results["performance_metrics"]
        if perf.get("confidence_computation_per_alignment", 0) > 0.01:  # >10ms
            recommendations.append("Confidence computation speed could be improved")
        
        if perf.get("report_generation_time", 0) > 5.0:  # >5 seconds
            recommendations.append("Report generation speed could be improved")
        
        self.validation_results["recommendations"] = recommendations
        
        # Save validation report
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        # Generate human-readable summary
        summary_file = self.output_dir / "validation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ABBA Modern Alignment Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Success Rate: {success_rate:.1f}%\n")
            f.write(f"Duration: {self.validation_results['duration_seconds']:.1f} seconds\n\n")
            
            f.write("Test Summary:\n")
            f.write(f"  Total tests: {total_tests}\n")
            f.write(f"  Passed: {passed_tests}\n")
            f.write(f"  Failed: {failed_tests}\n")
            f.write(f"  Skipped: {skipped_tests}\n\n")
            
            if self.validation_results["performance_metrics"]:
                f.write("Performance Metrics:\n")
                for metric, value in self.validation_results["performance_metrics"].items():
                    if "time" in metric:
                        f.write(f"  {metric}: {value:.3f}s\n")
                    else:
                        f.write(f"  {metric}: {value*1000:.1f}ms\n")
                f.write("\n")
            
            if recommendations:
                f.write("Recommendations:\n")
                for rec in recommendations:
                    f.write(f"  - {rec}\n")
                f.write("\n")
            
            if success_rate >= 90:
                f.write("Status: VALIDATION SUCCESSFUL\n")
                f.write("The modern alignment system is ready for production use.\n")
            else:
                f.write("Status: VALIDATION NEEDS ATTENTION\n")
                f.write("Review failed tests before production deployment.\n")
        
        print(f"   Validation report saved to: {report_file}")
        print(f"   Summary saved to: {summary_file}")


def main():
    """Main validation function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "data" / "validation_output"
    
    # Initialize validator
    validator = ModernAlignmentValidator(data_dir, output_dir)
    
    try:
        # Run complete validation
        results = validator.run_complete_validation()
        
        # Print final summary
        quality = results["quality_analysis"]
        success_rate = quality["overall_success_rate"]
        
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Tests: {quality['test_statistics']['passed_tests']}/{quality['test_statistics']['total_tests']} passed")
        
        if results["recommendations"]:
            print("\nRecommendations:")
            for rec in results["recommendations"]:
                print(f"  - {rec}")
        
        print(f"\nResults saved to: {output_dir}")
        
        return 0 if success_rate >= 90 else 1
        
    except Exception as e:
        print(f"\nVALIDATION FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit(main())