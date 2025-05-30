"""
Comprehensive test suite for the complete modern alignment system.

Tests all components including confidence ensemble, cross-validation,
semantic loss detection, and active learning.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.complete_modern_aligner import (
    CompleteModernAligner,
    AdvancedConfidenceEnsemble,
    CrossTranslationValidator,
    SemanticLossDetector,
    ActiveLearningManager,
    CompleteAlignment,
    AlignmentConfidence,
    AlignmentMethod,
    SemanticLoss,
    SemanticLossType,
)
from src.abba.interlinear.token_extractor import ExtractedToken
from src.abba.morphology import UnifiedMorphology, Language, MorphologyFeatures
from src.abba.verse_id import create_verse_id
from src.abba.parsers.hebrew_parser import HebrewVerse, HebrewWord
from src.abba.parsers.greek_parser import GreekVerse, GreekWord
from src.abba.parsers.translation_parser import TranslationVerse


class TestCompleteModernAligner(unittest.TestCase):
    """Test the complete modern alignment system."""

    def setUp(self):
        """Set up test fixtures."""
        self.aligner = CompleteModernAligner()

        # Create sample Hebrew verse
        self.hebrew_verse = self._create_sample_hebrew_verse()

        # Create sample Greek verse
        self.greek_verse = self._create_sample_greek_verse()

        # Create sample translation verses
        self.translation_verses = self._create_sample_translation_verses()

    def _create_sample_hebrew_verse(self) -> HebrewVerse:
        """Create a sample Hebrew verse for testing."""
        verse_id = create_verse_id("GEN", 1, 1)

        # Create Hebrew words with morphology
        words = [
            HebrewWord(
                text="בְּרֵאשִׁית", lemma="רֵאשִׁית", strong_number="H7225", morph="HR/Ncfsa", id="w1"
            ),
            HebrewWord(text="בָּרָא", lemma="בָּרָא", strong_number="H1254", morph="HVqp3ms", id="w2"),
            HebrewWord(text="אֱלֹהִים", lemma="אֱלֹהִים", strong_number="H430", morph="HNcmpa", id="w3"),
        ]

        return HebrewVerse(verse_id=verse_id, words=words, osis_id="Gen.1.1")

    def _create_sample_greek_verse(self) -> GreekVerse:
        """Create a sample Greek verse for testing."""
        verse_id = create_verse_id("JHN", 1, 1)

        # Create Greek words
        words = [
            GreekWord(text="Ἐν", lemma="ἐν", strong_number="G1722", morph="PREP", id="w1"),
            GreekWord(text="ἀρχῇ", lemma="ἀρχή", strong_number="G746", morph="N-DSF", id="w2"),
            GreekWord(text="ἦν", lemma="εἰμί", strong_number="G1510", morph="V-IAI-3S", id="w3"),
            GreekWord(text="ὁ", lemma="ὁ", strong_number="G3588", morph="T-NSM", id="w4"),
            GreekWord(text="λόγος", lemma="λόγος", strong_number="G3056", morph="N-NSM", id="w5"),
        ]

        return GreekVerse(verse_id=verse_id, words=words, tei_id="B04K1V1")

    def _create_sample_translation_verses(self) -> list[TranslationVerse]:
        """Create sample translation verses."""
        gen_verse_id = create_verse_id("GEN", 1, 1)
        john_verse_id = create_verse_id("JHN", 1, 1)

        return [
            TranslationVerse(
                verse_id=gen_verse_id,
                text="In the beginning God created the heavens and the earth.",
                original_book_name="Genesis",
                original_chapter=1,
                original_verse=1,
            ),
            TranslationVerse(
                verse_id=john_verse_id,
                text="In the beginning was the Word, and the Word was with God, and the Word was God.",
                original_book_name="John",
                original_chapter=1,
                original_verse=1,
            ),
        ]

    def test_initialization(self):
        """Test proper initialization of the aligner."""
        self.assertIsInstance(self.aligner.confidence_ensemble, AdvancedConfidenceEnsemble)
        self.assertIsInstance(self.aligner.cross_validator, CrossTranslationValidator)
        self.assertIsInstance(self.aligner.semantic_loss_detector, SemanticLossDetector)
        self.assertIsInstance(self.aligner.active_learning, ActiveLearningManager)

    def test_training_on_corpus(self):
        """Test training the aligner on a parallel corpus."""
        training_stats = self.aligner.train_on_corpus(
            hebrew_verses=[self.hebrew_verse],
            greek_verses=[self.greek_verse],
            translation_verses=self.translation_verses,
        )

        self.assertIn("original_verses_processed", training_stats)
        self.assertIn("translation_verses_processed", training_stats)
        self.assertIn("strongs_mappings_built", training_stats)

        # Check that Strong's mappings were built
        self.assertGreater(len(self.aligner.strongs_mappings), 0)

    def test_complete_verse_alignment(self):
        """Test complete verse alignment with all features."""
        # First train the system
        self.aligner.train_on_corpus(
            hebrew_verses=[self.hebrew_verse],
            greek_verses=[self.greek_verse],
            translation_verses=self.translation_verses,
        )

        # Test Hebrew verse alignment
        hebrew_trans = self.translation_verses[0]
        alignments = self.aligner.align_verse_complete(self.hebrew_verse, hebrew_trans)

        self.assertGreater(len(alignments), 0)

        for alignment in alignments:
            self.assertIsInstance(alignment, CompleteAlignment)
            self.assertIsInstance(alignment.confidence, AlignmentConfidence)
            self.assertGreaterEqual(alignment.confidence.overall_score, 0.0)
            self.assertLessEqual(alignment.confidence.overall_score, 1.0)

            # Check that alignment methods are recorded
            self.assertGreater(len(alignment.alignment_methods), 0)

            # Check that verse ID is set
            self.assertEqual(alignment.verse_id, "GEN.1.1")

    def test_greek_verse_alignment(self):
        """Test Greek verse alignment."""
        # Train first
        self.aligner.train_on_corpus(
            hebrew_verses=[self.hebrew_verse],
            greek_verses=[self.greek_verse],
            translation_verses=self.translation_verses,
        )

        # Test Greek verse
        greek_trans = self.translation_verses[1]
        alignments = self.aligner.align_verse_complete(self.greek_verse, greek_trans)

        self.assertGreater(len(alignments), 0)

        for alignment in alignments:
            self.assertIsInstance(alignment, CompleteAlignment)
            self.assertEqual(alignment.verse_id, "JHN.1.1")


class TestAdvancedConfidenceEnsemble(unittest.TestCase):
    """Test the advanced confidence ensemble system."""

    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = AdvancedConfidenceEnsemble()
        self.sample_alignment = self._create_sample_alignment()

    def _create_sample_alignment(self) -> CompleteAlignment:
        """Create a sample alignment for testing."""
        # Create mock morphology
        features = MorphologyFeatures(part_of_speech="noun")
        morphology = UnifiedMorphology(
            language=Language.HEBREW, features=features, original_code="HNcmpa"
        )

        token = ExtractedToken(
            text="אֱלֹהִים", strong_number="H430", lemma="אֱלֹהִים", morphology=morphology
        )

        return CompleteAlignment(
            source_tokens=[token],
            target_words=["God"],
            strong_numbers=["H430"],
            confidence=AlignmentConfidence(overall_score=0.0),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

    def test_confidence_computation(self):
        """Test confidence computation with all methods."""
        context = {"original_verse": Mock(), "translation_verse": Mock()}

        confidence = self.ensemble.compute_confidence(self.sample_alignment, context)

        self.assertIsInstance(confidence, AlignmentConfidence)
        self.assertGreaterEqual(confidence.overall_score, 0.0)
        self.assertLessEqual(confidence.overall_score, 1.0)

        # Check that method scores are computed
        self.assertIn(AlignmentMethod.STRONGS.value, confidence.method_scores)
        self.assertIn(AlignmentMethod.NEURAL.value, confidence.method_scores)
        self.assertIn(AlignmentMethod.SYNTACTIC.value, confidence.method_scores)

    def test_strongs_confidence(self):
        """Test Strong's-based confidence scoring."""
        confidence = self.ensemble._compute_strongs_confidence(self.sample_alignment)

        # Should be high confidence because Strong's number is present
        self.assertGreaterEqual(confidence, 0.7)

        # Test without Strong's numbers
        no_strongs_alignment = CompleteAlignment(
            source_tokens=[ExtractedToken(text="test")],
            target_words=["test"],
            strong_numbers=[],
            confidence=AlignmentConfidence(overall_score=0.0),
            alignment_methods=[],
        )

        no_strongs_confidence = self.ensemble._compute_strongs_confidence(no_strongs_alignment)
        self.assertEqual(no_strongs_confidence, 0.0)

    def test_uncertainty_factor_identification(self):
        """Test identification of uncertainty factors."""
        # Create alignment without Strong's numbers
        uncertain_alignment = CompleteAlignment(
            source_tokens=[ExtractedToken(text="test")],
            target_words=["word1", "word2", "word3", "word4"],  # Length mismatch (difference of 3)
            strong_numbers=[],
            confidence=AlignmentConfidence(overall_score=0.0),
            alignment_methods=[],
        )

        method_scores = {
            AlignmentMethod.STRONGS.value: 0.0,
            AlignmentMethod.SEMANTIC.value: 0.3,  # Low semantic similarity
        }

        factors = self.ensemble._identify_uncertainty_factors(method_scores, uncertain_alignment)

        self.assertIn("no_strongs_numbers", factors)
        self.assertIn("low_semantic_similarity", factors)
        self.assertIn("significant_length_mismatch", factors)


class TestCrossTranslationValidator(unittest.TestCase):
    """Test cross-translation validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = CrossTranslationValidator()

        # Create sample alignments for multiple translations
        self.sample_alignments = self._create_sample_alignments()

    def _create_sample_alignments(self) -> dict:
        """Create sample alignments for testing."""
        token = ExtractedToken(text="אֱלֹהִים", strong_number="H430", lemma="אֱלֹהִים")

        # KJV alignment
        kjv_alignment = CompleteAlignment(
            source_tokens=[token],
            target_words=["God"],
            strong_numbers=["H430"],
            confidence=AlignmentConfidence(overall_score=0.9),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

        # NIV alignment (same word)
        niv_alignment = CompleteAlignment(
            source_tokens=[token],
            target_words=["God"],
            strong_numbers=["H430"],
            confidence=AlignmentConfidence(overall_score=0.9),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

        # ESV alignment (different word)
        esv_alignment = CompleteAlignment(
            source_tokens=[token],
            target_words=["gods"],  # Different translation
            strong_numbers=["H430"],
            confidence=AlignmentConfidence(overall_score=0.8),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

        return {"KJV": [kjv_alignment], "NIV": [niv_alignment], "ESV": [esv_alignment]}

    def test_add_translation_alignments(self):
        """Test adding translation alignments."""
        verse_id = "GEN.1.1"

        for translation, alignments in self.sample_alignments.items():
            self.validator.add_translation_alignments(translation, verse_id, alignments)

        # Check that alignments were added
        self.assertEqual(len(self.validator.translation_alignments), 3)
        self.assertIn("KJV", self.validator.translation_alignments)
        self.assertIn("NIV", self.validator.translation_alignments)
        self.assertIn("ESV", self.validator.translation_alignments)

    def test_consistency_score_computation(self):
        """Test consistency score computation."""
        verse_id = "GEN.1.1"
        strong_number = "H430"

        # Add alignments
        for translation, alignments in self.sample_alignments.items():
            self.validator.add_translation_alignments(translation, verse_id, alignments)

        # Compute consistency score
        consistency = self.validator.compute_consistency_score(verse_id, strong_number)

        # Should be 2/3 = 0.67 (2 "God", 1 "gods")
        self.assertAlmostEqual(consistency, 2 / 3, places=2)

    def test_verse_validation(self):
        """Test validation of verse alignments."""
        verse_id = "GEN.1.1"

        # Add alignments
        for translation, alignments in self.sample_alignments.items():
            self.validator.add_translation_alignments(translation, verse_id, alignments)

        # Validate alignments
        validation_results = self.validator.validate_verse_alignments(
            verse_id, self.sample_alignments["KJV"]
        )

        self.assertEqual(validation_results["verse_id"], verse_id)
        self.assertEqual(validation_results["total_alignments"], 1)
        self.assertIn("H430", validation_results["consistency_scores"])
        self.assertGreaterEqual(validation_results["overall_consistency"], 0.0)

    def test_consistency_report_generation(self):
        """Test generation of consistency report."""
        verse_id = "GEN.1.1"

        # Add alignments
        for translation, alignments in self.sample_alignments.items():
            self.validator.add_translation_alignments(translation, verse_id, alignments)

        # Generate report
        report = self.validator.generate_consistency_report()

        self.assertIn("total_translations", report)
        self.assertIn("total_verses", report)
        self.assertIn("strong_number_analysis", report)
        self.assertIn("recommendations", report)

        self.assertEqual(report["total_translations"], 3)
        self.assertEqual(report["total_verses"], 1)


class TestSemanticLossDetector(unittest.TestCase):
    """Test semantic loss detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = SemanticLossDetector()

    def test_known_semantic_loss_detection(self):
        """Test detection of known semantic losses."""
        # Create alignment with known semantic loss (chesed)
        token = ExtractedToken(text="חֶסֶד", strong_number="H2617", lemma="חֶסֶד")

        alignment = CompleteAlignment(
            source_tokens=[token],
            target_words=["mercy"],
            strong_numbers=["H2617"],
            confidence=AlignmentConfidence(overall_score=0.8),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

        losses = self.detector.detect_semantic_losses(alignment)

        self.assertGreater(len(losses), 0)
        self.assertEqual(losses[0].loss_type, SemanticLossType.LEXICAL_RICHNESS)
        self.assertIn("chesed", losses[0].original_concept)

    def test_elohim_semantic_loss(self):
        """Test detection of Elohim plural majesty loss."""
        token = ExtractedToken(text="אֱלֹהִים", strong_number="H430", lemma="אֱלֹהִים")

        alignment = CompleteAlignment(
            source_tokens=[token],
            target_words=["God"],
            strong_numbers=["H430"],
            confidence=AlignmentConfidence(overall_score=0.9),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

        losses = self.detector.detect_semantic_losses(alignment)

        self.assertGreater(len(losses), 0)
        self.assertEqual(losses[0].loss_type, SemanticLossType.GRAMMATICAL_NUANCE)
        self.assertIn("plural", losses[0].description.lower())

    def test_no_semantic_loss(self):
        """Test case with no semantic losses."""
        token = ExtractedToken(
            text="בית", strong_number="H1004", lemma="בית"  # house - straightforward translation
        )

        alignment = CompleteAlignment(
            source_tokens=[token],
            target_words=["house"],
            strong_numbers=["H1004"],
            confidence=AlignmentConfidence(overall_score=0.9),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

        losses = self.detector.detect_semantic_losses(alignment)

        # Should be empty or very minimal
        self.assertLessEqual(len(losses), 1)


class TestActiveLearningManager(unittest.TestCase):
    """Test active learning functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ActiveLearningManager()
        self.sample_alignments = self._create_sample_alignments()

    def _create_sample_alignments(self) -> list[CompleteAlignment]:
        """Create sample alignments with varying confidence."""
        alignments = []

        # High confidence alignment
        high_conf = CompleteAlignment(
            source_tokens=[ExtractedToken(text="test1", strong_number="H1")],
            target_words=["test1"],
            strong_numbers=["H1"],
            confidence=AlignmentConfidence(overall_score=0.95, uncertainty_factors=[]),
            alignment_methods=[AlignmentMethod.STRONGS],
        )

        # Low confidence alignment
        low_conf = CompleteAlignment(
            source_tokens=[ExtractedToken(text="test2")],
            target_words=["test2"],
            strong_numbers=[],
            confidence=AlignmentConfidence(
                overall_score=0.3,
                uncertainty_factors=["no_strongs_numbers", "low_semantic_similarity"],
            ),
            alignment_methods=[AlignmentMethod.STATISTICAL],
        )

        # Medium confidence with conflicts
        conflicted = CompleteAlignment(
            source_tokens=[ExtractedToken(text="test3", strong_number="H3")],
            target_words=["test3"],
            strong_numbers=["H3"],
            confidence=AlignmentConfidence(
                overall_score=0.55,
                method_scores={
                    "strongs": 0.9,
                    "semantic": 0.2,  # Conflicting scores
                    "neural": 0.7,
                },
                uncertainty_factors=["conflicting_method_scores"],
            ),
            alignment_methods=[AlignmentMethod.STRONGS, AlignmentMethod.SEMANTIC],
        )

        alignments.extend([high_conf, low_conf, conflicted])
        return alignments

    def test_uncertain_alignment_identification(self):
        """Test identification of uncertain alignments."""
        uncertain = self.manager.identify_uncertain_alignments(
            self.sample_alignments, uncertainty_threshold=0.6
        )

        # Should identify the low confidence and conflicted alignments
        self.assertGreaterEqual(len(uncertain), 2)

        # Check that they're sorted by impact score
        self.assertTrue(hasattr(uncertain[0], "impact_score"))

    def test_review_batch_generation(self):
        """Test generation of review batch for human experts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "review_batch.json"

            # First identify uncertain alignments
            self.manager.identify_uncertain_alignments(self.sample_alignments)

            # Generate review batch
            batch = self.manager.generate_review_batch(output_path)

            self.assertIn("batch_id", batch)
            self.assertIn("cases", batch)
            self.assertGreater(len(batch["cases"]), 0)

            # Check that file was created
            self.assertTrue(output_path.exists())

            # Verify file contents
            with open(output_path, "r") as f:
                saved_batch = json.load(f)

            self.assertEqual(saved_batch["batch_id"], batch["batch_id"])

    def test_expert_feedback_processing(self):
        """Test processing of expert feedback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.json"

            # Create mock feedback data
            feedback_data = {
                "batch_id": "test_batch_1",
                "cases": [
                    {
                        "case_id": 1,
                        "current_confidence": 0.3,
                        "expert_feedback": {
                            "is_correct": True,
                            "confidence_rating": 0.8,
                            "notes": "Good alignment",
                        },
                    },
                    {
                        "case_id": 2,
                        "current_confidence": 0.6,
                        "expert_feedback": {
                            "is_correct": False,
                            "confidence_rating": 0.2,
                            "notes": "Incorrect alignment",
                        },
                    },
                ],
            }

            # Save feedback data
            with open(feedback_path, "w") as f:
                json.dump(feedback_data, f)

            # Process feedback
            summary = self.manager.process_expert_feedback(feedback_path)

            self.assertIn("batch_id", summary)
            self.assertIn("total_reviewed", summary)
            self.assertIn("accuracy", summary)
            self.assertIn("average_confidence_improvement", summary)

            self.assertEqual(summary["total_reviewed"], 2)
            self.assertEqual(summary["accuracy"], 0.5)  # 1 correct out of 2


class TestComprehensiveReport(unittest.TestCase):
    """Test comprehensive report generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.aligner = CompleteModernAligner()

        # Create sample alignments dictionary
        self.sample_alignments = self._create_sample_alignments_dict()

    def _create_sample_alignments_dict(self) -> dict[str, list[CompleteAlignment]]:
        """Create sample alignments for multiple verses."""
        alignments_dict = {}

        for verse_id in ["GEN.1.1", "GEN.1.2", "JHN.1.1"]:
            alignments = []

            for i in range(3):  # 3 alignments per verse
                alignment = CompleteAlignment(
                    source_tokens=[
                        ExtractedToken(text=f"word{i}", strong_number=f"H{i+1}", lemma=f"lemma{i}")
                    ],
                    target_words=[f"english{i}"],
                    strong_numbers=[f"H{i+1}"],
                    confidence=AlignmentConfidence(overall_score=0.5 + i * 0.2),
                    alignment_methods=[AlignmentMethod.STRONGS],
                    semantic_losses=[],
                    verse_id=verse_id,
                )
                alignments.append(alignment)

            alignments_dict[verse_id] = alignments

        return alignments_dict

    def test_comprehensive_report_generation(self):
        """Test generation of comprehensive quality report."""
        report = self.aligner.generate_comprehensive_report(self.sample_alignments)

        # Check main sections
        self.assertIn("summary", report)
        self.assertIn("confidence_distribution", report)
        self.assertIn("method_usage", report)
        self.assertIn("semantic_loss_analysis", report)
        self.assertIn("cross_translation_consistency", report)
        self.assertIn("active_learning", report)
        self.assertIn("recommendations", report)

        # Check summary statistics
        summary = report["summary"]
        self.assertEqual(summary["total_verses"], 3)
        self.assertEqual(summary["total_alignments"], 9)  # 3 verses * 3 alignments
        self.assertGreaterEqual(summary["average_confidence"], 0.0)
        self.assertLessEqual(summary["average_confidence"], 1.0)

        # Check confidence distribution
        confidence_dist = report["confidence_distribution"]
        self.assertIn("high_confidence", confidence_dist)
        self.assertIn("medium_confidence", confidence_dist)
        self.assertIn("low_confidence", confidence_dist)

        # Verify percentages add up to 100
        total_percentage = (
            confidence_dist["high_confidence"]["percentage"]
            + confidence_dist["medium_confidence"]["percentage"]
            + confidence_dist["low_confidence"]["percentage"]
        )
        self.assertAlmostEqual(total_percentage, 100.0, places=1)


def run_all_tests():
    """Run all tests and return results."""
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestCompleteModernAligner,
        TestAdvancedConfidenceEnsemble,
        TestCrossTranslationValidator,
        TestSemanticLossDetector,
        TestActiveLearningManager,
        TestComprehensiveReport,
    ]

    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
    }


if __name__ == "__main__":
    # Run tests directly
    results = run_all_tests()
    print(f"\nTest Results: {results}")
