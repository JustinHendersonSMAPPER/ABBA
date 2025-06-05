"""
Tests for the modern alignment system.

Comprehensive test suite covering the modern aligner with ML/statistical
controls and quality assurance features.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from abba.alignment.modern_aligner import (
    ModernAlignment,
    ConfidenceEnsemble,
    CrossTranslationValidator,
    ModernAlignmentPipeline,
    demonstrate_modern_pipeline,
    SPACY_AVAILABLE,
    ML_AVAILABLE
)
from abba.verse_id import VerseID
from abba.parsers.translation_parser import TranslationVerse
from abba.parsers.hebrew_parser import HebrewVerse, HebrewWord
from abba.parsers.greek_parser import GreekVerse, GreekWord
from abba.interlinear.token_extractor import ExtractedToken


class TestModernAlignment:
    """Test the ModernAlignment dataclass."""
    
    def test_modern_alignment_creation(self):
        """Test creating a modern alignment."""
        tokens = [ExtractedToken(text="אהב", strong_number="H157", lemma="אהב")]
        
        alignment = ModernAlignment(
            source_tokens=tokens,
            target_words=["love", "loves"],
            strong_numbers=["H157"],
            confidence_score=0.85,
            confidence_breakdown={"strongs": 0.9, "semantic": 0.8},
            alignment_methods=["strongs", "neural"],
            cross_translation_consistency=0.75,
            semantic_similarity=0.82,
            syntactic_compatibility=0.7,
            semantic_losses=[],
            cultural_context="Ancient Near Eastern covenant context",
            morphological_notes=["Qal perfect 3ms"],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        assert alignment.confidence_score == 0.85
        assert len(alignment.target_words) == 2
        assert alignment.strong_numbers == ["H157"]
        assert "strongs" in alignment.alignment_methods
        
    def test_modern_alignment_to_dict(self):
        """Test converting alignment to dictionary."""
        tokens = [ExtractedToken(text="אהב", strong_number="H157", lemma="אהב")]
        
        alignment = ModernAlignment(
            source_tokens=tokens,
            target_words=["love"],
            strong_numbers=["H157"],
            confidence_score=0.9,
            confidence_breakdown={"strongs": 0.9},
            alignment_methods=["strongs"],
            cross_translation_consistency=0.8,
            semantic_similarity=0.85,
            syntactic_compatibility=0.75,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        result = alignment.to_dict()
        
        assert result["source_text"] == "אהב"
        assert result["target_text"] == "love"
        assert result["confidence_score"] == 0.9
        assert result["quality_metrics"]["semantic_similarity"] == 0.85


class TestConfidenceEnsemble:
    """Test the confidence ensemble scoring system."""
    
    @pytest.fixture
    def ensemble(self):
        """Create a confidence ensemble."""
        return ConfidenceEnsemble()
    
    def test_ensemble_initialization(self, ensemble):
        """Test ensemble initialization."""
        assert ensemble.weights["strongs"] == 0.4
        assert ensemble.weights["semantic"] == 0.25
        assert ensemble.weights["neural"] == 0.25
        assert ensemble.weights["syntactic"] == 0.1
        assert sum(ensemble.weights.values()) == 1.0
        
    def test_compute_strongs_confidence(self, ensemble):
        """Test Strong's number confidence calculation."""
        alignment = ModernAlignment(
            source_tokens=[],
            target_words=[],
            strong_numbers=["H157"],
            confidence_score=0.0,
            confidence_breakdown={},
            alignment_methods=[],
            cross_translation_consistency=0.0,
            semantic_similarity=0.0,
            syntactic_compatibility=0.0,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        confidence = ensemble._compute_strongs_confidence(alignment)
        assert confidence == 0.9
        
        # Test without Strong's numbers
        alignment.strong_numbers = []
        confidence = ensemble._compute_strongs_confidence(alignment)
        assert confidence == 0.0
        
    @patch('abba.alignment.modern_aligner.ML_AVAILABLE', False)
    def test_compute_semantic_confidence_no_ml(self, ensemble):
        """Test semantic confidence when ML is not available."""
        alignment = Mock()
        context = {}
        
        confidence = ensemble._compute_semantic_confidence(alignment, context)
        assert confidence is None
        
    def test_compute_neural_confidence(self, ensemble):
        """Test neural confidence calculation."""
        tokens = [ExtractedToken(text="word", strong_number="H1", lemma="word")]
        
        alignment = ModernAlignment(
            source_tokens=tokens,
            target_words=["word"],
            strong_numbers=[],
            confidence_score=0.0,
            confidence_breakdown={},
            alignment_methods=[],
            cross_translation_consistency=0.0,
            semantic_similarity=0.0,
            syntactic_compatibility=0.0,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        context = {}
        confidence = ensemble._compute_neural_confidence(alignment, context)
        
        # 1:1 alignment should have high confidence
        assert confidence == 0.8
        
    def test_compute_syntactic_confidence(self, ensemble):
        """Test syntactic confidence calculation."""
        # Create tokens with morphology
        morphology = Mock()
        morphology.part_of_speech = "noun"
        
        token = ExtractedToken(text="word", strong_number="H1", lemma="word")
        token.morphology = morphology
        
        alignment = ModernAlignment(
            source_tokens=[token],
            target_words=["word"],
            strong_numbers=[],
            confidence_score=0.0,
            confidence_breakdown={},
            alignment_methods=[],
            cross_translation_consistency=0.0,
            semantic_similarity=0.0,
            syntactic_compatibility=0.0,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        confidence = ensemble._compute_syntactic_confidence(alignment)
        assert confidence == 0.8  # All same POS
        
    def test_compute_confidence_ensemble(self, ensemble):
        """Test overall confidence computation."""
        alignment = ModernAlignment(
            source_tokens=[],
            target_words=[],
            strong_numbers=["H157"],
            confidence_score=0.0,
            confidence_breakdown={},
            alignment_methods=[],
            cross_translation_consistency=0.0,
            semantic_similarity=0.0,
            syntactic_compatibility=0.0,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        context = {}
        
        # Mock the semantic model to None so semantic confidence returns None
        ensemble.semantic_model = None
        
        overall, breakdown = ensemble.compute_confidence(alignment, context)
        
        assert "strongs" in breakdown
        assert "neural" in breakdown
        assert "syntactic" in breakdown
        assert overall > 0  # Should have some confidence from Strong's


class TestCrossTranslationValidator:
    """Test cross-translation validation."""
    
    @pytest.fixture
    def validator(self):
        """Create a cross-translation validator."""
        return CrossTranslationValidator()
        
    def test_add_translation_alignment(self, validator):
        """Test adding translation alignments."""
        alignments = [
            ModernAlignment(
                source_tokens=[],
                target_words=["love"],
                strong_numbers=["H157"],
                confidence_score=0.9,
                confidence_breakdown={},
                alignment_methods=[],
                cross_translation_consistency=0.0,
                semantic_similarity=0.0,
                syntactic_compatibility=0.0,
                semantic_losses=[],
                cultural_context=None,
                morphological_notes=[],
                alternative_alignments=[],
                uncertainty_flags=[]
            )
        ]
        
        validator.add_translation_alignment("ESV", "GEN.1.1", alignments)
        
        assert "ESV" in validator.translation_alignments
        assert "GEN.1.1" in validator.translation_alignments["ESV"]
        
    def test_compute_consistency_score(self, validator):
        """Test consistency score computation."""
        # Add alignments from multiple translations
        alignment1 = ModernAlignment(
            source_tokens=[],
            target_words=["love"],
            strong_numbers=["H157"],
            confidence_score=0.9,
            confidence_breakdown={},
            alignment_methods=[],
            cross_translation_consistency=0.0,
            semantic_similarity=0.0,
            syntactic_compatibility=0.0,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        alignment2 = ModernAlignment(
            source_tokens=[],
            target_words=["love"],
            strong_numbers=["H157"],
            confidence_score=0.85,
            confidence_breakdown={},
            alignment_methods=[],
            cross_translation_consistency=0.0,
            semantic_similarity=0.0,
            syntactic_compatibility=0.0,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        alignment3 = ModernAlignment(
            source_tokens=[],
            target_words=["affection"],
            strong_numbers=["H157"],
            confidence_score=0.7,
            confidence_breakdown={},
            alignment_methods=[],
            cross_translation_consistency=0.0,
            semantic_similarity=0.0,
            syntactic_compatibility=0.0,
            semantic_losses=[],
            cultural_context=None,
            morphological_notes=[],
            alternative_alignments=[],
            uncertainty_flags=[]
        )
        
        validator.add_translation_alignment("ESV", "GEN.1.1", [alignment1])
        validator.add_translation_alignment("NIV", "GEN.1.1", [alignment2])
        validator.add_translation_alignment("KJV", "GEN.1.1", [alignment3])
        
        # 2 out of 3 translations use "love"
        consistency = validator.compute_consistency_score("GEN.1.1", "H157")
        assert consistency == 2/3
        
    def test_validate_alignment_set(self, validator):
        """Test validating a set of alignments."""
        alignments = [
            ModernAlignment(
                source_tokens=[],
                target_words=["love"],
                strong_numbers=["H157"],
                confidence_score=0.9,
                confidence_breakdown={},
                alignment_methods=[],
                cross_translation_consistency=0.0,
                semantic_similarity=0.0,
                syntactic_compatibility=0.0,
                semantic_losses=[],
                cultural_context=None,
                morphological_notes=[],
                alternative_alignments=[],
                uncertainty_flags=[]
            ),
            ModernAlignment(
                source_tokens=[],
                target_words=["God"],
                strong_numbers=["H430"],
                confidence_score=0.95,
                confidence_breakdown={},
                alignment_methods=[],
                cross_translation_consistency=0.0,
                semantic_similarity=0.0,
                syntactic_compatibility=0.0,
                semantic_losses=[],
                cultural_context=None,
                morphological_notes=[],
                alternative_alignments=[],
                uncertainty_flags=[]
            )
        ]
        
        scores = validator.validate_alignment_set("GEN.1.1", alignments)
        
        assert "H157" in scores
        assert "H430" in scores


class TestModernAlignmentPipeline:
    """Test the complete modern alignment pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create an alignment pipeline."""
        return ModernAlignmentPipeline()
        
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.confidence_ensemble is not None
        assert pipeline.cross_validator is not None
        assert isinstance(pipeline.strongs_mappings, dict)
        assert isinstance(pipeline.phrase_patterns, dict)
        
    def test_tokenize_english(self, pipeline):
        """Test English tokenization."""
        text = "In the beginning, God created the heavens and the earth."
        tokens = pipeline._tokenize_english(text)
        
        assert "in" in tokens
        assert "beginning" in tokens
        assert "god" in tokens
        assert "," not in tokens  # Punctuation removed
        assert "." not in tokens
        
    def test_update_strongs_mappings(self, pipeline):
        """Test updating Strong's mappings."""
        # Create mock verses
        hebrew_word = HebrewWord(
            text="אלהים",
            lemma="אלהים",
            strong_number="H430",
            morph="Ncmpa"
        )
        
        hebrew_verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[hebrew_word],
            osis_id="Gen.1.1"
        )
        
        trans_verse = TranslationVerse(
            verse_id=VerseID("GEN", 1, 1),
            text="In the beginning God created",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        
        pipeline._update_strongs_mappings(hebrew_verse, trans_verse)
        
        assert "H430" in pipeline.strongs_mappings
        assert len(pipeline.strongs_mappings["H430"]) > 0
        
    def test_normalize_strongs_mappings(self, pipeline):
        """Test normalizing Strong's mappings."""
        pipeline.strongs_mappings = {
            "H430": defaultdict(float, {"god": 10.0, "gods": 5.0})
        }
        
        pipeline._normalize_strongs_mappings()
        
        assert pipeline.strongs_mappings["H430"]["god"] == 10.0/15.0
        assert pipeline.strongs_mappings["H430"]["gods"] == 5.0/15.0
        
    def test_build_training_data(self, pipeline):
        """Test building training data."""
        # Create sample verses
        hebrew_word = HebrewWord(
            text="אלהים",
            lemma="אלהים",
            strong_number="H430",
            morph="Ncmpa"
        )
        
        hebrew_verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[hebrew_word],
            osis_id="Gen.1.1"
        )
        
        trans_verse = TranslationVerse(
            verse_id=VerseID("GEN", 1, 1),
            text="In the beginning God created",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        
        pipeline.build_training_data([hebrew_verse], [], [trans_verse])
        
        assert len(pipeline.strongs_mappings) > 0
        
    def test_align_by_strongs(self, pipeline):
        """Test Strong's-based alignment."""
        # Set up mappings
        pipeline.strongs_mappings = {
            "H430": {"god": 0.9, "gods": 0.1}
        }
        
        hebrew_word = HebrewWord(
            text="אלהים",
            lemma="אלהים",
            strong_number="H430",
            morph="Ncmpa"
        )
        
        hebrew_verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[hebrew_word],
            osis_id="Gen.1.1"
        )
        
        trans_words = ["in", "the", "beginning", "god", "created"]
        
        alignments = pipeline._align_by_strongs(hebrew_verse, trans_words)
        
        assert len(alignments) == 1
        assert alignments[0].strong_numbers == ["H430"]
        assert "god" in alignments[0].target_words
        
    def test_align_verse(self, pipeline):
        """Test aligning a complete verse."""
        # Set up mappings
        pipeline.strongs_mappings = {
            "H430": {"god": 0.9}
        }
        
        hebrew_word = HebrewWord(
            text="אלהים",
            lemma="אלהים",
            strong_number="H430",
            morph="Ncmpa"
        )
        
        hebrew_verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[hebrew_word],
            osis_id="Gen.1.1"
        )
        
        trans_verse = TranslationVerse(
            verse_id=VerseID("GEN", 1, 1),
            text="God",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        
        # Mock the confidence ensemble to avoid ML dependencies
        with patch.object(pipeline.confidence_ensemble, 'compute_confidence') as mock_conf:
            mock_conf.return_value = (0.85, {"strongs": 0.9, "neural": 0.8})
            
            alignments = pipeline.align_verse(hebrew_verse, trans_verse)
            
            assert len(alignments) > 0
            assert alignments[0].confidence_score == 0.85
            
    def test_generate_quality_report(self, pipeline):
        """Test generating quality report."""
        alignments = {
            "GEN.1.1": [
                ModernAlignment(
                    source_tokens=[],
                    target_words=["god"],
                    strong_numbers=["H430"],
                    confidence_score=0.9,
                    confidence_breakdown={},
                    alignment_methods=["strongs"],
                    cross_translation_consistency=0.8,
                    semantic_similarity=0.85,
                    syntactic_compatibility=0.75,
                    semantic_losses=[],
                    cultural_context=None,
                    morphological_notes=[],
                    alternative_alignments=[],
                    uncertainty_flags=[]
                ),
                ModernAlignment(
                    source_tokens=[],
                    target_words=["created"],
                    strong_numbers=["H1254"],
                    confidence_score=0.6,
                    confidence_breakdown={},
                    alignment_methods=["strongs", "neural"],
                    cross_translation_consistency=0.7,
                    semantic_similarity=0.65,
                    syntactic_compatibility=0.6,
                    semantic_losses=[],
                    cultural_context=None,
                    morphological_notes=[],
                    alternative_alignments=[],
                    uncertainty_flags=[]
                )
            ]
        }
        
        report = pipeline.generate_quality_report(alignments)
        
        assert report["total_alignments"] == 2
        assert report["average_confidence"] == 0.75
        assert report["confidence_distribution"]["high_confidence"] == 1
        assert report["confidence_distribution"]["medium_confidence"] == 1
        assert report["method_usage"]["strongs"] == 2
        assert report["method_usage"]["neural"] == 1
        
    def test_generate_recommendations(self, pipeline):
        """Test generating recommendations."""
        # Low confidence scores
        confidence_scores = [0.3, 0.4, 0.35, 0.5]
        method_usage = {"strongs": 2, "neural": 10}
        
        recommendations = pipeline._generate_recommendations(confidence_scores, method_usage)
        
        assert any("training data" in rec for rec in recommendations)
        assert any("low-confidence" in rec for rec in recommendations)
        assert any("Strong's coverage" in rec for rec in recommendations)


class TestDemonstration:
    """Test the demonstration function."""
    
    def test_demonstrate_modern_pipeline(self):
        """Test the demonstration function."""
        pipeline = demonstrate_modern_pipeline()
        
        assert isinstance(pipeline, ModernAlignmentPipeline)
        assert pipeline.confidence_ensemble is not None
        assert pipeline.cross_validator is not None