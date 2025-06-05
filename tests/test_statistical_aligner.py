"""
Tests for the statistical alignment system.

Comprehensive test suite covering the advanced statistical aligner with
Strong's anchoring, semantic loss detection, and multi-stage alignment.
"""

import pytest
from unittest.mock import Mock, patch
from collections import defaultdict

from abba.alignment.statistical_aligner import (
    AlignmentConfidence,
    SemanticLossType,
    SemanticLoss,
    EnhancedAlignment,
    StatisticalAligner,
    AlignmentPipeline
)
from abba.verse_id import VerseID
from abba.parsers.translation_parser import TranslationVerse
from abba.parsers.hebrew_parser import HebrewVerse, HebrewWord
from abba.parsers.greek_parser import GreekVerse, GreekWord
from abba.interlinear.token_extractor import ExtractedToken


class TestAlignmentConfidence:
    """Test alignment confidence enum."""
    
    def test_confidence_levels(self):
        """Test confidence level values."""
        assert AlignmentConfidence.HIGH.value == "high"
        assert AlignmentConfidence.MEDIUM.value == "medium"
        assert AlignmentConfidence.LOW.value == "low"
        assert AlignmentConfidence.UNCERTAIN.value == "uncertain"


class TestSemanticLossType:
    """Test semantic loss type enum."""
    
    def test_loss_types(self):
        """Test semantic loss type values."""
        assert SemanticLossType.LEXICAL_RICHNESS.value == "lexical_richness"
        assert SemanticLossType.ASPECTUAL_DETAIL.value == "aspectual_detail"
        assert SemanticLossType.CULTURAL_CONTEXT.value == "cultural_context"
        assert SemanticLossType.WORDPLAY.value == "wordplay"
        assert SemanticLossType.GRAMMATICAL_NUANCE.value == "grammatical_nuance"


class TestSemanticLoss:
    """Test semantic loss dataclass."""
    
    def test_semantic_loss_creation(self):
        """Test creating a semantic loss."""
        loss = SemanticLoss(
            loss_type=SemanticLossType.LEXICAL_RICHNESS,
            description="Hebrew 'chesed' has rich meaning",
            original_concept="chesed",
            translation_concept="mercy",
            explanation="English lacks single word for covenant faithfulness",
            severity=0.7
        )
        
        assert loss.loss_type == SemanticLossType.LEXICAL_RICHNESS
        assert loss.severity == 0.7
        assert "chesed" in loss.original_concept


class TestEnhancedAlignment:
    """Test enhanced alignment dataclass."""
    
    def test_enhanced_alignment_creation(self):
        """Test creating an enhanced alignment."""
        tokens = [ExtractedToken(text="אהב", strong_number="H157", lemma="אהב")]
        losses = [
            SemanticLoss(
                loss_type=SemanticLossType.LEXICAL_RICHNESS,
                description="Love concept",
                original_concept="ahav",
                translation_concept="love",
                explanation="Hebrew ahav includes covenant loyalty",
                severity=0.5
            )
        ]
        
        alignment = EnhancedAlignment(
            source_tokens=tokens,
            target_words=["love"],
            strong_numbers=["H157"],
            confidence=AlignmentConfidence.HIGH,
            confidence_score=0.85,
            alignment_method="strongs_exact",
            semantic_losses=losses,
            alternative_translations=["love", "affection", "cherish"],
            morphological_notes=["Qal perfect"],
            phrase_id=None
        )
        
        assert alignment.confidence == AlignmentConfidence.HIGH
        assert alignment.confidence_score == 0.85
        assert len(alignment.semantic_losses) == 1
        assert "love" in alignment.alternative_translations
        
    def test_enhanced_alignment_to_dict(self):
        """Test converting alignment to dictionary."""
        tokens = [ExtractedToken(text="אהב", strong_number="H157", lemma="אהב")]
        
        alignment = EnhancedAlignment(
            source_tokens=tokens,
            target_words=["love"],
            strong_numbers=["H157"],
            confidence=AlignmentConfidence.HIGH,
            confidence_score=0.9,
            alignment_method="strongs_exact",
            semantic_losses=[],
            alternative_translations=["love", "affection"],
            morphological_notes=["Qal perfect"],
            phrase_id="phrase_1"
        )
        
        result = alignment.to_dict()
        
        assert result["source_text"] == "אהב"
        assert result["target_text"] == "love"
        assert result["confidence"] == "high"
        assert result["confidence_score"] == 0.9
        assert result["phrase_id"] == "phrase_1"


class TestStatisticalAligner:
    """Test the statistical aligner."""
    
    @pytest.fixture
    def aligner(self):
        """Create a statistical aligner."""
        return StatisticalAligner()
        
    def test_aligner_initialization(self, aligner):
        """Test aligner initialization."""
        assert isinstance(aligner.strongs_to_english, dict)
        assert isinstance(aligner.translation_probs, dict)
        assert isinstance(aligner.phrase_patterns, dict)
        assert isinstance(aligner.semantic_loss_db, dict)
        assert isinstance(aligner.alignment_stats, defaultdict)
        
    def test_tokenize_english(self, aligner):
        """Test English tokenization."""
        text = "In the beginning, God created!"
        tokens = aligner._tokenize_english(text)
        
        assert tokens == ["in", "the", "beginning", "god", "created"]
        assert "," not in tokens
        assert "!" not in tokens
        
    def test_build_strongs_mapping(self, aligner):
        """Test building Strong's mappings."""
        # Create test data
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
        
        aligner._build_strongs_mapping([hebrew_verse], [trans_verse])
        
        assert "H430" in aligner.strongs_to_english
        assert "god" in aligner.strongs_to_english["H430"]
        assert aligner.strongs_to_english["H430"]["god"] == 1.0  # Normalized
        
    def test_learn_translation_probabilities(self, aligner):
        """Test learning translation probabilities."""
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
        
        aligner._learn_translation_probabilities([hebrew_verse], [trans_verse])
        
        assert ("אלהים", "god") in aligner.translation_probs
        assert aligner.translation_probs[("אלהים", "god")] == 1
        
    def test_build_semantic_loss_database(self, aligner):
        """Test building semantic loss database."""
        aligner._build_semantic_loss_database()
        
        assert "H2617" in aligner.semantic_loss_db  # chesed
        assert "G26" in aligner.semantic_loss_db  # agape
        assert "aorist" in aligner.semantic_loss_db
        
        # Check chesed loss
        chesed_losses = aligner.semantic_loss_db["H2617"]
        assert len(chesed_losses) > 0
        assert chesed_losses[0].loss_type == SemanticLossType.LEXICAL_RICHNESS
        
    def test_align_by_strongs(self, aligner):
        """Test Strong's-based alignment."""
        # Setup mappings
        aligner.strongs_to_english = {
            "H430": {"god": 0.9, "deity": 0.1}
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
            text="In the beginning God created",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        
        alignments = aligner._align_by_strongs(hebrew_verse, trans_verse)
        
        assert len(alignments) == 1
        assert alignments[0].strong_numbers == ["H430"]
        assert "god" in alignments[0].target_words
        assert alignments[0].confidence == AlignmentConfidence.HIGH
        assert alignments[0].alignment_method == "strongs_exact"
        
    def test_align_statistically(self, aligner):
        """Test statistical alignment."""
        # Setup translation probabilities
        aligner.translation_probs = {
            ("ברא", "created"): 0.8,
            ("ברא", "made"): 0.2
        }
        
        # Word without Strong's number
        hebrew_word = HebrewWord(
            text="ברא",
            lemma="ברא",
            strong_number=None,
            morph="V-Qp3ms"
        )
        
        hebrew_verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[hebrew_word],
            osis_id="Gen.1.1"
        )
        
        trans_verse = TranslationVerse(
            verse_id=VerseID("GEN", 1, 1),
            text="created",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        
        alignments = aligner._align_statistically(hebrew_verse, trans_verse, [])
        
        assert len(alignments) == 1
        assert "created" in alignments[0].target_words
        assert alignments[0].confidence == AlignmentConfidence.MEDIUM
        assert alignments[0].alignment_method == "statistical"
        
    def test_annotate_semantic_losses(self, aligner):
        """Test semantic loss annotation."""
        # Setup semantic loss database
        aligner.semantic_loss_db = {
            "H2617": [
                SemanticLoss(
                    loss_type=SemanticLossType.LEXICAL_RICHNESS,
                    description="chesed meaning",
                    original_concept="chesed",
                    translation_concept="mercy",
                    explanation="Complex Hebrew concept",
                    severity=0.7
                )
            ]
        }
        
        alignment = EnhancedAlignment(
            source_tokens=[],
            target_words=["mercy"],
            strong_numbers=["H2617"],
            confidence=AlignmentConfidence.HIGH,
            confidence_score=0.9,
            alignment_method="strongs",
            semantic_losses=[],
            alternative_translations=[],
            morphological_notes=[],
            phrase_id=None
        )
        
        aligner._annotate_semantic_losses([alignment])
        
        assert len(alignment.semantic_losses) == 1
        assert alignment.semantic_losses[0].original_concept == "chesed"
        
    def test_align_verse(self, aligner):
        """Test complete verse alignment."""
        # Setup aligner data
        aligner.strongs_to_english = {
            "H430": {"god": 0.9}
        }
        aligner.semantic_loss_db = {}
        
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
        
        alignments = aligner.align_verse(hebrew_verse, trans_verse)
        
        assert len(alignments) > 0
        assert any(a.strong_numbers == ["H430"] for a in alignments)
        
    def test_generate_search_indices(self, aligner):
        """Test search index generation."""
        alignments = {
            "GEN.1.1": [
                EnhancedAlignment(
                    source_tokens=[
                        ExtractedToken(
                            text="אלהים",
                            strong_number="H430",
                            lemma="אלהים",
                            morphology=Mock(spec=[])
                        )
                    ],
                    target_words=["God"],
                    strong_numbers=["H430"],
                    confidence=AlignmentConfidence.HIGH,
                    confidence_score=0.9,
                    alignment_method="strongs",
                    semantic_losses=[
                        SemanticLoss(
                            loss_type=SemanticLossType.CULTURAL_CONTEXT,
                            description="Elohim plural form",
                            original_concept="Elohim",
                            translation_concept="God",
                            explanation="Plural form lost in translation",
                            severity=0.3
                        )
                    ],
                    alternative_translations=[],
                    morphological_notes=[],
                    phrase_id=None
                )
            ]
        }
        
        indices = aligner.generate_search_indices(alignments)
        
        assert "god" in indices["english_to_verses"]
        assert "GEN.1.1" in indices["english_to_verses"]["god"]
        assert "H430" in indices["strongs_to_verses"]
        assert "GEN.1.1" in indices["strongs_to_verses"]["H430"]
        assert "Elohim" in indices["concept_to_verses"]
        
    def test_build_alignment_model(self, aligner):
        """Test building complete alignment model."""
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
        
        aligner.build_alignment_model([hebrew_verse], [trans_verse])
        
        assert len(aligner.strongs_to_english) > 0
        assert len(aligner.semantic_loss_db) > 0
        
    def test_identify_phrase_patterns(self, aligner):
        """Test phrase pattern identification."""
        # Create verses with multi-word phrases
        word1 = HebrewWord(
            text="בית",
            lemma="בית",
            strong_number="H1004",
            morph="Ncmsc"
        )
        word2 = HebrewWord(
            text="יהוה",
            lemma="יהוה",
            strong_number="H3068",
            morph="Np"
        )
        
        hebrew_verse = HebrewVerse(
            verse_id=VerseID("GEN", 28, 17),
            words=[word1, word2],
            osis_id="Gen.28.17"
        )
        
        trans_verse = TranslationVerse(
            verse_id=VerseID("GEN", 28, 17),
            text="house of the LORD",
            original_book_name="Genesis",
            original_chapter=28,
            original_verse=17
        )
        
        aligner._identify_phrase_patterns([hebrew_verse], [trans_verse])
        
        # Should identify bigram pattern
        assert "בית יהוה" in aligner.phrase_patterns
        assert "house of the LORD" in aligner.phrase_patterns["בית יהוה"]


class TestAlignmentPipeline:
    """Test the complete alignment pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create an alignment pipeline."""
        return AlignmentPipeline()
        
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert isinstance(pipeline.aligner, StatisticalAligner)
        assert isinstance(pipeline.search_indices, dict)
        
    def test_process_corpus(self, pipeline):
        """Test processing a corpus."""
        # Create test data
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
        
        greek_word = GreekWord(
            text="θεός",
            lemma="θεός",
            strong_number="G2316",
            morph="N-NSM"
        )
        
        greek_verse = GreekVerse(
            verse_id=VerseID("JHN", 1, 1),
            words=[greek_word],
            tei_id="B04K1V1"
        )
        
        trans_verse1 = TranslationVerse(
            verse_id=VerseID("GEN", 1, 1),
            text="God",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1
        )
        
        trans_verse2 = TranslationVerse(
            verse_id=VerseID("JHN", 1, 1),
            text="God",
            original_book_name="John",
            original_chapter=1,
            original_verse=1
        )
        
        alignments = pipeline.process_corpus(
            [hebrew_verse],
            [greek_verse],
            [trans_verse1, trans_verse2]
        )
        
        assert "GEN.1.1" in alignments
        assert "JHN.1.1" in alignments
        assert len(pipeline.search_indices) > 0
        
    def test_search_cross_language(self, pipeline):
        """Test cross-language search."""
        # Setup search indices
        pipeline.search_indices = {
            "english_to_verses": {"god": {"GEN.1.1", "JHN.1.1"}},
            "strongs_to_verses": {"H430": {"GEN.1.1"}},
            "concept_to_verses": {"Elohim": {"GEN.1.1"}},
            "morphology_to_verses": {"N-NSM": {"JHN.1.1"}}
        }
        
        # Test English search
        results = pipeline.search_cross_language("god", "english")
        assert "GEN.1.1" in results
        assert "JHN.1.1" in results
        
        # Test Strong's search
        results = pipeline.search_cross_language("H430", "strongs")
        assert "GEN.1.1" in results
        
        # Test concept search
        results = pipeline.search_cross_language("Elohim", "concept")
        assert "GEN.1.1" in results
        
        # Test morphology search
        results = pipeline.search_cross_language("N-NSM", "morphology")
        assert "JHN.1.1" in results
        
        # Test invalid search type
        results = pipeline.search_cross_language("test", "invalid")
        assert results == []