"""
Basic tests for cross-reference system.
"""

import pytest

from abba.cross_references.models import (
    CrossReference,
    ReferenceType,
    ReferenceRelationship,
    ReferenceConfidence,
    CitationMatch,
)
from abba.cross_references.parser import CrossReferenceParser
from abba.cross_references.classifier import ReferenceTypeClassifier as ReferenceClassifier
from abba.cross_references.confidence_scorer import ConfidenceScorer
from abba.cross_references.citation_tracker import CitationTracker
from abba.verse_id import VerseID, parse_verse_id


class TestCrossReferenceModels:
    """Test cross-reference data models."""
    
    def test_cross_reference_creation(self):
        """Test creating a cross-reference."""
        source = VerseID("MAT", 5, 17)
        target = VerseID("DEU", 5, 16)
        
        confidence = ReferenceConfidence(
            overall_score=0.95,
            textual_similarity=0.9,
            thematic_similarity=0.8,
            structural_similarity=0.7,
            scholarly_consensus=0.85,
            uncertainty_factors=[]
        )
        
        ref = CrossReference(
            source_verse=source,
            target_verse=target,
            reference_type=ReferenceType.DIRECT_QUOTE,
            relationship=ReferenceRelationship.QUOTES,
            confidence=confidence
        )
        
        assert ref.source_verse == source
        assert ref.target_verse == target
        assert ref.reference_type == ReferenceType.DIRECT_QUOTE
        assert ref.relationship == ReferenceRelationship.QUOTES
        assert ref.confidence.overall_score == 0.95
    
    def test_reverse_reference(self):
        """Test creating reverse reference."""
        source = VerseID("MAT", 5, 17)
        target = VerseID("DEU", 5, 16)
        
        confidence = ReferenceConfidence(
            overall_score=0.95,
            textual_similarity=0.9,
            thematic_similarity=0.8,
            structural_similarity=0.7,
            scholarly_consensus=0.85,
            uncertainty_factors=[]
        )
        
        ref = CrossReference(
            source_verse=source,
            target_verse=target,
            reference_type=ReferenceType.DIRECT_QUOTE,
            relationship=ReferenceRelationship.QUOTES,
            confidence=confidence
        )
        
        reverse = ref.get_reverse_reference()
        
        assert reverse.source_verse == target
        assert reverse.target_verse == source
        assert reverse.relationship == ReferenceRelationship.QUOTED_BY
        assert reverse.reference_type == ReferenceType.DIRECT_QUOTE
    
    def test_citation_match(self):
        """Test citation match model."""
        match = CitationMatch(
            source_verse=VerseID("MAT", 5, 17),
            target_verse=VerseID("DEU", 5, 16),
            source_text="Honor your father and mother",
            target_text="Honor your father and your mother",
            match_type=ReferenceType.DIRECT_QUOTE,
            text_similarity=0.95,
            word_matches=["honor", "father", "mother"]
        )
        
        assert match.source_verse.book == "MAT"
        assert match.target_verse.book == "DEU"
        assert match.text_similarity == 0.95
        assert len(match.word_matches) == 3


class TestCrossReferenceParser:
    """Test cross-reference parser."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = CrossReferenceParser()
        
        assert "Genesis" in parser.known_abbreviations
        assert parser.known_abbreviations["Genesis"] == "GEN"
        assert "Matthew" in parser.known_abbreviations
        assert parser.known_abbreviations["Matthew"] == "MAT"
    
    def test_parse_reference_string(self):
        """Test parsing reference strings."""
        parser = CrossReferenceParser()
        
        # Test simple reference
        refs = parser.parse_reference_string("John 3:16")
        assert len(refs) == 1
        assert refs[0].book == "JHN"
        assert refs[0].chapter == 3
        assert refs[0].verse == 16
        
        # Test multiple references
        refs = parser.parse_reference_string("Matt 5:17; Luke 10:25-28")
        assert len(refs) == 2
        assert refs[0].book == "MAT"
        assert refs[0].chapter == 5
        assert refs[0].verse == 17
        assert refs[1].book == "LUK"
        assert refs[1].chapter == 10
        assert refs[1].verse == 25  # First verse of range
    
    def test_normalize_book_name(self):
        """Test book name normalization."""
        parser = CrossReferenceParser()
        
        assert parser.normalize_book_name("Genesis") == "GEN"
        assert parser.normalize_book_name("Gen") == "GEN"
        assert parser.normalize_book_name("1 John") == "1JN"
        assert parser.normalize_book_name("Revelation") == "REV"


class TestReferenceClassifier:
    """Test reference type classification."""
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = ReferenceClassifier()
        assert classifier is not None
    
    def test_classify_direct_quote(self):
        """Test classifying direct quotes."""
        classifier = ReferenceClassifier()
        
        # Nearly identical text
        ref_type = classifier.classify(
            source_text="For God so loved the world",
            target_text="For God so loved the world",
            source_verse=VerseID("JHN", 3, 16),
            target_verse=VerseID("JHN", 3, 16)
        )
        
        assert ref_type == ReferenceType.DIRECT_QUOTE
    
    def test_classify_paraphrase(self):
        """Test classifying paraphrases."""
        classifier = ReferenceClassifier()
        
        # Similar but reworded
        ref_type = classifier.classify(
            source_text="Honor your father and mother",
            target_text="Give honor to your parents",
            source_verse=VerseID("MAT", 15, 4),
            target_verse=VerseID("EXO", 20, 12)
        )
        
        assert ref_type in [ReferenceType.PARAPHRASE, ReferenceType.PARTIAL_QUOTE, ReferenceType.ALLUSION]
    
    def test_classify_thematic_parallel(self):
        """Test classifying thematic parallels."""
        classifier = ReferenceClassifier()
        
        # Same theme, different wording
        ref_type = classifier.classify(
            source_text="Love your neighbor as yourself",
            target_text="Do unto others as you would have them do unto you",
            source_verse=VerseID("MAT", 22, 39),
            target_verse=VerseID("MAT", 7, 12)
        )
        
        assert ref_type in [ReferenceType.THEMATIC_PARALLEL, ReferenceType.HISTORICAL_PARALLEL]


class TestConfidenceScorer:
    """Test confidence scoring."""
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = ConfidenceScorer()
        assert scorer is not None
    
    def test_score_direct_quote(self):
        """Test scoring direct quotes."""
        scorer = ConfidenceScorer()
        
        confidence = scorer.calculate_confidence(
            source_text="For God so loved the world",
            target_text="For God so loved the world",
            ref_type=ReferenceType.DIRECT_QUOTE,
            relationship=ReferenceRelationship.QUOTES,
            source_verse=VerseID("JHN", 3, 16),
            target_verse=VerseID("JHN", 3, 16)
        )
        
        assert confidence.overall_score >= 0.8
        assert confidence.textual_similarity >= 0.95
    
    def test_score_partial_match(self):
        """Test scoring partial matches."""
        scorer = ConfidenceScorer()
        
        confidence = scorer.calculate_confidence(
            source_text="Honor your father and mother",
            target_text="Honor your father and your mother, that your days may be long",
            ref_type=ReferenceType.PARTIAL_QUOTE,
            relationship=ReferenceRelationship.QUOTES,
            source_verse=VerseID("MAT", 15, 4),
            target_verse=VerseID("EXO", 20, 12)
        )
        
        assert 0.5 <= confidence.overall_score <= 0.9
        assert confidence.textual_similarity > 0.5


class TestCitationTracker:
    """Test citation tracking functionality."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = CitationTracker()
        assert tracker is not None
    
    def test_add_citation(self):
        """Test creating cross-references from citations."""
        tracker = CitationTracker()
        
        # Create a citation match
        citation = CitationMatch(
            source_verse=VerseID("MAT", 5, 17),
            target_verse=VerseID("DEU", 5, 16),
            source_text="Do not think that I have come to abolish the Law",
            target_text="Honor your father and your mother",
            match_type=ReferenceType.ALLUSION,
            text_similarity=0.3,
            word_matches=["law"]
        )
        
        # Convert to cross-references
        cross_refs = tracker.create_citation_cross_references([citation])
        
        # Should create both forward and reverse references
        assert len(cross_refs) == 2
        
        # Check forward reference
        forward_ref = cross_refs[0]
        assert forward_ref.source_verse.book == "MAT"
        assert forward_ref.target_verse.book == "DEU"
        assert forward_ref.relationship == ReferenceRelationship.QUOTES
        
        # Check reverse reference
        reverse_ref = cross_refs[1]
        assert reverse_ref.source_verse.book == "DEU"
        assert reverse_ref.target_verse.book == "MAT"
        assert reverse_ref.relationship == ReferenceRelationship.QUOTED_BY
    
    def test_find_all_quotations(self):
        """Test analyzing quotation patterns."""
        tracker = CitationTracker()
        
        # Create multiple citation matches
        citations = []
        for chapter in [5, 15, 19]:
            citation = CitationMatch(
                source_verse=VerseID("MAT", chapter, 4),
                target_verse=VerseID("EXO", 20, 12),
                source_text="Honor your father and mother",
                target_text="Honor your father and your mother",
                match_type=ReferenceType.DIRECT_QUOTE,
                text_similarity=0.95,
                word_matches=["honor", "father", "mother"]
            )
            citations.append(citation)
        
        # Analyze patterns
        analysis = tracker.analyze_quotation_patterns(citations)
        
        assert analysis["total_citations"] == 3
        assert analysis["average_similarity"] > 0.9
        assert ("MAT", 3) in analysis["most_quoting_nt_books"]
        assert ("EXO", 3) in analysis["most_quoted_ot_books"]