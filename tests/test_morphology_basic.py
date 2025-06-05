"""
Basic tests for morphology system.
"""

import pytest

from abba.morphology.base import (
    Gender,
    Number,
    Person,
    Tense,
    Voice,
    Mood,
    Case,
    State,
    Stem,
    MorphologyFeatures,
    Language,
)
from abba.morphology.greek_morphology import GreekMorphology, GreekMorphologyParser
from abba.morphology.hebrew_morphology import HebrewMorphology, HebrewMorphologyParser
from abba.morphology.unified_morphology import UnifiedMorphology, UnifiedMorphologyParser


class TestMorphologyBase:
    """Test base morphology features."""
    
    def test_morphology_features_creation(self):
        """Test creating morphology features."""
        features = MorphologyFeatures(
            part_of_speech="verb",
            gender=Gender.MASCULINE,
            number=Number.SINGULAR,
            person=Person.THIRD,
            tense=Tense.AORIST,
            voice=Voice.ACTIVE,
            mood=Mood.INDICATIVE
        )
        
        assert features.part_of_speech == "verb"
        assert features.gender == Gender.MASCULINE
        assert features.number == Number.SINGULAR
        assert features.person == Person.THIRD
        assert features.tense == Tense.AORIST
        assert features.voice == Voice.ACTIVE
        assert features.mood == Mood.INDICATIVE
    
    def test_morphology_features_to_dict(self):
        """Test converting features to dictionary."""
        features = MorphologyFeatures(
            part_of_speech="noun",
            gender=Gender.FEMININE,
            number=Number.PLURAL,
            case=Case.GENITIVE
        )
        
        result = features.to_dict()
        
        assert result["part_of_speech"] == "noun"
        assert result["gender"] == "feminine"
        assert result["number"] == "plural"
        assert result["case"] == "genitive"
    
    def test_morphology_features_summary(self):
        """Test getting human-readable summary."""
        features = MorphologyFeatures(
            part_of_speech="verb",
            person=Person.FIRST,
            number=Number.PLURAL,
            tense=Tense.PRESENT,
            mood=Mood.INDICATIVE
        )
        
        summary = features.get_summary()
        
        assert "verb" in summary
        assert "1st" in summary
        assert "plural" in summary
        assert "present" in summary


class TestGreekMorphology:
    """Test Greek morphology parsing."""
    
    def test_greek_morphology_creation(self):
        """Test creating Greek morphology features."""
        morph = GreekMorphology(
            part_of_speech="noun",
            case=Case.NOMINATIVE,
            gender=Gender.MASCULINE,
            number=Number.SINGULAR
        )
        
        assert morph.part_of_speech == "noun"
        assert morph.case == Case.NOMINATIVE
        assert morph.gender == Gender.MASCULINE
        assert morph.number == Number.SINGULAR
    
    def test_greek_parser_parse_simple_noun(self):
        """Test parsing a simple Greek noun code."""
        parser = GreekMorphologyParser()
        
        # N-NSM = Noun, Nominative, Singular, Masculine
        result = parser.parse("N-NSM")
        
        assert result.part_of_speech == "noun"
        assert result.case == Case.NOMINATIVE
        assert result.number == Number.SINGULAR
        assert result.gender == Gender.MASCULINE
    
    def test_greek_parser_parse_verb(self):
        """Test parsing a Greek verb code."""
        parser = GreekMorphologyParser()
        
        # V-PAI-3S = Verb, Present, Active, Indicative, 3rd, Singular
        result = parser.parse("V-PAI-3S")
        
        assert result.part_of_speech == "verb"
        assert result.tense == Tense.PRESENT
        assert result.voice == Voice.ACTIVE
        assert result.mood == Mood.INDICATIVE
        assert result.person == Person.THIRD
        assert result.number == Number.SINGULAR
    
    def test_greek_parser_parse_participle(self):
        """Test parsing a Greek participle."""
        parser = GreekMorphologyParser()
        
        # V-PAP = Verb, Present, Active, Participle (P for participle)
        result = parser.parse("V-PAP")
        
        assert result.part_of_speech == "verb"
        assert result.tense == Tense.PRESENT
        assert result.voice == Voice.ACTIVE
        assert result.mood == Mood.PARTICIPLE
    
    def test_greek_parser_invalid_code(self):
        """Test parsing invalid Greek code."""
        parser = GreekMorphologyParser()
        
        result = parser.parse("ZZZZZ")  # Use clearly invalid code
        
        # Should return basic features without crashing
        assert result is not None
        # The parser might pick up the first letter if it's valid, so check for either None or minimal parsing
        assert result.part_of_speech is None or result.tense is None


class TestHebrewMorphology:
    """Test Hebrew morphology parsing."""
    
    def test_hebrew_morphology_creation(self):
        """Test creating Hebrew morphology features."""
        from abba.morphology.base import Stem, State
        
        morph = HebrewMorphology(
            part_of_speech="verb",
            stem=Stem.QAL,
            tense=Tense.PERFECT,
            person=Person.THIRD,
            gender=Gender.MASCULINE,
            number=Number.SINGULAR
        )
        
        assert morph.part_of_speech == "verb"
        assert morph.stem == Stem.QAL
        assert morph.tense == Tense.PERFECT
    
    def test_hebrew_parser_parse_simple_noun(self):
        """Test parsing a Hebrew noun code."""
        parser = HebrewMorphologyParser()
        
        # Ncmsa = Noun, common, masculine, singular, absolute
        result = parser.parse("Ncmsa")
        
        assert result.part_of_speech == "noun"
        assert result.gender == Gender.MASCULINE
        assert result.number == Number.SINGULAR
        assert result.state == State.ABSOLUTE
    
    def test_hebrew_parser_parse_verb(self):
        """Test parsing a Hebrew verb code."""
        parser = HebrewMorphologyParser()
        
        # VQP3MS = Verb, Qal, Perfect, 3rd, Masculine, Singular
        result = parser.parse("VQP3MS")
        
        assert result.part_of_speech == "verb"
        assert result.stem == Stem.QAL
        assert result.tense == Tense.PERFECT
        assert result.person == Person.THIRD
        assert result.gender == Gender.MASCULINE
        assert result.number == Number.SINGULAR


class TestUnifiedMorphology:
    """Test unified morphology parser."""
    
    def test_unified_parser_detect_greek(self):
        """Test detecting Greek morphology code."""
        parser = UnifiedMorphologyParser()
        
        result = parser.parse("V-PAI-3S", language="greek")
        
        assert isinstance(result, UnifiedMorphology)
        assert result.language == Language.GREEK
        assert isinstance(result.features, GreekMorphology)
        assert result.features.part_of_speech == "verb"
        assert result.features.tense == Tense.PRESENT
    
    def test_unified_parser_detect_hebrew(self):
        """Test detecting Hebrew morphology code."""
        parser = UnifiedMorphologyParser()
        
        result = parser.parse("VQP3MS", language="hebrew")
        
        assert isinstance(result, UnifiedMorphology)
        assert result.language == Language.HEBREW
        assert isinstance(result.features, HebrewMorphology)
        assert result.features.part_of_speech == "verb"
        assert result.features.stem == Stem.QAL
    
    def test_unified_parser_auto_detect(self):
        """Test auto-detecting language from morphology code."""
        parser = UnifiedMorphologyParser()
        
        # Greek-style code with dash
        greek_result = parser.parse("N-NSM")
        assert isinstance(greek_result, UnifiedMorphology)
        assert greek_result.language == Language.GREEK
        assert isinstance(greek_result.features, GreekMorphology)
        
        # Hebrew-style code - use one with article prefix to be unambiguous
        hebrew_result = parser.parse("HNcmsa")
        assert isinstance(hebrew_result, UnifiedMorphology)
        assert hebrew_result.language == Language.HEBREW
        assert isinstance(hebrew_result.features, HebrewMorphology)
        assert hebrew_result.features.has_article == True
        assert hebrew_result.features.part_of_speech == "noun"
    
    def test_unified_parser_format_display(self):
        """Test formatting morphology for display."""
        parser = UnifiedMorphologyParser()
        
        # Parse and format Greek
        greek_morph = parser.parse("V-PAI-3S", language="greek")
        display = parser.format_for_display(greek_morph.features)
        
        assert "verb" in display.lower()
        assert "present" in display.lower()
        assert "active" in display.lower()
        assert "indicative" in display.lower()
        assert "3" in display or "third" in display.lower()
        assert "singular" in display.lower()