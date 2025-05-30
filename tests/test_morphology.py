"""Unit tests for morphology parsing system."""

import pytest

from abba.morphology import (
    Gender,
    GreekMorphology,
    GreekMorphologyParser,
    HebrewMorphology,
    HebrewMorphologyParser,
    Language,
    Mood,
    Number,
    Person,
    Stem,
    Tense,
    UnifiedMorphology,
    UnifiedMorphologyParser,
    Voice,
)


class TestHebrewMorphologyParser:
    """Test Hebrew morphology parsing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.parser = HebrewMorphologyParser()

    def test_parse_noun(self) -> None:
        """Test parsing Hebrew noun morphology."""
        # Noun, common, masculine, singular, absolute
        morph = self.parser.parse("Ncmsa")

        assert morph.part_of_speech == "noun"
        assert morph.gender == Gender.MASCULINE
        assert morph.number == Number.SINGULAR
        assert morph.state.value == "absolute"

    def test_parse_verb_qal_perfect(self) -> None:
        """Test parsing Hebrew verb in Qal perfect."""
        # Verb, Qal, perfect, 3rd person, masculine, singular
        morph = self.parser.parse("Vqp3ms")

        assert morph.part_of_speech == "verb"
        assert morph.stem == Stem.QAL
        assert morph.tense == Tense.PERFECT
        assert morph.person == Person.THIRD
        assert morph.gender == Gender.MASCULINE
        assert morph.number == Number.SINGULAR

    def test_parse_verb_with_prefixes(self) -> None:
        """Test parsing Hebrew verb with prefixes."""
        # Conjunction + Article + Verb
        morph = self.parser.parse("CHVqp3ms")

        assert isinstance(morph, HebrewMorphology)
        assert morph.has_conjunction
        assert morph.has_article
        assert morph.part_of_speech == "verb"

    def test_parse_with_suffix(self) -> None:
        """Test parsing Hebrew word with pronominal suffix."""
        # Noun with 3rd person masculine singular suffix
        morph = self.parser.parse("Ncmsa/Sp3ms")

        assert morph.has_suffix
        assert morph.suffix_person == Person.THIRD
        assert morph.suffix_gender == Gender.MASCULINE
        assert morph.suffix_number == Number.SINGULAR

    def test_hebrew_summary(self) -> None:
        """Test Hebrew morphology summary generation."""
        morph = self.parser.parse("HNcmsa")
        summary = morph.get_hebrew_summary()

        assert "art" in summary
        assert "noun" in summary
        assert "masculine" in summary
        assert "singular" in summary

    def test_empty_morphology(self) -> None:
        """Test parsing empty morphology code."""
        morph = self.parser.parse("")

        assert morph.part_of_speech is None
        assert morph.gender is None
        assert morph.number is None

    def test_cache_functionality(self) -> None:
        """Test that morphology parsing is cached."""
        code = "Vqp3ms"

        # First parse
        morph1 = self.parser.parse(code)

        # Second parse should return same object
        morph2 = self.parser.parse(code)

        assert morph1 is morph2


class TestGreekMorphologyParser:
    """Test Greek morphology parsing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.parser = GreekMorphologyParser()

    def test_parse_verb_dash_format(self) -> None:
        """Test parsing Greek verb in dash format."""
        # Verb, Present Active Indicative, 3rd person Singular
        morph = self.parser.parse("V-PAI-3S")

        assert morph.part_of_speech == "verb"
        assert morph.tense == Tense.PRESENT
        assert morph.voice == Voice.ACTIVE
        assert morph.mood == Mood.INDICATIVE
        assert morph.person == Person.THIRD
        assert morph.number == Number.SINGULAR

    def test_parse_noun_dash_format(self) -> None:
        """Test parsing Greek noun in dash format."""
        # Noun, Nominative Singular Masculine
        morph = self.parser.parse("N-NSM")

        assert morph.part_of_speech == "noun"
        assert morph.case.value == "nominative"
        assert morph.number == Number.SINGULAR
        assert morph.gender == Gender.MASCULINE

    def test_parse_verb_compact_format(self) -> None:
        """Test parsing Greek verb in compact format."""
        # Verb Present Active Indicative 3rd Singular
        morph = self.parser.parse("VPAI3S")

        assert morph.part_of_speech == "verb"
        assert morph.tense == Tense.PRESENT
        assert morph.voice == Voice.ACTIVE
        assert morph.mood == Mood.INDICATIVE
        assert morph.person == Person.THIRD
        assert morph.number == Number.SINGULAR

    def test_parse_participle(self) -> None:
        """Test parsing Greek participle."""
        # Verb, Present Active Participle, Nominative Singular Masculine
        morph = self.parser.parse("V-PAP-NSM")

        assert morph.part_of_speech == "verb"
        assert morph.mood == Mood.PARTICIPLE
        assert morph.case.value == "nominative"
        assert morph.gender == Gender.MASCULINE
        assert morph.number == Number.SINGULAR

    def test_parse_article(self) -> None:
        """Test parsing Greek article."""
        # Article (T for Definite article), Nominative Singular Masculine
        morph = self.parser.parse("T-NSM")

        assert morph.part_of_speech == "article"
        assert morph.case.value == "nominative"
        assert morph.number == Number.SINGULAR
        assert morph.gender == Gender.MASCULINE

    def test_greek_summary(self) -> None:
        """Test Greek morphology summary generation."""
        morph = self.parser.parse("V-PAI-3S")
        summary = morph.get_greek_summary()

        assert "verb" in summary
        assert "present" in summary
        assert "active" in summary
        assert "indicative" in summary
        assert "3rd" in summary
        assert "singular" in summary

    def test_get_part_of_speech(self) -> None:
        """Test extracting part of speech."""
        assert self.parser.get_part_of_speech("V-PAI-3S") == "verb"
        assert self.parser.get_part_of_speech("N-NSM") == "noun"
        assert self.parser.get_part_of_speech("VPAI3S") == "verb"
        assert self.parser.get_part_of_speech("") is None


class TestUnifiedMorphologyParser:
    """Test unified morphology parsing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.parser = UnifiedMorphologyParser()

    def test_parse_hebrew(self) -> None:
        """Test parsing Hebrew through unified interface."""
        morph = self.parser.parse("Vqp3ms", Language.HEBREW)

        assert morph.language == Language.HEBREW
        assert isinstance(morph.features, HebrewMorphology)
        assert morph.features.part_of_speech == "verb"
        assert morph.original_code == "Vqp3ms"

    def test_parse_greek(self) -> None:
        """Test parsing Greek through unified interface."""
        morph = self.parser.parse("V-PAI-3S", Language.GREEK)

        assert morph.language == Language.GREEK
        assert isinstance(morph.features, GreekMorphology)
        assert morph.features.part_of_speech == "verb"
        assert morph.original_code == "V-PAI-3S"

    def test_auto_detect_greek(self) -> None:
        """Test automatic language detection for Greek."""
        # Dash format is typically Greek
        morph = self.parser.parse_auto_detect("V-PAI-3S")
        assert morph.language == Language.GREEK

        # Greek tense pattern
        morph = self.parser.parse_auto_detect("VPAI3S")
        assert morph.language == Language.GREEK

    def test_auto_detect_hebrew(self) -> None:
        """Test automatic language detection for Hebrew."""
        # Hebrew prefix pattern
        morph = self.parser.parse_auto_detect("HNcmsa")
        assert morph.language == Language.HEBREW

        # Hebrew stem pattern
        morph = self.parser.parse_auto_detect("Vqp3ms")
        assert morph.language == Language.HEBREW

    def test_is_verb(self) -> None:
        """Test verb identification."""
        morph = self.parser.parse("Vqp3ms", Language.HEBREW)
        assert morph.is_verb()

        morph = self.parser.parse("Ncmsa", Language.HEBREW)
        assert not morph.is_verb()

    def test_is_participle(self) -> None:
        """Test participle identification."""
        morph = self.parser.parse("V-PAP-NSM", Language.GREEK)
        assert morph.is_participle()

        morph = self.parser.parse("V-PAI-3S", Language.GREEK)
        assert not morph.is_participle()

    def test_compare_morphologies(self) -> None:
        """Test morphology comparison."""
        morph1 = self.parser.parse("Vqp3ms", Language.HEBREW)
        morph2 = self.parser.parse("V-PAI-3S", Language.GREEK)

        comparison = self.parser.compare_morphologies(morph1, morph2)

        assert comparison["languages"] == ["hebrew", "greek"]
        assert "agreements" in comparison
        assert "differences" in comparison
        assert comparison["agreements"]["part_of_speech"] == "verb"

    def test_morphology_statistics(self) -> None:
        """Test morphology statistics calculation."""
        morphologies = [
            self.parser.parse("Vqp3ms", Language.HEBREW),
            self.parser.parse("V-PAI-3S", Language.GREEK),
            self.parser.parse("Ncmsa", Language.HEBREW),
            self.parser.parse("N-NSM", Language.GREEK),
        ]

        stats = self.parser.get_morphology_statistics(morphologies)

        assert stats["total"] == 4
        assert stats["by_language"]["hebrew"] == 2
        assert stats["by_language"]["greek"] == 2
        assert stats["by_part_of_speech"]["verb"] == 2
        assert stats["by_part_of_speech"]["noun"] == 2


class TestMorphologyFeatures:
    """Test morphology feature classes."""

    def test_morphology_to_dict(self) -> None:
        """Test converting morphology features to dictionary."""
        morph = HebrewMorphology(
            part_of_speech="verb",
            stem=Stem.QAL,
            tense=Tense.PERFECT,
            person=Person.THIRD,
            gender=Gender.MASCULINE,
            number=Number.SINGULAR,
        )

        result = morph.to_dict()

        assert result["part_of_speech"] == "verb"
        assert result["stem"] == "qal"
        assert result["tense"] == "perfect"
        assert result["person"] == "3rd"
        assert result["gender"] == "masculine"
        assert result["number"] == "singular"

    def test_unified_morphology_to_dict(self) -> None:
        """Test converting unified morphology to dictionary."""
        parser = UnifiedMorphologyParser()
        morph = parser.parse("Vqp3ms", Language.HEBREW)

        result = morph.to_dict()

        assert result["language"] == "hebrew"
        assert result["original_code"] == "Vqp3ms"
        assert "features" in result
        assert "summary" in result
