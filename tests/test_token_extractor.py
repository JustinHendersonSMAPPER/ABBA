"""
Tests for the token extractor module.

Test coverage for extracting tokens from Hebrew and Greek biblical texts.
"""

import pytest
from typing import List

from abba.interlinear.token_extractor import (
    ExtractedToken,
    HebrewTokenExtractor,
    GreekTokenExtractor,
)
from abba.verse_id import VerseID
from abba.parsers.hebrew_parser import HebrewVerse, HebrewWord
from abba.parsers.greek_parser import GreekVerse, GreekWord
from abba.morphology.base import Language


class TestExtractedToken:
    """Test the ExtractedToken dataclass."""
    
    def test_token_creation(self):
        """Test creating an extracted token."""
        token = ExtractedToken(
            text="אֱלֹהִים",
            lemma="אֱלֹהִים",
            strong_number="H430",
            position=2,
            language=Language.HEBREW,
            gloss="God",
            transliteration="elohim"
        )
        
        assert token.text == "אֱלֹהִים"
        assert token.lemma == "אֱלֹהִים"
        assert token.strong_number == "H430"
        assert token.position == 2
        assert token.language == Language.HEBREW
        assert token.gloss == "God"
        assert token.transliteration == "elohim"
        
    def test_token_minimal(self):
        """Test token with minimal fields."""
        token = ExtractedToken(
            text="word",
            position=0
        )
        
        assert token.text == "word"
        assert token.position == 0
        assert token.lemma is None
        assert token.strong_number is None
        assert token.morphology is None
        assert token.gloss is None
        assert token.transliteration is None
        
    def test_token_to_dict(self):
        """Test converting token to dictionary."""
        token = ExtractedToken(
            text="λόγος",
            lemma="λόγος",
            strong_number="G3056",
            position=4,
            language=Language.GREEK,
            gloss="word",
            transliteration="logos"
        )
        
        token_dict = token.to_dict()
        
        assert token_dict["text"] == "λόγος"
        assert token_dict["position"] == 4
        assert token_dict["language"] == "greek"
        assert token_dict["lemma"] == "λόγος"
        assert token_dict["strong_number"] == "G3056"
        assert token_dict["gloss"] == "word"
        assert token_dict["transliteration"] == "logos"


class TestHebrewTokenExtractor:
    """Test the HebrewTokenExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a Hebrew token extractor."""
        return HebrewTokenExtractor()
        
    @pytest.fixture
    def hebrew_verse(self):
        """Create a sample Hebrew verse."""
        return HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[
                HebrewWord(
                    text="בְּרֵאשִׁית",
                    lemma="רֵאשִׁית",
                    strong_number="H7225",
                    morph="Ncfsa"
                ),
                HebrewWord(
                    text="בָּרָא",
                    lemma="בָּרָא",
                    strong_number="H1254",
                    morph="Vqp3ms"
                ),
                HebrewWord(
                    text="אֱלֹהִים",
                    lemma="אֱלֹהִים",
                    strong_number="H430",
                    morph="Ncmpa"
                )
            ],
            osis_id="Gen.1.1"
        )
        
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert hasattr(extractor, 'extract_tokens')
        assert hasattr(extractor, 'extract_tokens_from_text')
        assert hasattr(extractor, '_transliteration_map')
        
    def test_extract_tokens_from_verse(self, extractor, hebrew_verse):
        """Test extracting tokens from Hebrew verse."""
        tokens = extractor.extract_tokens(hebrew_verse)
        
        assert len(tokens) == 3
        
        # Check first token
        first_token = tokens[0]
        assert first_token.text == "בְּרֵאשִׁית"
        assert first_token.lemma == "רֵאשִׁית"
        assert first_token.strong_number == "H7225"
        assert first_token.position == 0
        assert first_token.language == Language.HEBREW
        assert first_token.transliteration is not None
        
        # Check morphology was parsed
        assert first_token.morphology is not None
        
    def test_extract_tokens_from_text(self, extractor):
        """Test extracting tokens from plain Hebrew text."""
        text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        tokens = extractor.extract_tokens_from_text(text)
        
        assert len(tokens) == 3
        
        # Check tokens were extracted
        assert tokens[0].text == "בְּרֵאשִׁית"
        assert tokens[1].text == "בָּרָא"
        assert tokens[2].text == "אֱלֹהִים"
        
        # Check positions
        assert tokens[0].position == 0
        assert tokens[1].position == 1
        assert tokens[2].position == 2
        
        # Check language
        for token in tokens:
            assert token.language == Language.HEBREW
            
    def test_transliteration(self, extractor):
        """Test Hebrew transliteration."""
        # Test basic consonants
        assert extractor._transliterate("א") == "'"
        assert extractor._transliterate("ב") == "b"
        assert extractor._transliterate("ג") == "g"
        assert extractor._transliterate("ש") == "sh"
        
        # Test final forms
        assert extractor._transliterate("ך") == "k"
        assert extractor._transliterate("ם") == "m"
        assert extractor._transliterate("ן") == "n"
        assert extractor._transliterate("ף") == "p"
        assert extractor._transliterate("ץ") == "ts"
        
    def test_empty_verse(self, extractor):
        """Test extracting from empty verse."""
        verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[],
            osis_id="Gen.1.1"
        )
        
        tokens = extractor.extract_tokens(verse)
        assert len(tokens) == 0
        
    def test_morphology_parsing(self, extractor, hebrew_verse):
        """Test that morphology is properly parsed."""
        tokens = extractor.extract_tokens(hebrew_verse)
        
        # Check first word (noun)
        noun_token = tokens[0]
        assert noun_token.morphology is not None
        assert noun_token.morphology.features.part_of_speech == "noun"
        
        # Check second word (verb)
        verb_token = tokens[1]
        assert verb_token.morphology is not None
        assert verb_token.morphology.features.part_of_speech == "verb"
        
    def test_word_with_gloss(self, extractor):
        """Test extracting word with gloss."""
        verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=[
                HebrewWord(
                    text="אֱלֹהִים",
                    lemma="אֱלֹהִים",
                    strong_number="H430",
                    morph="Ncmpa",
                    gloss="God"  # HebrewWord has gloss field
                )
            ],
            osis_id="Gen.1.1"
        )
        
        tokens = extractor.extract_tokens(verse)
        assert len(tokens) == 1
        assert tokens[0].gloss == "God"


class TestGreekTokenExtractor:
    """Test the GreekTokenExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a Greek token extractor."""
        return GreekTokenExtractor()
        
    @pytest.fixture
    def greek_verse(self):
        """Create a sample Greek verse."""
        return GreekVerse(
            verse_id=VerseID("JHN", 1, 1),
            words=[
                GreekWord(
                    text="Ἐν",
                    lemma="ἐν",
                    strong_number="G1722",
                    morph="P"
                ),
                GreekWord(
                    text="ἀρχῇ",
                    lemma="ἀρχή",
                    strong_number="G746",
                    morph="N-DSF"
                ),
                GreekWord(
                    text="ἦν",
                    lemma="εἰμί",
                    strong_number="G1510",
                    morph="V-IAI-3S"
                ),
                GreekWord(
                    text="ὁ",
                    lemma="ὁ",
                    strong_number="G3588",
                    morph="T-NSM"
                ),
                GreekWord(
                    text="λόγος",
                    lemma="λόγος",
                    strong_number="G3056",
                    morph="N-NSM"
                )
            ],
            tei_id="B01K1V1"
        )
        
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert hasattr(extractor, 'extract_tokens')
        assert hasattr(extractor, 'extract_tokens_from_text')
        assert hasattr(extractor, '_transliteration_map')
        
    def test_extract_tokens_from_verse(self, extractor, greek_verse):
        """Test extracting tokens from Greek verse."""
        tokens = extractor.extract_tokens(greek_verse)
        
        assert len(tokens) == 5
        
        # Check first token (preposition)
        first_token = tokens[0]
        assert first_token.text == "Ἐν"
        assert first_token.lemma == "ἐν"
        assert first_token.strong_number == "G1722"
        assert first_token.position == 0
        assert first_token.language == Language.GREEK
        assert first_token.transliteration is not None
        
        # Check last token (noun)
        last_token = tokens[-1]
        assert last_token.text == "λόγος"
        assert last_token.lemma == "λόγος"
        assert last_token.strong_number == "G3056"
        
    def test_extract_tokens_from_text(self, extractor):
        """Test extracting tokens from plain Greek text."""
        text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
        tokens = extractor.extract_tokens_from_text(text)
        
        assert len(tokens) == 5
        
        # Check tokens were extracted
        assert tokens[0].text == "Ἐν"
        assert tokens[1].text == "ἀρχῇ"
        assert tokens[2].text == "ἦν"
        assert tokens[3].text == "ὁ"
        assert tokens[4].text == "λόγος"
        
        # Check language
        for token in tokens:
            assert token.language == Language.GREEK
            
    def test_transliteration(self, extractor):
        """Test Greek transliteration."""
        # Test basic letters
        assert extractor._transliterate("α") == "a"
        assert extractor._transliterate("β") == "b"
        assert extractor._transliterate("λ") == "l"
        assert extractor._transliterate("ω") == "ō"
        
        # Test uppercase
        assert extractor._transliterate("Α") == "A"
        assert extractor._transliterate("Λ") == "L"
        assert extractor._transliterate("Ω") == "Ō"
        
        # Test digraphs
        assert extractor._transliterate("θ") == "th"
        assert extractor._transliterate("φ") == "ph"
        assert extractor._transliterate("χ") == "ch"
        assert extractor._transliterate("ψ") == "ps"
        
        # Test combinations
        assert extractor._transliterate("ου") == "ou"
        assert extractor._transliterate("ει") == "ei"
        assert extractor._transliterate("αι") == "ai"
        
    def test_strong_number_extraction(self, extractor):
        """Test extracting Strong's number from lemma."""
        # Test verse with Strong's in lemma
        verse = GreekVerse(
            verse_id=VerseID("JHN", 1, 1),
            words=[
                GreekWord(
                    text="λόγος",
                    lemma="λόγος G3056",  # Strong's in lemma
                    strong_number=None,   # Not in strong_number field
                    morph="N-NSM"
                )
            ],
            tei_id="B01K1V1"
        )
        
        tokens = extractor.extract_tokens(verse)
        
        assert len(tokens) == 1
        assert tokens[0].strong_number == "G3056"
        
    def test_empty_verse(self, extractor):
        """Test extracting from empty verse."""
        verse = GreekVerse(
            verse_id=VerseID("JHN", 1, 1),
            words=[],
            tei_id="B01K1V1"
        )
        
        tokens = extractor.extract_tokens(verse)
        assert len(tokens) == 0
        
    def test_morphology_parsing(self, extractor, greek_verse):
        """Test that morphology is properly parsed."""
        tokens = extractor.extract_tokens(greek_verse)
        
        # Check preposition
        prep_token = tokens[0]
        assert prep_token.morphology is not None
        assert prep_token.morphology.features.part_of_speech == "preposition"
        
        # Check noun
        noun_token = tokens[1]
        assert noun_token.morphology is not None
        assert noun_token.morphology.features.part_of_speech == "noun"
        
        # Check verb
        verb_token = tokens[2]
        assert verb_token.morphology is not None
        assert verb_token.morphology.features.part_of_speech == "verb"