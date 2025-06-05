"""
Tests for the interlinear generator module.

Test coverage for generating interlinear displays of biblical texts
with aligned original language and translation data.
"""

import pytest
from typing import List

from abba.interlinear.interlinear_generator import (
    InterlinearGenerator,
    InterlinearVerse,
    InterlinearWord,
)
from abba.verse_id import VerseID
from abba.parsers.hebrew_parser import HebrewVerse, HebrewWord
from abba.parsers.greek_parser import GreekVerse, GreekWord
from abba.parsers.translation_parser import TranslationVerse
from abba.interlinear.token_extractor import ExtractedToken
from abba.interlinear.token_alignment import TokenAlignment, AlignedToken
from abba.morphology.base import Language


class TestInterlinearWord:
    """Test the InterlinearWord dataclass."""
    
    def test_word_creation(self):
        """Test creating an interlinear word."""
        word = InterlinearWord(
            original_text="בְּרֵאשִׁית",
            transliteration="bereshit",
            lemma="רֵאשִׁית",
            strong_number="H7225",
            gloss="beginning",
            translation_words=["In", "beginning"],
            position=0,
            language=Language.HEBREW
        )
        
        assert word.original_text == "בְּרֵאשִׁית"
        assert word.transliteration == "bereshit"
        assert word.lemma == "רֵאשִׁית"
        assert word.strong_number == "H7225"
        assert word.gloss == "beginning"
        assert word.translation_words == ["In", "beginning"]
        
    def test_word_to_dict(self):
        """Test converting word to dictionary."""
        word = InterlinearWord(
            original_text="λόγος",
            transliteration="logos",
            lemma="λόγος",
            strong_number="G3056",
            gloss="word",
            language=Language.GREEK
        )
        
        word_dict = word.to_dict()
        
        assert word_dict["original"] == "λόγος"
        assert word_dict["transliteration"] == "logos"
        assert word_dict["strong_number"] == "G3056"
        assert word_dict["language"] == "greek"
        
    def test_word_display_lines(self):
        """Test getting display lines."""
        word = InterlinearWord(
            original_text="אֱלֹהִים",
            transliteration="elohim",
            morphology_gloss="Noun, masculine plural",
            gloss="God",
            translation_words=["God"]
        )
        
        lines = word.get_display_lines()
        
        assert lines["original"] == "אֱלֹהִים"
        assert lines["transliteration"] == "elohim"
        assert lines["morphology"] == "Noun, masculine plural"
        assert lines["gloss"] == "God"
        assert lines["translation"] == "God"


class TestInterlinearVerse:
    """Test the InterlinearVerse dataclass."""
    
    def test_verse_creation(self):
        """Test creating an interlinear verse."""
        words = [
            InterlinearWord(
                original_text="בְּרֵאשִׁית",
                transliteration="bereshit",
                position=0
            ),
            InterlinearWord(
                original_text="בָּרָא",
                transliteration="bara",
                position=1
            )
        ]
        
        verse = InterlinearVerse(
            verse_id=VerseID("GEN", 1, 1),
            language=Language.HEBREW,
            words=words,
            original_text="בְּרֵאשִׁית בָּרָא",
            transliteration="bereshit bara"
        )
        
        assert verse.verse_id.book == "GEN"
        assert verse.language == Language.HEBREW
        assert len(verse.words) == 2
        
    def test_verse_to_dict(self):
        """Test converting verse to dictionary."""
        verse = InterlinearVerse(
            verse_id=VerseID("JHN", 1, 1),
            language=Language.GREEK,
            words=[],
            original_text="Ἐν ἀρχῇ",
            translation_text="In the beginning"
        )
        
        verse_dict = verse.to_dict()
        
        assert verse_dict["verse_id"] == "JHN.1.1"
        assert verse_dict["language"] == "greek"
        assert verse_dict["original_text"] == "Ἐν ἀρχῇ"
        assert verse_dict["translation_text"] == "In the beginning"
        
    def test_aligned_display(self):
        """Test aligned interlinear display."""
        words = [
            InterlinearWord(
                original_text="בְּרֵאשִׁית",
                transliteration="bereshit",
                morphology_gloss="Noun",
                gloss="beginning",
                translation_words=["beginning"]
            ),
            InterlinearWord(
                original_text="בָּרָא",
                transliteration="bara",
                morphology_gloss="Verb",
                gloss="created",
                translation_words=["created"]
            )
        ]
        
        verse = InterlinearVerse(
            verse_id=VerseID("GEN", 1, 1),
            language=Language.HEBREW,
            words=words
        )
        
        display = verse.get_interlinear_display("aligned")
        
        assert "בְּרֵאשִׁית" in display
        assert "bereshit" in display
        assert "beginning" in display
        
    def test_sequential_display(self):
        """Test sequential interlinear display."""
        words = [
            InterlinearWord(
                original_text="λόγος",
                transliteration="logos",
                lemma="λόγος",
                strong_number="G3056",
                position=0
            )
        ]
        
        verse = InterlinearVerse(
            verse_id=VerseID("JHN", 1, 1),
            language=Language.GREEK,
            words=words
        )
        
        display = verse.get_interlinear_display("sequential")
        
        assert "Word 1:" in display
        assert "Original: λόγος" in display
        assert "Transliteration: logos" in display
        assert "Strong's: G3056" in display


class TestInterlinearGenerator:
    """Test the InterlinearGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create an interlinear generator."""
        return InterlinearGenerator()
        
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
        
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert hasattr(generator, 'hebrew_extractor')
        assert hasattr(generator, 'greek_extractor')
        
    def test_generate_hebrew_interlinear(self, generator, hebrew_verse):
        """Test generating Hebrew interlinear."""
        result = generator.generate_hebrew_interlinear(
            hebrew_verse,
            translation_text="In the beginning God created"
        )
        
        assert result.verse_id == hebrew_verse.verse_id
        assert result.language == Language.HEBREW
        assert len(result.words) == 3
        
        # Check first word
        first_word = result.words[0]
        assert first_word.original_text == "בְּרֵאשִׁית"
        assert first_word.strong_number == "H7225"
        
    def test_generate_greek_interlinear(self, generator, greek_verse):
        """Test generating Greek interlinear."""
        result = generator.generate_greek_interlinear(
            greek_verse,
            translation_text="In the beginning was the Word"
        )
        
        assert result.verse_id == greek_verse.verse_id
        assert result.language == Language.GREEK
        assert len(result.words) == 5
        
        # Check last word
        last_word = result.words[-1]
        assert last_word.original_text == "λόγος"
        assert last_word.strong_number == "G3056"
        
    def test_generate_with_alignment(self, generator, hebrew_verse):
        """Test generating with token alignment."""
        # Create mock alignment
        source_token = ExtractedToken(
            text="בְּרֵאשִׁית",
            position=0
        )
        
        from abba.interlinear.token_alignment import AlignmentType
        
        aligned_token = AlignedToken(
            source_tokens=[source_token],
            target_words=["In", "the", "beginning"],
            alignment_type=AlignmentType.ONE_TO_MANY,
            confidence=0.95
        )
        
        alignment = TokenAlignment(
            verse_id="GEN.1.1",
            source_language="hebrew",
            target_language="english",
            alignments=[aligned_token],
            unaligned_source=[],
            unaligned_target=[],
            alignment_score=0.9
        )
        
        result = generator.generate_hebrew_interlinear(
            hebrew_verse,
            translation_text="In the beginning God created",
            alignment=alignment
        )
        
        # Check that alignment was applied
        first_word = result.words[0]
        assert first_word.translation_words == ["In", "the", "beginning"]
        
    def test_parallel_interlinear(self, generator, hebrew_verse, greek_verse):
        """Test generating parallel interlinear."""
        verses = {
            "HEB": hebrew_verse,
            "GRK": greek_verse
        }
        
        translations = {
            "HEB": "In the beginning God created",
            "GRK": "In the beginning was the Word"
        }
        
        results = generator.generate_parallel_interlinear(verses, translations)
        
        assert len(results) == 2
        assert "HEB" in results
        assert "GRK" in results
        
        assert results["HEB"].language == Language.HEBREW
        assert results["GRK"].language == Language.GREEK
        
    def test_compare_interlinear_verses(self, generator, hebrew_verse, greek_verse):
        """Test comparing interlinear verses."""
        heb_interlinear = generator.generate_hebrew_interlinear(hebrew_verse)
        grk_interlinear = generator.generate_greek_interlinear(greek_verse)
        
        comparison = generator.compare_interlinear_verses(
            heb_interlinear,
            grk_interlinear
        )
        
        assert comparison["verse_id"] == "GEN.1.1"
        assert comparison["languages"] == ["hebrew", "greek"]
        assert "word_count_diff" in comparison
        assert "common_strongs" in comparison
        assert "morphology_comparison" in comparison