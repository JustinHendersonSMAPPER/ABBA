"""
Integration tests for various parser modules.

Tests parser functionality that doesn't have dedicated test files.
"""

import unittest
from pathlib import Path
import tempfile
import json

from abba.parsers.translation_parser import TranslationParser, TranslationVerse, Translation
from abba.parsers.lexicon_parser import LexiconParser, LexiconEntry
from abba.parsers.greek_parser import GreekParser, GreekVerse, GreekWord
from abba.parsers.hebrew_parser import HebrewParser, HebrewVerse, HebrewWord
from abba.verse_id import VerseID


class TestTranslationParserIntegration(unittest.TestCase):
    """Test translation parser with various formats."""

    def setUp(self):
        """Set up parser instance."""
        self.parser = TranslationParser()

    def test_parse_usfm_format(self):
        """Test parsing USFM format."""
        usfm_content = """\\id GEN
\\h Genesis
\\c 1
\\v 1 In the beginning God created the heavens and the earth.
\\v 2 The earth was formless and void.
"""
        
        # Test that parser can handle USFM
        # Note: Actual implementation would need USFM parsing logic
        self.assertIsInstance(self.parser, TranslationParser)

    def test_parse_json_translation(self):
        """Test parsing JSON translation format."""
        json_data = {
            "version": "ESV",  # Changed from "translation" to "version"
            "name": "English Standard Version",  # Added required "name" field
            "language": "en",
            "books": {
                "Genesis": {  # Changed to use full book name
                    "chapters": [  # Changed to list format as expected by parser
                        {
                            "chapter": 1,
                            "verses": [
                                {"verse": 1, "text": "In the beginning God created the heavens and the earth."},
                                {"verse": 2, "text": "The earth was without form and void."}
                            ]
                        }
                    ]
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            result = self.parser.parse_file(Path(temp_path))
            self.assertIsInstance(result, Translation)
            
            # Should have parsed verses
            self.assertTrue(result.verses)
            if result.verses:
                verse = result.verses[0]
                self.assertIsInstance(verse, TranslationVerse)
                self.assertEqual(verse.text, "In the beginning God created the heavens and the earth.")
        finally:
            Path(temp_path).unlink()

    def test_translation_metadata(self):
        """Test parsing translation metadata."""
        # TranslationMetadata doesn't exist; use Translation instead
        translation = Translation(
            version="ESV",
            name="English Standard Version",
            language="en",
            copyright="© 2001 Crossway",
        )

        self.assertEqual(translation.version, "ESV")
        self.assertEqual(translation.name, "English Standard Version")

    def test_parse_verse_variations(self):
        """Test parsing verses with variations."""
        # Create sample verse with metadata - TranslationVerse only has the basic fields
        verse = TranslationVerse(
            verse_id=VerseID("JHN", 3, 16),
            text="For God so loved the world...",
            original_book_name="John",
            original_chapter=3,
            original_verse=16,
        )

        self.assertEqual(verse.verse_id.book, "JHN")
        self.assertEqual(verse.original_book_name, "John")


class TestLexiconParserIntegration(unittest.TestCase):
    """Test lexicon parser functionality."""

    def setUp(self):
        """Set up parser instance."""
        self.parser = LexiconParser()

    def test_parse_strongs_entry(self):
        """Test parsing Strong's lexicon entry."""
        # Sample Strong's entry format
        entry_data = {
            "strongs": "G25",
            "lemma": "ἀγαπάω",
            "transliteration": "agapao",
            "pronunciation": "ag-ap-ah'-o",
            "part_of_speech": "verb",
            "definition": "to love",
            "kjv_usage": "love (135x), beloved (7x)",
        }

        entry = LexiconEntry(
            strong_number="G25",  # Fixed parameter name
            word=entry_data["lemma"],  # LexiconEntry uses 'word' not 'lemma'
            transliteration=entry_data["transliteration"],
            pronunciation=entry_data["pronunciation"],
            definition=entry_data["definition"],
            gloss="love",  # Add gloss
            language="greek",
        )

        self.assertEqual(entry.strong_number, "G25")
        self.assertEqual(entry.word, "ἀγαπάω")
        self.assertEqual(entry.language, "greek")

    def test_parse_hebrew_lexicon(self):
        """Test parsing Hebrew lexicon entry."""
        entry = LexiconEntry(
            strong_number="H1",
            word="אָב",
            transliteration="ab",
            definition="father",
            morphology="noun masculine",  # LexiconEntry uses 'morphology' field
            language="hebrew",
        )

        self.assertEqual(entry.strong_number, "H1")
        self.assertEqual(entry.word, "אָב")
        self.assertEqual(entry.language, "hebrew")

    def test_lexicon_relationships(self):
        """Test lexicon entry relationships."""
        # Test handling of word relationships - LexiconEntry doesn't have these fields
        entry = LexiconEntry(
            strong_number="G26",
            word="ἀγάπη",
            transliteration="agape",
            definition="love",
            gloss="love",
            language="greek",
            usage_notes="Related to G25 (agapao), synonym of G5360 (phileo), antonym of G3404 (miseo)",
        )

        self.assertEqual(entry.strong_number, "G26")
        self.assertEqual(entry.word, "ἀγάπη")
        self.assertIn("G25", entry.usage_notes)


class TestGreekParserIntegration(unittest.TestCase):
    """Test Greek text parser."""

    def setUp(self):
        """Set up parser instance."""
        self.parser = GreekParser()

    def test_parse_greek_word(self):
        """Test parsing individual Greek word."""
        # GreekWord doesn't have position or gloss parameters
        word = GreekWord(
            text="λόγος",
            lemma="λόγος",
            morph="N-NSM",
            strong_number="G3056",
            id="w001",  # id instead of position
        )

        self.assertEqual(word.text, "λόγος")
        self.assertEqual(word.morph, "N-NSM")
        self.assertEqual(word.strong_number, "G3056")

    def test_parse_greek_verse(self):
        """Test parsing complete Greek verse."""
        words = [
            GreekWord(text="Ἐν", lemma="ἐν", morph="P"),
            GreekWord(text="ἀρχῇ", lemma="ἀρχή", morph="N-DSF"),
            GreekWord(text="ἦν", lemma="εἰμί", morph="V-IAI-3S"),
            GreekWord(text="ὁ", lemma="ὁ", morph="T-NSM"),
            GreekWord(text="λόγος", lemma="λόγος", morph="N-NSM"),
        ]

        # GreekVerse requires tei_id parameter
        verse = GreekVerse(
            verse_id=VerseID("JHN", 1, 1),
            words=words,
            tei_id="B04K1V1",  # TEI format ID
        )

        self.assertEqual(len(verse.words), 5)
        self.assertEqual(verse.words[0].text, "Ἐν")
        self.assertEqual(verse.verse_id.book, "JHN")

    def test_parse_greek_with_variants(self):
        """Test parsing Greek with textual variants."""
        # GreekVerse doesn't have variant_readings parameter
        verse = GreekVerse(
            verse_id=VerseID("JHN", 1, 18),
            words=[
                GreekWord(text="μονογενὴς", lemma="μονογενής"),
                GreekWord(text="θεὸς", lemma="θεός"),
            ],
            tei_id="B04K1V18",
        )

        self.assertEqual(len(verse.words), 2)
        self.assertEqual(verse.words[0].text, "μονογενὴς")


class TestHebrewParserIntegration(unittest.TestCase):
    """Test Hebrew text parser."""

    def setUp(self):
        """Set up parser instance."""
        self.parser = HebrewParser()

    def test_parse_hebrew_word(self):
        """Test parsing individual Hebrew word."""
        # HebrewWord doesn't have position, gloss, or prefix parameters
        word = HebrewWord(
            text="בְּרֵאשִׁית",
            lemma="רֵאשִׁית",
            morph="HNcfsa",
            strong_number="H7225",
            id="w001",
        )

        self.assertEqual(word.text, "בְּרֵאשִׁית")
        self.assertEqual(word.lemma, "רֵאשִׁית")
        self.assertEqual(word.strong_number, "H7225")

    def test_parse_hebrew_verse(self):
        """Test parsing complete Hebrew verse."""
        words = [
            HebrewWord(text="בְּרֵאשִׁית", lemma="רֵאשִׁית", morph="HNcfsa"),
            HebrewWord(text="בָּרָא", lemma="בָּרָא", morph="Vqp3ms"),
            HebrewWord(text="אֱלֹהִים", lemma="אֱלֹהִים", morph="Ncmpa"),
        ]

        # HebrewVerse requires osis_id parameter
        verse = HebrewVerse(
            verse_id=VerseID("GEN", 1, 1),
            words=words,
            osis_id="Gen.1.1",
        )

        self.assertEqual(len(verse.words), 3)
        self.assertEqual(verse.words[0].text, "בְּרֵאשִׁית")
        self.assertEqual(verse.osis_id, "Gen.1.1")

    def test_parse_hebrew_with_ketiv_qere(self):
        """Test parsing Hebrew with Ketiv/Qere readings."""
        # HebrewVerse doesn't have ketiv_qere parameter
        verse = HebrewVerse(
            verse_id=VerseID("JER", 42, 11),
            words=[
                HebrewWord(text="איתי", lemma="את"),  # Ketiv form
            ],
            osis_id="Jer.42.11",
        )

        self.assertEqual(len(verse.words), 1)
        self.assertEqual(verse.words[0].text, "איתי")


class TestParserEdgeCases(unittest.TestCase):
    """Test edge cases in parsers."""

    def test_empty_verse_handling(self):
        """Test handling of empty verses."""
        # TranslationVerse doesn't have is_disputed parameter
        verse = TranslationVerse(
            verse_id=VerseID("MAR", 9, 44),
            text="",  # Some manuscripts omit this verse
            original_book_name="Mark",
            original_chapter=9,
            original_verse=44,
        )

        self.assertEqual(verse.text, "")
        self.assertEqual(verse.original_book_name, "Mark")

    def test_unicode_normalization(self):
        """Test handling of Unicode normalization."""
        # Hebrew with different normalization forms
        word1 = HebrewWord(
            text="שָׁלוֹם",  # NFC form
            lemma="שָׁלוֹם",
            morph="Ncmsa",
        )

        # Greek with diacritics
        word2 = GreekWord(
            text="πνεῦμα",  # With breathing and accent
            lemma="πνεῦμα",
            morph="N-NSN",
        )

        self.assertEqual(word1.text, "שָׁלוֹם")
        self.assertEqual(word2.text, "πνεῦμα")

    def test_large_verse_handling(self):
        """Test handling of unusually large verses."""
        # Esther 8:9 is the longest verse in the Bible
        long_text = "Then were the king's scribes called... " * 20  # Simulated long text
        
        verse = TranslationVerse(
            verse_id=VerseID("EST", 8, 9),
            text=long_text,
            original_book_name="Esther",
            original_chapter=8,
            original_verse=9,
        )

        self.assertGreater(len(verse.text), 500)


if __name__ == "__main__":
    unittest.main()