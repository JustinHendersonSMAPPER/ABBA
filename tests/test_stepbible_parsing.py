"""Tests for STEPBible data parsing functionality."""

import tempfile
import unittest
from pathlib import Path

from abba.bible_extractor import BibleExtractor
from abba.database import SQLiteManager


class TestSTEPBibleParsing(unittest.TestCase):
    """Test STEPBible parsing functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        self.stepbible_dir = self.data_dir / "stepbible"
        self.stepbible_dir.mkdir(exist_ok=True)

        self.extractor = BibleExtractor(str(self.data_dir))

        # Set up test database
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db_manager = SQLiteManager(self.db_path)
        self.db_manager.initialize_database()

    def test_parse_lexicon_hebrew(self):
        """Test parsing Hebrew lexicon from OpenScriptures XML."""
        lexicon_xml = """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
    <entry id="H1">
        <w pos="n-m" pron="awb" xlit="ab" xml:lang="heb">אָב</w>
        <meaning><def>father</def>, male parent</meaning>
        <usage>chief, father.</usage>
    </entry>
    <entry id="H6">
        <w pos="v" pron="aw-bad" xlit="abad" xml:lang="heb">אָבַד</w>
        <meaning><def>perish</def></meaning>
        <usage>break, destroy.</usage>
    </entry>
</lexicon>"""

        lexicon_file = self.stepbible_dir / "hebrew_strongs.xml"
        with open(lexicon_file, "w", encoding="utf-8") as f:
            f.write(lexicon_xml)

        result = self.extractor.parse_lexicon("hebrew", self.db_manager)
        self.assertTrue(result)

        lexicon_entry = self.db_manager.get_lexicon_entry("H1")
        self.assertIsNotNone(lexicon_entry)
        self.assertEqual(lexicon_entry["original_word"], "אָב")
        self.assertEqual(lexicon_entry["transliteration"], "ab")

    def test_parse_lexicon_greek(self):
        """Test parsing Greek lexicon from Abbott-Smith XML."""
        lexicon_xml = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.crosswire.org/2013/TEIOSIS/namespace">
 <teiHeader><fileDesc><titleStmt><title>Test</title></titleStmt>
 <publicationStmt><date>1922</date></publicationStmt>
 <sourceDesc><p>test</p></sourceDesc></fileDesc></teiHeader>
 <text><body>
  <entry n="ἀγάπη|G26">
    <form><orth>ἀγάπη</orth></form>
    <sense><gloss>love</gloss>, divine love</sense>
  </entry>
 </body></text>
</TEI>"""

        lexicon_file = self.stepbible_dir / "abbott_smith.xml"
        with open(lexicon_file, "w", encoding="utf-8") as f:
            f.write(lexicon_xml)

        result = self.extractor.parse_lexicon("greek", self.db_manager)
        self.assertTrue(result)

        lexicon_entry = self.db_manager.get_lexicon_entry("G26")
        self.assertIsNotNone(lexicon_entry)
        self.assertEqual(lexicon_entry["original_word"], "ἀγάπη")

    def test_parse_stepbible_morphology_greek(self):
        """Test parsing Greek morphology file."""
        morphology_content = """N-NSM\tNoun - Nominative Singular Masculine\tNoun Nominative Singular Masculine
V-PAI-3S\tVerb - Present Active Indicative 3rd Person Singular\tVerb Present Active Indicative 3rd Person Singular
# Comment line
ADJ-NSF\tAdjective - Nominative Singular Feminine\tAdjective Nominative Singular Feminine"""

        morphology_file = self.stepbible_dir / "greek_morphology.txt"
        with open(morphology_file, "w", encoding="utf-8") as f:
            f.write(morphology_content)

        result = self.extractor.parse_stepbible_morphology("greek", self.db_manager)
        self.assertTrue(result)

        morphology_info = self.db_manager.get_morphology_info("N-NSM")
        self.assertIsNotNone(morphology_info)
        self.assertEqual(morphology_info["description"], "Noun - Nominative Singular Masculine")

    def test_parse_stepbible_text_hebrew(self):
        """Test parsing Hebrew TAHOT text file."""
        tahot_content = (
            "TAHOT Gen-Deu - Test data\n\nFIELD DESCRIPTIONS:\nTest header content...\n\n"
            "Gen.1.1#01=L\tבְּ/רֵאשִׁ֖ית\tbe./re.Shit\tin/ beginning\tH9003/{H7225G}\t"
            "HR/Ncfsa\t\t\tH7225G\t\t\tH9003=ב=in/{H7225G=רֵאשִׁית=: beginning»first:1_beginning}\n"
            "Gen.1.1#02=L\tבָּרָ֣א\tba.Ra'\the created\t{H1254A}\tHVqp3ms\t\t\t"
            "H1254A\t\t\t{H1254A=בָּרָא=to create}\n"
            "Gen.1.1#03=L\tאֱלֹהִ֑ים\t'E.lo.Him\tGod\t{H0430G}\tHNcmpa\t\t\t"
            "H0430G\t\t\t{H0430G=אֱלֹהִים=God»LORD@Gen.1.1-Heb}\n"
        )

        tahot_file = self.stepbible_dir / "tahot_gen_deu.txt"
        with open(tahot_file, "w", encoding="utf-8") as f:
            f.write(tahot_content)

        result = self.extractor.parse_stepbible_text("tahot_gen_deu.txt", self.db_manager)
        self.assertTrue(result)

        words = self.db_manager.get_words_for_verse("Gen", 1, 1)
        self.assertTrue(len(words) >= 3)

    def test_parse_stepbible_text_greek(self):
        """Test parsing Greek TAGNT text file."""
        tagnt_content = (
            "TAGNT Mat-Jhn - Test data\n\nFIELD DESCRIPTIONS:\nTest header content...\n\n"
            "Mat.1.1#01=L\tΒίβλος\tBi.blos\tbook\t{G0976}\tGNnms\t\t\tG0976\t\t\t{G0976=βίβλος=book}\n"
            "Mat.1.1#02=L\tγενέσεως\tge.ne.se.os\tgeneration\t{G1078}\tGNgfs\t\t\t"
            "G1078\t\t\t{G1078=γένεσις=generation}\n"
            "Mat.1.1#03=L\tἸησοῦ\tI.e.sou\tJesus\t{G2424}\tGNgms\t\t\tG2424\t\t\t{G2424=Ἰησοῦς=Jesus}\n"
        )

        tagnt_file = self.stepbible_dir / "tagnt_mat_jhn.txt"
        with open(tagnt_file, "w", encoding="utf-8") as f:
            f.write(tagnt_content)

        result = self.extractor.parse_stepbible_text("tagnt_mat_jhn.txt", self.db_manager)
        self.assertTrue(result)

        words = self.db_manager.get_words_for_verse("Mat", 1, 1)
        self.assertTrue(len(words) >= 3)

    def test_parse_nonexistent_lexicon(self):
        """Test parsing non-existent lexicon files."""
        result = self.extractor.parse_lexicon("hebrew", self.db_manager)
        self.assertFalse(result)

    def test_parse_nonexistent_morphology(self):
        """Test parsing non-existent morphology files."""
        result = self.extractor.parse_stepbible_morphology("greek", self.db_manager)
        self.assertFalse(result)

    def test_parse_nonexistent_text(self):
        """Test parsing non-existent text files."""
        result = self.extractor.parse_stepbible_text("nonexistent.txt", self.db_manager)
        self.assertFalse(result)

    def test_parse_lexicon_invalid_language(self):
        """Test parsing lexicon with invalid language."""
        result = self.extractor.parse_lexicon("latin", self.db_manager)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
