"""Tests for STEPBible data parsing functionality."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_parse_stepbible_lexicon_hebrew(self):
        """Test parsing Hebrew lexicon file."""
        # Create test lexicon file
        lexicon_content = """H0001\tאָב\t'ab\tnoun\tfather\tThe male parent or ancestor
H0002\tאָבַד\t'abad\tverb\tperish\tTo be lost or destroyed
# Comment line
H0003\tאֵבֶל\t'ebel\tnoun\tmourning\tLamentation for the dead"""

        lexicon_file = self.stepbible_dir / "hebrew_lexicon.txt"
        with open(lexicon_file, "w", encoding="utf-8") as f:
            f.write(lexicon_content)

        # Test parsing
        result = self.extractor.parse_stepbible_lexicon("hebrew", self.db_manager)
        self.assertTrue(result)

        # Verify data was inserted
        lexicon_entry = self.db_manager.get_lexicon_entry("H0001")
        self.assertIsNotNone(lexicon_entry)
        self.assertEqual(lexicon_entry["original_word"], "אָב")
        self.assertEqual(lexicon_entry["transliteration"], "'ab")

    def test_parse_stepbible_morphology_greek(self):
        """Test parsing Greek morphology file."""
        # Create test morphology file
        morphology_content = """N-NSM\tNoun - Nominative Singular Masculine\tNoun Nominative Singular Masculine
V-PAI-3S\tVerb - Present Active Indicative 3rd Person Singular\tVerb Present Active Indicative 3rd Person Singular
# Comment line
ADJ-NSF\tAdjective - Nominative Singular Feminine\tAdjective Nominative Singular Feminine"""

        morphology_file = self.stepbible_dir / "greek_morphology.txt"
        with open(morphology_file, "w", encoding="utf-8") as f:
            f.write(morphology_content)

        # Test parsing
        result = self.extractor.parse_stepbible_morphology("greek", self.db_manager)
        self.assertTrue(result)

        # Verify data was inserted
        morphology_info = self.db_manager.get_morphology_info("N-NSM")
        self.assertIsNotNone(morphology_info)
        self.assertEqual(morphology_info["description"], "Noun - Nominative Singular Masculine")

    def test_parse_stepbible_text_hebrew(self):
        """Test parsing Hebrew TAHOT text file."""
        # Create test TAHOT file content (real STEPBible format)
        tahot_content = """TAHOT Gen-Deu - Test data

FIELD DESCRIPTIONS:
Test header content...

Gen.1.1#01=L	בְּ/רֵאשִׁ֖ית	be./re.Shit	in/ beginning	H9003/{H7225G}	HR/Ncfsa			H7225G			H9003=ב=in/{H7225G=רֵאשִׁית=: beginning»first:1_beginning}
Gen.1.1#02=L	בָּרָ֣א	ba.Ra'	he created	{H1254A}	HVqp3ms			H1254A			{H1254A=בָּרָא=to create}
Gen.1.1#03=L	אֱלֹהִ֑ים	'E.lo.Him	God	{H0430G}	HNcmpa			H0430G			{H0430G=אֱלֹהִים=God»LORD@Gen.1.1-Heb}
# Comment line
Gen.1.2#01=L	וְ/הָ/אָ֗רֶץ	ve./ha./'A.retz	and/ the/ earth	H9002/H9009/{H0776G}	HC/Td/Ncfsa			H0776G			H9002=ו=and/H9009=ה=the/{H0776G=אֶ֫רֶץ=: country;_planet»land:2_country;_planet}"""

        tahot_file = self.stepbible_dir / "tahot_gen_deu.txt"
        with open(tahot_file, "w", encoding="utf-8") as f:
            f.write(tahot_content)

        # Test parsing
        result = self.extractor.parse_stepbible_text("tahot_gen_deu.txt", self.db_manager)
        self.assertTrue(result)

        # Verify words were inserted
        words = self.db_manager.get_words_for_verse("Gen", 1, 1)
        self.assertTrue(len(words) >= 3)

        # Check specific word (cleaned Hebrew text)
        first_word = words[0]
        self.assertEqual(first_word["hebrew_text"], "בְּרֵאשִׁ֖ית")  # Should have cleaning applied
        self.assertEqual(first_word["strongs_primary"], "H7225G")
        self.assertEqual(first_word["translation"], "in/ beginning")

    def test_parse_stepbible_text_greek(self):
        """Test parsing Greek TAGNT text file."""
        # Create test TAGNT file content (real STEPBible format)
        tagnt_content = """TAGNT Mat-Jhn - Test data

FIELD DESCRIPTIONS:
Test header content...

Mat.1.1#01=L	Βίβλος	Bi.blos	book	{G0976}	GNnms			G0976			{G0976=βίβλος=book}
Mat.1.1#02=L	γενέσεως	ge.ne.se.os	generation	{G1078}	GNgfs			G1078			{G1078=γένεσις=generation}
Mat.1.1#03=L	Ἰησοῦ	I.e.sou	Jesus	{G2424}	GNgms			G2424			{G2424=Ἰησοῦς=Jesus}
# Comment line
Mat.1.2#01=L	Ἀβραὰμ	A.bra.am	Abraham	{G0011}	GNams			G0011			{G0011=Ἀβραάμ=Abraham}"""

        tagnt_file = self.stepbible_dir / "tagnt_mat_jhn.txt"
        with open(tagnt_file, "w", encoding="utf-8") as f:
            f.write(tagnt_content)

        # Test parsing
        result = self.extractor.parse_stepbible_text("tagnt_mat_jhn.txt", self.db_manager)
        self.assertTrue(result)

        # Verify words were inserted
        words = self.db_manager.get_words_for_verse("Mat", 1, 1)
        self.assertTrue(len(words) >= 3)

        # Check specific word
        first_word = words[0]
        self.assertEqual(first_word["greek_text"], "Βίβλος")
        self.assertEqual(first_word["strongs_primary"], "G0976")
        self.assertEqual(first_word["translation"], "book")

    def test_import_stepbible_data_success(self):
        """Test successful import of all STEPBible data."""
        # Create minimal test files
        self._create_test_files()

        # Test import
        result = self.extractor.import_stepbible_data(self.db_manager)
        self.assertTrue(result)

        # Verify data exists
        stats = self.db_manager.get_database_stats()
        self.assertGreater(stats["words"], 0)
        self.assertGreater(stats["lexicon"], 0)
        self.assertGreater(stats["morphology"], 0)

    def test_import_stepbible_data_no_directory(self):
        """Test import when STEPBible directory doesn't exist."""
        # Remove stepbible directory
        import shutil

        shutil.rmtree(self.stepbible_dir)

        result = self.extractor.import_stepbible_data(self.db_manager)
        self.assertFalse(result)

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent files."""
        result = self.extractor.parse_stepbible_lexicon("hebrew", self.db_manager)
        self.assertFalse(result)

        result = self.extractor.parse_stepbible_morphology("greek", self.db_manager)
        self.assertFalse(result)

        result = self.extractor.parse_stepbible_text("nonexistent.txt", self.db_manager)
        self.assertFalse(result)

    def _create_test_files(self):
        """Create minimal test files for successful import."""
        # Hebrew lexicon
        hebrew_lexicon = "H0001\tאָב\t'ab\tnoun\tfather\tThe male parent"
        with open(self.stepbible_dir / "hebrew_lexicon.txt", "w", encoding="utf-8") as f:
            f.write(hebrew_lexicon)

        # Greek lexicon
        greek_lexicon = "G0001\tἀ\ta\tparticle\tnot\tNegative particle"
        with open(self.stepbible_dir / "greek_lexicon.txt", "w", encoding="utf-8") as f:
            f.write(greek_lexicon)

        # Hebrew morphology
        hebrew_morph = "N-ms\tNoun masculine singular\tNoun masculine singular"
        with open(self.stepbible_dir / "hebrew_morphology.txt", "w", encoding="utf-8") as f:
            f.write(hebrew_morph)

        # Greek morphology
        greek_morph = "N-NSM\tNoun Nominative Singular Masculine\tNoun Nominative Singular Masculine"
        with open(self.stepbible_dir / "greek_morphology.txt", "w", encoding="utf-8") as f:
            f.write(greek_morph)

        # Hebrew text (real format)
        hebrew_text = "Gen.1.1#01=L\tבְּרֵאשִׁ֖ית\tbe.Shit\tbeginning\t{H7225G}\tNcfsa\t\t\tH7225G\t\t"
        with open(self.stepbible_dir / "tahot_gen_deu.txt", "w", encoding="utf-8") as f:
            f.write(hebrew_text)

        # Greek text (real format)
        greek_text = "Mat.1.1#01=L\tΒίβλος\tBi.blos\tbook\t{G0976}\tGNnms\t\t\tG0976\t\t"
        with open(self.stepbible_dir / "tagnt_mat_jhn.txt", "w", encoding="utf-8") as f:
            f.write(greek_text)


if __name__ == "__main__":
    unittest.main()
