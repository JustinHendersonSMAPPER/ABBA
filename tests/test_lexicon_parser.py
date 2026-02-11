"""Tests for the free/open-source lexicon parsers (OpenScriptures Hebrew, Abbott-Smith Greek)."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from abba.lexicon_parser import (
    _clean_text,
    _get_element_text,
    parse_abbott_smith_xml,
    parse_hebrew_strongs_xml,
)


class TestHelperFunctions(unittest.TestCase):
    """Test internal helper functions."""

    def test_clean_text_collapses_whitespace(self):
        self.assertEqual(_clean_text("  hello   world  "), "hello world")

    def test_clean_text_empty(self):
        self.assertEqual(_clean_text(""), "")

    def test_get_element_text_none(self):
        self.assertEqual(_get_element_text(None), "")

    def test_get_element_text_simple(self):
        elem = ET.fromstring("<root>hello</root>")
        self.assertEqual(_get_element_text(elem), "hello")

    def test_get_element_text_nested(self):
        elem = ET.fromstring("<root>a <child>b</child> c</root>")
        result = _get_element_text(elem)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)


class TestHebrewStrongsParser(unittest.TestCase):
    """Test OpenScriptures HebrewStrong.xml parser."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.xml_path = Path(self.temp_dir) / "HebrewStrong.xml"

    def _write_xml(self, content: str):
        with open(self.xml_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_parse_basic_entry(self):
        self._write_xml(
            """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
    <entry id="H1">
        <w pos="n-m" pron="awb" xlit="ab" xml:lang="heb">אָב</w>
        <meaning><def>father</def>, in a literal sense</meaning>
        <usage>chief, father.</usage>
    </entry>
</lexicon>"""
        )
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["strongs_number"], "H1")
        self.assertEqual(entry["original_word"], "אָב")
        self.assertEqual(entry["transliteration"], "ab")
        self.assertEqual(entry["part_of_speech"], "n-m")
        self.assertEqual(entry["gloss"], "father")
        self.assertEqual(entry["language"], "hebrew")
        self.assertIn("father", entry["definition"])

    def test_parse_multiple_entries(self):
        self._write_xml(
            """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
    <entry id="H1">
        <w pos="n-m" pron="awb" xlit="ab" xml:lang="heb">אָב</w>
        <meaning><def>father</def></meaning>
        <usage>father.</usage>
    </entry>
    <entry id="H6">
        <w pos="v" pron="aw-bad" xlit="abad" xml:lang="heb">אָבַד</w>
        <meaning><def>perish</def>, to wander away</meaning>
        <usage>break, destroy.</usage>
    </entry>
</lexicon>"""
        )
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["strongs_number"], "H1")
        self.assertEqual(entries[1]["strongs_number"], "H6")
        self.assertEqual(entries[1]["gloss"], "perish")

    def test_parse_aramaic_as_hebrew(self):
        """Aramaic entries should be stored with language='hebrew'."""
        self._write_xml(
            """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
    <entry id="H2">
        <w pos="n-m" pron="ab" xlit="ab" xml:lang="arc">אַב</w>
        <usage>father.</usage>
    </entry>
</lexicon>"""
        )
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["language"], "hebrew")

    def test_parse_entry_no_meaning(self):
        """Entries without <meaning> should fall back to usage for gloss."""
        self._write_xml(
            """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
    <entry id="H2">
        <w pos="n-m" pron="ab" xlit="ab" xml:lang="arc">אַב</w>
        <usage>father.</usage>
    </entry>
</lexicon>"""
        )
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(entries[0]["gloss"], "father")

    def test_parse_empty_file_returns_empty(self):
        self._write_xml(
            """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
</lexicon>"""
        )
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_parse_malformed_xml_returns_empty(self):
        self._write_xml("this is not xml")
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_parse_nonexistent_file_returns_empty(self):
        entries = parse_hebrew_strongs_xml(Path("/nonexistent/file.xml"))
        self.assertEqual(len(entries), 0)

    def test_transliteration_prefers_xlit(self):
        self._write_xml(
            """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
    <entry id="H1">
        <w pos="n-m" pron="awb" xlit="ab" xml:lang="heb">אָב</w>
        <meaning><def>father</def></meaning>
    </entry>
</lexicon>"""
        )
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(entries[0]["transliteration"], "ab")

    def test_transliteration_falls_back_to_pron(self):
        self._write_xml(
            """<?xml version="1.0" encoding="utf-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
    <entry id="H1">
        <w pos="n-m" pron="awb" xml:lang="heb">אָב</w>
        <meaning><def>father</def></meaning>
    </entry>
</lexicon>"""
        )
        entries = parse_hebrew_strongs_xml(self.xml_path)
        self.assertEqual(entries[0]["transliteration"], "awb")


class TestAbbottSmithParser(unittest.TestCase):
    """Test Abbott-Smith Greek Lexicon parser."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.xml_path = Path(self.temp_dir) / "abbott-smith.xml"

    def _write_xml(self, body_content: str):
        with open(self.xml_path, "w", encoding="utf-8") as f:
            f.write(
                f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.crosswire.org/2013/TEIOSIS/namespace">
 <teiHeader><fileDesc><titleStmt><title>Test</title></titleStmt>
 <publicationStmt><date>1922</date></publicationStmt>
 <sourceDesc><p>test</p></sourceDesc></fileDesc></teiHeader>
 <text><body>{body_content}</body></text>
</TEI>"""
            )

    def test_parse_basic_entry(self):
        self._write_xml(
            """
<entry n="ἀγάπη|G26">
  <form><orth>ἀγάπη</orth></form>
  <sense><gloss>love</gloss>, divine love, charity</sense>
</entry>"""
        )
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["strongs_number"], "G26")
        self.assertEqual(entry["original_word"], "ἀγάπη")
        self.assertEqual(entry["gloss"], "love")
        self.assertEqual(entry["language"], "greek")
        self.assertIn("love", entry["definition"])

    def test_parse_multiple_entries(self):
        self._write_xml(
            """
<entry n="ἀγάπη|G26">
  <form><orth>ἀγάπη</orth></form>
  <sense><gloss>love</gloss></sense>
</entry>
<entry n="πίστις|G4102">
  <form><orth>πίστις</orth></form>
  <sense><gloss>faith</gloss>, trust, belief</sense>
</entry>"""
        )
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["strongs_number"], "G26")
        self.assertEqual(entries[1]["strongs_number"], "G4102")
        self.assertEqual(entries[1]["gloss"], "faith")

    def test_skips_non_g_entries(self):
        """Only entries with G-prefixed Strong's numbers should be included."""
        self._write_xml(
            """
<entry n="word|H123">
  <form><orth>word</orth></form>
  <sense><gloss>test</gloss></sense>
</entry>
<entry n="word|G26">
  <form><orth>ἀγάπη</orth></form>
  <sense><gloss>love</gloss></sense>
</entry>"""
        )
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["strongs_number"], "G26")

    def test_skips_entry_without_n_attribute(self):
        self._write_xml(
            """
<entry>
  <form><orth>ἀγάπη</orth></form>
  <sense><gloss>love</gloss></sense>
</entry>"""
        )
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_parse_entry_without_gloss(self):
        self._write_xml(
            """
<entry n="word|G100">
  <form><orth>τεστ</orth></form>
  <sense>some definition text</sense>
</entry>"""
        )
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["gloss"], "")
        self.assertIn("some definition text", entries[0]["definition"])

    def test_parse_empty_body_returns_empty(self):
        self._write_xml("")
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_parse_malformed_xml_returns_empty(self):
        with open(self.xml_path, "w") as f:
            f.write("this is not xml")
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_parse_nonexistent_file_returns_empty(self):
        entries = parse_abbott_smith_xml(Path("/nonexistent/file.xml"))
        self.assertEqual(len(entries), 0)

    def test_long_definition_is_truncated(self):
        long_sense = "x" * 3000
        self._write_xml(
            f"""
<entry n="word|G100">
  <form><orth>τεστ</orth></form>
  <sense>{long_sense}</sense>
</entry>"""
        )
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertLessEqual(len(entries[0]["definition"]), 2010)

    def test_orth_overrides_n_attribute_word(self):
        self._write_xml(
            """
<entry n="old_form|G26">
  <form><orth>ἀγάπη</orth></form>
  <sense><gloss>love</gloss></sense>
</entry>"""
        )
        entries = parse_abbott_smith_xml(self.xml_path)
        self.assertEqual(entries[0]["original_word"], "ἀγάπη")


if __name__ == "__main__":
    unittest.main()
