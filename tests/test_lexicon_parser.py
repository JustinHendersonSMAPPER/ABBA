"""Tests for the free/open-source lexicon parsers.

Covers: OpenScriptures Hebrew, Abbott-Smith Greek, BDB Hebrew, Dodson Greek, Strong's Greek.
"""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from abba.lexicon_parser import (
    _clean_text,
    _get_element_text,
    _parse_tflsj_line,
    parse_abbott_smith_xml,
    parse_bdb_xml,
    parse_dodson_csv,
    parse_hebrew_strongs_xml,
    parse_strongs_greek_xml,
    parse_tflsj_txt,
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


class TestBDBParser(unittest.TestCase):
    """Test Brown-Driver-Briggs Hebrew Lexicon XML parser."""

    NS = "http://openscriptures.github.com/morphhb/namespace"

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bdb_path = Path(self.temp_dir) / "BrownDriverBriggs.xml"
        self.index_path = Path(self.temp_dir) / "LexicalIndex.xml"

    def _write_bdb(self, content: str):
        with open(self.bdb_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_index(self, content: str):
        with open(self.index_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _make_bdb_xml(self, entries_xml: str) -> str:
        return f'<?xml version="1.0" encoding="utf-8"?>' f'<lexicon xmlns="{self.NS}">{entries_xml}</lexicon>'

    def _make_index_xml(self, entries_xml: str) -> str:
        return (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f'<index xmlns="{self.NS}"><part xml:lang="heb">{entries_xml}</part></index>'
        )

    def test_parse_basic_entry(self):
        self._write_bdb(
            self._make_bdb_xml(
                """
                <part>
                    <section>
                        <entry id="a.ac.aa">
                            <w>אָבַד</w>
                            <pos>vb</pos>
                            <def>perish</def>
                        </entry>
                    </section>
                </part>
                """
            )
        )
        self._write_index(
            self._make_index_xml(
                """
                <entry id="aaf">
                    <w xlit="abad">אָבַד</w>
                    <pos>V</pos>
                    <def>perish</def>
                    <xref bdb="a.ac.aa" strong="6" twot="2"/>
                </entry>
                """
            )
        )
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["strongs_number"], "H0006")
        self.assertEqual(entry["source_lexicon"], "bdb")
        self.assertEqual(entry["original_word"], "אָבַד")
        self.assertEqual(entry["part_of_speech"], "vb")
        self.assertEqual(entry["gloss"], "perish")
        self.assertEqual(entry["language"], "hebrew")
        self.assertIn("perish", entry["definition"])

    def test_parse_multiple_strongs_for_one_bdb(self):
        """Multiple Strong's numbers can map to the same BDB entry."""
        self._write_bdb(
            self._make_bdb_xml(
                """
                <part>
                    <section>
                        <entry id="a.ae.ab">
                            <w>אָב</w>
                            <pos>n.m</pos>
                            <def>father</def>
                        </entry>
                    </section>
                </part>
                """
            )
        )
        self._write_index(
            self._make_index_xml(
                """
                <entry id="aac">
                    <w xlit="ab">אָב</w>
                    <xref bdb="a.ae.ab" strong="1" twot="4a"/>
                </entry>
                <entry id="aad">
                    <w xlit="ab">אַב</w>
                    <xref bdb="a.ae.ab" strong="2" twot="4a"/>
                </entry>
                """
            )
        )
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 2)
        strongs = {e["strongs_number"] for e in entries}
        self.assertEqual(strongs, {"H0001", "H0002"})

    def test_parse_entry_with_senses(self):
        self._write_bdb(
            self._make_bdb_xml(
                """
                <part>
                    <section>
                        <entry id="a.am.aa">
                            <w>אֶבֶן</w>
                            <pos>n.f</pos>
                            <def>stone</def>
                            <sense n="1"><def>in natural state</def></sense>
                            <sense n="2"><def>as material</def></sense>
                        </entry>
                    </section>
                </part>
                """
            )
        )
        self._write_index(
            self._make_index_xml(
                """
                <entry id="x">
                    <w xlit="eben">אֶבֶן</w>
                    <xref bdb="a.am.aa" strong="68" twot="9"/>
                </entry>
                """
            )
        )
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 1)
        self.assertIn("1.", entries[0]["definition"])
        self.assertIn("2.", entries[0]["definition"])

    def test_empty_bdb_returns_empty(self):
        self._write_bdb(self._make_bdb_xml(""))
        self._write_index(self._make_index_xml('<entry id="x"><w>w</w><xref bdb="missing" strong="1"/></entry>'))
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 0)

    def test_empty_index_returns_empty(self):
        self._write_bdb(
            self._make_bdb_xml('<part><section><entry id="a.a.a"><w>x</w><def>test</def></entry></section></part>')
        )
        self._write_index(self._make_index_xml(""))
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 0)

    def test_malformed_bdb_returns_empty(self):
        with open(self.bdb_path, "w") as f:
            f.write("not xml")
        self._write_index(self._make_index_xml(""))
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 0)

    def test_malformed_index_returns_empty(self):
        self._write_bdb(self._make_bdb_xml(""))
        with open(self.index_path, "w") as f:
            f.write("not xml")
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 0)

    def test_nonexistent_files_return_empty(self):
        entries = parse_bdb_xml(Path("/nonexistent/bdb.xml"), Path("/nonexistent/index.xml"))
        self.assertEqual(len(entries), 0)

    def test_long_definition_is_truncated(self):
        long_text = "x" * 3000
        self._write_bdb(
            self._make_bdb_xml(
                f"""
                <part><section>
                    <entry id="a.a.a">
                        <w>test</w>
                        <def>{long_text}</def>
                    </entry>
                </section></part>
                """
            )
        )
        self._write_index(self._make_index_xml('<entry id="x"><w>test</w><xref bdb="a.a.a" strong="1"/></entry>'))
        entries = parse_bdb_xml(self.bdb_path, self.index_path)
        self.assertEqual(len(entries), 1)
        self.assertLessEqual(len(entries[0]["definition"]), 2010)


class TestDodsonParser(unittest.TestCase):
    """Test Dodson Greek-English Lexicon CSV parser."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "dodson.csv"

    def _write_csv(self, content: str):
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_parse_basic_entry(self):
        self._write_csv('G0026,ἀγάπη,"agape",N:F,love,"love, benevolence, good will"\n')
        entries = parse_dodson_csv(self.csv_path)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["strongs_number"], "G0026")
        self.assertEqual(entry["source_lexicon"], "dodson")
        self.assertEqual(entry["original_word"], "ἀγάπη")
        self.assertEqual(entry["transliteration"], "agape")
        self.assertEqual(entry["part_of_speech"], "N:F")
        self.assertEqual(entry["gloss"], "love")
        self.assertIn("benevolence", entry["definition"])
        self.assertEqual(entry["language"], "greek")

    def test_parse_multiple_entries(self):
        self._write_csv(
            'G0026,ἀγάπη,"agape",N:F,love,"love, benevolence"\n' 'G4102,πίστις,"pistis",N:F,faith,"faith, trust"\n'
        )
        entries = parse_dodson_csv(self.csv_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["strongs_number"], "G0026")
        self.assertEqual(entries[1]["strongs_number"], "G4102")

    def test_skips_non_g_entries(self):
        self._write_csv("H0001,test,test,N,test,test\n" 'G0026,ἀγάπη,"agape",N:F,love,love\n')
        entries = parse_dodson_csv(self.csv_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["strongs_number"], "G0026")

    def test_skips_short_rows(self):
        self._write_csv("G0026,ἀγάπη,agape\n")
        entries = parse_dodson_csv(self.csv_path)
        self.assertEqual(len(entries), 0)

    def test_definition_falls_back_to_gloss(self):
        """When no definition column, gloss should be used."""
        self._write_csv("G0026,ἀγάπη,agape,N:F,love\n")
        entries = parse_dodson_csv(self.csv_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["definition"], "love")

    def test_empty_file_returns_empty(self):
        self._write_csv("")
        entries = parse_dodson_csv(self.csv_path)
        self.assertEqual(len(entries), 0)

    def test_nonexistent_file_returns_empty(self):
        entries = parse_dodson_csv(Path("/nonexistent/dodson.csv"))
        self.assertEqual(len(entries), 0)


class TestStrongsGreekParser(unittest.TestCase):
    """Test Strong's Greek Dictionary XML parser (morphgnt CC0)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.xml_path = Path(self.temp_dir) / "strongsgreek.xml"

    def _write_xml(self, entries_xml: str):
        with open(self.xml_path, "w", encoding="utf-8") as f:
            f.write(
                f"<?xml version='1.0' encoding='utf-8'?>"
                f"<strongsdictionary><prologue>test</prologue>"
                f"<entries>{entries_xml}</entries></strongsdictionary>"
            )

    def test_parse_basic_entry(self):
        self._write_xml(
            """
            <entry strongs="00026">
                <strongs>26</strongs>
                <greek BETA="A)GA/PH" unicode="ἀγάπη" translit="agápē"/>
                <pronunciation strongs="ag-ah'-pay"/>
                <strongs_derivation>from <strongsref language="GREEK" strongs="0025"/>;</strongs_derivation>
                <strongs_def>love, i.e. affection or benevolence</strongs_def>
                <kjv_def>--love, charity.</kjv_def>
            </entry>
            """
        )
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["strongs_number"], "G0026")
        self.assertEqual(entry["source_lexicon"], "strongs_greek")
        self.assertEqual(entry["original_word"], "ἀγάπη")
        self.assertEqual(entry["transliteration"], "agápē")
        self.assertEqual(entry["language"], "greek")
        self.assertIn("love", entry["gloss"])
        self.assertIn("love", entry["definition"])
        self.assertIn("affection", entry["definition"])

    def test_parse_multiple_entries(self):
        self._write_xml(
            """
            <entry strongs="00026">
                <strongs>26</strongs>
                <greek BETA="A" unicode="ἀγάπη" translit="agápē"/>
                <strongs_def>love</strongs_def>
                <kjv_def>--love.</kjv_def>
            </entry>
            <entry strongs="04102">
                <strongs>4102</strongs>
                <greek BETA="B" unicode="πίστις" translit="pístis"/>
                <strongs_def>persuasion, credence, faith</strongs_def>
                <kjv_def>--faith, belief.</kjv_def>
            </entry>
            """
        )
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["strongs_number"], "G0026")
        self.assertEqual(entries[1]["strongs_number"], "G4102")

    def test_includes_derivation_in_definition(self):
        self._write_xml(
            """
            <entry strongs="00002">
                <strongs>2</strongs>
                <greek BETA="A" unicode="Ἀαρών" translit="Aarṓn"/>
                <strongs_derivation>of Hebrew origin</strongs_derivation>
                <strongs_def>Aaron, the brother of Moses</strongs_def>
                <kjv_def>--Aaron.</kjv_def>
            </entry>
            """
        )
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertIn("Origin: of Hebrew origin", entries[0]["definition"])
        self.assertIn("Aaron", entries[0]["definition"])

    def test_skips_entry_without_strongs(self):
        self._write_xml("<entry><strongs>bad</strongs></entry>")
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_skips_empty_definition_entries(self):
        self._write_xml(
            """
            <entry strongs="99999">
                <strongs>99999</strongs>
                <greek BETA="X" unicode="" translit=""/>
            </entry>
            """
        )
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_gloss_strips_dashes_and_takes_first_word(self):
        self._write_xml(
            """
            <entry strongs="00018">
                <strongs>18</strongs>
                <greek BETA="A" unicode="ἀγαθός" translit="agathós"/>
                <strongs_def>a good thing</strongs_def>
                <kjv_def>--benefit, good(-s, things), well.</kjv_def>
            </entry>
            """
        )
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["gloss"], "benefit")

    def test_empty_file_returns_empty(self):
        self._write_xml("")
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_malformed_xml_returns_empty(self):
        with open(self.xml_path, "w") as f:
            f.write("not xml")
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 0)

    def test_nonexistent_file_returns_empty(self):
        entries = parse_strongs_greek_xml(Path("/nonexistent/strongs.xml"))
        self.assertEqual(len(entries), 0)

    def test_long_definition_is_truncated(self):
        long_text = "x" * 3000
        self._write_xml(
            f"""
            <entry strongs="00001">
                <strongs>1</strongs>
                <greek BETA="A" unicode="Α" translit="A"/>
                <strongs_def>{long_text}</strongs_def>
                <kjv_def>--Alpha.</kjv_def>
            </entry>
            """
        )
        entries = parse_strongs_greek_xml(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertLessEqual(len(entries[0]["definition"]), 2010)


class TestTFLSJParser(unittest.TestCase):
    """Test STEPBible TFLSJ (Tyndale Full LSJ Gloss) parser."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.txt_path = Path(self.temp_dir) / "tflsj.txt"

    def _write_txt(self, content: str):
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_parse_basic_entry(self):
        self._write_txt("G0026\tἀγάπη\tagape\tlove\tLove, goodwill, benevolence\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["strongs_number"], "G0026")
        self.assertEqual(entries[0]["original_word"], "ἀγάπη")
        self.assertEqual(entries[0]["transliteration"], "agape")
        self.assertEqual(entries[0]["gloss"], "love")
        self.assertEqual(entries[0]["definition"], "Love, goodwill, benevolence")
        self.assertEqual(entries[0]["source_lexicon"], "tflsj")
        self.assertEqual(entries[0]["language"], "greek")

    def test_skip_comment_lines(self):
        self._write_txt("# This is a comment\n$ Header line\nG0026\tἀγάπη\tagape\tlove\tDefinition\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 1)

    def test_skip_hebrew_entries(self):
        self._write_txt("H0001\tאָב\tab\tfather\tFather definition\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 0)

    def test_skip_empty_lines(self):
        self._write_txt("\n\n\nG0026\tἀγάπη\tagape\tlove\tDefinition\n\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 1)

    def test_multiple_entries(self):
        self._write_txt(
            "G0026\tἀγάπη\tagape\tlove\tLove definition\n"
            "G0032\tἄγγελος\tangelos\tmessenger\tMessenger definition\n"
            "G0040\tἅγιος\thagios\tholy\tSacred, holy\n"
        )
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 3)

    def test_numeric_only_strongs(self):
        self._write_txt("26\tἀγάπη\tagape\tlove\tDefinition\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["strongs_number"], "G0026")

    def test_entry_with_no_definition(self):
        self._write_txt("G0026\tἀγάπη\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 0)

    def test_gloss_used_as_definition_fallback(self):
        self._write_txt("G0026\tἀγάπη\tagape\tlove\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["definition"], "love")

    def test_long_definition_truncation(self):
        long_def = "a" * 4000
        self._write_txt(f"G0026\tἀγάπη\tagape\tlove\t{long_def}\n")
        entries = parse_tflsj_txt(self.txt_path)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["definition"].endswith("..."))
        self.assertLessEqual(len(entries[0]["definition"]), 3004)

    def test_missing_file(self):
        entries = parse_tflsj_txt(Path("/nonexistent/file.txt"))
        self.assertEqual(len(entries), 0)

    def test_parse_tflsj_line_none_for_empty(self):
        self.assertIsNone(_parse_tflsj_line(""))
        self.assertIsNone(_parse_tflsj_line("   "))
        self.assertIsNone(_parse_tflsj_line("# comment"))
        self.assertIsNone(_parse_tflsj_line("$ header"))

    def test_parse_tflsj_line_valid(self):
        entry = _parse_tflsj_line("G0026\tἀγάπη\tagape\tlove\tFull definition")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["strongs_number"], "G0026")


if __name__ == "__main__":
    unittest.main()
