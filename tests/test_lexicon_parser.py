"""Unit tests for lexicon_parser module."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from abba import LexiconEntry, LexiconParser


class TestLexiconEntry:
    """Test the LexiconEntry dataclass."""

    def test_creation(self) -> None:
        """Test creating LexiconEntry instances."""
        entry = LexiconEntry(
            strong_number="H1",
            word="אב",
            transliteration="ab",
            pronunciation="awb",
            definition="father",
            gloss="father",
            language="hebrew",
            morphology="n-m",
            etymology="from ancient root",
            usage_notes="commonly used",
        )

        assert entry.strong_number == "H1"
        assert entry.word == "אב"
        assert entry.transliteration == "ab"
        assert entry.pronunciation == "awb"
        assert entry.definition == "father"
        assert entry.gloss == "father"
        assert entry.language == "hebrew"
        assert entry.morphology == "n-m"
        assert entry.etymology == "from ancient root"
        assert entry.usage_notes == "commonly used"

    def test_creation_minimal(self) -> None:
        """Test creating LexiconEntry with minimal data."""
        entry = LexiconEntry(strong_number="H1", word="אב")

        assert entry.strong_number == "H1"
        assert entry.word == "אב"
        assert entry.language == "hebrew"  # Default
        assert entry.transliteration is None
        assert entry.definition is None

    def test_to_dict(self) -> None:
        """Test converting LexiconEntry to dictionary."""
        entry = LexiconEntry(
            strong_number="H1",
            word="אב",
            transliteration="ab",
            definition="father",
            language="hebrew",
        )

        expected = {
            "strong_number": "H1",
            "word": "אב",
            "transliteration": "ab",
            "pronunciation": None,
            "definition": "father",
            "gloss": None,
            "language": "hebrew",
            "morphology": None,
            "etymology": None,
            "usage_notes": None,
        }

        assert entry.to_dict() == expected


class TestLexiconParser:
    """Test the LexiconParser class."""

    def test_initialization(self) -> None:
        """Test parser initialization."""
        parser = LexiconParser()
        assert parser.namespace == {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

    def test_parse_file_not_found(self) -> None:
        """Test parsing non-existent file."""
        parser = LexiconParser()

        try:
            parser.parse_file("nonexistent.xml")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "not found" in str(e)

    def test_parse_file_valid_hebrew(self) -> None:
        """Test parsing valid Hebrew lexicon XML file."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <osisText osisIDWork="Strong" xml:lang="en">
                <div type="entry" n="1">
                    <w gloss="1a" lemma="אָב" morph="n-m" POS="awb" xlit="ab" xml:lang="heb">אב</w>
                    <list>
                        <item>1) father of an individual</item>
                        <item>2) of God as father of his people</item>
                    </list>
                    <note type="explanation">{father}</note>
                    <note type="exegesis">a primitive word</note>
                    <note type="translation">father.</note>
                </div>
                <div type="entry" n="2">
                    <w lemma="אָב" morph="n-m" xml:lang="heb">אב</w>
                    <list>
                        <item>1) father</item>
                    </list>
                    <note type="explanation">{father}</note>
                </div>
            </osisText>
        </osis>
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            parser = LexiconParser()
            entries = parser.parse_file(temp_path, "hebrew")

            assert len(entries) == 2

            # Check first entry
            entry1 = entries[0]
            assert entry1.strong_number == "H1"
            assert entry1.word == "אב"
            assert entry1.transliteration == "ab"
            assert entry1.pronunciation == "awb"
            assert entry1.morphology == "n-m"
            assert "father of an individual" in entry1.definition
            assert "of God as father" in entry1.definition
            assert entry1.gloss == "father"
            assert entry1.etymology == "a primitive word"
            assert entry1.usage_notes == "father."
            assert entry1.language == "hebrew"

            # Check second entry
            entry2 = entries[1]
            assert entry2.strong_number == "H2"
            assert entry2.word == "אב"
            assert entry2.definition == "1) father"
            assert entry2.gloss == "father"

        finally:
            Path(temp_path).unlink()

    def test_parse_file_valid_greek(self) -> None:
        """Test parsing valid Greek lexicon XML file."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <osisText osisIDWork="Strong" xml:lang="en">
                <div type="entry" n="1">
                    <w lemma="Α" morph="letter" POS="al'-fah" xlit="alpha" xml:lang="grc">Α</w>
                    <list>
                        <item>1) first letter of Greek alphabet</item>
                    </list>
                    <note type="explanation">{Alpha}</note>
                </div>
            </osisText>
        </osis>
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            parser = LexiconParser()
            entries = parser.parse_file(temp_path, "greek")

            assert len(entries) == 1

            entry = entries[0]
            assert entry.strong_number == "G1"
            assert entry.word == "Α"
            assert entry.language == "greek"

        finally:
            Path(temp_path).unlink()

    def test_parse_file_malformed_xml(self) -> None:
        """Test parsing malformed XML."""
        xml_content = "<osis><unclosed>"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            parser = LexiconParser()
            parser.parse_file(temp_path)
            assert False, "Should have raised ParseError"
        except ET.ParseError:
            pass
        finally:
            Path(temp_path).unlink()

    def test_parse_entry_element_minimal(self) -> None:
        """Test parsing entry element with minimal data."""
        parser = LexiconParser()

        xml_str = """
        <div type="entry" n="5" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w xml:lang="heb">test</w>
        </div>
        """
        element = ET.fromstring(xml_str)

        entry = parser._parse_entry_element(element, "hebrew")

        assert entry is not None
        assert entry.strong_number == "H5"
        assert entry.word == "test"
        assert entry.language == "hebrew"
        assert entry.definition is None
        assert entry.gloss is None

    def test_parse_entry_element_no_strong_number(self) -> None:
        """Test parsing entry element without Strong's number."""
        parser = LexiconParser()

        xml_str = """
        <div type="entry" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w xml:lang="heb">test</w>
        </div>
        """
        element = ET.fromstring(xml_str)

        entry = parser._parse_entry_element(element, "hebrew")

        assert entry is None

    def test_parse_entry_element_no_word(self) -> None:
        """Test parsing entry element without word element."""
        parser = LexiconParser()

        xml_str = """
        <div type="entry" n="5" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <note>No word element</note>
        </div>
        """
        element = ET.fromstring(xml_str)

        entry = parser._parse_entry_element(element, "hebrew")

        assert entry is None

    def test_parse_hebrew_lexicon(self) -> None:
        """Test parsing Hebrew lexicon from directory."""
        parser = LexiconParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            xml_content = """<?xml version="1.0"?>
            <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
                <osisText>
                    <div type="entry" n="1">
                        <w xml:lang="heb">אב</w>
                    </div>
                </osisText>
            </osis>
            """

            hebrew_file = Path(temp_dir) / "strongs_hebrew.xml"
            hebrew_file.write_text(xml_content, encoding="utf-8")

            entries = parser.parse_hebrew_lexicon(temp_dir)

            assert len(entries) == 1
            assert entries[0].language == "hebrew"
            assert entries[0].strong_number == "H1"

    def test_parse_greek_lexicon(self) -> None:
        """Test parsing Greek lexicon from directory."""
        parser = LexiconParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            xml_content = """<?xml version="1.0"?>
            <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
                <osisText>
                    <div type="entry" n="1">
                        <w xml:lang="grc">α</w>
                    </div>
                </osisText>
            </osis>
            """

            greek_file = Path(temp_dir) / "strongs_greek.xml"
            greek_file.write_text(xml_content, encoding="utf-8")

            entries = parser.parse_greek_lexicon(temp_dir)

            assert len(entries) == 1
            assert entries[0].language == "greek"
            assert entries[0].strong_number == "G1"

    def test_parse_both_lexicons(self) -> None:
        """Test parsing both Hebrew and Greek lexicons."""
        parser = LexiconParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            hebrew_content = """<?xml version="1.0"?>
            <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
                <osisText>
                    <div type="entry" n="1">
                        <w xml:lang="heb">אב</w>
                    </div>
                </osisText>
            </osis>
            """

            greek_content = """<?xml version="1.0"?>
            <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
                <osisText>
                    <div type="entry" n="1">
                        <w xml:lang="grc">α</w>
                    </div>
                </osisText>
            </osis>
            """

            hebrew_file = Path(temp_dir) / "strongs_hebrew.xml"
            hebrew_file.write_text(hebrew_content, encoding="utf-8")

            greek_file = Path(temp_dir) / "strongs_greek.xml"
            greek_file.write_text(greek_content, encoding="utf-8")

            result = parser.parse_both_lexicons(temp_dir)

            assert "hebrew" in result
            assert "greek" in result
            assert len(result["hebrew"]) == 1
            assert len(result["greek"]) == 1
            assert result["hebrew"][0].language == "hebrew"
            assert result["greek"][0].language == "greek"

    def test_get_lexicon_statistics(self) -> None:
        """Test getting lexicon statistics."""
        parser = LexiconParser()

        entries = [
            LexiconEntry(
                strong_number="H1",
                word="אב",
                definition="father",
                gloss="father",
                etymology="primitive",
                language="hebrew",
            ),
            LexiconEntry(
                strong_number="H2",
                word="אם",
                definition="mother",
                language="hebrew",
                # No gloss, no etymology
            ),
            LexiconEntry(
                strong_number="H3",
                word="בן",
                gloss="son",
                language="hebrew",
                # No definition, no etymology
            ),
        ]

        stats = parser.get_lexicon_statistics(entries)

        assert stats["total_entries"] == 3
        assert stats["entries_with_definitions"] == 2
        assert stats["entries_with_glosses"] == 2
        assert stats["entries_with_etymology"] == 1
        assert stats["language"] == "hebrew"

    def test_get_lexicon_statistics_empty(self) -> None:
        """Test statistics for empty entry list."""
        parser = LexiconParser()

        stats = parser.get_lexicon_statistics([])

        assert stats["total_entries"] == 0
        assert stats["entries_with_definitions"] == 0
        assert stats["entries_with_glosses"] == 0
        assert stats["entries_with_etymology"] == 0

    def test_create_strong_lookup(self) -> None:
        """Test creating Strong's number lookup dictionary."""
        parser = LexiconParser()

        entries = [
            LexiconEntry(strong_number="H1", word="אב", language="hebrew"),
            LexiconEntry(strong_number="H2", word="אם", language="hebrew"),
            LexiconEntry(strong_number="G1", word="α", language="greek"),
        ]

        lookup = parser.create_strong_lookup(entries)

        assert len(lookup) == 3
        assert "H1" in lookup
        assert "H2" in lookup
        assert "G1" in lookup
        assert lookup["H1"].word == "אב"
        assert lookup["G1"].word == "α"

    def test_search_by_word(self) -> None:
        """Test searching entries by word."""
        parser = LexiconParser()

        entries = [
            LexiconEntry(
                strong_number="H1",
                word="אב",
                transliteration="ab",
                gloss="father",
                language="hebrew",
            ),
            LexiconEntry(
                strong_number="H2",
                word="אם",
                transliteration="em",
                gloss="mother",
                language="hebrew",
            ),
            LexiconEntry(
                strong_number="H3", word="בן", transliteration="ben", gloss="son", language="hebrew"
            ),
        ]

        # Search by Hebrew word
        results = parser.search_by_word(entries, "אב")
        assert len(results) == 1
        assert results[0].strong_number == "H1"

        # Search by transliteration
        results = parser.search_by_word(entries, "em")
        assert len(results) == 1
        assert results[0].strong_number == "H2"

        # Search by gloss
        results = parser.search_by_word(entries, "son")
        assert len(results) == 1
        assert results[0].strong_number == "H3"

        # Search with no matches
        results = parser.search_by_word(entries, "xyz")
        assert len(results) == 0

        # Case insensitive search
        results = parser.search_by_word(entries, "FATHER")
        assert len(results) == 1
        assert results[0].strong_number == "H1"

    def test_gloss_extraction_with_braces(self) -> None:
        """Test extracting gloss text with curly braces."""
        parser = LexiconParser()

        xml_str = """
        <div type="entry" n="1" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w xml:lang="heb">אב</w>
            <note type="explanation">{father}</note>
        </div>
        """
        element = ET.fromstring(xml_str)

        entry = parser._parse_entry_element(element, "hebrew")

        assert entry is not None
        assert entry.gloss == "father"  # Braces removed

    def test_gloss_extraction_without_braces(self) -> None:
        """Test extracting gloss text without curly braces."""
        parser = LexiconParser()

        xml_str = """
        <div type="entry" n="1" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w xml:lang="heb">אב</w>
            <note type="explanation">father</note>
        </div>
        """
        element = ET.fromstring(xml_str)

        entry = parser._parse_entry_element(element, "hebrew")

        assert entry is not None
        assert entry.gloss == "father"

    def test_multiple_definition_items(self) -> None:
        """Test parsing multiple definition list items."""
        parser = LexiconParser()

        xml_str = """
        <div type="entry" n="1" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w xml:lang="heb">אב</w>
            <list>
                <item>1) father of an individual</item>
                <item>2) of God as father of his people</item>
                <item>3) head or founder of a household</item>
            </list>
        </div>
        """
        element = ET.fromstring(xml_str)

        entry = parser._parse_entry_element(element, "hebrew")

        assert entry is not None
        assert "father of an individual" in entry.definition
        assert "of God as father" in entry.definition
        assert "head or founder" in entry.definition
        # Should be joined with semicolons
        assert entry.definition.count(";") == 2
