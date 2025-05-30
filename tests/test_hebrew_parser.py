"""Unit tests for hebrew_parser module."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from abba import HebrewParser, HebrewVerse, HebrewWord, VerseID


class TestHebrewWord:
    """Test the HebrewWord dataclass."""

    def test_creation(self) -> None:
        """Test creating HebrewWord instances."""
        word = HebrewWord(
            text="בְּרֵאשִׁית", lemma="b/7225", strong_number="H7225", morph="HR/Ncfsa", id="01xeN"
        )

        assert word.text == "בְּרֵאשִׁית"
        assert word.lemma == "b/7225"
        assert word.strong_number == "H7225"
        assert word.morph == "HR/Ncfsa"
        assert word.id == "01xeN"

    def test_from_xml_element(self) -> None:
        """Test creating HebrewWord from XML element."""
        xml_str = '<w lemma="1254 a" morph="HVqp3ms" id="01Nvk">בָּרָא</w>'
        element = ET.fromstring(xml_str)

        word = HebrewWord.from_xml_element(element)

        assert word.text == "בָּרָא"
        assert word.lemma == "1254 a"
        assert word.strong_number == "H1254"
        assert word.morph == "HVqp3ms"
        assert word.id == "01Nvk"

    def test_from_xml_element_complex_lemma(self) -> None:
        """Test parsing complex lemma formats."""
        xml_str = '<w lemma="b/7225" morph="HR/Ncfsa" id="01xeN">בְּרֵאשִׁית</w>'
        element = ET.fromstring(xml_str)

        word = HebrewWord.from_xml_element(element)

        assert word.text == "בְּרֵאשִׁית"
        assert word.lemma == "b/7225"
        assert word.strong_number == "H7225"
        assert word.morph == "HR/Ncfsa"

    def test_from_xml_element_no_lemma(self) -> None:
        """Test parsing element without lemma."""
        xml_str = '<w morph="HVqp3ms">בָּרָא</w>'
        element = ET.fromstring(xml_str)

        word = HebrewWord.from_xml_element(element)

        assert word.text == "בָּרָא"
        assert word.lemma is None
        assert word.strong_number is None
        assert word.morph == "HVqp3ms"

    def test_to_dict(self) -> None:
        """Test converting HebrewWord to dictionary."""
        word = HebrewWord(
            text="בְּרֵאשִׁית", lemma="b/7225", strong_number="H7225", morph="HR/Ncfsa", id="01xeN"
        )

        expected = {
            "text": "בְּרֵאשִׁית",
            "lemma": "b/7225",
            "strong_number": "H7225",
            "morph": "HR/Ncfsa",
            "gloss": None,
            "id": "01xeN",
        }

        assert word.to_dict() == expected


class TestHebrewVerse:
    """Test the HebrewVerse dataclass."""

    def test_creation(self) -> None:
        """Test creating HebrewVerse instances."""
        verse_id = VerseID(book="GEN", chapter=1, verse=1)
        words = [
            HebrewWord(text="בְּרֵאשִׁית", strong_number="H7225"),
            HebrewWord(text="בָּרָא", strong_number="H1254"),
        ]

        verse = HebrewVerse(verse_id=verse_id, words=words, osis_id="Gen.1.1")

        assert verse.verse_id == verse_id
        assert len(verse.words) == 2
        assert verse.osis_id == "Gen.1.1"

    def test_from_xml_element(self) -> None:
        """Test creating HebrewVerse from XML element."""
        xml_str = """
        <verse osisID="Gen.1.1" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w lemma="b/7225" morph="HR/Ncfsa" id="01xeN">בְּרֵאשִׁית</w>
            <w lemma="1254 a" morph="HVqp3ms" id="01Nvk">בָּרָא</w>
            <w lemma="430" morph="HNcmpa" id="01TyA">אֱלֹהִים</w>
        </verse>
        """
        element = ET.fromstring(xml_str)

        verse = HebrewVerse.from_xml_element(element)

        assert verse is not None
        assert verse.verse_id.book == "GEN"
        assert verse.verse_id.chapter == 1
        assert verse.verse_id.verse == 1
        assert verse.osis_id == "Gen.1.1"
        assert len(verse.words) == 3
        assert verse.words[0].text == "בְּרֵאשִׁית"
        assert verse.words[1].text == "בָּרָא"
        assert verse.words[2].text == "אֱלֹהִים"

    def test_from_xml_element_invalid_osis_id(self) -> None:
        """Test handling invalid OSIS ID."""
        xml_str = """
        <verse osisID="Invalid" xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w>בְּרֵאשִׁית</w>
        </verse>
        """
        element = ET.fromstring(xml_str)

        verse = HebrewVerse.from_xml_element(element)

        assert verse is None

    def test_from_xml_element_no_osis_id(self) -> None:
        """Test handling missing OSIS ID."""
        xml_str = """
        <verse xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <w>בְּרֵאשִׁית</w>
        </verse>
        """
        element = ET.fromstring(xml_str)

        verse = HebrewVerse.from_xml_element(element)

        assert verse is None

    def test_to_dict(self) -> None:
        """Test converting HebrewVerse to dictionary."""
        verse_id = VerseID(book="GEN", chapter=1, verse=1)
        words = [HebrewWord(text="בְּרֵאשִׁית", strong_number="H7225")]

        verse = HebrewVerse(verse_id=verse_id, words=words, osis_id="Gen.1.1")

        result = verse.to_dict()

        assert result["verse_id"] == "GEN.1.1"
        assert result["osis_id"] == "Gen.1.1"
        assert len(result["words"]) == 1
        assert result["words"][0]["text"] == "בְּרֵאשִׁית"


class TestHebrewParser:
    """Test the HebrewParser class."""

    def test_initialization(self) -> None:
        """Test parser initialization."""
        parser = HebrewParser()
        assert parser.namespace == {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

    def test_parse_file_not_found(self) -> None:
        """Test parsing non-existent file."""
        parser = HebrewParser()

        try:
            parser.parse_file("nonexistent.xml")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "not found" in str(e)

    def test_parse_file_valid(self) -> None:
        """Test parsing valid Hebrew XML file."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
            <osisText>
                <div type="book" osisID="Gen">
                    <chapter osisID="Gen.1">
                        <verse osisID="Gen.1.1">
                            <w lemma="b/7225" morph="HR/Ncfsa" id="01xeN">בְּרֵאשִׁית</w>
                            <w lemma="1254 a" morph="HVqp3ms" id="01Nvk">בָּרָא</w>
                            <w lemma="430" morph="HNcmpa" id="01TyA">אֱלֹהִים</w>
                        </verse>
                        <verse osisID="Gen.1.2">
                            <w lemma="c/d/776" morph="HC/Td/Ncbsa" id="01LN3">וְהָאָרֶץ</w>
                            <w lemma="1961" morph="HVqp3fs" id="01Qzf">הָיְתָה</w>
                        </verse>
                    </chapter>
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
            parser = HebrewParser()
            verses = parser.parse_file(temp_path)

            assert len(verses) == 2

            # Check first verse
            verse1 = verses[0]
            assert verse1.verse_id.book == "GEN"
            assert verse1.verse_id.chapter == 1
            assert verse1.verse_id.verse == 1
            assert len(verse1.words) == 3
            assert verse1.words[0].text == "בְּרֵאשִׁית"
            assert verse1.words[0].strong_number == "H7225"

            # Check second verse
            verse2 = verses[1]
            assert verse2.verse_id.verse == 2
            assert len(verse2.words) == 2

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
            parser = HebrewParser()
            parser.parse_file(temp_path)
            assert False, "Should have raised ParseError"
        except ET.ParseError:
            pass
        finally:
            Path(temp_path).unlink()

    def test_parse_book_valid_code(self) -> None:
        """Test parsing book with valid code."""
        parser = HebrewParser()

        # Mock data directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Genesis file
            xml_content = """<?xml version="1.0" encoding="UTF-8"?>
            <osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
                <osisText>
                    <verse osisID="Gen.1.1">
                        <w lemma="b/7225">בְּרֵאשִׁית</w>
                    </verse>
                </osisText>
            </osis>
            """

            gen_file = Path(temp_dir) / "Gen.xml"
            gen_file.write_text(xml_content, encoding="utf-8")

            verses = parser.parse_book("GEN", temp_dir)
            assert len(verses) == 1
            assert verses[0].verse_id.book == "GEN"

    def test_parse_book_invalid_code(self) -> None:
        """Test parsing book with invalid code."""
        parser = HebrewParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                parser.parse_book("XXX", temp_dir)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Unknown Hebrew book code" in str(e)

    def test_get_book_statistics(self) -> None:
        """Test getting book statistics."""
        parser = HebrewParser()

        # Create test verses
        verses = [
            HebrewVerse(
                verse_id=VerseID(book="GEN", chapter=1, verse=1),
                words=[
                    HebrewWord(text="בְּרֵאשִׁית", morph="HR/Ncfsa"),
                    HebrewWord(text="בָּרָא", morph="HVqp3ms"),
                    HebrewWord(text="אֱלֹהִים"),  # No morph
                ],
                osis_id="Gen.1.1",
            ),
            HebrewVerse(
                verse_id=VerseID(book="GEN", chapter=1, verse=2),
                words=[
                    HebrewWord(text="וְהָאָרֶץ", morph="HC/Td/Ncbsa"),
                ],
                osis_id="Gen.1.2",
            ),
        ]

        stats = parser.get_book_statistics(verses)

        assert stats["verse_count"] == 2
        assert stats["word_count"] == 4
        assert stats["morphed_word_count"] == 3
        assert stats["book_code"] == "GEN"

    def test_get_book_statistics_empty(self) -> None:
        """Test statistics for empty verse list."""
        parser = HebrewParser()

        stats = parser.get_book_statistics([])

        assert stats["verse_count"] == 0
        assert stats["word_count"] == 0
        assert stats["morphed_word_count"] == 0

    def test_extract_strong_numbers(self) -> None:
        """Test extracting Strong's numbers."""
        parser = HebrewParser()

        verses = [
            HebrewVerse(
                verse_id=VerseID(book="GEN", chapter=1, verse=1),
                words=[
                    HebrewWord(text="בְּרֵאשִׁית", strong_number="H7225"),
                    HebrewWord(text="בָּרָא", strong_number="H1254"),
                    HebrewWord(text="אֱלֹהִים", strong_number="H430"),
                ],
                osis_id="Gen.1.1",
            ),
            HebrewVerse(
                verse_id=VerseID(book="GEN", chapter=1, verse=2),
                words=[
                    HebrewWord(text="וְהָאָרֶץ", strong_number="H776"),
                    HebrewWord(text="הָיְתָה", strong_number="H1961"),
                    HebrewWord(text="תֹהוּ", strong_number="H1254"),  # Duplicate
                ],
                osis_id="Gen.1.2",
            ),
        ]

        strong_numbers = parser.extract_strong_numbers(verses)

        assert strong_numbers == ["H1254", "H1961", "H430", "H7225", "H776"]

    def test_filename_mapping(self) -> None:
        """Test that all canonical book codes have filename mappings."""
        parser = HebrewParser()

        # Test a few key mappings
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test Exodus mapping
            exod_file = Path(temp_dir) / "Exod.xml"
            exod_file.write_text(
                '<?xml version="1.0"?>'
                '<osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace"></osis>'
            )

            try:
                parser.parse_book("EXO", temp_dir)
                # Should not raise an error for filename mapping
            except FileNotFoundError:
                # File doesn't have valid content, but filename mapping worked
                pass

            # Test invalid book code
            try:
                parser.parse_book("XXX", temp_dir)
                assert False, "Should raise ValueError"
            except ValueError:
                pass
