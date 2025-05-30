"""Unit tests for greek_parser module."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from abba import GreekParser, GreekVerse, GreekWord, VerseID


class TestGreekWord:
    """Test the GreekWord dataclass."""

    def test_creation(self) -> None:
        """Test creating GreekWord instances."""
        word = GreekWord(text="βιβλος", lemma="βιβλος", morph="n-f-s", id="w1")

        assert word.text == "βιβλος"
        assert word.lemma == "βιβλος"
        assert word.morph == "n-f-s"
        assert word.id == "w1"

    def test_from_xml_element(self) -> None:
        """Test creating GreekWord from XML element."""
        xml_str = '<w lemma="βιβλος" ana="n-f-s" xml:id="w1">βιβλος</w>'
        element = ET.fromstring(xml_str)

        word = GreekWord.from_xml_element(element)

        assert word.text == "βιβλος"
        assert word.lemma == "βιβλος"
        assert word.morph == "n-f-s"
        assert word.id == "w1"

    def test_from_xml_element_minimal(self) -> None:
        """Test parsing element with minimal attributes."""
        xml_str = "<w>γενεσεως</w>"
        element = ET.fromstring(xml_str)

        word = GreekWord.from_xml_element(element)

        assert word.text == "γενεσεως"
        assert word.lemma is None
        assert word.morph is None
        assert word.id is None

    def test_to_dict(self) -> None:
        """Test converting GreekWord to dictionary."""
        word = GreekWord(text="βιβλος", lemma="βιβλος", morph="n-f-s", id="w1")

        expected = {
            "text": "βιβλος",
            "lemma": "βιβλος",
            "morph": "n-f-s",
            "strong_number": None,
            "id": "w1",
        }

        assert word.to_dict() == expected


class TestGreekVerse:
    """Test the GreekVerse dataclass."""

    def test_creation(self) -> None:
        """Test creating GreekVerse instances."""
        verse_id = VerseID(book="MAT", chapter=1, verse=1)
        words = [
            GreekWord(text="βιβλος"),
            GreekWord(text="γενεσεως"),
        ]

        verse = GreekVerse(verse_id=verse_id, words=words, tei_id="B01K1V1")

        assert verse.verse_id == verse_id
        assert len(verse.words) == 2
        assert verse.tei_id == "B01K1V1"

    def test_from_xml_element(self) -> None:
        """Test creating GreekVerse from XML element."""
        xml_str = """
        <ab n="B01K1V1" xmlns="http://www.tei-c.org/ns/1.0">
            <w lemma="βιβλος">βιβλος</w>
            <w lemma="γενεσις">γενεσεως</w>
            <w lemma="ιησους">ιησου</w>
        </ab>
        """
        element = ET.fromstring(xml_str)

        verse = GreekVerse.from_xml_element(element, "MAT")

        assert verse is not None
        assert verse.verse_id.book == "MAT"
        assert verse.verse_id.chapter == 1
        assert verse.verse_id.verse == 1
        assert verse.tei_id == "B01K1V1"
        assert len(verse.words) == 3
        assert verse.words[0].text == "βιβλος"
        assert verse.words[1].text == "γενεσεως"
        assert verse.words[2].text == "ιησου"

    def test_from_xml_element_different_chapters(self) -> None:
        """Test parsing different chapter and verse numbers."""
        xml_str = '<ab n="B01K5V10"></ab>'
        element = ET.fromstring(xml_str)

        verse = GreekVerse.from_xml_element(element, "MAT")

        assert verse is not None
        assert verse.verse_id.chapter == 5
        assert verse.verse_id.verse == 10

    def test_from_xml_element_invalid_tei_id(self) -> None:
        """Test handling invalid TEI ID."""
        xml_str = '<ab n="Invalid"></ab>'
        element = ET.fromstring(xml_str)

        verse = GreekVerse.from_xml_element(element, "MAT")

        assert verse is None

    def test_from_xml_element_no_tei_id(self) -> None:
        """Test handling missing TEI ID."""
        xml_str = "<ab></ab>"
        element = ET.fromstring(xml_str)

        verse = GreekVerse.from_xml_element(element, "MAT")

        assert verse is None

    def test_to_dict(self) -> None:
        """Test converting GreekVerse to dictionary."""
        verse_id = VerseID(book="MAT", chapter=1, verse=1)
        words = [GreekWord(text="βιβλος")]

        verse = GreekVerse(verse_id=verse_id, words=words, tei_id="B01K1V1")

        result = verse.to_dict()

        assert result["verse_id"] == "MAT.1.1"
        assert result["tei_id"] == "B01K1V1"
        assert len(result["words"]) == 1
        assert result["words"][0]["text"] == "βιβλος"


class TestGreekParser:
    """Test the GreekParser class."""

    def test_initialization(self) -> None:
        """Test parser initialization."""
        parser = GreekParser()
        assert parser.namespace == {"tei": "http://www.tei-c.org/ns/1.0"}

    def test_parse_file_not_found(self) -> None:
        """Test parsing non-existent file."""
        parser = GreekParser()

        try:
            parser.parse_file("nonexistent.xml", "MAT")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "not found" in str(e)

    def test_parse_file_valid(self) -> None:
        """Test parsing valid Greek XML file."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            <text>
                <body>
                    <div type="book">
                        <div type="chapter">
                            <ab n="B01K1V1">
                                <w lemma="βιβλος">βιβλος</w>
                                <w lemma="γενεσις">γενεσεως</w>
                            </ab>
                            <ab n="B01K1V2">
                                <w lemma="αβρααμ">αβρααμ</w>
                                <w lemma="γενναω">εγεννησεν</w>
                            </ab>
                        </div>
                    </div>
                </body>
            </text>
        </TEI>
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            parser = GreekParser()
            verses = parser.parse_file(temp_path, "MAT")

            assert len(verses) == 2

            # Check first verse
            verse1 = verses[0]
            assert verse1.verse_id.book == "MAT"
            assert verse1.verse_id.chapter == 1
            assert verse1.verse_id.verse == 1
            assert len(verse1.words) == 2
            assert verse1.words[0].text == "βιβλος"

            # Check second verse
            verse2 = verses[1]
            assert verse2.verse_id.verse == 2
            assert len(verse2.words) == 2

        finally:
            Path(temp_path).unlink()

    def test_parse_file_malformed_xml(self) -> None:
        """Test parsing malformed XML."""
        xml_content = "<TEI><unclosed>"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            parser = GreekParser()
            parser.parse_file(temp_path, "MAT")
            assert False, "Should have raised ParseError"
        except ET.ParseError:
            pass
        finally:
            Path(temp_path).unlink()

    def test_parse_book_valid_code(self) -> None:
        """Test parsing book with valid code."""
        parser = GreekParser()

        # Mock data directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Matthew file
            xml_content = """<?xml version="1.0" encoding="utf-8"?>
            <TEI xmlns="http://www.tei-c.org/ns/1.0">
                <text>
                    <body>
                        <ab n="B01K1V1">
                            <w>βιβλος</w>
                        </ab>
                    </body>
                </text>
            </TEI>
            """

            mat_file = Path(temp_dir) / "MAT.xml"
            mat_file.write_text(xml_content, encoding="utf-8")

            verses = parser.parse_book("MAT", temp_dir)
            assert len(verses) == 1
            assert verses[0].verse_id.book == "MAT"

    def test_parse_book_filename_mapping(self) -> None:
        """Test book code to filename mapping."""
        parser = GreekParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test Mark (MAR.xml) and John (JOH.xml) mappings
            xml_content = """<?xml version="1.0"?>
            <TEI xmlns="http://www.tei-c.org/ns/1.0">
                <text><body><ab n="B01K1V1"><w>test</w></ab></body></text>
            </TEI>
            """

            # Create MAR.xml for Mark
            mar_file = Path(temp_dir) / "MAR.xml"
            mar_file.write_text(xml_content)

            verses = parser.parse_book("MRK", temp_dir)  # MRK maps to MAR.xml
            assert len(verses) == 1
            assert verses[0].verse_id.book == "MRK"

    def test_parse_book_invalid_code(self) -> None:
        """Test parsing book with invalid code."""
        parser = GreekParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                parser.parse_book("XXX", temp_dir)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Unknown Greek book code" in str(e)

    def test_get_book_statistics(self) -> None:
        """Test getting book statistics."""
        parser = GreekParser()

        # Create test verses
        verses = [
            GreekVerse(
                verse_id=VerseID(book="MAT", chapter=1, verse=1),
                words=[
                    GreekWord(text="βιβλος"),
                    GreekWord(text="γενεσεως"),
                ],
                tei_id="B01K1V1",
            ),
            GreekVerse(
                verse_id=VerseID(book="MAT", chapter=1, verse=2),
                words=[
                    GreekWord(text="αβρααμ"),
                ],
                tei_id="B01K1V2",
            ),
        ]

        stats = parser.get_book_statistics(verses)

        assert stats["verse_count"] == 2
        assert stats["word_count"] == 3
        assert stats["book_code"] == "MAT"

    def test_get_book_statistics_empty(self) -> None:
        """Test statistics for empty verse list."""
        parser = GreekParser()

        stats = parser.get_book_statistics([])

        assert stats["verse_count"] == 0
        assert stats["word_count"] == 0

    def test_extract_lemmas(self) -> None:
        """Test extracting lemmas."""
        parser = GreekParser()

        verses = [
            GreekVerse(
                verse_id=VerseID(book="MAT", chapter=1, verse=1),
                words=[
                    GreekWord(text="βιβλος", lemma="βιβλος"),
                    GreekWord(text="γενεσεως", lemma="γενεσις"),
                    GreekWord(text="ιησου", lemma="ιησους"),
                ],
                tei_id="B01K1V1",
            ),
            GreekVerse(
                verse_id=VerseID(book="MAT", chapter=1, verse=2),
                words=[
                    GreekWord(text="αβρααμ", lemma="αβρααμ"),
                    GreekWord(text="εγεννησεν", lemma="γενναω"),
                    GreekWord(text="τον", lemma="ο"),
                    GreekWord(text="text_no_lemma"),  # No lemma
                ],
                tei_id="B01K1V2",
            ),
        ]

        lemmas = parser.extract_lemmas(verses)

        # Should be sorted and unique
        assert lemmas == ["αβρααμ", "βιβλος", "γενεσις", "γενναω", "ιησους", "ο"]

    def test_get_all_nt_books(self) -> None:
        """Test getting all NT book codes."""
        books = GreekParser.get_all_nt_books()

        assert len(books) == 27  # NT has 27 books
        assert "MAT" in books
        assert "REV" in books
        assert "1CO" in books
        assert "3JN" in books

    def test_verse_id_parsing_edge_cases(self) -> None:
        """Test edge cases in verse ID parsing."""
        # Test different book numbers, chapters, verses
        test_cases = [
            ("B01K1V1", 1, 1, 1),  # Book 1, Chapter 1, Verse 1
            ("B27K22V21", 27, 22, 21),  # Book 27, Chapter 22, Verse 21
            ("B01K10V5", 1, 10, 5),  # Double digit chapter
        ]

        for tei_id, expected_book_num, expected_chapter, expected_verse in test_cases:
            xml_str = f'<ab n="{tei_id}"></ab>'
            element = ET.fromstring(xml_str)

            verse = GreekVerse.from_xml_element(element, "MAT")

            assert verse is not None
            assert verse.verse_id.chapter == expected_chapter
            assert verse.verse_id.verse == expected_verse
            assert verse.tei_id == tei_id
