"""Unit tests for translation_parser module."""

import json
import tempfile
from pathlib import Path

from abba import Translation, TranslationParser, TranslationVerse, VerseID


class TestTranslationVerse:
    """Test the TranslationVerse dataclass."""

    def test_creation(self) -> None:
        """Test creating TranslationVerse instances."""
        verse_id = VerseID(book="GEN", chapter=1, verse=1)
        verse = TranslationVerse(
            verse_id=verse_id,
            text="In the beginning God created the heavens and the earth.",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1,
        )

        assert verse.verse_id == verse_id
        assert verse.text == "In the beginning God created the heavens and the earth."
        assert verse.original_book_name == "Genesis"
        assert verse.original_chapter == 1
        assert verse.original_verse == 1

    def test_to_dict(self) -> None:
        """Test converting TranslationVerse to dictionary."""
        verse_id = VerseID(book="GEN", chapter=1, verse=1)
        verse = TranslationVerse(
            verse_id=verse_id,
            text="In the beginning God created the heavens and the earth.",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1,
        )

        expected = {
            "verse_id": "GEN.1.1",
            "text": "In the beginning God created the heavens and the earth.",
            "original_book_name": "Genesis",
            "original_chapter": 1,
            "original_verse": 1,
        }

        assert verse.to_dict() == expected


class TestTranslation:
    """Test the Translation dataclass."""

    def test_creation(self) -> None:
        """Test creating Translation instances."""
        translation = Translation(
            version="KJV", name="King James Version", language="en", copyright="Public Domain"
        )

        assert translation.version == "KJV"
        assert translation.name == "King James Version"
        assert translation.language == "en"
        assert translation.copyright == "Public Domain"
        assert translation.verses == []

    def test_creation_with_verses(self) -> None:
        """Test creating Translation with verses."""
        verse = TranslationVerse(
            verse_id=VerseID(book="GEN", chapter=1, verse=1),
            text="In the beginning...",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1,
        )

        translation = Translation(
            version="KJV", name="King James Version", language="en", verses=[verse]
        )

        assert len(translation.verses) == 1
        assert translation.verses[0] == verse

    def test_to_dict(self) -> None:
        """Test converting Translation to dictionary."""
        verse = TranslationVerse(
            verse_id=VerseID(book="GEN", chapter=1, verse=1),
            text="In the beginning...",
            original_book_name="Genesis",
            original_chapter=1,
            original_verse=1,
        )

        translation = Translation(
            version="KJV",
            name="King James Version",
            language="en",
            copyright="Public Domain",
            verses=[verse],
        )

        result = translation.to_dict()

        assert result["version"] == "KJV"
        assert result["name"] == "King James Version"
        assert result["language"] == "en"
        assert result["copyright"] == "Public Domain"
        assert result["verse_count"] == 1
        assert len(result["verses"]) == 1


class TestTranslationParser:
    """Test the TranslationParser class."""

    def test_initialization(self) -> None:
        """Test parser initialization."""
        parser = TranslationParser()
        assert parser is not None

    def test_parse_file_not_found(self) -> None:
        """Test parsing non-existent file."""
        parser = TranslationParser()

        try:
            parser.parse_file("nonexistent.json")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "not found" in str(e)

    def test_parse_file_valid(self) -> None:
        """Test parsing valid translation JSON file."""
        translation_data = {
            "version": "KJV",
            "name": "King James Version",
            "language": "en",
            "copyright": "Public Domain",
            "source": "bible.org",
            "website": "https://bible.org",
            "license_url": "https://creativecommons.org/publicdomain/",
            "books": {
                "Genesis": {
                    "name": "Genesis",
                    "abbr": "Gen",
                    "testament": "OT",
                    "chapters": [
                        {
                            "chapter": 1,
                            "verses": [
                                {
                                    "verse": 1,
                                    "text": (
                                        "In the beginning God created the heaven and the earth."
                                    ),
                                },
                                {
                                    "verse": 2,
                                    "text": "And the earth was without form, and void.",
                                },
                            ],
                        }
                    ],
                },
                "Matthew": {
                    "name": "Matthew",
                    "chapters": [
                        {
                            "chapter": 1,
                            "verses": [
                                {"verse": 1, "text": "The book of the generation of Jesus Christ."}
                            ],
                        }
                    ],
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(translation_data, f)
            temp_path = f.name

        try:
            parser = TranslationParser()
            translation = parser.parse_file(temp_path)

            # Check metadata
            assert translation.version == "KJV"
            assert translation.name == "King James Version"
            assert translation.language == "en"
            assert translation.copyright == "Public Domain"
            assert translation.source == "bible.org"
            assert translation.website == "https://bible.org"
            assert translation.license_url == "https://creativecommons.org/publicdomain/"

            # Check verses
            assert len(translation.verses) == 3  # 2 Genesis + 1 Matthew

            # Check first verse
            verse1 = translation.verses[0]
            assert verse1.verse_id.book == "GEN"
            assert verse1.verse_id.chapter == 1
            assert verse1.verse_id.verse == 1
            assert "In the beginning" in verse1.text
            assert verse1.original_book_name == "Genesis"

            # Check Matthew verse
            mat_verse = next(v for v in translation.verses if v.verse_id.book == "MAT")
            assert mat_verse.verse_id.chapter == 1
            assert mat_verse.verse_id.verse == 1
            assert "Jesus Christ" in mat_verse.text

        finally:
            Path(temp_path).unlink()

    def test_parse_file_malformed_json(self) -> None:
        """Test parsing malformed JSON."""
        json_content = '{"invalid": json}'

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            parser = TranslationParser()
            parser.parse_file(temp_path)
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            pass
        finally:
            Path(temp_path).unlink()

    def test_parse_file_missing_required_fields(self) -> None:
        """Test parsing JSON with missing required fields."""
        translation_data = {
            "language": "en",
            # Missing version and name
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(translation_data, f)
            temp_path = f.name

        try:
            parser = TranslationParser()
            parser.parse_file(temp_path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Missing required fields" in str(e)
        finally:
            Path(temp_path).unlink()

    def test_parse_file_unknown_books(self) -> None:
        """Test parsing with unknown book names."""
        translation_data = {
            "version": "TEST",
            "name": "Test Translation",
            "language": "en",
            "books": {
                "Genesis": {
                    "chapters": [{"chapter": 1, "verses": [{"verse": 1, "text": "Genesis text"}]}]
                },
                "Tobit": {  # Deuterocanonical book not in Protestant canon
                    "chapters": [{"chapter": 1, "verses": [{"verse": 1, "text": "Tobit text"}]}]
                },
                "UnknownBook": {  # Completely unknown book
                    "chapters": [{"chapter": 1, "verses": [{"verse": 1, "text": "Unknown text"}]}]
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(translation_data, f)
            temp_path = f.name

        try:
            parser = TranslationParser()
            translation = parser.parse_file(temp_path)

            # Should include Genesis and Tobit (both are known books)
            # UnknownBook should be skipped
            assert len(translation.verses) == 2
            
            # Check that we have Genesis and Tobit
            book_codes = {v.verse_id.book for v in translation.verses}
            assert "GEN" in book_codes
            assert "TOB" in book_codes  # Tobit is a valid deuterocanonical book

        finally:
            Path(temp_path).unlink()

    def test_parse_directory(self) -> None:
        """Test parsing directory of translation files."""
        parser = TranslationParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid translation file
            valid_data = {
                "version": "KJV",
                "name": "King James Version",
                "language": "en",
                "books": {
                    "Genesis": {
                        "chapters": [
                            {"chapter": 1, "verses": [{"verse": 1, "text": "In the beginning..."}]}
                        ]
                    }
                },
            }

            valid_file = Path(temp_dir) / "kjv.json"
            with open(valid_file, "w", encoding="utf-8") as f:
                json.dump(valid_data, f)

            # Create invalid translation file
            invalid_file = Path(temp_dir) / "invalid.json"
            with open(invalid_file, "w", encoding="utf-8") as f:
                f.write('{"invalid": json}')

            # Create non-JSON file
            text_file = Path(temp_dir) / "readme.txt"
            text_file.write_text("This is not JSON")

            translations = parser.parse_directory(temp_dir)

            # Should only include valid translation
            assert len(translations) == 1
            assert translations[0].version == "KJV"

    def test_parse_directory_with_limit(self) -> None:
        """Test parsing directory with limit."""
        parser = TranslationParser()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple valid files
            for i in range(5):
                data = {
                    "version": f"TEST{i}",
                    "name": f"Test Translation {i}",
                    "language": "en",
                    "books": {},
                }

                file_path = Path(temp_dir) / f"test{i}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)

            translations = parser.parse_directory(temp_dir, limit=3)

            # Should only parse first 3 files
            assert len(translations) == 3

    def test_get_translation_statistics(self) -> None:
        """Test getting translation statistics."""
        parser = TranslationParser()

        verses = [
            TranslationVerse(
                verse_id=VerseID(book="GEN", chapter=1, verse=1),
                text="In the beginning God created the heavens and the earth.",
                original_book_name="Genesis",
                original_chapter=1,
                original_verse=1,
            ),
            TranslationVerse(
                verse_id=VerseID(book="GEN", chapter=1, verse=2),
                text="And the earth was without form, and void.",
                original_book_name="Genesis",
                original_chapter=1,
                original_verse=2,
            ),
            TranslationVerse(
                verse_id=VerseID(book="MAT", chapter=1, verse=1),
                text="The book of the generation of Jesus Christ.",
                original_book_name="Matthew",
                original_chapter=1,
                original_verse=1,
            ),
        ]

        translation = Translation(
            version="KJV", name="King James Version", language="en", verses=verses
        )

        stats = parser.get_translation_statistics(translation)

        assert stats["version"] == "KJV"
        assert stats["name"] == "King James Version"
        assert stats["language"] == "en"
        assert stats["verse_count"] == 3
        assert stats["book_count"] == 2  # GEN and MAT
        assert stats["avg_verse_length"] > 0

    def test_get_translation_statistics_empty(self) -> None:
        """Test statistics for empty translation."""
        parser = TranslationParser()

        translation = Translation(version="EMPTY", name="Empty Translation", language="en")

        stats = parser.get_translation_statistics(translation)

        assert stats["verse_count"] == 0
        assert stats["book_count"] == 0
        assert stats["avg_verse_length"] == 0

    def test_normalize_book_names(self) -> None:
        """Test normalizing book names."""
        parser = TranslationParser()

        translation_data = {
            "books": {
                "Genesis": {},
                "1 Samuel": {},
                "Matthew": {},
                "Tobit": {},  # Deuterocanonical
                "Unknown": {},  # Unknown book
            }
        }

        mapping = parser.normalize_book_names(translation_data)

        assert mapping["Genesis"] == "GEN"
        assert mapping["1 Samuel"] == "1SA"
        assert mapping["Matthew"] == "MAT"
        assert mapping["Tobit"] == "TOB"  # Valid deuterocanonical book
        assert mapping["Unknown"] is None

    def test_extract_verse_variations(self) -> None:
        """Test extracting verse variations across translations."""
        parser = TranslationParser()

        translation1 = Translation(
            version="KJV",
            name="King James Version",
            language="en",
            verses=[
                TranslationVerse(
                    verse_id=VerseID(book="GEN", chapter=1, verse=1),
                    text="In the beginning God created the heaven and the earth.",
                    original_book_name="Genesis",
                    original_chapter=1,
                    original_verse=1,
                )
            ],
        )

        translation2 = Translation(
            version="ESV",
            name="English Standard Version",
            language="en",
            verses=[
                TranslationVerse(
                    verse_id=VerseID(book="GEN", chapter=1, verse=1),
                    text="In the beginning, God created the heavens and the earth.",
                    original_book_name="Genesis",
                    original_chapter=1,
                    original_verse=1,
                )
            ],
        )

        variations = parser.extract_verse_variations([translation1, translation2])

        assert "GEN.1.1" in variations
        assert len(variations["GEN.1.1"]) == 2
        assert "heaven and the earth" in variations["GEN.1.1"][0]  # KJV
        assert "heavens and the earth" in variations["GEN.1.1"][1]  # ESV

    def test_invalid_verse_data_handling(self) -> None:
        """Test handling of invalid verse data."""
        translation_data = {
            "version": "TEST",
            "name": "Test Translation",
            "language": "en",
            "books": {
                "Genesis": {
                    "chapters": [
                        {
                            "chapter": 1,
                            "verses": [
                                {"verse": 1, "text": "Valid verse"},
                                {"verse": 0, "text": "Invalid verse number"},  # Invalid
                                {"chapter": 1, "text": "Missing verse number"},  # Invalid
                                {"verse": 2, "text": ""},  # Empty text
                                {"verse": 3},  # Missing text
                            ],
                        },
                        {
                            "chapter": 0,  # Invalid chapter
                            "verses": [{"verse": 1, "text": "Invalid chapter"}],
                        },
                    ]
                }
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(translation_data, f)
            temp_path = f.name

        try:
            parser = TranslationParser()
            translation = parser.parse_file(temp_path)

            # Should only include valid verse
            assert len(translation.verses) == 1
            assert translation.verses[0].text == "Valid verse"

        finally:
            Path(temp_path).unlink()
