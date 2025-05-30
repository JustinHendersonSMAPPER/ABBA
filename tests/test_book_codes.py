"""Unit tests for book_codes module."""

from abba import (
    BOOK_INFO,
    BookCode,
    Canon,
    Testament,
    get_book_info,
    get_book_name,
    get_book_order,
    get_books_by_canon,
    get_books_by_testament,
    get_chapter_count,
    is_valid_book_code,
    normalize_book_name,
)


class TestBookCode:
    """Test the BookCode enum."""

    def test_book_code_values(self):
        """Test that BookCode enum has correct values."""
        assert BookCode.GEN.value == "GEN"
        assert BookCode.REV.value == "REV"
        assert BookCode.SA1.value == "1SA"
        assert BookCode.JN3.value == "3JN"

    def test_book_code_count(self):
        """Test that we have all 66 books."""
        # Count unique values (some enum names differ from values)
        unique_codes = set(code.value for code in BookCode)
        assert len(unique_codes) == 66


class TestBookInfo:
    """Test BOOK_INFO dictionary."""

    def test_all_books_present(self):
        """Test that all 66 books are in BOOK_INFO."""
        assert len(BOOK_INFO) == 66

    def test_book_info_structure(self):
        """Test that each book has required fields."""
        for code, info in BOOK_INFO.items():
            assert "name" in info
            assert "abbr" in info
            assert "chapters" in info
            assert "testament" in info
            assert isinstance(info["chapters"], int)
            assert isinstance(info["testament"], Testament)

    def test_testament_distribution(self):
        """Test correct number of OT and NT books."""
        ot_books = [b for b, info in BOOK_INFO.items() if info["testament"] == Testament.OLD]
        nt_books = [b for b, info in BOOK_INFO.items() if info["testament"] == Testament.NEW]
        assert len(ot_books) == 39
        assert len(nt_books) == 27


class TestNormalizeBookName:
    """Test normalize_book_name function."""

    def test_already_normalized(self):
        """Test that already normalized codes are returned as-is."""
        assert normalize_book_name("GEN") == "GEN"
        assert normalize_book_name("gen") == "GEN"
        assert normalize_book_name("REV") == "REV"

    def test_full_names(self):
        """Test normalizing full book names."""
        assert normalize_book_name("Genesis") == "GEN"
        assert normalize_book_name("genesis") == "GEN"
        assert normalize_book_name("Revelation") == "REV"
        assert normalize_book_name("1 Samuel") == "1SA"
        assert normalize_book_name("1 samuel") == "1SA"

    def test_abbreviations(self):
        """Test various abbreviation formats."""
        assert normalize_book_name("Gen") == "GEN"
        assert normalize_book_name("gen") == "GEN"
        assert normalize_book_name("Ge") == "GEN"
        assert normalize_book_name("Gn") == "GEN"
        assert normalize_book_name("Matt") == "MAT"
        assert normalize_book_name("Mt") == "MAT"

    def test_numbered_books(self):
        """Test various formats for numbered books."""
        # Different ways to write 1 Samuel
        assert normalize_book_name("1Sam") == "1SA"
        assert normalize_book_name("1 Sam") == "1SA"
        assert normalize_book_name("I Samuel") == "1SA"
        assert normalize_book_name("i samuel") == "1SA"
        assert normalize_book_name("1sa") == "1SA"
        assert normalize_book_name("1 sa") == "1SA"

    def test_special_cases(self):
        """Test special book name cases."""
        assert normalize_book_name("Song of Solomon") == "SNG"
        assert normalize_book_name("Song of Songs") == "SNG"
        assert normalize_book_name("Canticles") == "SNG"
        assert normalize_book_name("Psalms") == "PSA"
        assert normalize_book_name("Psalm") == "PSA"

    def test_invalid_names(self):
        """Test that invalid names return None."""
        assert normalize_book_name("Invalid") is None
        assert normalize_book_name("4 Maccabees") is None
        assert normalize_book_name("") is None


class TestGetBookInfo:
    """Test get_book_info function."""

    def test_valid_codes(self):
        """Test getting info for valid book codes."""
        gen_info = get_book_info("GEN")
        assert gen_info is not None
        assert gen_info["name"] == "Genesis"
        assert gen_info["abbr"] == "Gen"
        assert gen_info["chapters"] == 50
        assert gen_info["testament"] == Testament.OLD

    def test_case_insensitive(self):
        """Test that book codes are case-insensitive."""
        assert get_book_info("gen") == get_book_info("GEN")
        assert get_book_info("Rev") == get_book_info("REV")

    def test_invalid_codes(self):
        """Test that invalid codes return None."""
        assert get_book_info("XXX") is None
        assert get_book_info("") is None


class TestGetBookName:
    """Test get_book_name function."""

    def test_full_names(self):
        """Test getting full book names."""
        assert get_book_name("GEN") == "Genesis"
        assert get_book_name("REV") == "Revelation"
        assert get_book_name("1SA") == "1 Samuel"

    def test_abbreviations(self):
        """Test getting book abbreviations."""
        assert get_book_name("GEN", "abbr") == "Gen"
        assert get_book_name("REV", "abbr") == "Rev"
        assert get_book_name("1SA", "abbr") == "1Sam"

    def test_invalid_codes(self):
        """Test that invalid codes return None."""
        assert get_book_name("XXX") is None
        assert get_book_name("GEN", "invalid_form") is None


class TestGetBooksByTestament:
    """Test get_books_by_testament function."""

    def test_old_testament(self):
        """Test getting OT books."""
        ot_books = get_books_by_testament(Testament.OLD)
        assert len(ot_books) == 39
        assert "GEN" in ot_books
        assert "MAL" in ot_books
        assert "MAT" not in ot_books

    def test_new_testament(self):
        """Test getting NT books."""
        nt_books = get_books_by_testament(Testament.NEW)
        assert len(nt_books) == 27
        assert "MAT" in nt_books
        assert "REV" in nt_books
        assert "GEN" not in nt_books


class TestGetBooksByCanon:
    """Test get_books_by_canon function."""

    def test_protestant_canon(self):
        """Test Protestant canon (should have all 66 books)."""
        books = get_books_by_canon(Canon.PROTESTANT)
        assert len(books) == 66
        assert "GEN" in books
        assert "REV" in books

    def test_other_canons(self):
        """Test that other canons are defined (even if incomplete)."""
        # These are currently just placeholders
        catholic = get_books_by_canon(Canon.CATHOLIC)
        orthodox = get_books_by_canon(Canon.ORTHODOX)
        ethiopian = get_books_by_canon(Canon.ETHIOPIAN)

        # All should at least have the Protestant books for now
        assert len(catholic) >= 66
        assert len(orthodox) >= 66
        assert len(ethiopian) >= 66


class TestGetBookOrder:
    """Test get_book_order function."""

    def test_book_ordering(self):
        """Test that books are ordered correctly."""
        assert get_book_order("GEN") == 1
        assert get_book_order("EXO") == 2
        assert get_book_order("MAL") == 39
        assert get_book_order("MAT") == 40
        assert get_book_order("REV") == 66

    def test_case_insensitive(self):
        """Test that ordering is case-insensitive."""
        assert get_book_order("gen") == 1
        assert get_book_order("rev") == 66

    def test_invalid_codes(self):
        """Test that invalid codes return None."""
        assert get_book_order("XXX") is None
        assert get_book_order("") is None


class TestIsValidBookCode:
    """Test is_valid_book_code function."""

    def test_valid_codes(self):
        """Test valid book codes."""
        assert is_valid_book_code("GEN") is True
        assert is_valid_book_code("gen") is True
        assert is_valid_book_code("REV") is True
        assert is_valid_book_code("1SA") is True

    def test_invalid_codes(self):
        """Test invalid book codes."""
        assert is_valid_book_code("XXX") is False
        assert is_valid_book_code("Genesis") is False
        assert is_valid_book_code("") is False


class TestGetChapterCount:
    """Test get_chapter_count function."""

    def test_chapter_counts(self):
        """Test correct chapter counts for various books."""
        assert get_chapter_count("GEN") == 50
        assert get_chapter_count("PSA") == 150
        assert get_chapter_count("OBA") == 1
        assert get_chapter_count("MAT") == 28
        assert get_chapter_count("3JN") == 1
        assert get_chapter_count("REV") == 22

    def test_case_insensitive(self):
        """Test that function is case-insensitive."""
        assert get_chapter_count("gen") == 50
        assert get_chapter_count("rev") == 22

    def test_invalid_codes(self):
        """Test that invalid codes return None."""
        assert get_chapter_count("XXX") is None
        assert get_chapter_count("") is None


class TestBookAliases:
    """Test comprehensive book alias coverage."""

    def test_all_common_abbreviations(self):
        """Test that common abbreviations work."""
        test_cases = [
            # OT books
            ("Gn", "GEN"),
            ("Ex", "EXO"),
            ("Lv", "LEV"),
            ("Nm", "NUM"),
            ("Dt", "DEU"),
            ("Jos", "JOS"),
            ("Jg", "JDG"),
            ("Ru", "RUT"),
            # Numbered books
            ("1 Sam", "1SA"),
            ("2 Sam", "2SA"),
            ("1 Kgs", "1KI"),
            ("2 Kgs", "2KI"),
            ("1 Chr", "1CH"),
            ("2 Chr", "2CH"),
            # Prophets
            ("Is", "ISA"),
            ("Jr", "JER"),
            ("Ez", "EZK"),
            ("Dn", "DAN"),
            # NT books
            ("Mt", "MAT"),
            ("Mk", "MRK"),
            ("Lk", "LUK"),
            ("Jn", "JHN"),
            ("Ac", "ACT"),
            ("Ro", "ROM"),
            # Epistles
            ("1 Cor", "1CO"),
            ("2 Cor", "2CO"),
            ("Ga", "GAL"),
            ("Ep", "EPH"),
            ("1 Thess", "1TH"),
            ("2 Thess", "2TH"),
            ("1 Tim", "1TI"),
            ("2 Tim", "2TI"),
            # General epistles
            ("Heb", "HEB"),
            ("Jas", "JAS"),
            ("1 Pet", "1PE"),
            ("2 Pet", "2PE"),
            ("1 Jn", "1JN"),
            ("2 Jn", "2JN"),
            ("3 Jn", "3JN"),
            ("Jd", "JUD"),
            ("Re", "REV"),
        ]

        for abbr, expected in test_cases:
            assert normalize_book_name(abbr) == expected, f"Failed for {abbr}"
            assert normalize_book_name(abbr.lower()) == expected, f"Failed for {abbr.lower()}"
