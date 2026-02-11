"""End-to-end integration tests for the ABBA API.

These tests exercise the full stack: a seeded SQLite database is wired into
the FastAPI application via ``create_app(db_path=...)``, and every major
endpoint is hit through the Starlette ``TestClient`` to confirm that data
inserted into the database is properly accessible through the REST API.
"""

# pylint: disable=redefined-outer-name
import pytest
from fastapi.testclient import TestClient

from abba.api.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(seeded_db):
    """Create a TestClient backed by the seeded test database."""
    app = create_app(db_path=seeded_db)
    return TestClient(app)


# ---------------------------------------------------------------------------
# API Root
# ---------------------------------------------------------------------------


class TestAPIRoot:
    """Verify the API info endpoint returns expected metadata."""

    def test_root_returns_metadata(self, client):
        resp = client.get("/api/v1/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "ABBA Bible Study API"
        assert body["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# Verse Retrieval
# ---------------------------------------------------------------------------


class TestVerseEndpoints:
    """Test single-verse and chapter retrieval endpoints."""

    def test_get_verse_basic(self, client):
        """Retrieve Genesis 1:1 at basic depth."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1")
        assert resp.status_code == 200
        body = resp.json()
        assert "beginning" in body["text"].lower()
        assert body["chapter"] == 1
        assert body["verse"] == 1
        assert body["translation_id"] == "engbsb"
        # Basic depth should NOT include words
        assert body.get("words") is None

    def test_get_verse_standard_depth(self, client):
        """Standard depth includes original-language words."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=standard")
        assert resp.status_code == 200
        body = resp.json()
        assert body["words"] is not None
        assert len(body["words"]) == 7  # 7 Hebrew words in Gen 1:1
        # Verify first word is bereshit
        first_word = body["words"][0]
        assert first_word["original_text"] is not None
        assert first_word["strongs_number"] == "H7225"

    def test_get_verse_not_found(self, client):
        """Non-existent verse returns 404."""
        resp = client.get("/api/v1/verses/engbsb/1/99/99")
        assert resp.status_code == 404

    def test_get_chapter(self, client):
        """Retrieve full chapter of Genesis 1 (5 seeded verses)."""
        resp = client.get("/api/v1/verses/engbsb/1/1")
        assert resp.status_code == 200
        verses = resp.json()
        assert len(verses) == 5
        # Verify ordered by verse number
        verse_nums = [v["verse"] for v in verses]
        assert verse_nums == [1, 2, 3, 4, 5]

    def test_get_chapter_not_found(self, client):
        """Non-existent chapter returns 404."""
        resp = client.get("/api/v1/verses/engbsb/1/99")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Full-Text Search
# ---------------------------------------------------------------------------


class TestTextSearch:
    """Test full-text search across translations."""

    def test_search_finds_matching_verses(self, client):
        """Search for 'light' returns Genesis 1:3-1:5."""
        resp = client.get("/api/v1/search/text", params={"q": "light", "translation_id": "engbsb"})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 2  # 'light' appears in v3, v4, v5
        texts = [r["text"].lower() for r in results]
        assert all("light" in t for t in texts)

    def test_search_beginning(self, client):
        """Search for 'beginning' finds Gen 1:1 and John 1:1-1:2."""
        resp = client.get("/api/v1/search/text", params={"q": "beginning", "translation_id": "engbsb"})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 2
        book_ids = {r["book_id"] for r in results}
        # Should find results in both Genesis (1) and John (43)
        assert 1 in book_ids
        assert 43 in book_ids


# ---------------------------------------------------------------------------
# Strong's Number Search
# ---------------------------------------------------------------------------


class TestStrongsSearch:
    """Test searching by Strong's concordance numbers."""

    def test_search_hebrew_strongs(self, client):
        """Search for H0430 (elohim) finds the word in Genesis."""
        resp = client.get("/api/v1/search/strongs/H0430")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert results[0]["strongs_number"] == "H0430"
        assert results[0]["language"] == "hebrew"
        assert results[0]["original_text"] is not None

    def test_search_greek_strongs(self, client):
        """Search for G3056 (logos) finds the word in John."""
        resp = client.get("/api/v1/search/strongs/G3056")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert results[0]["strongs_number"] == "G3056"
        assert results[0]["language"] == "greek"


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------


class TestLexicon:
    """Test lexicon entry retrieval."""

    def test_get_hebrew_lexicon_entry(self, client):
        """Retrieve the lexicon entry for H7225 (reshit / beginning)."""
        resp = client.get("/api/v1/lexicon/H7225")
        assert resp.status_code == 200
        body = resp.json()
        assert body["strongs_number"] == "H7225"
        assert body["original_word"] == "רֵאשִׁית"
        assert body["language"] == "hebrew"
        assert "beginning" in body["definition"].lower()

    def test_get_greek_lexicon_entry(self, client):
        """Retrieve the lexicon entry for G3056 (logos / word)."""
        resp = client.get("/api/v1/lexicon/G3056")
        assert resp.status_code == 200
        body = resp.json()
        assert body["strongs_number"] == "G3056"
        assert body["original_word"] == "λόγος"
        assert body["language"] == "greek"

    def test_lexicon_not_found(self, client):
        """Non-existent Strong's number returns 404."""
        resp = client.get("/api/v1/lexicon/H0000")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Word Analysis
# ---------------------------------------------------------------------------


class TestWordAnalysis:
    """Test complete word analysis (word + lexicon + morphology)."""

    def test_hebrew_word_analysis(self, client):
        """Full analysis of Gen 1:1 word 1 (bereshit) returns word + lexicon + morphology."""
        resp = client.get("/api/v1/words/Gen/1/1/1")
        assert resp.status_code == 200
        body = resp.json()

        # Word data
        word = body["word"]
        assert word["word_num"] == 1
        assert word["original_text"] is not None
        assert word["language"] == "hebrew"

        # Lexicon should be populated
        lex = body["lexicon"]
        assert lex is not None
        assert lex["strongs_number"] == "H7225"
        assert "beginning" in lex["definition"].lower()

        # Morphology should be populated
        morph = body["morphology"]
        assert morph is not None
        assert morph["code"] == "HR/Ncfsa"
        assert "hebrew" in morph["description"].lower()

    def test_greek_word_analysis(self, client):
        """Full analysis of John 1:1 word 5 (Logos)."""
        resp = client.get("/api/v1/words/John/1/1/5")
        assert resp.status_code == 200
        body = resp.json()

        word = body["word"]
        assert word["language"] == "greek"

        lex = body["lexicon"]
        assert lex is not None
        assert lex["strongs_number"] == "G3056"

        morph = body["morphology"]
        assert morph is not None
        assert morph["code"] == "GNnms"

    def test_word_not_found(self, client):
        """Non-existent word position returns 404."""
        resp = client.get("/api/v1/words/Gen/1/1/99")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Books Metadata
# ---------------------------------------------------------------------------


class TestBooks:
    """Test book listing and metadata endpoints."""

    def test_list_books(self, client):
        """List books returns at least the seeded books."""
        resp = client.get("/api/v1/books")
        assert resp.status_code == 200
        books = resp.json()
        assert len(books) >= 2  # Genesis and John (may have dupes from two translations)
        names = {b["name"] for b in books}
        assert "Genesis" in names
        assert "John" in names

    def test_get_book_by_id(self, client):
        """Get specific book metadata by ID."""
        resp = client.get("/api/v1/books/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Genesis"
        assert body["testament"] == "old"

    def test_book_not_found(self, client):
        """Non-existent book returns 404."""
        resp = client.get("/api/v1/books/999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Analysis Endpoints
# ---------------------------------------------------------------------------


class TestAnalysis:
    """Test the linguistic analysis endpoints."""

    def test_morphology_analysis(self, client):
        """Morphology analysis returns patterns for Hebrew."""
        resp = client.get("/api/v1/analysis/morphology", params={"language": "hebrew"})
        assert resp.status_code == 200
        patterns = resp.json()
        assert len(patterns) >= 1
        # Each pattern should have the expected shape
        first = patterns[0]
        assert "pattern" in first
        assert "count" in first
        assert first["count"] >= 1

    def test_word_frequency(self, client):
        """Word frequency returns results for Hebrew words."""
        resp = client.get("/api/v1/analysis/frequency", params={"strongs_pattern": "H%"})
        assert resp.status_code == 200
        freqs = resp.json()
        assert len(freqs) >= 1
        # H0853 (et) appears twice in Gen 1:1
        h0853 = [f for f in freqs if f["strongs_number"] == "H0853"]
        assert len(h0853) == 1
        assert h0853[0]["frequency"] == 2

    def test_semantic_domain(self, client):
        """Semantic domain search for 'beginning' finds relevant entries."""
        resp = client.get("/api/v1/analysis/semantic-domain/beginning")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        strongs = {r["strongs_number"] for r in results}
        # Should find both Hebrew and Greek "beginning"
        assert "H7225" in strongs
        assert "G0746" in strongs


# ---------------------------------------------------------------------------
# Translation Comparison
# ---------------------------------------------------------------------------


class TestTranslationComparison:
    """Test comparing a verse across multiple translations."""

    def test_compare_gen1_1(self, client):
        """Compare Genesis 1:1 between BSB and KJV."""
        resp = client.get(
            "/api/v1/compare/Gen/1/1",
            params={"translations": ["engbsb", "engkjv"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["reference"] == "Gen 1:1"
        # Original language words should be present
        assert len(body["original_words"]) >= 1


# ---------------------------------------------------------------------------
# Data Integrity: round-trip verification
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    """Verify that data inserted into the database is faithfully returned by the API."""

    def test_verse_text_roundtrip(self, client):
        """Exact verse text survives the insert -> API -> response cycle."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1")
        assert resp.json()["text"] == "In the beginning God created the heavens and the earth."

    def test_lexicon_definition_roundtrip(self, client):
        """Lexicon definition text is preserved exactly."""
        resp = client.get("/api/v1/lexicon/H1254")
        assert resp.json()["definition"] == "to create, shape, form"

    def test_word_strongs_roundtrip(self, client):
        """Word Strong's numbers match what was inserted."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=standard")
        words = resp.json()["words"]
        expected_strongs = ["H7225", "H1254", "H0430", "H0853", "H8064", "H0853", "H0776"]
        actual_strongs = [w["strongs_number"] for w in words]
        assert actual_strongs == expected_strongs

    def test_database_stats_reflect_seeded_data(self, seeded_db_manager):
        """Database statistics match the number of seeded records."""
        stats = seeded_db_manager.get_database_stats()
        assert stats["translations"] == 2
        assert stats["verses"] == 10  # 5 BSB Gen + 3 BSB John + 2 KJV Gen
        assert stats["words"] >= 12  # 7 Gen Hebrew + 5 John Greek
        assert stats["lexicon"] >= 11
        assert stats["morphology"] >= 12
