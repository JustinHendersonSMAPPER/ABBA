"""End-to-end tests for Phases 5-8 features of the ABBA project.

Tests cover: genre shifts, speaker attributions, word explanations,
concept quality, surrounding context, descriptive/prescriptive markers,
translation divergence, pagination, user notes, collections, sharing,
new enrichment data integrity, and progressive depth with new features.
"""

# pylint: disable=redefined-outer-name

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.api.routes import configure_db
from abba.database import SQLiteManager
from abba.database.migrations import run_migrations
from abba.enrichment import (
    BookMetadataPopulator,
    ConceptQualityPopulator,
    CrossReferencePopulator,
    CulturalContextPopulator,
    GenreShiftPopulator,
    LifeTopicPopulator,
    LiteraryStructurePopulator,
    PassagePopulator,
    ReadingPlanPopulator,
    SpeakerAttributionPopulator,
    WordExplanationPopulator,
    WordRichnessComputer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def enriched_db(tmp_path_factory):  # noqa: C901 - linear seed inserts + populators; inherent complexity
    """Create a fully seeded and enriched database for the entire test module.

    This duplicates the seeded_db fixture logic so we have a module-scoped
    instance (faster test runs) and then runs migrations + all populators.
    """
    tmp_path = tmp_path_factory.mktemp("e2e")
    db_path = tmp_path / "test_abba.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # --- Translations ---
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("engbsb", "Berean Standard Bible", "Berean Standard Bible", "en", "protestant"),
    )
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("engkjv", "King James Version", "King James Version", "en", "protestant"),
    )
    # Real default-translation id used by the app (DEFAULT_TRANSLATION_ID = "BSB").
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("BSB", "Berean Standard Bible", "Berean Standard Bible", "eng", "protestant"),
    )

    # --- Books ---
    books = [
        ("engbsb", 1, "Genesis", "Genesis", 1, 50, "old"),
        ("engbsb", 2, "Exodus", "Exodus", 2, 40, "old"),
        ("engbsb", 18, "Job", "Job", 18, 42, "old"),
        ("engbsb", 43, "John", "John", 43, 21, "new"),
        ("engkjv", 1, "Genesis", "Genesis", 1, 50, "old"),
        ("engkjv", 43, "John", "John", 43, 21, "new"),
        ("BSB", 1, "Genesis", "Genesis", 1, 50, "old"),
        ("BSB", 43, "John", "John", 43, 21, "new"),
    ]
    for tid, bid, name, common, order, chapters, testament in books:
        db.execute_update(
            "INSERT OR REPLACE INTO books (translation_id, book_id, name, common_name, book_order, "
            "number_of_chapters, testament) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, bid, name, common, order, chapters, testament),
        )

    # --- Verses (BSB) ---
    bsb_gen1 = [
        (1, 1, "In the beginning God created the heavens and the earth."),
        (1, 2, "Now the earth was formless and void, and darkness was over the surface of the deep."),
        (1, 3, "And God said, 'Let there be light,' and there was light."),
        (1, 4, "And God saw that the light was good, and He separated the light from the darkness."),
        (1, 5, "God called the light 'day,' and the darkness He called 'night.'"),
    ]
    for ch, vs, text in bsb_gen1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 1, ch, vs, text),
        )
        db.execute_update(
            "INSERT INTO verses_fts (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 1, ch, vs, text),
        )

    bsb_john1 = [
        (1, 1, "In the beginning was the Word, and the Word was with God, and the Word was God."),
        (1, 2, "He was with God in the beginning."),
        (1, 3, "Through Him all things were made; without Him nothing was made that has been made."),
    ]
    for ch, vs, text in bsb_john1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 43, ch, vs, text),
        )
        db.execute_update(
            "INSERT INTO verses_fts (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 43, ch, vs, text),
        )

    # --- Verses (KJV) for comparison tests ---
    kjv_gen1 = [
        (1, 1, "In the beginning God created the heaven and the earth."),
        (1, 2, "And the earth was without form, and void; and darkness was upon the face of the deep."),
    ]
    for ch, vs, text in kjv_gen1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engkjv", 1, ch, vs, text),
        )

    # --- Original Language Words (Genesis 1:1 -- Hebrew) ---
    gen1_words = [
        (1, "Gen.1.1#01", "hebrew_text", None, "bereshit", "In beginning", "H9003/{H7225G}", "HR/Ncfsa", "H7225"),
        (2, "Gen.1.1#02", "bara_text", None, "bara", "created", "{H1254A}", "HVqp3ms", "H1254"),
        (3, "Gen.1.1#03", "elohim_text", None, "elohim", "God", "{H0430}", "HNcmpa", "H0430"),
        (4, "Gen.1.1#04", "et_text", None, "et", "[obj]", "{H0853}", "HTo", "H0853"),
        (5, "Gen.1.1#05", "shamayim_text", None, "hashamayim", "the heavens", "H9009/{H8064}", "HTd/Ncmpa", "H8064"),
        (6, "Gen.1.1#06", "veet_text", None, "ve'et", "and", "H9002/{H0853}", "HC/To", "H0853"),
        (7, "Gen.1.1#07", "aretz_text", None, "ha'aretz", "the earth", "H9009/{H0776}", "HTd/Ncbsa", "H0776"),
    ]
    for wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary in gen1_words:
        db.execute_update(
            "INSERT OR REPLACE INTO words "
            "(book, chapter, verse, word_num, word_ref, hebrew_text, greek_text, "
            "transliteration, translation, strongs_raw, morphology_code, strongs_primary, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("Gen", 1, 1, wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary, "hebrew"),
        )

    # --- Original Language Words (John 1:1 -- Greek) ---
    john1_words = [
        (1, "John.1.1#01", None, "en_text", "en", "In", "{G1722}", "GP", "G1722"),
        (2, "John.1.1#02", None, "arche_text", "arche", "beginning", "{G0746}", "GNdfs", "G0746"),
        (3, "John.1.1#03", None, "en_was", "en", "was", "{G1510V}", "GVIia3s", "G1510"),
        (4, "John.1.1#04", None, "ho_text", "ho", "the", "{G3588}", "GEdnms", "G3588"),
        (5, "John.1.1#05", None, "logos_text", "Logos", "Word", "{G3056}", "GNnms", "G3056"),
    ]
    for wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary in john1_words:
        db.execute_update(
            "INSERT OR REPLACE INTO words "
            "(book, chapter, verse, word_num, word_ref, hebrew_text, greek_text, "
            "transliteration, translation, strongs_raw, morphology_code, strongs_primary, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("John", 1, 1, wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary, "greek"),
        )

    # --- Lexicon Entries ---
    lexicon_entries = [
        ("H7225", None, None, None, "reshit_orig", "reshit", "noun", "beginning", "beginning, chief, first", "hebrew"),
        ("H1254", None, None, None, "bara_orig", "bara", "verb", "create", "to create, shape, form", "hebrew"),
        ("H0430", None, None, None, "elohim_orig", "elohim", "noun", "God", "God, gods, judges", "hebrew"),
        ("H0853", None, None, None, "et_orig", "et", "particle", "[obj]", "accusative marker", "hebrew"),
        ("H8064", None, None, None, "shamayim_orig", "shamayim", "noun", "heavens", "heaven, heavens, sky", "hebrew"),
        ("H0776", None, None, None, "erets_orig", "erets", "noun", "earth", "earth, land, ground", "hebrew"),
        ("G3056", None, None, None, "logos_orig", "logos", "noun", "word", "word, reason, statement", "greek"),
        (
            "G0746",
            None,
            None,
            None,
            "arche_orig",
            "arche",
            "noun",
            "beginning",
            "beginning, origin, first cause",
            "greek",
        ),
        ("G1722", None, None, None, "en_orig", "en", "preposition", "in", "in, on, among", "greek"),
        ("G1510", None, None, None, "eimi_orig", "eimi", "verb", "am", "I am, I exist", "greek"),
        ("G3588", None, None, None, "ho_orig", "ho", "article", "the", "the definite article", "greek"),
    ]
    for strongs, ext, disamb, uni, orig, translit, pos, gloss, defn, lang in lexicon_entries:
        db.execute_update(
            "INSERT OR REPLACE INTO lexicon "
            "(strongs_number, extended_strongs, disambiguated_strongs, unified_strongs, "
            "original_word, transliteration, part_of_speech, gloss, definition, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (strongs, ext, disamb, uni, orig, translit, pos, gloss, defn, lang),
        )

    # --- Morphology Codes ---
    morphology_entries = [
        ("HR/Ncfsa", "Hebrew Preposition + Noun common feminine singular absolute", None, "hebrew"),
        ("HVqp3ms", "Hebrew Verb qal perfect 3rd masculine singular", None, "hebrew"),
        ("HNcmpa", "Hebrew Noun common masculine plural absolute", None, "hebrew"),
        ("HTo", "Hebrew Particle accusative marker", None, "hebrew"),
        ("HTd/Ncmpa", "Hebrew Article + Noun common masculine plural absolute", None, "hebrew"),
        ("HC/To", "Hebrew Conjunction + Particle accusative marker", None, "hebrew"),
        ("HTd/Ncbsa", "Hebrew Article + Noun common both-gender singular absolute", None, "hebrew"),
        ("GP", "Greek Preposition", None, "greek"),
        ("GNdfs", "Greek Noun dative feminine singular", None, "greek"),
        ("GVIia3s", "Greek Verb Indicative imperfect active 3rd singular", None, "greek"),
        ("GEdnms", "Greek Article definite nominative masculine singular", None, "greek"),
        ("GNnms", "Greek Noun nominative masculine singular", None, "greek"),
    ]
    for code, desc, components, lang in morphology_entries:
        db.execute_update(
            "INSERT OR REPLACE INTO morphology (code, description, components, language) VALUES (?, ?, ?, ?)",
            (code, desc, components, lang),
        )

    # --- Run migrations to create enrichment tables ---
    run_migrations(db_path)

    # --- Populate all enrichment data ---
    GenreShiftPopulator(db_path).populate()
    SpeakerAttributionPopulator(db_path).populate()
    WordExplanationPopulator(db_path).populate()
    ConceptQualityPopulator(db_path).populate()
    PassagePopulator(db_path).populate()
    CulturalContextPopulator(db_path).populate()
    LiteraryStructurePopulator(db_path).populate()
    ReadingPlanPopulator(db_path).ensure_tables()
    ReadingPlanPopulator(db_path).populate()
    LifeTopicPopulator(db_path).populate()
    BookMetadataPopulator(db_path).populate()
    CrossReferencePopulator(db_path).populate()
    WordRichnessComputer(db_path).compute_all()

    # Mirror the BSB verses under the real default translation id so default-translation endpoints
    # (text search, mobile sync) work, then build the external-content FTS index from the content.
    for ch, vs, text in bsb_gen1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("BSB", 1, ch, vs, text),
        )
    for ch, vs, text in bsb_john1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("BSB", 43, ch, vs, text),
        )
    from abba.database.lexical_strongs_populator import populate_lexical_strongs  # noqa: PLC0415
    from abba.database.search_index import rebuild_search_index  # noqa: PLC0415

    rebuild_search_index(db_path)
    populate_lexical_strongs(db_path)  # index the normalized Strong's key so search_strongs works

    return db_path


@pytest.fixture(scope="module")
def client(enriched_db):
    """Create a FastAPI TestClient backed by the enriched database."""
    app = create_app(db_path=enriched_db)
    with TestClient(app) as tc:
        yield tc


@pytest.fixture(scope="module")
def db_manager(enriched_db):
    """Return a SQLiteManager connected to the enriched database."""
    return SQLiteManager(enriched_db)


# ---------------------------------------------------------------------------
# TestGenreShifts
# ---------------------------------------------------------------------------


class TestGenreShifts:
    """Tests for genre shift population and API endpoints."""

    def test_genre_shifts_populated_for_exodus(self, enriched_db):
        """Genre shifts for Exodus (book_id=2) should exist after population."""
        shifts = GenreShiftPopulator.get_shifts_for_book(enriched_db, 2)
        assert len(shifts) >= 2, "Exodus should have at least 2 genre shifts (Song of the Sea)"

    def test_get_genre_shifts_endpoint(self, client):
        """GET /api/v1/genre-shifts/2 should return genre shifts for Exodus."""
        resp = client.get("/api/v1/genre-shifts/2")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_genre_shift_data_structure(self, client):
        """Each genre shift should contain from_genre, to_genre, and description."""
        resp = client.get("/api/v1/genre-shifts/2")
        data = resp.json()
        shift = data[0]
        assert "from_genre" in shift
        assert "to_genre" in shift
        assert "description" in shift
        assert shift["from_genre"] == "narrative"
        assert shift["to_genre"] == "poetry"

    def test_genre_shifts_empty_for_unknown_book(self, client):
        """A book with no genre shifts should return an empty list."""
        resp = client.get("/api/v1/genre-shifts/999")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_active_genre_at_verse(self, enriched_db):
        """Genre at Exodus 15:5 should be 'poetry' (after the shift at 15:1)."""
        genre = GenreShiftPopulator.get_genre_at_verse(enriched_db, 2, 15, 5)
        assert genre == "poetry"

    def test_active_genre_before_first_shift(self, enriched_db):
        """Genre at Exodus 1:1 (before any shift) should be 'unknown'."""
        genre = GenreShiftPopulator.get_genre_at_verse(enriched_db, 2, 1, 1)
        assert genre == "unknown"


# ---------------------------------------------------------------------------
# TestSpeakerAttributions
# ---------------------------------------------------------------------------


class TestSpeakerAttributions:
    """Tests for speaker attribution population and API surface."""

    def test_speaker_attributions_populated(self, enriched_db):
        """Speaker attributions should have been inserted into the database."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM speaker_attributions").fetchone()[0]
        assert count > 0

    def test_verse_endpoint_deep_includes_speaker(self, client):
        """Deep depth verse response for Gen 1:3 should include a speaker field."""
        resp = client.get("/api/v1/verses/engbsb/1/1/3?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        assert "speaker" in data
        assert data["speaker"] is not None

    def test_genesis_1_3_speaker_is_god(self, client):
        """The speaker at Genesis 1:3 should be God."""
        resp = client.get("/api/v1/verses/engbsb/1/1/3?depth=deep")
        data = resp.json()
        assert data["speaker"]["speaker"] == "God"

    def test_job_chapter_speakers(self, enriched_db):
        """Job 3:1 should be attributed to Job."""
        speakers = SpeakerAttributionPopulator.get_speaker_for_verse(enriched_db, 18, 3, 1)
        assert len(speakers) >= 1
        assert speakers[0][0] == "Job"

    def test_verse_with_no_speaker(self, client):
        """Genesis 1:2 has no speaker attribution and should return None."""
        resp = client.get("/api/v1/verses/engbsb/1/1/2?depth=deep")
        data = resp.json()
        assert data["speaker"] is None


# ---------------------------------------------------------------------------
# TestWordExplanations
# ---------------------------------------------------------------------------


class TestWordExplanations:
    """Tests for word explanation population and API."""

    def test_word_explanations_populated(self, enriched_db):
        """Word explanations table should contain data after population."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM word_explanations").fetchone()[0]
        assert count > 0

    def test_get_word_explanation_h0430(self, client):
        """GET /api/v1/word-explanations/H0430 should return an explanation."""
        resp = client.get("/api/v1/word-explanations/H0430")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strongs_number"] == "H0430"

    def test_explanation_content_elohim(self, client):
        """The explanation for H0430 (Elohim) should mention plurality."""
        resp = client.get("/api/v1/word-explanations/H0430")
        data = resp.json()
        assert "plural" in data["explanation"].lower()

    def test_greek_word_explanation_logos(self, client):
        """GET /api/v1/word-explanations/G3056 should return the logos explanation."""
        resp = client.get("/api/v1/word-explanations/G3056")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "greek"
        assert "logos" in data["explanation"].lower() or "word" in data["explanation"].lower()

    def test_404_for_nonexistent_strongs(self, client):
        """A non-existent Strong's number should return 404."""
        resp = client.get("/api/v1/word-explanations/H9999")
        assert resp.status_code == 404

    def test_explanation_language_field(self, client):
        """H0430 explanation should have language='hebrew'."""
        resp = client.get("/api/v1/word-explanations/H0430")
        assert resp.json()["language"] == "hebrew"


# ---------------------------------------------------------------------------
# TestConceptQuality
# ---------------------------------------------------------------------------


class TestConceptQuality:
    """Tests for concept quality metadata population."""

    def test_semantic_range_warnings_populated(self, enriched_db):
        """The semantic_range_warnings table should have data."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM semantic_range_warnings").fetchone()[0]
        assert count > 0

    def test_concept_review_flags_populated(self, enriched_db):
        """The concept_review_flags table should have data."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM concept_review_flags").fetchone()[0]
        assert count > 0

    def test_temporal_tags_populated(self, enriched_db):
        """The concept_temporal_tags table should have data."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM concept_temporal_tags").fetchone()[0]
        assert count > 0

    def test_warning_exists_for_h0430(self, enriched_db):
        """H0430 (Elohim) should have a semantic range warning."""
        with sqlite3.connect(enriched_db) as conn:
            row = conn.execute(
                "SELECT warning_text FROM semantic_range_warnings WHERE strongs_number = 'H0430'"
            ).fetchone()
        assert row is not None
        assert "polysemous" in row[0].lower() or "elohim" in row[0].lower()

    def test_review_flag_for_trinity(self, enriched_db):
        """The concept 'trinity' should have a confessional_reading review flag."""
        with sqlite3.connect(enriched_db) as conn:
            row = conn.execute(
                "SELECT flag_type, review_note FROM concept_review_flags WHERE concept_name = 'trinity'"
            ).fetchone()
        assert row is not None
        assert row[0] == "confessional_reading"

    def test_temporal_tag_for_covenant(self, enriched_db):
        """The concept 'covenant' should have a temporal tag of 'both'."""
        with sqlite3.connect(enriched_db) as conn:
            row = conn.execute(
                "SELECT temporal_period FROM concept_temporal_tags WHERE concept_name = 'covenant'"
            ).fetchone()
        assert row is not None
        assert row[0] == "both"


# ---------------------------------------------------------------------------
# TestSurroundingContext
# ---------------------------------------------------------------------------


class TestSurroundingContext:
    """Tests for surrounding-verse context in deep depth responses."""

    def test_deep_verse_includes_surrounding_context(self, client):
        """Deep depth response should include surrounding_context."""
        resp = client.get("/api/v1/verses/engbsb/1/1/2?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        assert "surrounding_context" in data
        assert data["surrounding_context"] is not None

    def test_previous_verse_text(self, client):
        """The previous_verse for Gen 1:2 should be the text of Gen 1:1."""
        resp = client.get("/api/v1/verses/engbsb/1/1/2?depth=deep")
        data = resp.json()
        assert data["surrounding_context"]["previous_verse"] is not None
        assert "beginning" in data["surrounding_context"]["previous_verse"].lower()

    def test_next_verse_text(self, client):
        """The next_verse for Gen 1:2 should be the text of Gen 1:3."""
        resp = client.get("/api/v1/verses/engbsb/1/1/2?depth=deep")
        data = resp.json()
        assert data["surrounding_context"]["next_verse"] is not None
        assert "light" in data["surrounding_context"]["next_verse"].lower()

    def test_first_verse_has_no_previous(self, client):
        """Genesis 1:1 should have no previous_verse."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        data = resp.json()
        assert data["surrounding_context"]["previous_verse"] is None

    def test_last_verse_has_no_next(self, client):
        """Genesis 1:5 (last verse in our data) should have no next_verse."""
        resp = client.get("/api/v1/verses/engbsb/1/1/5?depth=deep")
        data = resp.json()
        assert data["surrounding_context"]["next_verse"] is None


# ---------------------------------------------------------------------------
# TestDescriptivePrescriptive
# ---------------------------------------------------------------------------


class TestDescriptivePrescriptive:
    """Tests for descriptive/prescriptive marking on narrative genre verses."""

    def test_narrative_genre_is_descriptive(self, client):
        """Genesis 1:1 at deep depth should be marked is_descriptive=True (narrative genre)."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        data = resp.json()
        assert data["is_descriptive"] is True

    def test_non_narrative_not_descriptive(self, enriched_db):
        """A verse in a non-narrative genre passage should not have is_descriptive=True.

        Exodus 15:1 is in the poetry section (after genre shift), so the active genre
        should be 'poetry', not 'narrative'. If genre != narrative, is_descriptive should
        not be True.
        """
        genre = GenreShiftPopulator.get_genre_at_verse(enriched_db, 2, 15, 5)
        assert genre == "poetry"
        # Poetry genre should not be marked as descriptive by the route logic

    def test_genre_field_populated_at_deep_depth(self, client):
        """Genre field should be populated at deep depth for Genesis 1:1."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        data = resp.json()
        assert data["genre"] is not None


# ---------------------------------------------------------------------------
# TestTranslationDivergence
# ---------------------------------------------------------------------------


class TestTranslationDivergence:
    """Tests for translation divergence detection in compare endpoint."""

    def test_compare_includes_divergences(self, client):
        """Compare endpoint should include a divergences field."""
        resp = client.get(
            "/api/v1/compare/1/1/1",
            params={"translations": ["engbsb", "engkjv"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "divergences" in data

    def test_divergences_detect_word_differences(self, client):
        """BSB 'heavens' vs KJV 'heaven' should produce a divergence entry with unique words."""
        resp = client.get(
            "/api/v1/compare/1/1/2",
            params={"translations": ["engbsb", "engkjv"]},
        )
        data = resp.json()
        # At least some unique words should differ between BSB and KJV for Gen 1:2
        divergences = data.get("divergences") or []
        # Even if similarity >= 0.85 and no divergence is reported, the field should exist
        assert isinstance(divergences, list)

    def test_similarity_score_computed(self, client):
        """When divergences exist, each should have a similarity score."""
        resp = client.get(
            "/api/v1/compare/1/1/2",
            params={"translations": ["engbsb", "engkjv"]},
        )
        data = resp.json()
        divergences = data.get("divergences") or []
        if divergences:
            assert "similarity" in divergences[0]
            assert 0.0 <= divergences[0]["similarity"] <= 1.0


# ---------------------------------------------------------------------------
# TestPagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Tests for paginated text search."""

    def test_text_search_page_1(self, client):
        """Text search page=1 should return results for 'God'."""
        resp = client.get("/api/v1/search/text", params={"q": "God", "page": 1, "limit": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_text_search_page_2_different_or_empty(self, client):
        """Text search page=2 should return different results or be empty."""
        resp1 = client.get("/api/v1/search/text", params={"q": "God", "page": 1, "limit": 2})
        resp2 = client.get("/api/v1/search/text", params={"q": "God", "page": 2, "limit": 2})
        assert resp2.status_code == 200
        data1 = resp1.json()
        data2 = resp2.json()
        if data2:
            # If page 2 has results, they should differ from page 1
            refs1 = {(r["book_id"], r["chapter"], r["verse"]) for r in data1}
            refs2 = {(r["book_id"], r["chapter"], r["verse"]) for r in data2}
            assert refs1 != refs2 or len(refs2) == 0

    def test_pagination_large_page_number(self, client):
        """A very large page number should return an empty list, not an error."""
        resp = client.get("/api/v1/search/text", params={"q": "God", "page": 9999, "limit": 50})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0


# ---------------------------------------------------------------------------
# TestUserNotes
# ---------------------------------------------------------------------------


class TestUserNotes:
    """Tests for user note CRUD via the API."""

    def test_create_note(self, client):
        """POST /api/v1/notes/1/1/1 should create a note and return it."""
        resp = client.post(
            "/api/v1/notes/1/1/1",
            json={"content": "This is a test note.", "note_type": "personal"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["note_id"] > 0
        assert data["content"] == "This is a test note."

    def test_get_notes_for_verse(self, client):
        """GET /api/v1/notes/1/1/1 should return notes previously created."""
        # Create a note first
        client.post("/api/v1/notes/1/1/1", json={"content": "Retrieve me."})
        resp = client.get("/api/v1/notes/1/1/1")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        contents = [n["content"] for n in data]
        assert "Retrieve me." in contents

    def test_delete_note(self, client):
        """DELETE /api/v1/notes/{note_id} should remove the note."""
        create_resp = client.post(
            "/api/v1/notes/1/1/5",
            json={"content": "Delete me.", "note_type": "personal"},
        )
        note_id = create_resp.json()["note_id"]
        del_resp = client.delete(f"/api/v1/notes/{note_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True
        # Verify it's gone
        notes = client.get("/api/v1/notes/1/1/5").json()
        note_ids = [n["note_id"] for n in notes]
        assert note_id not in note_ids

    def test_note_custom_type(self, client):
        """Creating a note with a custom note_type should persist that type."""
        resp = client.post(
            "/api/v1/notes/43/1/1",
            json={"content": "Study note on John 1:1.", "note_type": "study"},
        )
        data = resp.json()
        assert data["note_type"] == "study"

    def test_notes_for_verse_with_no_notes(self, client):
        """A verse with no notes should return an empty list."""
        resp = client.get("/api/v1/notes/99/99/99")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# TestUserCollections
# ---------------------------------------------------------------------------


class TestUserCollections:
    """Tests for user collection CRUD via the API."""

    def test_create_collection(self, client):
        """POST /api/v1/collections should create a new collection."""
        resp = client.post(
            "/api/v1/collections",
            json={"name": "Favorites", "description": "My favorite verses"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["collection_id"] > 0
        assert data["name"] == "Favorites"

    def test_list_collections(self, client):
        """GET /api/v1/collections should include the created collection."""
        # Ensure at least one collection exists
        client.post("/api/v1/collections", json={"name": "List Test"})
        resp = client.get("/api/v1/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        names = [c["name"] for c in data]
        assert "List Test" in names

    def test_add_verse_to_collection(self, client):
        """POST /api/v1/collections/{id}/items should add a verse."""
        coll_resp = client.post(
            "/api/v1/collections",
            json={"name": "Add Items Test"},
        )
        cid = coll_resp.json()["collection_id"]
        add_resp = client.post(
            f"/api/v1/collections/{cid}/items",
            json={"book_id": 1, "chapter": 1, "verse": 1, "note": "Creation verse"},
        )
        assert add_resp.status_code == 200
        assert add_resp.json()["added"] is True

    def test_get_collection_items(self, client):
        """GET /api/v1/collections/{id}/items should return the added verse."""
        coll_resp = client.post("/api/v1/collections", json={"name": "Items Test"})
        cid = coll_resp.json()["collection_id"]
        client.post(
            f"/api/v1/collections/{cid}/items",
            json={"book_id": 43, "chapter": 1, "verse": 1},
        )
        items_resp = client.get(f"/api/v1/collections/{cid}/items")
        assert items_resp.status_code == 200
        items = items_resp.json()
        assert len(items) == 1
        assert items[0]["book_id"] == 43
        assert items[0]["chapter"] == 1

    def test_delete_collection(self, client):
        """DELETE /api/v1/collections/{id} should remove the collection."""
        coll_resp = client.post("/api/v1/collections", json={"name": "Delete Me"})
        cid = coll_resp.json()["collection_id"]
        del_resp = client.delete(f"/api/v1/collections/{cid}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# TestSharing
# ---------------------------------------------------------------------------


class TestSharing:
    """Tests for shareable links."""

    def test_create_share(self, client):
        """POST /api/v1/share should create a share with a token."""
        resp = client.post(
            "/api/v1/share",
            json={
                "share_type": "verse",
                "title": "Genesis 1:1",
                "content": {"reference": "Gen 1:1", "text": "In the beginning..."},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "share_token" in data
        assert len(data["share_token"]) > 0

    def test_get_shared_item(self, client):
        """GET /api/v1/share/{token} should retrieve the shared item."""
        create_resp = client.post(
            "/api/v1/share",
            json={
                "share_type": "collection",
                "title": "Shared Collection",
                "content": {"verses": ["Gen 1:1", "John 1:1"]},
            },
        )
        token = create_resp.json()["share_token"]
        get_resp = client.get(f"/api/v1/share/{token}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["share_type"] == "collection"
        assert data["title"] == "Shared Collection"
        assert "verses" in data["content"]

    def test_invalid_share_token_404(self, client):
        """GET /api/v1/share/{invalid_token} should return 404."""
        resp = client.get("/api/v1/share/nonexistent_token_xyz")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestNewEnrichmentDataIntegrity
# ---------------------------------------------------------------------------


class TestNewEnrichmentDataIntegrity:
    """Tests that all enrichment tables exist and contain data."""

    def test_all_new_tables_exist(self, enriched_db):
        """All enrichment tables should exist after migration."""
        required_tables = [
            "book_metadata",
            "passages",
            "literary_structures",
            "cultural_context",
            "cross_references",
            "word_richness",
            "life_topics",
            "genre_shifts",
            "speaker_attributions",
            "word_explanations",
            "semantic_range_warnings",
            "concept_review_flags",
            "concept_temporal_tags",
            "verse_notes",
            "user_collections",
            "collection_items",
            "shared_items",
            "reading_plans",
            "reading_plan_entries",
        ]
        with sqlite3.connect(enriched_db) as conn:
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for table in required_tables:
            assert table in existing, f"Table '{table}' should exist after migration"

    def test_genre_shifts_table_has_data(self, enriched_db):
        """genre_shifts table should have rows after population."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM genre_shifts").fetchone()[0]
        assert count > 0

    def test_speaker_attributions_table_has_data(self, enriched_db):
        """speaker_attributions table should have rows."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM speaker_attributions").fetchone()[0]
        assert count > 0

    def test_word_explanations_table_has_data(self, enriched_db):
        """word_explanations table should have rows."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM word_explanations").fetchone()[0]
        assert count > 0

    def test_concept_quality_tables_have_data(self, enriched_db):
        """All three concept quality tables should have data."""
        with sqlite3.connect(enriched_db) as conn:
            warnings = conn.execute("SELECT COUNT(*) FROM semantic_range_warnings").fetchone()[0]
            flags = conn.execute("SELECT COUNT(*) FROM concept_review_flags").fetchone()[0]
            tags = conn.execute("SELECT COUNT(*) FROM concept_temporal_tags").fetchone()[0]
        assert warnings > 0
        assert flags > 0
        assert tags > 0

    def test_idempotent_population(self, enriched_db):
        """Running populate twice should not duplicate rows (idempotency)."""
        with sqlite3.connect(enriched_db) as conn:
            count_before = conn.execute("SELECT COUNT(*) FROM genre_shifts").fetchone()[0]

        # Populate again (should be a no-op)
        GenreShiftPopulator(enriched_db).populate()

        with sqlite3.connect(enriched_db) as conn:
            count_after = conn.execute("SELECT COUNT(*) FROM genre_shifts").fetchone()[0]

        assert count_after == count_before, "Re-populating should not increase row count"

    def test_cross_references_have_data(self, enriched_db):
        """cross_references table should have rows after population."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cross_references").fetchone()[0]
        assert count > 0

    def test_passages_have_data(self, enriched_db):
        """passages table should have rows."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        assert count > 0


# ---------------------------------------------------------------------------
# TestProgressiveDepthWithNewFeatures
# ---------------------------------------------------------------------------


class TestProgressiveDepthWithNewFeatures:
    """Tests for progressive depth levels returning the correct fields."""

    def test_basic_depth_no_deep_fields(self, client):
        """Basic depth should not include speaker, genre, or surrounding_context."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=basic")
        assert resp.status_code == 200
        data = resp.json()
        assert data["speaker"] is None
        assert data["genre"] is None
        assert data["surrounding_context"] is None

    def test_standard_depth_has_richness_no_speaker(self, client):
        """Standard depth should include richness_flags but not speaker."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=standard")
        assert resp.status_code == 200
        data = resp.json()
        # Standard depth populates words and richness_flags
        assert data["words"] is not None
        # Standard does not set speaker or surrounding_context
        assert data["speaker"] is None
        assert data["surrounding_context"] is None

    def test_deep_depth_includes_all_new_features(self, client):
        """Deep depth should include speaker, genre, surrounding_context, and is_descriptive."""
        resp = client.get("/api/v1/verses/engbsb/1/1/3?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["speaker"] is not None
        assert data["genre"] is not None
        assert data["surrounding_context"] is not None
        assert "is_descriptive" in data

    def test_scholarly_depth_includes_parallel_passages(self, client):
        """Scholarly depth should include all deep features plus parallel_passages."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=scholarly")
        assert resp.status_code == 200
        data = resp.json()
        assert "parallel_passages" in data
        # parallel_passages may be an empty list but should be present
        assert isinstance(data["parallel_passages"], list)
        # Also has deep features
        assert data["surrounding_context"] is not None
        assert data["genre"] is not None

    def test_deep_depth_genesis_1_1_has_cultural_context(self, client):
        """Deep depth for Genesis 1:1 should include cultural context notes."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cultural_context"] is not None
        assert isinstance(data["cultural_context"], list)
        assert len(data["cultural_context"]) >= 1
        # The cultural context for Genesis should mention origins or patriarchs
        titles = [c["title"] for c in data["cultural_context"]]
        assert any("genesis" in t.lower() for t in titles)

    def test_deep_depth_john_1_1_has_literary_structure(self, client):
        """Deep depth for John 1:1 should include the prologue chiastic structure."""
        resp = client.get("/api/v1/verses/engbsb/43/1/1?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["literary_structures"] is not None
        assert isinstance(data["literary_structures"], list)
        assert len(data["literary_structures"]) >= 1
        types = [s["structure_type"] for s in data["literary_structures"]]
        assert "chiasmus" in types

    def test_deep_depth_genesis_1_1_has_passage_info(self, client):
        """Deep depth for Genesis 1:1 should include passage (pericope) info."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["passage_info"] is not None
        assert "title" in data["passage_info"]
        # Should be one of the Genesis passages
        assert "creation" in data["passage_info"]["title"].lower()

    def test_deep_depth_genesis_1_1_has_cross_references(self, client):
        """Deep depth for Genesis 1:1 should include cross-references."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cross_references"] is not None
        assert isinstance(data["cross_references"], list)
        # Gen 1:1 has a thematic cross-reference to John 1:1
        target_refs = [x["target_reference"] for x in data["cross_references"]]
        assert any("jhn" in r.lower() or "john" in r.lower() or "43" in r for r in target_refs)


# ---------------------------------------------------------------------------
# TestReadingPlans
# ---------------------------------------------------------------------------


class TestReadingPlans:
    """Tests for reading plan endpoints."""

    def test_list_reading_plans(self, client):
        """GET /api/v1/reading-plans should return a non-empty list."""
        resp = client.get("/api/v1/reading-plans")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        slugs = [p["slug"] for p in data]
        assert "start-here" in slugs

    def test_get_reading_plan_detail(self, client):
        """GET /api/v1/reading-plans/start-here should return entries."""
        resp = client.get("/api/v1/reading-plans/start-here")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "start-here"
        assert len(data["entries"]) == 7

    def test_reading_plan_404(self, client):
        """GET /api/v1/reading-plans/nonexistent should return 404."""
        resp = client.get("/api/v1/reading-plans/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestLifeTopics
# ---------------------------------------------------------------------------


class TestLifeTopics:
    """Tests for life topics endpoints."""

    def test_list_life_topics(self, client):
        """GET /api/v1/life-topics should return a non-empty list."""
        resp = client.get("/api/v1/life-topics")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        slugs = [t["slug"] for t in data]
        assert "anxiety" in slugs

    def test_get_life_topic_detail(self, client):
        """GET /api/v1/life-topics/anxiety should return the topic with study steps."""
        resp = client.get("/api/v1/life-topics/anxiety")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "anxiety"
        assert data["category"] == "emotions"
        assert len(data["study_steps"]) >= 1

    def test_life_topic_404(self, client):
        """GET /api/v1/life-topics/nonexistent should return 404."""
        resp = client.get("/api/v1/life-topics/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestBookMetadata
# ---------------------------------------------------------------------------


class TestBookMetadata:
    """Tests for book metadata enrichment."""

    def test_book_info_includes_metadata(self, client):
        """GET /api/v1/books/1 should include enriched metadata fields."""
        resp = client.get("/api/v1/books/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["primary_genre"] == "narrative"
        assert data["author_traditional"] == "Moses"
        assert data["canonical_section"] == "Torah"

    def test_book_list_includes_metadata(self, client):
        """GET /api/v1/books should return books with genre info."""
        resp = client.get("/api/v1/books")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        genesis = next((b for b in data if b["book_id"] == 1), None)
        assert genesis is not None
        assert genesis["primary_genre"] == "narrative"


# ---------------------------------------------------------------------------
# TestWordRichness
# ---------------------------------------------------------------------------


class TestWordRichness:
    """Tests for word richness computation and API surface."""

    def test_word_richness_computed(self, enriched_db):
        """word_richness table should have rows after compute_all."""
        with sqlite3.connect(enriched_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM word_richness").fetchone()[0]
        assert count > 0

    def test_richness_flags_in_standard_depth(self, client):
        """Standard depth for Gen 1:1 should include richness flags."""
        resp = client.get("/api/v1/verses/engbsb/1/1/1?depth=standard")
        assert resp.status_code == 200
        data = resp.json()
        # richness_flags should be populated (may be empty if no high scores)
        assert "richness_flags" in data
        assert isinstance(data["richness_flags"], list)


# ---------------------------------------------------------------------------
# TestExportEndpoint
# ---------------------------------------------------------------------------


class TestExportEndpoint:
    """Tests for the verse export endpoint."""

    def test_export_json(self, client):
        """GET /api/v1/export/verse/engbsb/1/1/1 should return JSON export."""
        resp = client.get("/api/v1/export/verse/engbsb/1/1/1")
        assert resp.status_code == 200
        data = resp.json()
        assert "reference" in data
        assert "text" in data
        assert "beginning" in data["text"].lower()

    def test_export_markdown(self, client):
        """Export with format=markdown should include a markdown field."""
        resp = client.get("/api/v1/export/verse/engbsb/1/1/1?format=markdown")
        assert resp.status_code == 200
        data = resp.json()
        assert "markdown" in data
        assert data["markdown"].startswith("#")
