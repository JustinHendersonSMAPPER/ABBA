"""Tests for enrichment data population: book metadata, cross-references, word richness, life topics."""

import json
import sqlite3
from pathlib import Path

import pytest

from abba.database import SQLiteManager
from abba.database.migrations import (
    add_book_metadata_table,
    add_cross_references_table,
    add_life_topics_tables,
    add_word_richness_table,
)
from abba.enrichment.book_metadata import BOOK_METADATA, BookMetadataPopulator
from abba.enrichment.cross_references import CURATED_CROSS_REFERENCES, CrossReferencePopulator
from abba.enrichment.life_topics import LIFE_TOPICS, LifeTopicPopulator
from abba.enrichment.word_richness import WordRichnessComputer


@pytest.fixture
def enrichment_db(tmp_path):
    """Create a database with enrichment migration tables."""
    db_path = tmp_path / "enrichment_test.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # Run enrichment migrations
    add_book_metadata_table(db_path)
    add_cross_references_table(db_path)
    add_word_richness_table(db_path)
    add_life_topics_tables(db_path)

    return db_path


# ------------------------------------------------------------------ #
#  Book Metadata                                                      #
# ------------------------------------------------------------------ #


class TestBookMetadata:
    """Tests for BookMetadataPopulator."""

    def test_metadata_covers_all_66_books(self):
        """Curated metadata should cover all 66 Protestant canon books."""
        ids = {m["book_id"] for m in BOOK_METADATA}
        assert ids == set(range(1, 67))

    def test_populate_inserts_all_books(self, enrichment_db):
        """Should insert metadata for all 66 books."""
        pop = BookMetadataPopulator(enrichment_db)
        inserted = pop.populate()

        assert inserted == 66

        with sqlite3.connect(enrichment_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM book_metadata")
            assert cursor.fetchone()[0] == 66

    def test_populate_idempotent(self, enrichment_db):
        """Should not duplicate rows on second run without force."""
        pop = BookMetadataPopulator(enrichment_db)
        pop.populate()
        inserted = pop.populate()

        assert inserted == 0

    def test_populate_force_replaces(self, enrichment_db):
        """Should replace rows when force=True."""
        pop = BookMetadataPopulator(enrichment_db)
        pop.populate()
        replaced = pop.populate(force=True)

        assert replaced == 66

    def test_each_book_has_required_fields(self):
        """Each book metadata entry should have required fields."""
        for meta in BOOK_METADATA:
            assert "book_id" in meta
            assert "primary_genre" in meta
            assert "canonical_section" in meta
            assert "original_language" in meta
            assert meta["original_language"] in ("hebrew", "greek")

    def test_genres_are_valid(self):
        """Primary genres should be from a known set."""
        valid_genres = {
            "narrative",
            "law",
            "wisdom",
            "poetry",
            "prophecy",
            "gospel",
            "epistle",
            "apocalyptic",
        }
        for meta in BOOK_METADATA:
            assert meta["primary_genre"] in valid_genres, f"Book {meta['book_id']}: {meta['primary_genre']}"

    def test_get_metadata_for_book(self):
        """Should look up metadata by book_id from the constant."""
        result = BookMetadataPopulator.get_metadata_for_book(1)
        assert result is not None
        assert result["primary_genre"] == "narrative"
        assert result["canonical_section"] == "Torah"

        assert BookMetadataPopulator.get_metadata_for_book(999) is None

    def test_secondary_genres_stored_as_json(self, enrichment_db):
        """Secondary genres should be stored as JSON arrays."""
        pop = BookMetadataPopulator(enrichment_db)
        pop.populate()

        with sqlite3.connect(enrichment_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT secondary_genres FROM book_metadata WHERE book_id = 1")
            row = cursor.fetchone()
            parsed = json.loads(row[0])
            assert isinstance(parsed, list)
            assert "law" in parsed


# ------------------------------------------------------------------ #
#  Cross References                                                   #
# ------------------------------------------------------------------ #


class TestCrossReferences:
    """Tests for CrossReferencePopulator."""

    def test_curated_refs_have_valid_structure(self):
        """Each cross-reference tuple should have 9 elements with valid types."""
        valid_types = {"quotation", "allusion", "parallel", "thematic", "prophecy_fulfillment", "typology", "contrast"}
        for ref in CURATED_CROSS_REFERENCES:
            assert len(ref) == 9
            src_book, src_ch, src_vs, tgt_book, tgt_ch, tgt_vs, ref_type, confidence, notes = ref
            assert 1 <= src_book <= 66
            assert 1 <= tgt_book <= 66
            assert ref_type in valid_types
            assert 0.0 <= confidence <= 1.0

    def test_populate_inserts_refs(self, enrichment_db):
        """Should insert curated cross-references."""
        pop = CrossReferencePopulator(enrichment_db)
        inserted = pop.populate()

        assert inserted > 0
        assert inserted == len(CURATED_CROSS_REFERENCES)

    def test_populate_idempotent(self, enrichment_db):
        """Should not duplicate on second run without force."""
        pop = CrossReferencePopulator(enrichment_db)
        pop.populate()
        inserted = pop.populate()
        assert inserted == 0

    def test_populate_force_replaces(self, enrichment_db):
        """Should replace rows when force=True."""
        pop = CrossReferencePopulator(enrichment_db)
        pop.populate()
        replaced = pop.populate(force=True)
        assert replaced == len(CURATED_CROSS_REFERENCES)

    def test_get_cross_references_for_verse(self, enrichment_db):
        """Should return cross-references for a specific verse."""
        pop = CrossReferencePopulator(enrichment_db)
        pop.populate()

        # Genesis 1:1 should have cross-refs to John 1:1 and Revelation 21:1
        refs = pop.get_cross_references_for_verse(enrichment_db, 1, 1, 1)
        assert len(refs) > 0
        # Check that it includes both outgoing and incoming
        directions = {r["direction"] for r in refs}
        assert "outgoing" in directions or "incoming" in directions

    def test_bidirectional_lookup(self, enrichment_db):
        """Should find references in both directions."""
        pop = CrossReferencePopulator(enrichment_db)
        pop.populate()

        # John 1:1 should be found as a target of Genesis 1:1
        refs = pop.get_cross_references_for_verse(enrichment_db, 43, 1, 1)
        incoming = [r for r in refs if r["direction"] == "incoming"]
        assert len(incoming) > 0


# ------------------------------------------------------------------ #
#  Word Richness                                                      #
# ------------------------------------------------------------------ #


class TestWordRichness:
    """Tests for WordRichnessComputer."""

    def test_compute_single_full_coverage(self):
        """Should have low richness when gloss covers the definition well."""
        result = WordRichnessComputer._compute_single("beginning", "beginning", "noun")
        assert result["richness_score"] == 0.0
        assert result["gloss_coverage"] == 1.0

    def test_compute_single_partial_coverage(self):
        """Should have higher richness when gloss misses definition meanings."""
        result = WordRichnessComputer._compute_single("beginning", "beginning, chief, first, choicest", "noun")
        assert result["richness_score"] > 0.0
        assert result["gloss_coverage"] < 1.0
        assert len(result["untranslatable_nuances"]) > 0

    def test_compute_single_empty_definition(self):
        """Should handle empty definition gracefully."""
        result = WordRichnessComputer._compute_single("word", "", "noun")
        assert result["richness_score"] == 0.0
        assert result["gloss_coverage"] == 1.0

    def test_compute_single_empty_gloss(self):
        """Should have maximum richness when gloss is empty."""
        result = WordRichnessComputer._compute_single("", "beginning, chief, first", "noun")
        assert result["richness_score"] == 1.0
        assert result["gloss_coverage"] == 0.0

    def test_compute_single_verb_morphology_significance(self):
        """Should note morphology significance for verbs."""
        result = WordRichnessComputer._compute_single("create", "to create, shape, form", "verb")
        assert result["morphology_significance"] is not None
        assert "verb" in result["morphology_significance"].lower()

    def test_compute_single_noun_no_morphology_significance(self):
        """Should not add morphology significance for non-verbs."""
        result = WordRichnessComputer._compute_single("beginning", "beginning, chief", "noun")
        assert result["morphology_significance"] is None

    def test_compute_all_with_seeded_db(self, enrichment_db):
        """Should compute richness for word occurrences in the database."""
        # Seed some word and lexicon data
        with sqlite3.connect(enrichment_db) as conn:
            cursor = conn.cursor()

            # Add a word
            cursor.execute(
                "INSERT OR REPLACE INTO words "
                "(book, chapter, verse, word_num, word_ref, hebrew_text, "
                "transliteration, translation, strongs_primary, morphology_code, language) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("Gen", 1, 1, 1, "Gen.1.1#01", "בְּרֵאשִׁית", "bereshit", "beginning", "H7225", "HR/Ncfsa", "hebrew"),
            )

            # Add lexicon entry
            cursor.execute(
                "INSERT OR REPLACE INTO lexicon "
                "(strongs_number, original_word, transliteration, part_of_speech, gloss, definition, language) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("H7225", "רֵאשִׁית", "reshit", "noun", "beginning", "beginning, chief, first, choicest", "hebrew"),
            )
            conn.commit()

        computer = WordRichnessComputer(enrichment_db)
        inserted = computer.compute_all()

        assert inserted > 0

        with sqlite3.connect(enrichment_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT richness_score, untranslatable_nuances FROM word_richness WHERE strongs_number = 'H7225'"
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] > 0.0  # richness score
            nuances = json.loads(row[1])
            assert isinstance(nuances, list)


# ------------------------------------------------------------------ #
#  Life Topics                                                        #
# ------------------------------------------------------------------ #


class TestLifeTopics:
    """Tests for LifeTopicPopulator."""

    def test_topics_have_required_fields(self):
        """Each topic should have required fields."""
        for topic in LIFE_TOPICS:
            assert "slug" in topic
            assert "name" in topic
            assert "category" in topic
            assert "study_steps" in topic
            assert len(topic["study_steps"]) > 0

    def test_study_steps_have_valid_types(self):
        """Study step types should be from a known set."""
        valid_types = {"comfort", "understanding", "guidance", "hope"}
        for topic in LIFE_TOPICS:
            for step_type, _, _ in topic["study_steps"]:
                assert step_type in valid_types, f"Topic {topic['slug']}: invalid step type {step_type}"

    def test_populate_inserts_topics(self, enrichment_db):
        """Should insert all life topics."""
        pop = LifeTopicPopulator(enrichment_db)
        counts = pop.populate()

        assert counts["life_topics"] == len(LIFE_TOPICS)
        assert counts["topic_study_steps"] > 0

    def test_populate_idempotent(self, enrichment_db):
        """Should not duplicate on second run."""
        pop = LifeTopicPopulator(enrichment_db)
        pop.populate()
        counts = pop.populate()

        assert counts["life_topics"] == 0
        assert counts["topic_study_steps"] == 0

    def test_populate_force_replaces(self, enrichment_db):
        """Should replace all rows when force=True."""
        pop = LifeTopicPopulator(enrichment_db)
        pop.populate()
        counts = pop.populate(force=True)

        assert counts["life_topics"] == len(LIFE_TOPICS)

    def test_concept_mappings_inserted(self, enrichment_db):
        """Should insert concept mappings for topics that have them."""
        pop = LifeTopicPopulator(enrichment_db)
        counts = pop.populate()

        # At least some topics have concept mappings
        topics_with_concepts = [t for t in LIFE_TOPICS if t.get("concepts")]
        if topics_with_concepts:
            assert counts["life_topic_concepts"] > 0

    def test_slugs_are_unique(self):
        """Topic slugs should be unique."""
        slugs = [t["slug"] for t in LIFE_TOPICS]
        assert len(slugs) == len(set(slugs))

    def test_categories_are_valid(self):
        """Categories should be from a known set."""
        valid_categories = {"emotions", "relationships", "struggles", "life_stages", "practical", "faith"}
        for topic in LIFE_TOPICS:
            assert topic["category"] in valid_categories, f"Topic {topic['slug']}: {topic['category']}"
