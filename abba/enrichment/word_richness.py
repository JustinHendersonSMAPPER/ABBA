"""Word richness computation for meaning-loss detection."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class WordRichnessComputer:
    """Computes meaning-richness scores by comparing lexicon glosses to full definitions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def compute_all(self, force: bool = False) -> int:
        """Compute richness scores for all words in stepbible_verses that have lexicon entries.

        Args:
            force: If True, replace existing scores.

        Returns:
            Number of richness rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Ensure table exists
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='word_richness'")
            if cursor.fetchone()[0] == 0:
                logger.warning("word_richness table does not exist; run migrations first")
                return 0

            if not force:
                cursor.execute("SELECT COUNT(*) FROM word_richness")
                existing = cursor.fetchone()[0]
                if existing > 0:
                    logger.info("word_richness already has %d rows; use force=True to recompute", existing)
                    return 0

            if force:
                cursor.execute("DELETE FROM word_richness")

            # Get unique word occurrences with lexicon data
            cursor.execute(
                """
                SELECT DISTINCT
                    w.book, w.chapter, w.verse, w.word_num,
                    w.strongs_primary,
                    l.gloss, l.definition, l.part_of_speech
                FROM words w
                JOIN lexicon l ON w.strongs_primary = l.strongs_number
                WHERE w.strongs_primary IS NOT NULL
                  AND w.strongs_primary != ''
                  AND l.definition IS NOT NULL
                  AND l.definition != ''
                ORDER BY w.book, w.chapter, w.verse, w.word_num
                """
            )
            rows = cursor.fetchall()
            logger.info("Computing richness for %d word occurrences", len(rows))

            batch: List[tuple] = []
            for row in rows:
                score_data = self._compute_single(
                    gloss=row["gloss"] or "",
                    definition=row["definition"] or "",
                    part_of_speech=row["part_of_speech"] or "",
                )

                batch.append(
                    (
                        row["book"],
                        row["chapter"],
                        row["verse"],
                        row["word_num"],
                        row["strongs_primary"],
                        score_data["gloss_coverage"],
                        score_data["morphology_significance"],
                        json.dumps(score_data["untranslatable_nuances"]),
                        score_data.get("cultural_significance"),
                        score_data["richness_score"],
                    )
                )

                if len(batch) >= 500:
                    cursor.executemany(
                        "INSERT INTO word_richness "
                        "(book, chapter, verse, word_num, strongs_number, "
                        "gloss_coverage, morphology_significance, untranslatable_nuances, "
                        "cultural_significance, richness_score) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    inserted += len(batch)
                    batch.clear()

            if batch:
                cursor.executemany(
                    "INSERT INTO word_richness "
                    "(book, chapter, verse, word_num, strongs_number, "
                    "gloss_coverage, morphology_significance, untranslatable_nuances, "
                    "cultural_significance, richness_score) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                inserted += len(batch)

            conn.commit()

        logger.info("Computed word richness: %d rows inserted", inserted)
        return inserted

    @staticmethod
    def _compute_single(gloss: str, definition: str, part_of_speech: str) -> Dict[str, Any]:
        """Compute richness for a single word.

        The richness score indicates how much meaning is lost when translating
        to a single English gloss. Higher score = more meaning lost.

        Args:
            gloss: Short English gloss (e.g., "beginning").
            definition: Full lexicon definition (e.g., "beginning, chief, first, choicest").
            part_of_speech: Part of speech.

        Returns:
            Dictionary with richness metrics.
        """
        gloss_words = set(gloss.lower().split()) if gloss else set()
        def_words = set(definition.lower().replace(",", " ").replace(";", " ").split()) if definition else set()

        # Remove trivial words
        stop_words = {"a", "an", "the", "of", "to", "in", "for", "and", "or", "is", "it", "be", "that", "with"}
        gloss_words -= stop_words
        def_words -= stop_words

        if not def_words:
            return {
                "gloss_coverage": 1.0,
                "morphology_significance": None,
                "untranslatable_nuances": [],
                "richness_score": 0.0,
            }

        # Gloss coverage: what fraction of the definition is captured
        if gloss_words:
            coverage = len(gloss_words & def_words) / max(len(def_words), 1)
        else:
            coverage = 0.0

        # Richness score: inverse of coverage, scaled
        richness_score = round(1.0 - min(coverage, 1.0), 3)

        # Find nuances lost
        nuances = sorted(def_words - gloss_words)[:5]  # Top 5 meaning elements not in gloss

        # Morphology significance for verbs (tense/mood carries meaning)
        morph_sig = None
        if part_of_speech and part_of_speech.lower() in ("verb",):
            morph_sig = "Verb tense, mood, and voice carry meaning beyond the English gloss"

        return {
            "gloss_coverage": round(coverage, 3),
            "morphology_significance": morph_sig,
            "untranslatable_nuances": nuances,
            "richness_score": richness_score,
        }
