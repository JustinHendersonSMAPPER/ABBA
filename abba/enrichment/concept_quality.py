"""Concept definition quality review metadata.

Adds temporal tags, semantic range warnings, and review flags
to improve concept mapping accuracy.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (strongs_number, warning_text, frequency_note)
POLYSEMOUS_WARNINGS: List[Tuple[str, str, str]] = [
    (
        "H0430",
        "Elohim is highly polysemous: God, gods, angels, judges, mighty ones. "
        "Context determines referent. Searches will include all senses.",
        "Used ~2,600x — careful filtering required",
    ),
    (
        "H7307",
        "Ruach means spirit, wind, or breath depending on context. "
        "Mapping to a single concept (e.g., 'Holy Spirit') risks over-matching.",
        "Used ~378x across very different meanings",
    ),
    (
        "H6213",
        "Asah (do/make) is one of the most common Hebrew verbs. "
        "Matching on this Strong's alone will return thousands of irrelevant results.",
        "Used ~2,627x — almost always too broad as a filter",
    ),
    (
        "H1697",
        "Davar means word, thing, matter, or event. Its semantic range spans communication and concrete reality.",
        "Used ~1,439x — context-dependent meaning",
    ),
    (
        "H5414",
        "Natan (give) covers giving, placing, setting, appointing. "
        "Too broad for most concept searches without additional filtering.",
        "Used ~2,010x — high over-match risk",
    ),
    (
        "G3056",
        "Logos means word, reason, account, speech, or message. "
        "In John's Gospel it has a specialized theological meaning distinct from general usage.",
        "Used ~330x — meaning varies greatly by context",
    ),
    (
        "G2889",
        "Kosmos (world) can mean the physical creation, human society, "
        "or the world system opposed to God. John uses all three senses.",
        "Used ~186x — three distinct semantic domains",
    ),
    (
        "G4151",
        "Pneuma (spirit) refers to the Holy Spirit, human spirit, attitudes, "
        "winds, or demons. Careful contextual analysis required.",
        "Used ~379x — multiple distinct referents",
    ),
    (
        "G4100",
        "Pisteuo (believe) ranges from intellectual assent to committed trust. "
        "Different NT authors weight these nuances differently.",
        "Used ~241x — meaning spans a spectrum",
    ),
    (
        "G2316",
        "Theos (God) can refer to the Father, the Son, pagan deities, "
        "or be used in quotations from non-believers. Context is essential.",
        "Used ~1,317x — requires referent disambiguation",
    ),
]

# Concept review flags for theological accuracy
CONCEPT_REVIEW_FLAGS: List[Tuple[str, str, str]] = [
    (
        "trinity",
        "confessional_reading",
        "The Trinity is a post-biblical systematic theology construct. "
        "While trinitarian themes appear in Scripture, no single verse "
        "explicitly teaches the doctrine. Search results should note this.",
    ),
    (
        "original_sin",
        "confessional_reading",
        "The doctrine of original sin was systematized by Augustine. "
        "The relevant passages (Gen 3, Rom 5) describe human sinfulness "
        "but the specific doctrine involves interpretive tradition.",
    ),
    (
        "rapture",
        "confessional_reading",
        "The concept of a separate 'rapture' event is a specific "
        "eschatological interpretation, not a direct biblical term. "
        "1 Thess 4:17 uses 'caught up' in a broader context.",
    ),
    (
        "free_will",
        "confessional_reading",
        "Free will vs. sovereignty is a long-standing theological debate. "
        "Passages can be cited for multiple positions. Results should "
        "present the range of biblical evidence.",
    ),
]

# Temporal tags for concepts
TEMPORAL_TAGS: List[Tuple[str, str, str]] = [
    ("covenant", "both", "Covenant theology spans both testaments but evolves in form"),
    ("sacrifice", "both", "OT animal sacrifice → NT concept of Christ's sacrifice"),
    ("temple", "both", "OT physical temple → NT metaphorical and eschatological temple"),
    ("law", "ot_primary", "Torah law is OT-rooted; NT reinterprets its role"),
    ("grace", "nt_primary", "Charis as a dominant concept is distinctly NT emphasis"),
    ("justification", "nt_primary", "Pauline development of OT tsadaq concept"),
    ("kingdom_of_god", "both", "OT kingship of YHWH → NT kingdom proclamation by Jesus"),
    ("resurrection", "both", "OT hints → NT centerpiece of faith"),
    ("holy_spirit", "both", "OT ruach → NT pneuma with fuller personal characterization"),
    ("atonement", "both", "OT kaphar → NT hilasmos/hilasterion with christological focus"),
]


class ConceptQualityPopulator:
    """Populates concept quality review metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert all concept quality metadata."""
        total = 0
        total += self._populate_warnings(force)
        total += self._populate_review_flags(force)
        total += self._populate_temporal_tags(force)
        return total

    def _populate_warnings(self, force: bool) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if force:
                cursor.execute("DELETE FROM semantic_range_warnings")
            count = 0
            for strongs, warning, freq_note in POLYSEMOUS_WARNINGS:
                cursor.execute(
                    "SELECT COUNT(*) FROM semantic_range_warnings WHERE strongs_number = ?",
                    (strongs,),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO semantic_range_warnings (strongs_number, warning_text, frequency_note) "
                        "VALUES (?, ?, ?)",
                        (strongs, warning, freq_note),
                    )
                    count += 1
            conn.commit()
        logger.info("Populated %d semantic range warnings", count)
        return count

    def _populate_review_flags(self, force: bool) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if force:
                cursor.execute("DELETE FROM concept_review_flags")
            count = 0
            for concept, flag_type, note in CONCEPT_REVIEW_FLAGS:
                cursor.execute(
                    "SELECT COUNT(*) FROM concept_review_flags WHERE concept_name = ?",
                    (concept,),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO concept_review_flags (concept_name, flag_type, review_note) VALUES (?, ?, ?)",
                        (concept, flag_type, note),
                    )
                    count += 1
            conn.commit()
        logger.info("Populated %d concept review flags", count)
        return count

    def _populate_temporal_tags(self, force: bool) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if force:
                cursor.execute("DELETE FROM concept_temporal_tags")
            count = 0
            for concept, period, note in TEMPORAL_TAGS:
                cursor.execute(
                    "SELECT COUNT(*) FROM concept_temporal_tags WHERE concept_name = ?",
                    (concept,),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO concept_temporal_tags (concept_name, temporal_period, period_note) "
                        "VALUES (?, ?, ?)",
                        (concept, period, note),
                    )
                    count += 1
            conn.commit()
        logger.info("Populated %d concept temporal tags", count)
        return count
