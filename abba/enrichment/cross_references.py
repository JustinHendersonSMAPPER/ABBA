"""Cross-reference population for biblical passages."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Well-known cross-references sourced from public domain scholarship.
# Format: (source_book, source_ch, source_vs, target_book, target_ch, target_vs, ref_type, confidence, notes)
# ref_type: quotation, allusion, parallel, thematic, prophecy_fulfillment, typology, contrast
CURATED_CROSS_REFERENCES: List[Tuple[int, int, int, int, int, int, str, float, str]] = [
    # OT Quotations in NT
    (40, 1, 23, 23, 7, 14, "prophecy_fulfillment", 0.95, "Virgin birth prophecy"),
    (40, 2, 6, 33, 5, 2, "prophecy_fulfillment", 0.95, "Messiah born in Bethlehem"),
    (40, 2, 15, 28, 11, 1, "prophecy_fulfillment", 0.90, "Out of Egypt I called my son"),
    (40, 3, 3, 23, 40, 3, "quotation", 0.95, "Voice crying in the wilderness"),
    (40, 4, 4, 5, 8, 3, "quotation", 0.95, "Man shall not live by bread alone"),
    (40, 4, 6, 19, 91, 11, "quotation", 0.90, "Angels shall bear you up"),
    (40, 4, 7, 5, 6, 16, "quotation", 0.95, "Do not put the Lord to the test"),
    (40, 4, 10, 5, 6, 13, "quotation", 0.95, "Worship the Lord your God only"),
    (40, 21, 5, 38, 9, 9, "prophecy_fulfillment", 0.95, "King coming on a donkey"),
    (40, 27, 46, 19, 22, 1, "quotation", 0.95, "My God, why have you forsaken me"),
    (43, 1, 1, 1, 1, 1, "allusion", 0.90, "In the beginning"),
    (43, 12, 40, 23, 6, 10, "quotation", 0.90, "Blinded their eyes"),
    (44, 2, 17, 29, 2, 28, "quotation", 0.95, "Pour out my Spirit"),
    (44, 2, 25, 19, 16, 8, "quotation", 0.90, "I have set the Lord always before me"),
    (45, 1, 17, 35, 2, 4, "quotation", 0.95, "The righteous shall live by faith"),
    (45, 3, 10, 19, 14, 1, "quotation", 0.90, "None is righteous, no not one"),
    (45, 4, 3, 1, 15, 6, "quotation", 0.95, "Abraham believed God"),
    (45, 9, 33, 23, 28, 16, "quotation", 0.90, "Stumbling stone in Zion"),
    (58, 1, 5, 19, 2, 7, "quotation", 0.95, "You are my Son, today I have begotten you"),
    (58, 3, 7, 19, 95, 7, "quotation", 0.95, "Today if you hear his voice"),
    (58, 10, 5, 19, 40, 6, "quotation", 0.90, "Sacrifices you did not desire"),
    (66, 1, 7, 27, 7, 13, "allusion", 0.85, "Coming with the clouds"),
    # Synoptic Parallels
    (40, 3, 13, 41, 1, 9, "parallel", 0.95, "Baptism of Jesus"),
    (40, 3, 13, 42, 3, 21, "parallel", 0.95, "Baptism of Jesus"),
    (40, 14, 13, 41, 6, 31, "parallel", 0.95, "Feeding of the 5000"),
    (40, 14, 13, 42, 9, 10, "parallel", 0.95, "Feeding of the 5000"),
    (40, 14, 13, 43, 6, 1, "parallel", 0.95, "Feeding of the 5000"),
    (40, 26, 26, 41, 14, 22, "parallel", 0.95, "Last Supper institution"),
    (40, 26, 26, 42, 22, 19, "parallel", 0.95, "Last Supper institution"),
    (40, 28, 1, 41, 16, 1, "parallel", 0.95, "Resurrection morning"),
    (40, 28, 1, 42, 24, 1, "parallel", 0.95, "Resurrection morning"),
    (40, 28, 1, 43, 20, 1, "parallel", 0.95, "Resurrection morning"),
    # Thematic Links
    (1, 1, 1, 43, 1, 1, "thematic", 0.90, "In the beginning - creation"),
    (1, 3, 15, 45, 16, 20, "thematic", 0.85, "Crushing the serpent/Satan"),
    (1, 12, 1, 48, 3, 8, "thematic", 0.90, "Abrahamic blessing to nations"),
    (1, 22, 8, 43, 1, 29, "thematic", 0.85, "God will provide the lamb"),
    (2, 12, 46, 43, 19, 36, "typology", 0.90, "No bone shall be broken"),
    (2, 12, 3, 46, 5, 7, "typology", 0.90, "Christ our Passover lamb"),
    (3, 16, 15, 58, 9, 7, "typology", 0.90, "Day of Atonement and Christ"),
    (19, 22, 1, 40, 27, 46, "prophecy_fulfillment", 0.95, "My God, why have you forsaken me"),
    (19, 22, 18, 43, 19, 24, "prophecy_fulfillment", 0.90, "Casting lots for garments"),
    (23, 53, 5, 60, 2, 24, "prophecy_fulfillment", 0.90, "By his wounds we are healed"),
    (23, 53, 7, 44, 8, 32, "prophecy_fulfillment", 0.90, "Like a lamb led to slaughter"),
    (23, 61, 1, 42, 4, 18, "prophecy_fulfillment", 0.90, "Spirit of the Lord upon me"),
    (24, 31, 31, 58, 8, 8, "prophecy_fulfillment", 0.90, "New covenant"),
    (27, 7, 13, 40, 26, 64, "prophecy_fulfillment", 0.85, "Son of Man coming on clouds"),
    # Creation-Redemption Bookends
    (1, 1, 1, 66, 21, 1, "thematic", 0.85, "Creation and new creation"),
    (1, 2, 9, 66, 22, 2, "thematic", 0.85, "Tree of life"),
    (1, 3, 17, 66, 22, 3, "thematic", 0.85, "Curse and no more curse"),
    # Key Doctrinal Links
    (1, 15, 6, 45, 4, 3, "thematic", 0.95, "Righteousness by faith"),
    (5, 6, 4, 41, 12, 29, "quotation", 0.95, "Hear O Israel, the Lord is one"),
    (3, 19, 18, 40, 22, 39, "quotation", 0.95, "Love your neighbor as yourself"),
    (23, 6, 9, 40, 13, 14, "quotation", 0.90, "Hearing but not understanding"),
    (19, 110, 1, 40, 22, 44, "quotation", 0.95, "The Lord said to my Lord"),
    (19, 110, 4, 58, 5, 6, "quotation", 0.95, "Priest forever after Melchizedek"),
    (23, 40, 3, 41, 1, 3, "quotation", 0.95, "Prepare the way of the Lord"),
]


class CrossReferencePopulator:
    """Populates the cross_references table."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert cross-references into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Ensure table exists
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cross_references'")
            if cursor.fetchone()[0] == 0:
                logger.warning("cross_references table does not exist; run migrations first")
                return 0

            for ref in CURATED_CROSS_REFERENCES:
                src_book, src_ch, src_vs, tgt_book, tgt_ch, tgt_vs, ref_type, confidence, notes = ref
                try:
                    verb = "INSERT OR REPLACE" if force else "INSERT OR IGNORE"
                    cursor.execute(
                        f"{verb} INTO cross_references "  # noqa: S608
                        "(source_book_id, source_chapter, source_verse, "
                        "target_book_id, target_chapter, target_verse, "
                        "ref_type, confidence, source_dataset, notes) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (src_book, src_ch, src_vs, tgt_book, tgt_ch, tgt_vs, ref_type, confidence, "curated", notes),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error(
                        "Failed to insert cross-ref %s %d:%d -> %d:%d: %s", ref_type, src_ch, src_vs, tgt_ch, tgt_vs, e
                    )

            conn.commit()

        logger.info("Populated cross_references: %d rows inserted", inserted)
        return inserted

    def get_cross_references_for_verse(
        self, db_path: Path, book_id: int, chapter: int, verse: int
    ) -> List[Dict[str, Any]]:
        """Get all cross-references for a specific verse (as source or target).

        Args:
            db_path: Path to the database.
            book_id: Book ID.
            chapter: Chapter number.
            verse: Verse number.

        Returns:
            List of cross-reference dictionaries.
        """
        refs: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # As source
            cursor.execute(
                "SELECT target_book_id, target_chapter, target_verse, ref_type, confidence, notes "
                "FROM cross_references "
                "WHERE source_book_id = ? AND source_chapter = ? AND source_verse = ?",
                (book_id, chapter, verse),
            )
            for row in cursor.fetchall():
                refs.append(
                    {
                        "target_book_id": row[0],
                        "target_chapter": row[1],
                        "target_verse": row[2],
                        "ref_type": row[3],
                        "confidence": row[4],
                        "notes": row[5],
                        "direction": "outgoing",
                    }
                )

            # As target
            cursor.execute(
                "SELECT source_book_id, source_chapter, source_verse, ref_type, confidence, notes "
                "FROM cross_references "
                "WHERE target_book_id = ? AND target_chapter = ? AND target_verse = ?",
                (book_id, chapter, verse),
            )
            for row in cursor.fetchall():
                refs.append(
                    {
                        "source_book_id": row[0],
                        "source_chapter": row[1],
                        "source_verse": row[2],
                        "ref_type": row[3],
                        "confidence": row[4],
                        "notes": row[5],
                        "direction": "incoming",
                    }
                )

        return refs
