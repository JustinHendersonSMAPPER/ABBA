"""Genre-shift detection within biblical books.

Identifies locations where books transition between literary genres
(e.g., narrative → poetry in Exodus 15, Judges 5).
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (book_id, chapter, verse, from_genre, to_genre, description)
CURATED_GENRE_SHIFTS: List[Tuple[int, int, int, str, str, str]] = [
    # Exodus: narrative → poetry (Song of the Sea)
    (2, 15, 1, "narrative", "poetry", "Song of the Sea — Israel celebrates crossing the Red Sea"),
    (2, 15, 21, "poetry", "narrative", "Return to narrative after the Song of Miriam"),
    # Judges: narrative → poetry (Song of Deborah)
    (7, 5, 1, "narrative", "poetry", "Song of Deborah and Barak — victory hymn"),
    (7, 6, 1, "poetry", "narrative", "Return to narrative after Song of Deborah"),
    # 1 Samuel: narrative → poetry (Hannah's Prayer)
    (9, 2, 1, "narrative", "poetry", "Hannah's Prayer — prophetic hymn of thanksgiving"),
    (9, 2, 11, "poetry", "narrative", "Return to narrative after Hannah's Prayer"),
    # 2 Samuel: narrative → poetry (David's Lament)
    (10, 1, 19, "narrative", "poetry", "David's Lament over Saul and Jonathan"),
    (10, 2, 1, "poetry", "narrative", "Return to narrative after David's Lament"),
    # 2 Samuel: narrative → poetry (David's Song of Deliverance)
    (10, 22, 1, "narrative", "poetry", "David's Song of Deliverance (parallel to Psalm 18)"),
    (10, 23, 1, "poetry", "narrative", "Return to narrative after David's Song"),
    # Isaiah: prophecy → apocalyptic (Isaiah's Apocalypse)
    (23, 24, 1, "prophecy", "apocalyptic", "Isaiah's Apocalypse — cosmic judgment"),
    (23, 28, 1, "apocalyptic", "prophecy", "Return to prophetic woe oracles"),
    # Isaiah: prophecy → poetry (Servant Songs)
    (23, 42, 1, "prophecy", "poetry", "First Servant Song — the chosen servant"),
    (23, 53, 12, "poetry", "prophecy", "End of Fourth Servant Song, return to prophetic oracle"),
    # Jeremiah: prophecy → poetry (Lamentations within Jeremiah)
    (24, 9, 1, "prophecy", "poetry", "Jeremiah's personal lament"),
    (24, 10, 1, "poetry", "prophecy", "Return to prophetic message"),
    # Daniel: narrative → apocalyptic
    (27, 7, 1, "narrative", "apocalyptic", "Daniel's vision of four beasts — shift to apocalyptic"),
    (27, 8, 1, "apocalyptic", "apocalyptic", "Vision of the ram and goat continues apocalyptic"),
    # Habakkuk: prophecy → prayer/poetry
    (35, 3, 1, "prophecy", "poetry", "Habakkuk's prayer-hymn — a theophany"),
    # Jonah: narrative → poetry
    (32, 2, 2, "narrative", "poetry", "Jonah's prayer from the belly of the fish"),
    (32, 3, 1, "poetry", "narrative", "Return to narrative after Jonah's prayer"),
    # Numbers: narrative → poetry (Balaam's Oracles)
    (4, 23, 7, "narrative", "poetry", "Balaam's first oracle — prophetic poetry"),
    (4, 24, 25, "poetry", "narrative", "Return to narrative after Balaam's oracles"),
    # Deuteronomy: law → poetry (Song of Moses)
    (5, 32, 1, "law", "poetry", "Song of Moses — covenant lawsuit song"),
    (5, 33, 1, "poetry", "poetry", "Blessing of Moses — tribal blessings"),
    (5, 34, 1, "poetry", "narrative", "Return to narrative — death of Moses"),
    # Job: narrative → poetry → narrative
    (18, 3, 1, "narrative", "poetry", "Job's opening lament — shift to poetic dialogue"),
    (18, 42, 7, "poetry", "narrative", "Return to narrative epilogue"),
    # Ecclesiastes: wisdom → poetry (poem on time)
    (21, 3, 1, "wisdom", "poetry", "Poem on appointed times — 'a time for everything'"),
    (21, 3, 9, "poetry", "wisdom", "Return to wisdom reflection"),
    # Revelation: epistle → apocalyptic
    (66, 4, 1, "epistle", "apocalyptic", "Throne room vision — shift to apocalyptic visions"),
]


class GenreShiftPopulator:
    """Populates genre_shifts table with curated genre transitions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert curated genre shifts into the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if force:
                cursor.execute("DELETE FROM genre_shifts")

            count = 0
            for book_id, chapter, verse, from_genre, to_genre, description in CURATED_GENRE_SHIFTS:
                cursor.execute(
                    "SELECT COUNT(*) FROM genre_shifts WHERE book_id = ? AND chapter = ? AND verse = ?",
                    (book_id, chapter, verse),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO genre_shifts (book_id, chapter, verse, from_genre, to_genre, description) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (book_id, chapter, verse, from_genre, to_genre, description),
                    )
                    count += 1
            conn.commit()
        logger.info("Populated %d genre shifts", count)
        return count

    @staticmethod
    def get_shifts_for_book(db_path: Path, book_id: int) -> list:
        """Get all genre shifts within a book."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chapter, verse, from_genre, to_genre, description "
                "FROM genre_shifts WHERE book_id = ? ORDER BY chapter, verse",
                (book_id,),
            )
            return cursor.fetchall()

    @staticmethod
    def get_genre_at_verse(db_path: Path, book_id: int, chapter: int, verse: int) -> str:
        """Determine the active genre at a given verse by finding the most recent shift."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT to_genre FROM genre_shifts "
                "WHERE book_id = ? AND (chapter < ? OR (chapter = ? AND verse <= ?)) "
                "ORDER BY chapter DESC, verse DESC LIMIT 1",
                (book_id, chapter, chapter, verse),
            )
            row = cursor.fetchone()
            return row[0] if row else "unknown"
