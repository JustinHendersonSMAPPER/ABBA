"""Speaker attribution for quoted speech in biblical texts.

Identifies who is speaking in key passages to prevent misattribution
and support anti-proof-texting safeguards.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (book_id, start_chapter, start_verse, end_chapter, end_verse, speaker, context)
CURATED_ATTRIBUTIONS: List[Tuple[int, int, int, int, int, str, str]] = [
    # Genesis - God speaking
    (1, 1, 3, 1, 3, "God", "God commands light into existence"),
    (1, 1, 28, 1, 30, "God", "God blesses humanity and gives dominion"),
    (1, 2, 18, 2, 18, "God", "God declares it not good for man to be alone"),
    (1, 3, 1, 3, 5, "Serpent", "The serpent tempts Eve with deception"),
    (1, 3, 9, 3, 13, "God", "God questions Adam and Eve after the fall"),
    (1, 3, 14, 3, 19, "God", "God pronounces curses and the protoevangelium"),
    (1, 12, 1, 12, 3, "God", "God's call and covenant with Abram"),
    # Exodus
    (2, 3, 7, 3, 10, "God", "God speaks from the burning bush"),
    (2, 3, 14, 3, 14, "God", "God reveals His name — I AM WHO I AM"),
    (2, 20, 2, 20, 17, "God", "God speaks the Ten Commandments"),
    # Deuteronomy
    (5, 6, 4, 6, 9, "Moses", "Moses teaches the Shema"),
    # Job - key speakers
    (18, 1, 7, 1, 12, "Satan", "Satan challenges Job's faithfulness before God"),
    (18, 2, 9, 2, 10, "Job's wife", "Job's wife tells him to curse God"),
    (18, 3, 1, 3, 26, "Job", "Job curses the day of his birth"),
    (18, 4, 1, 4, 21, "Eliphaz", "Eliphaz's first speech — suffering implies sin"),
    (18, 8, 1, 8, 22, "Bildad", "Bildad's first speech — God doesn't pervert justice"),
    (18, 11, 1, 11, 20, "Zophar", "Zophar's first speech — accuses Job of hidden sin"),
    (18, 32, 6, 33, 33, "Elihu", "Elihu speaks — younger voice challenging all parties"),
    (18, 38, 1, 41, 34, "God", "God answers Job from the whirlwind"),
    (18, 42, 1, 42, 6, "Job", "Job's repentance and submission"),
    # Psalms
    (19, 2, 7, 2, 9, "God/Messiah", "God declares His decree — prophetic/messianic"),
    (19, 22, 1, 22, 1, "David/Messianic", "Cry of dereliction — quoted by Jesus on the cross"),
    (19, 110, 1, 110, 1, "God", "The LORD says to my Lord — messianic oracle"),
    # Isaiah
    (23, 6, 8, 6, 8, "Isaiah", "Isaiah volunteers — 'Here am I, send me'"),
    (23, 14, 12, 14, 15, "Narrator about King of Babylon", "Taunt against the king, often misapplied to Satan"),
    (23, 53, 1, 53, 12, "Isaiah/Prophetic", "Fourth Servant Song — prophetic voice about the suffering servant"),
    # Jeremiah
    (24, 29, 11, 29, 11, "God", "God's plans for Israel's future — spoken to exiles"),
    # Matthew
    (40, 4, 1, 4, 11, "Satan / Jesus", "Temptation of Jesus — Satan tempts, Jesus responds"),
    (40, 5, 3, 7, 27, "Jesus", "Sermon on the Mount — Jesus teaches"),
    (40, 16, 23, 16, 23, "Jesus", "Jesus rebukes Peter — 'Get behind me, Satan'"),
    (40, 23, 1, 23, 39, "Jesus", "Seven Woes against the Pharisees"),
    (40, 27, 46, 27, 46, "Jesus", "Cry from the cross — quoting Psalm 22:1"),
    # Mark
    (41, 1, 11, 1, 11, "God the Father", "Voice from heaven at Jesus' baptism"),
    (41, 5, 9, 5, 9, "Demons/Legion", "The demons identify themselves"),
    # Luke
    (42, 1, 46, 1, 55, "Mary", "The Magnificat — Mary's song"),
    (42, 1, 68, 1, 79, "Zechariah", "The Benedictus — Zechariah's prophecy"),
    (42, 2, 29, 2, 32, "Simeon", "The Nunc Dimittis — Simeon's song"),
    (42, 15, 11, 15, 32, "Jesus", "Parable of the Prodigal Son — Jesus narrates"),
    (42, 16, 19, 16, 31, "Jesus/Abraham", "Parable — Jesus narrates, Abraham speaks within"),
    # John
    (43, 1, 1, 1, 18, "John/Narrator", "Prologue — the evangelist's theological introduction"),
    (43, 3, 16, 3, 21, "Jesus or John/Narrator", "Debated: may be Jesus speaking or narrator's commentary"),
    (43, 8, 44, 8, 44, "Jesus", "Jesus speaking to opponents about the devil as father of lies"),
    (43, 14, 6, 14, 6, "Jesus", "Jesus declares: 'I am the way, the truth, and the life'"),
    # Acts
    (44, 5, 3, 5, 4, "Peter", "Peter confronts Ananias about lying to the Holy Spirit"),
    (44, 17, 22, 17, 31, "Paul", "Paul's speech at the Areopagus in Athens"),
    # Romans
    (45, 3, 10, 3, 18, "Paul quoting OT", "Catena of OT quotations — Paul's argument, not Paul's words"),
    (45, 7, 15, 7, 25, "Paul", "Paul's inner struggle — 'what I want to do I do not do'"),
    (45, 9, 20, 9, 21, "Paul", "Paul uses the potter/clay analogy — rhetorical argument"),
    # 1 Corinthians
    (46, 7, 10, 7, 11, "Paul quoting Jesus", "Paul transmits a command from the Lord about divorce"),
    (46, 7, 12, 7, 16, "Paul", "Paul gives his own judgment — 'I, not the Lord, say'"),
    # Philippians
    (50, 2, 6, 2, 11, "Paul quoting hymn", "Christ Hymn — likely pre-Pauline hymn, not Paul's own composition"),
    # Hebrews
    (58, 1, 5, 1, 14, "Author quoting OT", "Chain of OT quotations applied to Christ"),
    # James
    (59, 2, 19, 2, 19, "James", "James' rhetoric — 'even the demons believe'"),
    # Revelation
    (66, 1, 8, 1, 8, "God/Christ", "Alpha and Omega declaration"),
    (66, 2, 1, 3, 22, "Christ", "Letters to the seven churches — Jesus dictates"),
    (66, 22, 20, 22, 20, "Jesus / John", "Jesus: 'I am coming soon.' John: 'Amen. Come, Lord Jesus.'"),
]


class SpeakerAttributionPopulator:
    """Populates speaker_attributions table."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert curated speaker attributions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if force:
                cursor.execute("DELETE FROM speaker_attributions")

            count = 0
            for book_id, sc, sv, ec, ev, speaker, context in CURATED_ATTRIBUTIONS:
                cursor.execute(
                    "SELECT COUNT(*) FROM speaker_attributions "
                    "WHERE book_id = ? AND start_chapter = ? AND start_verse = ?",
                    (book_id, sc, sv),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO speaker_attributions "
                        "(book_id, start_chapter, start_verse, end_chapter, end_verse, speaker, context_note) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (book_id, sc, sv, ec, ev, speaker, context),
                    )
                    count += 1
            conn.commit()
        logger.info("Populated %d speaker attributions", count)
        return count

    @staticmethod
    def get_speaker_for_verse(db_path: Path, book_id: int, chapter: int, verse: int) -> list:
        """Get speaker attribution for a verse."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT speaker, context_note FROM speaker_attributions "
                "WHERE book_id = ? "
                "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
                "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) ",
                (book_id, chapter, chapter, verse, chapter, chapter, verse),
            )
            return cursor.fetchall()
