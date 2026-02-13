"""Pericope/passage boundary population for biblical text structure."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Curated passage boundaries (pericopes).
# Format: (book_id, start_ch, start_vs, end_ch, end_vs, title, genre, literary_type, display_order)
CURATED_PASSAGES: List[Tuple[int, int, int, int, int, str, str, str, int]] = [
    # Genesis
    (1, 1, 1, 1, 2, "Creation Prologue", "narrative", "cosmogony", 1),
    (1, 1, 1, 2, 3, "Creation Account", "narrative", "cosmogony", 2),
    (1, 2, 4, 2, 25, "Garden of Eden", "narrative", "origin_story", 3),
    (1, 3, 1, 3, 24, "The Fall", "narrative", "origin_story", 4),
    (1, 4, 1, 4, 16, "Cain and Abel", "narrative", "origin_story", 5),
    (1, 6, 1, 9, 17, "The Flood", "narrative", "judgment_narrative", 6),
    (1, 11, 1, 11, 9, "Tower of Babel", "narrative", "origin_story", 7),
    (1, 12, 1, 12, 9, "Call of Abram", "narrative", "covenant_narrative", 8),
    (1, 15, 1, 15, 21, "Abrahamic Covenant", "narrative", "covenant_narrative", 9),
    (1, 22, 1, 22, 19, "Binding of Isaac", "narrative", "test_narrative", 10),
    (1, 37, 1, 50, 26, "Joseph Narrative", "narrative", "novella", 11),
    # Exodus
    (2, 1, 1, 2, 10, "Birth of Moses", "narrative", "origin_story", 1),
    (2, 3, 1, 4, 17, "Burning Bush", "narrative", "theophany", 2),
    (2, 7, 14, 12, 36, "The Ten Plagues", "narrative", "judgment_narrative", 3),
    (2, 12, 1, 12, 51, "The Passover", "narrative", "liturgical_narrative", 4),
    (2, 14, 1, 14, 31, "Crossing the Red Sea", "narrative", "deliverance_narrative", 5),
    (2, 15, 1, 15, 21, "Song of the Sea", "poetry", "victory_song", 6),
    (2, 20, 1, 20, 17, "The Ten Commandments", "law", "apodictic_law", 7),
    (2, 25, 1, 31, 18, "Tabernacle Instructions", "law", "cultic_law", 8),
    (2, 32, 1, 32, 35, "The Golden Calf", "narrative", "apostasy_narrative", 9),
    # Leviticus
    (3, 16, 1, 16, 34, "Day of Atonement", "law", "cultic_law", 1),
    (3, 19, 1, 19, 37, "Holiness Code", "law", "ethical_law", 2),
    # Deuteronomy
    (5, 6, 4, 6, 9, "The Shema", "law", "creedal_statement", 1),
    (5, 28, 1, 28, 68, "Blessings and Curses", "law", "covenant_sanctions", 2),
    # Joshua
    (6, 1, 1, 1, 9, "Commission of Joshua", "narrative", "commission_narrative", 1),
    # Judges
    (7, 5, 1, 5, 31, "Song of Deborah", "poetry", "victory_song", 1),
    # Ruth
    (8, 1, 1, 4, 22, "Ruth and Boaz", "narrative", "novella", 1),
    # 1 Samuel
    (9, 17, 1, 17, 58, "David and Goliath", "narrative", "battle_narrative", 1),
    # 2 Samuel
    (10, 7, 1, 7, 29, "Davidic Covenant", "narrative", "covenant_narrative", 1),
    (10, 11, 1, 12, 25, "David and Bathsheba", "narrative", "royal_narrative", 2),
    # 1 Kings
    (11, 18, 1, 18, 46, "Elijah on Carmel", "narrative", "prophetic_narrative", 1),
    # Job
    (18, 1, 1, 2, 13, "Job's Testing", "narrative", "prose_framework", 1),
    (18, 38, 1, 41, 34, "God's Speech from the Whirlwind", "poetry", "divine_speech", 2),
    # Psalms
    (19, 1, 1, 1, 6, "The Two Ways", "poetry", "wisdom_psalm", 1),
    (19, 8, 1, 8, 9, "God's Majesty and Human Dignity", "poetry", "hymn", 2),
    (19, 22, 1, 22, 31, "My God, Why Have You Forsaken Me?", "poetry", "lament", 3),
    (19, 23, 1, 23, 6, "The Lord Is My Shepherd", "poetry", "trust_psalm", 4),
    (19, 51, 1, 51, 19, "Create in Me a Clean Heart", "poetry", "penitential_psalm", 5),
    (19, 91, 1, 91, 16, "Shelter of the Most High", "poetry", "trust_psalm", 6),
    (19, 103, 1, 103, 22, "Bless the Lord, O My Soul", "poetry", "hymn", 7),
    (19, 119, 1, 119, 176, "The Law of the Lord", "poetry", "acrostic_psalm", 8),
    (19, 139, 1, 139, 24, "You Have Searched Me and Known Me", "poetry", "hymn", 9),
    # Proverbs
    (20, 1, 1, 9, 18, "Wisdom's Call", "wisdom", "instruction", 1),
    (20, 31, 10, 31, 31, "The Virtuous Woman", "poetry", "acrostic_poem", 2),
    # Ecclesiastes
    (21, 1, 1, 1, 11, "Vanity of Vanities", "wisdom", "philosophical_reflection", 1),
    (21, 3, 1, 3, 15, "A Time for Everything", "poetry", "time_poem", 2),
    # Song of Solomon
    (22, 1, 1, 8, 14, "Song of Songs", "poetry", "love_poetry", 1),
    # Isaiah
    (23, 6, 1, 6, 13, "Isaiah's Commissioning", "narrative", "prophetic_call", 1),
    (23, 7, 14, 7, 14, "Immanuel Prophecy", "prophecy", "messianic_oracle", 2),
    (23, 9, 6, 9, 7, "Prince of Peace", "prophecy", "messianic_oracle", 3),
    (23, 40, 1, 40, 31, "Comfort My People", "prophecy", "salvation_oracle", 4),
    (23, 52, 13, 53, 12, "The Suffering Servant", "prophecy", "servant_song", 5),
    (23, 61, 1, 61, 3, "The Year of the Lord's Favor", "prophecy", "salvation_oracle", 6),
    # Jeremiah
    (24, 31, 31, 31, 34, "The New Covenant", "prophecy", "covenant_oracle", 1),
    # Lamentations
    (25, 1, 1, 5, 22, "Lament over Jerusalem", "poetry", "lament", 1),
    # Ezekiel
    (26, 37, 1, 37, 14, "Valley of Dry Bones", "prophecy", "vision_narrative", 1),
    # Daniel
    (27, 3, 1, 3, 30, "Fiery Furnace", "narrative", "court_tale", 1),
    (27, 7, 1, 7, 28, "Four Beasts Vision", "apocalyptic", "symbolic_vision", 2),
    # Hosea
    (28, 1, 1, 3, 5, "Hosea's Marriage", "prophecy", "symbolic_action", 1),
    # Joel
    (29, 2, 28, 2, 32, "Outpouring of the Spirit", "prophecy", "eschatological_oracle", 1),
    # Jonah
    (32, 1, 1, 4, 11, "Jonah and the Fish", "narrative", "prophetic_narrative", 1),
    # Micah
    (33, 5, 2, 5, 5, "Bethlehem Prophecy", "prophecy", "messianic_oracle", 1),
    (33, 6, 6, 6, 8, "What the Lord Requires", "prophecy", "ethical_exhortation", 2),
    # Habakkuk
    (35, 2, 4, 2, 4, "The Righteous Shall Live by Faith", "prophecy", "oracle_response", 1),
    # Malachi
    (39, 3, 1, 3, 5, "The Messenger of the Covenant", "prophecy", "messenger_oracle", 1),
    # Matthew
    (40, 1, 1, 1, 17, "Genealogy of Jesus", "narrative", "genealogy", 1),
    (40, 1, 18, 2, 23, "Birth of Jesus", "narrative", "infancy_narrative", 2),
    (40, 3, 1, 3, 17, "Baptism of Jesus", "narrative", "baptism_narrative", 3),
    (40, 4, 1, 4, 11, "Temptation of Jesus", "narrative", "test_narrative", 4),
    (40, 5, 1, 7, 29, "Sermon on the Mount", "discourse", "ethical_discourse", 5),
    (40, 5, 3, 5, 12, "The Beatitudes", "discourse", "beatitude", 6),
    (40, 6, 9, 6, 13, "The Lord's Prayer", "discourse", "prayer_instruction", 7),
    (40, 13, 1, 13, 52, "Kingdom Parables", "discourse", "parable_collection", 8),
    (40, 14, 13, 14, 21, "Feeding of the Five Thousand", "narrative", "miracle", 9),
    (40, 16, 13, 16, 20, "Peter's Confession", "narrative", "recognition_scene", 10),
    (40, 17, 1, 17, 13, "The Transfiguration", "narrative", "theophany", 11),
    (40, 21, 1, 21, 11, "Triumphal Entry", "narrative", "entry_narrative", 12),
    (40, 26, 17, 26, 30, "The Last Supper", "narrative", "farewell_meal", 13),
    (40, 26, 36, 26, 46, "Gethsemane", "narrative", "passion_narrative", 14),
    (40, 27, 1, 27, 66, "Crucifixion and Burial", "narrative", "passion_narrative", 15),
    (40, 28, 1, 28, 20, "Resurrection and Great Commission", "narrative", "resurrection_narrative", 16),
    # Mark
    (41, 1, 1, 1, 15, "Beginning of the Gospel", "narrative", "introduction", 1),
    (41, 4, 1, 4, 34, "Parables of the Kingdom", "discourse", "parable_collection", 2),
    (41, 10, 45, 10, 45, "Ransom Saying", "discourse", "mission_statement", 3),
    (41, 15, 1, 16, 8, "Passion and Resurrection", "narrative", "passion_narrative", 4),
    # Luke
    (42, 1, 1, 2, 52, "Infancy Narratives", "narrative", "infancy_narrative", 1),
    (42, 1, 46, 1, 55, "Magnificat", "poetry", "hymn", 2),
    (42, 2, 1, 2, 20, "Birth of Jesus", "narrative", "birth_narrative", 3),
    (42, 4, 16, 4, 30, "Nazareth Synagogue", "narrative", "inaugural_sermon", 4),
    (42, 10, 25, 10, 37, "Parable of the Good Samaritan", "discourse", "parable", 5),
    (42, 15, 1, 15, 32, "Lost Parables (Sheep, Coin, Son)", "discourse", "parable_trilogy", 6),
    (42, 22, 14, 22, 23, "Last Supper", "narrative", "farewell_meal", 7),
    (42, 24, 1, 24, 53, "Resurrection and Ascension", "narrative", "resurrection_narrative", 8),
    # John
    (43, 1, 1, 1, 18, "Prologue: The Word", "poetry", "hymnic_prologue", 1),
    (43, 2, 1, 2, 12, "Wedding at Cana", "narrative", "sign_narrative", 2),
    (43, 3, 1, 3, 21, "Nicodemus Discourse", "discourse", "dialogue", 3),
    (43, 4, 1, 4, 42, "Woman at the Well", "narrative", "encounter_narrative", 4),
    (43, 6, 1, 6, 71, "Bread of Life Discourse", "discourse", "revelation_discourse", 5),
    (43, 10, 1, 10, 21, "Good Shepherd Discourse", "discourse", "revelation_discourse", 6),
    (43, 11, 1, 11, 44, "Raising of Lazarus", "narrative", "sign_narrative", 7),
    (43, 13, 1, 17, 26, "Upper Room Discourse", "discourse", "farewell_discourse", 8),
    (43, 18, 1, 19, 42, "Passion Narrative", "narrative", "passion_narrative", 9),
    (43, 20, 1, 21, 25, "Resurrection Appearances", "narrative", "resurrection_narrative", 10),
    # Acts
    (44, 1, 1, 1, 11, "Ascension", "narrative", "transition_narrative", 1),
    (44, 2, 1, 2, 47, "Pentecost", "narrative", "founding_narrative", 2),
    (44, 9, 1, 9, 31, "Conversion of Paul", "narrative", "conversion_narrative", 3),
    (44, 17, 16, 17, 34, "Paul at Athens", "narrative", "speech_narrative", 4),
    # Romans
    (45, 1, 16, 1, 17, "Theme of Romans", "epistle", "thesis_statement", 1),
    (45, 3, 21, 3, 31, "Justification by Faith", "epistle", "theological_argument", 2),
    (45, 6, 1, 6, 23, "Dead to Sin, Alive in Christ", "epistle", "ethical_exhortation", 3),
    (45, 8, 1, 8, 39, "Life in the Spirit", "epistle", "theological_argument", 4),
    (45, 12, 1, 12, 21, "Living Sacrifice", "epistle", "ethical_exhortation", 5),
    # 1 Corinthians
    (46, 13, 1, 13, 13, "The Love Chapter", "epistle", "hymnic_prose", 1),
    (46, 15, 1, 15, 58, "Resurrection Chapter", "epistle", "theological_argument", 2),
    # 2 Corinthians
    (47, 5, 17, 5, 21, "New Creation in Christ", "epistle", "theological_statement", 1),
    # Galatians
    (48, 5, 22, 5, 26, "Fruit of the Spirit", "epistle", "ethical_list", 1),
    # Ephesians
    (49, 2, 1, 2, 10, "By Grace Through Faith", "epistle", "theological_argument", 1),
    (49, 6, 10, 6, 20, "Armor of God", "epistle", "metaphorical_exhortation", 2),
    # Philippians
    (50, 2, 5, 2, 11, "Christ Hymn", "poetry", "christological_hymn", 1),
    (50, 4, 4, 4, 9, "Rejoice in the Lord", "epistle", "ethical_exhortation", 2),
    # Colossians
    (51, 1, 15, 1, 20, "Supremacy of Christ", "poetry", "christological_hymn", 1),
    # 1 Thessalonians
    (52, 4, 13, 4, 18, "The Coming of the Lord", "epistle", "eschatological_instruction", 1),
    # Hebrews
    (58, 1, 1, 1, 4, "God Has Spoken by His Son", "epistle", "theological_prologue", 1),
    (58, 11, 1, 11, 40, "Hall of Faith", "epistle", "exempla_list", 2),
    (58, 12, 1, 12, 3, "Looking to Jesus", "epistle", "exhortation", 3),
    # James
    (59, 1, 2, 1, 4, "Trials and Joy", "epistle", "wisdom_instruction", 1),
    (59, 2, 14, 2, 26, "Faith and Works", "epistle", "theological_argument", 2),
    # 1 Peter
    (60, 2, 4, 2, 10, "Living Stones", "epistle", "ecclesiological_metaphor", 1),
    # 1 John
    (62, 1, 5, 2, 2, "God Is Light", "epistle", "theological_statement", 1),
    (62, 4, 7, 4, 21, "God Is Love", "epistle", "theological_statement", 2),
    # Revelation
    (66, 1, 1, 1, 20, "Vision of the Risen Christ", "apocalyptic", "vision_narrative", 1),
    (66, 2, 1, 3, 22, "Letters to the Seven Churches", "epistle", "prophetic_letters", 2),
    (66, 4, 1, 5, 14, "Throne Room Vision", "apocalyptic", "heavenly_vision", 3),
    (66, 12, 1, 12, 17, "The Woman and the Dragon", "apocalyptic", "symbolic_vision", 4),
    (66, 19, 11, 19, 21, "The Rider on the White Horse", "apocalyptic", "judgment_vision", 5),
    (66, 21, 1, 22, 5, "New Heaven and New Earth", "apocalyptic", "restoration_vision", 6),
    (66, 22, 6, 22, 21, "Epilogue: Come, Lord Jesus", "apocalyptic", "concluding_exhortation", 7),
]


class PassagePopulator:
    """Populates the passages table with pericope boundary data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert passage boundaries into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='passages'")
            if cursor.fetchone()[0] == 0:
                logger.warning("passages table does not exist; run migrations first")
                return 0

            if force:
                cursor.execute("DELETE FROM passages")

            for passage in CURATED_PASSAGES:
                book_id, s_ch, s_vs, e_ch, e_vs, title, genre, lit_type, display_order = passage
                try:
                    cursor.execute(
                        "SELECT COUNT(*) FROM passages "
                        "WHERE book_id = ? AND start_chapter = ? AND start_verse = ? AND title = ?",
                        (book_id, s_ch, s_vs, title),
                    )
                    if cursor.fetchone()[0] > 0:
                        continue
                    cursor.execute(
                        "INSERT INTO passages "
                        "(book_id, start_chapter, start_verse, end_chapter, end_verse, "
                        "title, genre, literary_type, display_order) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (book_id, s_ch, s_vs, e_ch, e_vs, title, genre, lit_type, display_order),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert passage '%s': %s", title, e)

            conn.commit()

        logger.info("Populated passages: %d rows inserted", inserted)
        return inserted

    @staticmethod
    def get_passages_for_chapter(db_path: Path, book_id: int, chapter: int) -> List[Dict[str, Any]]:
        """Get passage boundaries that overlap a given chapter.

        Args:
            db_path: Path to the database.
            book_id: Book ID.
            chapter: Chapter number.

        Returns:
            List of passage dictionaries.
        """
        passages: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT passage_id, title, genre, literary_type, structural_features, "
                "start_chapter, start_verse, end_chapter, end_verse "
                "FROM passages "
                "WHERE book_id = ? AND start_chapter <= ? AND end_chapter >= ? "
                "ORDER BY display_order",
                (book_id, chapter, chapter),
            )
            for row in cursor.fetchall():
                passages.append(
                    {
                        "passage_id": row[0],
                        "title": row[1],
                        "genre": row[2],
                        "literary_type": row[3],
                        "structural_features": row[4],
                        "start_chapter": row[5],
                        "start_verse": row[6],
                        "end_chapter": row[7],
                        "end_verse": row[8],
                    }
                )
        return passages

    @staticmethod
    def get_passage_for_verse(db_path: Path, book_id: int, chapter: int, verse: int) -> List[Dict[str, Any]]:
        """Get passages that contain a specific verse.

        Args:
            db_path: Path to the database.
            book_id: Book ID.
            chapter: Chapter number.
            verse: Verse number.

        Returns:
            List of passage dictionaries containing this verse.
        """
        passages: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT passage_id, title, genre, literary_type, "
                "start_chapter, start_verse, end_chapter, end_verse "
                "FROM passages "
                "WHERE book_id = ? "
                "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
                "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) "
                "ORDER BY display_order",
                (book_id, chapter, chapter, verse, chapter, chapter, verse),
            )
            for row in cursor.fetchall():
                passages.append(
                    {
                        "passage_id": row[0],
                        "title": row[1],
                        "genre": row[2],
                        "literary_type": row[3],
                        "start_chapter": row[4],
                        "start_verse": row[5],
                        "end_chapter": row[6],
                        "end_verse": row[7],
                    }
                )
        return passages
