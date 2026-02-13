"""Cultural context population for book-level introductions and annotations."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Curated book-level cultural context introductions.
# Format: (book_id, context_type, title, summary, detailed_content, time_period, geographic_region, confidence)
BOOK_INTRODUCTIONS: List[Tuple[int, str, str, str, str, str, str, str]] = [
    (
        1,
        "historical_background",
        "Genesis: Origins and Patriarchs",
        "Genesis covers creation through the patriarchal period, foundational to Israelite identity.",
        "Written in the ancient Near Eastern context where creation accounts were common. "
        "The patriarchal narratives reflect semi-nomadic life in the Fertile Crescent circa 2000-1800 BCE. "
        "Covenant-making ceremonies mirror Hittite suzerainty treaties. "
        "Understanding ANE parallels (Enuma Elish, Atrahasis) illuminates the distinctive theology.",
        "2000-1400 BCE",
        "Mesopotamia, Canaan, Egypt",
        "high",
    ),
    (
        2,
        "historical_background",
        "Exodus: Deliverance and Law",
        "Exodus narrates Israel's liberation from Egypt and covenant at Sinai.",
        "Set during the Late Bronze Age when Egypt controlled Canaan. "
        "The ten plagues directly challenged specific Egyptian deities. "
        "The covenant structure parallels ancient suzerainty treaties. "
        "The tabernacle design reflects ANE sacred space concepts.",
        "1446-1406 BCE or 1290-1250 BCE",
        "Egypt, Sinai Peninsula",
        "high",
    ),
    (
        3,
        "historical_background",
        "Leviticus: Holiness and Worship",
        "Leviticus provides Israel's worship system — sacrifices, purity, and priestly duties.",
        "Holiness meant separation for God's purposes. The sacrificial system used blood as a symbol "
        "of life given to atone. Purity laws distinguished Israel from surrounding cultures. "
        "The Day of Atonement (ch. 16) was the annual ritual for national cleansing.",
        "1446-1406 BCE",
        "Sinai Wilderness",
        "high",
    ),
    (
        5,
        "historical_background",
        "Deuteronomy: Covenant Renewal",
        "Moses' farewell speeches renewing the covenant before entering the Promised Land.",
        "Structured as an ancient Near Eastern treaty: preamble, historical prologue, stipulations, "
        "blessings/curses, and witnesses. The Shema (6:4-9) became Israel's central confession. "
        "The book addresses a transition generation preparing for settled agricultural life.",
        "1406 BCE",
        "Plains of Moab",
        "high",
    ),
    (
        18,
        "historical_background",
        "Job: Suffering and Divine Justice",
        "Job explores why the righteous suffer, challenging retribution theology.",
        "Set in the patriarchal period in the land of Uz (likely Edom). "
        "The dialogue form resembles ancient Near Eastern wisdom debates. "
        "Job's friends represent orthodox retribution theology — suffering proves sin. "
        "God's speeches from the whirlwind reframe the question entirely.",
        "Patriarchal period (setting); writing date debated",
        "Land of Uz",
        "medium",
    ),
    (
        19,
        "historical_background",
        "Psalms: Israel's Songbook",
        "The Psalms are Israel's collected prayers, praises, and laments spanning centuries.",
        "Psalms were sung in temple worship with musical accompaniment. "
        "Major types: hymns of praise, individual/communal laments, thanksgiving, royal, wisdom, "
        "and pilgrimage psalms. Hebrew poetry uses parallelism rather than rhyme. "
        "Many psalms have historical superscriptions linking them to David's life events.",
        "1000-400 BCE",
        "Jerusalem, various",
        "high",
    ),
    (
        20,
        "historical_background",
        "Proverbs: Wisdom for Daily Life",
        "Proverbs collects practical wisdom for living well in God's ordered world.",
        "Part of the ancient Near Eastern wisdom tradition (cf. Egyptian Instruction of Amenemope). "
        "Proverbs are general principles, not absolute promises. "
        "Lady Wisdom (chs. 1-9) personifies God's ordering principle in creation. "
        "The fear of the Lord is the foundational principle (1:7).",
        "950-700 BCE",
        "Jerusalem",
        "high",
    ),
    (
        23,
        "historical_background",
        "Isaiah: Prophet to Judah",
        "Isaiah prophesied during Assyria's rise, calling Judah to trust God alone.",
        "Isaiah ministered across four kings' reigns (740-680 BCE). "
        "Chapters 1-39 address the Assyrian crisis; 40-66 look beyond exile to restoration. "
        "The Suffering Servant songs (42, 49, 50, 52-53) are central to Christian interpretation. "
        "Isaiah's vision of universal salvation extends beyond Israel to all nations.",
        "740-680 BCE",
        "Jerusalem, Judah",
        "high",
    ),
    (
        40,
        "historical_background",
        "Matthew: Jesus as Jewish Messiah",
        "Matthew presents Jesus as the promised Messiah fulfilling Hebrew Scripture.",
        "Written primarily for Jewish Christians. Uses formula quotations ('this was to fulfill...') "
        "to connect Jesus to OT prophecy. Structures Jesus' teaching in five major discourses "
        "(parallel to the five books of Moses). Emphasizes the 'kingdom of heaven.' "
        "The Sermon on the Mount (chs. 5-7) presents Jesus as the authoritative interpreter of Torah.",
        "50-70 CE",
        "Antioch (likely)",
        "high",
    ),
    (
        41,
        "historical_background",
        "Mark: The Suffering Servant",
        "Mark is the earliest Gospel, emphasizing Jesus' actions and the cost of discipleship.",
        "Written for Roman Christians during persecution. Fast-paced narrative uses 'immediately' frequently. "
        "The 'Messianic Secret' — Jesus repeatedly tells people not to reveal his identity. "
        "The second half focuses on Jesus' journey to the cross. "
        "Discipleship means taking up one's cross (8:34).",
        "65-70 CE",
        "Rome (likely)",
        "high",
    ),
    (
        42,
        "historical_background",
        "Luke: Universal Savior",
        "Luke presents Jesus as savior for all people — Gentiles, women, poor, and marginalized.",
        "Written by a physician and travel companion of Paul. Unique parables: Good Samaritan, "
        "Prodigal Son, Rich Man and Lazarus. Emphasizes the Holy Spirit, prayer, and joy. "
        "Jesus' inaugural sermon in Nazareth (4:16-30) programmatically announces his mission. "
        "A two-volume work with Acts.",
        "60-80 CE",
        "Unknown (for Theophilus)",
        "high",
    ),
    (
        43,
        "historical_background",
        "John: The Divine Word",
        "John presents Jesus as the pre-existent Word of God through seven signs and discourses.",
        "Distinct from the Synoptics in structure, vocabulary, and chronology. "
        "Seven 'I am' statements reveal Jesus' identity (bread of life, light of the world, etc.). "
        "Seven signs (miracles) demonstrate his divine authority. "
        "The farewell discourse (chs. 13-17) is unique to John. "
        "Themes: light/darkness, belief/unbelief, above/below, life/death.",
        "85-95 CE",
        "Ephesus (tradition)",
        "high",
    ),
    (
        44,
        "historical_background",
        "Acts: The Early Church",
        "Acts narrates the Spirit-empowered expansion of the church from Jerusalem to Rome.",
        "Continues Luke's Gospel. Pentecost inaugurates the church age. "
        "The narrative follows two main figures: Peter (chs. 1-12) and Paul (chs. 13-28). "
        "Key speeches articulate early Christian theology. "
        "The Jerusalem Council (ch. 15) is pivotal for Gentile inclusion.",
        "30-62 CE (events); 62-80 CE (composition)",
        "Jerusalem to Rome",
        "high",
    ),
    (
        45,
        "historical_background",
        "Romans: The Gospel Explained",
        "Paul's most systematic letter explaining justification by faith and life in the Spirit.",
        "Written to a church Paul had not yet visited. Addresses Jew-Gentile tensions. "
        "Chapters 1-8: theological argument (sin, justification, sanctification, glorification). "
        "Chapters 9-11: Israel's role in God's plan. Chapters 12-16: practical ethics. "
        "The 'Romans Road' traces the logic of salvation.",
        "57 CE",
        "Corinth (written to Rome)",
        "high",
    ),
    (
        58,
        "historical_background",
        "Hebrews: Christ Superior to All",
        "Hebrews argues for Christ's superiority over angels, Moses, and the old covenant priesthood.",
        "Written to Jewish Christians tempted to return to Judaism. "
        "Uses extensive OT quotation and typological interpretation. "
        "Christ is the ultimate high priest after the order of Melchizedek. "
        "The 'Hall of Faith' (ch. 11) surveys OT examples of faithfulness.",
        "60-70 CE",
        "Unknown (possibly Rome)",
        "medium",
    ),
    (
        66,
        "historical_background",
        "Revelation: Apocalyptic Hope",
        "Revelation unveils God's ultimate victory over evil through vivid symbolic imagery.",
        "Written during Roman persecution (likely Domitian, 81-96 CE). "
        "Apocalyptic literature uses symbolic numbers, colors, and creatures. "
        "The number 7 (completeness) structures the book: 7 churches, seals, trumpets, bowls. "
        "Not primarily a timeline but a theological vision of God's sovereignty. "
        "Ends with the new creation — God dwelling with humanity forever.",
        "90-96 CE",
        "Patmos (written to seven churches in Asia Minor)",
        "high",
    ),
]


class CulturalContextPopulator:
    """Populates the cultural_context table with book-level introductions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert cultural context data into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cultural_context'")
            if cursor.fetchone()[0] == 0:
                logger.warning("cultural_context table does not exist; run migrations first")
                return 0

            if force:
                cursor.execute("DELETE FROM cultural_context WHERE context_type = 'historical_background'")

            for entry in BOOK_INTRODUCTIONS:
                book_id, ctx_type, title, summary, detailed, time_period, geo_region, confidence = entry
                try:
                    # Check for existing entry (no unique constraint on table)
                    cursor.execute(
                        "SELECT COUNT(*) FROM cultural_context WHERE book_id = ? AND context_type = ? AND title = ?",
                        (book_id, ctx_type, title),
                    )
                    if cursor.fetchone()[0] > 0:
                        continue
                    cursor.execute(
                        "INSERT INTO cultural_context "
                        "(book_id, context_type, title, summary, detailed_content, "
                        "time_period, geographic_region, confidence, display_priority) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (book_id, ctx_type, title, summary, detailed, time_period, geo_region, confidence, 1),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert cultural context for book %d: %s", book_id, e)

            conn.commit()

        logger.info("Populated cultural_context: %d rows inserted", inserted)
        return inserted

    @staticmethod
    def get_context_for_book(db_path: Path, book_id: int) -> List[Dict[str, Any]]:
        """Get cultural context entries for a book.

        Args:
            db_path: Path to the database.
            book_id: Book ID.

        Returns:
            List of cultural context dictionaries.
        """
        contexts: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_id, context_type, title, summary, detailed_content, "
                "time_period, geographic_region, confidence "
                "FROM cultural_context WHERE book_id = ? ORDER BY display_priority",
                (book_id,),
            )
            for row in cursor.fetchall():
                contexts.append(
                    {
                        "context_id": row[0],
                        "context_type": row[1],
                        "title": row[2],
                        "summary": row[3],
                        "detailed_content": row[4],
                        "time_period": row[5],
                        "geographic_region": row[6],
                        "confidence": row[7],
                    }
                )
        return contexts

    @staticmethod
    def get_context_for_verse(db_path: Path, book_id: int, chapter: int, verse: int) -> List[Dict[str, Any]]:
        """Get cultural context entries applicable to a specific verse.

        Includes both book-level and verse-range entries.

        Args:
            db_path: Path to the database.
            book_id: Book ID.
            chapter: Chapter number.
            verse: Verse number.

        Returns:
            List of cultural context dictionaries.
        """
        contexts: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Book-level (no chapter/verse specified)
            cursor.execute(
                "SELECT context_id, context_type, title, summary, detailed_content, "
                "time_period, geographic_region, confidence "
                "FROM cultural_context "
                "WHERE book_id = ? AND start_chapter IS NULL "
                "ORDER BY display_priority",
                (book_id,),
            )
            for row in cursor.fetchall():
                contexts.append(
                    {
                        "context_id": row[0],
                        "context_type": row[1],
                        "title": row[2],
                        "summary": row[3],
                        "detailed_content": row[4],
                        "time_period": row[5],
                        "geographic_region": row[6],
                        "confidence": row[7],
                    }
                )

            # Verse-range entries
            cursor.execute(
                "SELECT context_id, context_type, title, summary, detailed_content, "
                "time_period, geographic_region, confidence "
                "FROM cultural_context "
                "WHERE book_id = ? AND start_chapter IS NOT NULL "
                "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
                "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) "
                "ORDER BY display_priority",
                (book_id, chapter, chapter, verse, chapter, chapter, verse),
            )
            for row in cursor.fetchall():
                contexts.append(
                    {
                        "context_id": row[0],
                        "context_type": row[1],
                        "title": row[2],
                        "summary": row[3],
                        "detailed_content": row[4],
                        "time_period": row[5],
                        "geographic_region": row[6],
                        "confidence": row[7],
                    }
                )
        return contexts
