"""Reading plans and guided study paths for new and growing believers."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Curated reading plans.
# Format: (slug, name, description, category, estimated_days)
READING_PLANS: List[Tuple[str, str, str, str, int]] = [
    (
        "start-here",
        "Start Here: Introduction to the Bible",
        "A 7-day introduction to the most important passages for someone new to the Bible",
        "beginner",
        7,
    ),
    (
        "gospel-of-john",
        "The Gospel of John",
        "Read through the Gospel of John in 21 days — the best starting point for understanding Jesus",
        "beginner",
        21,
    ),
    (
        "psalms-30",
        "30 Days of Psalms",
        "One psalm a day for 30 days — learning to pray and praise with ancient Israel",
        "devotional",
        30,
    ),
    (
        "life-of-jesus",
        "The Life of Jesus",
        "Walk through key events in Jesus' life across all four Gospels",
        "topical",
        28,
    ),
    (
        "wisdom-literature",
        "Ancient Wisdom for Modern Life",
        "Proverbs, Ecclesiastes, and James — practical wisdom for daily decisions",
        "topical",
        21,
    ),
    (
        "redemption-story",
        "The Big Story: Creation to New Creation",
        "Trace the Bible's grand narrative in 14 key passages",
        "overview",
        14,
    ),
]

# Reading plan entries.
# Format: (plan_slug, day, book_id, start_chapter, start_verse, end_chapter, end_verse, title, reflection)
PLAN_ENTRIES: List[Tuple[str, int, int, int, int, int, int, str, str]] = [
    # Start Here (7 days)
    ("start-here", 1, 1, 1, 1, 1, 31, "In the Beginning", "What does this tell you about who God is?"),
    ("start-here", 2, 19, 23, 1, 23, 6, "The Lord Is My Shepherd", "How does God care for you like a shepherd?"),
    ("start-here", 3, 43, 1, 1, 1, 18, "The Word Became Flesh", "What does it mean that Jesus is the 'Word'?"),
    ("start-here", 4, 42, 15, 1, 15, 32, "The Lost Son Returns", "Which character do you identify with most?"),
    ("start-here", 5, 45, 8, 1, 8, 39, "Nothing Can Separate Us", "What fears does this passage address?"),
    ("start-here", 6, 40, 5, 1, 5, 16, "The Beatitudes", "Which beatitude speaks to your situation?"),
    ("start-here", 7, 66, 21, 1, 21, 8, "All Things New", "What gives you hope about the future?"),
    # Redemption Story (14 days)
    ("redemption-story", 1, 1, 1, 1, 2, 3, "Creation", "God creates a good world for humanity to inhabit"),
    ("redemption-story", 2, 1, 3, 1, 3, 24, "The Fall", "Sin enters the world and breaks relationship with God"),
    ("redemption-story", 3, 1, 12, 1, 12, 9, "Call of Abraham", "God begins a rescue plan through one family"),
    ("redemption-story", 4, 2, 14, 1, 14, 31, "The Exodus", "God delivers His people from slavery"),
    ("redemption-story", 5, 2, 20, 1, 20, 21, "The Covenant", "God gives His people a way to live"),
    ("redemption-story", 6, 10, 7, 1, 7, 29, "David's Kingdom", "God promises an everlasting king"),
    ("redemption-story", 7, 23, 53, 1, 53, 12, "Suffering Servant", "The prophets point to a coming savior"),
    ("redemption-story", 8, 42, 2, 1, 2, 20, "Jesus Is Born", "God enters the world as a human baby"),
    ("redemption-story", 9, 40, 5, 1, 7, 29, "Kingdom Teaching", "Jesus announces what God's kingdom looks like"),
    ("redemption-story", 10, 43, 11, 1, 11, 44, "Power Over Death", "Jesus shows he has authority over death itself"),
    ("redemption-story", 11, 42, 22, 14, 23, 56, "Cross and Death", "Jesus gives his life for the world"),
    ("redemption-story", 12, 43, 20, 1, 20, 31, "Resurrection", "Jesus conquers death and rises again"),
    ("redemption-story", 13, 44, 2, 1, 2, 47, "The Church Is Born", "The Spirit empowers God's people for mission"),
    ("redemption-story", 14, 66, 21, 1, 22, 5, "New Creation", "God makes all things new — the story ends in joy"),
    # Life of Jesus (28 days - key events)
    ("life-of-jesus", 1, 42, 1, 26, 2, 20, "Birth Announced and Fulfilled", "The extraordinary enters the ordinary"),
    ("life-of-jesus", 2, 40, 3, 1, 3, 17, "Baptism", "Jesus is publicly identified as God's Son"),
    ("life-of-jesus", 3, 40, 4, 1, 4, 11, "Temptation in the Wilderness", "Jesus faces and overcomes temptation"),
    ("life-of-jesus", 4, 42, 4, 16, 4, 30, "Nazareth Synagogue", "Jesus announces his mission"),
    ("life-of-jesus", 5, 40, 5, 1, 5, 48, "Sermon on the Mount (Part 1)", "The values of God's kingdom"),
    ("life-of-jesus", 6, 40, 6, 1, 6, 34, "Sermon on the Mount (Part 2)", "Trust, prayer, and priorities"),
    ("life-of-jesus", 7, 40, 7, 1, 7, 29, "Sermon on the Mount (Part 3)", "Building on the rock"),
    ("life-of-jesus", 8, 41, 1, 21, 1, 45, "Healing Ministry", "Jesus has authority over disease and demons"),
    ("life-of-jesus", 9, 41, 4, 35, 4, 41, "Calming the Storm", "Who is this that even wind and sea obey?"),
    ("life-of-jesus", 10, 40, 14, 13, 14, 33, "Feeding 5,000 and Walking on Water", "Jesus provides abundantly"),
    ("life-of-jesus", 11, 43, 4, 1, 4, 42, "Woman at the Well", "Jesus breaks social barriers with grace"),
    ("life-of-jesus", 12, 43, 9, 1, 9, 41, "Man Born Blind", "Physical and spiritual sight"),
    ("life-of-jesus", 13, 42, 10, 25, 10, 42, "Good Samaritan and Mary & Martha", "Love in action and attention"),
    ("life-of-jesus", 14, 42, 15, 1, 15, 32, "Lost Sheep, Coin, and Son", "God's joy in finding the lost"),
    ("life-of-jesus", 15, 40, 13, 1, 13, 52, "Parables of the Kingdom", "What God's kingdom is really like"),
    ("life-of-jesus", 16, 43, 11, 1, 11, 44, "Raising of Lazarus", "Jesus is the resurrection and the life"),
    ("life-of-jesus", 17, 41, 8, 27, 9, 13, "Peter's Confession and Transfiguration", "The turning point"),
    ("life-of-jesus", 18, 40, 21, 1, 21, 17, "Triumphal Entry", "The humble king arrives"),
    ("life-of-jesus", 19, 43, 13, 1, 13, 38, "Washing Feet", "The master becomes the servant"),
    ("life-of-jesus", 20, 43, 14, 1, 14, 31, "The Way, Truth, and Life", "Jesus prepares his disciples"),
    ("life-of-jesus", 21, 43, 15, 1, 15, 27, "The True Vine", "Abiding in Jesus"),
    ("life-of-jesus", 22, 43, 17, 1, 17, 26, "Jesus Prays for Us", "Hearing Jesus pray for his followers"),
    ("life-of-jesus", 23, 40, 26, 36, 26, 56, "Gethsemane", "Jesus faces the cross with honesty and obedience"),
    ("life-of-jesus", 24, 43, 18, 1, 19, 16, "Trial", "Injustice and the silence of the Lamb"),
    ("life-of-jesus", 25, 43, 19, 17, 19, 42, "Crucifixion", "It is finished"),
    ("life-of-jesus", 26, 43, 20, 1, 20, 31, "Resurrection", "He is risen!"),
    ("life-of-jesus", 27, 42, 24, 13, 24, 35, "Road to Emmaus", "Recognizing Jesus in the breaking of bread"),
    ("life-of-jesus", 28, 40, 28, 16, 28, 20, "Great Commission", "Go and make disciples"),
]


class ReadingPlanPopulator:
    """Populates reading_plans and reading_plan_entries tables."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def ensure_tables(self) -> None:
        """Create reading plan tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reading_plans (
                    slug TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    estimated_days INTEGER
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reading_plan_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_slug TEXT NOT NULL,
                    day_number INTEGER NOT NULL,
                    book_id INTEGER NOT NULL,
                    start_chapter INTEGER NOT NULL,
                    start_verse INTEGER NOT NULL,
                    end_chapter INTEGER NOT NULL,
                    end_verse INTEGER NOT NULL,
                    title TEXT,
                    reflection_question TEXT,
                    FOREIGN KEY (plan_slug) REFERENCES reading_plans(slug),
                    UNIQUE(plan_slug, day_number)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plan_entries_slug ON reading_plan_entries(plan_slug, day_number)"
            )
            conn.commit()

    def populate(self, force: bool = False) -> Dict[str, int]:
        """Insert reading plans into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Dictionary with counts of inserted rows.
        """
        counts: Dict[str, int] = {"plans": 0, "entries": 0}

        self.ensure_tables()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if force:
                cursor.execute("DELETE FROM reading_plan_entries")
                cursor.execute("DELETE FROM reading_plans")

            for slug, name, description, category, days in READING_PLANS:
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO reading_plans (slug, name, description, category, estimated_days) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (slug, name, description, category, days),
                    )
                    if cursor.rowcount > 0:
                        counts["plans"] += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert plan '%s': %s", slug, e)

            for entry in PLAN_ENTRIES:
                plan_slug, day, book_id, s_ch, s_vs, e_ch, e_vs, title, reflection = entry
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO reading_plan_entries "
                        "(plan_slug, day_number, book_id, start_chapter, start_verse, "
                        "end_chapter, end_verse, title, reflection_question) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (plan_slug, day, book_id, s_ch, s_vs, e_ch, e_vs, title, reflection),
                    )
                    if cursor.rowcount > 0:
                        counts["entries"] += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert plan entry day %d for '%s': %s", day, plan_slug, e)

            conn.commit()

        logger.info("Populated reading plans: %d plans, %d entries", counts["plans"], counts["entries"])
        return counts

    @staticmethod
    def get_plans(db_path: Path) -> List[Dict[str, Any]]:
        """Get all reading plans.

        Args:
            db_path: Path to the database.

        Returns:
            List of reading plan dictionaries.
        """
        plans: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT slug, name, description, category, estimated_days FROM reading_plans ORDER BY slug")
            for row in cursor.fetchall():
                plans.append(
                    {
                        "slug": row[0],
                        "name": row[1],
                        "description": row[2],
                        "category": row[3],
                        "estimated_days": row[4],
                    }
                )
        return plans

    @staticmethod
    def get_plan_entries(db_path: Path, plan_slug: str) -> List[Dict[str, Any]]:
        """Get entries for a specific reading plan.

        Args:
            db_path: Path to the database.
            plan_slug: Plan identifier.

        Returns:
            List of reading plan entry dictionaries.
        """
        entries: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT day_number, book_id, start_chapter, start_verse, "
                "end_chapter, end_verse, title, reflection_question "
                "FROM reading_plan_entries WHERE plan_slug = ? ORDER BY day_number",
                (plan_slug,),
            )
            for row in cursor.fetchall():
                entries.append(
                    {
                        "day_number": row[0],
                        "book_id": row[1],
                        "start_chapter": row[2],
                        "start_verse": row[3],
                        "end_chapter": row[4],
                        "end_verse": row[5],
                        "title": row[6],
                        "reflection_question": row[7],
                    }
                )
        return entries
