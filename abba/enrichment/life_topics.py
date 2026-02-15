"""Life topic population for everyday topical access to Scripture."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Curated life topics with concept mappings and study steps.
LIFE_TOPICS: List[Dict[str, Any]] = [
    {
        "slug": "anxiety",
        "name": "Anxiety & Worry",
        "category": "emotions",
        "description": "What the Bible says about overcoming fear, anxiety, and worry",
        "icon": "heart-pulse",
        "concepts": [("fear_of_god", "Understanding healthy vs unhealthy fear")],
        "study_steps": [
            ("comfort", "Philippians 4:6-7", "God's peace guards your heart when you bring anxieties to Him"),
            ("understanding", "Matthew 6:25-27", "Jesus teaches that worry cannot add a single hour to life"),
            ("guidance", "1 Peter 5:7", "Cast all your anxiety on God because He cares for you"),
            ("hope", "Isaiah 41:10", "Do not fear, for God is with you and will strengthen you"),
        ],
    },
    {
        "slug": "forgiveness",
        "name": "Forgiveness",
        "category": "relationships",
        "description": "Understanding and practicing forgiveness as God forgives us",
        "icon": "hand-heart",
        "concepts": [("forgiveness", "God's model of forgiveness")],
        "study_steps": [
            ("comfort", "1 John 1:9", "If we confess, God is faithful to forgive"),
            ("understanding", "Matthew 18:21-22", "Forgive not seven times but seventy-seven times"),
            ("guidance", "Ephesians 4:32", "Forgive each other as God forgave you in Christ"),
            ("hope", "Psalm 103:12", "As far as east from west, so far has He removed our sins"),
        ],
    },
    {
        "slug": "suffering",
        "name": "Suffering & Pain",
        "category": "struggles",
        "description": "Finding meaning and hope in the midst of suffering",
        "icon": "shield",
        "concepts": [("suffering", "Biblical perspective on suffering")],
        "study_steps": [
            ("comfort", "Psalm 34:18", "The Lord is close to the brokenhearted"),
            ("understanding", "Romans 8:28", "God works all things together for good"),
            ("guidance", "James 1:2-4", "Trials produce perseverance and maturity"),
            ("hope", "Revelation 21:4", "God will wipe every tear; no more death or pain"),
        ],
    },
    {
        "slug": "grief",
        "name": "Grief & Loss",
        "category": "emotions",
        "description": "Comfort and hope when facing loss and mourning",
        "icon": "cloud-rain",
        "concepts": [],
        "study_steps": [
            ("comfort", "Psalm 23:4", "Even in the valley of the shadow of death, God is with you"),
            ("understanding", "John 11:35", "Jesus wept — God understands grief"),
            ("guidance", "1 Thessalonians 4:13", "Grieve with hope, not as those without hope"),
            ("hope", "Psalm 30:5", "Weeping may endure for a night, but joy comes in the morning"),
        ],
    },
    {
        "slug": "purpose",
        "name": "Purpose & Calling",
        "category": "life_stages",
        "description": "Discovering God's purpose for your life",
        "icon": "compass",
        "concepts": [],
        "study_steps": [
            ("comfort", "Jeremiah 29:11", "Plans to prosper you and give you hope and a future"),
            ("understanding", "Ephesians 2:10", "Created in Christ for good works prepared in advance"),
            ("guidance", "Proverbs 3:5-6", "Trust in the Lord and He will direct your paths"),
            ("hope", "Philippians 1:6", "He who began a good work will carry it to completion"),
        ],
    },
    {
        "slug": "loneliness",
        "name": "Loneliness",
        "category": "emotions",
        "description": "God's presence and community when you feel alone",
        "icon": "user",
        "concepts": [],
        "study_steps": [
            ("comfort", "Deuteronomy 31:8", "The Lord goes before you; He will never leave you"),
            ("understanding", "Psalm 139:7-10", "There is nowhere you can go from God's presence"),
            ("guidance", "Hebrews 10:24-25", "Do not give up meeting together; encourage one another"),
            ("hope", "Matthew 28:20", "I am with you always, to the end of the age"),
        ],
    },
    {
        "slug": "anger",
        "name": "Anger",
        "category": "emotions",
        "description": "Understanding righteous anger and managing destructive anger",
        "icon": "flame",
        "concepts": [("wrath_of_god", "Understanding God's anger as righteous")],
        "study_steps": [
            ("comfort", "Psalm 4:4", "Be angry but do not sin; reflect in silence"),
            ("understanding", "James 1:19-20", "Be slow to anger — human anger does not produce God's righteousness"),
            ("guidance", "Ephesians 4:26-27", "Do not let the sun go down on your anger"),
            ("hope", "Proverbs 15:1", "A gentle answer turns away wrath"),
        ],
    },
    {
        "slug": "marriage",
        "name": "Marriage",
        "category": "relationships",
        "description": "Biblical wisdom for building a strong marriage",
        "icon": "rings",
        "concepts": [("marriage", "Biblical view of marriage")],
        "study_steps": [
            ("comfort", "Ecclesiastes 4:9-12", "Two are better than one; a cord of three is not quickly broken"),
            ("understanding", "Ephesians 5:25-28", "Love as Christ loved the church — sacrificial love"),
            ("guidance", "1 Corinthians 13:4-7", "Love is patient, love is kind"),
            ("hope", "Genesis 2:24", "The two shall become one flesh"),
        ],
    },
    {
        "slug": "temptation",
        "name": "Temptation",
        "category": "struggles",
        "description": "Resisting temptation and overcoming sin patterns",
        "icon": "shield-alert",
        "concepts": [("sexual_sin", "Resisting sexual temptation")],
        "study_steps": [
            ("comfort", "1 Corinthians 10:13", "God will not let you be tempted beyond what you can bear"),
            ("understanding", "James 1:13-15", "Temptation comes from our own desires, not from God"),
            ("guidance", "Hebrews 4:15-16", "Jesus was tempted yet without sin; approach the throne boldly"),
            ("hope", "Romans 6:14", "Sin shall not be your master; you are under grace"),
        ],
    },
    {
        "slug": "money",
        "name": "Money & Possessions",
        "category": "practical",
        "description": "Biblical wisdom about money, generosity, and contentment",
        "icon": "wallet",
        "concepts": [],
        "study_steps": [
            ("comfort", "Philippians 4:19", "God will supply every need according to His riches"),
            ("understanding", "Matthew 6:19-21", "Store up treasures in heaven, not on earth"),
            ("guidance", "Proverbs 3:9-10", "Honor the Lord with your wealth"),
            ("hope", "1 Timothy 6:6-8", "Godliness with contentment is great gain"),
        ],
    },
    {
        "slug": "identity",
        "name": "Identity in Christ",
        "category": "faith",
        "description": "Understanding who you are as a child of God",
        "icon": "fingerprint",
        "concepts": [("salvation", "Being made new in Christ")],
        "study_steps": [
            ("comfort", "2 Corinthians 5:17", "If anyone is in Christ, they are a new creation"),
            ("understanding", "Ephesians 1:3-6", "Chosen, adopted, and accepted in Christ"),
            ("guidance", "Romans 8:16-17", "The Spirit testifies that we are children of God"),
            ("hope", "1 Peter 2:9", "You are a chosen people, a royal priesthood"),
        ],
    },
    {
        "slug": "prayer",
        "name": "Prayer",
        "category": "faith",
        "description": "Learning to pray and deepen your conversation with God",
        "icon": "message-circle",
        "concepts": [("prayer", "Biblical models of prayer")],
        "study_steps": [
            ("comfort", "Psalm 145:18", "The Lord is near to all who call on Him in truth"),
            ("understanding", "Matthew 6:9-13", "Jesus teaches the disciples how to pray"),
            ("guidance", "1 Thessalonians 5:17", "Pray continually"),
            ("hope", "Romans 8:26", "The Spirit intercedes for us with groans beyond words"),
        ],
    },
]


class LifeTopicPopulator:
    """Populates life_topics, life_topic_concepts, and topic_study_steps tables."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> Dict[str, int]:
        """Insert life topics into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Dictionary with counts of inserted rows per table.
        """
        counts = {"life_topics": 0, "life_topic_concepts": 0, "topic_study_steps": 0}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Ensure tables exist
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='life_topics'")
            if cursor.fetchone()[0] == 0:
                logger.warning("life_topics table does not exist; run migrations first")
                return counts

            if force:
                cursor.execute("DELETE FROM topic_study_steps")
                cursor.execute("DELETE FROM life_topic_concepts")
                cursor.execute("DELETE FROM life_topics")

            for i, topic in enumerate(LIFE_TOPICS):
                try:
                    self._insert_single_topic(cursor, topic, i, counts)
                except sqlite3.Error as e:
                    logger.error("Failed to insert topic '%s': %s", topic["slug"], e)

            conn.commit()

        logger.info(
            "Populated life topics: %d topics, %d concept links, %d study steps",
            counts["life_topics"],
            counts["life_topic_concepts"],
            counts["topic_study_steps"],
        )
        return counts

    @staticmethod
    def _insert_single_topic(
        cursor: sqlite3.Cursor,
        topic: Dict[str, Any],
        display_order: int,
        counts: Dict[str, int],
    ) -> None:
        """Insert a single topic with its concepts and study steps."""
        cursor.execute(
            "INSERT OR IGNORE INTO life_topics (slug, name, category, description, icon, display_order) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (topic["slug"], topic["name"], topic["category"], topic["description"], topic.get("icon"), display_order),
        )
        if cursor.rowcount > 0:
            counts["life_topics"] += 1

        # Get the topic ID
        cursor.execute("SELECT id FROM life_topics WHERE slug = ?", (topic["slug"],))
        topic_row = cursor.fetchone()
        if not topic_row:
            return
        topic_id = topic_row[0]

        # Insert concept mappings
        for j, (concept_name, aspect) in enumerate(topic.get("concepts", [])):
            cursor.execute(
                "INSERT OR IGNORE INTO life_topic_concepts "
                "(topic_id, concept_name, relevance_aspect, display_order) "
                "VALUES (?, ?, ?, ?)",
                (topic_id, concept_name, aspect, j),
            )
            if cursor.rowcount > 0:
                counts["life_topic_concepts"] += 1

        # Insert study steps (check existing to avoid duplicates — no unique constraint)
        for step_order, (step_type, verse_ref, insight) in enumerate(topic.get("study_steps", [])):
            cursor.execute(
                "SELECT COUNT(*) FROM topic_study_steps WHERE topic_id = ? AND step_order = ?",
                (topic_id, step_order),
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO topic_study_steps "
                    "(topic_id, step_order, step_type, verse_reference, insight) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (topic_id, step_order, step_type, verse_ref, insight),
                )
                if cursor.rowcount > 0:
                    counts["topic_study_steps"] += 1
