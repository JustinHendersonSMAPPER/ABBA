"""Literary structure annotations for biblical passages."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Curated literary structures.
# Format: (book_id, s_ch, s_vs, e_ch, e_vs, structure_type, description, significance, elements_json, source)
CURATED_STRUCTURES: List[Tuple[int, int, int, int, int, str, str, str, str, str]] = [
    # Genesis - Flood narrative chiasmus
    (
        1,
        6,
        10,
        9,
        19,
        "chiasmus",
        "The Flood narrative is structured as an elaborate chiasm centered on 'God remembered Noah'",
        "The chiastic center (8:1) highlights God's covenant faithfulness as the theological core",
        json.dumps(
            [
                {"label": "A", "ref": "6:10", "text": "Noah's three sons"},
                {"label": "B", "ref": "6:22-7:5", "text": "Noah enters ark"},
                {"label": "C", "ref": "7:11", "text": "Flood begins"},
                {"label": "D", "ref": "7:24", "text": "150 days waters prevail"},
                {"label": "E", "ref": "8:1", "text": "GOD REMEMBERED NOAH (center)"},
                {"label": "D'", "ref": "8:3", "text": "150 days waters recede"},
                {"label": "C'", "ref": "8:13", "text": "Earth dried"},
                {"label": "B'", "ref": "8:18", "text": "Noah exits ark"},
                {"label": "A'", "ref": "9:18-19", "text": "Noah's three sons"},
            ]
        ),
        "Wenham, Genesis 1-15 (WBC)",
    ),
    # Psalm 8 - Chiastic structure
    (
        19,
        8,
        1,
        8,
        9,
        "chiasmus",
        "Psalm 8 forms a chiasm with the declaration of God's majesty as inclusio",
        "The chiastic structure centers on humanity's paradoxical smallness and glory",
        json.dumps(
            [
                {"label": "A", "ref": "8:1a", "text": "O LORD, how majestic is your name"},
                {"label": "B", "ref": "8:1b-2", "text": "Glory above the heavens"},
                {"label": "C", "ref": "8:3-4", "text": "What is man?"},
                {"label": "B'", "ref": "8:5-8", "text": "Crowned with glory, dominion"},
                {"label": "A'", "ref": "8:9", "text": "O LORD, how majestic is your name"},
            ]
        ),
        "Craigie, Psalms 1-50 (WBC)",
    ),
    # Psalm 119 - Acrostic
    (
        19,
        119,
        1,
        119,
        176,
        "acrostic",
        "Each of the 22 sections begins with a successive letter of the Hebrew alphabet",
        "The acrostic form celebrates the completeness of God's word from aleph to tav (A to Z)",
        json.dumps(
            [
                {"label": "Aleph", "ref": "119:1-8", "text": "Blessed are those whose way is blameless"},
                {"label": "Beth", "ref": "119:9-16", "text": "How can a young man keep his way pure?"},
                {"label": "Gimel", "ref": "119:17-24", "text": "Open my eyes to behold wondrous things"},
                {"label": "Tav", "ref": "119:169-176", "text": "Let my cry come before you, O LORD"},
            ]
        ),
        "Allen, Psalms 101-150 (WBC)",
    ),
    # Lamentations - Acrostic structure
    (
        25,
        1,
        1,
        5,
        22,
        "acrostic",
        "Lamentations 1-4 are acrostic poems; chapter 3 is a triple acrostic",
        "The acrostic form imposes order on the chaos of grief, containing lament within structure",
        json.dumps(
            [
                {"label": "Ch 1", "ref": "1:1-22", "text": "Single acrostic — Jerusalem personified"},
                {"label": "Ch 2", "ref": "2:1-22", "text": "Single acrostic — God's anger"},
                {"label": "Ch 3", "ref": "3:1-66", "text": "Triple acrostic — individual suffering and hope"},
                {"label": "Ch 4", "ref": "4:1-22", "text": "Single acrostic — siege horrors"},
                {"label": "Ch 5", "ref": "5:1-22", "text": "Non-acrostic — communal prayer (22 verses)"},
            ]
        ),
        "Provan, Lamentations (NCB)",
    ),
    # Proverbs 31:10-31 - Acrostic poem
    (
        20,
        31,
        10,
        31,
        31,
        "acrostic",
        "The Virtuous Woman poem is an acrostic spanning the Hebrew alphabet",
        "The acrostic celebrates the completeness of this woman's virtue — from A to Z",
        json.dumps(
            [
                {"label": "Aleph", "ref": "31:10", "text": "A wife of noble character who can find?"},
                {"label": "Tav", "ref": "31:31", "text": "Let her works praise her in the gates"},
            ]
        ),
        "Waltke, Proverbs 15-31 (NICOT)",
    ),
    # Isaiah 5 - Song of the Vineyard (parallelism)
    (
        23,
        5,
        1,
        5,
        7,
        "parallelism",
        "The Song of the Vineyard uses extended metaphor with synonymous parallelism",
        "Israel is God's vineyard; the parable form invites self-judgment before revealing the referent",
        json.dumps(
            [
                {"label": "A", "ref": "5:1-2", "text": "Vineyard preparation (tender care)"},
                {"label": "B", "ref": "5:3-4", "text": "Appeal to audience to judge"},
                {"label": "C", "ref": "5:5-6", "text": "Judgment pronounced on vineyard"},
                {"label": "D", "ref": "5:7", "text": "Interpretation: Israel is the vineyard"},
            ]
        ),
        "Oswalt, Isaiah 1-39 (NICOT)",
    ),
    # Amos 1-2 - Oracles against nations (pattern)
    (
        30,
        1,
        3,
        2,
        16,
        "inclusio",
        "Amos uses a 'for three transgressions, for four' pattern spiraling from nations to Israel",
        "The rhetorical pattern traps the audience — they agree God judges other nations, then discover they too are judged",
        json.dumps(
            [
                {"label": "1", "ref": "1:3", "text": "Damascus — for three transgressions, for four"},
                {"label": "2", "ref": "1:6", "text": "Gaza"},
                {"label": "3", "ref": "1:9", "text": "Tyre"},
                {"label": "4", "ref": "1:11", "text": "Edom"},
                {"label": "5", "ref": "1:13", "text": "Ammon"},
                {"label": "6", "ref": "2:1", "text": "Moab"},
                {"label": "7", "ref": "2:4", "text": "Judah"},
                {"label": "8", "ref": "2:6", "text": "ISRAEL (climax — the intended target)"},
            ]
        ),
        "Andersen & Freedman, Amos (AB)",
    ),
    # Matthew 5-7 Sermon on the Mount structure
    (
        40,
        5,
        1,
        7,
        29,
        "chiasmus",
        "The Sermon on the Mount has a chiastic structure centered on the Lord's Prayer",
        "The center (Lord's Prayer, 6:9-13) reveals that right relationship with God enables the ethical teaching",
        json.dumps(
            [
                {"label": "A", "ref": "5:3-16", "text": "Blessings of kingdom living"},
                {"label": "B", "ref": "5:17-48", "text": "Greater righteousness (6 antitheses)"},
                {"label": "C", "ref": "6:1-18", "text": "True piety — CENTER: Lord's Prayer (6:9-13)"},
                {"label": "B'", "ref": "6:19-7:12", "text": "Kingdom priorities"},
                {"label": "A'", "ref": "7:13-27", "text": "Warnings and call to action"},
            ]
        ),
        "Allison, Sermon on the Mount (1999)",
    ),
    # John's Prologue - Chiasmus
    (
        43,
        1,
        1,
        1,
        18,
        "chiasmus",
        "John's Prologue is structured as a chiasm centered on the incarnation",
        "The chiastic center (vv. 12-13) highlights receiving the Word and becoming children of God",
        json.dumps(
            [
                {"label": "A", "ref": "1:1", "text": "The Word with God"},
                {"label": "B", "ref": "1:2-5", "text": "Role in creation, life, light"},
                {"label": "C", "ref": "1:6-8", "text": "John's witness"},
                {"label": "D", "ref": "1:9-11", "text": "Coming into the world, rejected"},
                {"label": "E", "ref": "1:12-13", "text": "RECEIVING THE WORD — children of God"},
                {"label": "D'", "ref": "1:14", "text": "The Word became flesh, dwelt among us"},
                {"label": "C'", "ref": "1:15", "text": "John's witness"},
                {"label": "B'", "ref": "1:16-17", "text": "Grace and truth received"},
                {"label": "A'", "ref": "1:18", "text": "The only Son reveals God"},
            ]
        ),
        "Culpepper, The Gospel and Letters of John (1998)",
    ),
    # Philippians 2:5-11 Christ Hymn
    (
        50,
        2,
        5,
        2,
        11,
        "chiasmus",
        "The Christ Hymn follows a V-shaped descent/ascent pattern",
        "Christ's self-emptying descent leads to God's exaltation — the pattern for Christian life",
        json.dumps(
            [
                {"label": "A", "ref": "2:6", "text": "Form of God — equality with God"},
                {"label": "B", "ref": "2:7a", "text": "Emptied himself"},
                {"label": "C", "ref": "2:7b", "text": "Likeness of humanity"},
                {"label": "D", "ref": "2:8", "text": "DEATH ON A CROSS (lowest point)"},
                {"label": "C'", "ref": "2:9a", "text": "Highly exalted"},
                {"label": "B'", "ref": "2:9b", "text": "Name above every name"},
                {"label": "A'", "ref": "2:10-11", "text": "Every knee bows — Jesus is Lord"},
            ]
        ),
        "Fee, Philippians (NICNT)",
    ),
    # Hebrews 11 - Exempla pattern
    (
        58,
        11,
        1,
        11,
        40,
        "parallelism",
        "The 'Hall of Faith' uses the repeated 'by faith' anaphora to build a cumulative argument",
        "The repeated pattern creates momentum — faith has always been the way God's people live",
        json.dumps(
            [
                {"label": "Introduction", "ref": "11:1-3", "text": "Definition of faith"},
                {"label": "1", "ref": "11:4", "text": "By faith Abel..."},
                {"label": "2", "ref": "11:5", "text": "By faith Enoch..."},
                {"label": "3", "ref": "11:7", "text": "By faith Noah..."},
                {"label": "4", "ref": "11:8", "text": "By faith Abraham..."},
                {"label": "5", "ref": "11:20-22", "text": "By faith Isaac, Jacob, Joseph..."},
                {"label": "6", "ref": "11:23-29", "text": "By faith Moses..."},
                {"label": "Climax", "ref": "11:32-40", "text": "And what more shall I say?"},
            ]
        ),
        "Lane, Hebrews 9-13 (WBC)",
    ),
    # Revelation 4-5 - Heavenly throne room (inclusio)
    (
        66,
        4,
        1,
        5,
        14,
        "inclusio",
        "The throne room vision forms an inclusio with worship framing the revelation of the Lamb",
        "The scene establishes God's sovereign right to judge and the Lamb's worthiness to open the scroll",
        json.dumps(
            [
                {"label": "A", "ref": "4:1-3", "text": "Throne in heaven — one seated"},
                {"label": "B", "ref": "4:4-8", "text": "24 elders, 4 living creatures worship"},
                {"label": "C", "ref": "4:9-11", "text": "Worship: 'Worthy are you, our Lord and God'"},
                {"label": "D", "ref": "5:1-5", "text": "Sealed scroll — who is worthy?"},
                {"label": "C'", "ref": "5:6-10", "text": "The Lamb — 'Worthy are you'"},
                {"label": "B'", "ref": "5:11-13", "text": "Myriads worship the Lamb"},
                {"label": "A'", "ref": "5:14", "text": "The elders fell down and worshiped"},
            ]
        ),
        "Bauckham, The Theology of the Book of Revelation (1993)",
    ),
]


class LiteraryStructurePopulator:
    """Populates the literary_structures table."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert literary structures into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='literary_structures'")
            if cursor.fetchone()[0] == 0:
                logger.warning("literary_structures table does not exist; run migrations first")
                return 0

            if force:
                cursor.execute("DELETE FROM literary_structures")

            for struct in CURATED_STRUCTURES:
                book_id, s_ch, s_vs, e_ch, e_vs, stype, desc, significance, elements, source = struct
                try:
                    cursor.execute(
                        "SELECT COUNT(*) FROM literary_structures "
                        "WHERE book_id = ? AND start_chapter = ? AND start_verse = ? AND structure_type = ?",
                        (book_id, s_ch, s_vs, stype),
                    )
                    if cursor.fetchone()[0] > 0:
                        continue
                    cursor.execute(
                        "INSERT INTO literary_structures "
                        "(book_id, start_chapter, start_verse, end_chapter, end_verse, "
                        "structure_type, description, significance, elements, scholarly_source) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (book_id, s_ch, s_vs, e_ch, e_vs, stype, desc, significance, elements, source),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert literary structure for book %d: %s", book_id, e)

            conn.commit()

        logger.info("Populated literary_structures: %d rows inserted", inserted)
        return inserted

    @staticmethod
    def get_structures_for_verse(db_path: Path, book_id: int, chapter: int, verse: int) -> List[Dict[str, Any]]:
        """Get literary structures containing a specific verse.

        Args:
            db_path: Path to the database.
            book_id: Book ID.
            chapter: Chapter number.
            verse: Verse number.

        Returns:
            List of literary structure dictionaries.
        """
        structures: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT structure_id, structure_type, description, significance, elements, scholarly_source "
                "FROM literary_structures "
                "WHERE book_id = ? "
                "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
                "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) ",
                (book_id, chapter, chapter, verse, chapter, chapter, verse),
            )
            for row in cursor.fetchall():
                elements = []
                if row[4]:
                    try:
                        elements = json.loads(row[4])
                    except (json.JSONDecodeError, TypeError):
                        pass
                structures.append(
                    {
                        "structure_type": row[1],
                        "description": row[2],
                        "significance": row[3],
                        "elements": elements,
                        "scholarly_source": row[5],
                    }
                )
        return structures
