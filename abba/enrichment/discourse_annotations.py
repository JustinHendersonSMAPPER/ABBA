"""OpenText.org discourse annotation data for biblical passages."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Representative discourse annotations for key biblical passages.
# Based on OpenText.org discourse analysis methodology.
# Fields: book_id, start_chapter, start_verse, end_chapter, end_verse,
#         discourse_type (narrative/argument/exposition/hymn/dialogue),
#         function_label, relation_to_context (continuation/contrast/cause/result/elaboration),
#         description, prominence (0-3, where 3 is highest)
DISCOURSE_ANNOTATIONS: List[Dict[str, Any]] = [
    # =========================================================================
    # Genesis 1:1-2:3 — Creation Narrative
    # =========================================================================
    {
        "book_id": 1,
        "start_chapter": 1,
        "start_verse": 1,
        "end_chapter": 1,
        "end_verse": 2,
        "discourse_type": "narrative",
        "function_label": "narrative backbone",
        "relation_to_context": "continuation",
        "description": "Opening cosmological setting establishing God as sole creator; introduces the "
        "state of primordial chaos (tohu vabohu) before divine creative speech acts begin",
        "prominence": 3,
    },
    {
        "book_id": 1,
        "start_chapter": 1,
        "start_verse": 3,
        "end_chapter": 1,
        "end_verse": 5,
        "discourse_type": "dialogue",
        "function_label": "divine speech act",
        "relation_to_context": "result",
        "description": "First creative speech act: God commands light into existence. The pattern of "
        "command-fulfillment-evaluation-naming structures all subsequent creation days",
        "prominence": 3,
    },
    {
        "book_id": 1,
        "start_chapter": 1,
        "start_verse": 6,
        "end_chapter": 1,
        "end_verse": 31,
        "discourse_type": "narrative",
        "function_label": "structured repetition",
        "relation_to_context": "continuation",
        "description": "Six-day creation sequence following the recurring formula: divine speech, "
        "fulfillment, evaluation ('it was good'), and temporal marker ('evening and morning')",
        "prominence": 2,
    },
    {
        "book_id": 1,
        "start_chapter": 1,
        "start_verse": 26,
        "end_chapter": 1,
        "end_verse": 28,
        "discourse_type": "dialogue",
        "function_label": "divine speech act",
        "relation_to_context": "elaboration",
        "description": "Climactic divine deliberation ('Let us make') marking humanity's creation as "
        "the pinnacle of the creation narrative; shift from 'good' to 'very good'",
        "prominence": 3,
    },
    {
        "book_id": 1,
        "start_chapter": 2,
        "start_verse": 1,
        "end_chapter": 2,
        "end_verse": 3,
        "discourse_type": "narrative",
        "function_label": "closure",
        "relation_to_context": "result",
        "description": "Sabbath rest as theological conclusion to creation; God's cessation from "
        "work establishes the pattern of sacred time and the seven-day week",
        "prominence": 2,
    },
    # =========================================================================
    # John 1:1-18 — Prologue
    # =========================================================================
    {
        "book_id": 43,
        "start_chapter": 1,
        "start_verse": 1,
        "end_chapter": 1,
        "end_verse": 5,
        "discourse_type": "hymn",
        "function_label": "hymnic declaration",
        "relation_to_context": "continuation",
        "description": "Pre-existence and cosmic role of the Logos declared in elevated poetic style; "
        "echoes Genesis 1:1 to establish the Word's role in creation and as source of life and light",
        "prominence": 3,
    },
    {
        "book_id": 43,
        "start_chapter": 1,
        "start_verse": 6,
        "end_chapter": 1,
        "end_verse": 8,
        "discourse_type": "narrative",
        "function_label": "historical parenthesis",
        "relation_to_context": "elaboration",
        "description": "Narrative intrusion introducing John the Baptist as witness to the light; "
        "shifts from hymnic to historical mode to ground the theological in history",
        "prominence": 1,
    },
    {
        "book_id": 43,
        "start_chapter": 1,
        "start_verse": 9,
        "end_chapter": 1,
        "end_verse": 13,
        "discourse_type": "exposition",
        "function_label": "theological exposition",
        "relation_to_context": "contrast",
        "description": "The true light enters the world: rejection by 'his own' contrasted with reception "
        "by those given authority to become children of God. Central theological pivot of the prologue",
        "prominence": 3,
    },
    {
        "book_id": 43,
        "start_chapter": 1,
        "start_verse": 14,
        "end_chapter": 1,
        "end_verse": 14,
        "discourse_type": "hymn",
        "function_label": "incarnation declaration",
        "relation_to_context": "result",
        "description": "The incarnation statement ('the Word became flesh') — the theological climax "
        "of the prologue. The verb 'became' (egeneto) contrasts with the 'was' (en) of v.1",
        "prominence": 3,
    },
    {
        "book_id": 43,
        "start_chapter": 1,
        "start_verse": 15,
        "end_chapter": 1,
        "end_verse": 18,
        "discourse_type": "exposition",
        "function_label": "theological exposition",
        "relation_to_context": "elaboration",
        "description": "Concluding theological reflection: the community's witness to grace and truth "
        "received from the incarnate Word; the unique Son reveals the unseen Father",
        "prominence": 2,
    },
    # =========================================================================
    # Romans 8:1-39 — No Condemnation
    # =========================================================================
    {
        "book_id": 45,
        "start_chapter": 8,
        "start_verse": 1,
        "end_chapter": 8,
        "end_verse": 4,
        "discourse_type": "argument",
        "function_label": "thesis statement",
        "relation_to_context": "contrast",
        "description": "Opening declaration of 'no condemnation' for those in Christ Jesus; establishes "
        "the thesis contrasting the law of the Spirit with the law of sin and death",
        "prominence": 3,
    },
    {
        "book_id": 45,
        "start_chapter": 8,
        "start_verse": 5,
        "end_chapter": 8,
        "end_verse": 17,
        "discourse_type": "argument",
        "function_label": "logical argument",
        "relation_to_context": "cause",
        "description": "Sustained argument contrasting flesh and Spirit: those led by the Spirit are "
        "children of God and co-heirs with Christ. Builds from indicative to participatory identity",
        "prominence": 2,
    },
    {
        "book_id": 45,
        "start_chapter": 8,
        "start_verse": 18,
        "end_chapter": 8,
        "end_verse": 25,
        "discourse_type": "argument",
        "function_label": "eschatological hope",
        "relation_to_context": "contrast",
        "description": "Present suffering contrasted with future glory; creation's groaning and the "
        "believer's groaning both point to the hope of final redemption and adoption",
        "prominence": 2,
    },
    {
        "book_id": 45,
        "start_chapter": 8,
        "start_verse": 26,
        "end_chapter": 8,
        "end_verse": 30,
        "discourse_type": "argument",
        "function_label": "theological grounding",
        "relation_to_context": "elaboration",
        "description": "The Spirit's intercession and the golden chain of salvation (foreknew, "
        "predestined, called, justified, glorified) ground hope in God's sovereign purpose",
        "prominence": 2,
    },
    {
        "book_id": 45,
        "start_chapter": 8,
        "start_verse": 31,
        "end_chapter": 8,
        "end_verse": 39,
        "discourse_type": "argument",
        "function_label": "rhetorical climax",
        "relation_to_context": "result",
        "description": "Triumphant rhetorical climax with five unanswerable questions ('If God is for us, "
        "who can be against us?') culminating in the declaration that nothing can separate from God's love",
        "prominence": 3,
    },
    # =========================================================================
    # Psalm 23 — The Shepherd Psalm
    # =========================================================================
    {
        "book_id": 19,
        "start_chapter": 23,
        "start_verse": 1,
        "end_chapter": 23,
        "end_verse": 3,
        "discourse_type": "hymn",
        "function_label": "metaphorical description",
        "relation_to_context": "continuation",
        "description": "Extended shepherd metaphor in third person ('He leads me'): God provides rest, "
        "restoration, and right paths. The psalmist speaks about God as shepherd-provider",
        "prominence": 2,
    },
    {
        "book_id": 19,
        "start_chapter": 23,
        "start_verse": 4,
        "end_chapter": 23,
        "end_verse": 4,
        "discourse_type": "hymn",
        "function_label": "trust declaration",
        "relation_to_context": "contrast",
        "description": "Shift from third person to second person ('You are with me') at the darkest point "
        "of the psalm. The intimacy of address increases as the danger intensifies — theological center",
        "prominence": 3,
    },
    {
        "book_id": 19,
        "start_chapter": 23,
        "start_verse": 5,
        "end_chapter": 23,
        "end_verse": 6,
        "discourse_type": "hymn",
        "function_label": "trust declaration",
        "relation_to_context": "result",
        "description": "Metaphor shifts from shepherd to host: God prepares a table in enemy presence. "
        "Concludes with confident declaration of dwelling in God's house forever",
        "prominence": 2,
    },
]


class DiscourseAnnotationPopulator:
    """Populates the discourse_annotations table with passage-level discourse analysis."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _ensure_table(self, cursor: sqlite3.Cursor) -> bool:
        """Create the discourse_annotations table if it does not exist.

        Returns:
            True if table is ready, False otherwise.
        """
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS discourse_annotations (
                annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                start_chapter INTEGER NOT NULL,
                start_verse INTEGER NOT NULL,
                end_chapter INTEGER NOT NULL,
                end_verse INTEGER NOT NULL,
                discourse_type TEXT NOT NULL,
                function_label TEXT NOT NULL,
                relation_to_context TEXT,
                description TEXT,
                prominence INTEGER DEFAULT 0,
                UNIQUE(book_id, start_chapter, start_verse, end_chapter, end_verse, discourse_type)
            )
            """
        )
        return True

    def populate(self, force: bool = False) -> int:
        """Insert discourse annotations into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            self._ensure_table(cursor)

            if force:
                cursor.execute("DELETE FROM discourse_annotations")

            for annotation in DISCOURSE_ANNOTATIONS:
                try:
                    verb = "INSERT OR REPLACE" if force else "INSERT OR IGNORE"
                    cursor.execute(
                        f"{verb} INTO discourse_annotations "  # noqa: S608
                        "(book_id, start_chapter, start_verse, end_chapter, end_verse, "
                        "discourse_type, function_label, relation_to_context, description, prominence) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            annotation["book_id"],
                            annotation["start_chapter"],
                            annotation["start_verse"],
                            annotation["end_chapter"],
                            annotation["end_verse"],
                            annotation["discourse_type"],
                            annotation["function_label"],
                            annotation["relation_to_context"],
                            annotation["description"],
                            annotation["prominence"],
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error(
                        "Failed to insert discourse annotation for book %d %d:%d: %s",
                        annotation["book_id"],
                        annotation["start_chapter"],
                        annotation["start_verse"],
                        e,
                    )

            conn.commit()

        logger.info("Populated discourse_annotations: %d rows inserted", inserted)
        return inserted
