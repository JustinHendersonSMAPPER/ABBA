"""MACULA treebank clause-level syntax tree data for biblical passages."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Curated syntax tree data for key biblical verses.
# Based on MACULA Greek/Hebrew treebank clause-level analysis.
# Each node represents a syntactic unit within a verse's parse tree.
# Fields: book_id, chapter, verse, word_num (nullable for clause/phrase nodes),
#         node_type (sentence/clause/phrase/word), role (subject/predicate/object/modifier),
#         parent_id (reference to parent node, null for root), clause_type (main/temporal/relative/causal),
#         relation (description of syntactic relationship), depth (nesting level), text_content
SAMPLE_SYNTAX_TREES: List[Dict[str, Any]] = [
    # =========================================================================
    # Genesis 1:1 — "In the beginning God created the heavens and the earth"
    # Structure: Sentence -> Temporal-clause + Main-clause + Object-clause
    # =========================================================================
    # Root sentence node
    {
        "node_id": "gen_1_1_s1",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "sentence",
        "role": None,
        "parent_id": None,
        "clause_type": None,
        "relation": "root",
        "depth": 0,
        "text_content": "bereshit bara elohim et hashamayim ve'et ha'aretz",
    },
    # Temporal clause: "In the beginning"
    {
        "node_id": "gen_1_1_c1",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "clause",
        "role": "modifier",
        "parent_id": "gen_1_1_s1",
        "clause_type": "temporal",
        "relation": "temporal adjunct",
        "depth": 1,
        "text_content": "bereshit",
    },
    # Temporal clause word: bereshit
    {
        "node_id": "gen_1_1_w1",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": 1,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "gen_1_1_c1",
        "clause_type": None,
        "relation": "temporal prepositional phrase",
        "depth": 2,
        "text_content": "bereshit",
    },
    # Main clause: "God created"
    {
        "node_id": "gen_1_1_c2",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "clause",
        "role": None,
        "parent_id": "gen_1_1_s1",
        "clause_type": "main",
        "relation": "main clause",
        "depth": 1,
        "text_content": "bara elohim",
    },
    # Predicate: bara (created)
    {
        "node_id": "gen_1_1_w2",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": 2,
        "node_type": "word",
        "role": "predicate",
        "parent_id": "gen_1_1_c2",
        "clause_type": None,
        "relation": "verb (qal perfect 3ms)",
        "depth": 2,
        "text_content": "bara",
    },
    # Subject: elohim (God)
    {
        "node_id": "gen_1_1_w3",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": 3,
        "node_type": "word",
        "role": "subject",
        "parent_id": "gen_1_1_c2",
        "clause_type": None,
        "relation": "nominative subject",
        "depth": 2,
        "text_content": "elohim",
    },
    # Object clause: "the heavens and the earth"
    {
        "node_id": "gen_1_1_c3",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "clause",
        "role": "object",
        "parent_id": "gen_1_1_s1",
        "clause_type": "main",
        "relation": "object clause",
        "depth": 1,
        "text_content": "et hashamayim ve'et ha'aretz",
    },
    # Object marker + heavens phrase
    {
        "node_id": "gen_1_1_p1",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "phrase",
        "role": "object",
        "parent_id": "gen_1_1_c3",
        "clause_type": None,
        "relation": "direct object (accusative)",
        "depth": 2,
        "text_content": "et hashamayim",
    },
    # Word: et (object marker)
    {
        "node_id": "gen_1_1_w4",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": 4,
        "node_type": "word",
        "role": "object",
        "parent_id": "gen_1_1_p1",
        "clause_type": None,
        "relation": "accusative marker",
        "depth": 3,
        "text_content": "et",
    },
    # Word: hashamayim (the heavens)
    {
        "node_id": "gen_1_1_w5",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": 5,
        "node_type": "word",
        "role": "object",
        "parent_id": "gen_1_1_p1",
        "clause_type": None,
        "relation": "noun (masculine plural with article)",
        "depth": 3,
        "text_content": "hashamayim",
    },
    # Conjunction + earth phrase
    {
        "node_id": "gen_1_1_p2",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "phrase",
        "role": "object",
        "parent_id": "gen_1_1_c3",
        "clause_type": None,
        "relation": "coordinated direct object",
        "depth": 2,
        "text_content": "ve'et ha'aretz",
    },
    # Word: ve'et (and + object marker)
    {
        "node_id": "gen_1_1_w6",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": 6,
        "node_type": "word",
        "role": "object",
        "parent_id": "gen_1_1_p2",
        "clause_type": None,
        "relation": "conjunction + accusative marker",
        "depth": 3,
        "text_content": "ve'et",
    },
    # Word: ha'aretz (the earth)
    {
        "node_id": "gen_1_1_w7",
        "book_id": 1,
        "chapter": 1,
        "verse": 1,
        "word_num": 7,
        "node_type": "word",
        "role": "object",
        "parent_id": "gen_1_1_p2",
        "clause_type": None,
        "relation": "noun (feminine singular with article)",
        "depth": 3,
        "text_content": "ha'aretz",
    },
    # =========================================================================
    # John 1:1 — "In the beginning was the Word, and the Word was with God,
    #              and the Word was God"
    # Structure: Three coordinate clauses
    # =========================================================================
    # Root sentence node
    {
        "node_id": "jhn_1_1_s1",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "sentence",
        "role": None,
        "parent_id": None,
        "clause_type": None,
        "relation": "root",
        "depth": 0,
        "text_content": "en arche en ho logos kai ho logos en pros ton theon kai theos en ho logos",
    },
    # Clause 1: "en arche en ho logos" (In the beginning was the Word)
    {
        "node_id": "jhn_1_1_c1",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "clause",
        "role": None,
        "parent_id": "jhn_1_1_s1",
        "clause_type": "main",
        "relation": "main clause",
        "depth": 1,
        "text_content": "en arche en ho logos",
    },
    # Prepositional phrase: en arche
    {
        "node_id": "jhn_1_1_p1",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "phrase",
        "role": "modifier",
        "parent_id": "jhn_1_1_c1",
        "clause_type": None,
        "relation": "temporal prepositional phrase",
        "depth": 2,
        "text_content": "en arche",
    },
    # Word: en (in)
    {
        "node_id": "jhn_1_1_w1",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 1,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "jhn_1_1_p1",
        "clause_type": None,
        "relation": "preposition",
        "depth": 3,
        "text_content": "en",
    },
    # Word: arche (beginning)
    {
        "node_id": "jhn_1_1_w2",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 2,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "jhn_1_1_p1",
        "clause_type": None,
        "relation": "noun (dative feminine singular)",
        "depth": 3,
        "text_content": "arche",
    },
    # Word: en (was) — predicate of clause 1
    {
        "node_id": "jhn_1_1_w3",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 3,
        "node_type": "word",
        "role": "predicate",
        "parent_id": "jhn_1_1_c1",
        "clause_type": None,
        "relation": "verb (imperfect active indicative 3s)",
        "depth": 2,
        "text_content": "en",
    },
    # Subject phrase: ho logos
    {
        "node_id": "jhn_1_1_p2",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "phrase",
        "role": "subject",
        "parent_id": "jhn_1_1_c1",
        "clause_type": None,
        "relation": "nominative subject",
        "depth": 2,
        "text_content": "ho logos",
    },
    # Word: ho (the)
    {
        "node_id": "jhn_1_1_w4",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 4,
        "node_type": "word",
        "role": "subject",
        "parent_id": "jhn_1_1_p2",
        "clause_type": None,
        "relation": "article (nominative masculine singular)",
        "depth": 3,
        "text_content": "ho",
    },
    # Word: logos (Word)
    {
        "node_id": "jhn_1_1_w5",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 5,
        "node_type": "word",
        "role": "subject",
        "parent_id": "jhn_1_1_p2",
        "clause_type": None,
        "relation": "noun (nominative masculine singular)",
        "depth": 3,
        "text_content": "logos",
    },
    # Clause 2: "kai ho logos en pros ton theon" (and the Word was with God)
    {
        "node_id": "jhn_1_1_c2",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "clause",
        "role": None,
        "parent_id": "jhn_1_1_s1",
        "clause_type": "main",
        "relation": "coordinate clause",
        "depth": 1,
        "text_content": "kai ho logos en pros ton theon",
    },
    # Word: kai (and)
    {
        "node_id": "jhn_1_1_w6",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 6,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "jhn_1_1_c2",
        "clause_type": None,
        "relation": "coordinating conjunction",
        "depth": 2,
        "text_content": "kai",
    },
    # Subject phrase: ho logos
    {
        "node_id": "jhn_1_1_p3",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "phrase",
        "role": "subject",
        "parent_id": "jhn_1_1_c2",
        "clause_type": None,
        "relation": "nominative subject",
        "depth": 2,
        "text_content": "ho logos",
    },
    # Word: ho (the)
    {
        "node_id": "jhn_1_1_w7",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 7,
        "node_type": "word",
        "role": "subject",
        "parent_id": "jhn_1_1_p3",
        "clause_type": None,
        "relation": "article (nominative masculine singular)",
        "depth": 3,
        "text_content": "ho",
    },
    # Word: logos (Word)
    {
        "node_id": "jhn_1_1_w8",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 8,
        "node_type": "word",
        "role": "subject",
        "parent_id": "jhn_1_1_p3",
        "clause_type": None,
        "relation": "noun (nominative masculine singular)",
        "depth": 3,
        "text_content": "logos",
    },
    # Word: en (was) — predicate of clause 2
    {
        "node_id": "jhn_1_1_w9",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 9,
        "node_type": "word",
        "role": "predicate",
        "parent_id": "jhn_1_1_c2",
        "clause_type": None,
        "relation": "verb (imperfect active indicative 3s)",
        "depth": 2,
        "text_content": "en",
    },
    # Prepositional phrase: pros ton theon
    {
        "node_id": "jhn_1_1_p4",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "phrase",
        "role": "modifier",
        "parent_id": "jhn_1_1_c2",
        "clause_type": None,
        "relation": "locative prepositional phrase (relational)",
        "depth": 2,
        "text_content": "pros ton theon",
    },
    # Word: pros (with/toward)
    {
        "node_id": "jhn_1_1_w10",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 10,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "jhn_1_1_p4",
        "clause_type": None,
        "relation": "preposition (with accusative: face-to-face relationship)",
        "depth": 3,
        "text_content": "pros",
    },
    # Word: ton (the)
    {
        "node_id": "jhn_1_1_w11",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 11,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "jhn_1_1_p4",
        "clause_type": None,
        "relation": "article (accusative masculine singular)",
        "depth": 3,
        "text_content": "ton",
    },
    # Word: theon (God)
    {
        "node_id": "jhn_1_1_w12",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 12,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "jhn_1_1_p4",
        "clause_type": None,
        "relation": "noun (accusative masculine singular)",
        "depth": 3,
        "text_content": "theon",
    },
    # Clause 3: "kai theos en ho logos" (and the Word was God)
    {
        "node_id": "jhn_1_1_c3",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "clause",
        "role": None,
        "parent_id": "jhn_1_1_s1",
        "clause_type": "main",
        "relation": "coordinate clause",
        "depth": 1,
        "text_content": "kai theos en ho logos",
    },
    # Word: kai (and)
    {
        "node_id": "jhn_1_1_w13",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 13,
        "node_type": "word",
        "role": "modifier",
        "parent_id": "jhn_1_1_c3",
        "clause_type": None,
        "relation": "coordinating conjunction",
        "depth": 2,
        "text_content": "kai",
    },
    # Predicate nominative: theos (God) — anarthrous, predicate position
    {
        "node_id": "jhn_1_1_w14",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 14,
        "node_type": "word",
        "role": "predicate",
        "parent_id": "jhn_1_1_c3",
        "clause_type": None,
        "relation": "predicate nominative (anarthrous — qualitative, emphasizing nature)",
        "depth": 2,
        "text_content": "theos",
    },
    # Word: en (was) — copula of clause 3
    {
        "node_id": "jhn_1_1_w15",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 15,
        "node_type": "word",
        "role": "predicate",
        "parent_id": "jhn_1_1_c3",
        "clause_type": None,
        "relation": "verb (imperfect active indicative 3s, copula)",
        "depth": 2,
        "text_content": "en",
    },
    # Subject phrase: ho logos
    {
        "node_id": "jhn_1_1_p5",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": None,
        "node_type": "phrase",
        "role": "subject",
        "parent_id": "jhn_1_1_c3",
        "clause_type": None,
        "relation": "nominative subject",
        "depth": 2,
        "text_content": "ho logos",
    },
    # Word: ho (the)
    {
        "node_id": "jhn_1_1_w16",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 16,
        "node_type": "word",
        "role": "subject",
        "parent_id": "jhn_1_1_p5",
        "clause_type": None,
        "relation": "article (nominative masculine singular)",
        "depth": 3,
        "text_content": "ho",
    },
    # Word: logos (Word)
    {
        "node_id": "jhn_1_1_w17",
        "book_id": 43,
        "chapter": 1,
        "verse": 1,
        "word_num": 17,
        "node_type": "word",
        "role": "subject",
        "parent_id": "jhn_1_1_p5",
        "clause_type": None,
        "relation": "noun (nominative masculine singular)",
        "depth": 3,
        "text_content": "logos",
    },
]


class SyntaxTreePopulator:
    """Populates the syntax_trees table with clause-level parse tree data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _ensure_table(self, cursor: sqlite3.Cursor) -> bool:
        """Create the syntax_trees table if it does not exist.

        Returns:
            True if table is ready, False otherwise.
        """
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS syntax_trees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL UNIQUE,
                book_id INTEGER NOT NULL,
                chapter INTEGER NOT NULL,
                verse INTEGER NOT NULL,
                word_num INTEGER,
                node_type TEXT NOT NULL,
                role TEXT,
                parent_id TEXT,
                clause_type TEXT,
                relation TEXT,
                depth INTEGER NOT NULL DEFAULT 0,
                text_content TEXT
            )
            """
        )
        return True

    def populate(self, force: bool = False) -> int:
        """Insert syntax tree nodes into the database.

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
                cursor.execute("DELETE FROM syntax_trees")

            for node in SAMPLE_SYNTAX_TREES:
                try:
                    verb = "INSERT OR REPLACE" if force else "INSERT OR IGNORE"
                    cursor.execute(
                        f"{verb} INTO syntax_trees "  # noqa: S608
                        "(node_id, book_id, chapter, verse, word_num, node_type, role, "
                        "parent_id, clause_type, relation, depth, text_content) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            node["node_id"],
                            node["book_id"],
                            node["chapter"],
                            node["verse"],
                            node["word_num"],
                            node["node_type"],
                            node["role"],
                            node["parent_id"],
                            node["clause_type"],
                            node["relation"],
                            node["depth"],
                            node["text_content"],
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert syntax node %s: %s", node["node_id"], e)

            conn.commit()

        logger.info("Populated syntax_trees: %d rows inserted", inserted)
        return inserted
