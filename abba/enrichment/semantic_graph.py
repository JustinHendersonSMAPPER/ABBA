"""Semantic relationship graph for theological concept visualization."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Semantic relationships between biblical theological concepts.
# These relationships form a graph that can be used for concept navigation,
# visualization, and discovering thematic connections across Scripture.
# Fields: source_concept, target_concept,
#         relationship_type (synonym/antithetical/causal/enables/contrast/temporal/hierarchy),
#         weight (0.0-1.0 indicating strength of relationship),
#         evidence_count (approximate number of passages supporting the relationship),
#         shared_strongs_json (JSON list of Strong's numbers that connect the concepts)
SEMANTIC_RELATIONSHIPS: List[Dict[str, Any]] = [
    {
        "source_concept": "grace",
        "target_concept": "mercy",
        "relationship_type": "related",
        "weight": 0.85,
        "evidence_count": 45,
        "shared_strongs_json": json.dumps(["G5485", "H2617", "G1656"]),
    },
    {
        "source_concept": "faith",
        "target_concept": "trust",
        "relationship_type": "synonym",
        "weight": 0.90,
        "evidence_count": 120,
        "shared_strongs_json": json.dumps(["G4102", "H0539", "G4100"]),
    },
    {
        "source_concept": "sin",
        "target_concept": "redemption",
        "relationship_type": "antithetical",
        "weight": 0.80,
        "evidence_count": 85,
        "shared_strongs_json": json.dumps(["G0266", "G0629", "H1350"]),
    },
    {
        "source_concept": "covenant",
        "target_concept": "promise",
        "relationship_type": "causal",
        "weight": 0.75,
        "evidence_count": 65,
        "shared_strongs_json": json.dumps(["H1285", "G1242", "G1860"]),
    },
    {
        "source_concept": "love",
        "target_concept": "sacrifice",
        "relationship_type": "enables",
        "weight": 0.70,
        "evidence_count": 55,
        "shared_strongs_json": json.dumps(["G0026", "G2378", "H0157"]),
    },
    {
        "source_concept": "prophecy",
        "target_concept": "fulfillment",
        "relationship_type": "temporal",
        "weight": 0.85,
        "evidence_count": 95,
        "shared_strongs_json": json.dumps(["G4394", "G4137", "H5030"]),
    },
    {
        "source_concept": "law",
        "target_concept": "grace",
        "relationship_type": "contrast",
        "weight": 0.80,
        "evidence_count": 70,
        "shared_strongs_json": json.dumps(["G3551", "G5485", "H8451"]),
    },
    {
        "source_concept": "righteousness",
        "target_concept": "justice",
        "relationship_type": "synonym",
        "weight": 0.85,
        "evidence_count": 90,
        "shared_strongs_json": json.dumps(["G1343", "H6664", "G1342"]),
    },
    {
        "source_concept": "death",
        "target_concept": "resurrection",
        "relationship_type": "antithetical",
        "weight": 0.90,
        "evidence_count": 75,
        "shared_strongs_json": json.dumps(["G2288", "G0386", "H4194"]),
    },
    {
        "source_concept": "creation",
        "target_concept": "new creation",
        "relationship_type": "temporal",
        "weight": 0.75,
        "evidence_count": 30,
        "shared_strongs_json": json.dumps(["H1254", "G2937", "G2538"]),
    },
    {
        "source_concept": "holiness",
        "target_concept": "righteousness",
        "relationship_type": "related",
        "weight": 0.80,
        "evidence_count": 60,
        "shared_strongs_json": json.dumps(["H6918", "G0040", "G1343"]),
    },
    {
        "source_concept": "sin",
        "target_concept": "forgiveness",
        "relationship_type": "antithetical",
        "weight": 0.85,
        "evidence_count": 80,
        "shared_strongs_json": json.dumps(["G0266", "G0859", "H5545"]),
    },
    {
        "source_concept": "faith",
        "target_concept": "works",
        "relationship_type": "contrast",
        "weight": 0.75,
        "evidence_count": 40,
        "shared_strongs_json": json.dumps(["G4102", "G2041"]),
    },
    {
        "source_concept": "repentance",
        "target_concept": "forgiveness",
        "relationship_type": "causal",
        "weight": 0.80,
        "evidence_count": 50,
        "shared_strongs_json": json.dumps(["G3341", "G0859", "H7725"]),
    },
    {
        "source_concept": "kingdom",
        "target_concept": "authority",
        "relationship_type": "hierarchy",
        "weight": 0.70,
        "evidence_count": 55,
        "shared_strongs_json": json.dumps(["G0932", "G1849", "H4428"]),
    },
    {
        "source_concept": "spirit",
        "target_concept": "flesh",
        "relationship_type": "antithetical",
        "weight": 0.85,
        "evidence_count": 45,
        "shared_strongs_json": json.dumps(["G4151", "G4561", "H7307"]),
    },
    {
        "source_concept": "light",
        "target_concept": "darkness",
        "relationship_type": "antithetical",
        "weight": 0.90,
        "evidence_count": 65,
        "shared_strongs_json": json.dumps(["G5457", "G4655", "H0216"]),
    },
    {
        "source_concept": "wisdom",
        "target_concept": "knowledge",
        "relationship_type": "related",
        "weight": 0.75,
        "evidence_count": 50,
        "shared_strongs_json": json.dumps(["G4678", "G1108", "H2451"]),
    },
    {
        "source_concept": "salvation",
        "target_concept": "grace",
        "relationship_type": "causal",
        "weight": 0.85,
        "evidence_count": 70,
        "shared_strongs_json": json.dumps(["G4991", "G5485", "H3467"]),
    },
    {
        "source_concept": "covenant",
        "target_concept": "faithfulness",
        "relationship_type": "enables",
        "weight": 0.80,
        "evidence_count": 55,
        "shared_strongs_json": json.dumps(["H1285", "H0530", "G1242"]),
    },
    {
        "source_concept": "worship",
        "target_concept": "praise",
        "relationship_type": "synonym",
        "weight": 0.80,
        "evidence_count": 85,
        "shared_strongs_json": json.dumps(["G4352", "G0136", "H7812"]),
    },
    {
        "source_concept": "prayer",
        "target_concept": "worship",
        "relationship_type": "hierarchy",
        "weight": 0.70,
        "evidence_count": 60,
        "shared_strongs_json": json.dumps(["G4335", "G4352", "H8605"]),
    },
    {
        "source_concept": "truth",
        "target_concept": "deception",
        "relationship_type": "antithetical",
        "weight": 0.85,
        "evidence_count": 40,
        "shared_strongs_json": json.dumps(["G0225", "G0539", "H0571"]),
    },
    {
        "source_concept": "joy",
        "target_concept": "suffering",
        "relationship_type": "contrast",
        "weight": 0.70,
        "evidence_count": 35,
        "shared_strongs_json": json.dumps(["G5479", "G3804", "H8057"]),
    },
    {
        "source_concept": "baptism",
        "target_concept": "death",
        "relationship_type": "enables",
        "weight": 0.65,
        "evidence_count": 20,
        "shared_strongs_json": json.dumps(["G0908", "G2288"]),
    },
    {
        "source_concept": "atonement",
        "target_concept": "sacrifice",
        "relationship_type": "causal",
        "weight": 0.85,
        "evidence_count": 50,
        "shared_strongs_json": json.dumps(["H3722", "G2378", "G2435"]),
    },
    {
        "source_concept": "glory",
        "target_concept": "holiness",
        "relationship_type": "related",
        "weight": 0.75,
        "evidence_count": 45,
        "shared_strongs_json": json.dumps(["H3519", "G1391", "H6918"]),
    },
    {
        "source_concept": "hope",
        "target_concept": "faith",
        "relationship_type": "related",
        "weight": 0.80,
        "evidence_count": 55,
        "shared_strongs_json": json.dumps(["G1680", "G4102"]),
    },
    {
        "source_concept": "election",
        "target_concept": "covenant",
        "relationship_type": "causal",
        "weight": 0.70,
        "evidence_count": 35,
        "shared_strongs_json": json.dumps(["G1589", "H1285", "H0977"]),
    },
    {
        "source_concept": "judgment",
        "target_concept": "mercy",
        "relationship_type": "contrast",
        "weight": 0.80,
        "evidence_count": 40,
        "shared_strongs_json": json.dumps(["G2920", "G1656", "H4941"]),
    },
]


class SemanticGraphPopulator:
    """Populates the semantic_relationship_graph table for concept graph visualization."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _ensure_table(self, cursor: sqlite3.Cursor) -> bool:
        """Create the semantic_relationship_graph table if it does not exist.

        Returns:
            True if table is ready, False otherwise.
        """
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_relationship_graph (
                relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_concept TEXT NOT NULL,
                target_concept TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 0,
                shared_strongs_json TEXT,
                UNIQUE(source_concept, target_concept, relationship_type)
            )
            """
        )
        return True

    def populate(self, force: bool = False) -> int:
        """Insert semantic relationships into the database.

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
                cursor.execute("DELETE FROM semantic_relationship_graph")

            for rel in SEMANTIC_RELATIONSHIPS:
                try:
                    verb = "INSERT OR REPLACE" if force else "INSERT OR IGNORE"
                    cursor.execute(
                        f"{verb} INTO semantic_relationship_graph "  # noqa: S608
                        "(source_concept, target_concept, relationship_type, weight, "
                        "evidence_count, shared_strongs_json) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            rel["source_concept"],
                            rel["target_concept"],
                            rel["relationship_type"],
                            rel["weight"],
                            rel["evidence_count"],
                            rel["shared_strongs_json"],
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error(
                        "Failed to insert relationship %s -> %s: %s",
                        rel["source_concept"],
                        rel["target_concept"],
                        e,
                    )

            conn.commit()

        logger.info("Populated semantic_relationship_graph: %d rows inserted", inserted)
        return inserted
