"""
Clean, efficient semantic concept database schema.
Removes bloat and handles verse ranges properly.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class CleanSemanticSchema:
    """Efficient semantic concept database schema without bloat."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize_schema(self):
        """Create lean, efficient semantic concept tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Core concept definitions (unchanged - this is good)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concepts (
                    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    primary_strongs TEXT,      -- JSON array ["G26", "G25"]
                    extended_strongs TEXT,     -- JSON array ["G5368"]
                    semantic_keywords TEXT,    -- JSON array ["affection", "charity"]
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # CLEAN verse-concept relationships - handles ranges efficiently
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS verse_concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,

                    -- Range support: start = end for single verses
                    start_verse_id TEXT NOT NULL,  -- "1Co:13:1"
                    end_verse_id TEXT NOT NULL,    -- "1Co:13:13" for ranges, same as start for single

                    -- Match information
                    match_type TEXT CHECK(match_type IN ('lexical', 'semantic', 'thematic')),
                    confidence REAL DEFAULT 0.0,
                    semantic_score REAL,

                    -- Evidence (lean)
                    strongs_matched TEXT,          -- JSON array of matched Strong's
                    ollama_validated BOOLEAN DEFAULT 0,
                    ollama_confidence REAL,
                    ollama_reasoning TEXT,

                    -- Optional range description
                    range_description TEXT,        -- "Love Chapter", null for single verses
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(start_verse_id, end_verse_id, concept_id, match_type),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """
            )

            # Concept themes - for study organization
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_themes (
                    theme_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,
                    theme_name TEXT NOT NULL,
                    description TEXT,
                    key_insights TEXT,
                    display_order INTEGER DEFAULT 0,

                    UNIQUE(concept_id, theme_name),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """
            )

            # Theme verses - which verses belong to which themes
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS theme_verses (
                    theme_id INTEGER NOT NULL,
                    start_verse_id TEXT NOT NULL,
                    end_verse_id TEXT NOT NULL,

                    PRIMARY KEY(theme_id, start_verse_id, end_verse_id),
                    FOREIGN KEY (theme_id) REFERENCES concept_themes(theme_id)
                )
            """
            )

            # Concept relationships
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_relationships (
                    concept_id INTEGER NOT NULL,
                    related_concept_id INTEGER NOT NULL,
                    relationship_type TEXT CHECK(relationship_type IN
                        ('related', 'contrasting', 'foundation', 'expression', 'result')),
                    strength REAL DEFAULT 0.5,

                    PRIMARY KEY(concept_id, related_concept_id, relationship_type),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id),
                    FOREIGN KEY (related_concept_id) REFERENCES concepts(concept_id)
                )
            """
            )

            # Biblical stories related to concepts - narrative dimension for education
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_stories (
                    story_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,

                    story_title TEXT NOT NULL,
                    start_verse_id TEXT NOT NULL,
                    end_verse_id TEXT NOT NULL,
                    confidence_score REAL NOT NULL,

                    -- LLM discovery metadata (future-proofed field names)
                    discovery_question TEXT NOT NULL,     -- Question asked to find this story
                    discovery_model TEXT,                 -- Which model found it (ollama, openai, etc.)
                    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Story classification
                    story_type TEXT CHECK(story_type IN
                        ('parable', 'historical', 'prophecy', 'teaching', 'miracle', 'biography')),
                    testament TEXT CHECK(testament IN ('old', 'new')),

                    -- Additional context for study guides
                    story_summary TEXT,                   -- Brief summary
                    teaching_points TEXT,                 -- Key lessons (JSON array)
                    character_focus TEXT,                 -- Main characters (JSON array)

                    UNIQUE(concept_id, story_title, start_verse_id),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """
            )

            # Story discovery sessions - track iterative LLM querying
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS story_discovery_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,
                    discovery_question TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    confidence_threshold REAL DEFAULT 0.70,
                    stories_found INTEGER DEFAULT 0,
                    iterations_performed INTEGER DEFAULT 0,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """
            )

            # Processing stats
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_stats (
                    concept_id INTEGER PRIMARY KEY,
                    total_matches INTEGER DEFAULT 0,
                    lexical_matches INTEGER DEFAULT 0,
                    semantic_matches INTEGER DEFAULT 0,
                    stories_found INTEGER DEFAULT 0,
                    books_covered INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """
            )

            # Essential indexes only
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_concept ON verse_concepts(concept_id)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_start ON verse_concepts(start_verse_id)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_confidence ON verse_concepts(confidence DESC)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_type ON verse_concepts(match_type)",
                "CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)",
                "CREATE INDEX IF NOT EXISTS idx_theme_verses_theme ON theme_verses(theme_id)",
                "CREATE INDEX IF NOT EXISTS idx_stories_concept ON concept_stories(concept_id)",
                "CREATE INDEX IF NOT EXISTS idx_stories_confidence ON concept_stories(confidence_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_stories_type ON concept_stories(story_type)",
                "CREATE INDEX IF NOT EXISTS idx_discovery_sessions_concept ON story_discovery_sessions(concept_id)",
            ]

            for index in indexes:
                cursor.execute(index)

            # Useful views that JOIN with existing verse tables
            cursor.execute(
                """
                CREATE VIEW IF NOT EXISTS concept_verse_details AS
                SELECT
                    c.name as concept_name,
                    vc.start_verse_id,
                    vc.end_verse_id,
                    vc.match_type,
                    vc.confidence,
                    vc.semantic_score,
                    vc.range_description,
                    vc.ollama_validated,
                    vc.ollama_confidence
                FROM verse_concepts vc
                JOIN concepts c ON vc.concept_id = c.concept_id
            """
            )

            conn.commit()

    def add_concept_match(
        self,
        concept_name: str,
        verse_id: str,
        match_type: str,
        confidence: float,
        strongs_matched: Optional[List[str]] = None,
        semantic_score: Optional[float] = None,
        ollama_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a single verse match to a concept."""
        return self.add_concept_range(
            concept_name, verse_id, verse_id, match_type, confidence, strongs_matched, semantic_score, ollama_data
        )

    def add_concept_range(
        self,
        concept_name: str,
        start_verse_id: str,
        end_verse_id: str,
        match_type: str,
        confidence: float,
        strongs_matched: Optional[List[str]] = None,
        semantic_score: Optional[float] = None,
        ollama_data: Optional[Dict[str, Any]] = None,
        range_description: Optional[str] = None,
    ) -> bool:
        """Add a verse range match to a concept."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get or create concept
            cursor.execute("SELECT concept_id FROM concepts WHERE name = ?", (concept_name,))
            row = cursor.fetchone()
            if not row:
                return False

            concept_id = row[0]

            # Insert the match
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO verse_concepts
                    (concept_id, start_verse_id, end_verse_id, match_type, confidence,
                     semantic_score, strongs_matched, ollama_validated,
                     ollama_confidence, ollama_reasoning, range_description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        concept_id,
                        start_verse_id,
                        end_verse_id,
                        match_type,
                        confidence,
                        semantic_score,
                        json.dumps(strongs_matched) if strongs_matched else None,
                        bool(ollama_data) if ollama_data else False,
                        ollama_data.get("confidence") if ollama_data else None,
                        ollama_data.get("reasoning") if ollama_data else None,
                        range_description,
                    ),
                )

                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_verses_for_concept(self, concept_name: str, min_confidence: float = 0.5) -> List[Dict]:
        """Get all verses/ranges for a concept with actual verse text via JOIN."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # This assumes you have a 'verses' table with verse_id and text
            # If your verse table has different structure, adjust accordingly
            cursor.execute(
                """
                SELECT
                    vc.start_verse_id,
                    vc.end_verse_id,
                    vc.match_type,
                    vc.confidence,
                    vc.semantic_score,
                    vc.range_description,
                    vc.ollama_validated,
                    vc.strongs_matched,
                    c.name as concept_name,

                    -- Get verse text via JOIN (adjust table name as needed)
                    v1.text as start_verse_text,
                    CASE
                        WHEN vc.start_verse_id = vc.end_verse_id THEN NULL
                        ELSE v2.text
                    END as end_verse_text

                FROM verse_concepts vc
                JOIN concepts c ON vc.concept_id = c.concept_id
                LEFT JOIN verses v1 ON vc.start_verse_id = v1.verse_id
                LEFT JOIN verses v2 ON vc.end_verse_id = v2.verse_id

                WHERE c.name = ? AND vc.confidence >= ?
                ORDER BY vc.confidence DESC
            """,
                (concept_name, min_confidence),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_context_verses(self, verse_id: str, context_size: int = 2) -> List[Dict]:
        """Get surrounding verses for context (computed on-demand, not stored)."""
        # Parse verse_id to get book, chapter, verse
        parts = verse_id.split(":")
        if len(parts) != 3:
            return []

        book, chapter_str, verse_str = parts
        try:
            chapter = int(chapter_str)
            verse = int(verse_str)
        except ValueError:
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get surrounding verses (adjust table name as needed)
            cursor.execute(
                """
                SELECT verse_id, text
                FROM verses
                WHERE book = ? AND chapter = ?
                AND verse BETWEEN ? AND ?
                ORDER BY verse
            """,
                (book, chapter, max(1, verse - context_size), verse + context_size),
            )

            return [dict(row) for row in cursor.fetchall()]

    def export_concept_to_json(self, concept_name: str, output_path: Path) -> Optional[Dict[str, Any]]:  # noqa: C901
        """Export concept data to JSON without the bloat."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get concept
            cursor.execute("SELECT * FROM concepts WHERE name = ?", (concept_name,))
            concept_row = cursor.fetchone()
            if not concept_row:
                return None

            concept_id = concept_row["concept_id"]

            # Build lean export
            export_data: Dict[str, Any] = {
                "concept": {
                    "name": concept_row["name"],
                    "description": concept_row["description"],
                    "primary_strongs": json.loads(concept_row["primary_strongs"] or "[]"),
                    "extended_strongs": json.loads(concept_row["extended_strongs"] or "[]"),
                    "semantic_keywords": json.loads(concept_row["semantic_keywords"] or "[]"),
                },
                "matches": [],
                "themes": {},
                "stats": {},
            }

            # Get matches (using the view that JOINs with verses)
            cursor.execute(
                """
                SELECT * FROM concept_verse_details
                WHERE concept_name = ?
                ORDER BY confidence DESC
            """,
                (concept_name,),
            )

            for row in cursor.fetchall():
                match_data = {
                    "start_verse": row["start_verse_id"],
                    "end_verse": row["end_verse_id"] if row["end_verse_id"] != row["start_verse_id"] else None,
                    "type": row["match_type"],
                    "confidence": row["confidence"],
                }

                if row["semantic_score"]:
                    match_data["semantic_score"] = row["semantic_score"]
                if row["range_description"]:
                    match_data["description"] = row["range_description"]
                if row["ollama_validated"]:
                    match_data["ollama_confidence"] = row["ollama_confidence"]

                export_data["matches"].append(match_data)

            # Get themes
            cursor.execute(
                """
                SELECT ct.theme_name, ct.description, ct.key_insights,
                       tv.start_verse_id, tv.end_verse_id
                FROM concept_themes ct
                LEFT JOIN theme_verses tv ON ct.theme_id = tv.theme_id
                WHERE ct.concept_id = ?
                ORDER BY ct.display_order
            """,
                (concept_id,),
            )

            current_theme = None
            for row in cursor.fetchall():
                theme_name = row["theme_name"]
                if theme_name != current_theme:
                    export_data["themes"][theme_name] = {
                        "description": row["description"],
                        "key_insights": row["key_insights"],
                        "verses": [],
                    }
                    current_theme = theme_name

                if row["start_verse_id"]:
                    verse_ref = row["start_verse_id"]
                    if row["end_verse_id"] != row["start_verse_id"]:
                        verse_ref += f"-{row['end_verse_id']}"
                    export_data["themes"][theme_name]["verses"].append(verse_ref)

            # Get stories
            cursor.execute(
                """
                SELECT story_title, start_verse_id, end_verse_id, confidence_score,
                       story_type, story_summary, teaching_points, character_focus,
                       discovery_question, discovery_model
                FROM concept_stories
                WHERE concept_id = ?
                ORDER BY confidence_score DESC
            """,
                (concept_id,),
            )

            stories = []
            for row in cursor.fetchall():
                story_data = {
                    "title": row["story_title"],
                    "verses": (
                        f"{row['start_verse_id']}-{row['end_verse_id']}"
                        if row["end_verse_id"] != row["start_verse_id"]
                        else row["start_verse_id"]
                    ),
                    "confidence": row["confidence_score"],
                    "type": row["story_type"],
                    "summary": row["story_summary"],
                }

                if row["teaching_points"]:
                    story_data["teaching_points"] = json.loads(row["teaching_points"])
                if row["character_focus"]:
                    story_data["characters"] = json.loads(row["character_focus"])

                stories.append(story_data)

            export_data["stories"] = stories

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return export_data

    def discover_stories_for_concept(
        self,
        concept_name: str,
        discovery_question: Optional[str] = None,
        model_name: str = "ollama",
        confidence_threshold: float = 0.70,
        max_iterations: int = 5,
    ) -> List[Dict]:
        """
        Use iterative LLM prompting to discover biblical stories for a concept.

        Args:
            concept_name: Name of the concept to find stories for
            discovery_question: Optional custom question. If None, auto-generates from concept
            model_name: LLM model to use (ollama, openai, etc.)
            confidence_threshold: Minimum confidence score to store (default 0.70)
            max_iterations: Maximum "are there more?" rounds

        Example usage:
        discover_stories_for_concept("sexual_immorality", confidence_threshold=0.70)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get concept details
            cursor.execute("SELECT concept_id, name, description FROM concepts WHERE name = ?", (concept_name,))
            row = cursor.fetchone()
            if not row:
                return []

            concept_id, concept_name, concept_description = row

            # Auto-generate discovery question if not provided
            if discovery_question is None:
                discovery_question = self._generate_story_discovery_question(concept_name, concept_description)

            # Start discovery session
            cursor.execute(
                """
                INSERT INTO story_discovery_sessions
                (concept_id, discovery_question, model_used, confidence_threshold)
                VALUES (?, ?, ?, ?)
            """,
                (concept_id, discovery_question, model_name, confidence_threshold),
            )

            session_id = cursor.lastrowid
            stories_found = []
            iterations = 0

            # Initial prompt with exact format specification
            response_format = (
                "Respond with a short title of the story, followed by the Bible verses "
                "in parentheses, comma-separated. Also, after the parentheses, put a space "
                "and then a confidence score between 0.00 and 1.00. The format should look "
                "like this:\n\nThe Story about Something (Genesis 39:7-10) 0.87"
            )
            current_prompt = f"{discovery_question}\n\n{response_format}"

            while iterations < max_iterations:
                iterations += 1

                # Call LLM (placeholder - implement actual calling logic)
                llm_response = self._call_llm_for_stories(current_prompt, model_name)

                # Parse LLM response to extract stories
                new_stories = self._parse_story_response(  # type: ignore[attr-defined]  # pylint: disable=no-member
                    llm_response, concept_id, discovery_question, model_name
                )

                # Filter by confidence threshold
                qualified_stories = [s for s in new_stories if s["confidence_score"] >= confidence_threshold]

                if not qualified_stories:
                    break

                stories_found.extend(qualified_stories)

                # Store stories in database
                for story in qualified_stories:
                    try:
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO concept_stories
                            (concept_id, story_title, start_verse_id, end_verse_id,
                             confidence_score, discovery_question, discovery_model, story_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                concept_id,
                                story["story_title"],
                                story["start_verse_id"],
                                story["end_verse_id"],
                                story["confidence_score"],
                                discovery_question,
                                model_name,
                                story.get("story_type", "historical"),
                            ),
                        )
                    except sqlite3.IntegrityError:
                        pass  # Story already exists

                # Ask for more stories in subsequent iterations
                current_prompt = (
                    f"Are there more Bible stories about {concept_name} "
                    "that you can output that you haven't listed using the same format?"
                )

            # Update session stats
            cursor.execute(
                """
                UPDATE story_discovery_sessions
                SET stories_found = ?, iterations_performed = ?
                WHERE session_id = ?
            """,
                (len(stories_found), iterations, session_id),
            )

            conn.commit()
            return stories_found

    def _call_llm_for_stories(self, _prompt: str, _model_name: str) -> str:
        """
        Placeholder for LLM calls. Replace with actual implementation.
        This could call Ollama, OpenAI, Anthropic, etc.
        """
        # TODO: Implement actual LLM calling logic based on model_name  # pylint: disable=fixme
        # For Ollama: use requests to call http://localhost:11434/api/generate
        # For OpenAI: use openai.chat.completions.create()
        # For other cloud providers: implement respective APIs
        return ""

    def get_stories_for_concept(self, concept_name: str, min_confidence: float = 0.70) -> List[Dict]:
        """Get all stories for a concept above confidence threshold."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT cs.*, c.name as concept_name
                FROM concept_stories cs
                JOIN concepts c ON cs.concept_id = c.concept_id
                WHERE c.name = ? AND cs.confidence_score >= ?
                ORDER BY cs.confidence_score DESC
            """,
                (concept_name, min_confidence),
            )

            return [dict(row) for row in cursor.fetchall()]

    def _generate_story_discovery_question(self, concept_name: str, concept_description: str) -> str:
        """
        Auto-generate a discovery question from concept name and description.

        Args:
            concept_name: Name of the concept (e.g., "sexual_immorality")
            concept_description: Full description of the concept

        Returns:
            Formatted question for LLM story discovery
        """
        # Clean up concept name for display
        display_name = concept_name.replace("_", " ").title()

        # Shared response format instruction
        response_format = (
            "Respond with a short title of the story, followed by the Bible verses "
            "in parentheses, comma-separated. Also, after the parentheses, put a space "
            "and then a confidence score between 0.00 and 1.00. The format should look "
            "like this:\n\nThe Story about Something (Genesis 39:7-10) 0.87"
        )

        # Base question template
        if concept_description and len(concept_description.strip()) > 0:
            intro = (
                f"What Bible stories focus primarily on the concept of {display_name} ({concept_description.strip()})?"
            )
            question = f"{intro}\n\n{response_format}"
        else:
            intro = f"What Bible stories focus primarily on the biblical concept of {display_name}?"
            question = f"{intro}\n\n{response_format}"

        return question
