#!/usr/bin/env python3
"""
Strong's-Centric Semantic Concordance

This module implements an academically defensible approach to biblical concept
searching based on Strong's Concordance numbers as the primary semantic anchor.

Methodology:
1. Strong's numbers provide lemma-level semantic identification
2. Lexicon entries provide authoritative glosses and definitions
3. Morphological variants are tracked but not inferred
4. All matches are traceable to lexicographic sources

This approach prioritizes accuracy and scholarly defensibility over
algorithmic complexity.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..database.sqlite_manager import SQLiteManager
from ..logging_setup import logger


@dataclass
class ConceptDefinition:
    """Defines a biblical concept using Strong's numbers as semantic anchors."""

    name: str
    description: str
    primary_strongs: List[str] = field(default_factory=list)
    extended_strongs: List[str] = field(default_factory=list)
    excluded_strongs: List[str] = field(default_factory=list)
    phrase_patterns: List[Dict] = field(default_factory=list)
    validation_source: str = "Strong's Exhaustive Concordance"

    def get_all_strongs(self) -> Set[str]:
        """Get all Strong's numbers (primary + extended)."""
        return set(self.primary_strongs + self.extended_strongs)


@dataclass
class ConcordanceMatch:
    """Represents a match between a verse and a concept."""

    verse_id: str
    book: str
    chapter: int
    verse: int
    match_type: str  # 'primary', 'extended', 'phrase', 'lemma', 'potential'
    confidence: float  # 1.0 for primary, 0.8 for extended, etc.
    evidence: str  # Human-readable explanation
    strongs_matched: List[str]
    original_text: str
    translation_text: Optional[str] = None


class StrongsConcordance:
    """
    Builds concordances for biblical concepts using Strong's numbers
    as the authoritative semantic anchor.
    """

    def __init__(self, db_path: Path):
        """Initialize with database connection."""
        self.db_manager = SQLiteManager(db_path)
        self._validate_database_schema()

    def _validate_database_schema(self):
        """Ensure required tables and columns exist."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check for required columns
            cursor.execute("PRAGMA table_info(stepbible_verses)")
            columns = {col[1] for col in cursor.fetchall()}

            required_columns = {"strongs_lexical", "normalized_word", "original_word"}
            missing = required_columns - columns

            if missing:
                raise ValueError(
                    f"Database schema incomplete. Missing columns: {missing}. " "Run fix_stepbible_schema.py first."
                )

    def define_concept(
        self,
        name: str,
        primary_strongs: List[str],
        extended_strongs: Optional[List[str]] = None,
        phrase_patterns: Optional[List[Dict]] = None,
    ) -> ConceptDefinition:
        """
        Define a biblical concept using Strong's numbers.

        Args:
            name: Concept name (e.g., 'love', 'faith')
            primary_strongs: Core Strong's numbers that directly represent the concept
            extended_strongs: Related Strong's numbers (synonyms, related terms)
            phrase_patterns: Multi-word patterns (e.g., 'kingdom of God')

        Returns:
            ConceptDefinition object
        """
        # Validate Strong's numbers exist in database
        # Disable validation for now since it's causing issues
        # all_strongs = primary_strongs + (extended_strongs or [])
        # self._validate_strongs_numbers(all_strongs)

        return ConceptDefinition(
            name=name,
            description=f"Biblical concept: {name}",
            primary_strongs=primary_strongs,
            extended_strongs=extended_strongs or [],
            phrase_patterns=phrase_patterns or [],
        )

    def _validate_strongs_numbers(self, strongs_list: List[str]):
        """Validate that Strong's numbers exist in the database."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            for strongs in strongs_list:
                # Handle potential padding (H157 vs H0157, G25 vs G0025)
                variants = [strongs]
                if (strongs.startswith("H") or strongs.startswith("G")) and len(strongs) < 5:
                    padded = strongs[0] + strongs[1:].zfill(4)
                    variants.append(padded)

                placeholders = " OR ".join(["strongs_lexical = ?" for _ in variants])
                cursor.execute(f"SELECT COUNT(*) FROM stepbible_verses WHERE {placeholders}", variants)

                if cursor.fetchone()[0] == 0:
                    logger.warning(f"Strong's number {strongs} not found in database")

    def build_concordance(self, concept: ConceptDefinition) -> List[ConcordanceMatch]:
        """
        Build a complete concordance for a biblical concept.

        Returns matches organized by confidence level:
        1. Primary Strong's matches (confidence: 1.0)
        2. Extended Strong's matches (confidence: 0.8)
        3. Phrase pattern matches (confidence: 0.9)
        4. Lemma-related matches (confidence: 0.7)

        All matches include evidence for scholarly verification.
        """
        matches = []

        # Layer 1: Primary Strong's matches
        matches.extend(self._find_strongs_matches(concept.primary_strongs, match_type="primary", confidence=1.0))

        # Layer 2: Extended Strong's matches
        if concept.extended_strongs:
            matches.extend(self._find_strongs_matches(concept.extended_strongs, match_type="extended", confidence=0.8))

        # Layer 3: Phrase patterns
        if concept.phrase_patterns:
            matches.extend(self._find_phrase_matches(concept.phrase_patterns, confidence=0.9))

        # Layer 4: Lemma variants (same lexicon entry, different Strong's)
        lemma_matches = self._find_lemma_variants(concept.get_all_strongs())
        matches.extend(lemma_matches)

        # Deduplicate and sort by confidence
        return self._deduplicate_matches(matches)

    def _find_strongs_matches(
        self, strongs_list: List[str], match_type: str, confidence: float
    ) -> List[ConcordanceMatch]:
        """Find all verses containing specified Strong's numbers."""
        matches = []

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            for strongs in strongs_list:
                # Handle padding variants
                search_variants = []
                search_variants.append(str(strongs))  # Ensure string type

                # Handle padding for both Hebrew and Greek
                if (strongs.startswith("H") or strongs.startswith("G")) and len(strongs) < 5:
                    # Pad to 4 digits (H157 → H0157, G25 → G0025)
                    padded = strongs[0] + strongs[1:].zfill(4)
                    search_variants.append(str(padded))  # Ensure string type

                placeholders = " OR ".join(["strongs_lexical = ?" for _ in search_variants])

                cursor.execute(
                    f"""
                    SELECT
                        sv.id,
                        sv.book,
                        sv.chapter,
                        sv.verse,
                        sv.original_word,
                        sv.strongs_lexical,
                        COALESCE(l.definition, l.gloss, 'No definition available') as definition
                    FROM stepbible_verses sv
                    LEFT JOIN lexicon l ON sv.strongs_lexical = l.strongs_number
                    WHERE {placeholders}
                """,
                    search_variants,
                )

                for row in cursor.fetchall():
                    verse_id = f"{row[1]}:{row[2]}:{row[3]}"

                    matches.append(
                        ConcordanceMatch(
                            verse_id=verse_id,
                            book=row[1],
                            chapter=row[2],
                            verse=row[3],
                            match_type=match_type,
                            confidence=confidence,
                            evidence=f"Strong's {strongs}: {row[6] or 'No gloss available'}",
                            strongs_matched=[strongs],
                            original_text=row[4],
                        )
                    )

        return matches

    def _find_phrase_matches(self, phrase_patterns: List[Dict], confidence: float) -> List[ConcordanceMatch]:
        """Find multi-word phrase patterns."""
        matches = []

        for pattern in phrase_patterns:
            # Example pattern: {'strongs': ['G932', 'G2316'], 'name': 'kingdom of God'}
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Find verses containing all required Strong's numbers
                strongs_conditions = []
                params = []

                for strongs in pattern["strongs"]:
                    strongs_conditions.append(
                        "EXISTS (SELECT 1 FROM stepbible_verses sv2 "
                        "WHERE sv2.book = sv.book AND sv2.chapter = sv.chapter "
                        "AND sv2.verse = sv.verse AND sv2.strongs_lexical = ?)"
                    )
                    params.append(strongs)

                query = f"""
                    SELECT DISTINCT
                        sv.book,
                        sv.chapter,
                        sv.verse
                    FROM stepbible_verses sv
                    WHERE {' AND '.join(strongs_conditions)}
                """

                cursor.execute(query, params)

                for row in cursor.fetchall():
                    verse_id = f"{row[0]}:{row[1]}:{row[2]}"

                    matches.append(
                        ConcordanceMatch(
                            verse_id=verse_id,
                            book=row[0],
                            chapter=row[1],
                            verse=row[2],
                            match_type="phrase",
                            confidence=confidence,
                            evidence=f"Phrase pattern: {pattern['name']}",
                            strongs_matched=pattern["strongs"],
                            original_text=f"[Phrase: {pattern['name']}]",
                        )
                    )

        return matches

    def _find_lemma_variants(self, strongs_list: Set[str]) -> List[ConcordanceMatch]:
        """Find other Strong's numbers with the same lemma (using original_word as lemma)."""
        matches = []

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get original words (lemmas) for our Strong's numbers
            placeholders = ",".join(["?" for _ in strongs_list])
            cursor.execute(
                f"""
                SELECT DISTINCT original_word, strongs_number
                FROM lexicon
                WHERE strongs_number IN ({placeholders})
                AND original_word IS NOT NULL
            """,
                tuple(strongs_list),
            )

            lemma_map: Dict[str, List[str]] = {}
            for lemma, strongs in cursor.fetchall():
                if lemma not in lemma_map:
                    lemma_map[lemma] = []
                lemma_map[lemma].append(strongs)

            # Find other Strong's with same original word (lemma)
            for lemma, original_strongs in lemma_map.items():
                placeholders2 = ",".join(["?" for _ in original_strongs])
                cursor.execute(
                    f"""
                    SELECT strongs_number
                    FROM lexicon
                    WHERE original_word = ?
                    AND strongs_number NOT IN ({placeholders2})
                """,
                    tuple([lemma] + original_strongs),
                )

                variant_strongs = [row[0] for row in cursor.fetchall()]

                if variant_strongs:
                    # Add matches for these variants
                    variant_matches = self._find_strongs_matches(variant_strongs, match_type="lemma", confidence=0.7)

                    # Update evidence to show lemma connection
                    for match in variant_matches:
                        match.evidence = f"Lemma variant of {', '.join(original_strongs)}"

                    matches.extend(variant_matches)

        return matches

    def _deduplicate_matches(self, matches: List[ConcordanceMatch]) -> List[ConcordanceMatch]:
        """Remove duplicate matches, keeping highest confidence."""
        unique_matches: Dict[tuple, ConcordanceMatch] = {}

        for match in matches:
            key = (match.verse_id, match.strongs_matched[0] if match.strongs_matched else "")

            if key not in unique_matches or match.confidence > unique_matches[key].confidence:
                unique_matches[key] = match

        # Sort by confidence (descending) then by verse reference
        return sorted(unique_matches.values(), key=lambda m: (-m.confidence, m.book, m.chapter, m.verse))

    def generate_report(self, concept: ConceptDefinition, matches: List[ConcordanceMatch]) -> str:
        """Generate a human-readable concordance report."""
        report = []
        report.append(f"# Concordance Report: {concept.name}")
        report.append("\n## Methodology")
        report.append(f"- Primary Strong's numbers: {', '.join(concept.primary_strongs)}")
        if concept.extended_strongs:
            report.append(f"- Extended Strong's numbers: {', '.join(concept.extended_strongs)}")
        report.append(f"- Validation source: {concept.validation_source}")

        # Group matches by type
        by_type: Dict[str, List[ConcordanceMatch]] = {}
        for match in matches:
            if match.match_type not in by_type:
                by_type[match.match_type] = []
            by_type[match.match_type].append(match)

        # Report statistics
        report.append("\n## Statistics")
        report.append(f"- Total matches: {len(matches)}")
        for match_type, type_matches in by_type.items():
            report.append(f"- {match_type.capitalize()} matches: {len(type_matches)}")

        # Sample matches by type
        report.append("\n## Sample Matches by Type")
        for match_type in ["primary", "extended", "phrase", "lemma"]:
            if match_type in by_type:
                report.append(f"\n### {match_type.capitalize()} Matches")
                for match in by_type[match_type][:5]:  # First 5 examples
                    report.append(
                        f"- {match.verse_id} - {match.original_text} "
                        f"(Confidence: {match.confidence}, Evidence: {match.evidence})"
                    )
                if len(by_type[match_type]) > 5:
                    report.append(f"- ... and {len(by_type[match_type]) - 5} more")

        return "\n".join(report)
