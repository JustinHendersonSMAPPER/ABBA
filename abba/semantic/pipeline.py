"""Semantic analysis pipeline for biblical concept mapping and validation."""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ollama_analyzer import OllamaAnalyzer, SemanticAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ConceptDefinition:
    """Definition of a biblical concept for analysis."""

    name: str
    description: str
    hebrew_terms: List[str] = field(default_factory=list)
    greek_terms: List[str] = field(default_factory=list)
    strongs_numbers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class ConceptMappingResult:
    """Result of mapping a concept to verses."""

    concept: ConceptDefinition
    traditional_matches: List[str] = field(default_factory=list)  # Verse IDs from Strong's/keywords
    llm_validated_matches: List[str] = field(default_factory=list)  # LLM confirmed as relevant
    llm_discovered_matches: List[str] = field(default_factory=list)  # LLM found additional verses
    false_positives: List[str] = field(default_factory=list)  # Traditional matches rejected by LLM
    analysis_results: Dict[str, SemanticAnalysisResult] = field(default_factory=dict)
    processing_time: float = 0.0
    total_verses_analyzed: int = 0


class SemanticAnalysisPipeline:
    """Pipeline for semantic analysis and concept mapping using LLM validation."""

    def __init__(
        self,
        db_path: Path,
        ollama_host: str = "http://localhost:11434",
        ollama_models: List[str] = None,
        consensus_threshold: float = 0.7,
        batch_size: int = 100,
    ):
        """Initialize semantic analysis pipeline.

        Args:
            db_path: Path to ABBA SQLite database
            ollama_host: Ollama API endpoint
            ollama_models: List of models for consensus
            consensus_threshold: Agreement threshold for validation
            batch_size: Batch size for LLM processing
        """
        self.db_path = db_path
        self.analyzer = OllamaAnalyzer(
            host=ollama_host,
            models=ollama_models or ["llama4:scout", "command-r-plus:latest"],
            consensus_threshold=consensus_threshold,
            batch_size=batch_size,
        )

        # Cache for verse data to avoid repeated database queries
        self._verse_cache = {}

    def map_concept_to_verses(self, concept: ConceptDefinition) -> ConceptMappingResult:
        """Map a concept to biblical verses using traditional + LLM approach.

        This implements the three-phase approach:
        1. Traditional mapping (Strong's numbers, keywords)
        2. LLM validation of traditional matches
        3. Comprehensive LLM scanning for additional matches

        Args:
            concept: Concept definition to map

        Returns:
            ConceptMappingResult with all mappings and analysis
        """
        start_time = time.time()
        logger.info(f"Starting concept mapping for: {concept.name}")

        result = ConceptMappingResult(concept=concept)

        try:
            # Phase 1: Traditional mapping
            logger.info("Phase 1: Traditional mapping using Strong's numbers and keywords")
            result.traditional_matches = self._get_traditional_matches(concept)
            logger.info(f"Found {len(result.traditional_matches)} traditional matches")

            # Phase 2: LLM validation of traditional matches
            logger.info("Phase 2: LLM validation of traditional matches")
            if result.traditional_matches:
                validated, false_positives = self._validate_traditional_matches(concept, result.traditional_matches)
                result.llm_validated_matches = validated
                result.false_positives = false_positives
                result.total_verses_analyzed += len(result.traditional_matches)

                logger.info(f"LLM validated {len(validated)} matches, rejected {len(false_positives)}")

            # Phase 3: Comprehensive LLM scanning
            logger.info("Phase 3: Comprehensive LLM scanning for additional matches")
            additional_matches = self._comprehensive_verse_scan(concept, exclude_verses=set(result.traditional_matches))
            result.llm_discovered_matches = additional_matches
            logger.info(f"LLM discovered {len(additional_matches)} additional matches")

        except Exception as e:
            logger.error(f"Error in concept mapping for {concept.name}: {e}")
            raise

        finally:
            result.processing_time = time.time() - start_time
            logger.info(f"Concept mapping completed in {result.processing_time:.1f}s")

            # Log summary
            total_matches = len(result.llm_validated_matches) + len(result.llm_discovered_matches)
            logger.info(f"Final mapping for {concept.name}: {total_matches} relevant verses")

        return result

    def _get_traditional_matches(self, concept: ConceptDefinition) -> List[str]:
        """Get verses using traditional Strong's numbers and keyword matching.

        Args:
            concept: Concept definition

        Returns:
            List of verse IDs (book_id:chapter:verse format)
        """
        verse_ids = set()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Search by Strong's numbers
            for strongs in concept.strongs_numbers:
                cursor.execute(
                    """
                    SELECT DISTINCT book, chapter, verse
                    FROM stepbible_verses
                    WHERE strongs_primary = ? OR strongs_secondary LIKE ?
                """,
                    (strongs, f"%{strongs}%"),
                )

                for row in cursor.fetchall():
                    book_id = self._get_book_id(row[0])
                    if book_id:
                        verse_id = f"{book_id:03d}:{row[1]:03d}:{row[2]:03d}"
                        verse_ids.add(verse_id)

            # Search by Hebrew terms
            for hebrew_term in concept.hebrew_terms:
                cursor.execute(
                    """
                    SELECT DISTINCT book, chapter, verse
                    FROM stepbible_verses
                    WHERE original_word LIKE ?
                """,
                    (f"%{hebrew_term}%",),
                )

                for row in cursor.fetchall():
                    book_id = self._get_book_id(row[0])
                    if book_id:
                        verse_id = f"{book_id:03d}:{row[1]:03d}:{row[2]:03d}"
                        verse_ids.add(verse_id)

            # Search by Greek terms
            for greek_term in concept.greek_terms:
                cursor.execute(
                    """
                    SELECT DISTINCT book, chapter, verse
                    FROM stepbible_verses
                    WHERE original_word LIKE ?
                """,
                    (f"%{greek_term}%",),
                )

                for row in cursor.fetchall():
                    book_id = self._get_book_id(row[0])
                    if book_id:
                        verse_id = f"{book_id:03d}:{row[1]:03d}:{row[2]:03d}"
                        verse_ids.add(verse_id)

        return sorted(list(verse_ids))

    def _validate_traditional_matches(
        self, concept: ConceptDefinition, verse_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Validate traditional matches using LLM analysis.

        Args:
            concept: Concept definition
            verse_ids: List of verse IDs to validate

        Returns:
            Tuple of (validated_matches, false_positives)
        """
        validated = []
        false_positives = []

        # Get verse texts
        verses = []
        for verse_id in verse_ids:
            verse_data = self._get_verse_data(verse_id)
            if verse_data:
                verses.append((verse_data["text"], verse_id))

        # Batch analyze verses
        results = self.analyzer.batch_analyze_verses(verses, concept.name, concept.description)

        # Categorize results
        for result in results:
            verse_id = result.verse_reference

            # Consider relevant if high relevance score and consensus reached
            if result.relevance_score >= 0.5 and result.consensus_reached and not result.error:
                validated.append(verse_id)
            else:
                false_positives.append(verse_id)

        return validated, false_positives

    def _comprehensive_verse_scan(self, concept: ConceptDefinition, exclude_verses: set = None) -> List[str]:
        """Scan all verses for additional concept matches.

        This is the most time-intensive phase, scanning every verse
        in the database for potential relevance to the concept.

        Args:
            concept: Concept definition
            exclude_verses: Set of verse IDs to skip (already processed)

        Returns:
            List of additional verse IDs that match the concept
        """
        if exclude_verses is None:
            exclude_verses = set()

        additional_matches = []

        # Get all verses not already processed
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all unique verses
            cursor.execute("""
                SELECT DISTINCT book, chapter, verse
                FROM stepbible_verses
                WHERE original_word IS NOT NULL AND original_word != ''
                ORDER BY book, chapter, verse
            """)

            all_verses = []
            for row in cursor.fetchall():
                book_id = self._get_book_id(row[0])
                if book_id:
                    verse_id = f"{book_id:03d}:{row[1]:03d}:{row[2]:03d}"

                    # Skip if already processed
                    if verse_id not in exclude_verses:
                        verse_data = self._get_verse_data(verse_id)
                        if verse_data:
                            all_verses.append((verse_data["text"], verse_id))

        logger.info(f"Scanning {len(all_verses)} verses for concept {concept.name}")

        if all_verses:
            # Batch analyze all verses
            results = self.analyzer.batch_analyze_verses(all_verses, concept.name, concept.description)

            # Keep verses with high relevance and consensus
            for result in results:
                if (
                    result.relevance_score >= 0.6  # Higher threshold for discovery
                    and result.consensus_reached
                    and not result.error
                ):
                    additional_matches.append(result.verse_reference)

        return additional_matches

    def _get_verse_data(self, verse_id: str) -> Optional[Dict[str, Any]]:
        """Get verse data from database with caching.

        Args:
            verse_id: Verse ID in format "book_id:chapter:verse"

        Returns:
            Dict with verse data or None if not found
        """
        if verse_id in self._verse_cache:
            return self._verse_cache[verse_id]

        try:
            book_id, chapter, verse = map(int, verse_id.split(":"))

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get verse text from a reference translation (KJV)
                cursor.execute(
                    """
                    SELECT v.text, t.name as translation_name
                    FROM verses v
                    JOIN translations t ON v.translation_id = t.id
                    WHERE v.book_id = ? AND v.chapter = ? AND v.verse = ?
                    AND t.abbreviation = 'KJV'
                    LIMIT 1
                """,
                    (book_id, chapter, verse),
                )

                row = cursor.fetchone()
                if row:
                    verse_data = {
                        "text": row[0],
                        "translation": row[1],
                        "verse_id": verse_id,
                        "book_id": book_id,
                        "chapter": chapter,
                        "verse": verse,
                    }

                    # Cache the result
                    self._verse_cache[verse_id] = verse_data
                    return verse_data

        except Exception as e:
            logger.warning(f"Error getting verse data for {verse_id}: {e}")

        return None

    def _get_book_id(self, book_name: str) -> Optional[int]:
        """Convert book name to book ID.

        Args:
            book_name: Book abbreviation (e.g., 'Gen', 'Mat')

        Returns:
            Book ID (1-66) or None if not found
        """
        book_mapping = {
            "Gen": 1,
            "Exo": 2,
            "Lev": 3,
            "Num": 4,
            "Deu": 5,
            "Jos": 6,
            "Jdg": 7,
            "Rut": 8,
            "1Sa": 9,
            "2Sa": 10,
            "1Ki": 11,
            "2Ki": 12,
            "1Ch": 13,
            "2Ch": 14,
            "Ezr": 15,
            "Neh": 16,
            "Est": 17,
            "Job": 18,
            "Psa": 19,
            "Pro": 20,
            "Ecc": 21,
            "Sng": 22,
            "Isa": 23,
            "Jer": 24,
            "Lam": 25,
            "Ezk": 26,
            "Dan": 27,
            "Hos": 28,
            "Jol": 29,
            "Amo": 30,
            "Oba": 31,
            "Jon": 32,
            "Mic": 33,
            "Nam": 34,
            "Hab": 35,
            "Zep": 36,
            "Hag": 37,
            "Zec": 38,
            "Mal": 39,
            "Mat": 40,
            "Mrk": 41,
            "Luk": 42,
            "Jhn": 43,
            "Act": 44,
            "Rom": 45,
            "1Co": 46,
            "2Co": 47,
            "Gal": 48,
            "Eph": 49,
            "Php": 50,
            "Col": 51,
            "1Th": 52,
            "2Th": 53,
            "1Ti": 54,
            "2Ti": 55,
            "Tit": 56,
            "Phm": 57,
            "Heb": 58,
            "Jas": 59,
            "1Pe": 60,
            "2Pe": 61,
            "1Jn": 62,
            "2Jn": 63,
            "3Jn": 64,
            "Jud": 65,
            "Rev": 66,
        }

        return book_mapping.get(book_name)

    def save_concept_mapping(self, result: ConceptMappingResult) -> bool:
        """Save concept mapping results to database.

        Args:
            result: ConceptMappingResult to save

        Returns:
            True if saved successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create tables if they don't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS concept_definitions (
                        concept_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        hebrew_terms TEXT,
                        greek_terms TEXT,
                        strongs_numbers TEXT,
                        keywords TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS concept_verse_mappings (
                        concept_id TEXT,
                        verse_id TEXT,
                        validation_method TEXT,
                        relevance_score REAL,
                        confidence_score REAL,
                        validation_reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (concept_id, verse_id),
                        FOREIGN KEY (concept_id) REFERENCES concept_definitions(concept_id)
                    )
                """)

                # Insert/update concept definition
                concept_id = result.concept.name.lower().replace(" ", "_")
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO concept_definitions
                    (concept_id, name, description, hebrew_terms, greek_terms, strongs_numbers, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        concept_id,
                        result.concept.name,
                        result.concept.description,
                        ",".join(result.concept.hebrew_terms),
                        ",".join(result.concept.greek_terms),
                        ",".join(result.concept.strongs_numbers),
                        ",".join(result.concept.keywords),
                    ),
                )

                # Clear existing mappings for this concept
                cursor.execute(
                    """
                    DELETE FROM concept_verse_mappings WHERE concept_id = ?
                """,
                    (concept_id,),
                )

                # Insert validated traditional matches
                for verse_id in result.llm_validated_matches:
                    analysis = result.analysis_results.get(verse_id)
                    cursor.execute(
                        """
                        INSERT INTO concept_verse_mappings
                        (concept_id, verse_id, validation_method, relevance_score, confidence_score, validation_reason)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            concept_id,
                            verse_id,
                            "llm_validated",
                            analysis.relevance_score if analysis else 0.0,
                            analysis.confidence if analysis else 0.0,
                            analysis.reasoning if analysis else "Traditional match validated by LLM",
                        ),
                    )

                # Insert LLM discovered matches
                for verse_id in result.llm_discovered_matches:
                    analysis = result.analysis_results.get(verse_id)
                    cursor.execute(
                        """
                        INSERT INTO concept_verse_mappings
                        (concept_id, verse_id, validation_method, relevance_score, confidence_score, validation_reason)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            concept_id,
                            verse_id,
                            "llm_discovered",
                            analysis.relevance_score if analysis else 0.0,
                            analysis.confidence if analysis else 0.0,
                            analysis.reasoning if analysis else "Discovered by LLM comprehensive scan",
                        ),
                    )

                conn.commit()
                logger.info(f"Saved concept mapping for {result.concept.name} to database")
                return True

        except Exception as e:
            logger.error(f"Error saving concept mapping: {e}")
            return False
