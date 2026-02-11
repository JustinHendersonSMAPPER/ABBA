#!/usr/bin/env python3
"""
Semantic Concordance with Ollama Validation

This module extends the Strong's-centric concordance with semantic search
capabilities using embeddings and LLM validation to reduce false positives.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..database.sqlite_manager import SQLiteManager
from ..embeddings.chroma_manager import ChromaManager
from ..logging_setup import logger
from .ollama_analyzer import OllamaAnalyzer
from .strongs_concordance import ConceptDefinition, ConcordanceMatch, StrongsConcordance


@dataclass
class SemanticMatch(ConcordanceMatch):
    """Extended match with semantic scoring."""

    semantic_score: float = 0.0
    ollama_validation: Optional[str] = None
    ollama_confidence: float = 0.0
    is_semantic_only: bool = False


@dataclass
class ValidationResult:
    """Result from Ollama validation."""

    answer: str  # YES, MAYBE, NO
    confidence: float
    explanation: str
    model: str


class SemanticConcordance:
    """
    Combines Strong's concordance with embedding-based semantic search
    and Ollama validation for high-precision biblical concept mapping.
    """

    def __init__(self, db_path: Path, chroma_path: Path, ollama_config: Dict):
        """Initialize with database, embeddings, and Ollama."""
        self.strongs_concordance = StrongsConcordance(db_path)
        self.db_manager = SQLiteManager(db_path)
        self.chroma_manager = ChromaManager(chroma_path)
        self.ollama = OllamaAnalyzer(
            host=ollama_config.get("host", "http://localhost:11434"),
            models=ollama_config.get("models", ["llama3"]),
            consensus_threshold=ollama_config.get("consensus_threshold", 0.7),
            timeout=ollama_config.get("timeout", 30),
        )
        self.ollama_models = ollama_config.get("models", ["llama3"])
        self.consensus_threshold = ollama_config.get("consensus_threshold", 0.7)

    def build_semantic_concordance(
        self, concept: ConceptDefinition, max_semantic_results: int = 100, validate_semantic: bool = True
    ) -> List[SemanticMatch]:
        """
        Build a concordance combining lexical and semantic approaches.

        Process:
        1. Get Strong's-based matches (high precision)
        2. Build concept prototype from high-confidence matches
        3. Find semantically similar verses
        4. Validate semantic matches with Ollama
        5. Combine and rank all results
        """
        logger.info(f"Building semantic concordance for concept: {concept.name}")

        # Step 1: Get lexical matches
        lexical_matches = self.strongs_concordance.build_concordance(concept)
        logger.info(f"Found {len(lexical_matches)} lexical matches")

        # Convert to SemanticMatch objects
        all_matches = []
        lexical_verse_ids = set()

        for match in lexical_matches:
            semantic_match = SemanticMatch(
                verse_id=match.verse_id,
                book=match.book,
                chapter=match.chapter,
                verse=match.verse,
                match_type=match.match_type,
                confidence=match.confidence,
                evidence=match.evidence,
                strongs_matched=match.strongs_matched,
                original_text=match.original_text,
                translation_text=match.translation_text,
                semantic_score=0.0,  # Lexical matches don't need semantic score
                is_semantic_only=False,
            )
            all_matches.append(semantic_match)
            lexical_verse_ids.add(match.verse_id)

        # Step 2: Build concept prototypes
        hebrew_prototype, greek_prototype = self._build_concept_prototypes(
            lexical_matches[:20]  # Use top matches for prototype
        )

        # Step 3: Find semantically similar verses
        if hebrew_prototype is not None or greek_prototype is not None:
            semantic_candidates = self._find_semantic_matches(
                hebrew_prototype, greek_prototype, max_results=max_semantic_results, exclude_verse_ids=lexical_verse_ids
            )
            logger.info(f"Found {len(semantic_candidates)} semantic candidates")

            # Step 4: Validate semantic matches
            if validate_semantic and semantic_candidates:
                validated_matches = self._validate_semantic_matches(concept, semantic_candidates)
                all_matches.extend(validated_matches)
                logger.info(f"Validated {len(validated_matches)} semantic matches")

        # Step 5: Deduplicate by verse and rank
        deduplicated_matches = self._deduplicate_matches(all_matches)
        return self._rank_combined_results(deduplicated_matches)

    def _deduplicate_matches(self, matches: List[SemanticMatch]) -> List[SemanticMatch]:
        """
        Deduplicate matches by verse, merging multiple Strong's matches in the same verse.

        For verses with multiple matches (e.g., multiple Strong's numbers from same concept),
        creates a single match with combined evidence and highest confidence.
        """
        verse_groups = {}

        # Group matches by verse_id
        for match in matches:
            verse_id = match.verse_id
            if verse_id not in verse_groups:
                verse_groups[verse_id] = []
            verse_groups[verse_id].append(match)

        deduplicated = []

        for verse_id, verse_matches in verse_groups.items():
            if len(verse_matches) == 1:
                # No duplicates, keep as-is
                deduplicated.append(verse_matches[0])
            else:
                # Multiple matches for same verse - merge them
                merged = self._merge_verse_matches(verse_matches)
                deduplicated.append(merged)

        return deduplicated

    def _merge_verse_matches(self, matches: List[SemanticMatch]) -> SemanticMatch:
        """
        Merge multiple matches for the same verse into a single match.

        Uses the highest confidence and combines evidence from all matches.
        """
        # Sort by confidence (highest first)
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        best_match = sorted_matches[0]

        # Combine Strong's numbers from all matches
        all_strongs = []
        all_evidence = []

        for match in matches:
            all_strongs.extend(match.strongs_matched)
            if match.evidence and match.evidence not in all_evidence:
                all_evidence.append(match.evidence)

        # Remove duplicates while preserving order
        unique_strongs = []
        seen_strongs = set()
        for strongs in all_strongs:
            if strongs not in seen_strongs:
                unique_strongs.append(strongs)
                seen_strongs.add(strongs)

        # Create merged evidence
        combined_evidence = "; ".join(all_evidence)

        # Create new match with combined data
        return SemanticMatch(
            verse_id=best_match.verse_id,
            book=best_match.book,
            chapter=best_match.chapter,
            verse=best_match.verse,
            match_type=best_match.match_type,
            confidence=best_match.confidence,  # Use highest confidence
            semantic_score=best_match.semantic_score,
            ollama_validation=best_match.ollama_validation,
            ollama_confidence=best_match.ollama_confidence,
            evidence=combined_evidence,
            strongs_matched=unique_strongs,
            original_text=best_match.original_text,
            translation_text=best_match.translation_text,
            is_semantic_only=best_match.is_semantic_only,
        )

    def _convert_verse_id_to_embedding_format(self, verse_id: str) -> str:
        """
        Convert verse ID from Strong's format (1Co:2:9) to embedding format (046:002:009).

        Args:
            verse_id: Verse ID in format 'BookName:Chapter:Verse'

        Returns:
            Verse ID in format 'BookID:Chapter:Verse' with zero-padding
        """
        # Book name to book ID mapping (same as in semantic/pipeline.py)
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

        try:
            parts = verse_id.split(":")
            if len(parts) != 3:
                return verse_id  # Return original if format is unexpected

            book_name, chapter, verse = parts
            book_id = book_mapping.get(book_name)

            if book_id is None:
                return verse_id  # Return original if book not found

            # Format: BookID:Chapter:Verse with zero padding
            return f"{book_id:03d}:{int(chapter):03d}:{int(verse):03d}"

        except (ValueError, AttributeError):
            return verse_id  # Return original if conversion fails

    def _build_concept_prototypes(
        self, seed_matches: List[ConcordanceMatch]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build prototype embeddings from high-confidence matches."""
        hebrew_embeddings = []
        greek_embeddings = []

        for match in seed_matches:
            if match.confidence < 0.8:  # Only use high-confidence matches
                continue

            # Determine language from verse
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT language 
                    FROM stepbible_verses 
                    WHERE book = ? AND chapter = ? AND verse = ?
                """,
                    (match.book, match.chapter, match.verse),
                )

                languages = [row[0] for row in cursor.fetchall()]

                # Get embedding from appropriate collection
                if "hebrew" in languages:
                    try:
                        collection = self.chroma_manager.get_collection("original_verses")
                        embedding_id = self._convert_verse_id_to_embedding_format(match.verse_id)
                        result = collection.get(ids=[embedding_id], include=["embeddings"])
                        if result["embeddings"]:
                            hebrew_embeddings.append(np.array(result["embeddings"][0]))
                    except Exception as e:
                        logger.warning(f"Failed to get Hebrew embedding for {match.verse_id}: {e}")

                if "greek" in languages:
                    try:
                        collection = self.chroma_manager.get_collection("original_verses")
                        embedding_id = self._convert_verse_id_to_embedding_format(match.verse_id)
                        result = collection.get(ids=[embedding_id], include=["embeddings"])
                        if result["embeddings"]:
                            greek_embeddings.append(np.array(result["embeddings"][0]))
                    except Exception as e:
                        logger.warning(f"Failed to get Greek embedding for {match.verse_id}: {e}")

        # Calculate prototypes
        hebrew_prototype = np.mean(hebrew_embeddings, axis=0) if hebrew_embeddings else None
        greek_prototype = np.mean(greek_embeddings, axis=0) if greek_embeddings else None

        logger.info(f"Built prototypes - Hebrew: {hebrew_prototype is not None}, Greek: {greek_prototype is not None}")

        return hebrew_prototype, greek_prototype

    def _find_semantic_matches(
        self,
        hebrew_prototype: Optional[np.ndarray],
        greek_prototype: Optional[np.ndarray],
        max_results: int,
        exclude_verse_ids: Set[str],
    ) -> List[Dict]:
        """Find verses semantically similar to concept prototypes."""
        semantic_candidates = []

        collection = self.chroma_manager.get_collection("original_verses")

        # Search with Hebrew prototype
        if hebrew_prototype is not None:
            results = collection.query(
                query_embeddings=[hebrew_prototype.tolist()],
                n_results=max_results * 2,  # Get extra to account for filtering
                include=["metadatas", "distances"],
            )

            for i, verse_id in enumerate(results["ids"][0]):
                if verse_id not in exclude_verse_ids:
                    semantic_candidates.append(
                        {
                            "verse_id": verse_id,
                            "distance": results["distances"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "language": "hebrew",
                        }
                    )

        # Search with Greek prototype
        if greek_prototype is not None:
            results = collection.query(
                query_embeddings=[greek_prototype.tolist()],
                n_results=max_results * 2,
                include=["metadatas", "distances"],
            )

            for i, verse_id in enumerate(results["ids"][0]):
                if verse_id not in exclude_verse_ids and verse_id not in [c["verse_id"] for c in semantic_candidates]:
                    semantic_candidates.append(
                        {
                            "verse_id": verse_id,
                            "distance": results["distances"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "language": "greek",
                        }
                    )

        # Sort by distance and limit
        semantic_candidates.sort(key=lambda x: x["distance"])
        return semantic_candidates[:max_results]

    def _validate_semantic_matches(self, concept: ConceptDefinition, candidates: List[Dict]) -> List[SemanticMatch]:
        """Validate semantic matches using Ollama."""
        validated_matches = []

        for candidate in candidates:
            # Get verse data
            verse_data = self._get_verse_data(candidate["verse_id"])
            if not verse_data:
                continue

            # Validate with Ollama
            validation = self._validate_with_ollama(concept, verse_data)

            if validation.confidence >= 0.5:  # Accept YES and high-confidence MAYBE
                # Convert distance to similarity score (1 - normalized_distance)
                semantic_score = 1.0 - (candidate["distance"] / 2.0)  # Assuming max distance ~2

                match = SemanticMatch(
                    verse_id=candidate["verse_id"],
                    book=verse_data["book"],
                    chapter=verse_data["chapter"],
                    verse=verse_data["verse"],
                    match_type="semantic",
                    confidence=validation.confidence * 0.7,  # Scale down semantic confidence
                    evidence=f"Semantic match validated by {validation.model}",
                    strongs_matched=[],
                    original_text=verse_data["original_text"],
                    translation_text=None,
                    semantic_score=semantic_score,
                    ollama_validation=validation.answer,
                    ollama_confidence=validation.confidence,
                    is_semantic_only=True,
                )
                validated_matches.append(match)

        return validated_matches

    def _validate_with_ollama(self, concept: ConceptDefinition, verse_data: Dict) -> ValidationResult:
        """Validate a single match with Ollama."""
        # Use the OllamaAnalyzer's analyze_verse_for_concept method
        try:
            verse_ref = f"{verse_data['book']} {verse_data['chapter']}:{verse_data['verse']}"

            # Build verse text with original language and Strong's
            verse_text = verse_data.get("original_text", "")
            if verse_data.get("strongs_in_verse"):
                verse_text += f" (Strong's: {', '.join(verse_data.get('strongs_in_verse', []))})"

            # Use OllamaAnalyzer's method
            result = self.ollama.analyze_verse_for_concept(
                verse_text=verse_text,
                concept_name=concept.name,
                concept_description=concept.description,
                verse_reference=verse_ref,
            )

            # Convert result to our ValidationResult format
            # The analyze_verse_for_concept returns SemanticAnalysisResult with relevance_score
            if result.relevance_score >= 0.7:
                answer = "YES"
            elif result.relevance_score >= 0.4:
                answer = "MAYBE"
            else:
                answer = "NO"

            return ValidationResult(
                answer=answer, confidence=result.confidence, explanation=result.reasoning, model=self.ollama_models[0]
            )

        except Exception as e:
            logger.error(f"Ollama validation failed: {e}")
            return ValidationResult(
                answer="NO", confidence=0.0, explanation=f"Validation error: {str(e)}", model="error"
            )

    def _convert_embedding_id_to_verse_format(self, embedding_id: str) -> str:
        """
        Convert embedding ID format (046:003:016) back to verse format (Jhn:3:16).
        """
        # Reverse mapping from book ID to book name
        book_id_to_name = {
            1: "Gen",
            2: "Exo",
            3: "Lev",
            4: "Num",
            5: "Deu",
            6: "Jos",
            7: "Jdg",
            8: "Rut",
            9: "1Sa",
            10: "2Sa",
            11: "1Ki",
            12: "2Ki",
            13: "1Ch",
            14: "2Ch",
            15: "Ezr",
            16: "Neh",
            17: "Est",
            18: "Job",
            19: "Psa",
            20: "Pro",
            21: "Ecc",
            22: "Sng",
            23: "Isa",
            24: "Jer",
            25: "Lam",
            26: "Ezk",
            27: "Dan",
            28: "Hos",
            29: "Jol",
            30: "Amo",
            31: "Oba",
            32: "Jon",
            33: "Mic",
            34: "Nam",
            35: "Hab",
            36: "Zep",
            37: "Hag",
            38: "Zec",
            39: "Mal",
            40: "Mat",
            41: "Mrk",
            42: "Luk",
            43: "Jhn",
            44: "Act",
            45: "Rom",
            46: "1Co",
            47: "2Co",
            48: "Gal",
            49: "Eph",
            50: "Php",
            51: "Col",
            52: "1Th",
            53: "2Th",
            54: "1Ti",
            55: "2Ti",
            56: "Tit",
            57: "Phm",
            58: "Heb",
            59: "Jas",
            60: "1Pe",
            61: "2Pe",
            62: "1Jn",
            63: "2Jn",
            64: "3Jn",
            65: "Jud",
            66: "Rev",
        }

        try:
            parts = embedding_id.split(":")
            if len(parts) != 3:
                return embedding_id

            book_id = int(parts[0])
            chapter = int(parts[1])
            verse = int(parts[2])

            book_name = book_id_to_name.get(book_id)
            if not book_name:
                return embedding_id

            return f"{book_name}:{chapter}:{verse}"

        except (ValueError, AttributeError):
            return embedding_id

    def _get_verse_data(self, verse_id: str) -> Optional[Dict]:
        """Get verse data from database."""
        try:
            # Convert embedding format (046:003:016) to database format (Jhn:3:16)
            if verse_id and verse_id[0].isdigit():
                verse_id = self._convert_embedding_id_to_verse_format(verse_id)

            parts = verse_id.split(":")
            book = parts[0]
            chapter = int(parts[1])
            verse = int(parts[2])

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get all words in the verse
                cursor.execute(
                    """
                    SELECT 
                        original_word,
                        strongs_lexical,
                        morphology,
                        language
                    FROM stepbible_verses
                    WHERE book = ? AND chapter = ? AND verse = ?
                    ORDER BY id
                """,
                    (book, chapter, verse),
                )

                words = cursor.fetchall()
                if not words:
                    return None

                # Combine original text
                original_text = " ".join(w[0] for w in words if w[0])
                strongs_in_verse = [w[1] for w in words if w[1]]
                morphology = " ".join(w[2] for w in words if w[2])

                return {
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "original_text": original_text,
                    "strongs_in_verse": strongs_in_verse,
                    "morphology": morphology,
                    "language": words[0][3] if words else "unknown",
                }

        except Exception as e:
            logger.error(f"Failed to get verse data for {verse_id}: {e}")
            return None

    def _rank_combined_results(self, matches: List[SemanticMatch]) -> List[SemanticMatch]:
        """Rank combined lexical and semantic results."""
        # Calculate combined scores
        for match in matches:
            if match.is_semantic_only:
                # Semantic matches use weighted combination
                match.confidence = (
                    0.3 * match.semantic_score + 0.7 * match.ollama_confidence  # Embedding similarity  # LLM validation
                )
            # Lexical matches keep their original confidence

        # Sort by confidence, then by verse reference
        return sorted(matches, key=lambda m: (-m.confidence, m.book, m.chapter, m.verse))

    def generate_semantic_report(self, concept: ConceptDefinition, matches: List[SemanticMatch]) -> str:
        """Generate a detailed report including semantic matches."""
        report = []
        report.append(f"# Semantic Concordance Report: {concept.name}")
        report.append(f"\n## Methodology")
        report.append(f"- Primary Strong's numbers: {', '.join(concept.primary_strongs)}")
        report.append(f"- Semantic search: Original language embeddings")
        report.append(f"- Validation: Ollama LLM ({', '.join(self.ollama_models)})")

        # Separate match types
        lexical_matches = [m for m in matches if not m.is_semantic_only]
        semantic_matches = [m for m in matches if m.is_semantic_only]

        # Statistics
        report.append(f"\n## Statistics")
        report.append(f"- Total matches: {len(matches)}")
        report.append(f"- Lexical matches: {len(lexical_matches)}")
        report.append(f"- Semantic matches: {len(semantic_matches)}")

        if semantic_matches:
            validated_yes = sum(1 for m in semantic_matches if m.ollama_validation == "YES")
            validated_maybe = sum(1 for m in semantic_matches if m.ollama_validation == "MAYBE")
            report.append(f"- Semantic validation: {validated_yes} YES, {validated_maybe} MAYBE")

        # Sample matches
        report.append(f"\n## Top Lexical Matches")
        for match in lexical_matches[:5]:
            report.append(
                f"- {match.verse_id} - {match.original_text[:50]}... "
                f"(Confidence: {match.confidence:.2f}, Type: {match.match_type})"
            )

        if semantic_matches:
            report.append(f"\n## Top Semantic Matches")
            for match in semantic_matches[:5]:
                report.append(
                    f"- {match.verse_id} - {match.original_text[:50]}... "
                    f"(Confidence: {match.confidence:.2f}, Validation: {match.ollama_validation})"
                )

        return "\n".join(report)
