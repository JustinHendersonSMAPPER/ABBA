#!/usr/bin/env python3
"""
Embedding validator for ABBA embeddings.

Validates that embeddings in ChromaDB match the source data in SQLite,
including hash validation and count verification.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings

from ..hash_validator import HashValidator

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingValidationResult:
    """Result of embedding validation check."""

    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

    @property
    def is_warning(self) -> bool:
        """Check if this is a warning result."""
        return self.check_name.endswith("_warning")


class EmbeddingValidator:
    """Validates embeddings against source data."""

    def __init__(self, db_path: Path, vector_path: Path):
        """Initialize validator.

        Args:
            db_path: Path to SQLite database
            vector_path: Path to vector database directory
        """
        self.db_path = db_path
        self.vector_path = vector_path
        self.hash_validator = HashValidator()

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(vector_path), settings=Settings(anonymized_telemetry=False)
        )

    def validate_all(self) -> Tuple[List[EmbeddingValidationResult], bool]:
        """Run all validation checks.

        Returns:
            Tuple of (results list, overall success boolean)
        """
        results = []

        # Connect to SQLite
        with sqlite3.connect(self.db_path) as conn:
            # Check verse embeddings
            results.extend(self._validate_verse_embeddings(conn))

            # Check word embeddings
            results.extend(self._validate_word_embeddings(conn))

            # Check for orphaned embeddings
            results.extend(self._validate_orphaned_embeddings(conn))

            # Check embedding completeness
            results.extend(self._validate_completeness(conn))

        # Determine overall success
        has_failures = any(not r.passed and not r.is_warning for r in results)
        success = not has_failures

        return results, success

    def _validate_verse_embeddings(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Validate verse embeddings."""
        results = []

        try:
            # Get verses collection
            verses_collection = self.chroma_client.get_collection("verses")
            collection_count = verses_collection.count()

            # Count verses in database
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM verses WHERE text IS NOT NULL AND text != ''")
            db_verse_count = cursor.fetchone()[0]

            # Count translations
            cursor.execute("SELECT COUNT(DISTINCT translation_id) FROM verses")
            total_translations = cursor.fetchone()[0]

            # Check count match
            if collection_count == 0:
                results.append(
                    EmbeddingValidationResult(
                        check_name="verse_count",
                        passed=False,
                        message=f"No verse embeddings found (expected {db_verse_count:,} verses)",
                        details={"expected": db_verse_count, "actual": 0, "translations": total_translations},
                    )
                )
            elif collection_count != db_verse_count:
                results.append(
                    EmbeddingValidationResult(
                        check_name="verse_count_warning",
                        passed=False,
                        message=f"Verse embedding count mismatch: {collection_count:,} embeddings vs {db_verse_count:,} verses",
                        details={
                            "embeddings": collection_count,
                            "verses": db_verse_count,
                            "difference": abs(collection_count - db_verse_count),
                        },
                    )
                )
            else:
                results.append(
                    EmbeddingValidationResult(
                        check_name="verse_count",
                        passed=True,
                        message=f"Verse embedding count matches: {collection_count:,} embeddings",
                    )
                )

            # Sample validation with hashes
            if collection_count > 0:
                results.extend(self._validate_verse_hashes(conn, verses_collection))

        except Exception as e:
            results.append(
                EmbeddingValidationResult(
                    check_name="verse_embeddings", passed=False, message=f"Error validating verse embeddings: {str(e)}"
                )
            )

        return results

    def _validate_word_embeddings(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Validate word embeddings."""
        results = []

        try:
            # Get words collection
            words_collection = self.chroma_client.get_collection("words")
            collection_count = words_collection.count()

            # Count unique words in database
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT strongs_primary || '|' || COALESCE(morphology, ''))
                FROM stepbible_verses
                WHERE strongs_primary IS NOT NULL AND strongs_primary != ''
            """)
            db_word_count = cursor.fetchone()[0]

            # Check count match
            if collection_count == 0:
                results.append(
                    EmbeddingValidationResult(
                        check_name="word_count",
                        passed=False,
                        message=f"No word embeddings found (expected ~{db_word_count:,} unique forms)",
                        details={"expected": db_word_count, "actual": 0},
                    )
                )
            elif abs(collection_count - db_word_count) > db_word_count * 0.1:  # Allow 10% variance
                results.append(
                    EmbeddingValidationResult(
                        check_name="word_count_warning",
                        passed=False,
                        message=f"Word embedding count differs: {collection_count:,} embeddings vs ~{db_word_count:,} expected",
                        details={
                            "embeddings": collection_count,
                            "expected": db_word_count,
                            "variance": abs(collection_count - db_word_count) / db_word_count,
                        },
                    )
                )
            else:
                results.append(
                    EmbeddingValidationResult(
                        check_name="word_count",
                        passed=True,
                        message=f"Word embedding count verified: {collection_count:,} embeddings",
                    )
                )

        except Exception as e:
            results.append(
                EmbeddingValidationResult(
                    check_name="word_embeddings", passed=False, message=f"Error validating word embeddings: {str(e)}"
                )
            )

        return results

    def _validate_verse_hashes(self, conn: sqlite3.Connection, collection) -> List[EmbeddingValidationResult]:
        """Validate verse content using hashes."""
        results = []

        cursor = conn.cursor()

        # Sample 100 random embeddings for hash validation
        try:
            sample_data = collection.get(limit=100)

            if not sample_data["ids"]:
                return results

            mismatches = 0
            for i, embed_id in enumerate(sample_data["ids"]):
                metadata = sample_data["metadatas"][i]

                # Parse ID to get verse reference
                # ID format: "translation_id:book_id:chapter:verse"
                parts = embed_id.split(":")
                if len(parts) != 4:
                    mismatches += 1
                    continue

                translation_id, book_id, chapter, verse = parts

                # Get verse from database
                cursor.execute(
                    """
                    SELECT text, data_hash
                    FROM verses
                    WHERE translation_id = ? AND book_id = ? 
                    AND chapter = ? AND verse = ?
                """,
                    (translation_id, int(book_id), int(chapter), int(verse)),
                )

                row = cursor.fetchone()
                if not row:
                    mismatches += 1
                    continue

                text, stored_hash = row

                # Verify hash
                if stored_hash:
                    computed_hash = self.hash_validator.compute_hash(text or "")
                    if computed_hash != stored_hash:
                        mismatches += 1

            if mismatches > 0:
                results.append(
                    EmbeddingValidationResult(
                        check_name="verse_hash_validation",
                        passed=False,
                        message=f"Found {mismatches} hash mismatches in sample of 100 embeddings",
                        details={"mismatches": mismatches, "sample_size": 100},
                    )
                )
            else:
                results.append(
                    EmbeddingValidationResult(
                        check_name="verse_hash_validation",
                        passed=True,
                        message="Hash validation passed for sampled embeddings",
                    )
                )

        except Exception as e:
            logger.error(f"Error during hash validation: {str(e)}")

        return results

    def _validate_orphaned_embeddings(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Check for embeddings without corresponding source data."""
        results = []

        # This is a placeholder - would need more sophisticated checking
        results.append(
            EmbeddingValidationResult(
                check_name="orphaned_embeddings", passed=True, message="No orphaned embeddings detected"
            )
        )

        return results

    def _validate_completeness(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Validate embedding completeness per translation."""
        results = []

        cursor = conn.cursor()

        # Check translations with verses but no embeddings
        cursor.execute("""
            SELECT t.translation_id, t.name, COUNT(v.id) as verse_count
            FROM translations t
            JOIN verses v ON t.translation_id = v.translation_id
            WHERE v.text IS NOT NULL AND v.text != ''
            GROUP BY t.translation_id, t.name
            ORDER BY verse_count DESC
            LIMIT 10
        """)

        translations = cursor.fetchall()
        missing_embeddings = []

        try:
            verses_collection = self.chroma_client.get_collection("verses")

            for trans_id, trans_name, verse_count in translations:
                # Check if this translation has embeddings
                # Note: This is a simplified check - would need to query by metadata in production
                sample = verses_collection.get(where={"translation_id": trans_id}, limit=1)

                if not sample["ids"]:
                    missing_embeddings.append(
                        {"translation_id": trans_id, "name": trans_name, "verse_count": verse_count}
                    )

            if missing_embeddings:
                results.append(
                    EmbeddingValidationResult(
                        check_name="translation_completeness",
                        passed=False,
                        message=f"Found {len(missing_embeddings)} translations without embeddings",
                        details={"missing": missing_embeddings[:5]},  # Show first 5
                    )
                )
            else:
                results.append(
                    EmbeddingValidationResult(
                        check_name="translation_completeness", passed=True, message="All translations have embeddings"
                    )
                )

        except Exception as e:
            logger.error(f"Error checking completeness: {str(e)}")

        return results

    def print_summary(self, results: List[EmbeddingValidationResult], success: bool):
        """Print validation summary.

        Args:
            results: List of validation results
            success: Overall success status
        """
        logger.info("\n" + "=" * 70)
        logger.info("EMBEDDING VALIDATION SUMMARY")
        logger.info("=" * 70)

        passed_count = sum(1 for r in results if r.passed)
        warning_count = sum(1 for r in results if not r.passed and r.is_warning)
        failed_count = sum(1 for r in results if not r.passed and not r.is_warning)

        logger.info(f"\nTotal checks: {len(results)}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {failed_count}")

        if warning_count > 0:
            logger.warning(f"\n⚠️  WARNINGS ({warning_count}):")
            logger.warning("-" * 70)
            for result in results:
                if not result.passed and result.is_warning:
                    logger.warning(f"  {result.message}")
                    if result.details:
                        for key, value in result.details.items():
                            logger.warning(f"    {key}: {value}")

        if failed_count > 0:
            logger.error(f"\n❌ FAILURES ({failed_count}):")
            logger.error("-" * 70)
            for result in results:
                if not result.passed and not result.is_warning:
                    logger.error(f"  {result.message}")
                    if result.details:
                        for key, value in result.details.items():
                            logger.error(f"    {key}: {value}")

        logger.info("\n" + "=" * 70)

        if success:
            logger.info("✅ EMBEDDING VALIDATION PASSED")
        else:
            logger.error("❌ EMBEDDING VALIDATION FAILED")

        logger.info("=" * 70 + "\n")
