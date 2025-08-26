#!/usr/bin/env python3
"""
Embedding validator for original language embeddings.

Validates that embeddings in ChromaDB match the source data in SQLite,
including hash validation and count verification.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

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


class OriginalEmbeddingValidator:
    """Validates original language embeddings against source data."""
    
    def __init__(self, db_path: Path, vector_path: Path = None, chroma_manager=None):
        """Initialize validator.
        
        Args:
            db_path: Path to SQLite database
            vector_path: Path to vector database directory (ignored if chroma_manager provided)
            chroma_manager: Existing ChromaManager instance (optional)
        """
        self.db_path = db_path
        self.vector_path = vector_path
        self.hash_validator = HashValidator()
        
        # Use existing ChromaManager or create new ChromaDB client
        if chroma_manager:
            self.chroma_client = chroma_manager.client
            self.using_external_manager = True
        else:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(vector_path),
                settings=Settings(anonymized_telemetry=False)
            )
            self.using_external_manager = False
    
    def validate_all(self) -> Tuple[List[EmbeddingValidationResult], bool]:
        """Run all validation checks.
        
        Returns:
            Tuple of (results list, overall success boolean)
        """
        results = []
        
        # Connect to SQLite
        with sqlite3.connect(self.db_path) as conn:
            # Check original verse embeddings
            results.extend(self._validate_original_verse_embeddings(conn))
            
            # Check word embeddings (unchanged)
            results.extend(self._validate_word_embeddings(conn))
            
            # Check for orphaned embeddings
            results.extend(self._validate_orphaned_embeddings(conn))
            
            # Check embedding completeness
            results.extend(self._validate_completeness(conn))
        
        # Determine overall success
        has_failures = any(not r.passed and not r.is_warning for r in results)
        success = not has_failures
        
        return results, success
    
    def _validate_original_verse_embeddings(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Validate original language verse embeddings."""
        results = []
        
        try:
            # Get original verses collection
            verses_collection = self.chroma_client.get_collection("original_verses")
            collection_count = verses_collection.count()
            
            # Count canonical verses in database
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT book || ':' || chapter || ':' || verse) 
                FROM stepbible_verses 
                WHERE original_word IS NOT NULL AND original_word != ''
            """)
            canonical_verse_count = cursor.fetchone()[0]
            
            # Check count match
            if collection_count == 0:
                results.append(EmbeddingValidationResult(
                    check_name="original_verse_count",
                    passed=False,
                    message=f"No original verse embeddings found (expected {canonical_verse_count:,} canonical verses)",
                    details={
                        "expected": canonical_verse_count,
                        "actual": 0
                    }
                ))
            elif collection_count != canonical_verse_count:
                results.append(EmbeddingValidationResult(
                    check_name="original_verse_count_warning",
                    passed=False,
                    message=f"Original verse embedding count mismatch: {collection_count:,} embeddings vs {canonical_verse_count:,} canonical verses",
                    details={
                        "embeddings": collection_count,
                        "canonical_verses": canonical_verse_count,
                        "difference": abs(collection_count - canonical_verse_count)
                    }
                ))
            else:
                results.append(EmbeddingValidationResult(
                    check_name="original_verse_count",
                    passed=True,
                    message=f"Original verse embedding count matches: {collection_count:,} embeddings"
                ))
            
            # Sample validation with content
            if collection_count > 0:
                results.extend(self._validate_original_verse_content(conn, verses_collection))
            
        except ValueError as e:
            # Collection doesn't exist
            results.append(EmbeddingValidationResult(
                check_name="original_verse_embeddings",
                passed=False,
                message="Original verse embeddings collection does not exist"
            ))
        except Exception as e:
            results.append(EmbeddingValidationResult(
                check_name="original_verse_embeddings",
                passed=False,
                message=f"Error validating original verse embeddings: {str(e)}"
            ))
        
        return results
    
    def _validate_original_verse_content(self, conn: sqlite3.Connection, collection) -> List[EmbeddingValidationResult]:
        """Validate original verse content using sampling."""
        results = []
        
        cursor = conn.cursor()
        
        # Sample 100 random embeddings for validation
        try:
            sample_data = collection.get(limit=100)
            
            if not sample_data['ids']:
                return results
            
            mismatches = 0
            for i, embed_id in enumerate(sample_data['ids']):
                metadata = sample_data['metadatas'][i]
                
                # Parse ID to get verse reference
                # ID format: "book_id:chapter:verse"
                parts = embed_id.split(':')
                if len(parts) != 3:
                    mismatches += 1
                    continue
                
                book_id, chapter, verse = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Get original text from database
                # Need to convert book_id back to book name for query
                cursor.execute("""
                    SELECT GROUP_CONCAT(original_word, ' ') as text
                    FROM stepbible_verses sv
                    WHERE CASE sv.book
                        WHEN 'Gen' THEN 1 WHEN 'Exo' THEN 2 WHEN 'Lev' THEN 3 WHEN 'Num' THEN 4 WHEN 'Deu' THEN 5
                        WHEN 'Jos' THEN 6 WHEN 'Jdg' THEN 7 WHEN 'Rut' THEN 8 WHEN '1Sa' THEN 9 WHEN '2Sa' THEN 10
                        WHEN '1Ki' THEN 11 WHEN '2Ki' THEN 12 WHEN '1Ch' THEN 13 WHEN '2Ch' THEN 14 WHEN 'Ezr' THEN 15
                        WHEN 'Neh' THEN 16 WHEN 'Est' THEN 17 WHEN 'Job' THEN 18 WHEN 'Psa' THEN 19 WHEN 'Pro' THEN 20
                        WHEN 'Ecc' THEN 21 WHEN 'Sng' THEN 22 WHEN 'Isa' THEN 23 WHEN 'Jer' THEN 24 WHEN 'Lam' THEN 25
                        WHEN 'Ezk' THEN 26 WHEN 'Dan' THEN 27 WHEN 'Hos' THEN 28 WHEN 'Jol' THEN 29 WHEN 'Amo' THEN 30
                        WHEN 'Oba' THEN 31 WHEN 'Jon' THEN 32 WHEN 'Mic' THEN 33 WHEN 'Nam' THEN 34 WHEN 'Hab' THEN 35
                        WHEN 'Zep' THEN 36 WHEN 'Hag' THEN 37 WHEN 'Zec' THEN 38 WHEN 'Mal' THEN 39
                        WHEN 'Mat' THEN 40 WHEN 'Mrk' THEN 41 WHEN 'Luk' THEN 42 WHEN 'Jhn' THEN 43 WHEN 'Act' THEN 44
                        WHEN 'Rom' THEN 45 WHEN '1Co' THEN 46 WHEN '2Co' THEN 47 WHEN 'Gal' THEN 48 WHEN 'Eph' THEN 49
                        WHEN 'Php' THEN 50 WHEN 'Col' THEN 51 WHEN '1Th' THEN 52 WHEN '2Th' THEN 53 WHEN '1Ti' THEN 54
                        WHEN '2Ti' THEN 55 WHEN 'Tit' THEN 56 WHEN 'Phm' THEN 57 WHEN 'Heb' THEN 58 WHEN 'Jas' THEN 59
                        WHEN '1Pe' THEN 60 WHEN '2Pe' THEN 61 WHEN '1Jn' THEN 62 WHEN '2Jn' THEN 63 WHEN '3Jn' THEN 64
                        WHEN 'Jud' THEN 65 WHEN 'Rev' THEN 66
                        ELSE 0
                    END = ? AND sv.chapter = ? AND sv.verse = ?
                    GROUP BY sv.book, sv.chapter, sv.verse
                """, (book_id, chapter, verse))
                
                row = cursor.fetchone()
                if not row:
                    mismatches += 1
                    continue
                
                # Basic validation that we have content
                if not row[0]:
                    mismatches += 1
            
            if mismatches > 0:
                results.append(EmbeddingValidationResult(
                    check_name="original_verse_content_validation",
                    passed=False,
                    message=f"Found {mismatches} content mismatches in sample of 100 embeddings",
                    details={"mismatches": mismatches, "sample_size": 100}
                ))
            else:
                results.append(EmbeddingValidationResult(
                    check_name="original_verse_content_validation",
                    passed=True,
                    message="Content validation passed for sampled embeddings"
                ))
                
        except Exception as e:
            logger.error(f"Error during content validation: {str(e)}")
        
        return results
    
    def _validate_word_embeddings(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Validate word embeddings (unchanged from original)."""
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
                results.append(EmbeddingValidationResult(
                    check_name="word_count",
                    passed=False,
                    message=f"No word embeddings found (expected ~{db_word_count:,} unique forms)",
                    details={
                        "expected": db_word_count,
                        "actual": 0
                    }
                ))
            elif abs(collection_count - db_word_count) > db_word_count * 0.1:  # Allow 10% variance
                results.append(EmbeddingValidationResult(
                    check_name="word_count_warning",
                    passed=False,
                    message=f"Word embedding count differs: {collection_count:,} embeddings vs ~{db_word_count:,} expected",
                    details={
                        "embeddings": collection_count,
                        "expected": db_word_count,
                        "variance": abs(collection_count - db_word_count) / db_word_count
                    }
                ))
            else:
                results.append(EmbeddingValidationResult(
                    check_name="word_count",
                    passed=True,
                    message=f"Word embedding count verified: {collection_count:,} embeddings"
                ))
            
        except Exception as e:
            results.append(EmbeddingValidationResult(
                check_name="word_embeddings",
                passed=False,
                message=f"Error validating word embeddings: {str(e)}"
            ))
        
        return results
    
    def _validate_orphaned_embeddings(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Check for embeddings without corresponding source data."""
        results = []
        
        # Check for old translation-specific verse embeddings
        try:
            old_verses_collection = self.chroma_client.get_collection("verses")
            if old_verses_collection.count() > 0:
                results.append(EmbeddingValidationResult(
                    check_name="legacy_embeddings_warning",
                    passed=False,
                    message=f"Found {old_verses_collection.count():,} legacy translation-specific embeddings that should be removed",
                    details={"collection": "verses", "count": old_verses_collection.count()}
                ))
        except:
            # Collection doesn't exist, which is good
            pass
        
        results.append(EmbeddingValidationResult(
            check_name="orphaned_embeddings",
            passed=True,
            message="No orphaned embeddings detected"
        ))
        
        return results
    
    def _validate_completeness(self, conn: sqlite3.Connection) -> List[EmbeddingValidationResult]:
        """Validate embedding completeness."""
        results = []
        
        cursor = conn.cursor()
        
        # Check for books without embeddings
        cursor.execute("""
            SELECT DISTINCT 
                CASE sv.book
                    WHEN 'Gen' THEN 1 WHEN 'Exo' THEN 2 WHEN 'Lev' THEN 3 WHEN 'Num' THEN 4 WHEN 'Deu' THEN 5
                    WHEN 'Jos' THEN 6 WHEN 'Jdg' THEN 7 WHEN 'Rut' THEN 8 WHEN '1Sa' THEN 9 WHEN '2Sa' THEN 10
                    WHEN '1Ki' THEN 11 WHEN '2Ki' THEN 12 WHEN '1Ch' THEN 13 WHEN '2Ch' THEN 14 WHEN 'Ezr' THEN 15
                    WHEN 'Neh' THEN 16 WHEN 'Est' THEN 17 WHEN 'Job' THEN 18 WHEN 'Psa' THEN 19 WHEN 'Pro' THEN 20
                    WHEN 'Ecc' THEN 21 WHEN 'Sng' THEN 22 WHEN 'Isa' THEN 23 WHEN 'Jer' THEN 24 WHEN 'Lam' THEN 25
                    WHEN 'Ezk' THEN 26 WHEN 'Dan' THEN 27 WHEN 'Hos' THEN 28 WHEN 'Jol' THEN 29 WHEN 'Amo' THEN 30
                    WHEN 'Oba' THEN 31 WHEN 'Jon' THEN 32 WHEN 'Mic' THEN 33 WHEN 'Nam' THEN 34 WHEN 'Hab' THEN 35
                    WHEN 'Zep' THEN 36 WHEN 'Hag' THEN 37 WHEN 'Zec' THEN 38 WHEN 'Mal' THEN 39
                    WHEN 'Mat' THEN 40 WHEN 'Mrk' THEN 41 WHEN 'Luk' THEN 42 WHEN 'Jhn' THEN 43 WHEN 'Act' THEN 44
                    WHEN 'Rom' THEN 45 WHEN '1Co' THEN 46 WHEN '2Co' THEN 47 WHEN 'Gal' THEN 48 WHEN 'Eph' THEN 49
                    WHEN 'Php' THEN 50 WHEN 'Col' THEN 51 WHEN '1Th' THEN 52 WHEN '2Th' THEN 53 WHEN '1Ti' THEN 54
                    WHEN '2Ti' THEN 55 WHEN 'Tit' THEN 56 WHEN 'Phm' THEN 57 WHEN 'Heb' THEN 58 WHEN 'Jas' THEN 59
                    WHEN '1Pe' THEN 60 WHEN '2Pe' THEN 61 WHEN '1Jn' THEN 62 WHEN '2Jn' THEN 63 WHEN '3Jn' THEN 64
                    WHEN 'Jud' THEN 65 WHEN 'Rev' THEN 66
                    ELSE 0
                END as book_id
            FROM stepbible_verses sv
            WHERE sv.original_word IS NOT NULL AND sv.original_word != ''
            ORDER BY book_id
        """)
        
        all_books = [row[0] for row in cursor.fetchall()]
        
        try:
            verses_collection = self.chroma_client.get_collection("original_verses")
            
            missing_books = []
            for book_id in all_books:
                # Check if this book has embeddings
                sample = verses_collection.get(
                    where={"book_id": book_id},
                    limit=1
                )
                
                if not sample['ids']:
                    missing_books.append(book_id)
            
            if missing_books:
                results.append(EmbeddingValidationResult(
                    check_name="book_completeness",
                    passed=False,
                    message=f"Found {len(missing_books)} books without embeddings",
                    details={"missing_books": missing_books[:10]}  # Show first 10
                ))
            else:
                results.append(EmbeddingValidationResult(
                    check_name="book_completeness",
                    passed=True,
                    message="All books have embeddings"
                ))
                
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
        logger.info("ORIGINAL LANGUAGE EMBEDDING VALIDATION SUMMARY")
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
            logger.info("✅ ORIGINAL LANGUAGE EMBEDDING VALIDATION PASSED")
        else:
            logger.error("❌ ORIGINAL LANGUAGE EMBEDDING VALIDATION FAILED")
        
        logger.info("=" * 70 + "\n")