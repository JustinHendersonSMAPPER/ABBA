"""Post-import data validator to ensure metadata completeness."""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..parallel_import import Canon, get_translation_canon, BOOK_ID_MAP, ALL_KNOWN_EXTENDED_BOOKS

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    translation_id: str
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationSummary:
    """Summary of all validation results."""
    total_translations: int
    passed_translations: int
    failed_translations: int
    percentage: float
    failures: List[ValidationResult]
    warnings: List[ValidationResult]
    info_messages: List[ValidationResult] = None


class PostImportValidator:
    """Validates data integrity and metadata completeness after import."""
    
    def __init__(self, abba_db_path: Path, source_db_path: Path):
        """Initialize validator with database paths.
        
        Args:
            abba_db_path: Path to ABBA database
            source_db_path: Path to source bible.db
        """
        self.abba_db_path = abba_db_path
        self.source_db_path = source_db_path
        
    def validate_all_translations(self) -> ValidationSummary:
        """Run all validation checks on imported translations.
        
        Returns:
            ValidationSummary with results
        """
        results = []
        
        with sqlite3.connect(self.abba_db_path) as abba_conn:
            cursor = abba_conn.cursor()
            
            # Get translations that actually have verses imported (not just metadata)
            cursor.execute("""
                SELECT DISTINCT t.id 
                FROM translations t 
                INNER JOIN verses v ON t.id = v.translation_id 
                ORDER BY t.id
            """)
            translation_ids = [row[0] for row in cursor.fetchall()]
            
            if not translation_ids:
                return ValidationSummary(
                    total_translations=0,
                    passed_translations=0,
                    failed_translations=0,
                    percentage=0.0,
                    failures=[],
                    warnings=[]
                )
            
            # Run validation checks for each translation
            for trans_id in translation_ids:
                # Check metadata completeness
                results.extend(self._validate_translation_metadata(trans_id, abba_conn))
                
                # Check verse counts
                results.extend(self._validate_verse_counts(trans_id, abba_conn))
                
                # Check book coverage
                results.extend(self._validate_book_coverage(trans_id, abba_conn))
                
                # Check canon association
                results.extend(self._validate_canon_association(trans_id, abba_conn))
                
                # Check data integrity
                results.extend(self._validate_data_integrity(trans_id, abba_conn))
        
        # Summarize results
        failures = [r for r in results if not r.passed and r.check_name not in ["warning", "info"]]
        warnings = [r for r in results if not r.passed and r.check_name == "warning"]
        info_messages = [r for r in results if r.check_name == "info"]
        
        failed_translations = len(set(r.translation_id for r in failures))
        passed_translations = len(translation_ids) - failed_translations
        percentage = (passed_translations / len(translation_ids)) * 100 if translation_ids else 0
        
        return ValidationSummary(
            total_translations=len(translation_ids),
            passed_translations=passed_translations,
            failed_translations=failed_translations,
            percentage=percentage,
            failures=failures,
            warnings=warnings,
            info_messages=info_messages
        )
    
    def _validate_translation_metadata(self, trans_id: str, conn: sqlite3.Connection) -> List[ValidationResult]:
        """Check if translation has all required metadata fields."""
        results = []
        cursor = conn.cursor()
        
        # Check required fields
        cursor.execute("""
            SELECT id, name, english_name, language, canon
            FROM translations
            WHERE id = ?
        """, (trans_id,))
        
        row = cursor.fetchone()
        if not row:
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="metadata_exists",
                passed=False,
                message=f"Translation {trans_id} not found in database"
            ))
            return results
        
        _, name, english_name, language, canon = row
        
        # Check each required field
        if not name or name.strip() == '':
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="metadata_name",
                passed=False,
                message=f"Translation {trans_id} missing 'name' field"
            ))
        
        if not english_name or english_name.strip() == '':
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="metadata_english_name",
                passed=False,
                message=f"Translation {trans_id} missing 'english_name' field"
            ))
            
        if not language or language.strip() == '':
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="metadata_language",
                passed=False,
                message=f"Translation {trans_id} missing 'language' field"
            ))
            
        if not canon or canon.strip() == '':
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="metadata_canon",
                passed=False,
                message=f"Translation {trans_id} missing 'canon' field"
            ))
        
        # If all passed, add success result
        if not any(not r.passed for r in results):
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="metadata_complete",
                passed=True,
                message=f"All metadata fields present for {trans_id}"
            ))
        
        return results
    
    def _validate_verse_counts(self, trans_id: str, conn: sqlite3.Connection) -> List[ValidationResult]:
        """Validate verse counts match source."""
        results = []
        cursor = conn.cursor()
        
        # Get verse count from ABBA
        cursor.execute("""
            SELECT COUNT(*) FROM verses WHERE translation_id = ?
        """, (trans_id,))
        abba_count = cursor.fetchone()[0]
        
        # Get expected count from source (only mapped books)
        with sqlite3.connect(self.source_db_path) as source_conn:
            source_cursor = source_conn.cursor()
            
            # Count verses in mapped books only
            source_cursor.execute("""
                SELECT COUNT(*) 
                FROM ChapterVerse 
                WHERE translationId = ?
            """, (trans_id,))
            total_source_count = source_cursor.fetchone()[0]
            
            # Count verses in mapped books
            source_cursor.execute("""
                SELECT bookId, COUNT(*) as count
                FROM ChapterVerse 
                WHERE translationId = ?
                GROUP BY bookId
            """, (trans_id,))
            
            mapped_count = 0
            for book_id, count in source_cursor.fetchall():
                if BOOK_ID_MAP.get(book_id, 0) > 0:
                    mapped_count += count
        
        # Check if counts match
        if abba_count == mapped_count:
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="verse_count",
                passed=True,
                message=f"Verse count correct: {abba_count:,} verses",
                details={"imported": abba_count, "expected": mapped_count, "total_source": total_source_count}
            ))
        else:
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="verse_count",
                passed=False,
                message=f"Verse count mismatch: imported {abba_count:,}, expected {mapped_count:,}",
                details={"imported": abba_count, "expected": mapped_count, "total_source": total_source_count}
            ))
        
        return results
    
    def _validate_book_coverage(self, trans_id: str, conn: sqlite3.Connection) -> List[ValidationResult]:
        """Check if all expected books were imported."""
        results = []
        cursor = conn.cursor()
        
        # Get imported books
        cursor.execute("""
            SELECT DISTINCT book_id 
            FROM verses 
            WHERE translation_id = ?
            ORDER BY book_id
        """, (trans_id,))
        imported_books = set(row[0] for row in cursor.fetchall())
        
        # Get expected books from source
        with sqlite3.connect(self.source_db_path) as source_conn:
            source_cursor = source_conn.cursor()
            source_cursor.execute("""
                SELECT DISTINCT bookId
                FROM ChapterVerse 
                WHERE translationId = ?
            """, (trans_id,))
            
            source_books = []
            for row in source_cursor.fetchall():
                book_str = row[0]
                book_id = BOOK_ID_MAP.get(book_str, 0)
                if book_id > 0:
                    source_books.append(book_id)
        
        expected_books = set(source_books)
        
        # Check coverage
        missing_books = expected_books - imported_books
        extra_books = imported_books - expected_books
        
        if not missing_books and not extra_books:
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="book_coverage",
                passed=True,
                message=f"All {len(imported_books)} expected books imported",
                details={"imported": len(imported_books), "expected": len(expected_books)}
            ))
        else:
            if missing_books:
                results.append(ValidationResult(
                    translation_id=trans_id,
                    check_name="book_coverage",
                    passed=False,
                    message=f"Missing {len(missing_books)} books: {sorted(missing_books)}",
                    details={"missing": sorted(missing_books)}
                ))
            if extra_books:
                results.append(ValidationResult(
                    translation_id=trans_id,
                    check_name="book_coverage",
                    passed=False,
                    message=f"Extra {len(extra_books)} unexpected books: {sorted(extra_books)}",
                    details={"extra": sorted(extra_books)}
                ))
        
        return results
    
    def _validate_canon_association(self, trans_id: str, conn: sqlite3.Connection) -> List[ValidationResult]:
        """Validate canon detection and association."""
        results = []
        
        # Detect canon
        canon = get_translation_canon(trans_id, str(self.source_db_path))
        
        # Get book statistics
        with sqlite3.connect(self.source_db_path) as source_conn:
            source_cursor = source_conn.cursor()
            source_cursor.execute("""
                SELECT DISTINCT bookId
                FROM ChapterVerse 
                WHERE translationId = ?
            """, (trans_id,))
            
            all_books = [row[0] for row in source_cursor.fetchall()]
            
            # Categorize books
            standard_books = [b for b in all_books if BOOK_ID_MAP.get(b, 0) > 0]
            extended_books = [b for b in all_books if b in ALL_KNOWN_EXTENDED_BOOKS and BOOK_ID_MAP.get(b, 0) == 0]
            unknown_books = [b for b in all_books if BOOK_ID_MAP.get(b, 0) == 0 and b not in ALL_KNOWN_EXTENDED_BOOKS]
        
        # Add canon info
        results.append(ValidationResult(
            translation_id=trans_id,
            check_name="canon_detection",
            passed=True,
            message=f"Detected as {canon.value} canon",
            details={
                "canon": canon.value,
                "total_books": len(all_books),
                "standard_books": len(standard_books),
                "extended_books": len(extended_books),
                "unknown_books": len(unknown_books)
            }
        ))
        
        # Warn about unknown books
        if unknown_books:
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="warning",
                passed=False,
                message=f"Contains {len(unknown_books)} unrecognized books: {unknown_books}",
                details={"unknown_books": unknown_books}
            ))
        
        return results
    
    def _validate_data_integrity(self, trans_id: str, conn: sqlite3.Connection) -> List[ValidationResult]:
        """Check for actual import failures by comparing with source data."""
        results = []
        cursor = conn.cursor()
        
        # Check for duplicate verses
        cursor.execute("""
            SELECT book_id, chapter, verse, COUNT(*) as count
            FROM verses
            WHERE translation_id = ?
            GROUP BY book_id, chapter, verse
            HAVING COUNT(*) > 1
        """, (trans_id,))
        
        duplicates = cursor.fetchall()
        if duplicates:
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="duplicate_verses",
                passed=False,
                message=f"Found {len(duplicates)} duplicate verse entries",
                details={"duplicates": [(b, c, v, cnt) for b, c, v, cnt in duplicates[:10]]}  # First 10
            ))
        
        # Check for actual missing imports by comparing with source
        with sqlite3.connect(self.source_db_path) as source_conn:
            source_cursor = source_conn.cursor()
            
            # Get all verses that exist in source for this translation
            source_cursor.execute("""
                SELECT bookId, chapterNumber, number, text
                FROM ChapterVerse
                WHERE translationId = ?
            """, (trans_id,))
            
            source_verses = {}
            empty_in_source = 0
            
            for book_str, chapter, verse, text in source_cursor.fetchall():
                # Map book ID
                book_id = BOOK_ID_MAP.get(book_str, 0)
                if book_id > 0:  # Only track mappable books
                    key = (book_id, chapter, verse)
                    source_verses[key] = text
                    if not text or text.strip() == '':
                        empty_in_source += 1
            
            # Now check what was actually imported
            cursor.execute("""
                SELECT book_id, chapter, verse, text
                FROM verses
                WHERE translation_id = ?
            """, (trans_id,))
            
            imported_verses = {}
            for book_id, chapter, verse, text in cursor.fetchall():
                key = (book_id, chapter, verse)
                imported_verses[key] = text
            
            # Find verses that exist in source but not in imported
            missing_verses = set(source_verses.keys()) - set(imported_verses.keys())
            
            if missing_verses:
                # This is a real problem - verses exist in source but failed to import
                results.append(ValidationResult(
                    translation_id=trans_id,
                    check_name="missing_verses",
                    passed=False,
                    message=f"FAILED TO IMPORT {len(missing_verses)} verses that exist in source",
                    details={
                        "missing_count": len(missing_verses),
                        "examples": sorted(list(missing_verses))[:10]  # First 10 examples
                    }
                ))
                
                # Record these failures in the database
                cursor.executemany("""
                    INSERT OR IGNORE INTO failed_imports 
                    (translation_id, book_id, chapter, verse, reason)
                    VALUES (?, ?, ?, ?, 'Import failed')
                """, [(trans_id, book_id, chapter, verse) for book_id, chapter, verse in missing_verses])
                
                # Update translation record
                cursor.execute("""
                    UPDATE translations 
                    SET has_import_failures = 1, failed_verse_count = ?
                    WHERE id = ?
                """, (len(missing_verses), trans_id))
            
            # Report empty verses that are also empty in source (debug only)
            if empty_in_source > 0:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM verses
                    WHERE translation_id = ? AND (text IS NULL OR text = '')
                """, (trans_id,))
                
                empty_imported = cursor.fetchone()[0]
                if empty_imported > 0:
                    logger.debug(f"{trans_id}: Has {empty_imported} empty verses (also empty in source)")
        
        # If no issues found
        if not any(not r.passed for r in results):
            results.append(ValidationResult(
                translation_id=trans_id,
                check_name="data_integrity",
                passed=True,
                message="All source verses successfully imported"
            ))
        
        return results
    
    def print_summary(self, summary: ValidationSummary):
        """Print a formatted summary of validation results."""
        from ..logging_setup import get_logger
        logger = get_logger(__name__)
        
        logger.info("\n" + "="*70)
        logger.info("POST-IMPORT VALIDATION SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\nTotal translations validated: {summary.total_translations}")
        logger.info(f"Passed: {summary.passed_translations}")
        logger.info(f"Failed: {summary.failed_translations}")
        logger.info(f"Success rate: {summary.percentage:.1f}%")
        
        if summary.failures:
            logger.error(f"\n❌ FAILURES ({len(summary.failures)}):")
            logger.error("-" * 70)
            for failure in summary.failures:
                logger.error(f"  {failure.translation_id}: {failure.message}")
                if failure.details:
                    logger.error(f"    Details: {failure.details}")
        
        
        logger.info("\n" + "="*70)
        
        if summary.percentage < 100:
            logger.error("❌ VALIDATION FAILED - Not all translations passed validation")
            logger.error("Please review the failures above before continuing.")
        else:
            logger.info("✅ ALL VALIDATIONS PASSED - Safe to continue")
        
        logger.info("="*70 + "\n")