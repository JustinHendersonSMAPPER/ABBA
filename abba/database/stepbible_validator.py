"""STEPBible data validator with hash-based integrity checking."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mmh3

from abba.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class STEPBibleValidationResult:
    """Result of STEPBible validation check."""

    check_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


@dataclass
class STEPBibleValidationSummary:
    """Summary of all STEPBible validation results."""

    total_checks: int
    passed_checks: int
    failed_checks: int
    percentage: float
    failures: List[STEPBibleValidationResult]
    warnings: List[STEPBibleValidationResult]
    hash_mismatches: List[Dict]


class STEPBibleValidator:
    """Validates STEPBible data integrity and completeness."""

    # Expected files and their purposes
    EXPECTED_FILES: Dict[str, Dict[str, Any]] = {
        "tahot_gen_deu.txt": {"type": "hebrew_text", "books": ["Gen", "Exo", "Lev", "Num", "Deu"]},
        "tahot_jos_est.txt": {
            "type": "hebrew_text",
            "books": ["Jos", "Jdg", "Rut", "1Sa", "2Sa", "1Ki", "2Ki", "1Ch", "2Ch", "Ezr", "Neh", "Est"],
        },
        "tahot_job_sng.txt": {"type": "hebrew_text", "books": ["Job", "Psa", "Pro", "Ecc", "Sng"]},
        "tahot_isa_mal.txt": {
            "type": "hebrew_text",
            "books": [
                "Isa",
                "Jer",
                "Lam",
                "Ezk",
                "Dan",
                "Hos",
                "Jol",
                "Amo",
                "Oba",
                "Jon",
                "Mic",
                "Nam",
                "Hab",
                "Zep",
                "Hag",
                "Zec",
                "Mal",
            ],
        },
        "tagnt_mat_jhn.txt": {"type": "greek_text", "books": ["Mat", "Mrk", "Luk", "Jhn"]},
        "tagnt_act_rev.txt": {
            "type": "greek_text",
            "books": [
                "Act",
                "Rom",
                "1Co",
                "2Co",
                "Gal",
                "Eph",
                "Php",
                "Col",
                "1Th",
                "2Th",
                "1Ti",
                "2Ti",
                "Tit",
                "Phm",
                "Heb",
                "Jas",
                "1Pe",
                "2Pe",
                "1Jn",
                "2Jn",
                "3Jn",
                "Jud",
                "Rev",
            ],
        },
        "tbesh.txt": {"type": "hebrew_lexicon"},
        "tbesg.txt": {"type": "greek_lexicon"},
        "tehmc.txt": {"type": "hebrew_morphology"},
        "tegmc.txt": {"type": "greek_morphology"},
    }

    # Minimum expected counts for validation
    MIN_EXPECTED_COUNTS = {
        "hebrew_words": 280000,  # ~283k Hebrew words in OT
        "greek_words": 137000,  # ~137k Greek words in NT
        "hebrew_lexicon": 8000,  # ~8.7k Hebrew lexicon entries
        "greek_lexicon": 5000,  # ~5.6k Greek lexicon entries
        "hebrew_morphology": 1000,  # Morphology codes
        "greek_morphology": 1000,
    }

    def __init__(self, db_path: Path):
        """Initialize validator with database path."""
        self.db_path = db_path

    def validate_all(self) -> STEPBibleValidationSummary:
        """Run all validation checks on STEPBible data."""
        results = []
        hash_mismatches = []

        with sqlite3.connect(self.db_path) as conn:
            # Check file completeness
            results.extend(self._validate_file_completeness(conn))

            # Check word counts
            results.extend(self._validate_word_counts(conn))

            # Check lexicon completeness
            results.extend(self._validate_lexicon_completeness(conn))

            # Check morphology completeness
            results.extend(self._validate_morphology_completeness(conn))

            # Check data integrity with hashes
            integrity_results, mismatches = self._validate_data_integrity(conn)
            results.extend(integrity_results)
            hash_mismatches.extend(mismatches)

            # Check cross-references
            results.extend(self._validate_cross_references(conn))

            # Check for data anomalies
            results.extend(self._validate_data_anomalies(conn))

        # Summarize results
        failures = [r for r in results if not r.passed and "warning" not in r.check_name.lower()]
        warnings = [r for r in results if not r.passed and "warning" in r.check_name.lower()]

        failed_checks = len(failures)
        passed_checks = len(results) - len(failures) - len(warnings)
        total_checks = len(results)
        percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        return STEPBibleValidationSummary(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            percentage=percentage,
            failures=failures,
            warnings=warnings,
            hash_mismatches=hash_mismatches,
        )

    def _validate_file_completeness(self, conn: sqlite3.Connection) -> List[STEPBibleValidationResult]:
        """Check if all expected files were imported."""
        results = []
        cursor = conn.cursor()

        # Get imported files
        cursor.execute("SELECT DISTINCT source_file FROM stepbible_verses")
        imported_files = set(row[0] for row in cursor.fetchall())

        # Check text files
        text_files = [f for f, info in self.EXPECTED_FILES.items() if "text" in info["type"]]
        missing_files = set(text_files) - imported_files

        if missing_files:
            results.append(
                STEPBibleValidationResult(
                    check_name="file_completeness",
                    passed=False,
                    message=f"Missing {len(missing_files)} STEPBible text files",
                    details={"missing": sorted(missing_files)},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="file_completeness",
                    passed=True,
                    message=f"All {len(text_files)} STEPBible text files imported",
                )
            )

        return results

    def _validate_word_counts(self, conn: sqlite3.Connection) -> List[STEPBibleValidationResult]:
        """Validate word counts match expectations."""
        results = []
        cursor = conn.cursor()

        # Check Hebrew word count
        cursor.execute("SELECT COUNT(*) FROM stepbible_verses WHERE language = 'hebrew'")
        hebrew_count = cursor.fetchone()[0]

        if hebrew_count < self.MIN_EXPECTED_COUNTS["hebrew_words"]:
            results.append(
                STEPBibleValidationResult(
                    check_name="hebrew_word_count",
                    passed=False,
                    message=(
                        f"Hebrew word count too low: {hebrew_count:,} "
                        f"(expected >= {self.MIN_EXPECTED_COUNTS['hebrew_words']:,})"
                    ),
                    details={"actual": hebrew_count, "expected": self.MIN_EXPECTED_COUNTS["hebrew_words"]},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="hebrew_word_count", passed=True, message=f"Hebrew word count verified: {hebrew_count:,}"
                )
            )

        # Check Greek word count
        cursor.execute("SELECT COUNT(*) FROM stepbible_verses WHERE language = 'greek'")
        greek_count = cursor.fetchone()[0]

        if greek_count < self.MIN_EXPECTED_COUNTS["greek_words"]:
            results.append(
                STEPBibleValidationResult(
                    check_name="greek_word_count",
                    passed=False,
                    message=(
                        f"Greek word count too low: {greek_count:,} "
                        f"(expected >= {self.MIN_EXPECTED_COUNTS['greek_words']:,})"
                    ),
                    details={"actual": greek_count, "expected": self.MIN_EXPECTED_COUNTS["greek_words"]},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="greek_word_count", passed=True, message=f"Greek word count verified: {greek_count:,}"
                )
            )

        # Check book coverage
        for filename, info in self.EXPECTED_FILES.items():
            if "books" in info:
                cursor.execute(
                    """
                    SELECT DISTINCT book
                    FROM stepbible_verses
                    WHERE source_file = ?
                """,
                    (filename,),
                )

                imported_books = set(row[0] for row in cursor.fetchall())
                expected_books = set(info["books"])
                missing_books = expected_books - imported_books

                if missing_books:
                    results.append(
                        STEPBibleValidationResult(
                            check_name=f"book_coverage_{filename}",
                            passed=False,
                            message=f"{filename}: Missing {len(missing_books)} books",
                            details={"missing": sorted(missing_books)},
                        )
                    )

        return results

    def _validate_lexicon_completeness(self, conn: sqlite3.Connection) -> List[STEPBibleValidationResult]:
        """Check lexicon data completeness."""
        results = []
        cursor = conn.cursor()

        # Check Hebrew lexicon
        cursor.execute("SELECT COUNT(*) FROM lexicon WHERE language = 'hebrew'")
        hebrew_lex_count = cursor.fetchone()[0]

        if hebrew_lex_count < self.MIN_EXPECTED_COUNTS["hebrew_lexicon"]:
            results.append(
                STEPBibleValidationResult(
                    check_name="hebrew_lexicon_count",
                    passed=False,
                    message=(
                        f"Hebrew lexicon entries too low: {hebrew_lex_count:,} "
                        f"(expected >= {self.MIN_EXPECTED_COUNTS['hebrew_lexicon']:,})"
                    ),
                    details={"actual": hebrew_lex_count, "expected": self.MIN_EXPECTED_COUNTS["hebrew_lexicon"]},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="hebrew_lexicon_count",
                    passed=True,
                    message=f"Hebrew lexicon verified: {hebrew_lex_count:,} entries",
                )
            )

        # Check Greek lexicon
        cursor.execute("SELECT COUNT(*) FROM lexicon WHERE language = 'greek'")
        greek_lex_count = cursor.fetchone()[0]

        if greek_lex_count < self.MIN_EXPECTED_COUNTS["greek_lexicon"]:
            results.append(
                STEPBibleValidationResult(
                    check_name="greek_lexicon_count",
                    passed=False,
                    message=(
                        f"Greek lexicon entries too low: {greek_lex_count:,} "
                        f"(expected >= {self.MIN_EXPECTED_COUNTS['greek_lexicon']:,})"
                    ),
                    details={"actual": greek_lex_count, "expected": self.MIN_EXPECTED_COUNTS["greek_lexicon"]},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="greek_lexicon_count",
                    passed=True,
                    message=f"Greek lexicon verified: {greek_lex_count:,} entries",
                )
            )

        return results

    def _validate_morphology_completeness(self, conn: sqlite3.Connection) -> List[STEPBibleValidationResult]:
        """Check morphology data completeness."""
        results = []
        cursor = conn.cursor()

        # Check Hebrew morphology
        cursor.execute("SELECT COUNT(*) FROM morphology WHERE language = 'hebrew'")
        hebrew_morph_count = cursor.fetchone()[0]

        if hebrew_morph_count < self.MIN_EXPECTED_COUNTS["hebrew_morphology"]:
            results.append(
                STEPBibleValidationResult(
                    check_name="hebrew_morphology_count",
                    passed=False,
                    message=f"Hebrew morphology codes too low: {hebrew_morph_count:,}",
                    details={"actual": hebrew_morph_count, "expected": self.MIN_EXPECTED_COUNTS["hebrew_morphology"]},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="hebrew_morphology_count",
                    passed=True,
                    message=f"Hebrew morphology verified: {hebrew_morph_count:,} codes",
                )
            )

        # Check Greek morphology
        cursor.execute("SELECT COUNT(*) FROM morphology WHERE language = 'greek'")
        greek_morph_count = cursor.fetchone()[0]

        if greek_morph_count < self.MIN_EXPECTED_COUNTS["greek_morphology"]:
            results.append(
                STEPBibleValidationResult(
                    check_name="greek_morphology_count",
                    passed=False,
                    message=f"Greek morphology codes too low: {greek_morph_count:,}",
                    details={"actual": greek_morph_count, "expected": self.MIN_EXPECTED_COUNTS["greek_morphology"]},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="greek_morphology_count",
                    passed=True,
                    message=f"Greek morphology verified: {greek_morph_count:,} codes",
                )
            )

        return results

    def _validate_data_integrity(self, conn: sqlite3.Connection) -> Tuple[List[STEPBibleValidationResult], List[Dict]]:
        """Validate data integrity using hashes."""
        results: List[STEPBibleValidationResult] = []
        hash_mismatches: List[Dict[str, Any]] = []
        cursor = conn.cursor()

        # Check if we have hash data
        cursor.execute("SELECT COUNT(*) FROM stepbible_verses WHERE data_hash IS NOT NULL")
        hash_count = cursor.fetchone()[0]

        if hash_count == 0:
            results.append(
                STEPBibleValidationResult(
                    check_name="data_integrity_warning",
                    passed=False,
                    message="No hash data found - cannot verify data integrity",
                    details={"hash_count": 0},
                )
            )
            return results, hash_mismatches

        # Validate hashes for a sample of verses
        cursor.execute(
            """
            SELECT source_file, book, chapter, verse, word_number,
                   original_word, strongs_raw, morphology, data_hash
            FROM stepbible_verses
            WHERE data_hash IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 10000
        """
        )

        sample_words = cursor.fetchall()
        corrupt_count = 0

        for row in sample_words:
            source_file, book, chapter, verse, word_num, original, strongs, morph, stored_hash = row

            # Recalculate hash
            hash_data = f"{original or ''}|{strongs or ''}|{morph or ''}"
            calculated_hash = mmh3.hash(hash_data)

            # Check if hashes match
            if stored_hash != calculated_hash:
                corrupt_count += 1
                hash_mismatches.append(
                    {
                        "source_file": source_file,
                        "reference": f"{book} {chapter}:{verse} word {word_num}",
                        "issue": "Hash mismatch - data may be corrupted",
                        "stored": stored_hash,
                        "calculated": calculated_hash,
                    }
                )

        # Also check for missing critical data
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM stepbible_verses
            WHERE (original_word IS NULL OR original_word = '')
            AND (strongs_raw IS NULL OR strongs_raw = '')
            AND (morphology IS NULL OR morphology = '')
        """
        )

        empty_words = cursor.fetchone()[0]

        if corrupt_count > 0:
            results.append(
                STEPBibleValidationResult(
                    check_name="data_integrity",
                    passed=False,
                    message=f"Found {corrupt_count} hash mismatches in {len(sample_words)} sampled words",
                    details={"sample_size": len(sample_words), "corrupt": corrupt_count},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="data_integrity",
                    passed=True,
                    message=f"Data integrity verified for {len(sample_words)} sampled words",
                )
            )

        if empty_words > 100:  # Allow some empty entries
            results.append(
                STEPBibleValidationResult(
                    check_name="data_completeness_warning",
                    passed=False,
                    message=f"Found {empty_words:,} words with no linguistic data",
                    details={"empty_count": empty_words},
                )
            )

        return results, hash_mismatches

    def _validate_cross_references(self, conn: sqlite3.Connection) -> List[STEPBibleValidationResult]:
        """Validate cross-references between tables."""
        results = []
        cursor = conn.cursor()

        # Check Strong's numbers in verses have lexicon entries
        # Note: Strong's numbers should already be cleaned during import
        cursor.execute(
            """
            SELECT COUNT(DISTINCT sv.strongs_primary)
            FROM stepbible_verses sv
            WHERE sv.strongs_primary IS NOT NULL
            AND sv.strongs_primary != ''
            AND NOT EXISTS (
                SELECT 1 FROM lexicon l
                WHERE l.strongs_number = sv.strongs_primary
            )
        """
        )

        missing_lexicon = cursor.fetchone()[0]

        if missing_lexicon > 100:  # Should be much lower now with cleaned Strong's numbers
            results.append(
                STEPBibleValidationResult(
                    check_name="cross_reference_warning",
                    passed=False,
                    message=f"Found {missing_lexicon} Strong's numbers without lexicon entries",
                    details={"missing_count": missing_lexicon},
                )
            )
        else:
            results.append(
                STEPBibleValidationResult(
                    check_name="cross_reference",
                    passed=True,
                    message="Cross-references between verses and lexicon verified",
                )
            )

        return results

    def _validate_data_anomalies(self, conn: sqlite3.Connection) -> List[STEPBibleValidationResult]:
        """Check for data anomalies."""
        results = []
        cursor = conn.cursor()

        # Check for verses with too many words
        cursor.execute(
            """
            SELECT source_file, book, chapter, verse, COUNT(*) as word_count
            FROM stepbible_verses
            GROUP BY source_file, book, chapter, verse
            HAVING COUNT(*) > 100
        """
        )

        anomalies = cursor.fetchall()

        if anomalies:
            results.append(
                STEPBibleValidationResult(
                    check_name="data_anomaly_warning",
                    passed=False,
                    message=f"Found {len(anomalies)} verses with unusually high word counts (>100)",
                    details={"examples": anomalies[:5]},
                )
            )

        # Check for empty original words
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM stepbible_verses
            WHERE original_word IS NULL OR original_word = ''
        """
        )

        empty_words = cursor.fetchone()[0]

        if empty_words > 1000:  # Some empty words are expected
            results.append(
                STEPBibleValidationResult(
                    check_name="empty_words_warning",
                    passed=False,
                    message=f"Found {empty_words:,} words with empty original text",
                    details={"count": empty_words},
                )
            )

        return results

    def print_summary(self, summary: STEPBibleValidationSummary):
        """Print validation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("STEPBIBLE DATA VALIDATION SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nTotal checks: {summary.total_checks}")
        logger.info(f"Passed: {summary.passed_checks}")
        logger.info(f"Failed: {summary.failed_checks}")
        logger.info(f"Success rate: {summary.percentage:.1f}%")

        if summary.failures:
            logger.error(f"\n❌ FAILURES ({len(summary.failures)}):")
            logger.error("-" * 70)
            for failure in summary.failures:
                logger.error(f"  {failure.check_name}: {failure.message}")
                if failure.details:
                    logger.error(f"    Details: {failure.details}")

        if summary.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(summary.warnings)}):")
            logger.warning("-" * 70)
            for warning in summary.warnings:
                logger.warning(f"  {warning.message}")

        if summary.hash_mismatches:
            logger.error(f"\n🔒 HASH MISMATCHES ({len(summary.hash_mismatches)}):")
            logger.error("-" * 70)
            for mismatch in summary.hash_mismatches[:10]:  # Show first 10
                logger.error(f"  {mismatch['reference']} in {mismatch['source_file']}: {mismatch['issue']}")

        logger.info("\n" + "=" * 70)

        if summary.failed_checks > 0:
            logger.error("❌ STEPBIBLE VALIDATION FAILED")
            logger.error("Please review the failures above before continuing.")
        else:
            logger.info("✅ ALL STEPBIBLE VALIDATIONS PASSED")

        logger.info("=" * 70 + "\n")


def validate_stepbible_import(db_path: Path) -> bool:
    """Run STEPBible validation and return success status."""
    validator = STEPBibleValidator(db_path)
    summary = validator.validate_all()
    validator.print_summary(summary)

    # Store validation results
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stepbible_validation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_checks INTEGER,
                    passed_checks INTEGER,
                    failed_checks INTEGER,
                    success_rate REAL,
                    details TEXT
                )
            """
            )

            # Store results
            details = {
                "failures": [
                    {"check": f.check_name, "message": f.message, "details": f.details} for f in summary.failures
                ],
                "warnings": [
                    {"check": w.check_name, "message": w.message, "details": w.details} for w in summary.warnings
                ],
                "hash_mismatches": summary.hash_mismatches[:100],  # Store first 100
            }

            cursor.execute(
                """
                INSERT INTO stepbible_validation
                (total_checks, passed_checks, failed_checks, success_rate, details)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    summary.total_checks,
                    summary.passed_checks,
                    summary.failed_checks,
                    summary.percentage,
                    json.dumps(details),
                ),
            )

            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store validation results: {e}")

    # Return True only if all critical checks passed
    return summary.failed_checks == 0
