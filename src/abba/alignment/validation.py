"""
Validation system for verse alignment accuracy and completeness.

This module provides comprehensive validation tools to ensure verse mappings
are accurate, consistent, and complete across different versification systems.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..verse_id import VerseID, VerseRange, parse_verse_id
from ..versification import VersificationSystem
from .canon_support import Canon, CanonManager
from .unified_reference import UnifiedReferenceSystem
from .verse_mapper import EnhancedVerseMapper, MappingType, VerseMapping


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    error_count: int
    warning_count: int
    errors: List[str]
    warnings: List[str]
    details: Dict[str, any] = None

    def add_error(self, message: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(message)
        self.error_count += 1
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(message)
        self.warning_count += 1


@dataclass
class MappingValidationResult:
    """Result of mapping validation between two systems."""

    system1: VersificationSystem
    system2: VersificationSystem
    total_mappings: int
    round_trip_failures: int
    confidence_warnings: int
    missing_mappings: int
    validation_result: ValidationResult


class AlignmentValidator:
    """Validate verse alignment accuracy and completeness."""

    def __init__(self) -> None:
        """Initialize the alignment validator."""
        self.verse_mapper = EnhancedVerseMapper()
        self.canon_manager = CanonManager()
        self.unified_reference = UnifiedReferenceSystem()

    def validate_round_trip_mapping(
        self,
        system1: VersificationSystem,
        system2: VersificationSystem,
        sample_verses: Optional[List[str]] = None,
    ) -> MappingValidationResult:
        """
        Ensure verse mappings are reversible between two systems.

        Args:
            system1: First versification system
            system2: Second versification system
            sample_verses: Optional list of specific verses to test

        Returns:
            MappingValidationResult with validation details
        """
        result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
        )

        if sample_verses is None:
            # Use a standard set of test verses
            sample_verses = self._get_standard_test_verses()

        round_trip_failures = 0
        confidence_warnings = 0
        missing_mappings = 0

        for verse_str in sample_verses:
            verse_id = parse_verse_id(verse_str)
            if not verse_id:
                result.add_error(f"Invalid verse format: {verse_str}")
                continue

            # Map from system1 to system2
            forward_mapping = self.verse_mapper.map_verse(verse_id, system1, system2)

            if not forward_mapping.target_verses:
                missing_mappings += 1
                result.add_warning(
                    f"No mapping found for {verse_str} from {system1.value} to {system2.value}"
                )
                continue

            # Check confidence level
            if forward_mapping.confidence.value < 0.8:
                confidence_warnings += 1
                result.add_warning(
                    f"Low confidence mapping for {verse_str}: {forward_mapping.confidence.value}"
                )

            # For 1:1 mappings, test round-trip
            if len(forward_mapping.target_verses) == 1:
                target_verse = forward_mapping.target_verses[0]

                # Map back from system2 to system1
                reverse_mapping = self.verse_mapper.map_verse(target_verse, system2, system1)

                if not reverse_mapping.target_verses:
                    round_trip_failures += 1
                    result.add_error(
                        f"Round-trip failure: {verse_str} -> {target_verse} -> no mapping back"
                    )
                elif len(reverse_mapping.target_verses) == 1:
                    back_verse = reverse_mapping.target_verses[0]
                    if str(back_verse) != str(verse_id):
                        round_trip_failures += 1
                        result.add_error(
                            f"Round-trip inconsistency: {verse_str} -> {target_verse} -> {back_verse}"
                        )

        return MappingValidationResult(
            system1=system1,
            system2=system2,
            total_mappings=len(sample_verses),
            round_trip_failures=round_trip_failures,
            confidence_warnings=confidence_warnings,
            missing_mappings=missing_mappings,
            validation_result=result,
        )

    def check_coverage_completeness(
        self,
        canon: Canon,
        versification_system: VersificationSystem,
        provided_verses: List[VerseID],
    ) -> ValidationResult:
        """
        Verify all verses are mapped for a canon and versification system.

        Args:
            canon: Canon to check coverage for
            versification_system: Versification system being validated
            provided_verses: List of verse IDs that have been provided

        Returns:
            ValidationResult with coverage analysis
        """
        result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[], details={}
        )

        # Get expected books for this canon
        canon_books = self.canon_manager.get_canon_books(canon)
        provided_books = list(set(verse.book for verse in provided_verses))

        # Check book coverage
        missing_books = set(canon_books) - set(provided_books)
        extra_books = set(provided_books) - set(canon_books)

        if missing_books:
            for book in missing_books:
                result.add_error(f"Missing book: {book}")

        if extra_books:
            for book in extra_books:
                result.add_warning(f"Extra book not in {canon.value} canon: {book}")

        # Analyze verse distribution by book
        book_verse_counts = {}
        for verse in provided_verses:
            if verse.book not in book_verse_counts:
                book_verse_counts[verse.book] = 0
            book_verse_counts[verse.book] += 1

        # Check for books with very few verses (potential data issues)
        for book in canon_books:
            verse_count = book_verse_counts.get(book, 0)
            if verse_count == 0:
                continue  # Already reported as missing
            elif verse_count < 5:  # Arbitrary threshold for suspicious low counts
                result.add_warning(f"Book {book} has only {verse_count} verses - may be incomplete")

        # Store coverage statistics
        result.details = {
            "canon": canon.value,
            "versification_system": versification_system.value,
            "expected_books": len(canon_books),
            "provided_books": len(provided_books),
            "missing_books": list(missing_books),
            "extra_books": list(extra_books),
            "book_verse_counts": book_verse_counts,
            "coverage_percentage": (
                len(set(canon_books) & set(provided_books)) / len(canon_books) if canon_books else 0
            ),
        }

        return result

    def detect_mapping_conflicts(
        self, systems: Optional[List[VersificationSystem]] = None
    ) -> ValidationResult:
        """
        Find inconsistent mappings across versification systems.

        Args:
            systems: List of systems to check (default: all supported systems)

        Returns:
            ValidationResult with conflict analysis
        """
        result = ValidationResult(
            is_valid=True,
            error_count=0,
            warning_count=0,
            errors=[],
            warnings=[],
            details={"conflicts": []},
        )

        if systems is None:
            systems = list(self.verse_mapper.get_supported_systems())

        conflicts_found = []

        # Check all pairs of systems
        for i, sys1 in enumerate(systems):
            for j, sys2 in enumerate(systems[i + 1 :], i + 1):
                # Check consistency between sys1 and sys2
                consistency_valid = self.verse_mapper.validate_mapping_consistency(sys1, sys2)

                if not consistency_valid:
                    conflict_msg = f"Mapping inconsistency between {sys1.value} and {sys2.value}"
                    result.add_error(conflict_msg)
                    conflicts_found.append({"system1": sys1.value, "system2": sys2.value})

                # Check for problematic mappings
                problematic_mappings = self.verse_mapper.find_conflicting_mappings(sys1, sys2)

                for mapping in problematic_mappings:
                    if mapping.confidence.value < 0.5:
                        warning_msg = f"Low confidence mapping in {sys1.value}->{sys2.value}: {mapping.source_verse}"
                        result.add_warning(warning_msg)

        result.details["conflicts"] = conflicts_found
        return result

    def validate_canonical_id_generation(
        self, test_references: List[str], expected_canonical: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate that canonical ID generation produces expected results.

        Args:
            test_references: List of test verse references in various formats
            expected_canonical: Expected canonical forms (if known)

        Returns:
            ValidationResult with canonical ID validation
        """
        result = ValidationResult(
            is_valid=True,
            error_count=0,
            warning_count=0,
            errors=[],
            warnings=[],
            details={"processed_references": []},
        )

        processed = []

        for i, ref in enumerate(test_references):
            canonical_id = self.unified_reference.generate_canonical_id(ref)

            if canonical_id is None:
                result.add_error(f"Failed to generate canonical ID for: {ref}")
                processed.append({"input": ref, "output": None, "valid": False})
                continue

            canonical_str = str(canonical_id)
            is_valid = self.unified_reference.validate_canonical_id(canonical_id)

            if not is_valid:
                result.add_error(f"Generated invalid canonical ID: {canonical_str} from {ref}")

            # If expected results provided, compare
            if expected_canonical and i < len(expected_canonical):
                expected = expected_canonical[i]
                if canonical_str != expected:
                    result.add_error(
                        f"Canonical ID mismatch for {ref}: expected {expected}, got {canonical_str}"
                    )

            processed.append({"input": ref, "output": canonical_str, "valid": is_valid})

        result.details["processed_references"] = processed
        return result

    def _get_standard_test_verses(self) -> List[str]:
        """Get a standard set of verses for testing mappings."""
        return [
            "GEN.1.1",  # Basic verse
            "PSA.3.1",  # Psalm with title differences
            "PSA.23.1",  # Popular psalm
            "MAL.3.19",  # Malachi chapter boundary issue (Hebrew)
            "MAL.4.1",  # Malachi chapter boundary issue (English)
            "JOL.2.28",  # Joel chapter differences
            "JOL.3.1",  # Joel chapter differences
            "MAT.1.1",  # NT beginning
            "JHN.3.16",  # Famous verse
            "ROM.3.23",  # Key theological verse
            "REV.22.21",  # Bible ending
            "ACT.8.37",  # Textual variant (missing in some)
            "1JN.5.7",  # Comma Johanneum variant
        ]

    def comprehensive_validation(
        self, canon: Canon = Canon.PROTESTANT, systems: Optional[List[VersificationSystem]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Run comprehensive validation across all alignment components.

        Args:
            canon: Canon to use for validation
            systems: Versification systems to validate (default: all supported)

        Returns:
            Dictionary of validation results by component
        """
        if systems is None:
            systems = [VersificationSystem.MODERN, VersificationSystem.KJV, VersificationSystem.MT]

        results = {}

        # Validate mapping conflicts
        results["mapping_conflicts"] = self.detect_mapping_conflicts(systems)

        # Validate canonical ID generation
        test_refs = self._get_standard_test_verses()
        results["canonical_generation"] = self.validate_canonical_id_generation(test_refs)

        # Validate round-trip mappings for all system pairs
        for i, sys1 in enumerate(systems):
            for sys2 in systems[i + 1 :]:
                key = f"round_trip_{sys1.value}_{sys2.value}"
                results[key] = self.validate_round_trip_mapping(sys1, sys2)

        return results

    def generate_validation_report(
        self, validation_results: Dict[str, ValidationResult], output_format: str = "text"
    ) -> str:
        """
        Generate a comprehensive validation report.

        Args:
            validation_results: Results from comprehensive_validation
            output_format: Format for the report ("text" or "json")

        Returns:
            Formatted validation report
        """
        if output_format == "json":
            import json

            report_data = {}
            for key, result in validation_results.items():
                if hasattr(result, "validation_result"):
                    # MappingValidationResult
                    vr = result.validation_result
                    report_data[key] = {
                        "is_valid": vr.is_valid,
                        "error_count": vr.error_count,
                        "warning_count": vr.warning_count,
                        "errors": vr.errors,
                        "warnings": vr.warnings,
                        "system1": result.system1.value,
                        "system2": result.system2.value,
                        "total_mappings": result.total_mappings,
                    }
                else:
                    # ValidationResult
                    report_data[key] = {
                        "is_valid": result.is_valid,
                        "error_count": result.error_count,
                        "warning_count": result.warning_count,
                        "errors": result.errors,
                        "warnings": result.warnings,
                    }
            return json.dumps(report_data, indent=2)

        # Text format
        lines = ["ABBA Verse Alignment Validation Report", "=" * 50, ""]

        # Calculate totals, handling both ValidationResult and MappingValidationResult
        total_errors = 0
        total_warnings = 0
        for r in validation_results.values():
            if hasattr(r, "validation_result"):
                # MappingValidationResult
                total_errors += r.validation_result.error_count
                total_warnings += r.validation_result.warning_count
            elif hasattr(r, "error_count"):
                # ValidationResult
                total_errors += r.error_count
                total_warnings += r.warning_count

        lines.append(f"Overall Status: {'PASS' if total_errors == 0 else 'FAIL'}")
        lines.append(f"Total Errors: {total_errors}")
        lines.append(f"Total Warnings: {total_warnings}")
        lines.append("")

        for component, result in validation_results.items():
            lines.append(f"Component: {component}")

            # Handle both ValidationResult and MappingValidationResult
            if hasattr(result, "validation_result"):
                # MappingValidationResult
                vr = result.validation_result
                lines.append(f"  Status: {'PASS' if vr.is_valid else 'FAIL'}")
                lines.append(f"  Errors: {vr.error_count}")
                lines.append(f"  Warnings: {vr.warning_count}")

                if vr.errors:
                    lines.append("  Error Details:")
                    for error in vr.errors:
                        lines.append(f"    - {error}")

                if vr.warnings:
                    lines.append("  Warnings:")
                    for warning in vr.warnings:
                        lines.append(f"    - {warning}")
            else:
                # ValidationResult
                lines.append(f"  Status: {'PASS' if result.is_valid else 'FAIL'}")
                lines.append(f"  Errors: {result.error_count}")
                lines.append(f"  Warnings: {result.warning_count}")

                if result.errors:
                    lines.append("  Error Details:")
                    for error in result.errors:
                        lines.append(f"    - {error}")

                if result.warnings:
                    lines.append("  Warnings:")
                    for warning in result.warnings:
                        lines.append(f"    - {warning}")

            lines.append("")

        return "\n".join(lines)
