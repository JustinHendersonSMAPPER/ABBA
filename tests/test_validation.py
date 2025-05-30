"""Unit tests for validation module."""

from typing import List

import pytest

from abba.alignment.canon_support import Canon
from abba.alignment.validation import (
    AlignmentValidator,
    MappingValidationResult,
    ValidationResult,
)
from abba.verse_id import VerseID, parse_verse_id
from abba.versification import VersificationSystem


class TestValidationResult:
    """Test the ValidationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating ValidationResult instances."""
        result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
        )

        assert result.is_valid
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.errors == []
        assert result.warnings == []

    def test_add_error(self) -> None:
        """Test adding errors to validation result."""
        result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
        )

        result.add_error("Test error")

        assert not result.is_valid
        assert result.error_count == 1
        assert "Test error" in result.errors

    def test_add_warning(self) -> None:
        """Test adding warnings to validation result."""
        result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
        )

        result.add_warning("Test warning")

        assert result.is_valid  # Warnings don't invalidate result
        assert result.warning_count == 1
        assert "Test warning" in result.warnings

    def test_multiple_errors_and_warnings(self) -> None:
        """Test adding multiple errors and warnings."""
        result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
        )

        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        assert not result.is_valid
        assert result.error_count == 2
        assert result.warning_count == 2
        assert len(result.errors) == 2
        assert len(result.warnings) == 2


class TestMappingValidationResult:
    """Test the MappingValidationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating MappingValidationResult instances."""
        validation_result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
        )

        mapping_result = MappingValidationResult(
            system1=VersificationSystem.MODERN,
            system2=VersificationSystem.KJV,
            total_mappings=100,
            round_trip_failures=0,
            confidence_warnings=5,
            missing_mappings=2,
            validation_result=validation_result,
        )

        assert mapping_result.system1 == VersificationSystem.MODERN
        assert mapping_result.system2 == VersificationSystem.KJV
        assert mapping_result.total_mappings == 100
        assert mapping_result.round_trip_failures == 0
        assert mapping_result.confidence_warnings == 5
        assert mapping_result.missing_mappings == 2
        assert mapping_result.validation_result == validation_result


class TestAlignmentValidator:
    """Test the AlignmentValidator class."""

    def test_initialization(self) -> None:
        """Test validator initialization."""
        validator = AlignmentValidator()

        assert validator.verse_mapper is not None
        assert validator.canon_manager is not None
        assert validator.unified_reference is not None

    def test_validate_round_trip_mapping_same_system(self) -> None:
        """Test round-trip validation for the same system."""
        validator = AlignmentValidator()

        result = validator.validate_round_trip_mapping(
            VersificationSystem.MODERN,
            VersificationSystem.MODERN,
            ["GEN.1.1", "MAT.1.1", "REV.22.21"],
        )

        assert result.system1 == VersificationSystem.MODERN
        assert result.system2 == VersificationSystem.MODERN
        assert result.total_mappings == 3
        assert result.round_trip_failures == 0  # Same system should always work

    def test_validate_round_trip_mapping_different_systems(self) -> None:
        """Test round-trip validation between different systems."""
        validator = AlignmentValidator()

        result = validator.validate_round_trip_mapping(
            VersificationSystem.MT, VersificationSystem.MODERN, ["GEN.1.1", "PSA.23.1"]
        )

        assert result.system1 == VersificationSystem.MT
        assert result.system2 == VersificationSystem.MODERN
        assert result.total_mappings == 2
        # May have some failures due to versification differences

    def test_validate_round_trip_mapping_with_textual_variants(self) -> None:
        """Test round-trip validation with textual variants."""
        validator = AlignmentValidator()

        result = validator.validate_round_trip_mapping(
            VersificationSystem.KJV,
            VersificationSystem.MODERN,
            ["ACT.8.37", "1JN.5.7"],  # Known textual variants
        )

        assert result.total_mappings == 2
        # Should detect missing mappings for textual variants
        assert result.missing_mappings > 0 or result.round_trip_failures > 0

    def test_validate_round_trip_mapping_with_invalid_verse(self) -> None:
        """Test round-trip validation with invalid verse."""
        validator = AlignmentValidator()

        result = validator.validate_round_trip_mapping(
            VersificationSystem.MODERN,
            VersificationSystem.MODERN,
            ["GEN.1.1", "INVALID.VERSE", "MAT.1.1"],
        )

        assert result.validation_result.error_count > 0
        assert any("Invalid verse format" in error for error in result.validation_result.errors)

    def test_validate_round_trip_mapping_default_test_verses(self) -> None:
        """Test round-trip validation with default test verses."""
        validator = AlignmentValidator()

        result = validator.validate_round_trip_mapping(
            VersificationSystem.MODERN, VersificationSystem.MODERN
        )

        # Should use standard test verses
        assert result.total_mappings > 0
        assert result.round_trip_failures == 0  # Same system

    def test_check_coverage_completeness_full_coverage(self) -> None:
        """Test coverage validation with complete coverage."""
        validator = AlignmentValidator()

        # Create a list of all Protestant canon verses (simplified)
        protestant_books = validator.canon_manager.get_canon_books(Canon.PROTESTANT)
        test_verses = [parse_verse_id(f"{book}.1.1") for book in protestant_books]

        result = validator.check_coverage_completeness(
            Canon.PROTESTANT, VersificationSystem.MODERN, test_verses
        )

        assert result.is_valid
        assert result.error_count == 0
        assert result.details["coverage_percentage"] == 1.0
        assert result.details["missing_books"] == []

    def test_check_coverage_completeness_missing_books(self) -> None:
        """Test coverage validation with missing books."""
        validator = AlignmentValidator()

        # Provide only a few verses
        test_verses = [
            parse_verse_id("GEN.1.1"),
            parse_verse_id("MAT.1.1"),
            parse_verse_id("REV.22.21"),
        ]

        result = validator.check_coverage_completeness(
            Canon.PROTESTANT, VersificationSystem.MODERN, test_verses
        )

        assert not result.is_valid
        assert result.error_count > 0
        assert len(result.details["missing_books"]) > 0
        assert result.details["coverage_percentage"] < 1.0

    def test_check_coverage_completeness_extra_books(self) -> None:
        """Test coverage validation with extra books."""
        validator = AlignmentValidator()

        # Include both Protestant books and one with invalid ID
        test_verses = [parse_verse_id("GEN.1.1"), parse_verse_id("MAT.1.1")]
        # Add a non-existent verse that should trigger extra book warning
        from abba.verse_id import VerseID

        test_verses.append(VerseID(book="XXX", chapter=1, verse=1))  # Non-existent book

        result = validator.check_coverage_completeness(
            Canon.PROTESTANT, VersificationSystem.MODERN, test_verses
        )

        # Should have warnings about extra books
        assert result.warning_count > 0
        assert "XXX" in result.details["extra_books"]

    def test_check_coverage_completeness_low_verse_counts(self) -> None:
        """Test coverage validation with suspiciously low verse counts."""
        validator = AlignmentValidator()

        # Provide very few verses per book
        test_verses = [
            parse_verse_id("GEN.1.1"),
            parse_verse_id("EXO.1.1"),
            parse_verse_id("MAT.1.1"),
        ]

        result = validator.check_coverage_completeness(
            Canon.PROTESTANT, VersificationSystem.MODERN, test_verses
        )

        # Should have warnings about books with few verses
        assert result.warning_count > 0

    def test_detect_mapping_conflicts_no_conflicts(self) -> None:
        """Test conflict detection when no conflicts exist."""
        validator = AlignmentValidator()

        result = validator.detect_mapping_conflicts([VersificationSystem.MODERN])

        # Single system should have no conflicts
        assert result.is_valid or result.error_count == 0

    def test_detect_mapping_conflicts_with_systems(self) -> None:
        """Test conflict detection across multiple systems."""
        validator = AlignmentValidator()

        systems = [VersificationSystem.MODERN, VersificationSystem.MT, VersificationSystem.KJV]
        result = validator.detect_mapping_conflicts(systems)

        assert isinstance(result, ValidationResult)
        # May find conflicts between systems

    def test_detect_mapping_conflicts_default_systems(self) -> None:
        """Test conflict detection with default systems."""
        validator = AlignmentValidator()

        result = validator.detect_mapping_conflicts()

        assert isinstance(result, ValidationResult)
        assert "conflicts" in result.details

    def test_validate_canonical_id_generation_valid_references(self) -> None:
        """Test canonical ID generation validation with valid references."""
        validator = AlignmentValidator()

        test_refs = ["Genesis 1:1", "Matthew 1:1", "Revelation 22:21"]
        result = validator.validate_canonical_id_generation(test_refs)

        assert result.is_valid or result.error_count == 0
        assert len(result.details["processed_references"]) == 3

    def test_validate_canonical_id_generation_invalid_references(self) -> None:
        """Test canonical ID generation validation with invalid references."""
        validator = AlignmentValidator()

        test_refs = ["Genesis 1:1", "INVALID REFERENCE", "Matthew 1:1"]
        result = validator.validate_canonical_id_generation(test_refs)

        assert result.error_count > 0
        assert len(result.details["processed_references"]) == 3

        # Check that invalid reference was marked as failed
        processed = result.details["processed_references"]
        invalid_ref = next(ref for ref in processed if ref["input"] == "INVALID REFERENCE")
        assert not invalid_ref["valid"]

    def test_validate_canonical_id_generation_with_expected_results(self) -> None:
        """Test canonical ID generation with expected results."""
        validator = AlignmentValidator()

        test_refs = ["Genesis 1:1", "Matthew 1:1"]
        expected = ["GEN.1.1", "MAT.1.1"]

        result = validator.validate_canonical_id_generation(test_refs, expected)

        # Check that expected results are compared
        processed = result.details["processed_references"]
        assert len(processed) == 2

    def test_comprehensive_validation_default_parameters(self) -> None:
        """Test comprehensive validation with default parameters."""
        validator = AlignmentValidator()

        results = validator.comprehensive_validation()

        assert isinstance(results, dict)
        assert "mapping_conflicts" in results
        assert "canonical_generation" in results
        # Should have round-trip results for system pairs

    def test_comprehensive_validation_custom_parameters(self) -> None:
        """Test comprehensive validation with custom parameters."""
        validator = AlignmentValidator()

        systems = [VersificationSystem.MODERN, VersificationSystem.KJV]
        results = validator.comprehensive_validation(Canon.CATHOLIC, systems)

        assert isinstance(results, dict)
        # Should include round-trip validation for specified systems
        round_trip_keys = [key for key in results.keys() if key.startswith("round_trip")]
        assert len(round_trip_keys) > 0

    def test_generate_validation_report_text_format(self) -> None:
        """Test generating validation report in text format."""
        validator = AlignmentValidator()

        # Create sample validation results
        sample_result = ValidationResult(
            is_valid=True, error_count=0, warning_count=1, errors=[], warnings=["Sample warning"]
        )

        results = {"test_component": sample_result}
        report = validator.generate_validation_report(results, "text")

        assert isinstance(report, str)
        assert "ABBA Verse Alignment Validation Report" in report
        assert "test_component" in report
        assert "Sample warning" in report

    def test_generate_validation_report_json_format(self) -> None:
        """Test generating validation report in JSON format."""
        validator = AlignmentValidator()

        # Create sample validation results
        sample_result = ValidationResult(
            is_valid=False, error_count=1, warning_count=0, errors=["Sample error"], warnings=[]
        )

        results = {"test_component": sample_result}
        report = validator.generate_validation_report(results, "json")

        assert isinstance(report, str)
        # Should be valid JSON
        import json

        parsed = json.loads(report)
        assert "test_component" in parsed
        assert parsed["test_component"]["is_valid"] is False
        assert "Sample error" in parsed["test_component"]["errors"]

    def test_standard_test_verses(self) -> None:
        """Test that standard test verses are reasonable."""
        validator = AlignmentValidator()

        test_verses = validator._get_standard_test_verses()

        assert isinstance(test_verses, list)
        assert len(test_verses) > 0

        # Should include some known challenging verses
        assert "PSA.3.1" in test_verses  # Psalm title differences
        assert "MAL.3.19" in test_verses or "MAL.4.1" in test_verses  # Chapter boundary
        assert "ACT.8.37" in test_verses  # Textual variant

        # All should be valid verse IDs
        for verse_str in test_verses:
            verse_id = parse_verse_id(verse_str)
            assert verse_id is not None

    def test_validation_with_empty_input(self) -> None:
        """Test validation with empty input lists."""
        validator = AlignmentValidator()

        # Empty verse list for round-trip validation
        result = validator.validate_round_trip_mapping(
            VersificationSystem.MODERN, VersificationSystem.MODERN, []
        )

        assert result.total_mappings == 0
        assert result.round_trip_failures == 0

    def test_validation_result_aggregation(self) -> None:
        """Test that validation results properly aggregate statistics."""
        validator = AlignmentValidator()

        results = validator.comprehensive_validation()

        # Calculate total errors and warnings
        total_errors = 0
        total_warnings = 0
        for r in results.values():
            if hasattr(r, "validation_result"):
                # MappingValidationResult
                total_errors += r.validation_result.error_count
                total_warnings += r.validation_result.warning_count
            else:
                # ValidationResult
                total_errors += r.error_count
                total_warnings += r.warning_count

        assert total_errors >= 0
        assert total_warnings >= 0

        # Generate report and check that totals are included
        report = validator.generate_validation_report(results, "text")
        assert f"Total Errors: {total_errors}" in report
        assert f"Total Warnings: {total_warnings}" in report

    def test_generate_validation_report_with_mapping_results(self) -> None:
        """Test generating validation report with MappingValidationResult."""
        validator = AlignmentValidator()

        # Create a MappingValidationResult
        vr = ValidationResult(
            is_valid=False,
            error_count=1,
            warning_count=2,
            errors=["Test error"],
            warnings=["Warning 1", "Warning 2"],
        )
        mapping_result = MappingValidationResult(
            system1=VersificationSystem.MT,
            system2=VersificationSystem.MODERN,
            total_mappings=100,
            round_trip_failures=1,
            confidence_warnings=2,
            missing_mappings=0,
            validation_result=vr,
        )

        results = {"mapping_test": mapping_result}
        report = validator.generate_validation_report(results, "text")

        assert "mapping_test" in report
        assert "Test error" in report
        assert "Warning 1" in report
        assert "Warning 2" in report

    def test_validation_edge_cases(self) -> None:
        """Test validation edge cases for better coverage."""
        validator = AlignmentValidator()

        # Test with no errors or warnings
        clean_result = ValidationResult(
            is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
        )

        # Test report generation with only clean results
        results = {"clean": clean_result}
        report = validator.generate_validation_report(results, "text")
        assert "PASS" in report
        assert "Total Errors: 0" in report

        # Test error result edge case
        error_result = ValidationResult(
            is_valid=False,
            error_count=2,
            warning_count=0,
            errors=["Error 1", "Error 2"],
            warnings=[],
        )

        # Generate JSON report to cover that branch
        json_report = validator.generate_validation_report({"errors": error_result}, "json")
        assert isinstance(json_report, str)

        # Parse JSON to verify structure
        import json

        parsed = json.loads(json_report)
        assert "errors" in parsed
        assert parsed["errors"]["error_count"] == 2
        assert parsed["errors"]["is_valid"] is False

        # Test JSON generation for MappingValidationResult to increase coverage
        mvr = MappingValidationResult(
            system1=VersificationSystem.MT,
            system2=VersificationSystem.MODERN,
            total_mappings=10,
            round_trip_failures=0,
            confidence_warnings=0,
            missing_mappings=0,
            validation_result=ValidationResult(
                is_valid=True, error_count=0, warning_count=0, errors=[], warnings=[]
            ),
        )

        # This should handle the MappingValidationResult in JSON generation
        json_report2 = validator.generate_validation_report({"mapping": mvr}, "json")
        parsed2 = json.loads(json_report2)
        # MappingValidationResult should be converted to its validation_result in JSON
        assert "mapping" in parsed2
        assert parsed2["mapping"]["is_valid"] is True
        assert parsed2["mapping"]["system1"] == "MT"
        assert parsed2["mapping"]["system2"] == "Modern"
        assert parsed2["mapping"]["total_mappings"] == 10
