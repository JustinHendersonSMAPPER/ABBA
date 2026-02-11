"""Tests for import tracking functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from abba.database.import_tracker import ImportTracker


class TestImportTracker:
    """Test cases for ImportTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.tracker_file = Path(self.temp_dir) / "test_import_status.json"
        self.tracker = ImportTracker(self.tracker_file)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.tracker_file.exists():
            self.tracker_file.unlink()
        Path(self.temp_dir).rmdir()

    def test_initialization_new_file(self):
        """Test tracker initialization with new file."""
        assert self.tracker.tracker_file == self.tracker_file
        assert self.tracker.status == {"translations": {}, "stepbible_files": {}, "last_update": None}

    def test_initialization_existing_file(self):
        """Test tracker initialization with existing file."""
        # Create existing file
        existing_data = {
            "translations": {"eng_kjv": {"imported": True, "timestamp": "2024-01-01T12:00:00"}},
            "stepbible_files": {"test_file": {"imported": True, "timestamp": "2024-01-01T12:00:00"}},
            "last_update": "2024-01-01T12:00:00",
        }

        with open(self.tracker_file, "w") as f:
            json.dump(existing_data, f)

        tracker = ImportTracker(self.tracker_file)
        assert tracker.status == existing_data

    def test_translation_tracking(self):
        """Test translation import tracking."""
        # Initially not imported
        assert not self.tracker.is_translation_imported("eng_kjv")
        assert self.tracker.get_translation_import_time("eng_kjv") is None

        # Mark as imported
        self.tracker.mark_translation_imported("eng_kjv")

        # Should now be imported
        assert self.tracker.is_translation_imported("eng_kjv")
        import_time = self.tracker.get_translation_import_time("eng_kjv")
        assert import_time is not None
        assert isinstance(import_time, str)

        # File should be saved
        assert self.tracker_file.exists()
        with open(self.tracker_file, "r") as f:
            data = json.load(f)
        assert "eng_kjv" in data["translations"]
        assert data["translations"]["eng_kjv"]["imported"] is True

    def test_stepbible_file_tracking(self):
        """Test STEPBible file tracking."""
        file_type = "tahot"
        file_name = "TAHOT.txt"

        # Initially not imported
        assert not self.tracker.is_stepbible_file_imported(file_type, file_name)

        # Mark as imported
        self.tracker.mark_stepbible_file_imported(file_type, file_name)

        # Should now be imported
        assert self.tracker.is_stepbible_file_imported(file_type, file_name)

        # File should be saved
        assert self.tracker_file.exists()
        with open(self.tracker_file, "r") as f:
            data = json.load(f)
        assert f"{file_type}:{file_name}" in data["stepbible_files"]

    def test_reset_tracking(self):
        """Test resetting tracking data."""
        # Add some data
        self.tracker.mark_translation_imported("eng_kjv")
        self.tracker.mark_stepbible_file_imported("tahot", "TAHOT.txt")

        # Verify data exists
        assert self.tracker.is_translation_imported("eng_kjv")
        assert self.tracker.is_stepbible_file_imported("tahot", "TAHOT.txt")

        # Reset with confirmation
        with patch("builtins.input", return_value="yes"):
            self.tracker.reset(confirm=True)

        # Should be cleared
        assert not self.tracker.is_translation_imported("eng_kjv")
        assert not self.tracker.is_stepbible_file_imported("tahot", "TAHOT.txt")

    def test_reset_tracking_no_confirm(self):
        """Test resetting without confirmation."""
        # Add some data
        self.tracker.mark_translation_imported("eng_kjv")

        # Reset without confirmation - should not clear
        with patch("builtins.input", return_value="no"):
            self.tracker.reset(confirm=True)

        # Should still exist
        assert self.tracker.is_translation_imported("eng_kjv")

    def test_get_import_summary(self):
        """Test import summary generation."""
        # Add some data
        self.tracker.mark_translation_imported("eng_kjv")
        self.tracker.mark_translation_imported("eng_asv")
        self.tracker.mark_stepbible_file_imported("tahot", "TAHOT.txt")

        summary = self.tracker.get_import_summary()

        assert summary["translations_imported"] == 2
        assert summary["stepbible_files"]["tahot"] == 1
        assert summary["last_update"] is not None

    def test_file_corruption_handling(self):
        """Test handling of corrupted tracker file."""
        # Create corrupted JSON file
        with open(self.tracker_file, "w") as f:
            f.write("invalid json content")

        # Should handle gracefully and reset
        tracker = ImportTracker(self.tracker_file)
        assert tracker.status == {"translations": {}, "stepbible_files": {}, "last_update": None}

    def test_save_error_handling(self):
        """Test error handling during save operations."""
        # Make directory read-only to cause save error
        self.tracker_file.parent.chmod(0o444)

        try:
            # This should not raise an exception
            self.tracker.mark_translation_imported("eng_kjv")
            # The save might fail, but the in-memory state should update
            assert self.tracker.is_translation_imported("eng_kjv")
        finally:
            # Restore permissions
            self.tracker_file.parent.chmod(0o755)

    def test_concurrent_access_simulation(self):
        """Test behavior with simulated concurrent access."""
        # Create first tracker
        tracker1 = ImportTracker(self.tracker_file)

        # Mark something as imported
        tracker1.mark_translation_imported("eng_kjv")

        # Create second tracker (simulates another process)
        tracker2 = ImportTracker(self.tracker_file)

        # Both should see the imported translation
        assert tracker1.is_translation_imported("eng_kjv")
        assert tracker2.is_translation_imported("eng_kjv")

        # Make changes in both
        tracker1.mark_translation_imported("eng_asv")
        tracker2.mark_stepbible_file_imported("tahot", "TAHOT.txt")

        # Create third tracker to check final state
        tracker3 = ImportTracker(self.tracker_file)

        # Should have data from the last save
        assert tracker3.is_stepbible_file_imported("tahot", "TAHOT.txt")

    def test_default_tracker_file_location(self):
        """Test default tracker file location."""
        # Create tracker without specifying file
        default_tracker = ImportTracker()

        expected_path = Path("bible_data") / ".import_status.json"
        assert default_tracker.tracker_file == expected_path

    def test_translation_list_methods(self):
        """Test methods that work with translation lists."""
        translations = ["eng_kjv", "eng_asv", "spa_rvr"]

        # Mark some as imported
        self.tracker.mark_translation_imported("eng_kjv")
        self.tracker.mark_translation_imported("spa_rvr")

        # Test filtering
        imported = [t for t in translations if self.tracker.is_translation_imported(t)]
        not_imported = [t for t in translations if not self.tracker.is_translation_imported(t)]

        assert imported == ["eng_kjv", "spa_rvr"]
        assert not_imported == ["eng_asv"]

    def test_stepbible_file_categories(self):
        """Test tracking different categories of STEPBible files."""
        files = [
            ("tahot", "TAHOT1.txt"),
            ("tahot", "TAHOT2.txt"),
            ("tagnt", "TAGNT.txt"),
            ("lexicon", "TBESH.txt"),
            ("morphology", "TEHMC.txt"),
        ]

        # Mark files as imported
        for file_type, file_name in files:
            self.tracker.mark_stepbible_file_imported(file_type, file_name)

        # Check summary by category
        summary = self.tracker.get_import_summary()

        assert summary["stepbible_files"]["tahot"] == 2
        assert summary["stepbible_files"]["tagnt"] == 1
        assert summary["stepbible_files"]["lexicon"] == 1
        assert summary["stepbible_files"]["morphology"] == 1

        # Check individual files
        for file_type, file_name in files:
            assert self.tracker.is_stepbible_file_imported(file_type, file_name)


class TestImportTrackerIntegration:
    """Integration tests for ImportTracker with actual database operations."""

    @pytest.fixture
    def mock_database(self):
        """Mock database for integration testing."""
        db_mock = MagicMock()
        db_mock.get_database_stats.return_value = {"verses": 31102, "words": 8674, "translations": 5}
        return db_mock

    def test_import_workflow_simulation(self, mock_database):
        """Test complete import workflow with tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker_file = Path(temp_dir) / "import_status.json"
            tracker = ImportTracker(tracker_file)

            # Simulate translation import workflow
            translations = ["eng_kjv", "eng_asv", "spa_rvr"]

            # Initial state - nothing imported
            assert all(not tracker.is_translation_imported(t) for t in translations)

            # Import translations one by one
            for translation in translations:
                # Simulate import operation
                # ... database import would happen here ...

                # Mark as imported
                tracker.mark_translation_imported(translation)

                # Verify tracking
                assert tracker.is_translation_imported(translation)

            # Simulate STEPBible import
            stepbible_files = [("tahot", "TAHOT1.txt"), ("tagnt", "TAGNT.txt"), ("lexicon", "TBESH.txt")]

            for file_type, file_name in stepbible_files:
                # Simulate file processing
                # ... file import would happen here ...

                tracker.mark_stepbible_file_imported(file_type, file_name)

            # Verify final state
            summary = tracker.get_import_summary()
            assert summary["translations_imported"] == 3
            assert sum(summary["stepbible_files"].values()) == 3

            # Simulate restart - tracker should remember state
            new_tracker = ImportTracker(tracker_file)
            assert new_tracker.get_import_summary()["translations_imported"] == 3
