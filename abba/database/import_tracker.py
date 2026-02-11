"""Track import progress to avoid re-processing data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ImportTracker:
    """Track which data imports have been completed."""

    def __init__(self, tracker_file: Optional[Path] = None):
        """Initialize import tracker.

        Args:
            tracker_file: Path to tracking file (defaults to bible_data/.import_status.json)
        """
        if tracker_file is None:
            tracker_file = Path("bible_data") / ".import_status.json"
        self.tracker_file = tracker_file
        self.status = self._load_status()

    def _load_status(self) -> Dict[str, Any]:
        """Load status from file or create new."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start fresh
                return self._create_empty_status()
        return self._create_empty_status()

    def _create_empty_status(self) -> Dict[str, Any]:
        """Create empty status structure."""
        return {
            "schema_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "imports": {
                "translations": {},  # translation_id: timestamp
                "stepbible": {  # file_type: timestamp
                    "tahot": {},  # filename: timestamp
                    "tagnt": {},
                    "lexicon": {},
                    "morphology": {},
                },
            },
            "metadata": {"last_update": None, "total_imports": 0},
        }

    def _save_status(self):
        """Save current status to file."""
        # Ensure directory exists
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)

        # Update metadata
        self.status["metadata"]["last_update"] = datetime.now().isoformat()

        # Write with pretty formatting
        with open(self.tracker_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def is_translation_imported(self, translation_id: str) -> bool:
        """Check if a translation has been imported.

        Args:
            translation_id: Translation identifier (e.g., 'KJV')

        Returns:
            True if already imported
        """
        return translation_id in self.status["imports"]["translations"]

    def mark_translation_imported(self, translation_id: str):
        """Mark a translation as imported.

        Args:
            translation_id: Translation identifier
        """
        self.status["imports"]["translations"][translation_id] = datetime.now().isoformat()
        self.status["metadata"]["total_imports"] += 1
        self._save_status()

    def is_stepbible_file_imported(self, file_type: str, filename: str) -> bool:
        """Check if a STEPBible file has been imported.

        Args:
            file_type: Type of file ('tahot', 'tagnt', 'lexicon', 'morphology')
            filename: Name of the file

        Returns:
            True if already imported
        """
        if file_type not in self.status["imports"]["stepbible"]:
            return False
        return filename in self.status["imports"]["stepbible"][file_type]

    def mark_stepbible_file_imported(self, file_type: str, filename: str):
        """Mark a STEPBible file as imported.

        Args:
            file_type: Type of file
            filename: Name of the file
        """
        if file_type not in self.status["imports"]["stepbible"]:
            self.status["imports"]["stepbible"][file_type] = {}

        self.status["imports"]["stepbible"][file_type][filename] = datetime.now().isoformat()
        self.status["metadata"]["total_imports"] += 1
        self._save_status()

    def get_import_summary(self) -> Dict[str, Any]:
        """Get summary of import status.

        Returns:
            Dictionary with import statistics
        """
        translations = self.status["imports"]["translations"]
        stepbible = self.status["imports"]["stepbible"]

        return {
            "translations_imported": len(translations),
            "translation_list": list(translations.keys()),
            "stepbible_files": {file_type: len(files) for file_type, files in stepbible.items()},
            "last_update": self.status["metadata"]["last_update"],
            "total_imports": self.status["metadata"]["total_imports"],
        }

    def reset(self, confirm: bool = False):
        """Reset all import tracking.

        Args:
            confirm: Must be True to actually reset
        """
        if confirm:
            self.status = self._create_empty_status()
            self._save_status()

    def remove_translation(self, translation_id: str):
        """Remove a translation from imported status.

        Args:
            translation_id: Translation to remove
        """
        if translation_id in self.status["imports"]["translations"]:
            del self.status["imports"]["translations"][translation_id]
            self._save_status()

    def get_translation_import_time(self, translation_id: str) -> Optional[str]:
        """Get when a translation was imported.

        Args:
            translation_id: Translation identifier

        Returns:
            ISO timestamp or None if not imported
        """
        return self.status["imports"]["translations"].get(translation_id)

    def clear_stepbible_tracking(self):
        """Clear all STEPBible import tracking records."""
        # Clear all STEPBible file tracking
        for file_type in self.status["imports"]["stepbible"]:
            self.status["imports"]["stepbible"][file_type] = {}

        # Also clear the overall complete marker
        if "complete" in self.status["imports"]["stepbible"]:
            del self.status["imports"]["stepbible"]["complete"]

        self._save_status()
