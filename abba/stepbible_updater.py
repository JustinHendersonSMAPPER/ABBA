"""STEPBible data update checker and manager."""

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from .bible_extractor import BibleExtractor

logger = logging.getLogger(__name__)


class STEPBibleUpdater:
    """Manages checking and updating STEPBible data files."""

    def __init__(self, data_dir: Path):
        """Initialize updater with data directory.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.stepbible_dir = self.data_dir / "stepbible"
        self.temp_dir: Optional[Path] = None

    def check_for_updates(self) -> Tuple[bool, Dict[str, bool]]:
        """Check if STEPBible files have been updated.

        Returns:
            Tuple of (has_updates, file_changes_dict)
            where file_changes_dict maps filename to True if changed
        """
        # Get current file hashes
        current_hashes = self._get_file_hashes(self.stepbible_dir)

        if not current_hashes:
            logger.info("No existing STEPBible files found")
            return False, {}

        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)
            temp_stepbible_dir = self.temp_dir / "stepbible"
            temp_stepbible_dir.mkdir()

            # Download files to temp directory
            extractor = BibleExtractor(str(self.temp_dir))

            if not extractor.download_stepbible_data():
                logger.error("Failed to download STEPBible files for comparison")
                return False, {}

            # Get new file hashes
            new_hashes = self._get_file_hashes(temp_stepbible_dir)

            # Compare hashes
            file_changes = {}
            has_updates = False

            logger.info("Comparing %s STEPBible files...", len(current_hashes))

            for filename, current_hash in current_hashes.items():
                new_hash = new_hashes.get(filename)

                if new_hash and new_hash != current_hash:
                    file_changes[filename] = True
                    has_updates = True
                    logger.debug("Update detected for %s", filename)
                else:
                    file_changes[filename] = False

            # Check for new files
            for filename in new_hashes:
                if filename not in current_hashes:
                    file_changes[filename] = True
                    has_updates = True
                    logger.debug("New file detected: %s", filename)

            if has_updates:
                # Keep the temporary files for update
                self._backup_and_update_files(temp_stepbible_dir, file_changes)

            return has_updates, file_changes

    def _get_file_hashes(self, directory: Path) -> Dict[str, str]:
        """Get SHA256 hashes for all files in directory.

        Args:
            directory: Directory to scan

        Returns:
            Dict mapping filename to SHA256 hash
        """
        hashes: Dict[str, str] = {}

        if not directory.exists():
            return hashes

        for file_path in directory.glob("*.txt"):
            if file_path.name == "ATTRIBUTION.txt":
                continue  # Skip attribution file

            try:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        sha256_hash.update(chunk)
                hashes[file_path.name] = sha256_hash.hexdigest()
            except Exception as e:
                logger.error("Error hashing %s: %s", file_path, e)

        return hashes

    def _backup_and_update_files(self, temp_stepbible_dir: Path, file_changes: Dict[str, bool]):
        """Backup current files and update with new ones.

        Args:
            temp_stepbible_dir: Temporary directory with new files
            file_changes: Dict of files that have changed
        """
        # Create backup directory
        backup_dir = self.stepbible_dir.parent / "stepbible_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir()

        try:
            # Backup current files
            logger.info("Backing up current STEPBible files to %s", backup_dir)
            for file_path in self.stepbible_dir.glob("*.txt"):
                shutil.copy2(file_path, backup_dir)

            # Update changed files
            updated_count = 0
            for filename, changed in file_changes.items():
                if changed:
                    src = temp_stepbible_dir / filename
                    dst = self.stepbible_dir / filename

                    if src.exists():
                        logger.info("Updating %s", filename)
                        shutil.copy2(src, dst)
                        updated_count += 1

            logger.info("Successfully updated %s files", updated_count)

            # Update the tracking to force re-import
            self._mark_stepbible_for_reimport()

        except Exception as e:
            logger.error("Error updating files: %s", e)
            # Restore from backup
            logger.info("Restoring from backup...")
            for file_path in backup_dir.glob("*.txt"):
                shutil.copy2(file_path, self.stepbible_dir)
            raise
        finally:
            # Clean up backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

    def _mark_stepbible_for_reimport(self):
        """Mark STEPBible data as needing re-import."""
        # Remove the import tracking for STEPBible
        import_status_file = self.data_dir / ".import_status.json"

        if import_status_file.exists():
            import json

            try:
                with open(import_status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Remove STEPBible import tracking
                if "stepbible_files" in data:
                    data["stepbible_files"] = {}
                    logger.info("Cleared STEPBible import tracking")

                with open(import_status_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error("Error updating import status: %s", e)

    def get_update_summary(self, file_changes: Dict[str, bool]) -> str:
        """Get a human-readable summary of updates.

        Args:
            file_changes: Dict of files that have changed

        Returns:
            Summary string
        """
        changed_files = [f for f, changed in file_changes.items() if changed]

        if not changed_files:
            return "No STEPBible updates found - all files are up to date"

        summary = f"\n{len(changed_files)} STEPBible file(s) have been updated since last run:\n"
        for filename in sorted(changed_files):
            summary += f"  ✓ {filename}\n"

        return summary
