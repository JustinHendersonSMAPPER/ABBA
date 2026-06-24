"""Deployment preparation and configuration validation for ABBA."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _validate_paths(config: Any) -> List[str]:
    """Validate path-related configuration."""
    issues: List[str] = []
    if not config.data_dir.exists():
        issues.append(f"Data directory does not exist: {config.data_dir}")
    if config.database_path and not config.database_path.parent.exists():
        issues.append(f"Database parent directory does not exist: {config.database_path.parent}")
    return issues


def _validate_numeric(config: Any) -> List[str]:
    """Validate numeric configuration values."""
    issues: List[str] = []
    if config.parallel_workers is not None and config.parallel_workers < 1:
        issues.append("parallel_workers must be at least 1")
    if config.connection_pool_size < 1:
        issues.append("connection_pool_size must be at least 1")
    if config.max_results < 1:
        issues.append("max_results must be at least 1")
    if not 0.0 <= config.similarity_threshold <= 1.0:
        issues.append("similarity_threshold must be between 0.0 and 1.0")
    if config.search_cache_size < 0:
        issues.append("search_cache_size cannot be negative")
    if config.ollama_timeout < 1:
        issues.append("ollama_timeout must be at least 1 second")
    if not 0.0 <= config.ollama_consensus_threshold <= 1.0:
        issues.append("ollama_consensus_threshold must be between 0.0 and 1.0")
    if config.memory_limit is not None and config.memory_limit < 64:
        issues.append("memory_limit should be at least 64 MB")
    if not config.embedding_model_english:
        issues.append("embedding_model_english is required")
    return issues


def validate_config(config: Any) -> List[str]:
    """Validate an ABBAConfig for common misconfiguration.

    Args:
        config: ABBAConfig instance to validate.

    Returns:
        List of warning/error messages. Empty means valid.
    """
    issues: List[str] = []
    issues.extend(_validate_paths(config))
    issues.extend(_validate_numeric(config))
    return issues


def backup_database(db_path: Path, backup_dir: Optional[Path] = None) -> Path:
    """Create a backup of the ABBA database.

    Args:
        db_path: Path to the database to back up.
        backup_dir: Directory for backups (default: same directory as db).

    Returns:
        Path to the backup file.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if backup_dir is None:
        backup_dir = db_path.parent

    backup_dir.mkdir(parents=True, exist_ok=True)

    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    backup_path = backup_dir / backup_name

    # Use SQLite's backup API for consistency
    source = sqlite3.connect(db_path)
    dest = sqlite3.connect(backup_path)
    try:
        source.backup(dest)
        logger.info("Database backed up to %s", backup_path)
    finally:
        source.close()
        dest.close()

    return backup_path


def restore_database(backup_path: Path, target_path: Path) -> None:
    """Restore a database from a backup.

    Args:
        backup_path: Path to the backup file.
        target_path: Path where the database should be restored.
    """
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    source = sqlite3.connect(backup_path)
    dest = sqlite3.connect(target_path)
    try:
        source.backup(dest)
        logger.info("Database restored from %s to %s", backup_path, target_path)
    finally:
        source.close()
        dest.close()


def get_database_stats(db_path: Path) -> Dict[str, Any]:
    """Get statistics about the ABBA database.

    Args:
        db_path: Path to the database.

    Returns:
        Dict with table counts and sizes.
    """
    if not db_path.exists():
        return {"error": "Database not found"}

    stats: Dict[str, Any] = {"db_path": str(db_path), "size_mb": round(db_path.stat().st_size / (1024 * 1024), 2)}

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        table_counts: Dict[str, int] = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM [{table}]")  # noqa: S608
                table_counts[table] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                table_counts[table] = -1

        stats["tables"] = table_counts
        stats["total_tables"] = len(tables)

    return stats


def create_install_script(output_dir: Path) -> Path:
    """Generate a shell installation script.

    Args:
        output_dir: Directory to write the script.

    Returns:
        Path to the generated script.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "install.sh"

    script = """#!/bin/bash
# ABBA Installation Script
set -e

echo "=== ABBA Bible Study Tool Installation ==="
echo ""

# Check Python version
python3 --version 2>/dev/null || { echo "Python 3 is required"; exit 1; }

# Check uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install dependencies
echo "Installing dependencies..."
uv sync

# Initialize data directory
echo "Setting up data directory..."
mkdir -p bible_data

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Quick start:"
echo "  uv run python abba/main.py --english-only   # Download English translations"
echo "  uv run python abba/main.py --serve           # Start the API server"
echo "  # Open http://localhost:8000/docs for API documentation"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)
    logger.info("Install script created at %s", script_path)
    return script_path


def prepare_distribution(project_dir: Path) -> Dict[str, Any]:
    """Prepare the project for distribution.

    Validates the project is ready for distribution and
    returns a report of what would be included.

    Args:
        project_dir: Root project directory.

    Returns:
        Distribution readiness report.
    """
    report: Dict[str, Any] = {"ready": True, "issues": [], "included_files": []}

    # Check for required files
    required = ["pyproject.toml", "abba/__init__.py", "abba/main.py"]
    for req in required:
        path = project_dir / req
        if path.exists():
            report["included_files"].append(req)
        else:
            report["issues"].append(f"Missing required file: {req}")
            report["ready"] = False

    # Check for sensitive files that should not be distributed
    sensitive = [".env", "credentials.json", "*.key", "*.pem"]
    for pattern in sensitive:
        matches = list(project_dir.glob(pattern))
        for match in matches:
            report["issues"].append(f"Sensitive file found: {match.name} — exclude from distribution")

    return report
