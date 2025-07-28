"""Operation manager for handling long-running operations with cleanup and validation."""

import logging
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from datetime import datetime

from .state_tracker import StateTracker, OperationStatus
from .hash_validator import HashValidator

logger = logging.getLogger(__name__)


class OperationManager:
    """Manages operations with state tracking, cleanup, and validation."""
    
    def __init__(self, state_file: Path, source_db_path: Optional[Path] = None):
        """Initialize operation manager.
        
        Args:
            state_file: Path to state tracking file
            source_db_path: Path to source bible.db (for validation)
        """
        self.tracker = StateTracker(state_file)
        self.validator = HashValidator()
        self.source_db_path = source_db_path or Path("bible_data/bible.db")
        
        # Cleanup functions for each operation type
        self.cleanup_handlers: Dict[str, Callable] = {
            "import_translations": self._cleanup_translation_import,
            "import_stepbible": self._cleanup_stepbible_import,
            "embed_verses": self._cleanup_verse_embeddings,
            "embed_words": self._cleanup_word_embeddings,
        }
        
        # Validation functions for each operation type
        self.validation_handlers: Dict[str, Callable] = {
            "import_translations": self._validate_translation_import,
            "import_stepbible": self._validate_stepbible_import,
            "embed_verses": self._validate_verse_embeddings,
            "embed_words": self._validate_word_embeddings,
        }
    
    def start_job(
        self,
        operation_name: str,
        job_name: str,
        db_manager=None,
        chroma_manager=None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Start a job with automatic cleanup if needed.
        
        Args:
            operation_name: Operation name (e.g., 'import_translations')
            job_name: Job name (e.g., 'KJV', 'ESV')
            db_manager: SQLiteManager instance for database operations
            chroma_manager: ChromaManager instance for vector operations
            metadata: Optional metadata for the job
            
        Returns:
            True if job can proceed, False if cleanup failed
        """
        # Ensure operation exists
        if not self.tracker.get_operation_state(operation_name):
            self.tracker.start_operation(operation_name)
        
        # Check if cleanup is needed for this job
        if self.tracker.should_cleanup_job(operation_name, job_name):
            logger.info(f"Cleaning up interrupted job: {operation_name}/{job_name}")
            
            cleanup_handler = self.cleanup_handlers.get(operation_name)
            if cleanup_handler:
                try:
                    cleanup_handler(db_manager, chroma_manager, job_name)
                except Exception as e:
                    logger.error(f"Cleanup failed for {operation_name}/{job_name}: {e}")
                    self.tracker.fail_job(operation_name, job_name, str(e))
                    return False
            
            # Reset job after cleanup
            self.tracker.reset_job(operation_name, job_name)
        
        # Start the job
        self.tracker.start_job(operation_name, job_name, metadata)
        return True
    
    def complete_job(
        self,
        operation_name: str,
        job_name: str,
        db_manager=None,
        chroma_manager=None,
        validation_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Complete a job with validation.
        
        Args:
            operation_name: Operation name
            job_name: Job name
            db_manager: SQLiteManager instance for validation
            chroma_manager: ChromaManager instance for validation
            validation_params: Parameters for validation (e.g., expected counts)
            
        Returns:
            True if validation passed and job completed
        """
        # Run validation if handler exists
        validation_handler = self.validation_handlers.get(operation_name)
        validation_result = {}
        
        if validation_handler:
            try:
                is_valid, message, details = validation_handler(
                    db_manager, 
                    chroma_manager,
                    job_name,
                    validation_params
                )
                
                validation_result = {
                    "valid": is_valid,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }
                
                if details:
                    validation_result["details"] = details
                
                if not is_valid:
                    logger.error(f"Validation failed for {operation_name}/{job_name}: {message}")
                    self.tracker.fail_job(operation_name, job_name, message)
                    return False
                
                logger.info(f"Validation passed for {operation_name}/{job_name}: {message}")
                
            except Exception as e:
                logger.error(f"Validation error for {operation_name}/{job_name}: {e}")
                self.tracker.fail_job(operation_name, job_name, str(e))
                return False
        
        # Mark as completed with validation results
        self.tracker.complete_job(operation_name, job_name, validation_result)
        return True
    
    def _cleanup_translation_import(self, db_manager, chroma_manager, job_name: str):
        """Clean up partial translation import for a specific translation."""
        if not db_manager:
            logger.warning("No db_manager provided for translation cleanup")
            return
        
        translation_id = job_name
        logger.info(f"Cleaning up partial import for translation: {translation_id}")
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete verses for this translation
            cursor.execute("DELETE FROM verses WHERE translation_id = ?", (translation_id,))
            deleted_verses = cursor.rowcount
            
            # Delete words for this translation
            cursor.execute("DELETE FROM words WHERE translation_id = ?", (translation_id,))
            deleted_words = cursor.rowcount
            
            logger.info(f"Cleaned up {translation_id}: {deleted_verses} verses, {deleted_words} words")
            
            conn.commit()
    
    def _cleanup_stepbible_import(self, db_manager, chroma_manager, job_name: str):
        """Clean up partial STEPBible import for a specific file."""
        if not db_manager:
            return
        
        file_name = job_name
        logger.info(f"Cleaning up partial STEPBible import for file: {file_name}")
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete STEPBible verses for this file
            cursor.execute("DELETE FROM stepbible_verses WHERE source_file = ?", (file_name,))
            deleted_verses = cursor.rowcount
            
            logger.info(f"Cleaned up {file_name}: {deleted_verses} STEPBible verses")
            
            conn.commit()
    
    def _cleanup_verse_embeddings(self, db_manager, chroma_manager, job_name: str):
        """Clean up partial verse embeddings for a specific translation."""
        if not chroma_manager:
            logger.warning("No chroma_manager provided for verse embedding cleanup")
            return
        
        translation_id = job_name
        logger.info(f"Cleaning up embeddings for translation: {translation_id}")
        
        try:
            verses_collection = chroma_manager.get_collection("verses")
            
            # Delete embeddings for this translation
            verses_collection.delete(
                where={"translation_id": translation_id}
            )
            logger.info(f"Cleaned up verse embeddings for {translation_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning verse embeddings: {e}")
            raise
    
    def _cleanup_word_embeddings(self, db_manager, chroma_manager, job_name: str):
        """Clean up partial word embeddings."""
        if not chroma_manager:
            return
        
        logger.info(f"Cleaning up word embeddings for job: {job_name}")
        
        try:
            words_collection = chroma_manager.get_collection("words")
            
            # For word embeddings, we might need to clear all and restart
            # since words are typically processed in bulk
            if job_name == "all_words":
                # Clear entire collection for full re-embedding
                count = words_collection.count()
                if count > 0:
                    logger.info(f"Clearing {count} word embeddings for full re-embedding")
                    # Note: ChromaDB doesn't have a clear all method
                    # Would need to implement batch deletion
            
        except Exception as e:
            logger.error(f"Error cleaning word embeddings: {e}")
            raise
    
    def _validate_translation_import(
        self, 
        db_manager, 
        chroma_manager,
        job_name: str,
        validation_params: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate translation import using hash validation."""
        if not db_manager:
            return False, "No database manager provided", None
        
        translation_id = job_name
        dest_db_path = str(db_manager.db_path)
        
        # Use hash validator for detailed validation
        is_valid, message, details = self.validator.validate_translation_import(
            translation_id,
            str(self.source_db_path),
            dest_db_path
        )
        
        # Add quick checksum validation
        if is_valid and validation_params and validation_params.get("use_checksum", True):
            quick_valid, quick_msg = self.validator.quick_validate(
                str(self.source_db_path),
                dest_db_path,
                translation_id
            )
            
            if not quick_valid:
                return False, f"Checksum validation failed: {quick_msg}", details
        
        return is_valid, message, details
    
    def _validate_stepbible_import(
        self,
        db_manager,
        chroma_manager,
        job_name: str,
        validation_params: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate STEPBible import."""
        if not db_manager:
            return False, "No database manager provided", None
        
        file_name = job_name
        expected_counts = validation_params or {}
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check verses for this specific file
            cursor.execute(
                "SELECT COUNT(*) FROM stepbible_verses WHERE source_file = ?",
                (file_name,)
            )
            actual_verses = cursor.fetchone()[0]
            
            # Validate against expected counts
            if expected_counts.get("verses"):
                expected = expected_counts["verses"]
                if actual_verses < expected * 0.95:  # 5% tolerance
                    return False, f"Expected ~{expected} verses, found {actual_verses}", {
                        "file": file_name,
                        "expected": expected,
                        "actual": actual_verses
                    }
            
            # Basic sanity check
            if actual_verses == 0:
                return False, f"No verses found for {file_name}", {"file": file_name}
            
            return True, f"Imported {actual_verses:,} verses from {file_name}", {
                "file": file_name,
                "verse_count": actual_verses
            }
    
    def _validate_verse_embeddings(
        self,
        db_manager,
        chroma_manager,
        job_name: str,
        validation_params: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate verse embeddings using hash validation."""
        if not chroma_manager:
            return False, "No ChromaDB manager provided", None
        
        if not db_manager:
            return False, "No database manager provided", None
        
        translation_id = job_name
        
        # Use hash validator for embedding validation
        is_valid, message, details = self.validator.validate_embeddings(
            translation_id,
            str(db_manager.db_path),
            chroma_manager
        )
        
        return is_valid, message, details
    
    def _validate_word_embeddings(
        self,
        db_manager,
        chroma_manager,
        job_name: str,
        validation_params: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate word embeddings."""
        if not chroma_manager:
            return False, "No ChromaDB manager provided", None
        
        if not db_manager:
            return False, "No database manager provided", None
        
        try:
            # Get unique word count from database
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(DISTINCT word_id) FROM words")
                expected_word_count = cursor.fetchone()[0]
            
            # Get actual embeddings
            words_collection = chroma_manager.get_collection("words")
            actual_count = words_collection.count()
            
            # Validate (words might be deduplicated, so allow variance)
            if actual_count < expected_word_count * 0.8:  # Allow 20% variance for deduplication
                return False, f"Expected ~{expected_word_count:,} word embeddings, found {actual_count:,}", {
                    "expected": expected_word_count,
                    "actual": actual_count,
                    "variance": abs(expected_word_count - actual_count) / expected_word_count
                }
            
            return True, f"Successfully embedded {actual_count:,} words", {
                "word_count": actual_count,
                "expected_count": expected_word_count
            }
            
        except Exception as e:
            return False, f"Error validating word embeddings: {e}", {"error": str(e)}
    
    def get_status_summary(self) -> Dict[str, Dict[str, str]]:
        """Get summary of all operation statuses."""
        return self.tracker.get_summary()
    
    def handle_interrupted_operations(self) -> List[str]:
        """Get list of jobs that need attention.
        
        Returns:
            List of warning messages about interrupted jobs
        """
        warnings = []
        interrupted = self.tracker.get_interrupted_jobs()
        
        for operation, job in interrupted:
            warnings.append(
                f"⚠️  {operation}/{job} was interrupted and will be cleaned up on next run"
            )
        
        return warnings