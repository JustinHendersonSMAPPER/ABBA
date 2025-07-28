"""Parallel import system for ABBA with optimized performance."""

import logging
import multiprocessing as mp
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Iterator
import time

from tqdm import tqdm
import mmh3

from .operation_manager import OperationManager
from .hash_validator import HashValidator

logger = logging.getLogger(__name__)

# Book ID mapping from 3-letter codes to numeric IDs
BOOK_ID_MAP = {
    # Old Testament
    'GEN': 1, 'EXO': 2, 'LEV': 3, 'NUM': 4, 'DEU': 5,
    'JOS': 6, 'JDG': 7, 'RUT': 8, '1SA': 9, '2SA': 10,
    '1KI': 11, '2KI': 12, '1CH': 13, '2CH': 14, 'EZR': 15,
    'NEH': 16, 'EST': 17, 'JOB': 18, 'PSA': 19, 'PRO': 20,
    'ECC': 21, 'SNG': 22, 'ISA': 23, 'JER': 24, 'LAM': 25,
    'EZK': 26, 'DAN': 27, 'HOS': 28, 'JOL': 29, 'AMO': 30,
    'OBA': 31, 'JON': 32, 'MIC': 33, 'NAM': 34, 'HAB': 35,
    'ZEP': 36, 'HAG': 37, 'ZEC': 38, 'MAL': 39,
    # New Testament
    'MAT': 40, 'MRK': 41, 'LUK': 42, 'JHN': 43, 'ACT': 44,
    'ROM': 45, '1CO': 46, '2CO': 47, 'GAL': 48, 'EPH': 49,
    'PHP': 50, 'COL': 51, '1TH': 52, '2TH': 53, '1TI': 54,
    '2TI': 55, 'TIT': 56, 'PHM': 57, 'HEB': 58, 'JAS': 59,
    '1PE': 60, '2PE': 61, '1JN': 62, '2JN': 63, '3JN': 64,
    'JUD': 65, 'REV': 66
}


@dataclass
class ImportJob:
    """Represents a single import job."""
    translation_id: str
    source_db_path: str
    dest_db_path: str
    batch_size: int = 1000
    

@dataclass 
class ImportResult:
    """Results from an import job."""
    translation_id: str
    success: bool
    verse_count: int
    word_count: int
    duration: float
    error: Optional[str] = None


class ParallelImporter:
    """Handles parallel import of Bible translations with optimal performance."""
    
    def __init__(
        self,
        source_db_path: Path,
        dest_db_path: Path,
        operation_manager: Optional[OperationManager] = None,
        max_workers: Optional[int] = None
    ):
        """Initialize parallel importer.
        
        Args:
            source_db_path: Path to source bible.db
            dest_db_path: Path to destination abba.db
            operation_manager: Optional operation manager for state tracking
            max_workers: Maximum parallel workers (defaults to CPU count)
        """
        self.source_db_path = Path(source_db_path)
        self.dest_db_path = Path(dest_db_path)
        self.operation_manager = operation_manager
        self.max_workers = max_workers or mp.cpu_count()
        self.hash_validator = HashValidator()
        
        # Ensure destination database is initialized
        self._ensure_dest_schema()
        
    def _ensure_dest_schema(self):
        """Ensure destination database has proper schema including hash columns."""
        with sqlite3.connect(str(self.dest_db_path)) as conn:
            cursor = conn.cursor()
            
            # Check if content_hash column exists
            cursor.execute("""
                SELECT COUNT(*) FROM pragma_table_info('verses') 
                WHERE name='content_hash'
            """)
            
            if cursor.fetchone()[0] == 0:
                # Add hash column if missing
                cursor.execute("ALTER TABLE verses ADD COLUMN content_hash INTEGER")
                conn.commit()
                logger.info("Added content_hash column to verses table")
    
    def import_translations_parallel(
        self,
        translation_ids: List[str],
        use_processes: bool = True,
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, ImportResult]:
        """Import multiple translations in parallel.
        
        Args:
            translation_ids: List of translation IDs to import
            use_processes: Use processes (True) or threads (False)
            batch_size: Number of verses to process in each batch
            show_progress: Show progress bars
            
        Returns:
            Dictionary mapping translation_id to ImportResult
        """
        results = {}
        
        # Create jobs
        jobs = [
            ImportJob(
                translation_id=tid,
                source_db_path=str(self.source_db_path),
                dest_db_path=str(self.dest_db_path),
                batch_size=batch_size
            )
            for tid in translation_ids
        ]
        
        # Show detailed progress for single translation regardless of worker count
        if len(jobs) == 1:
            # Single translation - show detailed verse-by-verse progress
            job = jobs[0]
            if show_progress:
                # Get total verse count
                with sqlite3.connect(job.source_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM ChapterVerse WHERE translationId = ?",
                        (job.translation_id,)
                    )
                    total_verses = cursor.fetchone()[0]
                
                pbar = tqdm(
                    total=total_verses,
                    desc=f"Importing {job.translation_id}",
                    unit="verses"
                )
                
                def progress_callback(current, total):
                    pbar.n = current
                    pbar.refresh()
                
                result = self._import_single_translation(job, progress_callback)
                pbar.close()
            else:
                result = self._import_single_translation(job)
            
            results[job.translation_id] = result
            return results
        
        # Multiple translations - show overall progress
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._import_single_translation, job): job
                for job in jobs
            }
            
            # Progress tracking
            if show_progress:
                pbar = tqdm(
                    total=len(jobs),
                    desc="Importing translations",
                    unit="translations"
                )
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results[job.translation_id] = result
                    
                    if show_progress:
                        pbar.update(1)
                        if result.success:
                            pbar.set_postfix({
                                'current': job.translation_id,
                                'verses': f"{result.verse_count:,}",
                                'rate': f"{result.verse_count/result.duration:.0f} v/s" if result.duration > 0 else "N/A"
                            })
                    
                except Exception as e:
                    logger.error(f"Failed to import {job.translation_id}: {e}")
                    results[job.translation_id] = ImportResult(
                        translation_id=job.translation_id,
                        success=False,
                        verse_count=0,
                        word_count=0,
                        duration=0,
                        error=str(e)
                    )
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        return results
    
    @staticmethod
    def _import_single_translation(job: ImportJob, progress_callback=None) -> ImportResult:
        """Import a single translation (runs in separate process/thread).
        
        Args:
            job: Import job details
            progress_callback: Optional callback for progress updates
            
        Returns:
            ImportResult with details
        """
        start_time = time.time()
        verse_count = 0
        word_count = 0
        
        try:
            # Each process gets its own connections
            source_conn = sqlite3.connect(job.source_db_path)
            source_conn.row_factory = sqlite3.Row
            dest_conn = sqlite3.connect(job.dest_db_path)
            
            # Enable optimizations
            dest_conn.execute("PRAGMA synchronous = OFF")
            dest_conn.execute("PRAGMA journal_mode = WAL")
            dest_conn.execute("PRAGMA cache_size = 10000")
            dest_conn.execute("PRAGMA temp_store = MEMORY")
            
            try:
                # Start transaction
                dest_conn.execute("BEGIN TRANSACTION")
                
                # Get total verse count for progress tracking
                source_cursor = source_conn.cursor()
                source_cursor.execute(
                    "SELECT COUNT(*) FROM ChapterVerse WHERE translationId = ?",
                    (job.translation_id,)
                )
                total_verses = source_cursor.fetchone()[0]
                
                # Import verses in batches with progress
                verse_count = ParallelImporter._import_verses_batch_with_progress(
                    source_conn, dest_conn, job.translation_id, job.batch_size,
                    total_verses, progress_callback
                )
                
                # Import words in batches
                # Skip word import - words come from STEPBible data, not bible.db
                word_count = 0
                # word_count = ParallelImporter._import_words_batch(
                #     source_conn, dest_conn, job.translation_id, job.batch_size
                # )
                
                # Commit transaction
                dest_conn.commit()
                
                duration = time.time() - start_time
                
                return ImportResult(
                    translation_id=job.translation_id,
                    success=True,
                    verse_count=verse_count,
                    word_count=word_count,
                    duration=duration
                )
                
            finally:
                source_conn.close()
                dest_conn.close()
                
        except Exception as e:
            duration = time.time() - start_time
            return ImportResult(
                translation_id=job.translation_id,
                success=False,
                verse_count=verse_count,
                word_count=word_count,
                duration=duration,
                error=str(e)
            )
    
    @staticmethod
    def _import_verses_batch(
        source_conn: sqlite3.Connection,
        dest_conn: sqlite3.Connection,
        translation_id: str,
        batch_size: int
    ) -> int:
        """Import verses in batches with hash calculation.
        
        Returns:
            Number of verses imported
        """
        source_cursor = source_conn.cursor()
        dest_cursor = dest_conn.cursor()
        
        # Get verses for translation
        source_cursor.execute("""
            SELECT translationId, bookId, chapterNumber, number, text
            FROM ChapterVerse
            WHERE translationId = ?
            ORDER BY bookId, chapterNumber, number
        """, (translation_id,))
        
        verse_count = 0
        batch = []
        hash_validator = HashValidator()
        
        for row in source_cursor:
            # Map book ID from string to integer
            book_str = row[1]  # bookId
            book_id = BOOK_ID_MAP.get(book_str, 0)
            if book_id == 0:
                logger.warning(f"Unknown book ID: {book_str}")
                continue
            
            # Extract values
            translation_id = row[0]  # translationId
            chapter = row[2]  # chapterNumber
            verse = row[3]  # number
            text = row[4] or ''  # text
            
            # Calculate hash
            content_hash = hash_validator.hash_verse(
                translation_id,
                book_id,
                chapter,
                verse,
                text
            )
            
            batch.append((
                translation_id,
                book_id,
                chapter,
                verse,
                text,
                content_hash
            ))
            
            # Insert batch when full
            if len(batch) >= batch_size:
                dest_cursor.executemany("""
                    INSERT OR REPLACE INTO verses 
                    (translation_id, book_id, chapter, verse, text, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, batch)
                verse_count += len(batch)
                batch = []
        
        # Insert remaining batch
        if batch:
            dest_cursor.executemany("""
                INSERT OR REPLACE INTO verses 
                (translation_id, book_id, chapter, verse, text, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
            verse_count += len(batch)
        
        return verse_count
    
    @staticmethod
    def _import_verses_batch_with_progress(
        source_conn: sqlite3.Connection,
        dest_conn: sqlite3.Connection,
        translation_id: str,
        batch_size: int,
        total_verses: int,
        progress_callback=None
    ) -> int:
        """Import verses in batches with hash calculation and progress tracking.
        
        Returns:
            Number of verses imported
        """
        source_cursor = source_conn.cursor()
        dest_cursor = dest_conn.cursor()
        
        # Get verses for translation
        source_cursor.execute("""
            SELECT translationId, bookId, chapterNumber, number, text
            FROM ChapterVerse
            WHERE translationId = ?
            ORDER BY bookId, chapterNumber, number
        """, (translation_id,))
        
        verse_count = 0
        batch = []
        hash_validator = HashValidator()
        
        for row in source_cursor:
            # Map book ID from string to integer
            book_str = row[1]  # bookId
            book_id = BOOK_ID_MAP.get(book_str, 0)
            if book_id == 0:
                logger.warning(f"Unknown book ID: {book_str}")
                continue
            
            # Extract values
            translation_id = row[0]  # translationId
            chapter = row[2]  # chapterNumber
            verse = row[3]  # number
            text = row[4] or ''  # text
            
            # Calculate hash
            content_hash = hash_validator.hash_verse(
                translation_id,
                book_id,
                chapter,
                verse,
                text
            )
            
            batch.append((
                translation_id,
                book_id,
                chapter,
                verse,
                text,
                content_hash
            ))
            
            # Insert batch when full
            if len(batch) >= batch_size:
                dest_cursor.executemany("""
                    INSERT OR REPLACE INTO verses 
                    (translation_id, book_id, chapter, verse, text, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, batch)
                verse_count += len(batch)
                
                # Update progress
                if progress_callback:
                    progress_callback(verse_count, total_verses)
                
                batch = []
        
        # Insert remaining batch
        if batch:
            dest_cursor.executemany("""
                INSERT OR REPLACE INTO verses 
                (translation_id, book_id, chapter, verse, text, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
            verse_count += len(batch)
            
            # Final progress update
            if progress_callback:
                progress_callback(verse_count, total_verses)
        
        return verse_count
    
    @staticmethod
    def _import_words_batch(
        source_conn: sqlite3.Connection,
        dest_conn: sqlite3.Connection,
        translation_id: str,
        batch_size: int
    ) -> int:
        """Import words in batches.
        
        Returns:
            Number of words imported
        """
        source_cursor = source_conn.cursor()
        dest_cursor = dest_conn.cursor()
        
        # First, ensure verses exist and get their IDs
        verse_map = {}
        dest_cursor.execute("""
            SELECT id, book_id, chapter, verse 
            FROM verses 
            WHERE translation_id = ?
        """, (translation_id,))
        
        for row in dest_cursor:
            key = (row[1], row[2], row[3])  # book_id, chapter, verse
            verse_map[key] = row[0]  # id
        
        # Get words for translation
        source_cursor.execute("""
            SELECT w.*, v.book_id, v.chapter, v.verse
            FROM words w
            JOIN verses v ON w.verse_id = v.id
            WHERE v.translation_id = ?
        """, (translation_id,))
        
        word_count = 0
        batch = []
        
        for row in source_cursor:
            # Map to destination verse_id
            verse_key = (row['book_id'], row['chapter'], row['verse'])
            dest_verse_id = verse_map.get(verse_key)
            
            if dest_verse_id:
                batch.append((
                    dest_verse_id,
                    row['word_id'],
                    row['position'],
                    row['word'],
                    row['translation_id']
                ))
                
                # Insert batch when full
                if len(batch) >= batch_size:
                    dest_cursor.executemany("""
                        INSERT OR REPLACE INTO words 
                        (verse_id, word_id, position, word, translation_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, batch)
                    word_count += len(batch)
                    batch = []
        
        # Insert remaining batch
        if batch:
            dest_cursor.executemany("""
                INSERT OR REPLACE INTO words 
                (verse_id, word_id, position, word, translation_id)
                VALUES (?, ?, ?, ?, ?)
            """, batch)
            word_count += len(batch)
        
        return word_count
    
    def import_with_validation(
        self,
        translation_ids: List[str],
        validate_after: bool = True,
        use_processes: bool = False  # Threads often better for I/O
    ) -> Dict[str, Any]:
        """Import translations with state tracking and validation.
        
        Args:
            translation_ids: Translations to import
            validate_after: Run validation after import
            use_processes: Use processes vs threads
            
        Returns:
            Summary of import results
        """
        if not self.operation_manager:
            # Just do parallel import without state tracking
            return self.import_translations_parallel(
                translation_ids, 
                use_processes=use_processes
            )
        
        results = {}
        
        # Start operation
        self.operation_manager.tracker.start_operation("import_translations")
        
        # Import each translation with state tracking
        for tid in translation_ids:
            # Start job
            if self.operation_manager.start_job("import_translations", tid):
                try:
                    # Import this translation
                    import_results = self.import_translations_parallel(
                        [tid],
                        use_processes=use_processes
                    )
                    
                    result = import_results[tid]
                    results[tid] = result
                    
                    if result.success and validate_after:
                        # Validate and complete
                        success = self.operation_manager.complete_job(
                            "import_translations",
                            tid,
                            validation_params={
                                "expected_verses": result.verse_count
                            }
                        )
                        
                        if success:
                            logger.info(f"Successfully imported and validated {tid}")
                        else:
                            logger.error(f"Validation failed for {tid}")
                            result.success = False
                    
                except Exception as e:
                    logger.error(f"Failed to import {tid}: {e}")
                    self.operation_manager.tracker.fail_job(
                        "import_translations", tid, str(e)
                    )
                    results[tid] = ImportResult(
                        translation_id=tid,
                        success=False,
                        verse_count=0,
                        word_count=0,
                        duration=0,
                        error=str(e)
                    )
        
        return results


class OptimizedSQLiteManager:
    """SQLite manager with connection pooling and optimizations."""
    
    def __init__(self, db_path: Path, pool_size: int = 10):
        """Initialize with connection pool.
        
        Args:
            db_path: Path to database
            pool_size: Number of connections in pool
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = Queue(maxsize=pool_size)
        self._lock = Lock()
        
        # Pre-create connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create optimized connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        # Optimizations for bulk operations
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)
    
    def close_all(self):
        """Close all connections in pool."""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()


def benchmark_import_methods(
    source_db: Path,
    dest_db: Path,
    translation_ids: List[str]
) -> Dict[str, Any]:
    """Benchmark different import methods.
    
    Args:
        source_db: Source database path
        dest_db: Destination database path  
        translation_ids: Translations to test with
        
    Returns:
        Benchmark results
    """
    import shutil
    import tempfile
    
    results = {}
    
    # Test different configurations
    configs = [
        ("Sequential", {"use_parallel": False}),
        ("Threads-2", {"use_processes": False, "max_workers": 2}),
        ("Threads-4", {"use_processes": False, "max_workers": 4}),
        ("Threads-CPU", {"use_processes": False, "max_workers": mp.cpu_count()}),
        ("Process-2", {"use_processes": True, "max_workers": 2}),
        ("Process-4", {"use_processes": True, "max_workers": 4}),
        ("Process-CPU", {"use_processes": True, "max_workers": mp.cpu_count()}),
    ]
    
    for name, config in configs:
        # Create temp destination
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            temp_db = Path(tmp.name)
        
        try:
            # Copy schema to temp
            shutil.copy2(dest_db, temp_db)
            
            # Time the import
            start = time.time()
            
            if config.get("use_parallel", True):
                importer = ParallelImporter(
                    source_db, 
                    temp_db,
                    max_workers=config.get("max_workers", mp.cpu_count())
                )
                
                import_results = importer.import_translations_parallel(
                    translation_ids,
                    use_processes=config.get("use_processes", True),
                    show_progress=False
                )
            else:
                # Sequential baseline
                importer = ParallelImporter(source_db, temp_db, max_workers=1)
                import_results = {}
                for tid in translation_ids:
                    job = ImportJob(tid, str(source_db), str(temp_db))
                    import_results[tid] = ParallelImporter._import_single_translation(job)
            
            duration = time.time() - start
            
            # Calculate totals
            total_verses = sum(r.verse_count for r in import_results.values())
            total_words = sum(r.word_count for r in import_results.values())
            
            results[name] = {
                "duration": duration,
                "verses": total_verses,
                "words": total_words,
                "verses_per_second": total_verses / duration if duration > 0 else 0,
                "config": config
            }
            
        finally:
            temp_db.unlink(missing_ok=True)
    
    return results