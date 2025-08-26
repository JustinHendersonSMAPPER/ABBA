"""Parallel STEPBible import for better performance."""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

from tqdm import tqdm

from .hash_validator import HashValidator

logger = logging.getLogger(__name__)


def _clean_strongs_number(strongs: str) -> str:
    """Clean Strong's number to base form.
    
    Examples:
        {H0001G} -> H0001
        {H0001G}/H9020 -> H0001
        G0001G -> G0001
        H1234 -> H1234
        H3899H}\H9016\ \H9018 -> H3899
    """
    if not strongs:
        return strongs
    
    # Remove all backslashes and spaces
    strongs = strongs.replace('\\', '').replace(' ', '')
    
    # Remove curly braces and everything after them
    if '{' in strongs:
        strongs = strongs.split('{')[0]
    if '}' in strongs:
        strongs = strongs.split('}')[0]
        
    # Take first part before any slash
    if '/' in strongs:
        strongs = strongs.split('/')[0]
    
    # Extract just the primary Strong's number (letter + 4 digits)
    import re
    match = re.match(r'([GH]\d{4})', strongs)
    if match:
        return match.group(1)
    
    # If no match, return empty string to avoid bad data
    return ''


@dataclass
class StepBibleJob:
    """Represents a STEPBible parsing job."""
    filename: str
    filepath: str
    file_type: str
    language: str
    dest_db_path: str
    chunk_start: int
    chunk_end: int
    chunk_lines: List[str]


@dataclass
class StepBibleResult:
    """Results from a STEPBible parsing job."""
    filename: str
    chunk_id: int
    success: bool
    words_parsed: int
    duration: float
    error: Optional[str] = None
    word_data: Optional[List[Dict[str, Any]]] = None


class ParallelStepBibleImporter:
    """Handles parallel import of STEPBible text files."""
    
    def __init__(
        self,
        stepbible_dir: Path,
        dest_db_path: Path,
        max_workers: Optional[int] = None
    ):
        """Initialize parallel STEPBible importer.
        
        Args:
            stepbible_dir: Path to STEPBible data directory
            dest_db_path: Path to destination database
            max_workers: Maximum parallel workers (defaults to CPU count)
        """
        self.stepbible_dir = Path(stepbible_dir)
        self.dest_db_path = Path(dest_db_path)
        self.max_workers = max_workers or mp.cpu_count()
    
    def parse_file_parallel(
        self,
        filename: str,
        file_type: str,
        show_progress: bool = True
    ) -> Tuple[bool, int]:
        """Parse a STEPBible file using parallel processing.
        
        Args:
            filename: Name of the file to parse
            file_type: Type of file (tahot, tagnt)
            show_progress: Show progress bar
            
        Returns:
            Tuple of (success, word_count)
        """
        filepath = self.stepbible_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return False, 0
        
        # Determine language
        language = "hebrew" if filename.startswith("tahot") else "greek"
        
        # Read file and split into chunks
        with open(filepath, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        # Filter out header lines and empty lines
        data_lines = []
        for i, line in enumerate(all_lines):
            line = line.strip()
            if (line and 
                not line.startswith(("#", "=", "TAHOT", "TAGNT", "FIELD")) and
                "STEPBible.org" not in line and
                "Data created by" not in line and
                "licence allows" not in line and
                "." in line and "#" in line and "=" in line and "\t" in line):
                data_lines.append((i, line))
        
        if not data_lines:
            logger.warning(f"No data lines found in {filename}")
            return True, 0
        
        # Calculate chunk size for parallel processing
        chunk_size = max(1000, len(data_lines) // (self.max_workers * 4))
        chunks = []
        
        for i in range(0, len(data_lines), chunk_size):
            chunk_data = data_lines[i:i + chunk_size]
            chunks.append({
                'chunk_id': len(chunks),
                'start_line': chunk_data[0][0] if chunk_data else 0,
                'end_line': chunk_data[-1][0] if chunk_data else 0,
                'lines': [line for _, line in chunk_data]
            })
        
        if show_progress:
            from .logging_setup import get_logger
            logger = get_logger(__name__)
            logger.debug(f"  Processing {filename}: {len(data_lines):,} data lines in {len(chunks)} chunks...")
        
        # Process chunks in parallel
        total_words = 0
        results = []
        
        # Use processes for CPU-bound parsing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {}
            for chunk in chunks:
                job = StepBibleJob(
                    filename=filename,
                    filepath=str(filepath),
                    file_type=file_type,
                    language=language,
                    dest_db_path=str(self.dest_db_path),
                    chunk_start=chunk['start_line'],
                    chunk_end=chunk['end_line'],
                    chunk_lines=chunk['lines']
                )
                
                future = executor.submit(self._parse_chunk, job, chunk['chunk_id'])
                future_to_chunk[future] = chunk['chunk_id']
            
            # Process results with progress bar
            if show_progress:
                pbar = tqdm(
                    total=len(chunks),
                    desc=f"    {filename}",
                    unit="chunks"
                )
            
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success and result.word_data:
                        total_words += len(result.word_data)
                        
                        # Batch insert word data
                        self._insert_word_batch(result.word_data)
                    
                    if show_progress:
                        pbar.update(1)
                        pbar.set_postfix({'words': f'{total_words:,}'})
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id}: {e}")
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        # Summary
        success = len([r for r in results if r.success]) == len(chunks)
        
        from .logging_setup import get_logger
        logger = get_logger(__name__)
        
        if total_words > 0:
            logger.debug(f"✓ Parsed {total_words:,} {language} words from {filename}")
        else:
            logger.warning(f"✗ No words parsed from {filename}")
        
        return success, total_words
    
    @staticmethod
    def _parse_chunk(job: StepBibleJob, chunk_id: int) -> StepBibleResult:
        """Parse a chunk of STEPBible data (runs in separate process).
        
        Args:
            job: Job details
            chunk_id: Chunk identifier
            
        Returns:
            StepBibleResult with parsed data
        """
        start_time = time.time()
        words_parsed = 0
        word_data = []
        
        try:
            is_greek = job.language == "greek"
            
            for line in job.chunk_lines:
                try:
                    # Parse data lines that match pattern: Book.Chapter.Verse#WordNum=Source
                    parts = line.split("\t")
                    if len(parts) < 9:  # Need at least 9 parts for essential data
                        continue
                    
                    # Parse reference: Gen.1.1#01=L
                    ref_part = parts[0]
                    if "=" not in ref_part:
                        continue
                    
                    ref_no_source = ref_part.split("=")[0]  # Remove =L part
                    
                    # Extract book.chapter.verse#word
                    if "#" not in ref_no_source:
                        continue
                    
                    verse_ref, word_num_str = ref_no_source.split("#")
                    
                    # Parse book.chapter.verse
                    ref_parts = verse_ref.split(".")
                    if len(ref_parts) < 3:
                        continue
                    
                    book = ref_parts[0]
                    chapter = int(ref_parts[1])
                    verse = int(ref_parts[2])
                    word_num = int(word_num_str)
                    
                    # Extract data based on language
                    if is_greek:
                        # Greek format parsing
                        greek_with_trans = parts[1] if len(parts) > 1 else ""
                        english_gloss = parts[2] if len(parts) > 2 else ""
                        strongs_morph = parts[3] if len(parts) > 3 else ""
                        
                        # Extract Greek text and transliteration
                        if "(" in greek_with_trans and ")" in greek_with_trans:
                            greek_text = greek_with_trans.split("(")[0].strip()
                            transliteration = greek_with_trans.split("(")[1].rstrip(")")
                        else:
                            greek_text = greek_with_trans
                            transliteration = ""
                        
                        # Extract Strong's and morphology
                        if "=" in strongs_morph:
                            strongs_primary = strongs_morph.split("=")[0]
                            morphology_code = strongs_morph.split("=")[1]
                        else:
                            strongs_primary = strongs_morph
                            morphology_code = ""
                        
                        word_info = {
                            "source": job.filename,  # Use the actual filename
                            "book": book,
                            "chapter": chapter,
                            "verse": verse,
                            "word_number": word_num,
                            "original_word": greek_text,
                            "transliteration": transliteration,
                            "english": english_gloss.strip("[]"),
                            "strongs_primary": _clean_strongs_number(strongs_primary),
                            "morphology": morphology_code,
                            "language": "greek"
                        }
                    else:
                        # Hebrew format parsing
                        hebrew_text = parts[1] if len(parts) > 1 else ""
                        transliteration = parts[2] if len(parts) > 2 else ""
                        translation = parts[3] if len(parts) > 3 else ""
                        strongs_raw = parts[4] if len(parts) > 4 else ""
                        morphology = parts[5] if len(parts) > 5 else ""
                        strongs_primary = parts[13] if len(parts) > 13 else strongs_raw
                        
                        # Clean the Strong's number for primary field
                        cleaned_primary = _clean_strongs_number(strongs_primary or strongs_raw)
                        
                        word_info = {
                            "source": job.filename,  # Use the actual filename
                            "book": book,
                            "chapter": chapter,
                            "verse": verse,
                            "word_number": word_num,
                            "original_word": hebrew_text,
                            "transliteration": transliteration,
                            "english": translation,
                            "strongs_raw": strongs_raw,
                            "strongs_primary": cleaned_primary,
                            "morphology": morphology,
                            "language": "hebrew"
                        }
                    
                    word_data.append(word_info)
                    words_parsed += 1
                    
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue
            
            duration = time.time() - start_time
            
            return StepBibleResult(
                filename=job.filename,
                chunk_id=chunk_id,
                success=True,
                words_parsed=words_parsed,
                duration=duration,
                word_data=word_data
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return StepBibleResult(
                filename=job.filename,
                chunk_id=chunk_id,
                success=False,
                words_parsed=0,
                duration=duration,
                error=str(e)
            )
    
    def _insert_word_batch(self, word_data: List[Dict[str, Any]]):
        """Insert a batch of parsed words into the database.
        
        Args:
            word_data: List of word dictionaries to insert
        """
        if not word_data:
            return
        
        import sqlite3
        import mmh3
        
        with sqlite3.connect(self.dest_db_path) as conn:
            cursor = conn.cursor()
            
            # Prepare batch data
            batch = []
            for word in word_data:
                # Calculate hash of the word data for integrity checking
                hash_data = f"{word.get('original_word', '')}|{word.get('strongs_raw', '')}|{word.get('morphology', '')}"
                data_hash = mmh3.hash(hash_data)
                
                batch.append((
                    word.get('source', ''),
                    word.get('book', ''),
                    word.get('chapter', 0),
                    word.get('verse', 0),
                    word.get('word_number', 0),
                    word.get('original_word', ''),
                    word.get('transliteration', ''),
                    word.get('english', ''),
                    word.get('strongs_raw', ''),
                    word.get('strongs_primary', ''),
                    word.get('morphology', ''),
                    word.get('language', ''),
                    data_hash
                ))
            
            # Batch insert
            cursor.executemany("""
                INSERT OR REPLACE INTO stepbible_verses 
                (source_file, book, chapter, verse, word_number, original_word,
                 transliteration, english, strongs_raw, strongs_primary,
                 morphology, language, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            
            conn.commit()