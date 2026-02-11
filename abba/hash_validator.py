"""Fast hash-based validation for ABBA data integrity."""

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import mmh3  # MurmurHash3

logger = logging.getLogger(__name__)


class HashValidator:
    """Validates data integrity using fast MurmurHash3 hashing."""

    # Consistent seed for reproducible hashes
    HASH_SEED = 42

    def __init__(self):
        """Initialize hash validator."""
        self._hash_cache: Dict[str, int] = {}

    def hash_verse(self, translation_id: str, book_id: int, chapter: int, verse: int, text: str) -> int:
        """Generate fast hash for a verse.

        Args:
            translation_id: Translation identifier
            book_id: Book number (1-66)
            chapter: Chapter number
            verse: Verse number
            text: Verse text content

        Returns:
            32-bit integer hash
        """
        # Create deterministic key including location and content
        key = f"{translation_id}:{book_id:03d}:{chapter:03d}:{verse:03d}:{text}"
        return mmh3.hash(key, seed=self.HASH_SEED)

    def hash_word(
        self, word: str, strongs: Optional[str] = None, morph: Optional[str] = None, position: Optional[int] = None
    ) -> int:
        """Generate fast hash for a word entry.

        Args:
            word: The word text
            strongs: Strong's number (optional)
            morph: Morphology code (optional)
            position: Word position in verse (optional)

        Returns:
            32-bit integer hash
        """
        # Include all identifying information
        key_parts = [word]
        if strongs:
            key_parts.append(strongs)
        if morph:
            key_parts.append(morph)
        if position is not None:
            key_parts.append(str(position))

        key = "|".join(key_parts)
        return mmh3.hash(key, seed=self.HASH_SEED)

    def hash_embedding_source(self, content: str, metadata: Dict[str, Any]) -> int:
        """Generate hash for embedding source content.

        Args:
            content: The text that was embedded
            metadata: Additional metadata included in embedding

        Returns:
            32-bit integer hash
        """
        # Sort metadata keys for consistency
        meta_str = "|".join(f"{k}:{v}" for k, v in sorted(metadata.items()))
        key = f"{content}|{meta_str}"
        return mmh3.hash(key, seed=self.HASH_SEED)

    def validate_translation_import(
        self, translation_id: str, source_db_path: str, dest_db_path: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate translation import using streaming hash comparison.

        Args:
            translation_id: Translation to validate
            source_db_path: Path to source bible.db
            dest_db_path: Path to destination abba.db

        Returns:
            Tuple of (is_valid, message, details)
        """
        mismatches = []
        missing_verses = []
        verse_count = 0

        try:
            with sqlite3.connect(source_db_path) as source_conn:
                source_conn.row_factory = sqlite3.Row
                source_cursor = source_conn.cursor()

                with sqlite3.connect(dest_db_path) as dest_conn:
                    dest_conn.row_factory = sqlite3.Row
                    dest_cursor = dest_conn.cursor()

                    # Stream verses from source
                    source_cursor.execute(
                        """
                        SELECT book_id, chapter, verse, text 
                        FROM verses 
                        WHERE translation_id = ?
                        ORDER BY book_id, chapter, verse
                    """,
                        (translation_id,),
                    )

                    for row in source_cursor:
                        verse_count += 1

                        # Calculate expected hash
                        expected_hash = self.hash_verse(
                            translation_id, row["book_id"], row["chapter"], row["verse"], row["text"] or ""
                        )

                        # Check destination
                        dest_cursor.execute(
                            """
                            SELECT text, content_hash 
                            FROM verses 
                            WHERE translation_id = ? 
                            AND book_id = ? 
                            AND chapter = ? 
                            AND verse = ?
                        """,
                            (translation_id, row["book_id"], row["chapter"], row["verse"]),
                        )

                        dest_row = dest_cursor.fetchone()

                        if not dest_row:
                            missing_verses.append(f"{row['book_id']}:{row['chapter']}:{row['verse']}")
                        else:
                            # Verify hash matches
                            actual_hash = dest_row["content_hash"]
                            if actual_hash is None:
                                # Calculate hash if not stored
                                actual_hash = self.hash_verse(
                                    translation_id, row["book_id"], row["chapter"], row["verse"], dest_row["text"] or ""
                                )

                            if actual_hash != expected_hash:
                                mismatches.append(
                                    {
                                        "location": f"{row['book_id']}:{row['chapter']}:{row['verse']}",
                                        "expected_hash": expected_hash,
                                        "actual_hash": actual_hash,
                                    }
                                )

                        # Stop after finding significant issues
                        if len(mismatches) + len(missing_verses) > 10:
                            break

            # Prepare results
            if missing_verses or mismatches:
                details = {
                    "verses_checked": verse_count,
                    "missing_count": len(missing_verses),
                    "mismatch_count": len(mismatches),
                    "missing_verses": missing_verses[:5],  # First 5
                    "mismatches": mismatches[:5],  # First 5
                }

                message = f"Validation failed: {len(missing_verses)} missing, {len(mismatches)} mismatched"
                return False, message, details

            # Check total count
            with sqlite3.connect(dest_db_path) as dest_conn:
                cursor = dest_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM verses WHERE translation_id = ?", (translation_id,))
                dest_count = cursor.fetchone()[0]

            if dest_count != verse_count:
                return (
                    False,
                    f"Count mismatch: expected {verse_count}, found {dest_count}",
                    {"expected_count": verse_count, "actual_count": dest_count},
                )

            return (
                True,
                f"Successfully validated {verse_count} verses",
                {"verse_count": verse_count, "translation_id": translation_id},
            )

        except Exception as e:
            return False, f"Validation error: {str(e)}", {"error": str(e)}

    def calculate_aggregate_checksum(self, db_path: str, translation_id: str) -> Tuple[Optional[int], int]:
        """Calculate aggregate checksum for entire translation.

        Uses XOR of all verse hashes for fast validation.

        Args:
            db_path: Path to database
            translation_id: Translation to checksum

        Returns:
            Tuple of (checksum, verse_count)
        """
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Check if content_hash column exists
                cursor.execute("""
                    SELECT COUNT(*) FROM pragma_table_info('verses') 
                    WHERE name='content_hash'
                """)
                has_hash_column = cursor.fetchone()[0] > 0

                if has_hash_column:
                    # Use stored hashes (fastest)
                    cursor.execute(
                        """
                        SELECT 
                            COALESCE(
                                (SELECT 
                                    CAST(SUM(content_hash) AS INTEGER) & 0xFFFFFFFF
                                FROM verses 
                                WHERE translation_id = ?
                                AND content_hash IS NOT NULL),
                                0
                            ) as checksum,
                            COUNT(*) as verse_count
                        FROM verses 
                        WHERE translation_id = ?
                    """,
                        (translation_id, translation_id),
                    )
                else:
                    # Calculate hashes on the fly
                    cursor.execute(
                        """
                        SELECT book_id, chapter, verse, text
                        FROM verses 
                        WHERE translation_id = ?
                        ORDER BY book_id, chapter, verse
                    """,
                        (translation_id,),
                    )

                    checksum = 0
                    verse_count = 0

                    for row in cursor:
                        verse_hash = self.hash_verse(
                            translation_id, row[0], row[1], row[2], row[3] or ""  # book_id  # chapter  # verse  # text
                        )
                        checksum ^= verse_hash  # XOR for aggregation
                        verse_count += 1

                    return checksum & 0xFFFFFFFF, verse_count

                result = cursor.fetchone()
                return result[0] or 0, result[1] or 0

        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return None, 0

    def validate_embeddings(
        self, translation_id: str, db_path: str, chroma_manager
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate embeddings match source verses.

        Args:
            translation_id: Translation to validate
            db_path: Path to database with verses
            chroma_manager: ChromaManager instance

        Returns:
            Tuple of (is_valid, message, details)
        """
        try:
            # Get expected verse hashes from database
            expected_hashes = {}
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Get verse data to calculate hashes
                cursor.execute(
                    """
                    SELECT book_id, chapter, verse, text
                    FROM verses 
                    WHERE translation_id = ?
                """,
                    (translation_id,),
                )

                for row in cursor:
                    verse_id = f"{translation_id}:{row[0]:03d}:{row[1]:03d}:{row[2]:03d}"
                    verse_hash = self.hash_verse(
                        translation_id, row[0], row[1], row[2], row[3] or ""  # book_id  # chapter  # verse  # text
                    )
                    expected_hashes[verse_id] = verse_hash

            # Get embeddings from ChromaDB
            verses_collection = chroma_manager.get_collection("verses")

            # Query embeddings for this translation
            results = verses_collection.get(where={"translation_id": translation_id}, include=["metadatas"])

            if not results["ids"]:
                return (
                    False,
                    f"No embeddings found for {translation_id}",
                    {"expected_count": len(expected_hashes), "actual_count": 0},
                )

            # Validate each embedding
            missing_in_db = []
            hash_mismatches = []

            for i, verse_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]

                if verse_id not in expected_hashes:
                    missing_in_db.append(verse_id)
                    continue

                # Check if source content hash matches
                source_hash = metadata.get("source_hash")
                if source_hash and source_hash != expected_hashes[verse_id]:
                    hash_mismatches.append(
                        {"verse_id": verse_id, "expected": expected_hashes[verse_id], "actual": source_hash}
                    )

            # Check for missing embeddings
            embedded_ids = set(results["ids"])
            missing_embeddings = [vid for vid in expected_hashes.keys() if vid not in embedded_ids]

            # Prepare results
            if missing_in_db or hash_mismatches or missing_embeddings:
                details = {
                    "expected_count": len(expected_hashes),
                    "actual_count": len(results["ids"]),
                    "missing_in_db": len(missing_in_db),
                    "hash_mismatches": len(hash_mismatches),
                    "missing_embeddings": len(missing_embeddings),
                    "sample_missing": missing_embeddings[:5],
                }

                message = (
                    f"Validation failed: {len(missing_embeddings)} missing embeddings, "
                    f"{len(hash_mismatches)} hash mismatches"
                )
                return False, message, details

            return (
                True,
                f"Successfully validated {len(results['ids'])} embeddings",
                {"embedding_count": len(results["ids"]), "translation_id": translation_id},
            )

        except Exception as e:
            return False, f"Validation error: {str(e)}", {"error": str(e)}

    def quick_validate(self, source_db_path: str, dest_db_path: str, translation_id: str) -> Tuple[bool, str]:
        """Quick validation using aggregate checksums.

        Very fast validation that compares aggregate checksums.
        Good for initial checks before detailed validation.

        Args:
            source_db_path: Path to source database
            dest_db_path: Path to destination database
            translation_id: Translation to validate

        Returns:
            Tuple of (is_valid, message)
        """
        source_checksum, source_count = self.calculate_aggregate_checksum(source_db_path, translation_id)

        dest_checksum, dest_count = self.calculate_aggregate_checksum(dest_db_path, translation_id)

        if source_checksum is None or dest_checksum is None:
            return False, "Failed to calculate checksums"

        if source_count != dest_count:
            return False, f"Count mismatch: {source_count} vs {dest_count}"

        if source_checksum != dest_checksum:
            return False, f"Checksum mismatch: {source_checksum} vs {dest_checksum}"

        return True, f"Quick validation passed: {dest_count} verses"
