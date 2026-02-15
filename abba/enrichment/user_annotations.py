"""User annotation features: notes, collections, and sharing.

Provides note-taking, verse saving/bookmarking, and shareable
collections for personal Bible study.
"""

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UserAnnotationManager:
    """Manages user notes, collections, and sharing."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    # --- Notes ---

    def create_note(
        self,
        book_id: int,
        chapter: int,
        verse: int,
        content: str,
        note_type: str = "personal",
    ) -> int:
        """Create a note attached to a verse."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO verse_notes (book_id, chapter, verse, content, note_type) VALUES (?, ?, ?, ?, ?)",
                (book_id, chapter, verse, content, note_type),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_notes_for_verse(self, book_id: int, chapter: int, verse: int) -> List[Dict[str, Any]]:
        """Get all notes for a verse."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT note_id, content, note_type, created_at, updated_at "
                "FROM verse_notes WHERE book_id = ? AND chapter = ? AND verse = ? "
                "ORDER BY created_at DESC",
                (book_id, chapter, verse),
            )
            return [
                {
                    "note_id": r[0],
                    "content": r[1],
                    "note_type": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                }
                for r in cursor.fetchall()
            ]

    def update_note(self, note_id: int, content: str) -> bool:
        """Update a note's content."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE verse_notes SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE note_id = ?",
                (content, note_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_note(self, note_id: int) -> bool:
        """Delete a note."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM verse_notes WHERE note_id = ?", (note_id,))
            conn.commit()
            return cursor.rowcount > 0

    # --- Collections ---

    def create_collection(self, name: str, description: str = "") -> int:
        """Create a new verse collection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_collections (name, description) VALUES (?, ?)",
                (name, description),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all user collections with verse counts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT c.collection_id, c.name, c.description, c.created_at, "
                "COUNT(ci.id) as verse_count "
                "FROM user_collections c "
                "LEFT JOIN collection_items ci ON c.collection_id = ci.collection_id "
                "GROUP BY c.collection_id ORDER BY c.created_at DESC"
            )
            return [
                {
                    "collection_id": r[0],
                    "name": r[1],
                    "description": r[2],
                    "created_at": r[3],
                    "verse_count": r[4],
                }
                for r in cursor.fetchall()
            ]

    def add_to_collection(
        self,
        collection_id: int,
        book_id: int,
        chapter: int,
        verse: int,
        note: str = "",
    ) -> bool:
        """Add a verse to a collection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO collection_items (collection_id, book_id, chapter, verse, note) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (collection_id, book_id, chapter, verse, note),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # Already in collection

    def get_collection_items(self, collection_id: int) -> List[Dict[str, Any]]:
        """Get all items in a collection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, book_id, chapter, verse, note, added_at "
                "FROM collection_items WHERE collection_id = ? ORDER BY added_at",
                (collection_id,),
            )
            return [
                {
                    "id": r[0],
                    "book_id": r[1],
                    "chapter": r[2],
                    "verse": r[3],
                    "note": r[4],
                    "added_at": r[5],
                }
                for r in cursor.fetchall()
            ]

    def remove_from_collection(self, collection_id: int, book_id: int, chapter: int, verse: int) -> bool:
        """Remove a verse from a collection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM collection_items WHERE collection_id = ? AND book_id = ? AND chapter = ? AND verse = ?",
                (collection_id, book_id, chapter, verse),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_collection(self, collection_id: int) -> bool:
        """Delete a collection and all its items."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM collection_items WHERE collection_id = ?", (collection_id,))
            cursor.execute("DELETE FROM user_collections WHERE collection_id = ?", (collection_id,))
            conn.commit()
            return cursor.rowcount > 0

    # --- Sharing ---

    def create_share(
        self,
        share_type: str,
        content: Dict[str, Any],
        title: str = "",
    ) -> str:
        """Create a shareable link for content. Returns share token."""
        token = uuid.uuid4().hex[:12]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO shared_items (share_token, share_type, title, content_json) VALUES (?, ?, ?, ?)",
                (token, share_type, title, json.dumps(content)),
            )
            conn.commit()
        return token

    def get_shared_item(self, token: str) -> Optional[Dict[str, Any]]:
        """Retrieve a shared item by token."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT share_type, title, content_json, created_at FROM shared_items WHERE share_token = ?",
                (token,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "share_type": row[0],
                "title": row[1],
                "content": json.loads(row[2]) if row[2] else {},
                "created_at": row[3],
            }
