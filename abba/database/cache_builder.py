"""Build precomputed verse annotation cache for fast deep-depth queries.

Iterates all unique verse references from the words table and precomputes
annotation data (richness flags, cross-references, cultural context,
passage info, literary structures, speaker attribution, active genre)
into the verse_annotations_cache table.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _query_words(conn: sqlite3.Connection, book: str, chapter: int, verse: int) -> List[Dict[str, Any]]:
    """Get original language words for a verse."""
    cursor = conn.execute(
        "SELECT word_num, hebrew_text, greek_text, transliteration, "
        "translation, strongs_primary, morphology_code, language "
        "FROM words WHERE book = ? AND chapter = ? AND verse = ? "
        "ORDER BY word_num",
        (book, chapter, verse),
    )
    return [
        {
            "word_num": r[0],
            "original_text": r[1] or r[2],
            "transliteration": r[3],
            "english_gloss": r[4],
            "strongs_number": r[5],
            "morphology_code": r[6],
            "language": r[7],
        }
        for r in cursor.fetchall()
    ]


def _query_richness(conn: sqlite3.Connection, book: str, chapter: int, verse: int) -> List[Dict[str, Any]]:
    """Get richness flags for a verse."""
    try:
        cursor = conn.execute(
            "SELECT wr.word_num, wr.strongs_number, wr.richness_score, "
            "wr.untranslatable_nuances, wr.morphology_significance, "
            "l.original_word, l.gloss, l.definition "
            "FROM word_richness wr "
            "LEFT JOIN lexicon l ON wr.strongs_number = l.strongs_number "
            "WHERE wr.book = ? AND wr.chapter = ? AND wr.verse = ? "
            "AND wr.richness_score > 0.3 "
            "ORDER BY wr.richness_score DESC",
            (book, chapter, verse),
        )
        flags = []
        for r in cursor.fetchall():
            nuances: List[str] = []
            if r[3]:
                try:
                    nuances = json.loads(r[3])
                except (json.JSONDecodeError, TypeError):
                    pass
            flags.append(
                {
                    "word_num": r[0],
                    "strongs_number": r[1],
                    "richness_score": r[2],
                    "untranslatable_nuances": nuances,
                    "morphology_significance": r[4],
                    "original_word": r[5],
                    "english_gloss": r[6],
                    "full_definition": r[7],
                }
            )
        return flags
    except sqlite3.OperationalError:
        return []


def _query_cross_refs(conn: sqlite3.Connection, book_id: int, chapter: int, verse: int) -> List[Dict[str, Any]]:
    """Get cross-references for a verse (outgoing + incoming)."""
    refs: List[Dict[str, Any]] = []
    try:
        cursor = conn.execute(
            "SELECT target_book_id, target_chapter, target_verse, ref_type, confidence, notes "
            "FROM cross_references "
            "WHERE source_book_id = ? AND source_chapter = ? AND source_verse = ?",
            (book_id, chapter, verse),
        )
        for r in cursor.fetchall():
            refs.append(
                {
                    "target_reference": f"{r[0]} {r[1]}:{r[2]}",
                    "ref_type": r[3],
                    "confidence": r[4] or 0.8,
                    "notes": r[5],
                }
            )
        cursor = conn.execute(
            "SELECT source_book_id, source_chapter, source_verse, ref_type, confidence, notes "
            "FROM cross_references "
            "WHERE target_book_id = ? AND target_chapter = ? AND target_verse = ?",
            (book_id, chapter, verse),
        )
        for r in cursor.fetchall():
            refs.append(
                {
                    "target_reference": f"{r[0]} {r[1]}:{r[2]}",
                    "ref_type": r[3],
                    "confidence": r[4] or 0.8,
                    "notes": r[5],
                }
            )
    except sqlite3.OperationalError:
        pass
    return refs


def _query_cultural_context(conn: sqlite3.Connection, book_id: int) -> List[Dict[str, Any]]:
    """Get book-level cultural context."""
    try:
        cursor = conn.execute(
            "SELECT context_id, context_type, title, summary, detailed_content, "
            "time_period, geographic_region, confidence "
            "FROM cultural_context "
            "WHERE book_id = ? AND start_chapter IS NULL "
            "ORDER BY display_priority",
            (book_id,),
        )
        return [
            {
                "context_id": r[0],
                "context_type": r[1],
                "title": r[2],
                "summary": r[3],
                "detailed_content": r[4],
                "time_period": r[5],
                "geographic_region": r[6],
                "confidence": r[7],
            }
            for r in cursor.fetchall()
        ]
    except sqlite3.OperationalError:
        return []


def _query_passage_info(conn: sqlite3.Connection, book_id: int, chapter: int, verse: int) -> Optional[Dict[str, Any]]:
    """Get the innermost passage containing a verse."""
    try:
        cursor = conn.execute(
            "SELECT passage_id, title, genre, literary_type, structural_features, "
            "start_chapter, start_verse, end_chapter, end_verse "
            "FROM passages "
            "WHERE book_id = ? "
            "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
            "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) "
            "ORDER BY (end_chapter - start_chapter) ASC, (end_verse - start_verse) ASC "
            "LIMIT 1",
            (book_id, chapter, chapter, verse, chapter, chapter, verse),
        )
        r = cursor.fetchone()
        if r:
            features: List[str] = []
            if r[4]:
                try:
                    features = json.loads(r[4])
                except (json.JSONDecodeError, TypeError):
                    pass
            return {
                "passage_id": r[0],
                "title": r[1],
                "genre": r[2],
                "literary_type": r[3],
                "structural_features": features,
                "start_chapter": r[5],
                "start_verse": r[6],
                "end_chapter": r[7],
                "end_verse": r[8],
            }
    except sqlite3.OperationalError:
        pass
    return None


def _query_literary_structures(
    conn: sqlite3.Connection, book_id: int, chapter: int, verse: int
) -> List[Dict[str, Any]]:
    """Get literary structures containing a verse."""
    try:
        cursor = conn.execute(
            "SELECT structure_type, description, significance, elements "
            "FROM literary_structures "
            "WHERE book_id = ? "
            "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
            "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) ",
            (book_id, chapter, chapter, verse, chapter, chapter, verse),
        )
        structures = []
        for r in cursor.fetchall():
            elems: List[str] = []
            if r[3]:
                try:
                    elems = json.loads(r[3])
                except (json.JSONDecodeError, TypeError):
                    pass
            structures.append(
                {
                    "structure_type": r[0],
                    "description": r[1],
                    "significance": r[2],
                    "elements": elems,
                }
            )
        return structures
    except sqlite3.OperationalError:
        return []


def _query_speaker(conn: sqlite3.Connection, book_id: int, chapter: int, verse: int) -> Optional[Dict[str, Any]]:
    """Get speaker attribution for a verse."""
    try:
        cursor = conn.execute(
            "SELECT speaker, context_note FROM speaker_attributions "
            "WHERE book_id = ? "
            "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
            "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) "
            "LIMIT 1",
            (book_id, chapter, chapter, verse, chapter, chapter, verse),
        )
        r = cursor.fetchone()
        if r:
            return {"speaker": r[0], "context_note": r[1]}
    except sqlite3.OperationalError:
        pass
    return None


def _query_active_genre(conn: sqlite3.Connection, book_id: int, chapter: int, verse: int) -> Optional[str]:
    """Determine the active literary genre at a verse."""
    try:
        cursor = conn.execute(
            "SELECT to_genre FROM genre_shifts "
            "WHERE book_id = ? AND (chapter < ? OR (chapter = ? AND verse <= ?)) "
            "ORDER BY chapter DESC, verse DESC LIMIT 1",
            (book_id, chapter, chapter, verse),
        )
        r = cursor.fetchone()
        if r:
            return str(r[0])
    except sqlite3.OperationalError:
        pass
    try:
        cursor = conn.execute(
            "SELECT primary_genre FROM book_metadata WHERE book_id = ?",
            (book_id,),
        )
        r = cursor.fetchone()
        if r:
            return str(r[0])
    except sqlite3.OperationalError:
        pass
    return None


def _get_book_id_mapping(conn: sqlite3.Connection) -> Dict[str, int]:
    """Build mapping from book name (used in words table) to book_id."""
    mapping: Dict[str, int] = {}
    try:
        cursor = conn.execute("SELECT DISTINCT book FROM words")
        book_names = [str(r[0]) for r in cursor.fetchall()]
    except sqlite3.OperationalError:
        return mapping

    try:
        cursor = conn.execute("SELECT book_id, name FROM books GROUP BY book_id, name")
        for r in cursor.fetchall():
            bid, name = int(r[0]), str(r[1])
            mapping[name] = bid
            # Also try matching word table names to book_id
            for bn in book_names:
                if name.lower().startswith(bn.lower()[:3]) or bn.lower().startswith(name.lower()[:3]):
                    mapping[bn] = bid
    except sqlite3.OperationalError:
        pass
    return mapping


def build_annotation_cache(db_path: Path) -> int:
    """Build the verse annotation cache for all unique verses.

    Args:
        db_path: Path to the ABBA database

    Returns:
        Number of cache entries created
    """
    start_time = time.perf_counter()
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")

    # Get all unique verse references from the words table
    cursor = conn.execute("SELECT DISTINCT book, chapter, verse FROM words ORDER BY book, chapter, verse")
    verses = cursor.fetchall()
    logger.info("Building annotation cache for %d unique verses", len(verses))

    book_id_map = _get_book_id_mapping(conn)

    # Cultural context is book-level, so cache per book
    cultural_cache: Dict[int, List[Dict[str, Any]]] = {}

    count = 0
    for book_name, chapter, verse_num in verses:
        book_id = book_id_map.get(str(book_name))
        if book_id is None:
            continue

        words = _query_words(conn, str(book_name), int(chapter), int(verse_num))
        richness = _query_richness(conn, str(book_name), int(chapter), int(verse_num))
        xrefs = _query_cross_refs(conn, book_id, int(chapter), int(verse_num))

        if book_id not in cultural_cache:
            cultural_cache[book_id] = _query_cultural_context(conn, book_id)
        cultural = cultural_cache[book_id]

        passage = _query_passage_info(conn, book_id, int(chapter), int(verse_num))
        structures = _query_literary_structures(conn, book_id, int(chapter), int(verse_num))
        speaker = _query_speaker(conn, book_id, int(chapter), int(verse_num))
        genre = _query_active_genre(conn, book_id, int(chapter), int(verse_num))

        conn.execute(
            "INSERT OR REPLACE INTO verse_annotations_cache ("
            "book_id, chapter, verse, words_json, richness_flags_json, "
            "cross_references_json, cultural_context_json, passage_info_json, "
            "literary_structures_json, speaker_json, active_genre"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                book_id,
                int(chapter),
                int(verse_num),
                json.dumps(words) if words else None,
                json.dumps(richness) if richness else None,
                json.dumps(xrefs) if xrefs else None,
                json.dumps(cultural) if cultural else None,
                json.dumps(passage) if passage else None,
                json.dumps(structures) if structures else None,
                json.dumps(speaker) if speaker else None,
                genre,
            ),
        )
        count += 1

        if count % 1000 == 0:
            conn.commit()
            logger.info("  Cached %d/%d verses...", count, len(verses))

    conn.commit()
    conn.close()

    elapsed = time.perf_counter() - start_time
    logger.info("Annotation cache built: %d entries in %.1fs", count, elapsed)
    return count
