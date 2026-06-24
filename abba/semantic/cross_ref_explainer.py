"""Cross-reference explanation engine: promotes TSK candidates to explained cross_references."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import requests

from abba.api.constants import DEFAULT_TRANSLATION_ID
from abba.database.sqlite_manager import SQLiteManager
from abba.provenance import Provenance, ProvenanceStore, TrustTier

logger = logging.getLogger(__name__)

OLLAMA_URL: str = os.environ.get("ABBA_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("ABBA_OLLAMA_MODEL", "qwen2.5:14b")
CONFIDENCE_THRESHOLD: float = float(os.environ.get("ABBA_XREF_CONFIDENCE", "0.60"))

# Pipeline version stamped in every provenance record
_PIPELINE_VERSION = "0.1.0"

# Maps numeric book_id → 3-letter STEP/STEPBible book code used in stepbible_verses.book
BOOK_ID_TO_STEP_CODE: dict[int, str] = {
    1: "Gen",
    2: "Exo",
    3: "Lev",
    4: "Num",
    5: "Deu",
    6: "Jos",
    7: "Jdg",
    8: "Rut",
    9: "1Sa",
    10: "2Sa",
    11: "1Ki",
    12: "2Ki",
    13: "1Ch",
    14: "2Ch",
    15: "Ezr",
    16: "Neh",
    17: "Est",
    18: "Job",
    19: "Psa",
    20: "Pro",
    21: "Ecc",
    22: "Sng",
    23: "Isa",
    24: "Jer",
    25: "Lam",
    26: "Ezk",
    27: "Dan",
    28: "Hos",
    29: "Jol",
    30: "Amo",
    31: "Oba",
    32: "Jon",
    33: "Mic",
    34: "Nam",
    35: "Hab",
    36: "Zep",
    37: "Hag",
    38: "Zec",
    39: "Mal",
    40: "Mat",
    41: "Mrk",
    42: "Luk",
    43: "Jhn",
    44: "Act",
    45: "Rom",
    46: "1Co",
    47: "2Co",
    48: "Gal",
    49: "Eph",
    50: "Php",
    51: "Col",
    52: "1Th",
    53: "2Th",
    54: "1Ti",
    55: "2Ti",
    56: "Tit",
    57: "Phm",
    58: "Heb",
    59: "Jas",
    60: "1Pe",
    61: "2Pe",
    62: "1Jn",
    63: "2Jn",
    64: "3Jn",
    65: "Jud",
    66: "Rev",
}

PROMPT_TEMPLATE = (
    "You help Bible readers understand why two passages are cross-referenced. "
    "In 1-2 plain, denominationally-neutral sentences, explain why they are linked, "
    "grounded ONLY in the shared idea given. No doctrinal claims beyond the texts.\n\n"
    "Passage A — {ref_a}: {text_a}\n"
    "Passage B — {ref_b}: {text_b}\n"
    "Shared idea: {shared_idea}.\n\n"
    "Explanation:"
)


def _ollama_generate(prompt: str, model: str, url: str, timeout: int = 120) -> str:
    """Call Ollama /api/generate and return the generated text.

    Args:
        prompt: The prompt to send to the model.
        model: Ollama model name (e.g. ``qwen2.5:14b``).
        url: Base URL of the Ollama server.
        timeout: Request timeout in seconds.

    Returns:
        The stripped generated text.

    Raises:
        RuntimeError: On HTTP error or empty response.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 180},
    }
    response = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def compute_shared_strongs(
    db: SQLiteManager,
    src_book_id: int,
    src_ch: int,
    src_v: int,
    tgt_book_id: int,
    tgt_ch: int,
    tgt_v: int,
) -> list[str]:
    """Return Strong's numbers that appear in both verses (via stepbible_verses.lexical_strongs).

    The ``stepbible_verses`` table has one row per word, with a ``lexical_strongs``
    column holding the normalised canonical Strong's key (e.g. ``H0430``).
    Rows where ``lexical_strongs`` is NULL or empty are skipped.

    Args:
        db: Open SQLiteManager instance.
        src_book_id: Numeric book ID for the source verse.
        src_ch: Source chapter.
        src_v: Source verse.
        tgt_book_id: Numeric book ID for the target verse.
        tgt_ch: Target chapter.
        tgt_v: Target verse.

    Returns:
        Sorted list of Strong's numbers shared by both verses.
    """
    src_code = BOOK_ID_TO_STEP_CODE.get(src_book_id, "")
    tgt_code = BOOK_ID_TO_STEP_CODE.get(tgt_book_id, "")
    if not src_code or not tgt_code:
        return []

    src_rows = db.execute_query(
        "SELECT lexical_strongs FROM stepbible_verses "
        "WHERE book = ? AND chapter = ? AND verse = ? AND lexical_strongs IS NOT NULL AND lexical_strongs != ''",
        (src_code, src_ch, src_v),
    )
    tgt_rows = db.execute_query(
        "SELECT lexical_strongs FROM stepbible_verses "
        "WHERE book = ? AND chapter = ? AND verse = ? AND lexical_strongs IS NOT NULL AND lexical_strongs != ''",
        (tgt_code, tgt_ch, tgt_v),
    )

    src_strongs: set[str] = {r[0] for r in src_rows}
    tgt_strongs: set[str] = {r[0] for r in tgt_rows}
    return sorted(src_strongs & tgt_strongs)


def score_confidence(anchor_phrase: Optional[str], shared_strongs: list[str]) -> float:
    """Compute a confidence score for a candidate cross-reference.

    Design rationale:
    - If the TSK anchor phrase is present, that is the primary grounding signal
      (base 0.7).  Anchor-less candidates get 0.3.
    - Each shared Strong's number adds 0.1, capped at 0.3 bonus, so
      cross-testament links that legitimately share no lexeme can still be
      promoted if anchored.

    Args:
        anchor_phrase: TSK anchor word/phrase or None/empty.
        shared_strongs: Strong's numbers shared by both verses.

    Returns:
        Confidence in [0.0, 1.0].
    """
    base = 0.7 if (anchor_phrase and anchor_phrase.strip()) else 0.3
    bonus = min(0.3, 0.1 * len(shared_strongs))
    return min(1.0, base + bonus)


def build_prompt(
    ref_a: str,
    text_a: str,
    ref_b: str,
    text_b: str,
    anchor: Optional[str],
) -> str:
    """Build the Ollama prompt for a cross-reference pair.

    Args:
        ref_a: Human-readable reference for the source verse (e.g. ``John 3:16``).
        text_a: English text of the source verse.
        ref_b: Human-readable reference for the target verse.
        text_b: English text of the target verse.
        anchor: TSK anchor phrase, or None/empty for a generic label.

    Returns:
        Formatted prompt string.
    """
    if anchor and anchor.strip():
        shared_idea = f'the word/idea "{anchor}"'
    else:
        shared_idea = "a related theme"
    return PROMPT_TEMPLATE.format(
        ref_a=ref_a,
        text_a=text_a,
        ref_b=ref_b,
        text_b=text_b,
        shared_idea=shared_idea,
    )


def explain_candidate(
    db: SQLiteManager,
    candidate: dict[str, Any],
    model: str,
    url: str,
) -> Optional[dict[str, Any]]:
    """Generate an explanation for a single cross-reference candidate.

    Fetches verse texts, computes shared Strong's numbers, scores confidence,
    builds a prompt, and calls Ollama.

    Args:
        db: Open SQLiteManager instance.
        candidate: Row dict from ``cross_reference_candidates``.
        model: Ollama model name.
        url: Ollama base URL.

    Returns:
        Dict with all result fields, or None if either verse text is missing.
    """
    src_book_id: int = candidate["source_book_id"]
    src_ch: int = candidate["source_chapter"]
    src_v: int = candidate["source_verse"]
    tgt_book_id: int = candidate["target_book_id"]
    tgt_ch: int = candidate["target_chapter"]
    tgt_v: int = candidate["target_verse"]
    anchor_phrase: Optional[str] = candidate.get("anchor_phrase")

    # Fetch source verse text
    src_rows = db.execute_query(
        "SELECT text FROM verses WHERE book_id = ? AND chapter = ? AND verse = ? AND translation_id = ?",
        (src_book_id, src_ch, src_v, DEFAULT_TRANSLATION_ID),
    )
    if not src_rows:
        logger.debug("explain_candidate: no text for source %d %d:%d", src_book_id, src_ch, src_v)
        return None
    text_a: str = src_rows[0][0]

    # Fetch target verse text
    tgt_rows = db.execute_query(
        "SELECT text FROM verses WHERE book_id = ? AND chapter = ? AND verse = ? AND translation_id = ?",
        (tgt_book_id, tgt_ch, tgt_v, DEFAULT_TRANSLATION_ID),
    )
    if not tgt_rows:
        logger.debug("explain_candidate: no text for target %d %d:%d", tgt_book_id, tgt_ch, tgt_v)
        return None
    text_b: str = tgt_rows[0][0]

    # Build human-readable references using books table, fall back to STEP code
    src_name = _lookup_book_name(db, src_book_id)
    tgt_name = _lookup_book_name(db, tgt_book_id)
    ref_a = f"{src_name} {src_ch}:{src_v}"
    ref_b = f"{tgt_name} {tgt_ch}:{tgt_v}"

    shared_strongs = compute_shared_strongs(db, src_book_id, src_ch, src_v, tgt_book_id, tgt_ch, tgt_v)
    confidence = score_confidence(anchor_phrase, shared_strongs)
    prompt = build_prompt(ref_a, text_a, ref_b, text_b, anchor_phrase)
    explanation = _ollama_generate(prompt, model, url)

    return {
        "source_book_id": src_book_id,
        "source_chapter": src_ch,
        "source_verse": src_v,
        "target_book_id": tgt_book_id,
        "target_chapter": tgt_ch,
        "target_verse": tgt_v,
        "anchor_phrase": anchor_phrase,
        "ref_a": ref_a,
        "ref_b": ref_b,
        "confidence": confidence,
        "shared_strongs": shared_strongs,
        "explanation": explanation,
    }


def _lookup_book_name(db: SQLiteManager, book_id: int) -> str:
    """Return a display name for a book, consulting the books table first.

    Falls back to the STEP code (e.g. ``Jhn``) if the books table has no
    matching row for the default translation.

    Args:
        db: Open SQLiteManager instance.
        book_id: Numeric book ID.

    Returns:
        Human-readable book name or STEP code fallback.
    """
    rows = db.execute_query(
        "SELECT name FROM books WHERE book_id = ? AND translation_id = ? LIMIT 1",
        (book_id, DEFAULT_TRANSLATION_ID),
    )
    if rows:
        return str(rows[0][0])
    return BOOK_ID_TO_STEP_CODE.get(book_id, str(book_id))


def generate_explanations(
    db_path: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
    threshold: float = CONFIDENCE_THRESHOLD,
    limit: Optional[int] = None,
    source_book_id: Optional[int] = None,
    source_chapter: Optional[int] = None,
    source_verse: Optional[int] = None,
) -> dict[str, Any]:
    """Promote TSK candidates that pass the confidence gate into cross_references.

    Iterates ``cross_reference_candidates``, optionally filtered by source
    book/chapter/verse.  For each candidate that is not already in
    ``cross_references``, generates an AI explanation via Ollama and inserts a
    new row, plus a provenance record.

    Args:
        db_path: Path to the ABBA SQLite database.
        model: Ollama model name.
        url: Ollama base URL.
        threshold: Minimum confidence required to promote a candidate.
        limit: Maximum number of candidates to process (None = all).
        source_book_id: Filter to this source book.
        source_chapter: Filter to this source chapter (requires source_book_id).
        source_verse: Filter to this source verse (requires source_chapter).

    Returns:
        Stats dict with keys: processed, promoted, skipped_low_conf,
        skipped_no_text, skipped_existing.
    """
    db = SQLiteManager(db_path)
    db.initialize_database()

    stats: dict[str, int] = {
        "processed": 0,
        "promoted": 0,
        "skipped_low_conf": 0,
        "skipped_no_text": 0,
        "skipped_existing": 0,
    }

    # Build the SELECT query with optional filters
    query = (
        "SELECT source_book_id, source_chapter, source_verse, "
        "target_book_id, target_chapter, target_verse, anchor_phrase, source_dataset "
        "FROM cross_reference_candidates WHERE 1=1"
    )
    params: list[Any] = []
    if source_book_id is not None:
        query += " AND source_book_id = ?"
        params.append(source_book_id)
    if source_chapter is not None:
        query += " AND source_chapter = ?"
        params.append(source_chapter)
    if source_verse is not None:
        query += " AND source_verse = ?"
        params.append(source_verse)
    query += " ORDER BY source_book_id, source_chapter, source_verse"
    if limit is not None:
        query += f" LIMIT {limit}"

    rows = db.execute_query(query, tuple(params) if params else None)

    for row in rows:
        stats["processed"] += 1
        candidate: dict[str, Any] = {
            "source_book_id": row[0],
            "source_chapter": row[1],
            "source_verse": row[2],
            "target_book_id": row[3],
            "target_chapter": row[4],
            "target_verse": row[5],
            "anchor_phrase": row[6],
            "source_dataset": row[7],
        }

        # Skip if already in cross_references
        existing = db.execute_query(
            "SELECT ref_id FROM cross_references "
            "WHERE source_book_id = ? AND source_chapter = ? AND source_verse = ? "
            "AND target_book_id = ? AND target_chapter = ? AND target_verse = ?",
            (
                candidate["source_book_id"],
                candidate["source_chapter"],
                candidate["source_verse"],
                candidate["target_book_id"],
                candidate["target_chapter"],
                candidate["target_verse"],
            ),
        )
        if existing:
            stats["skipped_existing"] += 1
            continue

        result = explain_candidate(db, candidate, model, url)
        if result is None:
            stats["skipped_no_text"] += 1
            continue

        if result["confidence"] < threshold:
            stats["skipped_low_conf"] += 1
            continue

        # Insert into cross_references and capture the new ref_id
        ref_id = _insert_cross_reference(db, result)

        # Record provenance
        prov = Provenance(
            entity_type="cross_reference",
            entity_id=str(ref_id),
            source="ollama",
            source_detail=f"model={model}; anchor={result['anchor_phrase']!r}",
            trust_tier=TrustTier.GENERATED,
            trust_rationale=(
                f"AI-generated explanation grounded in TSK anchor {result['anchor_phrase']!r} "
                f"and shared Strong's {result['shared_strongs']}. "
                f"Confidence {result['confidence']:.2f} >= threshold {threshold:.2f}."
            ),
            generated_by=model,
            grounding={
                "anchor_phrase": result["anchor_phrase"],
                "shared_strongs": result["shared_strongs"],
                "source_dataset": candidate.get("source_dataset", "TSK"),
            },
            confidence=result["confidence"],
            pipeline_version=_PIPELINE_VERSION,
        )
        ProvenanceStore(db).record(prov)

        stats["promoted"] += 1
        logger.info(
            "Promoted %s → %s (conf=%.2f)",
            result["ref_a"],
            result["ref_b"],
            result["confidence"],
        )

    return stats


def _insert_cross_reference(db: SQLiteManager, result: dict[str, Any]) -> int:
    """Insert one row into cross_references and return its ref_id.

    Uses a direct connection so we can capture ``lastrowid``.

    Args:
        db: Open SQLiteManager instance.
        result: Dict returned by :func:`explain_candidate`.

    Returns:
        The ``ref_id`` of the newly inserted row.
    """
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO cross_references (
                source_book_id, source_chapter, source_verse,
                target_book_id, target_chapter, target_verse,
                ref_type, confidence, source_dataset, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["source_book_id"],
                result["source_chapter"],
                result["source_verse"],
                result["target_book_id"],
                result["target_chapter"],
                result["target_verse"],
                "TSK",
                result["confidence"],
                "TSK+ollama",
                result["explanation"],
            ),
        )
        conn.commit()
        ref_id: int = cursor.lastrowid or 0
        if ref_id == 0:
            # Row existed (INSERT OR IGNORE was a no-op); fetch the existing id
            existing = conn.execute(
                "SELECT ref_id FROM cross_references "
                "WHERE source_book_id=? AND source_chapter=? AND source_verse=? "
                "AND target_book_id=? AND target_chapter=? AND target_verse=?",
                (
                    result["source_book_id"],
                    result["source_chapter"],
                    result["source_verse"],
                    result["target_book_id"],
                    result["target_chapter"],
                    result["target_verse"],
                ),
            ).fetchone()
            ref_id = existing[0] if existing else 0
    return ref_id
