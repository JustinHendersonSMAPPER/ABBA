"""Concurrent, resumable driver for the FULL cross-reference explanation pass.

Reuses the merged engine (abba.semantic.cross_ref_explainer) but parallelises the
Ollama calls (continuous batching on the GPU gives ~3x) while relying on SQLite's
busy-timeout to serialise the brief writes. Resumable: candidates already in
cross_references are skipped, so re-running continues where it left off.

Run detached:  nohup uv run python claude/run_full_xref.py > /tmp/xref_full.log 2>&1 &
Test a slice:  uv run python claude/run_full_xref.py --source 45 1 16   (Romans 1:16)
               uv run python claude/run_full_xref.py --limit 50
Tune workers:  --workers 12
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import concurrent.futures as cf
from typing import Any

from abba.database.sqlite_manager import SQLiteManager
from abba.provenance import Provenance, ProvenanceStore, TrustTier
from abba.semantic import cross_ref_explainer as eng

DB_PATH = "bible_data/abba.db"
_COLS = (
    "source_book_id",
    "source_chapter",
    "source_verse",
    "target_book_id",
    "target_chapter",
    "target_verse",
    "anchor_phrase",
    "source_dataset",
)

# One SQLiteManager per worker thread (migrations run once per thread, not per candidate).
_tl = threading.local()


def _db() -> SQLiteManager:
    db = getattr(_tl, "db", None)
    if db is None:
        db = SQLiteManager(DB_PATH)
        _tl.db = db
    return db


def _already_done(db: SQLiteManager, c: dict[str, Any]) -> bool:
    return bool(
        db.execute_query(
            "SELECT 1 FROM cross_references WHERE source_book_id=? AND source_chapter=? AND source_verse=? "
            "AND target_book_id=? AND target_chapter=? AND target_verse=? LIMIT 1",
            (
                c["source_book_id"],
                c["source_chapter"],
                c["source_verse"],
                c["target_book_id"],
                c["target_chapter"],
                c["target_verse"],
            ),
        )
    )


def _process(c: dict[str, Any], model: str, url: str, threshold: float) -> str:
    """Process one candidate. Returns an outcome key. Never raises (errors -> 'error')."""
    try:
        db = _db()
        if _already_done(db, c):
            return "existing"
        # Fast gate: a MEANINGFUL anchor always scores >= 0.7 (passes). Anchor-less and
        # stopword/"See on" anchors can fall below threshold, so pre-compute their score
        # and skip the expensive LLM call when they can't pass anyway.
        anchor = c.get("anchor_phrase")
        if not eng.is_meaningful_anchor(anchor):
            shared = eng.compute_shared_strongs(
                db,
                c["source_book_id"],
                c["source_chapter"],
                c["source_verse"],
                c["target_book_id"],
                c["target_chapter"],
                c["target_verse"],
            )
            if eng.score_confidence(anchor, shared) < threshold:
                return "low_conf"
        result = eng.explain_candidate(db, c, model, url)
        if result is None:
            return "no_text"
        if result["confidence"] < threshold:
            return "low_conf"
        ref_id = eng._insert_cross_reference(db, result)
        ProvenanceStore(db).record(
            Provenance(
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
                    "source_dataset": c.get("source_dataset", "TSK"),
                },
                confidence=result["confidence"],
                pipeline_version=eng._PIPELINE_VERSION,
            )
        )
        return "promoted"
    except Exception as exc:  # noqa: BLE001 - one bad candidate must not stop the run; it's retried on resume
        sys.stderr.write(f"error on {c['source_book_id']}:{c['source_chapter']}:{c['source_verse']} -> {exc}\n")
        return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", nargs=3, type=int, metavar=("BOOK", "CH", "V"), default=None)
    ap.add_argument("--batch", type=int, default=2000, help="candidates submitted per wave")
    args = ap.parse_args()

    model, url, threshold = eng.OLLAMA_MODEL, eng.OLLAMA_URL, eng.CONFIDENCE_THRESHOLD

    read_db = SQLiteManager(DB_PATH)
    where = "WHERE 1=1"
    params: list[Any] = []
    if args.source:
        where += " AND source_book_id=? AND source_chapter=? AND source_verse=?"
        params += args.source
    q = (
        f"SELECT {', '.join(_COLS)} FROM cross_reference_candidates {where} "
        "ORDER BY source_book_id, source_chapter, source_verse, target_book_id, target_chapter, target_verse"
    )
    if args.limit:
        q += f" LIMIT {args.limit}"
    rows = read_db.execute_query(q, tuple(params) if params else None)
    candidates = [dict(zip(_COLS, r)) for r in rows]
    total = len(candidates)

    stats = {k: 0 for k in ("processed", "promoted", "existing", "no_text", "low_conf", "error")}
    start = time.time()
    print(f"[xref-full] model={model} workers={args.workers} threshold={threshold} candidates={total:,}", flush=True)

    def log_progress() -> None:
        done = stats["processed"]
        elapsed = max(1e-6, time.time() - start)
        rate = done / elapsed
        remaining = total - done
        eta_h = (remaining / rate / 3600) if rate > 0 else 0
        print(
            f"[xref-full] {done:,}/{total:,} | promoted {stats['promoted']:,} "
            f"existing {stats['existing']:,} low_conf {stats['low_conf']:,} no_text {stats['no_text']:,} "
            f"err {stats['error']:,} | {rate:.1f}/s | ETA {eta_h:.1f}h",
            flush=True,
        )

    # Submit in waves to bound in-flight futures + checkpoint progress.
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i in range(0, total, args.batch):
            wave = candidates[i : i + args.batch]
            futs = [ex.submit(_process, c, model, url, threshold) for c in wave]
            for fut in cf.as_completed(futs):
                outcome = fut.result()
                stats[outcome] = stats.get(outcome, 0) + 1
                stats["processed"] += 1
                if stats["processed"] % 500 == 0:
                    log_progress()

    log_progress()
    print(f"[xref-full] DONE in {(time.time()-start)/3600:.2f}h -> {stats}", flush=True)


if __name__ == "__main__":
    main()
