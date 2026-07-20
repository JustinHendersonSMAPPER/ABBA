"""Audit promoted cross-reference explanations for grounding + accuracy.

Read-only (WAL allows concurrent reads while the run continues). Prints aggregate
health stats + a stratified random sample with both verse texts so quality can be judged.
"""

from __future__ import annotations

import sqlite3
import textwrap

DB = r"C:\Users\jhend\github\ABBA\bible_data\abba.db"
TR = "BSB"


def vtext(c: sqlite3.Connection, b: int, ch: int, v: int) -> str:
    r = c.execute(
        "SELECT text FROM verses WHERE book_id=? AND chapter=? AND verse=? AND translation_id=?",
        (b, ch, v, TR),
    ).fetchone()
    return r[0] if r else "(no text)"


def bname(c: sqlite3.Connection, b: int) -> str:
    r = c.execute("SELECT name FROM books WHERE book_id=? AND translation_id=? LIMIT 1", (b, TR)).fetchone()
    return r[0] if r else str(b)


def main() -> None:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    # ---- aggregate health over AI-generated promotions ----
    total = c.execute("SELECT COUNT(*) FROM cross_references WHERE source_dataset='TSK+ollama'").fetchone()[0]
    empty = c.execute(
        "SELECT COUNT(*) FROM cross_references WHERE source_dataset='TSK+ollama' AND (notes IS NULL OR TRIM(notes)='')"
    ).fetchone()[0]
    # truncation heuristic: explanation not ending in sentence punctuation
    trunc = c.execute(
        "SELECT COUNT(*) FROM cross_references WHERE source_dataset='TSK+ollama' "
        "AND notes IS NOT NULL AND TRIM(notes)!='' "
        "AND SUBSTR(TRIM(notes),-1) NOT IN ('.','!','?','\"',')')"
    ).fetchone()[0]
    lens = c.execute(
        "SELECT MIN(LENGTH(notes)), AVG(LENGTH(notes)), MAX(LENGTH(notes)) "
        "FROM cross_references WHERE source_dataset='TSK+ollama' AND notes IS NOT NULL"
    ).fetchone()
    leaky = c.execute(
        "SELECT COUNT(*) FROM cross_references WHERE source_dataset='TSK+ollama' AND LOWER(notes) LIKE '%<think>%'"
    ).fetchone()[0]
    # confidence histogram
    conf = c.execute(
        "SELECT confidence, COUNT(*) FROM cross_references WHERE source_dataset='TSK+ollama' "
        "GROUP BY confidence ORDER BY confidence"
    ).fetchall()

    print("=" * 96)
    print("AGGREGATE HEALTH (source_dataset='TSK+ollama')")
    print(f"  total promoted         : {total:,}")
    print(f"  empty explanations     : {empty}")
    print(f"  <think> leaks          : {leaky}")
    print(f"  not ending in . ! ? \" ): {trunc}  (possible truncation)")
    print(f"  explanation length     : min={lens[0]} avg={lens[1]:.0f} max={lens[2]} chars")
    print(f"  confidence histogram   : {[(round(x,2), n) for x, n in conf]}")

    # ---- stratified random sample: spread across source books ----
    print("\n" + "=" * 96)
    print("STRATIFIED RANDOM SAMPLE (one per distinct source book, up to 18)")
    rows = c.execute(
        """
        SELECT cr.ref_id, cr.source_book_id, cr.source_chapter, cr.source_verse,
               cr.target_book_id, cr.target_chapter, cr.target_verse, cr.confidence, cr.notes
        FROM cross_references cr
        JOIN (
            SELECT source_book_id, MIN(ref_id) AS rid
            FROM cross_references WHERE source_dataset='TSK+ollama'
            GROUP BY source_book_id
            ORDER BY RANDOM() LIMIT 18
        ) pick ON cr.ref_id = pick.rid
        ORDER BY cr.source_book_id, cr.source_chapter, cr.source_verse
        """
    ).fetchall()
    _print_samples(c, rows)

    # ---- targeted: anchor-less promotions (passed via shared Strong's, not phrase) ----
    print("\n" + "=" * 96)
    print("TARGETED SAMPLE: anchor-less promotions (grounded ONLY on shared Strong's)")
    rows = c.execute(
        """
        SELECT cr.ref_id, cr.source_book_id, cr.source_chapter, cr.source_verse,
               cr.target_book_id, cr.target_chapter, cr.target_verse, cr.confidence, cr.notes
        FROM cross_references cr
        JOIN provenance p ON p.entity_type='cross_reference' AND p.entity_id=CAST(cr.ref_id AS TEXT)
        WHERE cr.source_dataset='TSK+ollama'
          AND (p.grounding_json LIKE '%"anchor_phrase": null%' OR p.grounding_json LIKE '%"anchor_phrase": ""%')
        ORDER BY RANDOM() LIMIT 6
        """
    ).fetchall()
    _print_samples(c, rows, show_grounding=True)


def _print_samples(c: sqlite3.Connection, rows: list, show_grounding: bool = False) -> None:
    for (rid, sb, sc, sv, tb, tc, tv, conf, notes) in rows:
        g = c.execute(
            "SELECT grounding_json, generated_by, trust_tier FROM provenance "
            "WHERE entity_type='cross_reference' AND entity_id=? LIMIT 1",
            (str(rid),),
        ).fetchone()
        grounding = g[0] if g else "(no provenance!)"
        print("\n" + "-" * 96)
        print(f"ref {rid} | conf {conf} | tier {g[2] if g else '?'} | by {g[1] if g else '?'}")
        print(f"  SRC {bname(c,sb)} {sc}:{sv} — {vtext(c,sb,sc,sv)}")
        print(f"  TGT {bname(c,tb)} {tc}:{tv} — {vtext(c,tb,tc,tv)}")
        if show_grounding:
            print(f"  grounding: {grounding}")
        print("  WHY: " + "\n       ".join(textwrap.wrap(notes or "(empty)", 92)))


if __name__ == "__main__":
    main()
