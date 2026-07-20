"""Remove rows that the two fixes invalidate, so resume re-evaluates them cleanly.

Deletes (cross_references + their provenance) where:
  (a) the explanation contains CJK characters (language drift), OR
  (b) the candidate anchor is non-empty but NOT meaningful (stopword-only / "See on").
Anchor-less rows promoted via shared Strong's are KEPT (legitimately grounded).

Run with --apply to actually delete; default is a dry run.
"""

from __future__ import annotations

import sqlite3
import sys

from abba.semantic import cross_ref_explainer as eng

DB = r"C:\Users\jhend\github\ABBA\bible_data\abba.db"


def main() -> None:
    apply = "--apply" in sys.argv
    c = sqlite3.connect(DB)
    rows = c.execute(
        """
        SELECT cr.ref_id, cr.notes, cc.anchor_phrase
        FROM cross_references cr
        LEFT JOIN cross_reference_candidates cc
          ON cc.source_book_id=cr.source_book_id AND cc.source_chapter=cr.source_chapter
         AND cc.source_verse=cr.source_verse AND cc.target_book_id=cr.target_book_id
         AND cc.target_chapter=cr.target_chapter AND cc.target_verse=cr.target_verse
        WHERE cr.source_dataset='TSK+ollama'
        """
    ).fetchall()

    cjk_ids, weak_ids = [], []
    for ref_id, notes, anchor in rows:
        if notes and eng._contains_cjk(notes):
            cjk_ids.append(ref_id)
        elif anchor and anchor.strip() and not eng.is_meaningful_anchor(anchor):
            weak_ids.append(ref_id)

    to_delete = sorted(set(cjk_ids) | set(weak_ids))
    print(f"promoted rows scanned : {len(rows):,}")
    print(f"  CJK-contaminated    : {len(cjk_ids):,}")
    print(f"  weak-anchor (revoked): {len(weak_ids):,}")
    print(f"  TOTAL to delete     : {len(to_delete):,}")
    print(f"  remaining after     : {len(rows) - len(to_delete):,}")

    if not apply:
        print("\n(dry run — re-run with --apply to delete)")
        return

    cur = c.cursor()
    deleted_prov = 0
    for i in range(0, len(to_delete), 500):
        chunk = to_delete[i : i + 500]
        qmarks = ",".join("?" * len(chunk))
        cur.execute(f"DELETE FROM provenance WHERE entity_type='cross_reference' AND entity_id IN ({qmarks})",
                    [str(x) for x in chunk])
        deleted_prov += cur.rowcount
        cur.execute(f"DELETE FROM cross_references WHERE ref_id IN ({qmarks})", chunk)
    c.commit()
    print(f"\nDELETED {len(to_delete):,} cross_references + {deleted_prov:,} provenance rows")
    print(f"cross_references now: {c.execute('SELECT COUNT(*) FROM cross_references').fetchone()[0]:,}")


if __name__ == "__main__":
    main()
