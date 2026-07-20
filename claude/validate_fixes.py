"""Live validation of the two fixes against real Ollama, using the UPDATED engine."""

from __future__ import annotations

from abba.database.sqlite_manager import SQLiteManager
from abba.semantic import cross_ref_explainer as eng

DB = "bible_data/abba.db"

# Pairs that drifted to Chinese under the old prompt (book_id, ch, v) x2 + anchor
DRIFT_PAIRS = [
    (23, 53, 5, 60, 2, 24, "chastisement"),   # Isa 53:5 -> 1Pet 2:24
    (1, 3, 3, 18, 1, 11, "touch"),            # Gen 3:3  -> Job 1:11
    (66, 21, 4, 23, 65, 18, "neither sorrow"),  # Rev 21:4 -> Isa 65:18
    (1, 3, 19, 21, 5, 15, "till"),            # Gen 3:19 -> Ecc 5:15
]

# Anchor-hygiene cases: these should now be treated as NON-meaningful (base 0.3)
JUNK = ["that", "I will", "they", "for it is", "See on Matt 4:1", "thou shalt"]


def main() -> None:
    db = SQLiteManager(DB)
    print("=== FIX 1: language drift (regenerate known-drift pairs x2 each) ===")
    clean = total = 0
    for (sb, sc, sv, tb, tc, tv, anchor) in DRIFT_PAIRS:
        cand = {
            "source_book_id": sb, "source_chapter": sc, "source_verse": sv,
            "target_book_id": tb, "target_chapter": tc, "target_verse": tv,
            "anchor_phrase": anchor, "source_dataset": "TSK",
        }
        for _ in range(2):
            total += 1
            res = eng.explain_candidate(db, cand, eng.OLLAMA_MODEL, eng.OLLAMA_URL)
            if res is None:
                print(f"  DEFERRED (no clean English): {sb} {sc}:{sv} -> {tb} {tc}:{tv}")
                continue
            has_cjk = eng._contains_cjk(res["explanation"])
            clean += 0 if has_cjk else 1
            flag = " <CJK!>" if has_cjk else ""
            print(f"  [{anchor}]{flag} {res['explanation'][:120]}")
    print(f"  -> {clean}/{total} clean English\n")

    print("=== FIX 2: anchor hygiene (junk anchors must score 0.3 base) ===")
    for a in JUNK:
        meaningful = eng.is_meaningful_anchor(a)
        score_no_strongs = eng.score_confidence(a, [])
        print(f"  {a!r:22} meaningful={meaningful}  conf(no strongs)={score_no_strongs}  -> "
              f"{'PROMOTED' if score_no_strongs >= 0.6 else 'DEFERRED'}")


if __name__ == "__main__":
    main()
