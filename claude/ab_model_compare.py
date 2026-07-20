"""A/B compare cross-ref explanation quality: qwen2.5:14b vs qwen3:14b (thinking off).

Pulls a few real candidates, builds the production prompt, and generates with both
models so we can judge whether the newer generation is worth switching the live run.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from abba.database.sqlite_manager import SQLiteManager
from abba.semantic import cross_ref_explainer as eng

URL = "http://localhost:11434"

# Representative real pairs: (src_book, src_ch, src_v, tgt_book, tgt_ch, tgt_v, anchor)
PAIRS = [
    (43, 3, 16, 1, 22, 12, "gave"),        # John 3:16 -> Gen 22:12  (lexical anchor, cross-testament)
    (45, 1, 16, 19, 110, 2, "for it is"),  # Rom 1:16 -> Psa 110:2   (phrase anchor)
    (1, 1, 1, 43, 1, 1, "beginning"),      # Gen 1:1 -> John 1:1     (iconic thematic)
    (40, 5, 8, 19, 24, 4, "pure"),         # Matt 5:8 -> Psa 24:4    (Beatitude -> Psalm)
]


def gen(model: str, prompt: str, think: bool, num_predict: int) -> tuple[str, float]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }
    if not think:
        payload["think"] = False  # Ollama: disable reasoning trace for hybrid models
    t = time.time()
    r = requests.post(f"{URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    body = r.json()
    return (body.get("response", "").strip(), time.time() - t)


def main() -> None:
    db = SQLiteManager("bible_data/abba.db")
    for (sb, sc, sv, tb, tc, tv, anchor) in PAIRS:
        sa = db.execute_query(
            "SELECT text FROM verses WHERE book_id=? AND chapter=? AND verse=? AND translation_id=?",
            (sb, sc, sv, eng.DEFAULT_TRANSLATION_ID),
        )
        ta = db.execute_query(
            "SELECT text FROM verses WHERE book_id=? AND chapter=? AND verse=? AND translation_id=?",
            (tb, tc, tv, eng.DEFAULT_TRANSLATION_ID),
        )
        if not sa or not ta:
            print(f"!! missing text for {sb} {sc}:{sv} or {tb} {tc}:{tv}; skipping")
            continue
        ref_a = f"{eng._lookup_book_name(db, sb)} {sc}:{sv}"
        ref_b = f"{eng._lookup_book_name(db, tb)} {tc}:{tv}"
        prompt = eng.build_prompt(ref_a, sa[0][0], ref_b, ta[0][0], anchor)

        print("\n" + "=" * 90)
        print(f"{ref_a}  ->  {ref_b}   (anchor: {anchor!r})")
        print("-" * 90)
        try:
            q25, t25 = gen("qwen2.5:14b", prompt, think=True, num_predict=180)
            print(f"[qwen2.5:14b  {t25:4.1f}s] {q25}")
        except Exception as e:  # noqa: BLE001
            print(f"[qwen2.5:14b] ERROR {e}")
        try:
            q3, t3 = gen("qwen3:14b", prompt, think=False, num_predict=256)
            tag = " <HAS-THINK-TAGS!>" if "<think>" in q3.lower() else ""
            print(f"[qwen3:14b    {t3:4.1f}s]{tag} {q3}")
        except Exception as e:  # noqa: BLE001
            print(f"[qwen3:14b] ERROR {e}")


if __name__ == "__main__":
    main()
