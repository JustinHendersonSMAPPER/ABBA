"""A/B: qwen2.5:14b vs huihui_ai/qwen3.5-abliterated:9B (thinking off, /no_think fallback)."""

from __future__ import annotations

import time
from typing import Any

import requests

from abba.database.sqlite_manager import SQLiteManager
from abba.semantic import cross_ref_explainer as eng

URL = "http://localhost:11434"
Q25 = "qwen2.5:14b"
Q35 = "huihui_ai/qwen3.5-abliterated:9B"

PAIRS = [
    (43, 3, 16, 1, 22, 12, "gave"),
    (45, 1, 16, 19, 110, 2, "for it is"),
    (1, 1, 1, 43, 1, 1, "beginning"),
    (40, 5, 8, 19, 24, 4, "pure"),
]


def gen(model: str, prompt: str, think: bool, num_predict: int, no_think_tag: bool = False) -> tuple[str, float]:
    p = prompt + (" /no_think" if no_think_tag else "")
    payload: dict[str, Any] = {
        "model": model,
        "prompt": p,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }
    if not think:
        payload["think"] = False
    t = time.time()
    r = requests.post(f"{URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip(), time.time() - t


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
        ref_a = f"{eng._lookup_book_name(db, sb)} {sc}:{sv}"
        ref_b = f"{eng._lookup_book_name(db, tb)} {tc}:{tv}"
        prompt = eng.build_prompt(ref_a, sa[0][0], ref_b, ta[0][0], anchor)

        print("\n" + "=" * 92)
        print(f"{ref_a}  ->  {ref_b}   (anchor: {anchor!r})")
        print("-" * 92)
        out, dt = gen(Q25, prompt, think=True, num_predict=180)
        print(f"[qwen2.5:14b           {dt:4.1f}s] {out}")

        # abliterated 9B: try think:false first
        out, dt = gen(Q35, prompt, think=False, num_predict=300)
        note = ""
        if not out or "<think>" in out.lower():
            out2, dt2 = gen(Q35, prompt, think=False, num_predict=300, no_think_tag=True)
            note = f"  (think:false empty/leaky -> /no_think retry {dt2:.1f}s)"
            out, dt = out2, dt2
        flag = " <EMPTY!>" if not out else (" <THINK-LEAK!>" if "<think>" in out.lower() else "")
        print(f"[qwen3.5-ablit:9B      {dt:4.1f}s]{flag}{note} {out}")


if __name__ == "__main__":
    main()
