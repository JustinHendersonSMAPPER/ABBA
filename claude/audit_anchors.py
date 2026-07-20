"""Quantify the two failure modes the sample surfaced: stopword anchors + truncation."""

from __future__ import annotations

import sqlite3
import textwrap

DB = r"C:\Users\jhend\github\ABBA\bible_data\abba.db"

# Function/stop words: anchors made ONLY of these carry no semantic grounding.
STOP = {
    "the", "a", "an", "and", "or", "but", "for", "nor", "so", "yet", "of", "to", "in", "on",
    "at", "by", "with", "from", "as", "that", "this", "these", "those", "it", "is", "was",
    "are", "were", "be", "been", "he", "she", "they", "we", "i", "you", "him", "her", "them",
    "his", "their", "your", "my", "me", "us", "who", "whom", "which", "what", "shall", "will",
    "not", "no", "all", "into", "unto", "out", "up", "down", "then", "there", "here", "when",
    "do", "did", "have", "has", "had", "them", "thee", "thou", "thy", "ye", "let", "if",
}


def is_stopword_anchor(anchor: str) -> bool:
    toks = [t for t in anchor.lower().replace(",", " ").replace(".", " ").split() if t]
    return len(toks) > 0 and all(t in STOP for t in toks)


def main() -> None:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    # Join promoted rows to their candidate anchor.
    rows = c.execute(
        """
        SELECT cc.anchor_phrase, COUNT(*) n
        FROM cross_references cr
        JOIN cross_reference_candidates cc
          ON cc.source_book_id=cr.source_book_id AND cc.source_chapter=cr.source_chapter
         AND cc.source_verse=cr.source_verse AND cc.target_book_id=cr.target_book_id
         AND cc.target_chapter=cr.target_chapter AND cc.target_verse=cr.target_verse
        WHERE cr.source_dataset='TSK+ollama'
        GROUP BY cc.anchor_phrase
        """
    ).fetchall()

    total = sum(n for _, n in rows)
    anchored = sum(n for a, n in rows if a and a.strip())
    stop = sum(n for a, n in rows if a and a.strip() and is_stopword_anchor(a))
    print(f"promoted (joined to candidate): {total:,}")
    print(f"  anchored                     : {anchored:,}")
    print(f"  STOPWORD-only anchors        : {stop:,}  ({100*stop/max(1,anchored):.1f}% of anchored)")

    print("\nTop 25 anchor phrases among promoted (* = stopword-only):")
    for a, n in sorted(rows, key=lambda x: -x[1])[:25]:
        tag = "*" if (a and a.strip() and is_stopword_anchor(a)) else " "
        print(f"  {tag} {n:6,}  {a!r}")

    # ---- truncation check: explanations not ending in sentence punctuation ----
    print("\n" + "=" * 90)
    print("TRUNCATION CHECK — sample of explanations not ending in . ! ? \" )")
    tr = c.execute(
        "SELECT ref_id, notes FROM cross_references WHERE source_dataset='TSK+ollama' "
        "AND notes IS NOT NULL AND TRIM(notes)!='' "
        "AND SUBSTR(TRIM(notes),-1) NOT IN ('.','!','?','\"',')') LIMIT 6"
    ).fetchall()
    for rid, n in tr:
        print(f"\n  ref {rid} (ends: ...{n[-50:]!r})")
        print("    " + "\n    ".join(textwrap.wrap(n, 86)))


if __name__ == "__main__":
    main()
