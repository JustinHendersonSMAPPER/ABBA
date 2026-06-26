# D5 — Historical / Cultural Context via PD Dictionary Entity-Linking

**Status:** ✅ **v1 BUILT & LIVE (GPU-free), 2026-06-25.** Ingest + `ref_target` linking + display
all shipped — Easton's (3,961 entries) ingested, **23,063 source-vouched links across 12,551 verses**,
surfaced in StudyView's "Historical Context" section with a Tier-A "📚 Sourced" provenance chip. The
**LLM-summary stage is deliberately deferred** (it needs the GPU, which the cross-ref run is using);
v1 shows the verbatim PD article (a fact) directly, so no AI is required for trustworthy context.
**Decision reference:** [`deferred-decisions.md` D5](../deferred-decisions.md).
**Author:** groundwork pass + v1 build, 2026-06-25.

**Populate (idempotent):** `import_easton_entries(db, "bible_data/sources/Easton.zip")` →
`link_dictionary_entries(db)` (see `claude/d5_ingest.py` + `claude/d5_link.py`). Re-fetch Easton.zip
from the CrossWire URL in §2.

---

## 1. Goal

Surface trustworthy historical/cultural context for a verse without inventing anything, and without
the doctrinal slant that verse-keyed PD commentaries (Barnes/JFB/Matthew Henry) carry. We do this by
ingesting a **public-domain Bible dictionary** as entry articles (headword → article), then linking a
verse's proper nouns / key terms to those headwords using **high-confidence exact matches only**. An
LLM later *summarizes* a matched article (never fabricates), and every surfaced item carries
provenance naming the matched entity as its grounding.

This mirrors the proven TSK cross-reference pipeline: a clean PD source → a staging/entry table →
conservative grounding → optional LLM step that only ever paraphrases facts → provenance + trust-gate.

## 2. Source selection (PD-only)

**Chosen first source: Easton's Bible Dictionary (1897).**

- **Public domain:** M.G. Easton died 1894; the 3rd edition (Thomas Nelson, 1897) is PD worldwide.
- **Exact artifact:** the CrossWire SWORD **"Easton"** module — the same provider/family as the TSK
  module we already trust (D1). Its `mods.d/easton.conf` declares:
  - `DistributionLicense=Public Domain`
  - `About=... Public Domain -- Copy Freely ... M.G. Easton ... Illustrated Bible Dictionary, Third
    Edition, published by Thomas Nelson, 1897.`
  - `TextSource=CCEL`, `SourceType=TEI`, `ModDrv=zLD`
- **Download URL (gitignored target `bible_data/sources/Easton.zip`):**
  `https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/Easton.zip`
- **Why Easton over ISBE (1915) first:** Easton is small (~3,963 entries), clean, single-author, and
  already TEI-structured — ideal to validate the linking approach end-to-end before scaling to ISBE's
  breadth. ISBE remains the next source once the pipeline is proven; the schema is source-agnostic
  (`source` column), so adding ISBE is additive.

### 2a. Format note — why a bespoke parser (not pysword)

TSK ships as a SWORD `zCom` (commentary) module, which pysword reads. Easton ships as a SWORD **`zLD`
(compressed dictionary)** module, and **pysword does not implement the dictionary driver**. So
`abba/sources/easton.py` decodes the `zLD` container directly. The format (validated against the real
3,963-entry module):

| File | Meaning |
|------|---------|
| `easton.idx` | N × 8 bytes `(dat_offset, dat_size)`, one per entry, sorted by headword. |
| `easton.dat` | per-entry key record: `KEY` + `\r\n` + `\0` + 4-byte running entry index. |
| `easton.zdx` | M × 8 bytes `(zdt_offset, zdt_size)`, one per compressed block. |
| `easton.zdt` | M zlib blocks. Each decompressed block = `uint32 count` + `count × (offset,size)` + article payloads, in global entry order. |

Concatenating blocks in order yields every entry in order, so the parser walks blocks/sub-entries
sequentially and reads the headword straight from each article's `<entryFree n="...">` / `<title>`.
Articles are TEI; verse citations appear as `<ref osisRef="Bible:Book.C.V">`, which the parser also
extracts (`ref_targets`) — a free, author-supplied verse signal we can use as a *secondary* link cue.

## 3. Data model

`dictionary_entries` (added to the schema layer in `migrations.py`, **deliberately not registered in
`_MIGRATIONS`** so it is only created when a dictionary is actually ingested — never auto-applied to
the live DB by a stray `run_migrations`):

```sql
CREATE TABLE dictionary_entries (
    entry_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    headword            TEXT NOT NULL,        -- verbatim, e.g. "Zuph, Land of"
    headword_normalized TEXT NOT NULL,        -- exact-match key, e.g. "ZUPH LAND OF"
    article             TEXT NOT NULL,        -- plain text, TEI stripped
    ref_targets         TEXT,                 -- JSON array of OSIS refs the article cites
    source              TEXT NOT NULL,        -- "Easton's Bible Dictionary (1897)"
    license             TEXT NOT NULL,        -- "Public Domain"
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, headword)                  -- idempotent re-import
);
CREATE INDEX idx_dict_entries_headword ON dictionary_entries(headword_normalized);
CREATE INDEX idx_dict_entries_source   ON dictionary_entries(source);
```

`headword_normalized` (uppercase, parentheticals removed, punctuation→space, whitespace collapsed) is
the exact-match key the linker will join against. Storing it precomputed keeps linking a cheap indexed
equality lookup rather than per-query string munging.

## 4. Pipeline

```
Easton.zip (zLD, PD)
   └─ abba/sources/easton.py        iter_easton_entries() -> DictionaryEntry(headword, article, ref_targets)
        └─ abba/database/easton_importer.py
              ├─ INSERT OR IGNORE into dictionary_entries (idempotent)
              └─ provenance: Tier A (AUTHORITATIVE) per entry, confidence = NULL (it is a fact)

[FUTURE — not built yet]
   verse proper nouns / key terms
        └─ exact-match linker  (headword_normalized == candidate term, high confidence ONLY)
              └─ verse_dictionary_links  (verse, entry_id, match_term, confidence, match_method)
                    └─ LLM summarizer  (summarize the MATCHED article only; never invent)
                          └─ cultural_context row + provenance Tier B (GENERATED, grounded in entry_id)
                                └─ trust-gate (D2 threshold) decides surface vs. drop
```

### 4a. Ingest (built)

`import_easton_entries(db_path, zip_path)` is idempotent (`INSERT OR IGNORE` on `UNIQUE(source,
headword)`) and writes one **Tier A** provenance record per entry (`entity_type='dictionary_entry'`,
`trust_tier='A'`, `confidence=NULL`, `generated_by=NULL`, grounding = the headword + cited refs). Tier
A because the article is verbatim PD text — a fact ingested unaltered, not an inference.

### 4b. Linking (future — conservative, exact-only)

Candidate terms for a verse come from (in priority order):
1. **Proper nouns** in the verse (people/places) — the highest-precision signal.
2. **Author-supplied back-references:** an Easton entry whose `ref_targets` include this verse is a
   strong, source-vouched link (the dictionary itself points at the verse).
3. Key domain terms (e.g. "phylactery", "Sabbath") — only when they exact-match a headword.

Match rule: `normalize(term) == headword_normalized` **exactly**. No fuzzy/stemmed/substring matching
in v1 — a wrong context is worse than none (see §5). A match record stores the term, the method
(`proper_noun` / `ref_target` / `key_term`), and a confidence; only matches at/above the D2 threshold
are eligible to be summarized.

### 4c. LLM summary (future — paraphrase only)

The LLM receives **only the matched article text** and is instructed to summarize/condense it for the
verse's context, denominationally neutral (D3), with an explicit "do not add facts not in the
provided article" instruction — identical discipline to the cross-ref explainer. Output is **Tier B
(GENERATED)** provenance with `generated_by=<model>`, `confidence` in [0,1], and
`grounding={dictionary_entry_id, headword, source}` so the exact PD article backing the summary is
always auditable.

## 5. Open risks & how the trust-gate handles them

| Risk | Mitigation |
|------|------------|
| **Disambiguation** — many headwords are ambiguous ("Mary", "Antioch", "Herod"). | v1 only links *unambiguous* exact matches; ambiguous headwords (multiple entries normalize equal, or a term maps to several people/places) are **not** auto-linked — they're deferred (Tier C) for editorial review rather than guessed. The `ref_target` method sidesteps ambiguity entirely: the dictionary already chose the verse. |
| **False matches** — a common word coincidentally equals a headword ("A", "Ass", "Bread"). | Restrict candidates to proper nouns + curated key terms; never link generic vocabulary. Exact-match-only eliminates fuzzy false positives. A stop-headword list filters trivially-common single-letter / common-word entries. |
| **Wrong context surfaced** | Two gates: (a) the linker confidence must clear the D2 threshold; (b) the LLM summary is itself gated and Tier-B — if it can't be grounded in the matched article it's dropped, not shown (no-dead-data). |
| **Source drift / provenance** | Every entry and every summary carries a provenance row naming the PD source, the matched entity, the model (for summaries), and confidence. `ProvenanceStore.export_all()` makes the whole chain publicly auditable. |
| **Encoding artifacts** | The CCEL source has stray CP1252 bytes despite the UTF-8 declaration; the parser decodes UTF-8 → CP1252 → lossy, so no entry is ever dropped on a decode error. |

## 6. Deliverables in this groundwork pass

- `abba/sources/easton.py` — zLD parser → `DictionaryEntry(headword, article, ref_targets)`.
- `abba/database/migrations.py::add_dictionary_entries_table` — schema (not auto-registered).
- `abba/database/easton_importer.py` — idempotent import + Tier-A provenance.
- `tests/test_easton.py` — self-contained (synthetic zLD fixture in `tmp_path`, no network/live-DB/LLM).

## 7. Not done / needs a decision

- **Linker + LLM-summary stages** are specified but unbuilt (needs Ollama + its own build cycle).
- **`verse_dictionary_links` table + proper-noun extraction** are sketched in §4b, not implemented.
- **Bulk-load to the live DB** is intentionally not performed here. To ingest after review:
  download `Easton.zip` to `bible_data/sources/`, then
  `import_easton_entries("bible_data/abba.db", "bible_data/sources/Easton.zip")`.
- **Second source (ISBE 1915)** deferred until the Easton pipeline is proven end-to-end.
