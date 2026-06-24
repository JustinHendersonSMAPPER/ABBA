# ABBA Study UI Roadmap — Design

- **Date:** 2026-06-23
- **Status:** Draft for review
- **Owner:** Justin Henderson
- **Type:** Product/data roadmap (not a single feature spec)

---

## 1. Purpose & Main Objective

ABBA should be a Bible-study UI that helps people study scripture with three pillars:

1. **Original-language analysis** (Hebrew/Greek text, morphology, Strong's, lexicon).
2. **Trustworthy historical/cultural context.**
3. **Correlations that come _with explanations_** — not bare cross-reference lists that
   "nobody can vouch for."

**Main objective (the north star):** help someone *new to the Bible transition to deeper
learning*. The same interface must serve an everyday layperson **and** a scholar:

- A layperson sees **plain-language meaning first**.
- Depth — original language, morphology, manuscript detail, scholarly apparatus — is revealed
  **progressively, on demand, never forced**.
- The journey from *"what does this verse mean?"* → *"why does this Hebrew word matter, and how
  does it connect across scripture?"* **is the product**.

This objective is a design constraint, not just an audience description: it converts the existing
`basic / deep / scholarly` depth dial from a preference toggle into a **guided progression**.

---

## 2. Current State (after the English-only import)

Discovery (2026-06-23) across the SQLite DB, ChromaDB, the FastAPI backend, and the Vue frontend
found a wide-but-hollow system: the UI (~14 routes) and API (~65 endpoints) are largely built, but
most of the *enrichment data* they read is empty.

### Populated (real data, ready to use)

| Data | Volume | Location |
|---|---|---|
| English translations | 49 translations / 1.16M verses | `bible_data/abba.db` `verses` |
| Original-language words (Hebrew/Greek) | 425K words w/ morphology, translit, gloss, Strong's | `stepbible_verses` |
| Strong's lexicon | 14,105 entries | `lexicon` |
| Morphology codes | 2,756 | `morphology` |
| Semantic embeddings | 50,142 (verses + words), validated 100% | ChromaDB `bible_data/vectors/` |

### Empty (0 rows) — the work of this roadmap

`cross_references`, `cultural_context`, `concept_definitions`, `concept_verse_mappings`,
`word_concepts`, `semantic_concepts`, `syntax_trees`, `discourse_annotations`,
`manuscript_variants`, `semantic_domains` (Louw-Nida), `life_topics`, `reading_plans`,
`passages`/`genre_shifts`, `books`/`book_metadata`.

### Key gaps confirmed in code

- **No "why" anywhere.** Cross-references carry only an optional free-text `notes` column; no part
  of the backend generates or stores *why* two passages relate.
- **Semantic search is wired but not lit.** `/search/semantic` exists and embeddings exist, but the
  query path silently degrades to full-text search.
- **`/audio` is a stub.** Returns placeholder URLs.

---

## 3. Guiding Principles (locked with the user)

1. **Breadth-first ("light up everything").** Fill the empty tables behind the *existing* UI/API
   rather than building new screens. The bottleneck is data, not UI.
2. **Hybrid sourcing.** Ingest authoritative facts; **generate the "why"** on top of them.
3. **Trust gate (the deferral rule).** AI may *explain* trustworthy facts but must **never be the
   _source_ of facts**. Any feature whose underlying data would be untrustworthy — or could only be
   fabricated by AI — is **deferred** to a gated backlog, not shipped now.
4. **Provenance & public scrutiny.** Every enrichment row carries a complete, auditable record:
   `source` + `source_detail` (citation/version/URL), `trust_tier`, a stored **`trust_rationale`**
   (the *why-trusted* answer in plain words), `generated_by` (model id or null), `grounding` (the
   exact facts an explanation was built on), `confidence` (0.00–1.00 for LLM output), and
   `created_at` / `pipeline_version` for reproducibility. The UI exposes all of it behind a
   "**Why is this here?**" disclosure on every element (📚 *Sourced* vs 🤖 *AI-assisted*), and it is
   **exportable**. Anyone — user or critic — must be able to ask *where did this come from, is it
   trusted, and why* and get a straight answer. Without this we're back to correlations nobody can
   vouch for.
5. **Progressive disclosure.** One UI, layered depth. The on-ramp from new reader to deep study is a
   first-class concern, serving layperson and scholar from the same screens.
6. **No dead data (the strongest rule).** No correlation is ever shown to the reader unless we can
   explain *why* it exists. A cross-reference with no grounded explanation is not displayed at all —
   ever. This directly answers the original complaint about printed-Bible marginal references whose
   rationale "nobody knows." If we can't say why, we don't show it.

---

## 4. Strategy: Risk-tiered sequencing (Approach C)

Of the three sequencing options considered — (A) vertical slice per pillar, (B) horizontal
data-layer-first, (C) risk-tiered — we chose **C**. It *is* "light up everything," executed in the
correct dependency order (you cannot generate a "why" before you have the link), and it
operationalizes the trust gate directly: trustworthy facts first, grounded generation second, UI
polish third, untrustworthy/AI-only features explicitly parked.

Accepted trade-off: a few impressive-looking screens (manuscript variants, Louw-Nida domains) stay
intentionally dark until a trustworthy source is sourced — which is precisely the desired behavior.

**Refinement (the "no dead data" rule):** correlations are *explanation-gated*. Cross-reference
links are never a standalone deliverable — they are raw input to the explanation engine, and only
links that receive a grounded, guardrail-passing explanation are persisted as visible
`cross_references`. Because generation time is not a constraint (see §7, Models & compute), we
explain the **entire** candidate set rather than a top-N subset.

---

## 5. Trust-tier classification (the heart of the roadmap)

Every empty table sorts into one tier by the trust gate:

### Tier A — Ingest now (authoritative open facts)

| Feature | Source | License |
|---|---|---|
| Cross-reference **link candidates** *(staging only — never displayed unexplained, see Tier B)* | Treasury of Scripture Knowledge | Public domain |
| Hebrew + Greek **syntax trees** | MACULA / Clear-Bible treebanks | CC BY 4.0 |
| **Book metadata** (testament, chapters, genre, traditional author/date) | Compiled reference data | Open |

*(Already populated: original-language words, morphology, Strong's lexicon, embeddings.)*

### Tier B — Generate now, grounded in facts + visibly labeled

| Feature | Grounding (the "facts") | AI's role |
|---|---|---|
| **Explained correlations** *(differentiator — the only form in which cross-refs ship)* | TSK link + shared Strong's #, shared lemmas, embedding similarity, thematic overlap | Write the "why", cite the grounding; **a link with no passing explanation is discarded, not shown** |
| **Concept → verse mappings** | Strong's-anchored concordance (`StrongsConcordance` + `concepts.yaml`) | Validate/expand via Ollama |
| **Historical/cultural context** | Public-domain reference works — ISBE (1915), Easton's (1897), Smith's — keyed to people/places/terms | Summarize & link; **never invent** |
| **Semantic search** | Existing validated ChromaDB embeddings | None — just wire the query path |

**Historical-context trust rule:** PD works are well-regarded but reflect early-20th-c. scholarship.
They will be **labeled with source + date**, favored for factual/geographic/cultural entries over
interpretive theology, and flagged as replaceable later by better-licensed modern sources.

### Tier C — Defer (gated backlog: untrustworthy or AI-only facts)

| Feature | Why deferred | Entry gate to un-defer |
|---|---|---|
| **Manuscript variants** | Needs a real critical apparatus; NA28/BHS copyrighted | Identify + license an open apparatus (e.g., SBLGNT apparatus, CNTR open data) |
| **Louw-Nida semantic domains** | Official lexicon is copyrighted (UBS) | Evaluate/license an open semantic-domain substitute |
| **Discourse annotations** | OpenText.org coverage/licensing partial | Confirm coverage + license |
| **Audio** | Endpoint is a stub | Identify a real audio source |
| **Life topics / reading plans** | Editorial content; AI-draftable but needs human review | Editorial review workflow + clear "draft" labeling |

---

## 6. Phased Roadmap

Every phase must clear the existing `CLAUDE.md` gates before it is "done": **80%+ test coverage**,
`ruff format` clean, `ruff check` clean, `pyright` clean, tests passing.

### Phase 0 — Trust, provenance & audit foundation *(prerequisite)*

- Add a shared **audit/provenance record** to enrichment tables (migration): `source`,
  `source_detail` (citation/version/URL), `trust_tier`, `trust_rationale` (plain-words *why-trusted*),
  `generated_by` (model id or null), `grounding` (JSON of the facts an explanation used),
  `confidence` (0.00–1.00 for LLM output; N/A for ingested facts), `created_at`, `pipeline_version`.
- Expose the record via API and a reusable **"Why is this here?"** disclosure component (trust chip →
  expands to source, trust rationale, grounding, confidence) and make it **exportable** (the
  "open to public scrutiny" requirement).
- **Wire `/search/semantic` end-to-end** against the existing ChromaDB embeddings (low effort,
  immediate payoff — embeddings already exist and are validated).
- Define the **progressive-disclosure contract** (what `basic` vs `deep` vs `scholarly` reveals per
  screen) so later phases build against it.

**Exit criteria:** audit columns live + migrated; "Why is this here?" disclosure renders source +
rationale + confidence on a sample element; provenance is queryable via API and exportable; semantic
search returns real vector results behind a labeled UI.

### Phase 1 — Ingest Tier-A facts

- Cross-reference link **candidates** (TSK, public domain) → a **staging table**
  (`cross_reference_candidates`), *not* the user-visible `cross_references` table. These are inputs to
  Phase 2 and are never displayed on their own.
- Book metadata → `books` / `book_metadata` *(directly displayable fact)*.
- Syntax trees (MACULA) → `syntax_trees` *(directly displayable fact; larger ingestion; can run in parallel)*.

**Exit criteria:** book info and original-language/syntax screens render real data with source labels;
the cross-ref candidate staging table is populated; importer is idempotent and covered by tests.

### Phase 2 — The explanation engine + grounded context *(the differentiator)*

- Build-time pass extending `OllamaAnalyzer`: for **every** candidate cross-ref link, generate an
  explanation **grounded** in shared Strong's #, shared lemmas, embedding similarity, and theme.
  Only links whose explanation passes the grounding guardrail are written to the visible
  `cross_references` table (cached, labeled AI-assisted, grounding shown beneath the prose).
  **Links that fail are discarded — no dead data.** Full coverage; generation time is not a constraint.
- Apply the same rule to **parallel passages** (vocabulary-based): show a "why" or don't show them.
- Concept → verse mappings via the Strong's-anchored concordance + Ollama validation →
  `concept_definitions`, `concept_verse_mappings`, `word_concepts`.
- Historical/cultural context grounded in PD reference works → `cultural_context`.
- **Confidence (0.00–1.00):** every LLM explanation stores a confidence that **blends objective
  grounding signals** (count of shared Strong's #, lexical overlap, embedding-similarity score) with
  the model's self-assessment — not the self-report alone, which is unreliable. Stored, surfaced in
  the UI, and exportable.
- **Hallucination guardrails / the gate:** an explanation is promoted to a visible cross-reference
  only if it (a) cites grounding facts and (b) meets a minimum confidence threshold; otherwise the
  explanation *and its link* are discarded (the "no dead data" rule). A sampling/eval pass + the
  existing community-feedback endpoints catch drift.

**Exit criteria:** every visible cross-reference shows a plain-language "why" with its grounding cited;
zero unexplained correlations exist in the user-facing tables; concept and context panels populate
with labeled provenance; guardrail/eval pass documented.

### Phase 3 — Progressive-disclosure UX & honest empty states

- Apply trust chips and "expand the why" across all lit-up routes.
- Replace blank Tier-C panels with graceful **"not yet available — needs a trustworthy source"**
  states (never silent blanks).
- Polish the **original-language word-study on-ramp** (pillar #1 + main objective): plain meaning →
  one click to transliteration/morphology → one click to lexicon + grounded cross-scripture usage.
- Make the depth dial a *guided progression* (suggest the next layer, don't just expose toggles).

**Exit criteria:** a new reader can move from "what does this verse mean" to original-language and
explained-correlation depth without dead ends; Tier-C screens show honest, branded placeholders.

### Phase 4+ — Deferred backlog (gated)

Work the Tier-C table only as each **entry gate** is satisfied. Each item, when un-deferred, becomes
its own brainstorm → spec → plan cycle.

---

## 7. Cross-cutting concerns

- **Provenance schema** is the backbone (Phase 0) — everything else assumes it.
- **Transparency & public scrutiny:** every enrichment element answers *where from / trusted? / why /
  how confident* via a stored audit record, exposed in-UI and exportable. LLM output is reproducible
  (store model + version, prompt inputs, and grounding) so any claim can be re-derived or challenged.
- **Models & compute:** explanation generation runs on a Qwen-class model via Ollama — either Ollama
  **cloud** or **local** on 5090-class GPU(s) (single or multiple). Generation time is explicitly
  *not* a constraint, which is what makes full-coverage explanation (no top-N) feasible. All
  generation is **build-time**; no LLM runs at query time.
- **Hallucination guardrails:** grounding-citation requirement, eval sampling, community feedback.
  A correlation with no passing explanation is dropped (the "no dead data" rule).
- **Licensing diligence:** verify each source's license *before* ingestion; record it in provenance.
  Confirmed: **TSK = public domain**; **OpenBible.info = CC BY** (attribution, *not* PD) — see §9.
- **Progressive disclosure:** one component contract drives layperson↔scholar layering.
- **Performance:** generation is build-time (no LLM at query time), consistent with existing design.

---

## 8. Explicit non-goals (YAGNI for this roadmap)

- No new feature *screens* — the UI surface already exceeds the data.
- No user authentication / accounts work.
- No multilingual expansion beyond English for now.
- No offline-sync wiring (composable exists but stays unused).
- No query-time LLM (keep the build-time architecture).
- Tier-C features are **not** built now (that's the whole point of the gate).

---

## 9. Open questions / risks

- **MACULA syntax-tree ingestion size/format** — needs a spike to estimate effort.
- **Cross-ref source license (decision pending):** TSK is public domain; OpenBible.info is CC BY
  (attribution required), so it does *not* meet the "public domain only" bar literally.
  Recommendation: **TSK-only** — it is the public-domain base OpenBible.info itself derives from, and
  OpenBible's main added value (community vote-ranking) is moot now that we explain the *entire* set
  rather than a top-N. Include OpenBible.info only if CC-BY attribution is acceptable (recorded in
  provenance).
- **Ollama explanation throughput at full coverage** (~340–500K candidate links) — long but
  acceptable (time is not a constraint); needs batching, caching, idempotent resume, and a
  discard-on-fail path. *Resolved:* full coverage, no top-N prioritization.
- **Historical-context PD sources are dated** — acceptable now with labels; plan a later upgrade.

---

## 10. Success definition

A new reader opens a verse, understands it in plain language, taps once to see the Hebrew/Greek
behind a key word, taps again to see *why* a related passage connects — with every fact sourced and
every AI explanation labeled. A scholar uses the same screens to reach morphology, syntax, and
(eventually) apparatus-level depth. Nothing on screen is a correlation nobody can explain.
