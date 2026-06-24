# Deferred Decisions — ABBA

Decisions that are **not answerable with high confidence** without your input. Work continues
around these; each lists my recommended default so nothing is blocked silently. Resolve any of
these by editing the **Decision** line.

> Convention: `Status: OPEN` = needs your call. `Status: PROCEEDING (default)` = I'm building to the
> stated default and will adjust if you choose otherwise. `Status: RESOLVED` = decided.

---

## D1 — TSK cross-reference data source & license
**Question:** Which exact *public-domain* Treasury of Scripture Knowledge distribution do we ingest?
OpenBible.info's file is CC-BY (you said PD-only). Candidate PD sources need verification (raw TSK
text/XML distributions exist in several Bible-software exports and GitHub repos of varying license
hygiene).
**Why deferred:** Picking a source is a licensing judgment I shouldn't make blind. Ingesting the
wrong file would violate the PD-only rule.
**Recommended default / how I'm proceeding:** Build the importer + parser to a documented TSK
schema, fixture-tested, and a staging table — but DO NOT bulk-load until you confirm a source URL.
**Status:** PROCEEDING (default) — importer built & tested against a fixture; bulk load awaits a confirmed PD source.

## D2 — Minimum confidence threshold for the "no dead data" gate
**Question:** Below what confidence (0.00–1.00) is an explained cross-reference discarded rather than shown?
**Why deferred:** Needs empirical calibration against real LLM output, which needs Ollama running.
**Recommended default:** Start at **0.60**, make it a config value, tune after a sample generation run.
**Status:** PROCEEDING (default 0.60, config-driven).

## D3 — Theological / denominational stance of AI explanations
**Question:** Should generated "why" explanations be denominationally **neutral/descriptive**, follow a
specific tradition, or present multiple views?
**Why deferred:** High-impact on prompt design and trust; a genuine editorial/values decision that is yours.
**Recommended default:** **Denominationally neutral & descriptive** — explanations restricted to
linguistic/textual grounding (shared Strong's, shared lemma, thematic/semantic overlap, historical
fact from PD sources); avoid doctrinal claims; where traditions differ, describe rather than adjudicate.
**Status:** OPEN (building the prompt to the neutral default, easily swapped).

## D4 — LLM model & Ollama topology for the build-time generation run
**Question:** Exact model tag (e.g. a specific Qwen) and cloud vs local-on-5090 for the bulk explanation run.
**Why deferred:** An ops/environment choice; depends on your Ollama setup.
**Recommended default:** Build the generation client **model-agnostic** (model name from config/env,
Ollama base URL from config), default to a Qwen tag, batched + resumable.
**Status:** PROCEEDING (default, model-agnostic + config-driven).

## D5 — Historical/cultural context PD source & entity-linking scope
**Question:** Which PD reference works to ingest first (ISBE 1915 / Easton's / Smith's) and how granular
the entity linking (key context to people/places/terms vs. to verse ranges)?
**Why deferred:** Scope + source-quality judgment; several valid approaches.
**Recommended default:** Defer until after cross-references ship; when started, prefer ISBE (1915) for
breadth, key entries to verse ranges first (simpler), entity-level linking later.
**Status:** OPEN (not started; later phase).

## D6 — Canon scope for v1
**Question:** Protestant 66 only, or broader canon, for the first complete experience?
**Recommended default:** **Protestant 66** (matches the current English-only import); canon-aware code
already exists for later expansion.
**Status:** PROCEEDING (default 66).

## D7 — Single-user vs multi-user
**Question:** Notes/collections are currently global (no auth). Is the target single-user/local, or multi-user with accounts?
**Recommended default:** **Single-user / local** for now (roadmap non-goal: no auth). Keep data model
compatible with a later `user_id` addition.
**Status:** PROCEEDING (default single-user).

## D8 — Original-language word-study follow-ups (Pillar 1)
Pillar 1 now works end-to-end (verse → original-language word chips → lexicon). Remaining refinements,
deferred because they need careful STEP-format work or are low-confidence one-shot:
- **"Find all occurrences" of a Strong's number** (`/search/strongs/{n}` → `SQLiteManager.search_strongs`)
  still queries the empty legacy `words` table; should query `stepbible_verses` matched on the
  *normalized* Strong's (the stored `strongs_primary` is padded/prefixed, so SQL-side normalization or a
  precomputed normalized column is needed).
- **`normalize_strongs` uppercase suffix semantics:** an uppercase trailing letter is a STEP
  language/disambiguation marker (strip it) while a lowercase suffix is a homonym marker (keep it).
  A direct `strongs_primary` with an uppercase suffix is currently returned un-normalized — rare, and
  the "right" behavior is ambiguous, so left as-is.
- **Morphology descriptions / part-of-speech:** `morphology_code` is shown raw; expanding STEP morph
  codes to human descriptions needs a parser/lookup (the `morphology` table has 2,756 code defs).
- **Multi-brace compound roots** (only the first `{...}` is used) and **Aramaic** (coded as Hebrew).
**Status:** OPEN (Pillar 1 functional; these are enhancements).
