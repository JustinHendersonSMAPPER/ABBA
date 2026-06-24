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
**Status:** ✅ **RESOLVED** (merged `d45d0cf`). Found + verified the **CrossWire SWORD "TSK" module**
— the original ~1880 Treasury of Scripture Knowledge, `DistributionLicense=Public Domain` (NOT the
CC-BY OpenBible compilation, NOT the copyrighted "enhanced" TSKe). Downloaded to
`bible_data/sources/TSK.zip` (gitignored; re-fetch from
`https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/TSK.zip`). The importer
(`abba/database/tsk_importer.py` + `abba/sources/tsk.py`, via pysword/MIT) loaded **578,350**
grounded cross-references (source verse → target verse + **anchor phrase**) into
`cross_reference_candidates`. Validated: Gen 1:1→John 1:1/Heb 1:10; John 3:16→Gen 22:12 (anchor "gave").

## D2 — Minimum confidence threshold for the "no dead data" gate
**Question:** Below what confidence (0.00–1.00) is an explained cross-reference discarded rather than shown?
**Why deferred:** Needs empirical calibration against real LLM output, which needs Ollama running.
**Recommended default:** Start at **0.60**, make it a config value, tune after a sample generation run.
**Status:** ✅ **IN USE** (engine merged). Threshold **0.60**, env `ABBA_XREF_CONFIDENCE`. Confidence =
0.7 (TSK anchor present) + min(0.3, 0.1·shared-Strong's), capped — so anchored links pass and
unanchored ones (can't be grounded) are dropped. On the John 3:16 sample, 19/19 passed. Tune the env if you want a stricter gate.

## D3 — Theological / denominational stance of AI explanations
**Question:** Should generated "why" explanations be denominationally **neutral/descriptive**, follow a
specific tradition, or present multiple views?
**Why deferred:** High-impact on prompt design and trust; a genuine editorial/values decision that is yours.
**Recommended default:** **Denominationally neutral & descriptive** — explanations restricted to
linguistic/textual grounding (shared Strong's, shared lemma, thematic/semantic overlap, historical
fact from PD sources); avoid doctrinal claims; where traditions differ, describe rather than adjudicate.
**Status:** ✅ **IN USE** (engine merged) at the neutral/descriptive default — the prompt explicitly
asks for "plain, denominationally-neutral" explanations grounded ONLY in the shared idea + verse texts,
"no doctrinal claims beyond the texts". Sample output is on-spec (e.g. John 3:16→Gen 22:12 describes the
shared 'giving an only son' theme without adjudicating doctrine). Still OPEN for your review if you want a
different stance — it's a one-line prompt change in `abba/semantic/cross_ref_explainer.py::build_prompt`.

## D4 — LLM model & Ollama topology for the build-time generation run
**Question:** Exact model tag (e.g. a specific Qwen) and cloud vs local-on-5090 for the bulk explanation run.
**Why deferred:** An ops/environment choice; depends on your Ollama setup.
**Recommended default:** Build the generation client **model-agnostic** (model name from config/env,
Ollama base URL from config), default to a Qwen tag, batched + resumable.
**Status:** ✅ **IN USE** with a caveat. The engine is model-agnostic (env `ABBA_OLLAMA_MODEL` /
`ABBA_OLLAMA_URL`, resumable). **The cloud `qwen3.5:397b-cloud` returned "this model requires a
subscription, upgrade for access"** — this Ollama instance isn't authenticated to the subscription
(needs `ollama signin` on the box). So I downloaded + defaulted to **local `qwen2.5:14b`** (clean
instruct, fits the 5090, ~1.3s/explanation warm) and validated quality. **To use the 397B cloud model
for the full run, sign Ollama into the subscription account, then set `ABBA_OLLAMA_MODEL=qwen3.5:397b-cloud`.**
Full run command (sequential): `uv run python -c "from abba.semantic.cross_ref_explainer import generate_explanations; print(generate_explanations('bible_data/abba.db'))"` (idempotent/resumable; ~578K candidates).

**FULL RUN LAUNCHED 2026-06-24 (~18:01) via a concurrent driver** `claude/run_full_xref.py` (10 workers, thread-local DB handles, reuses the engine; pre-filters the 20.7% anchor-less candidates so they skip the LLM unless 3+ shared Strong's). Detached (`Start-Process`, PID in `claude/xref_full.pid`, logs `claude/xref_full.out.log`/`.err.log`). **Bottleneck: `OLLAMA_NUM_PARALLEL=4`** (a persistent User env var the box owner set) caps the 5090 at 4 concurrent generations → ~2.3 LLM/s → **ETA ~2.3 days** for the 458,823 anchored candidates. Left the user's deliberate Ollama config untouched (they said time doesn't matter). **Optional ~2x speedup:** set `OLLAMA_NUM_PARALLEL=8`–12, restart Ollama; the run auto-accelerates on resume (idempotent skip-existing). Monitor: `Get-Content claude/xref_full.out.log -Tail 3` (logs every 500). Resume after any interruption: re-run the same `Start-Process` command.

## D5 — Historical/cultural context PD source & entity-linking scope
**Question:** Which PD reference works to ingest first (ISBE 1915 / Easton's / Smith's) and how granular
the entity linking (key context to people/places/terms vs. to verse ranges)?
**Why deferred:** Scope + source-quality judgment; several valid approaches.
**Recommended default:** Defer until after cross-references ship; when started, prefer ISBE (1915) for
breadth, key entries to verse ranges first (simpler), entity-level linking later.
**Status:** OPEN — researched (2026-06-24), **NOT a clean high-confidence one-shot like TSK was.**
Source landscape: Easton's (1897), ISBE (1915), Smith's are public domain and available as CrossWire
SWORD dictionary modules — but those are **entry-keyed** (headword → article: "Bethlehem", "Pharisee"),
NOT verse-keyed. Mapping them to per-verse `cultural_context` requires **entity-linking** (identify the
people/places/terms in each verse, disambiguate, attach the right article) — an accuracy-sensitive NER
problem; wrong context is worse than none (trust-gate). The verse-keyed PD sources that DO exist
(Barnes, JFB, Matthew Henry commentaries) are **interpretive/doctrinal**, which conflicts with the
neutral-factual ethos (D3). **Recommended approach when tackled:** ingest a PD dictionary (Easton's is
small/clean) as entry articles; do conservative entity-linking (match the verse's proper nouns +
key terms to headwords, only high-confidence exact matches), and have the LLM *summarize* the matched
article (never invent), each with provenance + the matched entity as grounding. This is its own
brainstorm→build cycle, not a quick win — left for explicit go-ahead.

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
- ~~**"Find all occurrences" of a Strong's number**~~ — **DONE** (merged `faad8bd`): added an indexed
  `stepbible_verses.lexical_strongs` column (precomputed normalized key) + populator; `search_strongs`
  queries it; `/search/strongs` returns canonically-ordered occurrences with full book names + BSB text;
  the Word Study page shows "Appears in N verses" + a linked occurrence list.
- **`normalize_strongs` uppercase suffix semantics:** an uppercase trailing letter is a STEP
  language/disambiguation marker (strip it) while a lowercase suffix is a homonym marker (keep it).
  A direct `strongs_primary` with an uppercase suffix is currently returned un-normalized — rare, and
  the "right" behavior is ambiguous, so left as-is.
- **Morphology descriptions / part-of-speech:** `morphology_code` is shown raw; expanding STEP morph
  codes to human descriptions needs a parser/lookup (the `morphology` table has 2,756 code defs).
- **Multi-brace compound roots** (only the first `{...}` is used) and **Aramaic** (coded as Hebrew).
**Status:** OPEN (Pillar 1 functional; these are enhancements).
