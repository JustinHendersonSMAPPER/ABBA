# Phase 0b — Frontend UX + Provenance Disclosure Plan

**Goal:** Make the weak UI materially better and wire the Phase 0a provenance + semantic backend into the UI.

**Verification (no unit-test runner exists; Playwright e2e is flaky on Windows):** every change must keep
`cd frontend && npx vue-tsc --noEmit` at **0 errors** and `npm run build` **succeeding**. A final visual
pass via the running app where practical.

**Stack:** Vue 3.4 + TS 5.9 + Vite 8 + Pinia + Vue Router. API client: `src/composables/useApi.ts`
(base `/api/v1`, dev-proxied to `http://localhost:8000`). CSS variables for theming in `src/App.vue`.

Derived from the frontend audit (P1–P12 + integration points A/B).

---

## Pass 1 — Foundation, correctness & polish (high-confidence)

1. **`useApi` loading semaphore (P3, real bug).** `src/composables/useApi.ts` — the single shared
   `loading` ref is set false by the first of several parallel calls (StudyView fires `Promise.all`).
   Make `call()` ref-counted: increment on start, decrement in `finally`, `loading=false` only at 0.
   Preserve the last error but don't clear it prematurely.
2. **Dark-mode color fixes (P7).** Replace literal `background: white` / `color: white` with
   `var(--color-surface)` / `var(--color-text)` in: `views/LifeTopicNavigator.vue` (card + search input),
   `views/ReadingPlans.vue` (card), `components/WordJourneyCard.vue`.
3. **Reading typography (P10).** `src/App.vue` `.app-main { font-size: 1.125rem }`; add per-verse block
   spacing in `ReadingPane.vue` so passages are scannable; raise verse-number contrast.
4. **Accessibility basics (P11).** `App.vue` `<div class="app-main">` → `<main>`; verse-number `<sup>`
   click targets in `ReadingPane.vue` → `<button>` (keyboard-focusable); global `:focus-visible`
   outline using `--color-accent`; ensure TranslationLens words are reachable (button or tabindex+keydown).
5. **Default reading target + skeleton (P1).** `ReadingPane.vue`: default to John 1 on first load
   (`selectedBook='JHN'`, `selectedChapter='1'`, call `loadChapter()` after books resolve); show a
   skeleton while books load.
6. **StudyView chapter-route fix (P8).** `/study/:book/:chapter` without a verse currently renders a
   `ChapterData` as if it were a verse. Redirect to `/study/:book/:chapter/1` (router or in-view guard).
7. **Reusable `LoadingState.vue` (P2).** A small spinner/skeleton component; replace the plain
   "Loading…" text in ReadingPane, StudyView, SearchResults, LifeTopicNavigator, LexiconView, ConceptExplorer.
8. **Add `"typecheck": "vue-tsc --noEmit"`** to `frontend/package.json` scripts.

## Pass 2 — Provenance disclosure + semantic-search surfacing (Phase 0b core)

9. **`getProvenance` + types.** Add to `useApi.ts`: `getProvenance(entityType, entityId)` →
   `GET /api/v1/provenance/{entityType}/{entityId}`. Add `ProvenanceData` to `src/types/api.ts`
   (`entity_type, entity_id, source, source_detail?, trust_tier, trust_rationale, generated_by?,
   grounding, confidence?, pipeline_version`).
10. **`ProvenanceChip.vue`.** Self-fetching inline component (props `entityType`, `entityId`): renders
    📚 *Sourced* (trust_tier A) or 🤖 *AI-assisted* (B) badge; click expands a popover with source,
    trust_rationale, and confidence (as %). Gracefully renders nothing on 404/null. Place beside
    enrichment renders: ContextSidebar cultural items, StudyView cross-references & cultural/historical,
    ManuscriptVariants header, ConceptExplorer concept cards.
11. **Search results value (P9, integration B).** Add `score?: number` to `SearchResult`; render a
    relevance bar/percentage; map raw `match_type` ("exact"/"semantic"/"both") to human labels;
    add a semantic-specific empty state ("try a natural-language theme like 'comfort in suffering'").

## Deferred to a later pass (documented, not silently dropped)
- Nav IA overhaul / mobile bottom-tab (P4) — bigger design change; the touch-dropdown bug fix is the
  high-confidence slice and is included; full redesign deferred.
- Onboarding re-trigger + DepthDial inline descriptions (P5, P6) — next frontend pass.
- ContextSidebar population from ReadingPane (P12) — next pass.
