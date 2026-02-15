# ABBA Frontend Completion Plan: Wire All Backend APIs + UX Expert Recommendations

## Current State

- **55 backend endpoints**, all fully implemented
- **47 API methods** in `useApi.js`, 18 never called from any component
- **12 components** exist (SyntaxTreeView, DiscourseView, ManuscriptVariants accept props but no view fetches data for them independently)
- **9 routes** defined, all functional
- **StudyView** already renders SyntaxTreeView/DiscourseView/ManuscriptVariants if `verseData` contains those fields — but the depth-aware verse endpoint must actually return them at scholarly depth

The key insight: **StudyView already conditionally renders** `SyntaxTreeView`, `DiscourseView`, and `ManuscriptVariants` from `verseData` props. The gap is that the verse API may not populate those fields, AND there's no way to independently explore these features. The plan addresses both.

---

## Phase A: Enhance StudyView with Scholarly Data Fetching (Priority 1)

**Goal:** When depth=scholarly, StudyView should actively fetch syntax/discourse/variants data and merge it into the display, rather than only relying on the verse endpoint returning it.

### A1. StudyView: Fetch scholarly data at deep/scholarly depth
**File:** `frontend/src/views/StudyView.vue`

In `loadVerse()`, after the existing `Promise.all([getContext, getCrossReferences])`:
- When `depth === 'deep' || depth === 'scholarly'` AND a specific verse is selected:
  - Fetch `getSyntaxTree(book, chapter, verse)`
  - Fetch `getDiscourseUnits(book, chapter, verse)`
  - Fetch `getManuscriptVariants(book, chapter, verse)`
- Merge results into `verseData` so existing `v-if` guards render them
- Use `Promise.all` for parallel fetching (non-blocking)

### A2. StudyView: Add Concept Feedback buttons
**File:** `frontend/src/views/StudyView.vue`

When viewing a verse that appears in concept search results, show thumbs up/down feedback buttons that call `submitConceptFeedback()`. This can be a small inline component beneath cross-references:
- "Was this verse relevant to [concept]?" with Relevant / Irrelevant / Partial buttons
- Only visible at deep+ depth

### A3. StudyView: Add Audio Player
**File:** `frontend/src/views/StudyView.vue`

Add an audio play button in the study-actions bar:
- On click, call `getAudioResource(book, chapter)`
- If audio URL returned, render a native `<audio>` element with controls
- Simple, non-intrusive — appears alongside Export/Share/Bookmark buttons
- Per UX expert: ambient, not overwhelming

---

## Phase B: New Routes & Views for Unused APIs (Priority 2)

### B1. Shared Content Viewer — `/shared/:token`
**New file:** `frontend/src/views/SharedView.vue`

Currently shares are created (`createShare`) but there's no route to view them. The share URL generated in StudyView (`/share/${token}`) leads to a 404.

- New route: `{ path: '/shared/:token', component: SharedView }`
- Calls `getShare(token)` on mount
- Displays shared verse/collection/note with read-only styling
- Includes "Open in ABBA" link to the full study view
- Clean, minimal page suitable for link sharing

### B2. Semantic Domain Browser — `/domains`
**New file:** `frontend/src/views/SemanticDomainBrowser.vue`

A hierarchical browser for Louw-Nida semantic domains.

- New route: `{ path: '/domains', component: SemanticDomainBrowser }`
- On mount, calls `getSemanticDomains()` (top-level domains)
- Click a domain → calls `getSemanticDomains(parentCode)` for children
- Each domain shows words in it (expandable) via the existing `/semantic-domains/{code}/words` endpoint
- Click a word → navigates to `/lexicon/{strongs}`
- Per UX expert: **breadcrumb navigation** for domain hierarchy, card-based word display
- Add nav link under "Words" as secondary nav item or accessible from LexiconView

### B3. Community Hub — `/community`
**New file:** `frontend/src/views/CommunityView.vue`

Unified view for community contributions and concept proposals.

**Two tabs:**
1. **Contributions tab:**
   - Lists contributions via `listContributions()`
   - Filter by status (pending/approved/rejected) and book
   - "Submit Contribution" form → `createContribution(data)`
   - Review interface (for moderators) → `reviewContribution(id, decision, note)`

2. **Concept Proposals tab:**
   - Lists proposals via `listConceptProposals()`
   - "Propose a Concept" form → `createConceptProposal(data)`
   - Shows status badges (pending/approved/rejected)

Per UX expert: frame as "Help improve ABBA" — community contribution, not academic editing.

### B4. Morphology & Frequency Analysis — `/analysis`
**New file:** `frontend/src/views/AnalysisView.vue`

Exposes the `/analysis/morphology` and `/analysis/frequency` endpoints.

**Two sections:**
1. **Word Frequency:**
   - Input for Strong's pattern filter, min frequency
   - Table/chart of word frequency results
   - Click word → `/lexicon/{strongs}`

2. **Morphology Patterns:**
   - Language selector (Hebrew/Greek)
   - Pattern input (optional)
   - Results as categorized cards showing morphological patterns

Per UX expert: only visible at Study/Analyze depth — this is scholarly tooling. Add a link from the LexiconView "See frequency data" rather than top-level nav.

### B5. Multilingual Search — Add to SearchResults
**File:** `frontend/src/views/SearchResults.vue` (modify existing)

Add a 4th search mode option to the existing search dropdown:
- `<option value="multilingual">Multilingual Search</option>`
- When selected, show additional fields: source language selector, target translations (comma-separated)
- Calls `multilingualSearch(query, sourceLang, translations)`
- Results render identically to semantic search results

This avoids a whole new view — it's just another search mode in the existing search page.

---

## Phase C: Enhance Existing Views with Missing API Data (Priority 3)

### C1. ReadingPane: Passage Boundaries & Genre Shifts
**File:** `frontend/src/views/ReadingPane.vue`

When a book+chapter is loaded at standard+ depth:
- Call `getPassages(bookId, chapter)` to get pericope boundaries
- Call `getGenreShifts(bookId)` to get genre transitions
- Render passage headings (e.g., "The Sermon on the Mount") as subtle dividers between verse groups
- Show genre shift indicators if a genre change occurs within the chapter
- Per UX expert: **ambient, not intrusive** — thin horizontal rule with passage title above verse groups

### C2. ReadingPane: Audio Player
**File:** `frontend/src/views/ReadingPane.vue`

Add a play button in the reading controls bar:
- Calls `getAudioResource(selectedBook, selectedChapter)`
- Shows mini audio player below controls when audio is available
- Per UX expert: reading + listening is the #1 use case for Bible apps

### C3. ReadingPane: Click verse number → Study
**File:** `frontend/src/views/ReadingPane.vue`

Currently verse numbers are just `<sup>` elements. Make them clickable:
- Click verse number → navigate to `/study/{book}/{chapter}/{verse}`
- Per UX expert: the primary discovery path from casual reading to deep study

### C4. LexiconView: Semantic Domain Navigation
**File:** `frontend/src/views/LexiconView.vue`

The domain badges already render, but they're not clickable. Make them link to the domain browser:
- Click domain badge → `/domains?highlight={domain_code}`
- Add "Explore related words in this domain" prompt

### C5. ConceptExplorer: Concept Feedback Integration
**File:** `frontend/src/views/ConceptExplorer.vue`

When viewing a concept's verses, add small feedback buttons per verse:
- Relevant / Irrelevant / Partial
- Calls `submitConceptFeedback(conceptName, verseId, feedbackType)`
- Shows aggregated feedback summary via `getConceptFeedbackSummary()`
- Per UX expert: light touch — small thumbs up/down, not a form

### C6. Life Topic Navigator: Topic Search
**File:** `frontend/src/views/LifeTopicNavigator.vue`

Currently loads all topics on mount. Add a search bar that calls the unused `life-topics/search` endpoint:
- Add search input above topic grid
- On input, call the search endpoint for filtered results
- Falls back to full list when search is cleared

---

## Phase D: Navigation & UX Expert Recommendations (Priority 4)

### D1. Update App.vue Navigation
**File:** `frontend/src/App.vue`

Add new routes to navigation. Per UX expert guidance (**max 5 visible at once**), use a progressive nav strategy:
- **Primary nav (always visible):** Read, Search, Topics, Plans, Discover
- **"More" dropdown (click to expand):** Compare, Words, Domains, Collections, Community, Analysis
- This keeps the nav clean while making all features accessible
- Mobile: hamburger menu with all items

### D2. Add the new routes to main.js
**File:** `frontend/src/main.js`

Add routes:
```js
{ path: '/shared/:token', name: 'shared', component: () => import('./views/SharedView.vue') }
{ path: '/domains', name: 'domains', component: () => import('./views/SemanticDomainBrowser.vue') }
{ path: '/community', name: 'community', component: () => import('./views/CommunityView.vue') }
{ path: '/analysis', name: 'analysis', component: () => import('./views/AnalysisView.vue') }
```

### D3. Onboarding / First-Run Experience
**New file:** `frontend/src/components/OnboardingOverlay.vue`

Per UX expert: "no clear starting point kills adoption." On first visit:
- Show a lightweight overlay/modal with 3 steps:
  1. "Welcome to ABBA" — explains depth dial (Read/Understand/Study/Analyze)
  2. "Start with a Topic" — links to Life Topics or Reading Plans
  3. "Or just read" — dismisses to ReadingPane pre-loaded with John 1
- Uses `localStorage` flag (`abba-onboarded`) to only show once
- "Show again" accessible from a help icon in nav

### D4. Mobile Navigation Improvements
**File:** `frontend/src/App.vue`

Per UX expert ("most Bible reading happens on phones"):
- Add hamburger menu toggle for `<600px` screens
- Nav links collapse into a vertical slide-out menu
- Touch-friendly tap targets (min 44px)
- Depth dial becomes a compact dropdown on mobile

---

## Phase E: Backend Verification & E2E Tests (Priority 5)

### E1. Verify verse endpoint returns scholarly data
Check that `GET /verses/{translation}/{book}/{chapter}/{verse}?depth=scholarly` actually returns `syntax_tree`, `discourse_units`, and `manuscript_variants` fields. If not, the StudyView fetch-and-merge approach (Phase A1) handles this independently.

### E2. Add E2E tests for new views
**New files in `frontend/e2e/`:**
- `shared-view.spec.js` — tests share link viewing
- `semantic-domains.spec.js` — tests domain browsing
- `community.spec.js` — tests contribution submission/listing
- `analysis.spec.js` — tests morphology/frequency views
- `multilingual-search.spec.js` — tests multilingual search mode
- `audio-player.spec.js` — tests audio playback UI
- `onboarding.spec.js` — tests first-run experience

### E3. Update existing E2E tests
- `study-features.spec.js` — add tests for scholarly depth data loading
- `reading-pane.spec.js` — add tests for passage boundaries, verse click-to-study
- `search.spec.js` — add test for multilingual search mode

---

## Implementation Order

| Step | Phase | Description | New Files | Modified Files |
|------|-------|-------------|-----------|----------------|
| 1 | A1 | StudyView scholarly fetching | 0 | StudyView.vue |
| 2 | C3 | ReadingPane verse → study links | 0 | ReadingPane.vue |
| 3 | C1 | ReadingPane passage boundaries | 0 | ReadingPane.vue |
| 4 | B1 | SharedView route | 1 | main.js |
| 5 | B5 | Multilingual search mode | 0 | SearchResults.vue |
| 6 | C6 | Topic search integration | 0 | LifeTopicNavigator.vue |
| 7 | A3+C2 | Audio player (Study + Reading) | 1 (AudioPlayer component) | StudyView.vue, ReadingPane.vue |
| 8 | B2 | Semantic Domain Browser | 1 | main.js |
| 9 | C4 | LexiconView domain links | 0 | LexiconView.vue |
| 10 | C5+A2 | Concept feedback buttons | 0 | ConceptExplorer.vue, StudyView.vue |
| 11 | B3 | Community Hub | 1 | main.js |
| 12 | B4 | Analysis View | 1 | main.js |
| 13 | D1 | Nav restructure (More dropdown) | 0 | App.vue |
| 14 | D2 | Register new routes | 0 | main.js |
| 15 | D3 | Onboarding overlay | 1 | App.vue |
| 16 | D4 | Mobile nav improvements | 0 | App.vue |
| 17 | E1-3 | E2E tests | 7 | 3 existing test files |

**Total: 6 new Vue files, 13 new E2E test files, ~10 modified files**

---

## API Coverage After Plan Completion

| API Method | Used Before | Used After | Where |
|-----------|:-----------:|:----------:|-------|
| `getSyntaxTree()` | No | Yes | StudyView (scholarly depth) |
| `getDiscourseUnits()` | No | Yes | StudyView (scholarly depth) |
| `getBookDiscourse()` | No | Yes | StudyView (book-level context) |
| `getManuscriptVariants()` | No | Yes | StudyView (scholarly depth) |
| `getSignificantVariants()` | No | Yes | CommunityView (notable variants list) |
| `multilingualSearch()` | No | Yes | SearchResults (4th search mode) |
| `createContribution()` | No | Yes | CommunityView |
| `listContributions()` | No | Yes | CommunityView |
| `reviewContribution()` | No | Yes | CommunityView |
| `createConceptProposal()` | No | Yes | CommunityView |
| `listConceptProposals()` | No | Yes | CommunityView |
| `submitConceptFeedback()` | No | Yes | ConceptExplorer, StudyView |
| `getAudioResource()` | No | Yes | ReadingPane, StudyView |
| `mobileSync()` | No | Yes | (Deferred — PWA/native scope) |
| `getPassages()` | No | Yes | ReadingPane |
| `getGenreShifts()` | No | Yes | ReadingPane |
| `getSemanticDomains()` | No | Yes | SemanticDomainBrowser |
| `getShare()` | No | Yes | SharedView |

**Result: 17 of 18 unused methods now wired up.** Only `mobileSync()` is deferred (requires PWA/native app infrastructure).

---

## UX Expert Principles Applied

1. **Progressive disclosure** — scholarly features only at deep/scholarly depth
2. **Max 5 nav items visible** — "More" dropdown for secondary features
3. **Lead with "so what"** — audio player, passage context, life topics front and center
4. **Mobile-first** — hamburger nav, touch targets, responsive domain browser
5. **Enrich, never undermine** — manuscript variants framed as "textual tradition richness"
6. **Clear starting point** — onboarding overlay guides new users
7. **Community framing** — "Help improve ABBA" not "submit academic corrections"
