# Critical Findings — app non-functional with real data (2026-06-23)

Discovered during end-to-end verification (live FastAPI + Vite + Playwright) of the populated DB.
The frontend and backend were evidently built against divergent contracts and never integration-tested
end-to-end. Three breakages make core flows (reading, search) return nothing or wrong data.

## F1 — `books` table is empty → no book navigation
`GET /api/v1/books` returns `[]`. `books` table has 0 rows. The reading page's book selector can't
populate, so a new user cannot navigate at all.
- **Source of truth:** `bible.eng.db` has a `Book` table (per translation: `name`, `commonName`,
  `order` = canonical 1–66 book number, `numberOfChapters`). The import never copied it.
- **Fix:** populate `abba.db.books`. Chosen approach: derive from `abba.db.verses` (which already has
  the correct abba `translation_id` + numeric `book_id`; `number_of_chapters = MAX(chapter)`) plus a
  canonical 66-book name/testament table — self-contained, no source-DB/translation-id mapping issues.
  Integrate into the import pipeline AND backfill the existing DB.

## F2 — default `translation_id = "engbsb"` does not exist (0 verses)
Hardcoded in ~8 places (`abba/api/models.py:669`, `routes.py:365/674/2254/2450/2591/2599`,
`semantic_search.py:348`, plus frontend literals). The BSB text is actually imported under id **`BSB`**
(31,086 verses). Effect: text search returns `[]`; semantic search returns verses with empty `text`.
- **Fix:** introduce a single `DEFAULT_TRANSLATION_ID` (config-driven, default `"BSB"`) and replace the
  scattered `"engbsb"` literals. (Translation-id scheme is itself inconsistent — most are `eng_xxx`, BSB
  is bare `BSB`; default to the known-good BSB.)

## F3 — verse-route contract mismatch (reading broken end-to-end)
- Frontend: `GET /verses/{book}/{chapter}/{verse}?depth=` and `/verses/{book}/{chapter}?depth=` — book
  is a 3-letter **code** (e.g. `JHN`), **no** `translation_id`.
- Backend: `GET /verses/{translation_id}/{book_id}/{chapter}/{verse}` and
  `/verses/{translation_id}/{book_id}/{chapter}` — `translation_id` first, **numeric** `book_id`.
- So `/verses/JHN/1/1` is parsed as `translation_id=JHN, book_id=1, chapter=1` → wrong/empty.
- **Fix (planned, verified in-browser):** reconcile the contract. Decision: frontend resolves book
  codes → numeric `book_id` (via the now-populated `/books`) and calls the backend's real route with a
  `translation_id` defaulting to BSB. Keep human-readable book codes in front-end URLs; translate at
  the API boundary. Backend verse routes stay as-is (they have tests + a stable shape).

## Why this matters / status
These explain the "weak UX": it's not styling, the data flows don't connect. Fixing them is the highest
-value work and is high-confidence. Being fixed on branch `fix/core-data-and-api-contract`, verified
end-to-end with the running app. Tracked in the autonomous ledger.
