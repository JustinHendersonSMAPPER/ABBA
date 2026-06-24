# fix-f3-report: Frontend API Contract Reconciliation

**Branch:** fix/core-data-and-api-contract
**Date:** 2026-06-23

---

## Changes Made

### 1. `frontend/src/types/api.ts`

- **Added `VerseResponse` interface** matching the real backend shape:
  `reference`, `book_name`, `chapter`, `verse`, `text`, `translation_id` (always present);
  `words`, `richness_flags`, `cultural_context`, `cross_references`, `literary_structures`,
  `manuscript_variants`, `syntax_tree`, `discourse_units`, `semantic_domains`, `concepts`,
  `surrounding_context`, `speaker`, `genre`, `parallel_passages` (optional/nullable).
- **Added `RichnessFlag`, `CulturalNote`, `CrossRef` interfaces** (used by `VerseResponse`).
- **Fixed `BookInfo`** from `{ id: string; name: string; chapters: number; testament?: string }`
  to `{ book_id: number; name: string; common_name?: string; testament: string; chapter_count: number; ... }`
  matching `abba/api/models.py:BookInfo`.
- **Fixed `SearchResult.book_id`** from `string` to `number`.
- Retained legacy `VerseData`, `ChapterData`, `ContextData`, `CrossReference` interfaces for
  backward compatibility with components not in scope (they are no longer imported by callers).

### 2. `frontend/src/composables/useApi.ts`

- **Exported `DEFAULT_TRANSLATION = 'BSB'`** constant at module level.
- **`getVerse` signature changed** from `(book: string, chapter, verse, depth?)` to
  `(translationId: string, bookId: number, chapter, verse, depth?)`.
  New URL: `/verses/${translationId}/${bookId}/${chapter}/${verse}?depth=${depth}`.
  Return type: `VerseResponse | null`.
- **`getChapter` signature changed** from `(book: string, chapter, depth?)` to
  `(translationId: string, bookId: number, chapter, depth?)`.
  New URL: `/verses/${translationId}/${bookId}/${chapter}?depth=${depth}`.
  Return type: `VerseResponse[] | null`.
- **Removed `getCrossReferences`** (was calling `/verses/.../cross-references` — endpoint does not exist).
- **Removed `getContext`** (was calling `/verses/.../context` — endpoint does not exist).
- **Fixed `getAudioResource`** default `translationId` from `'engbsb'` to `DEFAULT_TRANSLATION`.
- **`getSyntaxTree` return type** updated from `VerseData['syntax_tree']` to `VerseResponse['syntax_tree']`.
- `DEFAULT_TRANSLATION` added to the `UseApiReturn` interface and return object.
- Removed unused imports: `ChapterData`, `ContextData`, `CrossReference`, `VerseData`.

### 3. `frontend/src/views/ReadingPane.vue`

- Book `<option>` now uses `:key="book.book_id" :value="book.book_id"`.
- `selectedBook` ref type changed from `string` to `number` (default `0`).
- `selectedChapter` ref type changed from `string` to `number` (default `0`).
- `chapterData` ref replaced with `chapterVerses: ref<VerseResponse[]>([])`.
- Default book: `books.value.find(b => b.book_id === 43) || books.value[0]` (John = 43).
- `onBookChange()` reads `book.chapter_count` (was `book.chapters`).
- `loadChapter()` calls `api.getChapter(api.DEFAULT_TRANSLATION, selectedBook.value, selectedChapter.value, depth)`.
- Template iterates `chapterVerses` using `v.verse` (number) and `v.text` fields.
- Verse-link navigates to `/study/${selectedBook}/${selectedChapter}/${v.verse}`.
- `LiteraryModeIndicator` now reads from `chapterVerses[0]?.genre` / `chapterVerses[0]?.literary_structures`.
- `richness_flags` mapped correctly from `RichnessFlag` objects.

### 4. `frontend/src/views/StudyView.vue`

- `verseData` ref type changed from `VerseData | null` to `VerseResponse | null`.
- Removed `contextData` and `crossRefs` local refs (data now read from embedded `VerseResponse` fields).
- `loadVerse()` calls `api.getVerse(api.DEFAULT_TRANSLATION, Number(book.value), chapter.value, verse.value, depth.value)`.
- **Removed calls to `api.getContext()` and `api.getCrossReferences()`** — non-existent endpoints.
- Context section reads `verseData.cultural_context` (empty today → panel hidden by `v-if`).
- Cross-references section reads `verseData.cross_references` using `CrossRef.book_id` (numeric).
- Cross-ref router-links now: `/study/${ref.book_id}/${ref.chapter}/${ref.verse}`.
- Scholarly fetch still calls `getSyntaxTree`, `getDiscourseUnits`, `getManuscriptVariants` with string book_id (those endpoints take numeric book_id — passed from route param string which encodes the number).
- `doExport()` changed from `'engbsb'` to `api.DEFAULT_TRANSLATION`.
- `primary_concept` reference removed — replaced with `concepts[0].name` lookup.
- `parallel_translations` section replaced with `parallel_passages`.
- `contextStore.setContext` calls cast `CulturalNote`/`CrossRef` arrays to `Record<string, unknown>[]` to satisfy store's index-signature requirement.

### 5. `frontend/src/views/SearchResults.vue`

- `resultLink()` simplified: reads `r.book_id` (now `number | undefined`) directly instead of
  casting through `Record<string, unknown>`.

### 6. `frontend/src/views/TranslationCompare.vue`

- `selectedTranslations` default changed from `['engbsb', 'engkjv']` to `['BSB']`.
- `availableTranslations` entry `{ id: 'engbsb', ... }` changed to `{ id: 'BSB', ... }`.
- Book `<option>` uses `:key="book.book_id" :value="String(book.book_id)"`.
- `onBookChange()` reads `book.chapter_count` (was `book.chapters`) and finds by `String(b.book_id)`.
- `onMounted` getBooks handling updated to match array vs object pattern.

---

## Calls Changed / Removed

| Call | Before | After |
|---|---|---|
| `getVerse` URL | `/verses/{book}/{ch}/{v}?depth=` | `/verses/BSB/{bookId}/{ch}/{v}?depth=` |
| `getChapter` URL | `/verses/{book}/{ch}?depth=` | `/verses/BSB/{bookId}/{ch}?depth=` |
| `getContext` | called (endpoint 404) | **REMOVED** |
| `getCrossReferences` | called (endpoint 404) | **REMOVED** |
| `getAudioResource` default | `'engbsb'` | `'BSB'` |
| `exportVerse` translation arg | `'engbsb'` | `api.DEFAULT_TRANSLATION` |

---

## Verification

```
vue-tsc --noEmit  →  0 errors
npm run build     →  ✓ built in 449ms, 0 errors
```
