# Fix F1 & F2 Report

## What Was Changed

### F2 — Replace hardcoded `"engbsb"` with `DEFAULT_TRANSLATION_ID = "BSB"`

**New file:** `abba/api/constants.py`
- Defines `DEFAULT_TRANSLATION_ID = "BSB"` with module-level docstring.

**Modified files:**

| File | Change |
|------|--------|
| `abba/api/models.py` | Added `from .constants import DEFAULT_TRANSLATION_ID`; replaced `translation_id: str = "engbsb"` in `AudioResource` |
| `abba/api/routes.py` | Added `from .constants import DEFAULT_TRANSLATION_ID`; replaced 5 occurrences of `"engbsb"`: two `Query()` defaults (`/search/text`, `/search/semantic`), the multilingual fallback list, the audio endpoint `Query()` default, and both the SQL literal and the `tid=` literal in the mobile sync handler |
| `abba/api/semantic_search.py` | Added `from .constants import DEFAULT_TRANSLATION_ID`; replaced `translation_id: str = "engbsb"` in `SemanticSearchAPI.hybrid_search()` |

The mobile-sync SQL was also converted from a string literal (`WHERE v.translation_id = 'engbsb'`) to a bound parameter (`WHERE v.translation_id = ?`, `(DEFAULT_TRANSLATION_ID, bid)`).

---

### F1 — Populate the empty `books` table from the `verses` table

**New file:** `abba/database/books_populator.py`
- `populate_books(db_path: Path) -> int` — idempotent, DELETE + INSERT in one transaction.
- Derives `(translation_id, book_id, MAX(chapter))` from `verses`.
- Skips `book_id` outside 1–66.
- Uses the canonical 66-name map (`CANONICAL_BOOK_NAMES`).
- Testament: `'old'` for books 1–39, `'new'` for 40–66 (matches schema CHECK constraint).
- `book_order = book_id`, `name = common_name = canonical name`.
- Returns row count inserted.

**Modified files:**

| File | Change |
|------|--------|
| `abba/cli.py` | Added `--populate-books` argument + `should_populate_books()` method |
| `abba/main.py` | (1) Standalone `--populate-books` handler (early return after backfill); (2) automatic call to `populate_books()` after verses import completes |
| `abba/api/routes.py` | `list_books()` now filters `WHERE translation_id = ?` (bound to `DEFAULT_TRANSLATION_ID`), groups by `book_id` to deduplicate, and orders by `book_id` |

---

## Backfill Command

To populate an existing database without running the full pipeline:

```bash
uv run python -c "from pathlib import Path; from abba.database.books_populator import populate_books; print(populate_books(Path('bible_data/abba.db')))"
```

Or via the CLI flag:

```bash
uv run python abba/main.py --populate-books
```

---

## Test Outputs

All 10 tests pass (`uv run pytest tests/test_default_translation.py tests/test_books_populator.py -v`):

```
tests/test_default_translation.py::test_default_translation_id_value PASSED
tests/test_default_translation.py::test_default_translation_importable PASSED
tests/test_books_populator.py::test_populate_books_returns_count PASSED
tests/test_books_populator.py::test_populate_books_genesis_metadata PASSED
tests/test_books_populator.py::test_populate_books_john_metadata PASSED
tests/test_books_populator.py::test_populate_books_idempotent PASSED
tests/test_books_populator.py::test_populate_books_empty_db_returns_zero PASSED
tests/test_books_populator.py::test_populate_books_skips_out_of_range_book_id PASSED
tests/test_books_populator.py::test_get_books_endpoint_returns_default_translation_books PASSED
tests/test_books_populator.py::test_get_books_endpoint_empty_when_no_books_populated PASSED
```

Quality gates: `ruff check` — 0 errors, `ruff format --check` — 0 issues, `pyright` — 0 errors.

---

## `GET /api/v1/books` Response Shape

```json
[
  {
    "book_id": 1,
    "name": "Genesis",
    "common_name": "Genesis",
    "testament": "old",
    "chapter_count": 50,
    "primary_genre": null,
    "secondary_genres": [],
    "author_traditional": null,
    "date_range": null,
    "original_audience": null,
    "literary_features": [],
    "reading_context": null,
    "canonical_section": null,
    "passages": null
  }
]
```

- Filtered to `DEFAULT_TRANSLATION_ID` ("BSB") only.
- One entry per `book_id` (no duplicates across multiple translation rows).
- Ordered by `book_id` ascending.
- `chapter_count` is derived from `MAX(chapter)` in the verses table.
