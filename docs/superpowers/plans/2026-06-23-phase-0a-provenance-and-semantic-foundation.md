# Phase 0a — Provenance & Semantic Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend foundation for "open to public scrutiny" — a central, auditable provenance store with an API — and wire the existing (but disconnected) semantic search end-to-end.

**Architecture:** A single `provenance` table keyed by `(entity_type, entity_id)` records *where data came from, whether it's trusted, why, and (for AI output) a 0.00–1.00 confidence*. A `Provenance` dataclass enforces the trust rules; a `ProvenanceStore` persists/retrieves them via the existing `SQLiteManager`; two read-only API endpoints expose and export the records. Separately, the `/search/semantic` endpoint — currently full-text-only — is routed through the already-complete `SemanticSearchAPI.hybrid_search()` with graceful FTS fallback.

**Tech Stack:** Python 3, SQLite (`sqlite3` stdlib via `SQLiteManager`), FastAPI + Pydantic, ChromaDB + Sentence-Transformers (already present), pytest, uv, ruff, pyright.

**Scope note:** This is **Phase 0a** of the roadmap (`docs/superpowers/specs/2026-06-23-abba-study-ui-roadmap-design.md`). It is backend-only and independently testable. The frontend "Why is this here?" disclosure component and the progressive-disclosure contract are **Phase 0b** (separate plan, Vue/Playwright toolchain).

## Global Constraints

- **Line length:** 120 (ruff formatter).
- **Quality gates (all must pass before "done"):** `uv run ruff format .`, `uv run ruff check .` (zero violations), `nox -s typing` (pyright zero errors on `abba/`), `nox -s tests` (pytest), **80% minimum coverage** on new/modified code.
- **Type hints:** required on every function signature in `abba/`.
- **Dependencies:** add only via `uv add` / `uv add --dev` (none expected for this plan — all libraries already present).
- **Migrations:** follow the existing idempotent pattern in `abba/database/migrations.py` (check `sqlite_master`, return `bool`, register in `_MIGRATIONS`).
- **Confidence semantics:** `confidence` is `0.00–1.00` and is **required** whenever `generated_by` is set (AI output); it is `None` for ingested facts.
- **Trust tiers:** `A` = authoritative ingested fact, `B` = AI-generated grounded in facts, `C` = deferred.

---

## File Structure

- `abba/database/migrations.py` — **Modify.** Add `add_provenance_table()` and register it in `_MIGRATIONS`.
- `abba/provenance.py` — **Create.** `TrustTier` enum, `Provenance` dataclass (with validation), `ProvenanceStore` (persistence).
- `abba/api/models.py` — **Modify.** Add `ProvenanceRecord` Pydantic response model.
- `abba/api/routes.py` — **Modify.** Add provenance singleton + 2 endpoints; add semantic singleton + rewrite `/search/semantic`.
- `tests/test_provenance_migration.py` — **Create.**
- `tests/test_provenance_model.py` — **Create.**
- `tests/test_provenance_store.py` — **Create.**
- `tests/test_api_provenance.py` — **Create.**
- `tests/test_semantic_search_endpoint.py` — **Create.**

---

## Task 1: Provenance table migration

**Files:**
- Modify: `abba/database/migrations.py` (add function near line 543, after `add_cross_references_table`; register in `_MIGRATIONS` list ~line 1392)
- Test: `tests/test_provenance_migration.py`

**Interfaces:**
- Produces: `add_provenance_table(db_path: Path) -> bool` and a `provenance` table with columns `(id, entity_type, entity_id, source, source_detail, trust_tier, trust_rationale, generated_by, grounding_json, confidence, pipeline_version, created_at)` and `UNIQUE(entity_type, entity_id)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_migration.py`:

```python
"""Tests for the provenance table migration."""

import sqlite3
from pathlib import Path

from abba.database.migrations import add_provenance_table


def test_add_provenance_table_creates_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    sqlite3.connect(db_path).close()  # create an empty database file

    assert add_provenance_table(db_path) is True

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='provenance'"
        ).fetchone()
    assert row is not None


def test_add_provenance_table_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    sqlite3.connect(db_path).close()

    add_provenance_table(db_path)
    assert add_provenance_table(db_path) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance_migration.py -v`
Expected: FAIL with `ImportError: cannot import name 'add_provenance_table'`.

- [ ] **Step 3: Write minimal implementation**

In `abba/database/migrations.py`, add this function after `add_cross_references_table` (after line 542):

```python
def add_provenance_table(db_path: Path) -> bool:
    """Add the central provenance table for auditable data attribution.

    One uniform audit record per enrichment element, keyed by
    (entity_type, entity_id): where it came from, whether it is trusted, why,
    and — for AI output — a 0.00-1.00 confidence.

    Args:
        db_path: Path to the database

    Returns:
        True if migration was needed and succeeded, False if already exists
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='provenance'")
            if cursor.fetchone()[0] > 0:
                logger.debug("provenance table already exists")
                return False

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_detail TEXT,
                    trust_tier TEXT NOT NULL CHECK(trust_tier IN ('A', 'B', 'C')),
                    trust_rationale TEXT NOT NULL,
                    generated_by TEXT,
                    grounding_json TEXT,
                    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
                    pipeline_version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entity_type, entity_id)
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_entity ON provenance(entity_type, entity_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_tier ON provenance(trust_tier)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_source ON provenance(source)")
            conn.commit()
            logger.info("Added provenance table")
            return True
    except Exception as e:
        logger.error("Failed to add provenance table: %s", e)
        raise
```

Then register it in the `_MIGRATIONS` list (after the `add_user_annotation_tables` entry, before the closing `]` near line 1392):

```python
    (add_user_annotation_tables, "user annotation tables"),
    # Phase 0a provenance foundation
    (add_provenance_table, "provenance table"),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance_migration.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add abba/database/migrations.py tests/test_provenance_migration.py
git commit -m "feat(provenance): add central provenance audit table migration"
```

---

## Task 2: Provenance dataclass + TrustTier enum

**Files:**
- Create: `abba/provenance.py`
- Test: `tests/test_provenance_model.py`

**Interfaces:**
- Produces:
  - `class TrustTier(str, Enum)` with members `AUTHORITATIVE = "A"`, `GENERATED = "B"`, `DEFERRED = "C"`.
  - `@dataclass Provenance(entity_type: str, entity_id: str, source: str, trust_tier: TrustTier, trust_rationale: str, pipeline_version: str, source_detail: Optional[str] = None, generated_by: Optional[str] = None, grounding: Dict[str, Any] = {}, confidence: Optional[float] = None)` with `__post_init__` validation and `grounding_json() -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_model.py`:

```python
"""Tests for the Provenance dataclass and TrustTier enum."""

import pytest

from abba.provenance import Provenance, TrustTier


def test_valid_ingested_fact_has_no_confidence() -> None:
    p = Provenance(
        entity_type="cross_reference",
        entity_id="42",
        source="TSK",
        trust_tier=TrustTier.AUTHORITATIVE,
        trust_rationale="Public-domain scholarly compilation (~1880), widely used standard.",
        pipeline_version="0.1.0",
    )
    assert p.confidence is None
    assert p.trust_tier is TrustTier.AUTHORITATIVE


def test_llm_output_requires_confidence() -> None:
    with pytest.raises(ValueError, match="confidence is missing"):
        Provenance(
            entity_type="cross_reference",
            entity_id="42",
            source="ollama",
            trust_tier=TrustTier.GENERATED,
            trust_rationale="Explanation grounded in 2 shared Strong's numbers.",
            pipeline_version="0.1.0",
            generated_by="qwen3.5:cloud",
        )


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        Provenance(
            entity_type="x",
            entity_id="1",
            source="ollama",
            trust_tier=TrustTier.GENERATED,
            trust_rationale="r",
            pipeline_version="0.1.0",
            generated_by="qwen",
            confidence=1.5,
        )


def test_blank_rationale_rejected() -> None:
    with pytest.raises(ValueError, match="trust_rationale"):
        Provenance(
            entity_type="x",
            entity_id="1",
            source="TSK",
            trust_tier=TrustTier.AUTHORITATIVE,
            trust_rationale="",
            pipeline_version="0.1.0",
        )


def test_grounding_json_is_stable() -> None:
    p = Provenance(
        entity_type="cross_reference",
        entity_id="1",
        source="ollama",
        trust_tier=TrustTier.GENERATED,
        trust_rationale="r",
        pipeline_version="0.1.0",
        generated_by="qwen",
        confidence=0.8,
        grounding={"b": 2, "a": 1},
    )
    assert p.grounding_json() == '{"a": 1, "b": 2}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'abba.provenance'`.

- [ ] **Step 3: Write minimal implementation**

Create `abba/provenance.py`:

```python
"""Auditable provenance records for ABBA enrichment data.

Every generated or ingested enrichment element carries a provenance record so
that anyone can ask: where did this come from, is it trusted, why, and (for
AI-generated content) how confident are we (0.00-1.00). This is the backbone of
the project's "open to public scrutiny" requirement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TrustTier(str, Enum):
    """Trust classification for an enrichment element."""

    AUTHORITATIVE = "A"  # ingested from an authoritative open dataset (a fact)
    GENERATED = "B"  # AI-generated, grounded in authoritative facts
    DEFERRED = "C"  # parked: untrustworthy or AI-only; not surfaced


@dataclass
class Provenance:
    """A complete, auditable attribution record for one enrichment element."""

    entity_type: str
    entity_id: str
    source: str
    trust_tier: TrustTier
    trust_rationale: str
    pipeline_version: str
    source_detail: Optional[str] = None
    generated_by: Optional[str] = None
    grounding: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.entity_type:
            raise ValueError("entity_type is required")
        if not self.entity_id:
            raise ValueError("entity_id is required")
        if not self.source:
            raise ValueError("source is required")
        if not self.trust_rationale:
            raise ValueError("trust_rationale is required (the 'why trusted' answer)")
        if not isinstance(self.trust_tier, TrustTier):
            self.trust_tier = TrustTier(self.trust_tier)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.generated_by is not None and self.confidence is None:
            raise ValueError("generated_by is set (AI output) but confidence is missing")

    def grounding_json(self) -> str:
        """Serialize grounding facts to deterministic JSON for storage."""
        return json.dumps(self.grounding, ensure_ascii=False, sort_keys=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance_model.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add abba/provenance.py tests/test_provenance_model.py
git commit -m "feat(provenance): add Provenance dataclass with trust/confidence validation"
```

---

## Task 3: ProvenanceStore (persistence)

**Files:**
- Modify: `abba/provenance.py` (append `ProvenanceStore`)
- Test: `tests/test_provenance_store.py`

**Interfaces:**
- Consumes: `SQLiteManager.execute_update(query, params) -> int` and `SQLiteManager.execute_query(query, params) -> list[sqlite3.Row]` (row access by column name), `Provenance`, `TrustTier` from Task 2.
- Produces: `class ProvenanceStore(db: SQLiteManager)` with `record(prov: Provenance) -> None`, `get(entity_type: str, entity_id: str) -> Optional[Provenance]`, `export_all() -> list[dict[str, Any]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_store.py`:

```python
"""Tests for ProvenanceStore persistence."""

from pathlib import Path

from abba.database.sqlite_manager import SQLiteManager
from abba.provenance import Provenance, ProvenanceStore, TrustTier


def _store(tmp_path: Path) -> ProvenanceStore:
    # SQLiteManager runs all migrations on init, creating the provenance table.
    db = SQLiteManager(tmp_path / "test.db")
    return ProvenanceStore(db)


def test_record_and_get_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    p = Provenance(
        entity_type="cross_reference",
        entity_id="100",
        source="ollama",
        trust_tier=TrustTier.GENERATED,
        trust_rationale="Grounded in shared Strong's H0430 + similarity 0.87.",
        pipeline_version="0.1.0",
        generated_by="qwen3.5:cloud",
        grounding={"shared_strongs": ["H0430"], "similarity": 0.87},
        confidence=0.82,
    )
    store.record(p)

    got = store.get("cross_reference", "100")
    assert got is not None
    assert got.confidence == 0.82
    assert got.trust_tier is TrustTier.GENERATED
    assert got.grounding["shared_strongs"] == ["H0430"]


def test_record_upserts_on_conflict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    base = {
        "entity_type": "cross_reference",
        "entity_id": "1",
        "trust_tier": TrustTier.AUTHORITATIVE,
        "trust_rationale": "r",
        "pipeline_version": "0.1.0",
    }
    store.record(Provenance(source="TSK", **base))
    store.record(Provenance(source="TSK+OpenBible", **base))

    got = store.get("cross_reference", "1")
    assert got is not None
    assert got.source == "TSK+OpenBible"
    assert len(store.export_all()) == 1


def test_get_missing_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get("cross_reference", "nope") is None


def test_export_all_returns_dicts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(
        Provenance(
            entity_type="cultural_context",
            entity_id="5",
            source="ISBE-1915",
            trust_tier=TrustTier.GENERATED,
            trust_rationale="Summarized from public-domain ISBE (1915) entry.",
            pipeline_version="0.1.0",
            generated_by="qwen3.5:cloud",
            confidence=0.7,
        )
    )
    exported = store.export_all()
    assert exported[0]["source"] == "ISBE-1915"
    assert exported[0]["confidence"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'ProvenanceStore'`.

- [ ] **Step 3: Write minimal implementation**

Append to `abba/provenance.py`. First add the imports needed for typing at the top of the file (under the existing `from typing import ...` line):

```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import sqlite3

    from .database.sqlite_manager import SQLiteManager
```

Then append the class:

```python
class ProvenanceStore:
    """Persists and retrieves Provenance records via SQLiteManager."""

    _COLUMNS = (
        "entity_type, entity_id, source, source_detail, trust_tier, "
        "trust_rationale, generated_by, grounding_json, confidence, pipeline_version"
    )

    def __init__(self, db: "SQLiteManager") -> None:
        self.db = db

    def record(self, prov: Provenance) -> None:
        """Upsert a provenance record, keyed by (entity_type, entity_id)."""
        self.db.execute_update(
            """
            INSERT INTO provenance (
                entity_type, entity_id, source, source_detail, trust_tier,
                trust_rationale, generated_by, grounding_json, confidence, pipeline_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                source=excluded.source,
                source_detail=excluded.source_detail,
                trust_tier=excluded.trust_tier,
                trust_rationale=excluded.trust_rationale,
                generated_by=excluded.generated_by,
                grounding_json=excluded.grounding_json,
                confidence=excluded.confidence,
                pipeline_version=excluded.pipeline_version
            """,
            (
                prov.entity_type,
                prov.entity_id,
                prov.source,
                prov.source_detail,
                prov.trust_tier.value,
                prov.trust_rationale,
                prov.generated_by,
                prov.grounding_json(),
                prov.confidence,
                prov.pipeline_version,
            ),
        )

    def get(self, entity_type: str, entity_id: str) -> Optional[Provenance]:
        """Return the provenance record for one element, or None."""
        rows = self.db.execute_query(
            f"SELECT {self._COLUMNS} FROM provenance WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        )
        if not rows:
            return None
        return self._row_to_provenance(rows[0])

    def export_all(self) -> List[Dict[str, Any]]:
        """Return every provenance record as a plain dict (for public scrutiny)."""
        rows = self.db.execute_query(
            f"SELECT {self._COLUMNS} FROM provenance ORDER BY entity_type, entity_id"
        )
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_provenance(row: "sqlite3.Row") -> Provenance:
        return Provenance(
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            source=row["source"],
            source_detail=row["source_detail"],
            trust_tier=TrustTier(row["trust_tier"]),
            trust_rationale=row["trust_rationale"],
            generated_by=row["generated_by"],
            grounding=json.loads(row["grounding_json"]) if row["grounding_json"] else {},
            confidence=row["confidence"],
            pipeline_version=row["pipeline_version"],
        )

    @staticmethod
    def _row_to_dict(row: "sqlite3.Row") -> Dict[str, Any]:
        return {
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "source": row["source"],
            "source_detail": row["source_detail"],
            "trust_tier": row["trust_tier"],
            "trust_rationale": row["trust_rationale"],
            "generated_by": row["generated_by"],
            "grounding": json.loads(row["grounding_json"]) if row["grounding_json"] else {},
            "confidence": row["confidence"],
            "pipeline_version": row["pipeline_version"],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance_store.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add abba/provenance.py tests/test_provenance_store.py
git commit -m "feat(provenance): add ProvenanceStore with upsert, get, and export"
```

---

## Task 4: Provenance API model + endpoints

**Files:**
- Modify: `abba/api/models.py` (add `ProvenanceRecord`)
- Modify: `abba/api/routes.py` (imports, `_AppState`, `_get_provenance`, 2 endpoints)
- Test: `tests/test_api_provenance.py`

**Interfaces:**
- Consumes: `ProvenanceStore`, `Provenance` (Task 3); `create_app`, `configure_db`, `SQLiteManager` (existing).
- Produces: `GET /api/v1/provenance/export -> List[ProvenanceRecord]`; `GET /api/v1/provenance/{entity_type}/{entity_id} -> ProvenanceRecord` (404 if missing).

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_provenance.py`:

```python
"""Tests for the provenance API endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.database.sqlite_manager import SQLiteManager
from abba.provenance import Provenance, ProvenanceStore, TrustTier


def _client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "test.db"
    store = ProvenanceStore(SQLiteManager(db_path))  # migrates + seeds
    store.record(
        Provenance(
            entity_type="cross_reference",
            entity_id="7",
            source="TSK",
            trust_tier=TrustTier.AUTHORITATIVE,
            trust_rationale="Public-domain compilation (~1880).",
            pipeline_version="0.1.0",
        )
    )
    app = create_app(db_path=db_path)  # configures routes to use db_path
    return TestClient(app)


def test_get_provenance_returns_record(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/api/v1/provenance/cross_reference/7")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "TSK"
    assert body["trust_tier"] == "A"
    assert body["confidence"] is None


def test_get_provenance_missing_returns_404(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/api/v1/provenance/cross_reference/999")
    assert resp.status_code == 404


def test_export_provenance_lists_records(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/api/v1/provenance/export")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_provenance.py -v`
Expected: FAIL — endpoints return 404 for all paths (routes not defined) / `ImportError` for `ProvenanceRecord`.

- [ ] **Step 3a: Add the response model**

In `abba/api/models.py`, confirm these imports exist at the top (add any missing): `from typing import Any, Dict, Optional` and `from pydantic import BaseModel, Field`. Then add:

```python
class ProvenanceRecord(BaseModel):
    """Auditable attribution record exposed for public scrutiny."""

    entity_type: str
    entity_id: str
    source: str
    source_detail: Optional[str] = None
    trust_tier: str
    trust_rationale: str
    generated_by: Optional[str] = None
    grounding: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    pipeline_version: str
```

- [ ] **Step 3b: Wire the routes**

In `abba/api/routes.py`:

1. Add imports near the existing imports (top of file):

```python
from ..provenance import ProvenanceStore
```

2. Add `ProvenanceRecord` to the `from .models import (...)` block (keep alphabetical-ish ordering near `PassageInfo`).

3. Add a field to `_AppState` (after `analysis_api`):

```python
    provenance_store: Optional[ProvenanceStore] = None
```

4. Add the singleton accessor after `_get_analysis()`:

```python
def _get_provenance() -> ProvenanceStore:
    """Get or create the ProvenanceStore singleton."""
    if _state.provenance_store is None:
        _state.provenance_store = ProvenanceStore(_get_db())
    return _state.provenance_store
```

5. Also set it in `configure_db()` so tests using a fresh DB pick it up. Add this line inside `configure_db`, after `_state.analysis_api = AnalysisAPI(db_manager)`:

```python
    _state.provenance_store = ProvenanceStore(db_manager)
```

6. Add the two endpoints (place near the root endpoint, e.g. after `api_root`). **`export` MUST be declared before the parameterized route.**

```python
@router.get("/provenance/export", response_model=List[ProvenanceRecord], tags=["provenance"])
async def export_provenance() -> List[ProvenanceRecord]:
    """Export every provenance record for public scrutiny."""
    store = _get_provenance()
    return [ProvenanceRecord(**rec) for rec in store.export_all()]


@router.get("/provenance/{entity_type}/{entity_id}", response_model=ProvenanceRecord, tags=["provenance"])
async def get_provenance(entity_type: str, entity_id: str) -> ProvenanceRecord:
    """Return the audit record for one enrichment element."""
    prov = _get_provenance().get(entity_type, entity_id)
    if prov is None:
        raise HTTPException(status_code=404, detail="No provenance record found")
    return ProvenanceRecord(
        entity_type=prov.entity_type,
        entity_id=prov.entity_id,
        source=prov.source,
        source_detail=prov.source_detail,
        trust_tier=prov.trust_tier.value,
        trust_rationale=prov.trust_rationale,
        generated_by=prov.generated_by,
        grounding=prov.grounding,
        confidence=prov.confidence,
        pipeline_version=prov.pipeline_version,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_provenance.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add abba/api/models.py abba/api/routes.py tests/test_api_provenance.py
git commit -m "feat(api): expose provenance records via /provenance endpoints"
```

---

## Task 5: Wire semantic search end-to-end

**Files:**
- Modify: `abba/api/routes.py` (imports, `_AppState`, `_get_semantic`, refactor `/search/semantic`)
- Test: `tests/test_semantic_search_endpoint.py`

**Interfaces:**
- Consumes: `SemanticSearchAPI.hybrid_search(query_text, translation_id, n_results) -> List[HybridSearchResult]` (fields: `book_id, chapter, verse, text, translation_id, book_name, score, match_type, semantic_similarity, explanation`); `ChromaManager(persist_path=str)`; `EmbeddingModelManager(cache_dir=str)`; `ConfigManager().get_config()` with `.vectors_path` and `.data_dir`.
- Produces: `_get_semantic() -> Optional[SemanticSearchAPI]`; `_fts_only_semantic_results(search_text, translation_id, limit) -> List[SemanticSearchResult]`; a `/search/semantic` that uses hybrid search when available and falls back to FTS.

- [ ] **Step 1: Write the failing test**

Create `tests/test_semantic_search_endpoint.py`:

```python
"""Tests for the /search/semantic endpoint wiring (hybrid + FTS fallback)."""

from pathlib import Path

from fastapi.testclient import TestClient

import abba.api.routes as routes
from abba.api.app import create_app
from abba.api.semantic_search import HybridSearchResult
from abba.database.sqlite_manager import SQLiteManager


def _client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "test.db"
    SQLiteManager(db_path)  # migrate an empty DB
    return TestClient(create_app(db_path=db_path))


def test_semantic_endpoint_uses_hybrid_when_available(tmp_path: Path, monkeypatch) -> None:
    class FakeSemantic:
        def hybrid_search(self, query_text, translation_id="engbsb", n_results=20):
            return [
                HybridSearchResult(
                    book_id=43,
                    chapter=3,
                    verse=16,
                    text="For God so loved the world...",
                    translation_id=translation_id,
                    book_name="John",
                    score=0.91,
                    match_type="both",
                    semantic_similarity=0.88,
                    explanation="Matched both text and meaning",
                )
            ]

    monkeypatch.setattr(routes, "_get_semantic", lambda: FakeSemantic())
    resp = _client(tmp_path).get("/api/v1/search/semantic", params={"q": "love"})
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["match_type"] == "both"
    assert data[0]["explanation"] == "Matched both text and meaning"


def test_semantic_endpoint_falls_back_to_fts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(routes, "_get_semantic", lambda: None)
    resp = _client(tmp_path).get("/api/v1/search/semantic", params={"q": "God"})
    # Fallback must still return a (possibly empty) list, never a 500.
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_semantic_search_endpoint.py -v`
Expected: FAIL — `AttributeError: module 'abba.api.routes' has no attribute '_get_semantic'`.

- [ ] **Step 3a: Add imports and state**

In `abba/api/routes.py`:

1. Add import near the other `.` imports:

```python
from .semantic_search import HybridSearchResult, SemanticSearchAPI
```

2. Add fields to `_AppState` (after `provenance_store`):

```python
    semantic_api: Optional[SemanticSearchAPI] = None
    semantic_unavailable: bool = False
```

- [ ] **Step 3b: Add the semantic singleton and FTS helper**

Add after `_get_provenance()`:

```python
def _get_semantic() -> Optional[SemanticSearchAPI]:
    """Build the semantic search API lazily.

    Returns None (and remembers the failure) if ChromaDB vectors or the
    embedding models are unavailable, so callers degrade gracefully to FTS.
    """
    if _state.semantic_api is None and not _state.semantic_unavailable:
        try:
            from ..embeddings.chroma_manager import ChromaManager
            from ..embeddings.model_manager import EmbeddingModelManager

            config = ConfigManager().get_config()
            chroma = ChromaManager(persist_path=str(config.vectors_path))
            models = EmbeddingModelManager(cache_dir=str(config.data_dir / "models"))
            _state.semantic_api = SemanticSearchAPI(_get_db(), chroma, models)
        except Exception:  # noqa: BLE001 - any failure means "degrade to FTS"
            _state.semantic_unavailable = True
            return None
    return _state.semantic_api


def _fts_only_semantic_results(
    search_text: str, translation_id: str, limit: int
) -> List[SemanticSearchResult]:
    """Full-text fallback when semantic search is unavailable."""
    db = _get_db()
    results: List[SemanticSearchResult] = []
    try:
        fts_rows = db.search_verses(translation_id, search_text, limit * 2)
        for rank, row in enumerate(fts_rows):
            row_keys = row.keys() if hasattr(row, "keys") else []
            results.append(
                SemanticSearchResult(
                    book_id=row["book_id"],
                    chapter=row["chapter"],
                    verse=row["verse"],
                    text=row["text"],
                    book_name=row["book_name"] if "book_name" in row_keys else "",
                    score=round(1.0 - (rank / max(len(fts_rows), 1)), 3),
                    match_type="exact",
                    explanation=f"Text match (rank {rank + 1})",
                    translation_id=translation_id,
                )
            )
    except Exception:  # noqa: BLE001, S110 - best-effort fallback; failure is non-fatal
        pass
    return results
```

- [ ] **Step 3c: Replace the endpoint body**

Replace the existing `semantic_search` endpoint body (routes.py:592–631) — keep the decorator and signature (lines 577–584) — with:

```python
    parsed = parse_query(q)
    testament_override = testament or parsed.testament_filter
    book_override = book_id or parsed.book_filter
    search_text = parsed.text or q

    semantic = _get_semantic()
    if semantic is not None:
        hybrid = semantic.hybrid_search(search_text, translation_id=translation_id, n_results=limit * 2)
        results = [
            SemanticSearchResult(
                book_id=h.book_id,
                chapter=h.chapter,
                verse=h.verse,
                text=h.text,
                book_name=h.book_name,
                score=round(h.score, 3),
                match_type=h.match_type,
                explanation=h.explanation,
                translation_id=h.translation_id,
            )
            for h in hybrid
        ]
    else:
        results = _fts_only_semantic_results(search_text, translation_id, limit)

    # Apply filters
    if testament_override:
        t = "old" if testament_override in ("old", "ot") else "new"
        allowed = set(range(1, 40)) if t == "old" else set(range(40, 67))
        results = [r for r in results if r.book_id in allowed]

    if book_override:
        results = [r for r in results if r.book_id == book_override]

    return results[:limit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_semantic_search_endpoint.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full gate and commit**

```bash
uv run ruff format .
uv run ruff check --fix .
nox -s typing
uv run pytest tests/test_provenance_migration.py tests/test_provenance_model.py tests/test_provenance_store.py tests/test_api_provenance.py tests/test_semantic_search_endpoint.py -v
git add abba/api/routes.py tests/test_semantic_search_endpoint.py
git commit -m "feat(api): route /search/semantic through hybrid search with FTS fallback"
```

---

## Self-Review

**1. Spec coverage (against Phase 0 of the roadmap spec):**
- ✅ Audit/provenance record with `source`, `source_detail`, `trust_tier`, `trust_rationale`, `generated_by`, `grounding`, `confidence`, `created_at`, `pipeline_version` → Task 1 (schema) + Task 2 (model) + Task 3 (store).
- ✅ Confidence 0.00–1.00, required for LLM output → Task 2 validation + Task 1 CHECK constraint.
- ✅ Exposed via API + exportable → Task 4 (`/provenance/{...}` + `/provenance/export`).
- ✅ Wire `/search/semantic` end-to-end with graceful fallback → Task 5.
- ⏸️ **Deferred to Phase 0b (out of scope here):** the frontend "Why is this here?" disclosure component and the progressive-disclosure contract. Noted in the Scope note.

**2. Placeholder scan:** No TBD/TODO; every code step contains complete code; every command has expected output.

**3. Type consistency:** `Provenance`, `TrustTier`, `ProvenanceStore` names are consistent across Tasks 2–4. `ProvenanceRecord` (Pydantic) is distinct from `Provenance` (dataclass) by design — API boundary vs domain object. `_get_semantic`, `_get_provenance`, `_fts_only_semantic_results` are referenced exactly as defined. `HybridSearchResult` fields used in Task 5 match `abba/api/semantic_search.py` lines 82–96.

---

## Open follow-ups (not this plan)
- **Phase 0b:** frontend trust-chip / "Why is this here?" disclosure + progressive-disclosure contract.
- **Phase 1+:** the TSK candidate-staging importer and the explanation engine will be the first real *writers* to `ProvenanceStore`.
