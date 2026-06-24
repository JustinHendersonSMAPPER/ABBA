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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import sqlite3

    from .database.sqlite_manager import SQLiteManager


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
        rows = self.db.execute_query(f"SELECT {self._COLUMNS} FROM provenance ORDER BY entity_type, entity_id")
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
