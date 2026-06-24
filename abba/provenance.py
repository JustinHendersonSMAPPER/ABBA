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
