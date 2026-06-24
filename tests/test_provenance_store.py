"""Tests for ProvenanceStore persistence."""

from pathlib import Path

from abba.database.sqlite_manager import SQLiteManager
from abba.provenance import Provenance, ProvenanceStore, TrustTier


def _store(tmp_path: Path) -> ProvenanceStore:
    # SQLiteManager runs all migrations on init, creating the provenance table.
    db = SQLiteManager(tmp_path / "test.db")
    db.initialize_database()
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
    assert exported[0]["trust_tier"] == "B"
    assert exported[0]["grounding"] == {}
