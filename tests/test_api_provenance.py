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
