"""Tests for the /search/semantic endpoint wiring (hybrid + FTS fallback)."""

from pathlib import Path

from fastapi.testclient import TestClient

from abba.api import routes
from abba.api.app import create_app
from abba.api.semantic_search import HybridSearchResult
from abba.database.sqlite_manager import SQLiteManager


def _client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "test.db"
    SQLiteManager(db_path).initialize_database()  # migrate an empty DB
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

    monkeypatch.setattr(routes, "_get_semantic", lambda: FakeSemantic())  # noqa: PLW0108
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


def test_semantic_endpoint_falls_back_when_hybrid_raises(tmp_path: Path, monkeypatch) -> None:
    class BrokenSemantic:
        def hybrid_search(self, query_text, translation_id="engbsb", n_results=20):
            raise RuntimeError("chroma down")

    monkeypatch.setattr(routes, "_get_semantic", lambda: BrokenSemantic())  # noqa: PLW0108
    resp = _client(tmp_path).get("/api/v1/search/semantic", params={"q": "love"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
