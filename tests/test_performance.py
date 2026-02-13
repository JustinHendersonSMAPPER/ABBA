"""Performance tests for ABBA API endpoints.

Tests response times and throughput for various API operations
against the seeded test database.
"""

# pylint: disable=redefined-outer-name
import time

import pytest
from fastapi.testclient import TestClient

from abba.api.app import create_app
from abba.api.routes import configure_db
from abba.database import SQLiteManager
from abba.database.migrations import run_migrations
from abba.enrichment import BookMetadataPopulator, CrossReferencePopulator, PassagePopulator
from abba.performance import BenchmarkResult, BenchmarkSuite, ConnectionPool, run_benchmark


@pytest.fixture
def perf_client(seeded_db):
    """Create a test client with enrichment data for performance testing."""
    run_migrations(seeded_db)
    BookMetadataPopulator(seeded_db).populate()
    CrossReferencePopulator(seeded_db).populate()
    PassagePopulator(seeded_db).populate()

    db = SQLiteManager(seeded_db)
    configure_db(db)
    app = create_app()
    yield TestClient(app)


class TestBenchmarkUtilities:
    """Test the benchmark infrastructure."""

    def test_run_benchmark_measures_time(self):
        result = run_benchmark("test_sleep", lambda: time.sleep(0.001), iterations=3, target_ms=100.0)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_sleep"
        assert result.iterations == 3
        assert result.duration_ms > 0
        assert result.avg_ms > 0

    def test_benchmark_pass_detection(self):
        result = run_benchmark("fast_op", lambda: None, iterations=10, target_ms=100.0)
        assert result.passed is True

    def test_benchmark_suite_tracking(self):
        suite = BenchmarkSuite()
        r1 = BenchmarkResult(name="a", duration_ms=10.0, iterations=1, target_ms=20.0, passed=True)
        r2 = BenchmarkResult(name="b", duration_ms=50.0, iterations=1, target_ms=20.0, passed=False)
        suite.add(r1)
        suite.add(r2)
        assert suite.passed == 1
        assert suite.failed == 1
        summary = suite.summary()
        assert summary["total_benchmarks"] == 2

    def test_benchmark_result_avg_calculation(self):
        result = BenchmarkResult(name="calc", duration_ms=100.0, iterations=10)
        assert result.avg_ms == 10.0


class TestConnectionPool:
    """Test connection pool functionality."""

    def test_pool_creation(self, seeded_db):
        pool = ConnectionPool(str(seeded_db), pool_size=3)
        assert pool.pool_size == 3
        pool.close_all()

    def test_pool_acquire_release(self, seeded_db):
        pool = ConnectionPool(str(seeded_db), pool_size=2)
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        assert conn1 is not None
        assert conn2 is not None
        pool.release(conn1)
        pool.release(conn2)
        pool.close_all()

    def test_pool_acquire_beyond_size(self, seeded_db):
        pool = ConnectionPool(str(seeded_db), pool_size=1)
        conn1 = pool.acquire()
        conn2 = pool.acquire()  # Should create a new one
        assert conn1 is not None
        assert conn2 is not None
        pool.release(conn1)
        pool.release(conn2)
        pool.close_all()

    def test_pool_connection_works(self, seeded_db):
        pool = ConnectionPool(str(seeded_db), pool_size=1)
        conn = pool.acquire()
        cursor = conn.execute("SELECT COUNT(*) FROM verses")
        count = cursor.fetchone()[0]
        assert count > 0
        pool.release(conn)
        pool.close_all()


class TestAPIPerformance:
    """Test API endpoint response times."""

    def test_basic_verse_response_time(self, perf_client):
        start = time.perf_counter()
        resp = perf_client.get("/api/v1/verses/engbsb/1/1/1?depth=basic")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        # Basic retrieval should be fast (generous limit for test env)
        assert elapsed_ms < 500, f"Basic verse took {elapsed_ms:.0f}ms"

    def test_search_response_time(self, perf_client):
        start = time.perf_counter()
        resp = perf_client.get("/api/v1/search/text?q=beginning")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        assert elapsed_ms < 500, f"Text search took {elapsed_ms:.0f}ms"

    def test_book_list_response_time(self, perf_client):
        start = time.perf_counter()
        resp = perf_client.get("/api/v1/books")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        assert elapsed_ms < 500, f"Book list took {elapsed_ms:.0f}ms"

    def test_multiple_verse_requests(self, perf_client):
        """Test throughput of multiple sequential requests."""
        start = time.perf_counter()
        for vs in range(1, 6):
            resp = perf_client.get(f"/api/v1/verses/engbsb/1/1/{vs}?depth=basic")
            assert resp.status_code == 200
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 5
        assert avg_ms < 500, f"Average verse request took {avg_ms:.0f}ms"
