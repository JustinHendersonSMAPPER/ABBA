"""Performance benchmarking and profiling utilities for ABBA."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    duration_ms: float
    iterations: int = 1
    avg_ms: float = 0.0
    target_ms: float = 0.0
    passed: bool = True

    def __post_init__(self) -> None:
        if self.iterations > 0:
            self.avg_ms = self.duration_ms / self.iterations


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: List[BenchmarkResult] = field(default_factory=list)
    total_ms: float = 0.0
    passed: int = 0
    failed: int = 0

    def add(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
        self.total_ms += result.duration_ms
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict."""
        return {
            "total_benchmarks": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "total_ms": round(self.total_ms, 2),
            "results": [
                {
                    "name": r.name,
                    "avg_ms": round(r.avg_ms, 2),
                    "target_ms": r.target_ms,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


def run_benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 10,
    target_ms: float = 100.0,
) -> BenchmarkResult:
    """Run a benchmark function multiple times and measure performance.

    Args:
        name: Name of the benchmark.
        func: Function to benchmark (called with no arguments).
        iterations: Number of iterations to run.
        target_ms: Target average time in milliseconds.

    Returns:
        BenchmarkResult with timing data.
    """
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = (time.perf_counter() - start) * 1000

    result = BenchmarkResult(
        name=name,
        duration_ms=elapsed,
        iterations=iterations,
        target_ms=target_ms,
        passed=elapsed / max(iterations, 1) <= target_ms,
    )
    status = "PASS" if result.passed else "FAIL"
    logger.info(
        "[%s] %s: %.2fms avg (target: %.0fms, %d iterations)",
        status,
        name,
        result.avg_ms,
        target_ms,
        iterations,
    )
    return result


class ConnectionPool:
    """Simple connection pool for SQLite database connections.

    Maintains a pool of reusable database connections for
    concurrent FastAPI request handling.
    """

    def __init__(self, db_path: str, pool_size: int = 10) -> None:
        import sqlite3
        import threading

        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._sqlite3 = sqlite3

        # Pre-create connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.append(conn)

    def _create_connection(self) -> Any:
        """Create a new database connection with standard pragmas."""
        conn = self._sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = self._sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
        # Pool exhausted, create a new connection
        return self._create_connection()

    def release(self, conn: Any) -> None:
        """Return a connection to the pool."""
        with self._lock:
            if len(self._pool) < self.pool_size:
                self._pool.append(conn)
            else:
                conn.close()

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()


def run_api_benchmarks(
    db_path: str,
    iterations: int = 10,
) -> Optional[BenchmarkSuite]:
    """Run performance benchmarks against the API layer.

    Args:
        db_path: Path to the database file.
        iterations: Number of iterations per benchmark.

    Returns:
        BenchmarkSuite with results, or None if setup fails.
    """
    try:
        from .api.app import create_app
        from .api.routes import configure_db
        from .database import SQLiteManager
    except ImportError:
        logger.warning("Cannot import API modules for benchmarking")
        return None

    from fastapi.testclient import TestClient

    db = SQLiteManager(db_path)
    configure_db(db)
    app = create_app()
    client = TestClient(app)

    suite = BenchmarkSuite()

    # Benchmark: basic verse retrieval (<5ms target)
    suite.add(
        run_benchmark(
            "basic_verse_retrieval",
            lambda: client.get("/api/v1/verses/engbsb/1/1/1?depth=basic"),
            iterations=iterations,
            target_ms=5.0,
        )
    )

    # Benchmark: standard verse retrieval (<30ms target)
    suite.add(
        run_benchmark(
            "standard_verse_retrieval",
            lambda: client.get("/api/v1/verses/engbsb/1/1/1?depth=standard"),
            iterations=iterations,
            target_ms=30.0,
        )
    )

    # Benchmark: deep verse retrieval (<100ms target)
    suite.add(
        run_benchmark(
            "deep_verse_retrieval",
            lambda: client.get("/api/v1/verses/engbsb/1/1/1?depth=deep"),
            iterations=iterations,
            target_ms=100.0,
        )
    )

    # Benchmark: text search (<50ms target)
    suite.add(
        run_benchmark(
            "text_search",
            lambda: client.get("/api/v1/search/text?q=beginning"),
            iterations=iterations,
            target_ms=50.0,
        )
    )

    # Benchmark: book metadata (<10ms target)
    suite.add(
        run_benchmark(
            "book_metadata",
            lambda: client.get("/api/v1/books"),
            iterations=iterations,
            target_ms=10.0,
        )
    )

    logger.info("Benchmark summary: %s", suite.summary())
    return suite
