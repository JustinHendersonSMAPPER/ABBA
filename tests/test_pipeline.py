"""Tests for the data pipeline orchestration module."""

import tempfile
import unittest
from pathlib import Path

from abba.pipeline import DataPipeline, PipelineResult, PipelineStep, create_default_pipeline


class TestPipelineStep(unittest.TestCase):
    """Test PipelineStep dataclass."""

    def test_basic_step(self):
        """Verify default PipelineStep field values."""
        step = PipelineStep(name="test", description="A test step", fn=lambda **_: None)
        self.assertEqual(step.name, "test")
        self.assertEqual(step.description, "A test step")
        self.assertEqual(step.depends_on, [])
        self.assertFalse(step.optional)

    def test_step_with_deps(self):
        """Verify step stores dependency list."""
        step = PipelineStep(
            name="step2",
            description="Depends on step1",
            fn=lambda **_: None,
            depends_on=["step1"],
        )
        self.assertEqual(step.depends_on, ["step1"])


class TestPipelineResult(unittest.TestCase):
    """Test PipelineResult dataclass."""

    def test_success_result(self):
        """Verify success result fields."""
        result = PipelineResult(step_name="test", success=True, duration_seconds=1.5, message="OK")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_failure_result(self):
        """Verify failure result fields."""
        result = PipelineResult(step_name="test", success=False, duration_seconds=0.1, error="boom")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "boom")


class TestDataPipeline(unittest.TestCase):
    """Test DataPipeline orchestration."""

    def setUp(self):
        """Create temp directory and paths for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir()

    def test_empty_pipeline(self):
        """An empty pipeline runs with no results."""
        pipeline = DataPipeline(self.db_path, self.data_dir)
        results = pipeline.run()
        self.assertEqual(len(results), 0)

    def test_single_step_success(self):
        """A single passing step returns success result."""
        pipeline = DataPipeline(self.db_path, self.data_dir)
        pipeline.add_step(PipelineStep(name="step1", description="Test step", fn=lambda **_: None))
        results = pipeline.run()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].step_name, "step1")

    def test_dependency_ordering(self):
        """Steps execute in dependency order regardless of registration order."""
        pipeline = DataPipeline(self.db_path, self.data_dir)
        order = []

        pipeline.add_step(
            PipelineStep(
                name="step2",
                description="Second",
                fn=lambda **_: order.append("step2"),
                depends_on=["step1"],
            )
        )
        pipeline.add_step(
            PipelineStep(
                name="step1",
                description="First",
                fn=lambda **_: order.append("step1"),
            )
        )
        pipeline.run()
        self.assertEqual(order, ["step1", "step2"])

    def test_mandatory_step_failure_stops_pipeline(self):
        """A mandatory step failure stops subsequent steps."""
        pipeline = DataPipeline(self.db_path, self.data_dir)

        def fail(**_kwargs):
            raise RuntimeError("Step failed")

        pipeline.add_step(PipelineStep(name="step1", description="Fails", fn=fail))
        pipeline.add_step(
            PipelineStep(
                name="step2",
                description="Should not run",
                fn=lambda **_: None,
                depends_on=["step1"],
            )
        )
        results = pipeline.run()
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("Step failed", results[0].error)

    def test_optional_step_failure_continues(self):
        """An optional step failure allows subsequent steps to continue."""
        pipeline = DataPipeline(self.db_path, self.data_dir)

        def fail(**_kwargs):
            raise RuntimeError("Optional failed")

        pipeline.add_step(PipelineStep(name="step1", description="Optional fail", fn=fail, optional=True))
        pipeline.add_step(
            PipelineStep(
                name="step2",
                description="Should still run",
                fn=lambda **_: None,
                depends_on=["step1"],
            )
        )
        results = pipeline.run()
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].success)
        self.assertTrue(results[1].success)

    def test_skip_steps(self):
        """Skipped steps are excluded from execution."""
        pipeline = DataPipeline(self.db_path, self.data_dir)
        pipeline.add_step(PipelineStep(name="step1", description="Skipped", fn=lambda **_: None))
        pipeline.add_step(
            PipelineStep(
                name="step2",
                description="Runs",
                fn=lambda **_: None,
                depends_on=["step1"],
            )
        )
        results = pipeline.run(skip=["step1"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].step_name, "step2")

    def test_summary(self):
        """Summary includes step names and status."""
        pipeline = DataPipeline(self.db_path, self.data_dir)
        pipeline.add_step(PipelineStep(name="step1", description="Test", fn=lambda **_: None))
        pipeline.run()
        summary = pipeline.summary()
        self.assertIn("Pipeline Summary", summary)
        self.assertIn("step1", summary)
        self.assertIn("OK", summary)

    def test_results_property(self):
        """Results property returns a copy of run results."""
        pipeline = DataPipeline(self.db_path, self.data_dir)
        pipeline.add_step(PipelineStep(name="step1", description="Test", fn=lambda **_: None))
        pipeline.run()
        results = pipeline.results
        self.assertEqual(len(results), 1)


class TestCreateDefaultPipeline(unittest.TestCase):
    """Test the default pipeline factory."""

    def test_creates_pipeline_with_steps(self):
        """Default pipeline includes init, lexicons, and enrichment steps."""
        temp_dir = tempfile.mkdtemp()
        pipeline = create_default_pipeline(
            db_path=Path(temp_dir) / "test.db",
            data_dir=Path(temp_dir) / "data",
        )
        self.assertIsInstance(pipeline, DataPipeline)
        order = pipeline._resolve_order()  # pylint: disable=protected-access
        self.assertIn("init_database", order)
        self.assertIn("import_lexicons", order)
        self.assertIn("run_enrichment", order)

    def test_init_database_step_runs(self):
        """The init_database step creates the SQLite database file."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "data" / "test.db"
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        pipeline = create_default_pipeline(db_path=db_path, data_dir=data_dir)
        results = pipeline.run(skip=["import_lexicons", "run_enrichment"])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertTrue(db_path.exists())


if __name__ == "__main__":
    unittest.main()
