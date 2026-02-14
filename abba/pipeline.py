"""Data pipeline orchestration for the full ABBA build process.

Coordinates importing translations, STEPBible data, lexicons,
enrichment data, and embedding generation in the correct order.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in the data pipeline."""

    name: str
    description: str
    fn: Callable[..., Any]
    depends_on: List[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class PipelineResult:
    """Result of a pipeline step execution."""

    step_name: str
    success: bool
    duration_seconds: float
    message: str = ""
    error: Optional[str] = None


class DataPipeline:
    """Orchestrates the full ABBA data build pipeline.

    Steps execute in dependency order.  Each step receives the config
    and db_path and is responsible for its own idempotency.
    """

    def __init__(self, db_path: Path, data_dir: Path) -> None:
        self.db_path = db_path
        self.data_dir = data_dir
        self._steps: Dict[str, PipelineStep] = {}
        self._results: List[PipelineResult] = []

    def add_step(self, step: PipelineStep) -> None:
        """Register a pipeline step."""
        self._steps[step.name] = step

    @property
    def results(self) -> List[PipelineResult]:
        """Return results from the most recent run."""
        return list(self._results)

    def _resolve_order(self) -> List[str]:
        """Topologically sort steps by their dependencies."""
        visited: Dict[str, bool] = {}
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited[name] = True
            step = self._steps.get(name)
            if step is None:
                return
            for dep in step.depends_on:
                visit(dep)
            order.append(name)

        for name in self._steps:
            visit(name)
        return order

    def run(self, skip: Optional[List[str]] = None) -> List[PipelineResult]:
        """Execute all pipeline steps in dependency order.

        Args:
            skip: Optional list of step names to skip.

        Returns:
            List of PipelineResult for each executed step.
        """
        skip = skip or []
        self._results = []
        order = self._resolve_order()

        logger.info("Pipeline execution order: %s", ", ".join(order))

        for step_name in order:
            if step_name in skip:
                logger.info("Skipping step: %s", step_name)
                continue

            step = self._steps[step_name]
            logger.info("Running step: %s — %s", step.name, step.description)

            start = time.time()
            try:
                step.fn(db_path=self.db_path, data_dir=self.data_dir)
                duration = time.time() - start
                result = PipelineResult(
                    step_name=step.name,
                    success=True,
                    duration_seconds=round(duration, 2),
                    message=f"Completed in {duration:.1f}s",
                )
            except Exception as exc:
                duration = time.time() - start
                result = PipelineResult(
                    step_name=step.name,
                    success=False,
                    duration_seconds=round(duration, 2),
                    error=str(exc),
                )
                if not step.optional:
                    logger.error("Pipeline step %s failed: %s", step.name, exc)
                    self._results.append(result)
                    break
                logger.warning("Optional step %s failed: %s (continuing)", step.name, exc)

            self._results.append(result)

        return self._results

    def summary(self) -> str:
        """Return a human-readable summary of the last run."""
        lines = ["Pipeline Summary", "=" * 40]
        for r in self._results:
            status = "OK" if r.success else "FAIL"
            line = f"  [{status}] {r.step_name}: {r.duration_seconds}s"
            if r.error:
                line += f" — {r.error}"
            lines.append(line)
        total = sum(r.duration_seconds for r in self._results)
        ok_count = sum(1 for r in self._results if r.success)
        lines.append(f"\n  {ok_count}/{len(self._results)} steps succeeded ({total:.1f}s total)")
        return "\n".join(lines)


def _step_init_database(db_path: Path, **_kwargs: Any) -> None:
    """Initialize database schema and run migrations."""
    from .database import SQLiteManager  # pylint: disable=import-outside-toplevel
    from .database.migrations import run_migrations  # pylint: disable=import-outside-toplevel

    db_path.parent.mkdir(parents=True, exist_ok=True)
    mgr = SQLiteManager(db_path)
    mgr.initialize_database()
    run_migrations(db_path)


def _step_import_lexicons(data_dir: Path, **_kwargs: Any) -> None:
    """Import all available lexicon files."""
    from .lexicon_parser import (  # pylint: disable=import-outside-toplevel
        parse_abbott_smith_xml,
        parse_bdb_xml,
        parse_dodson_csv,
        parse_hebrew_strongs_xml,
        parse_strongs_greek_xml,
        parse_tflsj_txt,
    )

    lexicon_dir = data_dir / "lexicons"
    if not lexicon_dir.exists():
        logger.info("No lexicon directory at %s — skipping lexicon import", lexicon_dir)
        return

    parsers = [
        ("hebrew_strongs.xml", parse_hebrew_strongs_xml),
        ("abbott_smith.xml", parse_abbott_smith_xml),
        ("dodson.csv", parse_dodson_csv),
        ("strongs_greek.xml", parse_strongs_greek_xml),
        ("tflsj.txt", parse_tflsj_txt),
    ]

    for filename, parser in parsers:
        path = lexicon_dir / filename
        if path.exists():
            entries = parser(path)
            logger.info("Parsed %d entries from %s", len(entries), filename)

    # BDB requires both files
    bdb_path = lexicon_dir / "bdb.xml"
    index_path = lexicon_dir / "lexical_index.xml"
    if bdb_path.exists() and index_path.exists():
        entries = parse_bdb_xml(bdb_path, index_path)
        logger.info("Parsed %d BDB entries", len(entries))


def _step_run_enrichment(db_path: Path, **_kwargs: Any) -> None:
    """Run all enrichment data populators."""
    from .enrichment import (  # pylint: disable=import-outside-toplevel
        BookMetadataPopulator,
        ConceptQualityPopulator,
        CrossReferencePopulator,
        CulturalContextPopulator,
        DiscourseAnnotationPopulator,
        GenreShiftPopulator,
        LifeTopicPopulator,
        LiteraryStructurePopulator,
        ManuscriptVariantPopulator,
        PassagePopulator,
        ReadingPlanPopulator,
        SemanticDomainPopulator,
        SemanticGraphPopulator,
        SpeakerAttributionPopulator,
        SyntaxTreePopulator,
        WordExplanationPopulator,
        WordRichnessComputer,
    )

    populator_classes: List[Any] = [
        BookMetadataPopulator,
        PassagePopulator,
        LiteraryStructurePopulator,
        CrossReferencePopulator,
        CulturalContextPopulator,
        GenreShiftPopulator,
        SpeakerAttributionPopulator,
        WordExplanationPopulator,
        WordRichnessComputer,
        LifeTopicPopulator,
        ReadingPlanPopulator,
        SemanticDomainPopulator,
        SemanticGraphPopulator,
        DiscourseAnnotationPopulator,
        SyntaxTreePopulator,
        ManuscriptVariantPopulator,
        ConceptQualityPopulator,
    ]

    for cls in populator_classes:
        try:
            pop = cls(db_path)
            count: int = pop.populate()
            logger.info("Populated %s: %d rows", cls.__name__, count)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Enrichment %s failed: %s", cls.__name__, exc)


def create_default_pipeline(db_path: Path, data_dir: Path) -> DataPipeline:
    """Create a pipeline with all standard ABBA build steps.

    Args:
        db_path: Path to the ABBA SQLite database.
        data_dir: Root data directory (bible_data/).

    Returns:
        Configured DataPipeline ready to run.
    """
    pipeline = DataPipeline(db_path, data_dir)

    pipeline.add_step(
        PipelineStep(
            name="init_database",
            description="Initialize database schema and run migrations",
            fn=_step_init_database,
        )
    )

    pipeline.add_step(
        PipelineStep(
            name="import_lexicons",
            description="Import lexicon data (Strong's, BDB, Abbott-Smith, Dodson, TFLSJ)",
            fn=_step_import_lexicons,
            depends_on=["init_database"],
            optional=True,
        )
    )

    pipeline.add_step(
        PipelineStep(
            name="run_enrichment",
            description="Populate enrichment data (metadata, cross-refs, passages, etc.)",
            fn=_step_run_enrichment,
            depends_on=["init_database"],
        )
    )

    return pipeline
