# CLAUDE.md

## Project Overview

ABBA (Annotated Bible and Background Analysis) is a Python framework for biblical text analysis using SQLite for structured data and ChromaDB for semantic search. Uses uv for dependency management.

## Commands

```bash
# Setup
uv sync                                 # Create .venv and install all deps (incl. dev group)

# Quality (MUST pass before any phase is complete)
uv run ruff format .                    # Format (replaces black + isort)
uv run ruff check --fix .               # Lint + autofix (replaces flake8 + pylint + bandit)
nox -s lint                             # ruff format --check + ruff check
nox -s typing                           # pyright
nox -s tests                            # pytest + coverage
nox -s security                         # ruff flake8-bandit (S) rules

# Run
uv run python abba/main.py              # Main app
uv run python abba/main.py --list       # List translations

# Single test
uv run pytest tests/test_file.py::test_name -v
```

## Phase Completion Requirements

**Every phase MUST meet ALL of these before being marked complete:**
1. **80% minimum test coverage** for new/modified code (goal: 95% where practical)
2. **ruff format**: Zero formatting issues (line length: 120; replaces black + isort)
3. **ruff check**: Zero violations (replaces flake8 + pylint + bandit)
4. **pyright**: Zero type errors on `abba/` source (replaces mypy)
5. Tests must actually pass

Update `claude/checklist.md` when completing phase items.

## Architecture

```
bible.db / STEPBible → SQLite (abba.db) → Ollama → ChromaDB
                              ↓                        ↓
                       Direct Queries          Semantic Queries
                              ↓                        ↓
                          Unified API ←←←←←←←←←←←←←←←←
```

- **Config priority**: CLI args > .env > config.json > defaults
- **Env vars**: Prefixed `ABBA_` (e.g., `ABBA_DATA_DIR`)
- **Embeddings**: Sentence Transformers (NOT Ollama). English: `e5-large-v2`, Multilingual: `multilingual-e5-base`
- **Semantic analysis**: Ollama (for concept extraction/validation only)
- **Lexicons**: OpenScriptures HebrewStrong.xml (CC BY 4.0), Abbott-Smith Greek (public domain)

## Data Sources

- **bible.helloao.org**: Bible translations via `bible.db` (MIT license API, public domain BSB)
- **STEPBible-Data**: Hebrew/Greek texts, morphology (CC BY 4.0, Tyndale House Cambridge)
  - TAHOT (Hebrew OT): 4 files, TAGNT (Greek NT): 2 files
  - Morphology: Hebrew (TEHMC) + Greek (TEGMC)
- **OpenScriptures**: HebrewStrong.xml - Hebrew/Aramaic lexicon (CC BY 4.0 / public domain)
- **Abbott-Smith**: Greek Lexicon TEI XML (public domain, 1922)

## Key Design Decisions

- **Verse-level alignment** for cross-linguistic mapping (not word-level)
- **Strong's-Centric Semantic Mapping** for concept search (lexicographic, not algorithmic)
- **Build-time LLM processing** - no LLM needed at search time
- **Canon-aware imports**: Protestant (66), Catholic (73), Orthodox (76+), Ethiopian (81)
- **MurmurHash3** for data integrity validation
- **Parallel processing**: Auto-detects CPU count, threads for I/O, processes for CPU

## Development Phases

See `claude/checklist.md` for detailed status:
1. **Phase 1**: SQLite database foundation - **COMPLETED**
2. **Phase 2**: Vector database and embeddings
3. **Phase 3**: Semantic search
4. **Phase 4**: Concept mapping
5. **Phase 5**: Performance optimization
6. **Phase 6**: Testing and documentation

## Code Standards

- **Ruff**: 120 char line length; formatter + linter (config in `pyproject.toml` under `[tool.ruff]`)
- **Pyright**: type checker (config in `pyproject.toml` under `[tool.pyright]`)
- **Type hints**: Required on all function signatures in `abba/`
- Use `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (tooling) for new dependencies
- `@pytest.mark.integration` for tests needing external services
- Place debug/temp files in `claude/` folder (gitignored)
