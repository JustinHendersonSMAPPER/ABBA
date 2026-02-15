# CLAUDE.md

## Project Overview

ABBA (Annotated Bible and Background Analysis) is a Python framework for biblical text analysis using SQLite for structured data and ChromaDB for semantic search. Uses Poetry for dependency management.

## Commands

```bash
# Setup
poetry install

# Quality (MUST pass before any phase is complete)
poetry run black --line-length 120 .    # Format
poetry run isort --profile black .      # Sort imports
nox -s lint                             # black, isort, flake8, pylint
nox -s typing                           # mypy
nox -s tests                            # pytest + coverage
nox -s security                         # bandit

# Run
poetry run python abba/main.py          # Main app
poetry run python abba/main.py --list   # List translations

# Single test
poetry run pytest tests/test_file.py::test_name -v
```

## Phase Completion Requirements

**Every phase MUST meet ALL of these before being marked complete:**
1. **80% minimum test coverage** for new/modified code (goal: 95% where practical)
2. **black + isort**: Zero formatting issues (line length: 120)
3. **flake8**: Zero violations
4. **pylint**: Zero findings (ignore the score; 0 warnings/errors is the standard)
5. **mypy**: Zero type errors on `abba/` source
6. Tests must actually pass

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

- **Black**: 120 char line length
- **isort**: black-compatible profile
- **Type hints**: Required on all function signatures in `abba/`
- Use `poetry add` for new dependencies
- `@pytest.mark.integration` for tests needing external services
- Place debug/temp files in `claude/` folder (gitignored)
