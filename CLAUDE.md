# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABBA (Annotated Bible and Background Analysis) is a comprehensive Python framework for biblical text analysis that combines linguistic precision with semantic intelligence. The project is transitioning from a simple Bible data extractor to a dual-database architecture using SQLite for structured data and vector databases for semantic search.

## Key Development Commands

### Setup and Dependencies
```bash
# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Running the Application
```bash
# Run main application (automatically generates all missing data including embeddings)
python abba/main.py

# Or use poetry
poetry run python abba/main.py

# Common options
python abba/main.py --list                    # List available translations
python abba/main.py --force-download          # Force re-download of bible.db
python abba/main.py --translations KJV ESV    # Extract specific translations

# Embedding generation (optional - main.py auto-generates missing embeddings)
python abba/main.py --embed-verses            # Force re-generate verse embeddings
python abba/main.py --embed-words             # Force re-generate word embeddings
python abba/main.py --embed-all               # Force re-generate all embeddings
python abba/main.py --force-reembed           # Force overwrite existing embeddings
```

### Testing and Quality Checks
```bash
# Run full test suite with coverage
nox -s tests

# Run linting (includes pylint, flake8, isort, flynt)
nox -s lint

# Run type checking
nox -s typing

# Run security checks
nox -s security

# Run individual test
poetry run pytest tests/test_specific.py::test_function_name
```

### Code Formatting
```bash
# Format with black (line length: 120)
poetry run black .

# Sort imports
poetry run isort .
```

## Architecture Overview

### Current State
The project currently extracts Bible translations from `bible.db` into JSON files. Key components:

- **Configuration Hierarchy**: CLI args > .env file > config.json > defaults
- **Data Flow**: bible.db → BibleExtractor → JSON files
- **STEPBible Integration**: Downloads Hebrew/Greek texts with morphology and lexicons

### Planned Architecture (per checklist.md)
```
STEPBible Files → SQLite Database → Ollama Processing → Vector Database
                           ↓                                    ↓
                   Direct Queries                      Semantic Queries
                           ↓                                    ↓
                       Unified API ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

### Configuration System
- **Priority**: CLI arguments override environment variables override config files
- **Environment Variables**: Prefixed with `ABBA_` (e.g., `ABBA_DATA_DIR`)
- **Config Files**: JSON format, specified via `--config-file`

### Key Design Decisions

1. **Dual Embedding Models**:
   - English: `intfloat/e5-large-v2` (1024 dimensions)
   - Multilingual: `intfloat/multilingual-e5-base` (768 dimensions)
   - Using Sentence Transformers, NOT Ollama for embeddings

2. **Multi-Model Semantic Analysis**:
   - Default models: `llama4:scout,command-r-plus:latest`
   - Consensus-based scoring for reliability
   - Ollama used for concept extraction, NOT embeddings

3. **Enhanced Context Embeddings**:
   - Combine verse text + original language + morphology + Strong's numbers
   - Language-specific pipelines for optimal accuracy

## STEPBible Data Structure

The system downloads 10 files + attribution:
- **TAHOT** (Hebrew OT): 4 files split by book ranges
- **TAGNT** (Greek NT): 2 files split by book ranges  
- **Lexicons**: Hebrew (TBESH) and Greek (TBESG)
- **Morphology**: Hebrew (TEHMC) and Greek (TEGMC) code explanations

Files use tab-separated format with complex field structures including Strong's numbers, morphology codes, and transliterations.

## Development Priorities

Follow the phases in `claude/checklist.md`:
1. **Phase 1**: SQLite database foundation (modify extractor to read directly from bible.db) ✅ **COMPLETED**
2. **Phase 2**: Vector database and embedding integration
3. **Phase 3**: Semantic search implementation
4. **Phase 4**: Concept mapping system
5. **Phase 5**: Performance optimization
6. **Phase 6**: Testing and documentation

**Important**: When completing tasks, always update the checklist in `claude/checklist.md` to mark items as `[x]` instead of `[ ]`.

## Alignment Strategy

The project uses **verse-level alignment** as the primary method for cross-linguistic text mapping. This pragmatic approach:

- Handles empty/blank verses by storing them as empty strings
- Preserves verse numbering across all translations
- Provides sufficient alignment for 99% of use cases
- Avoids complex linguistic challenges that have proven difficult historically

**Advanced semantic alignment** (word-level, semantic units, etc.) is considered a future research topic, not a current implementation goal. See `claude/docs/CROSS_LINGUISTIC_ALIGNMENT_PROPOSAL.md` for analysis of the challenges. This decision keeps the project focused on delivering practical value without getting bogged down in time-consuming areas that may not work as intended.

## Code Quality Standards

**All code must meet these requirements before merging:**

1. **Test Coverage**: Minimum 95% coverage required
   - Run `nox -s tests` to check coverage
   - Coverage report generated in `reports/coverage.xml`
   - No untested code paths allowed

2. **Code Quality Tools** - All must pass with zero issues:
   - **Black**: Auto-formatter with 120 character line length
   - **Pylint**: Full code analysis must pass
   - **Mypy**: Type checking with strict mode
   - **Bandit**: Security analysis must show no issues
   - **Flake8**: Style guide enforcement
   - **isort**: Import sorting (black-compatible profile)

3. **Pre-commit Verification**:
   ```bash
   # Run all checks before committing
   poetry run black .
   poetry run isort .
   nox -s lint    # Runs pylint, flake8, isort, flynt
   nox -s typing  # Runs mypy
   nox -s security # Runs bandit
   nox -s tests   # Ensures >95% coverage
   ```

## Claude Development Files

**All Claude-specific test scripts, debug files, and documentation should be placed in the `claude/` folder for easier cleanup.**

### Claude Folder Organization
```
claude/
├── scripts/          # Debug and test scripts
├── documentation/    # Claude-generated docs and analysis
├── temp_files/       # Temporary files for testing
└── notes/           # Development notes and findings
```

### Guidelines for Claude Files
- Place all debug scripts (like `debug_stepbible.py`) in `claude/scripts/`
- Place temporary test files in `claude/temp_files/`
- Place analysis and documentation in `claude/documentation/`
- Clean up `claude/` folder before final commits
- Add `claude/` to `.gitignore` if needed to prevent accidental commits

## Important Notes

- The project uses Poetry for dependency management - always use `poetry add` for new dependencies
- Black formatter is configured with 120 character line length
- Tests should be marked with `@pytest.mark.integration` if they require external services
- The codebase follows a configuration-first approach with extensive CLI and environment variable support
- All new database features should support both CLI arguments and environment variables
- Embedding models and Ollama models serve different purposes - keep them separate
- Every new feature must include comprehensive tests to maintain >95% coverage
- All code must pass ALL quality checks - no exceptions
- **Use `claude/` folder for all Claude-generated debugging files and temporary scripts**