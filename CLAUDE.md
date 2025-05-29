# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABBA (Annotated Bible and Background Analysis) is a data format specification project for biblical study. The project aims to create an extensible, structured data format for presenting biblical texts with comprehensive annotations, cross-references, and contextual information.

## Development Commands

### Quick Start with Make
```bash
# Install all dependencies
make install

# Run tests
make test

# Run linters
make lint

# Format code
make format

# Run all checks
make check

# See all available commands
make help
```

### Detailed Poetry Commands

#### Setup
```bash
# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

#### Development
```bash
# Run formatters
poetry run black src tests
poetry run isort src tests

# Run linters
poetry run flake8 src tests
poetry run pylint src tests
poetry run mypy src tests
poetry run bandit -r src
poetry run vulture src

# Run all linters at once
poetry run pre-commit run --all-files
```

#### Testing
```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=abba --cov-report=html --cov-report=term

# Run specific test file
poetry run pytest tests/test_specific.py

# Run tests matching pattern
poetry run pytest -k "test_pattern"
```

#### Building
```bash
# Build the package
poetry build
```

## Core Architectural Goals

The ABBA format is designed to support:

1. **Multi-version alignment** - Normalizing verse identifiers across different Bible translations
2. **Interlinear parsing** - Supporting original language (Greek/Hebrew) analysis with morphological data
3. **Layered annotation system** - Topics, themes, and cross-references at verse, section, and chapter levels
4. **Temporal data integration** - Timeline and historical context embedding
5. **Multi-canon support** - Modular system for different biblical canons (Protestant, Catholic, Orthodox, etc.)
6. **Citation tracking** - Mapping Old Testament quotes in the New Testament
7. **Manuscript variant support** - Linking to textual criticism data

## Technology Stack

- **Language**: Python 3.11-3.13
- **Package Manager**: Poetry
- **Testing**: pytest, coverage
- **Linting**: black, isort, flake8, pylint, mypy, bandit, vulture
- **License**: MIT

## Project Structure

```
abba/
├── src/
│   └── abba/         # Main package code
├── tests/            # Test files
├── docs/             # Documentation
└── pyproject.toml    # Poetry configuration
```

## Key Documentation

- **[Architecture Documentation](docs/ARCHITECTURE.md)** - Comprehensive multi-format architecture supporting SQLite, OpenSearch, Graph DB, and static files
- **[Canonical Format Specification](docs/CANONICAL_FORMAT.md)** - Detailed specification of the source data format
- **[Example Data](examples/canonical_sample.json)** - Sample canonical format data

## Key Design Considerations

When implementing the ABBA format:
- Support multiple backend formats (SQLite, OpenSearch, static JSON, Graph DB) for different use cases
- Implement format-agnostic query interfaces for maximum portability
- Optimize each format for its specific use case (e.g., static files for CDNs, SQLite for mobile)
- Focus on extensibility to support various biblical scholarship use cases
- Ensure proper handling of verse ID normalization across different versification systems
- Design data structures that can efficiently handle cross-references and annotations at multiple levels
- Consider performance implications for large-scale biblical text processing
- Maintain type safety with mypy annotations throughout the codebase
- Ensure comprehensive test coverage for all functionality