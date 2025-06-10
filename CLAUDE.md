# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABBA (Annotated Bible and Background Analysis) is a comprehensive framework for biblical text analysis. The project provides advanced data processing capabilities including multi-language support, morphological analysis, timeline tracking, and multiple export formats. 

**Current Status**: Version 1.0 - Production Ready with ~90%+ test coverage (867+ tests passing)

### Key Features

- **Multi-Canon Support**: Protestant, Catholic, Orthodox, Ethiopian, Syriac, Samaritan, and Hebrew Bible canons
- **Original Language Processing**: Advanced Hebrew and Greek morphological analysis
- **ML-Powered Annotations**: Zero-shot, few-shot, and BERT-based classification
- **Timeline Management**: BCE date handling with uncertainty modeling
- **Cross-Reference System**: Citation tracking with confidence scoring
- **Multiple Export Formats**: SQLite, JSON, OpenSearch, and Graph DB support
- **Interlinear Display**: Token extraction and alignment for original languages
- **Unicode Support**: RTL handling, transliteration, and font detection

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
├── src/abba/              # Main package code
│   ├── alignment/         # Text alignment systems
│   ├── annotations/       # ML-powered annotation engines
│   ├── canon/            # Biblical canon support
│   ├── cross_references/ # Cross-reference management
│   ├── export/           # Export format implementations
│   ├── interlinear/      # Interlinear display generation
│   ├── language/         # Unicode, RTL, transliteration
│   ├── manuscript/       # Manuscript variant support
│   ├── morphology/       # Hebrew/Greek morphological analysis
│   ├── parsers/          # Text parsing systems
│   ├── timeline/         # Historical timeline support
│   └── schemas/          # JSON schema definitions
├── tests/                 # Comprehensive test suite (867+ tests)
├── docs/                  # Architecture and design documentation
├── examples/              # Sample data and usage examples
├── scripts/               # Utility and demonstration scripts
└── pyproject.toml         # Poetry configuration
```

## Key Documentation

- **[Architecture Documentation](docs/ARCHITECTURE.md)** - Comprehensive multi-format architecture supporting SQLite, OpenSearch, Graph DB, and static files
- **[Canonical Format Specification](docs/CANONICAL_FORMAT.md)** - Detailed specification of the source data format
- **[Example Data](examples/canonical_sample.json)** - Sample canonical format data
- **[Automatic Annotations](docs/AUTOMATIC_ANNOTATIONS.md)** - ML-powered annotation generation
- **[Search Methodology](docs/SEARCH_METHODOLOGY.md)** - Cross-language search strategies
- **[Data Integrity](docs/DATA_INTEGRITY.md)** - Validation and quality assurance
- **[Modern Alignment Stack](docs/MODERN_ALIGNMENT_STACK.md)** - Advanced text alignment system

## Running Demonstrations

The project includes several demonstration scripts to showcase functionality:

```bash
# Demonstrate working features across all modules
python scripts/demonstrate_working_features.py

# Validate the complete pipeline
python scripts/validate_pipeline.py

# Test modern alignment system
python scripts/validate_modern_alignment.py

# Train alignment models
python scripts/train_modern_alignment.py

# Check test status
python scripts/test_status_check.py
```

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
- Place any test files specific to Claude testing in claude_scripts and any files saved from these should go to claude_output

## Key Implementation Notes

### BCE Date Handling
The timeline system uses an encoding convention for BCE dates to work around Python datetime limitations:
```python
from abba.timeline.models import create_bce_date

# Create a BCE date (e.g., 1446 BCE)
date = create_bce_date(1446)  # Internally encoded as year 3554
```

### Canon Support
The system supports multiple canons with proper book ordering:
```python
from abba.canon.registry import CanonRegistry

registry = CanonRegistry()
catholic_canon = registry.get_canon("catholic")
books = catholic_canon.get_books()  # Returns book codes in canonical order
```

### Morphology Analysis
Both Hebrew and Greek morphological analysis are supported:
```python
from abba.morphology.hebrew_morphology import HebrewMorphology
from abba.morphology.greek_morphology import GreekMorphology

# Hebrew example
hebrew = HebrewMorphology()
morph_data = hebrew.parse_morph_code("Ncmpa")  # Noun, common, masculine, plural, absolute

# Greek example  
greek = GreekMorphology()
is_participle = greek.is_participle("V-PAN")  # Returns True for participles
```

### Export Formats
The system provides both full-featured and minimal exporters:
```python
# Full SQLite export with FTS5 support
from abba.export.sqlite_exporter import SQLiteExporter, SQLiteConfig

config = SQLiteConfig(output_path="bible.db", enable_fts5=True)
exporter = SQLiteExporter(config)

# Minimal SQLite for simple use cases
from abba.export.minimal_sqlite import MinimalSQLiteExporter

exporter = MinimalSQLiteExporter("verses.db")
exporter.add_verse("GEN.1.1", "GEN", 1, 1, "In the beginning...")
exporter.finalize()
```

## Common Tasks

### Running Tests for Specific Modules
```bash
# Test morphology systems
pytest tests/test_morphology.py -v

# Test export functionality
pytest tests/test_export_*.py -v

# Test with coverage for a specific module
pytest tests/test_interlinear.py --cov=abba.interlinear
```

### Checking Code Quality
```bash
# Run all quality checks
make check

# Run specific linters
poetry run mypy src/abba/morphology
poetry run flake8 src/abba/export
```

### Building Documentation
```bash
# Generate API documentation (when implemented)
make docs

# View test coverage report
make test-coverage
open htmlcov/index.html
```