# ABBA
Annotated Bible and Background Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-90%25+-brightgreen.svg)](https://github.com/jhenderson/ABBA)

## Overview

The Annotated Bible and Background Analysis (ABBA) project makes the Bible more approachable through advanced data processing and multi-format export capabilities. ABBA provides a comprehensive framework for biblical text analysis, supporting multiple languages, canons, and export formats.

## 🚀 Current Status

**Version 1.0 - Production Ready**

All major features have been implemented and tested with ~90%+ test coverage. The system successfully handles:

- ✅ Multi-version Bible alignment
- ✅ Hebrew/Greek morphological analysis  
- ✅ Advanced annotations with ML support
- ✅ Timeline tracking with BCE date handling
- ✅ Cross-reference management
- ✅ Multiple export formats (SQLite, JSON, OpenSearch, Graph DBs)
- ✅ Support for Protestant, Catholic, Orthodox, and Ethiopian canons

## 📚 Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)** - Comprehensive guide to ABBA's multi-format architecture
- **[Canonical Format Specification](docs/CANONICAL_FORMAT.md)** - Detailed specification of the source data format
- **[Automatic Annotations](docs/AUTOMATIC_ANNOTATIONS.md)** - ML-powered annotation generation
- **[Search Methodology](docs/SEARCH_METHODOLOGY.md)** - Cross-language search strategies
- **[Data Integrity](docs/DATA_INTEGRITY.md)** - Validation and quality assurance
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and development setup

## 🛠️ Installation

```bash
# Install from source
git clone https://github.com/jhenderson/ABBA.git
cd ABBA
poetry install

# For development with all features
poetry install --with dev,test

# Run tests
make test

# See all available commands
make help
```

## 🎯 Key Features

### 1. **Multi-Canon Support**
Support for different biblical traditions:
- Protestant (66 books)
- Catholic (73 books)  
- Orthodox (76+ books)
- Ethiopian Orthodox (81 books)

```python
from abba.canon.registry import CanonRegistry

registry = CanonRegistry()
catholic_canon = registry.get_canon("catholic")
books = catholic_canon.get_books()
```

### 2. **Original Language Processing**
Advanced morphological analysis for Hebrew and Greek:

```python
from abba.morphology.hebrew_morphology import HebrewMorphology
from abba.morphology.greek_morphology import GreekMorphology

# Parse Hebrew morphology
hebrew = HebrewMorphology()
morph_data = hebrew.parse_morph_code("Ncmpa")  # Noun, common, masculine, plural, absolute

# Check Greek participles
greek = GreekMorphology()
is_participle = greek.is_participle("V-PAN")  # Verb, Participle, Active, Nominative
```

### 3. **Timeline with BCE Support**
Proper handling of historical dates including BCE:

```python
from abba.timeline.models import Event, create_bce_date

exodus = Event(
    name="The Exodus",
    time_point=TimePoint(
        exact_date=create_bce_date(1446),  # 1446 BCE
        confidence=0.7
    )
)
```

### 4. **Advanced Annotations**
ML-powered annotation system with multiple approaches:

```python
from abba.annotations.annotation_engine import AnnotationEngine

engine = AnnotationEngine()
annotations = await engine.annotate_verse(
    verse_id=VerseID("JHN", 1, 1),
    text="In the beginning was the Word...",
    annotation_types=[AnnotationType.THEOLOGICAL_THEME, AnnotationType.CHRISTOLOGICAL]
)
```

### 5. **Multiple Export Formats**

#### SQLite Export
```python
from abba.export.sqlite_exporter import SQLiteExporter, SQLiteConfig

config = SQLiteConfig(
    output_path="bible.db",
    enable_fts5=True,  # Full-text search support
    create_indices=True
)

exporter = SQLiteExporter(config)
await exporter.export(dataset)
```

#### JSON Export
```python
from abba.export.json_exporter import StaticJSONExporter, JSONConfig

config = JSONConfig(
    output_path="./json_export",
    split_by_book=True,
    create_search_indices=True,
    pretty_print=True
)

exporter = StaticJSONExporter(config)
await exporter.export(dataset)
```

#### Minimal Exports
For simple use cases:

```python
from abba.export.minimal_sqlite import MinimalSQLiteExporter

exporter = MinimalSQLiteExporter("verses.db")
exporter.add_verse("GEN.1.1", "GEN", 1, 1, "In the beginning...")
exporter.finalize()
```

### 6. **Cross-Reference Tracking**
Comprehensive cross-reference system:

```python
from abba.cross_references.models import CrossReference, ReferenceType

xref = CrossReference(
    source_verse=VerseID("GEN", 1, 1),
    target_verse=VerseID("JHN", 1, 1),
    reference_type=ReferenceType.PARALLEL,
    confidence=0.95,
    metadata={"theme": "creation", "linguistic_parallel": True}
)
```

## 📊 Architecture

ABBA uses a modular architecture supporting multiple backends:

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Parsers** | Text processing | Hebrew, Greek, translation parsing |
| **Alignment** | Version mapping | Statistical & ML-based alignment |
| **Annotations** | Content enrichment | Zero-shot, few-shot, BERT models |
| **Export** | Output generation | SQLite, JSON, OpenSearch, Graph |
| **Canon** | Tradition support | Multi-canon with proper ordering |
| **Timeline** | Historical data | BCE dates, uncertainty handling |

## 🧪 Testing

The project maintains ~90%+ test coverage:

```bash
# Run all tests
make test

# Run with coverage report
make test-coverage

# Run specific test file
pytest tests/test_morphology.py

# Run with verbose output
pytest -v tests/
```

## 🔧 Development

### Quick Start
```bash
# Install development dependencies
poetry install --with dev,test

# Run linters
make lint

# Format code
make format

# Run all checks
make check
```

### Project Structure
```
ABBA/
├── src/abba/          # Main package
│   ├── alignment/     # Text alignment systems
│   ├── annotations/   # ML annotation engines
│   ├── canon/         # Biblical canon support
│   ├── export/        # Export formats
│   ├── morphology/    # Language analysis
│   ├── parsers/       # Text parsers
│   └── timeline/      # Historical timeline
├── tests/             # Comprehensive test suite
├── docs/              # Documentation
└── examples/          # Usage examples
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

Areas where contributions are especially welcome:
- Additional language parsers
- More export formats
- Enhanced ML models for annotations
- Additional canon support
- Performance optimizations

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Bible translation data from open source projects
- Strong's Concordance for lexical data
- Morphological databases from various scholarly sources
- The broader digital humanities community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jhenderson/ABBA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jhenderson/ABBA/discussions)
- **Documentation**: [Full Docs](https://github.com/jhenderson/ABBA/tree/main/docs)

---

*ABBA - Making biblical texts accessible through modern technology*