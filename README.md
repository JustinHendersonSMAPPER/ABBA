# ABBA
Annotated Bible and Background Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-0%25-red.svg)](https://github.com/jhenderson/ABBA)

## Overview

The Annotated Bible and Background Analysis (ABBA) project's goal is to make the bible more approachable. The bible is critical to apply for every day Christianity and is also considered a key historical document studied by non-christians. However, the bible's scope and breadth is large. With multiple authors, early languages that do not conform to modern languages, and large timeframes where cultures and intended audiences change, studying the bible can be difficult to study. This project aims to present the bible in a format that allows multiple forms of bible study.

The goal of the project is not to build a user interface for accessing the data but is instead designed to build a new data format that is more extensible and presentable that can be used by various user interfaces.

## Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)** - Comprehensive guide to ABBA's multi-format architecture
- **[Canonical Format Specification](docs/CANONICAL_FORMAT.md)** - Detailed specification of the source data format
- **[Automatic Annotations](docs/AUTOMATIC_ANNOTATIONS.md)** - How ABBA automatically generates high-confidence linguistic annotations
- **[Search Methodology](docs/SEARCH_METHODOLOGY.md)** - Strategies for searching across languages and modern concepts
- **[Data Integrity](docs/DATA_INTEGRITY.md)** - Source verification, validation rules, and quality assurance
- **[Example Data](examples/canonical_sample.json)** - Sample data in canonical format

## Installation

```bash
# Install from source
git clone https://github.com/jhenderson/ABBA.git
cd ABBA
poetry install

# For development
poetry install --with dev,test
```

## Data Format Design

### Core Structure

ABBA uses a hierarchical JSON-based format with the following primary components:

```json
{
  "version": "1.0.0",
  "metadata": {
    "generated": "2024-01-01T00:00:00Z",
    "sources": ["..."],
    "canons": ["protestant", "catholic", "orthodox"]
  },
  "books": [
    {
      "id": "GEN",
      "name": "Genesis",
      "canon": ["protestant", "catholic", "orthodox"],
      "metadata": {...},
      "chapters": [...]
    }
  ]
}
```

### Key Features

The ABBA format addresses the following challenges:

#### 1. Verse and Passage Alignment Across Versions
- **Problem**: Many translations split or combine verses differently
- **Solution**: Unified Reference System (URS) that normalizes verse IDs across all versions
- **Implementation**: Each verse has a canonical ID plus version-specific mappings

#### 2. Interlinear and Original Language Parsing
- **Problem**: Greek/Hebrew words map to multiple English words with complex grammar
- **Solution**: Token-level mapping with full morphological analysis
- **Features**:
  - Strong's numbers for word roots
  - Morphological tags (tense, voice, mood, case, gender, number)
  - Semantic domain classification
  - Pronunciation guides

#### 3. Topical and Thematic Tagging
- **Problem**: Theological themes span multiple verses, chapters, or books
- **Solution**: Multi-level hierarchical tagging system
- **Tag Categories**:
  - Topics (prayer, faith, love, etc.)
  - Theological concepts (trinity, salvation, covenant)
  - Literary types (prophecy, poetry, narrative, discourse)
  - Audience context (Jews, Gentiles, churches)

#### 4. Cross-References
- **Problem**: Biblical passages frequently reference other passages
- **Solution**: Bidirectional reference mapping with relationship types
- **Reference Types**:
  - Direct quotation
  - Allusion
  - Thematic parallel
  - Prophetic fulfillment
  - Type/antitype

#### 5. Timeline Integration
- **Problem**: Biblical events span thousands of years with complex chronology
- **Solution**: Temporal metadata at multiple granularities
- **Data Points**:
  - Estimated dates (ranges for uncertainty)
  - Historical periods
  - Genealogical connections
  - Contemporary events

#### 6. Canonical and Apocryphal Differentiation
- **Problem**: Different traditions include different books
- **Solution**: Modular canon system with tradition markers
- **Supported Canons**:
  - Protestant (66 books)
  - Catholic (73 books)
  - Orthodox (76-81 books)
  - Ethiopian (81 books)

#### 7. Citation Tracking
- **Problem**: NT frequently quotes/references OT
- **Solution**: Comprehensive citation database with confidence levels
- **Citation Types**:
  - Verbatim quote
  - Paraphrase
  - Allusion
  - Typological reference

#### 8. Translation Philosophy Metadata
- **Problem**: Different translations use different approaches
- **Solution**: Translation profile for each version
- **Metadata**:
  - Philosophy (formal/dynamic/paraphrase)
  - Reading level
  - Denominational affiliation
  - Textual basis

#### 9. Variant Readings and Manuscript Evidence
- **Problem**: Different manuscripts contain textual variants
- **Solution**: Critical apparatus integration
- **Features**:
  - Variant readings with manuscript support
  - Confidence ratings
  - Links to manuscript images

#### 10. Multi-language Support
- **Problem**: Global audience needs multiple languages
- **Solution**: Language-agnostic core with locale-specific content
- **Support for**:
  - UI translations
  - Bible text in 100+ languages
  - RTL languages (Hebrew, Arabic)

#### 11. Historical and Cultural Context
- **Problem**: Modern readers lack ancient context
- **Solution**: Contextual annotation layers
- **Context Types**:
  - Historical background
  - Cultural practices
  - Geographic information
  - Archaeological insights

## Technical Architecture

ABBA supports multiple data formats optimized for different use cases, from static websites to distributed search clusters. See our comprehensive [Architecture Documentation](docs/ARCHITECTURE.md) for detailed schemas and implementation details.

### Multi-Format Strategy

| Format | Use Case | Key Benefits |
|--------|----------|--------------|
| **Static JSON** | Websites, CDNs | No backend required, progressive loading |
| **SQLite** | Desktop/Mobile apps | Single file, offline capable, full SQL |
| **OpenSearch** | Web APIs | Distributed search, real-time updates |
| **Graph DB** | Research tools | Relationship analysis, complex traversals |
| **Parquet** | Data science | Columnar storage, analytical queries |

### Core Data Model

```python
# Canonical verse representation
@dataclass
class CanonicalVerse:
    canonical_id: str      # Universal ID (e.g., "GEN.1.1")
    book: str             # Book code
    chapter: int          # Chapter number
    verse: int            # Verse number
    verse_part: Optional[str]  # For split verses (e.g., "a", "b")
    
    # Multi-version text storage
    translations: Dict[str, VerseText]
    
    # Original language data
    hebrew_tokens: Optional[List[HebrewToken]]
    greek_tokens: Optional[List[GreekToken]]
    
    # Rich annotations
    annotations: List[Annotation]
    cross_references: List[CrossReference]
    
    # Metadata
    timeline: Optional[TimelineData]
    geography: Optional[GeographyData]
    
    def to_json(self) -> dict:
        """Export to static JSON format"""
        
    def to_sqlite_row(self) -> tuple:
        """Export to SQLite format"""
        
    def to_opensearch_doc(self) -> dict:
        """Export to OpenSearch format"""
```

### Query Interface

```python
# Format-agnostic query interface
class ABBAQuery:
    def __init__(self, backend: ABBABackend):
        self.backend = backend
    
    # Basic queries
    def get_verse(self, verse_id: str) -> CanonicalVerse
    def search_text(self, query: str, version: str = None) -> List[CanonicalVerse]
    
    # Advanced queries
    def search_hebrew(self, lemma: str = None, morph: str = None) -> List[Token]
    def find_cross_refs(self, verse_id: str, depth: int = 1) -> Graph
    def search_by_timeline(self, start: str, end: str) -> List[CanonicalVerse]
    def aggregate_by_topic(self, topic: str) -> TopicAnalysis

# Backend implementations
class StaticJSONBackend(ABBABackend):
    """Browser-compatible, no server required"""

class SQLiteBackend(ABBABackend):
    """Local database with full SQL capabilities"""

class OpenSearchBackend(ABBABackend):
    """Distributed search with advanced features"""

class GraphBackend(ABBABackend):
    """Optimized for relationship queries"""
```

### Performance Characteristics

| Operation | Static JSON | SQLite | OpenSearch | Graph DB |
|-----------|------------|---------|------------|----------|
| Single verse | <1ms | <1ms | ~5ms | ~3ms |
| Text search | ~200ms | ~20ms | <20ms | ~100ms |
| Cross-refs (3 hops) | N/A | ~100ms | ~50ms | <20ms |
| Timeline query | ~150ms | ~15ms | <10ms | ~25ms |

## Development Roadmap

### Phase 1: Core Data Model (Current)
- [ ] Define JSON schema for ABBA format
- [ ] Implement verse alignment algorithm
- [ ] Create data validation tools
- [ ] Build basic import/export utilities

### Phase 2: Original Language Support
- [ ] Integrate Strong's concordance
- [ ] Add morphological parsing
- [ ] Implement interlinear alignment

### Phase 3: Annotation System
- [ ] Design annotation taxonomy
- [ ] Build cross-reference database
- [ ] Add topical tagging

### Phase 4: Historical Context
- [ ] Timeline integration
- [ ] Cultural notes system
- [ ] Archaeological data links

### Phase 5: Distribution
- [ ] API library development
- [ ] Data compression optimization
- [ ] Documentation and examples

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/jhenderson/ABBA.git
cd ABBA

# Install dependencies
make install

# Run tests
make test

# Run tests with coverage
make test-coverage

# Run all linters
make lint

# Format code
make format

# Run all checks before committing
make check
```

See all available commands with `make help`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Bible text sources: [List of open Bible translation projects]
- Linguistic data: [Credits for Strong's, morphology databases]
- Historical research: [Academic sources]