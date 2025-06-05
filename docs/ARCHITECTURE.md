# ABBA Multi-Format Architecture

## Overview

ABBA (Annotated Bible and Background Analysis) is a comprehensive framework for biblical text analysis, supporting multiple data formats optimized for different use cases, from static websites to distributed search clusters. The system is production-ready with ~90%+ test coverage.

## Core Design Principles

1. **Single Source of Truth**: All formats derive from a canonical data model
2. **Query Optimization**: Each format optimized for its primary use case
3. **Progressive Enhancement**: Simple formats for basic needs, complex formats for advanced features
4. **Interoperability**: Standard formats for maximum compatibility
5. **Incremental Updates**: Support for efficient delta updates
6. **Multi-Language Support**: Full Hebrew and Greek morphological analysis
7. **Extensible Annotations**: ML-powered annotation system with multiple approaches

## Data Formats

### 1. Static File Formats (No Backend Required)

#### Hierarchical JSON Structure
```
abba-data/
├── manifest.json              # Version info, checksums, metadata
├── books/
│   ├── gen/
│   │   ├── book.json         # Book metadata
│   │   ├── chapters/
│   │   │   ├── 1.json        # Chapter data with verses
│   │   │   └── ...
│   │   └── search/
│   │       ├── words.json    # Word frequency index
│   │       └── topics.json   # Topic/theme index
│   └── ...
├── cross-refs/
│   ├── forward.json          # Source → targets
│   └── reverse.json          # Target → sources
├── search/
│   ├── concordance.json      # Global word index
│   ├── topics.json           # Global topic taxonomy
│   └── timeline.json         # Chronological index
└── versions/
    ├── esv/
    │   └── ... (same structure)
    └── ...
```

**Optimized for**:
- Static site generators
- CDN distribution
- Offline apps
- Browser-based searching

#### Compressed Bundle Format
```python
# Binary format using MessagePack or CBOR
{
    "header": {
        "version": "1.0.0",
        "compression": "zstd",
        "indices": {...}
    },
    "data": compressed_binary_data,
    "index": {
        "verse_offsets": [...],     # Quick verse lookup
        "search_trees": {...},       # Pre-built search structures
        "bloom_filters": {...}       # Probabilistic search
    }
}
```

### 2. SQLite Database Schema

```sql
-- Core verse storage with FTS5 for full-text search
CREATE TABLE verses (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT UNIQUE NOT NULL,  -- e.g., "GEN.1.1"
    book_id TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    verse_part TEXT,  -- For split verses like "1a", "1b"
    UNIQUE(book_id, chapter, verse, verse_part)
);

-- Version-specific text with full-text search
CREATE VIRTUAL TABLE verse_text USING fts5(
    verse_id,
    version_id,
    text,
    text_normalized,  -- Lowercase, no punctuation for better search
    content=verse_text,
    tokenize='porter unicode61'
);

-- Original language tokens
CREATE TABLE tokens (
    id INTEGER PRIMARY KEY,
    verse_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    surface_form TEXT NOT NULL,
    lemma TEXT NOT NULL,
    strongs TEXT,
    morph_code TEXT,  -- Standardized morphology
    gloss TEXT,
    FOREIGN KEY (verse_id) REFERENCES verses(id),
    INDEX idx_lemma (lemma),
    INDEX idx_strongs (strongs)
);

-- Hierarchical annotations with materialized paths
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,  -- topic, theme, note, etc.
    path TEXT NOT NULL,  -- e.g., "/theology/salvation/justification"
    name TEXT NOT NULL,
    parent_id INTEGER,
    metadata JSON,
    FOREIGN KEY (parent_id) REFERENCES annotations(id),
    INDEX idx_path (path),
    INDEX idx_type (type)
);

-- Many-to-many verse annotations with scope
CREATE TABLE verse_annotations (
    verse_id INTEGER NOT NULL,
    annotation_id INTEGER NOT NULL,
    scope TEXT NOT NULL,  -- verse, passage, chapter, book
    confidence REAL DEFAULT 1.0,
    metadata JSON,
    PRIMARY KEY (verse_id, annotation_id, scope),
    FOREIGN KEY (verse_id) REFERENCES verses(id),
    FOREIGN KEY (annotation_id) REFERENCES annotations(id)
);

-- Cross-references with relationship types
CREATE TABLE cross_references (
    id INTEGER PRIMARY KEY,
    source_verse_id INTEGER NOT NULL,
    source_end_verse_id INTEGER,  -- For ranges
    target_verse_id INTEGER NOT NULL,
    target_end_verse_id INTEGER,
    type TEXT NOT NULL,  -- quote, allusion, parallel, etc.
    confidence REAL DEFAULT 1.0,
    metadata JSON,
    FOREIGN KEY (source_verse_id) REFERENCES verses(id),
    FOREIGN KEY (target_verse_id) REFERENCES verses(id),
    INDEX idx_source (source_verse_id),
    INDEX idx_target (target_verse_id)
);

-- Timeline and historical data
CREATE TABLE timeline_events (
    id INTEGER PRIMARY KEY,
    date_start TEXT,  -- ISO 8601 or BCE notation
    date_end TEXT,
    date_precision TEXT,  -- year, decade, century
    event_type TEXT,
    description TEXT,
    metadata JSON
);

-- R-tree index for efficient range queries
CREATE VIRTUAL TABLE timeline_index USING rtree(
    id,
    min_year, max_year  -- Numeric year for range queries
);

-- Cached search results for common queries
CREATE TABLE search_cache (
    query_hash TEXT PRIMARY KEY,
    query_type TEXT NOT NULL,
    parameters JSON NOT NULL,
    results JSON NOT NULL,
    created_at INTEGER NOT NULL,
    hit_count INTEGER DEFAULT 0
);

-- Views for common access patterns
CREATE VIEW verse_with_text AS
SELECT 
    v.*,
    GROUP_CONCAT(
        json_object('version', vt.version_id, 'text', vt.text),
        ','
    ) as translations
FROM verses v
LEFT JOIN verse_text vt ON v.id = vt.verse_id
GROUP BY v.id;

-- Indexes for performance
CREATE INDEX idx_verse_lookup ON verses(book_id, chapter, verse);
CREATE INDEX idx_annotation_verse ON verse_annotations(annotation_id, verse_id);
CREATE INDEX idx_timeline_date ON timeline_events(date_start, date_end);
```

**Optimized for**:
- Desktop applications
- Mobile apps
- Single-user scenarios
- Complex local queries

### 3. OpenSearch/Elasticsearch Schema

```json
// Primary verse index
{
  "mappings": {
    "properties": {
      "canonical_id": { 
        "type": "keyword",
        "fields": {
          "parts": {
            "type": "text",
            "analyzer": "verse_id_analyzer"
          }
        }
      },
      "book": { "type": "keyword" },
      "chapter": { "type": "integer" },
      "verse": { "type": "integer" },
      "verse_part": { "type": "keyword" },
      
      // Nested for multiple versions
      "translations": {
        "type": "nested",
        "properties": {
          "version": { "type": "keyword" },
          "text": { 
            "type": "text",
            "analyzer": "standard",
            "fields": {
              "exact": { "type": "keyword" },
              "stemmed": { "analyzer": "english" },
              "phonetic": { "analyzer": "phonetic" }
            }
          },
          "language": { "type": "keyword" }
        }
      },
      
      // Original language data
      "tokens": {
        "type": "nested",
        "properties": {
          "position": { "type": "integer" },
          "surface": { "type": "keyword" },
          "lemma": { 
            "type": "keyword",
            "fields": {
              "transliterated": { "type": "text" }
            }
          },
          "strongs": { "type": "keyword" },
          "morph": { "type": "keyword" },
          "gloss": { "type": "text" }
        }
      },
      
      // Hierarchical annotations
      "annotations": {
        "type": "nested",
        "properties": {
          "type": { "type": "keyword" },
          "path": { 
            "type": "text",
            "analyzer": "path_hierarchy"
          },
          "value": { "type": "keyword" },
          "confidence": { "type": "float" }
        }
      },
      
      // Geographic data
      "location": {
        "type": "geo_point"
      },
      
      // Timeline data
      "date_range": {
        "type": "date_range",
        "format": "yyyy||yyyy-MM||yyyy-MM-dd||epoch_millis"
      },
      
      // Suggest functionality
      "suggest": {
        "type": "completion",
        "contexts": [
          {
            "name": "book",
            "type": "category"
          }
        ]
      }
    }
  },
  
  "settings": {
    "analysis": {
      "analyzer": {
        "verse_id_analyzer": {
          "tokenizer": "verse_id_tokenizer"
        },
        "phonetic": {
          "tokenizer": "standard",
          "filter": ["lowercase", "double_metaphone"]
        }
      },
      "tokenizer": {
        "verse_id_tokenizer": {
          "type": "pattern",
          "pattern": "[.]"
        }
      }
    }
  }
}

// Separate indices for different query patterns
// Cross-reference index for graph-like queries
{
  "cross_references": {
    "properties": {
      "source": { "type": "keyword" },
      "target": { "type": "keyword" },
      "type": { "type": "keyword" },
      "chain": {  // For multi-hop references
        "type": "keyword"
      }
    }
  }
}

// Topic hierarchy index
{
  "topics": {
    "properties": {
      "id": { "type": "keyword" },
      "name": { "type": "text" },
      "path": { 
        "type": "text",
        "analyzer": "path_hierarchy"
      },
      "verse_count": { "type": "integer" },
      "related_topics": { "type": "keyword" }
    }
  }
}

// Timeline index for temporal queries
{
  "timeline": {
    "properties": {
      "event_id": { "type": "keyword" },
      "date_range": { "type": "date_range" },
      "event_type": { "type": "keyword" },
      "related_verses": { "type": "keyword" },
      "description": { "type": "text" }
    }
  }
}
```

**Optimized for**:
- Web APIs
- Distributed search
- Complex aggregations
- Multi-language search
- Fuzzy matching
- Real-time updates

### 4. Graph Database Format (Neo4j/ArangoDB)

```cypher
// Node types
(:Verse {
  canonical_id: "GEN.1.1",
  book: "GEN",
  chapter: 1,
  verse: 1
})

(:Version {
  id: "ESV",
  name: "English Standard Version",
  language: "en"
})

(:Word {
  surface: "בְּרֵאשִׁית",
  lemma: "רֵאשִׁית",
  strongs: "H7225"
})

(:Topic {
  id: "creation",
  name: "Creation",
  path: "/theology/creation"
})

(:Person {
  id: "abraham",
  name: "Abraham"
})

(:Place {
  id: "jerusalem",
  name: "Jerusalem",
  coordinates: [35.2137, 31.7683]
})

(:Event {
  id: "exodus",
  name: "The Exodus",
  date_start: "1446 BCE"
})

// Relationships
(:Verse)-[:HAS_TEXT {text: "In the beginning..."}]->(:Version)
(:Verse)-[:CONTAINS {position: 1}]->(:Word)
(:Verse)-[:REFERENCES {type: "quote"}]->(:Verse)
(:Verse)-[:ADDRESSES]->(:Topic)
(:Verse)-[:MENTIONS]->(:Person)
(:Verse)-[:LOCATED_IN]->(:Place)
(:Verse)-[:DESCRIBES]->(:Event)
(:Topic)-[:PARENT_OF]->(:Topic)
(:Person)-[:CHILD_OF]->(:Person)
(:Event)-[:FOLLOWS]->(:Event)

// Example: Find all verses about faith that reference Abraham
MATCH (t:Topic {name: "Faith"})<-[:ADDRESSES]-(v:Verse)-[:MENTIONS]->(p:Person {name: "Abraham"})
RETURN v
```

**Optimized for**:
- Relationship traversal
- Complex cross-references
- Genealogies
- Thematic connections
- Social network analysis

### 5. Analytical Formats

#### Apache Parquet for Data Science
```python
# Columnar format for efficient analytical queries
verses_df = {
    'canonical_id': ['GEN.1.1', 'GEN.1.2', ...],
    'book': ['GEN', 'GEN', ...],
    'chapter': [1, 1, ...],
    'verse': [1, 2, ...],
    'text_esv': ['In the beginning...', ...],
    'text_niv': ['In the beginning...', ...],
    'word_count': [10, 12, ...],
    'sentiment_score': [0.8, 0.7, ...],
    'topics': [['creation', 'god'], ['spirit'], ...]
}

# Optimized for:
# - Statistical analysis
# - Machine learning
# - Bulk processing
# - Columnar compression
```

## Data Pipeline Architecture

```yaml
# Pipeline configuration
pipeline:
  source:
    format: "canonical_json"
    path: "source/canonical/"
    
  stages:
    - name: "validate"
      processor: "SchemaValidator"
      
    - name: "enrich"
      processors:
        - "StrongsEnricher"
        - "CrossReferenceResolver"
        - "TopicTagger"
        
    - name: "generate"
      outputs:
        - format: "static_json"
          path: "dist/static/"
          options:
            compression: "gzip"
            
        - format: "sqlite"
          path: "dist/abba.db"
          options:
            fts: true
            
        - format: "elasticsearch"
          endpoint: "http://localhost:9200"
          options:
            bulk_size: 1000
            
        - format: "parquet"
          path: "dist/analytics/"
          options:
            compression: "snappy"
            
    - name: "optimize"
      processors:
        - "SearchIndexBuilder"
        - "CacheWarmer"
        
    - name: "validate_output"
      processor: "OutputValidator"
```

## Query Optimization Strategies

### 1. Static Files
- Pre-computed indices in JSON
- Bloom filters for existence checks
- Client-side caching strategies
- Progressive loading

### 2. SQLite
- Covering indices for common queries
- Materialized views for complex joins
- FTS5 for full-text search
- R-tree for range queries

### 3. OpenSearch
- Index templates for consistency
- Careful shard allocation
- Query result caching
- Aggregation optimization

### 4. Graph Database
- Indexed properties for entry points
- Optimized traversal patterns
- Relationship property indices
- Query result caching

## Current Implementation Status

### Core Modules (Production Ready)

1. **Parsers** (`src/abba/parsers/`)
   - Hebrew parser with full morphological support
   - Greek parser with extensive grammatical analysis
   - Translation parser for modern versions
   - Lexicon parser for Strong's and other references

2. **Alignment System** (`src/abba/alignment/`)
   - Statistical aligner for cross-version mapping
   - Modern aligner with ML support
   - Bridge tables for complex verse mappings
   - Unified reference system

3. **Morphology** (`src/abba/morphology/`)
   - Hebrew morphology with complete parsing
   - Greek morphology including participle detection
   - Unified morphology interface

4. **Annotations** (`src/abba/annotations/`)
   - Zero-shot classification using Hugging Face models
   - Few-shot learning with example-based classification
   - BERT adapter for contextual understanding
   - Quality control and confidence scoring
   - Topic discovery and taxonomy management

5. **Timeline System** (`src/abba/timeline/`)
   - BCE date handling with special encoding
   - Event and period modeling
   - Temporal graph construction
   - Uncertainty modeling
   - Visualization components

6. **Export System** (`src/abba/export/`)
   - SQLite exporter with FTS5 support
   - JSON exporter with hierarchical structure
   - OpenSearch exporter (fully async)
   - Graph database exporters (Neo4j, ArangoDB)
   - Pipeline orchestration
   - Validation framework

7. **Multi-Canon Support** (`src/abba/canon/`)
   - Protestant, Catholic, Orthodox, Ethiopian canons
   - Canon comparison and validation
   - Versification system handling

8. **Cross-References** (`src/abba/cross_references/`)
   - Citation parser and tracker
   - Confidence scoring
   - Relationship classification

9. **Interlinear Support** (`src/abba/interlinear/`)
   - Token extraction and alignment
   - Display generation
   - Lexicon integration

### Testing & Quality

- **Test Coverage**: ~90%+ (867+ tests passing)
- **Code Quality**: Enforced through black, isort, flake8, pylint, mypy
- **Type Safety**: Full mypy type annotations
- **Documentation**: Comprehensive docstrings and architectural docs

## Performance Benchmarks

| Query Type | Static JSON | SQLite | OpenSearch | Graph DB |
|------------|------------|---------|------------|----------|
| Single verse lookup | <1ms | <1ms | ~5ms | ~3ms |
| Word search (single book) | ~10ms | <5ms | <10ms | ~20ms |
| Word search (whole Bible) | ~200ms | ~20ms | <20ms | ~100ms |
| Cross-ref traversal (1 hop) | ~50ms | ~10ms | ~15ms | <5ms |
| Cross-ref traversal (3 hops) | N/A | ~100ms | ~50ms | <20ms |
| Topic aggregation | ~100ms | ~30ms | <10ms | ~15ms |
| Timeline range query | ~150ms | ~15ms | <10ms | ~25ms |

## Deployment Architecture

### Static Deployment (CDN/Edge)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│     CDN     │────▶│  S3/GCS     │
│    Cache    │     │   (Cache)   │     │  (Origin)   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Edge Worker │ (Optional)
                    │  (Search)   │
                    └─────────────┘
```

### Database Deployment
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   API GW    │────▶│  Lambda/    │
│     App     │     │  (Cache)    │     │  Container  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                           ┌───────────────────┴───────────────┐
                           ▼                                   ▼
                    ┌─────────────┐                   ┌─────────────┐
                    │   SQLite    │                   │ OpenSearch  │
                    │   (Read)    │                   │  (Search)   │
                    └─────────────┘                   └─────────────┘
```

### Graph Database Deployment
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  GraphQL    │────▶│   Apollo    │────▶│   Neo4j/    │
│   Client    │     │   Server    │     │  ArangoDB   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Redis    │
                    │   (Cache)   │
                    └─────────────┘
```

## Usage Examples

### Basic Usage
```python
from abba.export.pipeline import ExportPipeline
from abba.parsers.hebrew_parser import HebrewParser
from abba.parsers.greek_parser import GreekParser

# Create pipeline
pipeline = ExportPipeline()

# Add parsers
pipeline.add_parser(HebrewParser())
pipeline.add_parser(GreekParser())

# Configure exports
pipeline.add_exporter("sqlite", {"output_path": "bible.db"})
pipeline.add_exporter("json", {"output_path": "./static"})

# Run pipeline
await pipeline.run()
```

### Advanced Annotation
```python
from abba.annotations.annotation_engine import AnnotationEngine
from abba.verse_id import VerseID

# Create annotation engine
engine = AnnotationEngine()

# Annotate verse
verse_id = VerseID("JHN", 3, 16)
text = "For God so loved the world..."

annotations = await engine.annotate_verse(
    verse_id=verse_id,
    text=text,
    annotation_types=[
        AnnotationType.THEOLOGICAL_THEME,
        AnnotationType.CROSS_REFERENCE,
        AnnotationType.KEY_TERM
    ]
)
```

### Timeline Visualization
```python
from abba.timeline.models import Event, create_bce_date
from abba.timeline.visualization import TimelineVisualizer

# Create events
events = [
    Event(
        id="exodus",
        name="The Exodus",
        description="Israel leaves Egypt",
        time_point=TimePoint(
            exact_date=create_bce_date(1446),
            confidence=0.7
        )
    )
]

# Create visualizer
visualizer = TimelineVisualizer()

# Generate visualization
viz_data = visualizer.create_timeline_visualization(events)
svg = visualizer.export_svg(viz_data)
```

## Selection Guide

| Use Case | Recommended Format | Why |
|----------|-------------------|-----|
| Static website | Hierarchical JSON | No backend needed, CDN-friendly |
| Mobile app | SQLite | Offline capability, single file |
| Web API | OpenSearch | Scalable, flexible search |
| Research tool | Graph DB + Parquet | Complex queries, analytics |
| Desktop app | SQLite | Fast, local, full-featured |
| Microservice | Protocol Buffers | Compact, fast parsing |

## Implementation Priority

1. **Phase 1**: Canonical JSON format + SQLite (covers 80% of use cases)
2. **Phase 2**: Static JSON distribution (enables web usage)
3. **Phase 3**: OpenSearch integration (enables advanced search)
4. **Phase 4**: Graph database (enables relationship analysis)
5. **Phase 5**: Analytical formats (enables research)