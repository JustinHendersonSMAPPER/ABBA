# ABBA User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Customization Options](#customization-options)
6. [Export Formats](#export-formats)
7. [Command Line Interface](#command-line-interface)
8. [Python API](#python-api)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Quick Start

Get started with ABBA in 5 minutes:

```bash
# Install ABBA
git clone https://github.com/jhenderson/ABBA.git
cd ABBA
poetry install

# Run a simple export
python -m abba export --format sqlite --output bible.db

# Or use the Python API
python
>>> from abba.export.minimal_sqlite import MinimalSQLiteExporter
>>> exporter = MinimalSQLiteExporter("verses.db")
>>> exporter.add_verse("GEN.1.1", "GEN", 1, 1, "In the beginning God created...")
>>> exporter.finalize()
```

## Installation

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Poetry (for dependency management)
- Git

### Full Installation

```bash
# Clone the repository
git clone https://github.com/jhenderson/ABBA.git
cd ABBA

# Install with all features
poetry install --with dev,test

# Verify installation
poetry run pytest --version
poetry run python -c "import abba; print(abba.__version__)"
```

### Minimal Installation

```bash
# For production use only
poetry install --only main

# For specific features
poetry install --extras "ml"  # Machine learning features
poetry install --extras "opensearch"  # OpenSearch support
```

## Basic Usage

### 1. Simple Verse Export

```python
from abba.export.minimal_json import MinimalJSONExporter
from abba.verse_id import VerseID

# Create exporter
exporter = MinimalJSONExporter("verses.json")

# Add verses
exporter.add_verse(
    verse_id="GEN.1.1",
    book="GEN",
    chapter=1,
    verse=1,
    text="In the beginning God created the heavens and the earth."
)

# Save to file
exporter.finalize()
```

### 2. Parse Original Language Text

```python
from abba.parsers.hebrew_parser import HebrewParser
from abba.parsers.greek_parser import GreekParser

# Parse Hebrew
hebrew_parser = HebrewParser()
hebrew_verse = hebrew_parser.parse_verse(
    "GEN.1.1",
    "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃"
)

# Parse Greek
greek_parser = GreekParser()
greek_verse = greek_parser.parse_verse(
    "JHN.1.1",
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν"
)
```

### 3. Multi-Canon Support

```python
from abba.canon.registry import CanonRegistry

# Get available canons
registry = CanonRegistry()
print("Available canons:", registry.list_canons())

# Work with specific canon
catholic = registry.get_canon("catholic")
print(f"Catholic canon has {len(catholic.get_books())} books")

# Check if book is in canon
if catholic.contains_book("TOB"):  # Tobit
    print("Tobit is in the Catholic canon")
```

## Advanced Features

### 1. Text Alignment

```python
from abba.alignment.modern_aligner import ModernAlignmentPipeline
from abba.parsers.hebrew_parser import HebrewVerse, HebrewWord
from abba.parsers.translation_parser import TranslationVerse
from abba.verse_id import VerseID

# Create pipeline
pipeline = ModernAlignmentPipeline()

# Prepare verses
hebrew_verse = HebrewVerse(
    verse_id=VerseID("GEN", 1, 1),
    words=[
        HebrewWord(text="בְּרֵאשִׁית", lemma="רֵאשִׁית", strong_number="H7225"),
        HebrewWord(text="בָּרָא", lemma="בָּרָא", strong_number="H1254"),
    ],
    osis_id="Gen.1.1"
)

translation = TranslationVerse(
    verse_id=VerseID("GEN", 1, 1),
    text="In the beginning God created",
    original_book_name="Genesis",
    original_chapter=1,
    original_verse=1
)

# Align verses
alignment = pipeline.align_verse(hebrew_verse, translation)
print(f"Alignment confidence: {alignment.confidence}")
```

### 2. ML-Powered Annotations

```python
import asyncio
from abba.annotations.annotation_engine import AnnotationEngine
from abba.annotations.models import AnnotationType
from abba.verse_id import VerseID

async def annotate_verse():
    engine = AnnotationEngine()
    
    annotations = await engine.annotate_verse(
        verse_id=VerseID("JHN", 3, 16),
        text="For God so loved the world that he gave his only Son...",
        annotation_types=[
            AnnotationType.THEOLOGICAL_THEME,
            AnnotationType.KEY_TERM,
            AnnotationType.CROSS_REFERENCE
        ]
    )
    
    for ann in annotations:
        print(f"- {ann.type}: {ann.value} (confidence: {ann.confidence})")

# Run async function
asyncio.run(annotate_verse())
```

### 3. Timeline Management

```python
from abba.timeline.models import Event, TimePoint, create_bce_date
from abba.timeline.visualization import TimelineVisualizer

# Create historical events
events = [
    Event(
        id="abraham_born",
        name="Abraham Born",
        description="Birth of Abraham",
        time_point=TimePoint(
            exact_date=create_bce_date(2166),
            confidence=0.7
        )
    ),
    Event(
        id="exodus",
        name="The Exodus",
        description="Israel leaves Egypt",
        time_point=TimePoint(
            exact_date=create_bce_date(1446),
            confidence=0.8
        )
    ),
    Event(
        id="david_king",
        name="David Becomes King",
        description="David crowned king of Israel",
        time_point=TimePoint(
            exact_date=create_bce_date(1010),
            confidence=0.9
        )
    )
]

# Create visualization
visualizer = TimelineVisualizer()
viz_data = visualizer.create_timeline_visualization(events)

# Export as SVG
svg = visualizer.export_svg(viz_data, width=800, height=400)
with open("biblical_timeline.svg", "w") as f:
    f.write(svg)
```

### 4. Cross-Reference Detection

```python
from abba.cross_references.parser import CrossReferenceParser
from abba.cross_references.models import ReferenceType

# Initialize parser
parser = CrossReferenceParser()

# Parse cross-references
source_text = "All have sinned and fall short of the glory of God"
references = parser.parse_references(
    source_verse=VerseID("ROM", 3, 23),
    source_text=source_text,
    target_corpus=["GEN", "PSA", "ISA"]  # Books to search
)

# Display found references
for ref in references:
    print(f"{ref.source_verse} → {ref.target_verse}")
    print(f"  Type: {ref.reference_type}")
    print(f"  Confidence: {ref.confidence}")
```

## Customization Options

### 1. Export Configuration

```python
from abba.export.sqlite_exporter import SQLiteExporter, SQLiteConfig
from abba.export.json_exporter import StaticJSONExporter, JSONConfig

# SQLite with custom options
sqlite_config = SQLiteConfig(
    output_path="custom_bible.db",
    enable_fts5=True,              # Full-text search
    create_indices=True,           # Performance indices
    include_morphology=True,       # Original language data
    include_cross_refs=True,       # Cross-references
    compression_level=6            # 0-9, higher = smaller file
)

# JSON with custom structure
json_config = JSONConfig(
    output_path="./bible_json",
    split_by_book=True,           # Separate file per book
    split_by_chapter=True,        # Separate file per chapter
    create_search_indices=True,   # Search optimization
    include_strongs=True,         # Strong's numbers
    pretty_print=True,            # Human-readable
    chunk_size=100                # Verses per chunk
)
```

### 2. Parser Options

```python
# Hebrew parser with custom settings
from abba.parsers.hebrew_parser import HebrewParser, HebrewParserConfig

config = HebrewParserConfig(
    normalize_final_forms=True,    # כ → ך normalization
    preserve_cantillation=False,   # Remove cantillation marks
    include_ketiv_qere=True        # Include both written/read forms
)

parser = HebrewParser(config)

# Greek parser with options
from abba.parsers.greek_parser import GreekParser, GreekParserConfig

config = GreekParserConfig(
    normalize_sigma=True,          # σ/ς normalization  
    preserve_breathing=True,       # Keep breathing marks
    include_variants=True          # Textual variants
)

parser = GreekParser(config)
```

### 3. Alignment Customization

```python
from abba.alignment.modern_aligner import ModernAlignmentConfig

config = ModernAlignmentConfig(
    # Confidence thresholds
    min_strongs_confidence=0.8,
    min_statistical_confidence=0.6,
    min_neural_confidence=0.7,
    
    # Algorithm weights
    strongs_weight=0.35,
    statistical_weight=0.25,
    neural_weight=0.25,
    syntactic_weight=0.15,
    
    # Performance options
    use_cache=True,
    batch_size=32,
    num_workers=4
)

pipeline = ModernAlignmentPipeline(config)
```

## Export Formats

### 1. SQLite Export

Full-featured relational database:

```python
from abba.export.sqlite_exporter import SQLiteExporter
from abba.export.pipeline import ExportPipeline

# Create pipeline
pipeline = ExportPipeline()

# Configure SQLite export
exporter = SQLiteExporter({
    "output_path": "bible.db",
    "enable_fts5": True,
    "create_indices": True
})

# Add to pipeline
pipeline.add_exporter(exporter)

# Run export
await pipeline.run(source_data)
```

Features:
- Full-text search with FTS5
- Relational structure
- Optimized indices
- Compact file size

### 2. JSON Export

Hierarchical JSON for web applications:

```python
from abba.export.json_exporter import StaticJSONExporter

exporter = StaticJSONExporter({
    "output_path": "./bible_json",
    "split_by_book": True,
    "create_search_indices": True
})

# Export creates structure:
# bible_json/
#   ├── manifest.json
#   ├── books/
#   │   ├── GEN/
#   │   │   ├── book.json
#   │   │   └── chapters/
#   │   │       ├── 1.json
#   │   │       └── ...
#   └── indices/
#       └── search.json
```

### 3. OpenSearch Export

For scalable search infrastructure:

```python
from abba.export.opensearch_exporter import OpenSearchExporter

exporter = OpenSearchExporter({
    "cluster_url": "https://localhost:9200",
    "index_name": "bible-verses",
    "username": "admin",
    "password": "admin",
    "bulk_size": 500,
    "use_ssl": True
})

# Features:
# - Distributed search
# - Real-time indexing
# - Complex aggregations
# - Multi-language support
```

### 4. Graph Database Export

For relationship analysis:

```python
from abba.export.graph_exporter import Neo4jExporter

exporter = Neo4jExporter({
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password",
    "database": "bible"
})

# Creates nodes: Verse, Person, Place, Topic
# Creates relationships: REFERENCES, MENTIONS, FOLLOWS
```

## Command Line Interface

### Basic Commands

```bash
# Export to SQLite
abba export --format sqlite --output bible.db

# Export to JSON with options
abba export --format json --output ./bible_json \
    --split-by-book --pretty-print

# Parse specific files
abba parse hebrew --input hebrew_text.xml --output parsed.json
abba parse greek --input greek_text.xml --output parsed.json

# Generate annotations
abba annotate --input verses.json --output annotated.json \
    --types theological,cross-reference

# Align texts
abba align --source hebrew.json --target english.json \
    --output alignments.json --method modern
```

### Advanced CLI Usage

```bash
# Custom configuration file
abba export --config my_config.yaml

# Batch processing
find ./sources -name "*.xml" | \
    xargs -P 4 -I {} abba parse --input {} --output {}.json

# Pipeline with multiple outputs
abba pipeline \
    --source ./canonical \
    --outputs sqlite:bible.db,json:./json,opensearch:localhost:9200

# Validation
abba validate --schema canonical --input data.json
```

## Python API

### Core Classes

```python
# Verse identification
from abba.verse_id import VerseID, parse_verse_id, parse_verse_range

verse = VerseID("ROM", 3, 23)
verse_with_part = VerseID("ROM", 3, 23, part="a")

# Parse from string
verse = parse_verse_id("ROM.3.23")
range = parse_verse_range("ROM.3.23-25")

# Canon management
from abba.canon.registry import CanonRegistry
from abba.canon.models import Canon, Testament

registry = CanonRegistry()
protestant = registry.get_canon("protestant")

# Morphology
from abba.morphology.hebrew_morphology import HebrewMorphology
from abba.morphology.greek_morphology import GreekMorphology

hebrew = HebrewMorphology()
greek = GreekMorphology()

# Export pipeline
from abba.export.pipeline import ExportPipeline
from abba.export.validation import OutputValidator

pipeline = ExportPipeline()
validator = OutputValidator()
```

### Async Operations

```python
import asyncio
from abba.export.pipeline import ExportPipeline

async def main():
    pipeline = ExportPipeline()
    
    # Add async exporters
    pipeline.add_exporter("opensearch", {...})
    
    # Run pipeline
    await pipeline.run_async()
    
    # Validate results
    results = await pipeline.get_results()
    print(f"Exported {results['total_verses']} verses")

asyncio.run(main())
```

## Examples

### Example 1: Build a Bible Study App Database

```python
from abba.export.sqlite_exporter import SQLiteExporter, SQLiteConfig
from abba.canon.registry import CanonRegistry
import asyncio

async def build_study_database():
    # Setup
    config = SQLiteConfig(
        output_path="bible_study.db",
        enable_fts5=True,
        include_morphology=True,
        include_cross_refs=True,
        include_timeline=True
    )
    
    exporter = SQLiteExporter(config)
    registry = CanonRegistry()
    
    # Export Protestant canon
    protestant = registry.get_canon("protestant")
    
    for book_code in protestant.get_books():
        print(f"Processing {book_code}...")
        # Load and export book data
        await exporter.export_book(book_code)
    
    await exporter.finalize()
    print("Database created successfully!")

asyncio.run(build_study_database())
```

### Example 2: Generate Scripture Memory Cards

```python
from abba.verse_id import parse_verse_id
from abba.annotations.annotation_engine import AnnotationEngine
import json

async def create_memory_cards(verse_refs):
    engine = AnnotationEngine()
    cards = []
    
    for ref in verse_refs:
        verse_id = parse_verse_id(ref)
        
        # Get verse text
        text = get_verse_text(verse_id)  # Your implementation
        
        # Get key themes
        annotations = await engine.annotate_verse(
            verse_id=verse_id,
            text=text,
            annotation_types=[AnnotationType.KEY_TERM]
        )
        
        # Create card
        card = {
            "reference": ref,
            "text": text,
            "key_terms": [a.value for a in annotations],
            "difficulty": calculate_difficulty(text)
        }
        
        cards.append(card)
    
    # Save cards
    with open("memory_cards.json", "w") as f:
        json.dump(cards, f, indent=2)

# Usage
verses = ["JHN.3.16", "ROM.8.28", "PSA.23.1"]
asyncio.run(create_memory_cards(verses))
```

### Example 3: Cross-Reference Network

```python
from abba.cross_references.models import CrossReference
from abba.export.graph_exporter import GraphMLExporter
import networkx as nx

def build_reference_network(references: list[CrossReference]):
    G = nx.DiGraph()
    
    for ref in references:
        # Add nodes
        G.add_node(str(ref.source_verse), type="verse")
        G.add_node(str(ref.target_verse), type="verse")
        
        # Add edge with attributes
        G.add_edge(
            str(ref.source_verse),
            str(ref.target_verse),
            type=ref.reference_type.value,
            confidence=ref.confidence
        )
    
    # Export to GraphML
    exporter = GraphMLExporter()
    exporter.export_graph(G, "cross_references.graphml")
    
    # Find important verses (high PageRank)
    pagerank = nx.pagerank(G)
    important = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Most referenced verses:")
    for verse, score in important:
        print(f"  {verse}: {score:.4f}")
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'abba'
# Solution: Ensure you're in the project directory and have run:
poetry install
poetry shell  # Activate virtual environment
```

**2. Memory Issues with Large Exports**
```python
# Use streaming export for large datasets
from abba.export.streaming import StreamingExporter

exporter = StreamingExporter(
    chunk_size=1000,  # Process 1000 verses at a time
    max_memory_mb=512  # Limit memory usage
)
```

**3. Slow Performance**
```python
# Enable parallel processing
from abba.config import Config

Config.set_parallel_workers(8)
Config.enable_caching(True)
Config.set_batch_size(100)
```

**4. Character Encoding Issues**
```python
# Force UTF-8 encoding
from abba.parsers.base import ParserConfig

config = ParserConfig(
    encoding="utf-8",
    handle_bom=True,
    normalize_unicode=True
)
```

### Debug Mode

Enable detailed logging:

```python
import logging
from abba.config import Config

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode
Config.debug_mode = True

# Verbose output
Config.verbose = True
```

### Getting Help

1. Check documentation: `/docs` folder
2. Run tests: `make test`
3. View examples: `/examples` folder
4. GitHub Issues: Report bugs or request features
5. Discussions: Ask questions in GitHub Discussions

## Next Steps

- Explore the [API Documentation](API.md)
- Read about [Data Flow and Algorithms](DATA_FLOW_AND_ALGORITHMS.md)
- Check out [Example Scripts](/examples)
- Contribute to the project on [GitHub](https://github.com/jhenderson/ABBA)