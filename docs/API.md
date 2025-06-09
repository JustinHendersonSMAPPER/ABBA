# ABBA API Documentation

This document provides comprehensive API documentation for the key modules in the ABBA framework.

## Table of Contents

1. [Parsers](#parsers)
2. [Morphology](#morphology)
3. [Alignment](#alignment)
4. [Annotations](#annotations)
5. [Timeline](#timeline)
6. [Export](#export)
7. [Canon Support](#canon-support)
8. [Cross References](#cross-references)
9. [Interlinear](#interlinear)

## Parsers

### Hebrew Parser

```python
from abba.parsers.hebrew_parser import HebrewParser, HebrewVerse, HebrewWord

# Initialize parser
parser = HebrewParser()

# Parse a verse
verse_data = {
    "verse_id": "GEN.1.1",
    "text": "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ",
    "words": [
        {
            "text": "בְּרֵאשִׁית",
            "lemma": "רֵאשִׁית",
            "strong": "H7225",
            "morph": "Ncfsa",
            "gloss": "beginning"
        }
        # ... more words
    ]
}

verse = parser.parse_verse(verse_data)

# Access parsed data
for word in verse.words:
    print(f"{word.text} ({word.lemma}) - {word.gloss}")
    if word.morphology:
        print(f"  Part of speech: {word.morphology.part_of_speech}")
```

### Greek Parser

```python
from abba.parsers.greek_parser import GreekParser, GreekVerse, GreekWord

# Initialize parser
parser = GreekParser()

# Parse a verse
verse_data = {
    "verse_id": "JHN.1.1",
    "text": "Ἐν ἀρχῇ ἦν ὁ Λόγος",
    "words": [
        {
            "text": "Ἐν",
            "lemma": "ἐν",
            "strong": "G1722",
            "morph": "P",
            "gloss": "in"
        }
        # ... more words
    ]
}

verse = parser.parse_verse(verse_data)

# Check if word is a participle
for word in verse.words:
    if parser.morphology.is_participle(word.morph):
        print(f"{word.text} is a participle")
```

### Translation Parser

```python
from abba.parsers.translation_parser import TranslationParser, TranslationVerse

parser = TranslationParser()

# Parse modern translation
verse_data = {
    "verse_id": "GEN.1.1",
    "translation_id": "ESV",
    "text": "In the beginning, God created the heavens and the earth."
}

verse = parser.parse_verse(verse_data)
```

## Morphology

### Hebrew Morphology

```python
from abba.morphology.hebrew_morphology import HebrewMorphology

morph = HebrewMorphology()

# Parse morphology code
morph_data = morph.parse_morph_code("Ncfsa")
print(f"Part of speech: {morph_data.part_of_speech}")  # noun
print(f"Gender: {morph_data.gender}")  # feminine
print(f"Number: {morph_data.number}")  # singular
print(f"State: {morph_data.state}")    # absolute

# Check word type
print(morph.is_verb("Vqp3ms"))        # True
print(morph.is_noun("Ncfsa"))         # True
print(morph.is_participle("Vqrpms"))  # True
```

### Greek Morphology

```python
from abba.morphology.greek_morphology import GreekMorphology

morph = GreekMorphology()

# Parse morphology code
morph_data = morph.parse_morph_code("V-PAI-3S")
print(f"Part of speech: {morph_data.part_of_speech}")  # verb
print(f"Tense: {morph_data.tense}")    # present
print(f"Voice: {morph_data.voice}")    # active
print(f"Mood: {morph_data.mood}")      # indicative
print(f"Person: {morph_data.person}")  # 3
print(f"Number: {morph_data.number}")  # singular

# Check if participle
print(morph.is_participle("V-PAN"))   # True
```

## Alignment

### Statistical Aligner

```python
from abba.alignment.statistical_aligner import StatisticalAligner

aligner = StatisticalAligner()

# Align two versions
source_verses = [...]  # List of verses from one version
target_verses = [...]  # List of verses from another version

alignments = aligner.align_versions(source_verses, target_verses)

for alignment in alignments:
    print(f"{alignment.source_id} -> {alignment.target_id}")
    print(f"Confidence: {alignment.confidence}")
```

### Bridge Tables

```python
from abba.alignment.bridge_tables import BridgeTable

# Create bridge table for complex verse mappings
bridge = BridgeTable()

# Add mapping for split verses
bridge.add_mapping(
    source_refs=["GEN.1.1"],
    target_refs=["GEN.1.1a", "GEN.1.1b"],
    mapping_type="split",
    confidence=1.0
)

# Query mappings
targets = bridge.get_target_refs("GEN.1.1")
```

## Annotations

### Annotation Engine

```python
from abba.annotations.annotation_engine import AnnotationEngine
from abba.annotations.models import AnnotationType
from abba.verse_id import VerseID

# Initialize engine
engine = AnnotationEngine()

# Annotate a verse
verse_id = VerseID("JHN", 3, 16)
text = "For God so loved the world..."

annotations = await engine.annotate_verse(
    verse_id=verse_id,
    text=text,
    annotation_types=[
        AnnotationType.THEOLOGICAL_THEME,
        AnnotationType.KEY_TERM,
        AnnotationType.CROSS_REFERENCE,
        AnnotationType.CHRISTOLOGICAL
    ]
)

# Process results
for annotation in annotations:
    print(f"{annotation.type}: {annotation.value}")
    print(f"Confidence: {annotation.confidence}")
    print(f"Metadata: {annotation.metadata}")
```

### Zero-Shot Classification

```python
from abba.annotations.zero_shot_classifier import ZeroShotClassifier

classifier = ZeroShotClassifier()

# Classify text
text = "For by grace you have been saved through faith"
labels = ["salvation", "faith", "grace", "works", "law"]

results = classifier.classify(text, labels)

for label, score in results:
    print(f"{label}: {score:.3f}")
```

### Few-Shot Classification

```python
from abba.annotations.few_shot_classifier import FewShotClassifier

classifier = FewShotClassifier()

# Train with examples
examples = [
    ("God is love", "love"),
    ("Love your neighbor", "love"),
    ("Faith comes by hearing", "faith"),
    ("By faith Abraham obeyed", "faith")
]

classifier.train(examples)

# Classify new text
result = classifier.classify("Love one another")
print(f"Classification: {result.label} (confidence: {result.confidence})")
```

## Timeline

### Event Management

```python
from abba.timeline.models import Event, TimePoint, EventType, CertaintyLevel, create_bce_date

# Create an event
event = Event(
    id="exodus",
    name="The Exodus",
    description="The Israelites leave Egypt",
    event_type=EventType.POINT,
    time_point=TimePoint(
        exact_date=create_bce_date(1446),  # 1446 BCE
        confidence=0.7
    ),
    certainty_level=CertaintyLevel.PROBABLE,
    verse_refs=[VerseID("EXO", 12, 31), VerseID("EXO", 12, 41)],
    categories=["religious", "political"],
    scholarly_sources=["Kitchen, K.A.", "Hoffmeier, J.K."]
)

# Create a time period
period = TimePeriod(
    period_id="judges",
    name="Period of Judges",
    start=TimePoint(exact_date=create_bce_date(1375)),
    end=TimePoint(exact_date=create_bce_date(1050))
)
```

### Timeline Visualization

```python
from abba.timeline.visualization import TimelineVisualizer, VisualizationConfig

# Configure visualization
config = VisualizationConfig(
    width=1400,
    height=600,
    show_uncertainty=True,
    color_scheme="default"
)

# Create visualizer
visualizer = TimelineVisualizer(config)

# Create visualization
events = [event1, event2, event3]  # List of Event objects
viz_data = visualizer.create_timeline_visualization(events)

# Export to SVG
svg = visualizer.export_svg(viz_data)

# Apply filters
from abba.timeline.filter import UserPreferences

preferences = UserPreferences(
    min_confidence=0.6,
    min_certainty_level=CertaintyLevel.POSSIBLE
)

filtered_viz = visualizer.create_timeline_visualization(events, preferences)
```

## Export

### Export Pipeline

```python
from abba.export.pipeline import ExportPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    source_dir="./data/canonical",
    output_dir="./output",
    parallel_workers=4
)

# Create pipeline
pipeline = ExportPipeline(config)

# Add exporters
from abba.export.sqlite_exporter import SQLiteExporter, SQLiteConfig
from abba.export.json_exporter import StaticJSONExporter, JSONConfig

# SQLite export
sqlite_config = SQLiteConfig(
    output_path="./output/bible.db",
    enable_fts5=True,
    create_indices=True
)
pipeline.add_exporter(SQLiteExporter(sqlite_config))

# JSON export
json_config = JSONConfig(
    output_path="./output/json",
    split_by_book=True,
    create_search_indices=True,
    compression="gzip"
)
pipeline.add_exporter(StaticJSONExporter(json_config))

# Run pipeline
result = await pipeline.run()
print(f"Exported {result.total_items} items in {result.duration}s")
```

### Minimal Exporters

```python
# Simple SQLite export
from abba.export.minimal_sqlite import MinimalSQLiteExporter

exporter = MinimalSQLiteExporter("verses.db")

# Add verses
exporter.add_verse("GEN.1.1", "GEN", 1, 1, "In the beginning...")
exporter.add_verse("GEN.1.2", "GEN", 1, 2, "And the earth was...")

# Finalize
exporter.finalize()

# Simple JSON export
from abba.export.minimal_json import MinimalJSONExporter

exporter = MinimalJSONExporter("./output")
exporter.add_verse("GEN.1.1", {"text": "In the beginning...", "book": "GEN"})
exporter.finalize()
```

## Canon Support

```python
from abba.canon.registry import CanonRegistry
from abba.canon.models import Canon

# Get canon registry
registry = CanonRegistry()

# Get specific canon
catholic_canon = registry.get_canon("catholic")
print(f"Books: {len(catholic_canon.books)}")
print(f"OT Books: {catholic_canon.ot_book_count}")
print(f"NT Books: {catholic_canon.nt_book_count}")

# Check if book is in canon
is_in_canon = catholic_canon.contains_book("TOB")  # Tobit
print(f"Tobit in Catholic canon: {is_in_canon}")

# Compare canons
protestant = registry.get_canon("protestant")
catholic = registry.get_canon("catholic")

comparison = registry.compare_canons(protestant, catholic)
print(f"Books only in Catholic: {comparison.unique_to_second}")
```

## Cross References

```python
from abba.cross_references.parser import CrossReferenceParser
from abba.cross_references.models import CrossReference, ReferenceType

# Parse cross references
parser = CrossReferenceParser()

# Parse from annotation
ref_text = "See also Matt 5:17; Luke 24:44"
refs = parser.parse_reference_text(ref_text, current_verse=VerseID("JHN", 1, 1))

for ref in refs:
    print(f"{ref.source_verse} -> {ref.target_verse}")
    print(f"Type: {ref.reference_type}")

# Create cross reference
xref = CrossReference(
    source_verse=VerseID("ROM", 3, 23),
    target_verse=VerseID("GEN", 3, 1),
    reference_type=ReferenceType.ALLUSION,
    confidence=0.8,
    metadata={"theme": "fall", "keyword": "sin"}
)
```

## Interlinear

```python
from abba.interlinear.interlinear_generator import InterlinearGenerator
from abba.interlinear.token_extractor import extract_hebrew_tokens

# Generate interlinear display
generator = InterlinearGenerator()

# With Hebrew verse and English translation
hebrew_verse = HebrewVerse(...)  # From Hebrew parser
translation = TranslationVerse(...)  # From translation parser

display = generator.generate_display(
    original_verse=hebrew_verse,
    translation_verse=translation,
    format=DisplayFormat.STANDARD
)

# Access display data
for row in display.rows:
    print(f"{row.original_text} - {row.transliteration}")
    print(f"  Lemma: {row.lemma}")
    print(f"  Gloss: {row.gloss}")
    print(f"  Translation: {' '.join(row.translation_words)}")

# Extract tokens
tokens = extract_hebrew_tokens(hebrew_verse)
for token in tokens:
    print(f"{token.text} ({token.strong_number})")
```

## Error Handling

All API methods follow consistent error handling patterns:

```python
from abba.exceptions import (
    ABBAError,
    ParsingError,
    ValidationError,
    ExportError,
    AlignmentError
)

try:
    # API operations
    result = parser.parse_verse(data)
except ParsingError as e:
    print(f"Parsing failed: {e.message}")
    print(f"Context: {e.context}")
except ValidationError as e:
    print(f"Validation failed: {e.errors}")
except ABBAError as e:
    # Generic ABBA error
    print(f"Operation failed: {e}")
```

## Async Operations

Many operations support async execution for better performance:

```python
import asyncio

async def process_verses():
    # Annotation engine is async
    engine = AnnotationEngine()
    
    tasks = []
    for verse_id, text in verses:
        task = engine.annotate_verse(verse_id, text)
        tasks.append(task)
    
    # Process in parallel
    results = await asyncio.gather(*tasks)
    return results

# Run async function
results = asyncio.run(process_verses())
```

## Configuration

Most modules support configuration through dataclasses:

```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    option1: str = "default"
    option2: int = 10
    option3: bool = True

# Pass to module
module = MyModule(MyConfig(
    option1="custom",
    option2=20
))
```

## Utility Functions

### Verse ID Management

```python
from abba.verse_id import parse_verse_id, parse_verse_range, format_verse_id, get_verse_parts

# Parse verse references
verse = parse_verse_id("ROM.3.23")  # Returns VerseID object
verse_range = parse_verse_range("ROM.3.23-25")  # Returns VerseRange object

# Format verse ID
formatted = format_verse_id(verse)  # Returns "ROM.3.23"

# Get verse parts (for split verses)
parts = get_verse_parts("ROM.3.23")  # Returns [ROM.3.23, ROM.3.23a, ROM.3.23b]
```

### Book Code Utilities

```python
from abba.book_codes import normalize_book_name, get_book_name, is_valid_book_code

# Normalize book names
code = normalize_book_name("Genesis")  # Returns "GEN"
code = normalize_book_name("1 Corinthians")  # Returns "1CO"

# Get book names
name = get_book_name("GEN")  # Returns "Genesis"
name = get_book_name("GEN", form="abbr")  # Returns "Gen"

# Validate book codes
is_valid = is_valid_book_code("GEN")  # Returns True
is_valid = is_valid_book_code("XXX")  # Returns False
```

### Unicode Utilities

```python
from abba.language.unicode_utils import UnicodeNormalizer, HebrewNormalizer, GreekNormalizer

# General normalization
normalizer = UnicodeNormalizer()
clean_text = normalizer.clean_text("Text with  extra   spaces")
normalized = normalizer.normalize_nfc("é")  # Composed form

# Hebrew specific
hebrew = HebrewNormalizer()
stripped = hebrew.strip_hebrew_points("בְּרֵאשִׁית")  # Returns "בראשית"
normalized = hebrew.normalize_final_forms("אבגדך")  # Normalizes final letters

# Greek specific
greek = GreekNormalizer()
stripped = greek.strip_greek_accents("λόγος")  # Returns "λογος"
normalized = greek.normalize_final_sigma("ΛΟΓΟΣ")  # Handles final sigma
```

## Best Practices

### 1. Error Handling

Always handle potential errors appropriately:

```python
from abba.exceptions import ABBAError, ValidationError

try:
    result = parse_verse_id("invalid reference")
except ValidationError as e:
    # Handle invalid input
    logger.error(f"Invalid verse reference: {e}")
    return None
```

### 2. Resource Management

Use context managers for resources:

```python
from abba.export.sqlite_exporter import SQLiteExporter

# Automatic cleanup
with SQLiteExporter(config) as exporter:
    exporter.export_verses(verses)
# Connection automatically closed
```

### 3. Async Best Practices

Use async for I/O operations:

```python
import asyncio
from abba.annotations.annotation_engine import AnnotationEngine

async def annotate_batch(verses):
    engine = AnnotationEngine()
    
    # Process in batches for memory efficiency
    batch_size = 100
    results = []
    
    for i in range(0, len(verses), batch_size):
        batch = verses[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            engine.annotate_verse(v.id, v.text) 
            for v in batch
        ])
        results.extend(batch_results)
    
    return results
```

### 4. Configuration Management

Use environment variables for sensitive data:

```python
import os
from abba.export.opensearch_exporter import OpenSearchConfig

config = OpenSearchConfig(
    cluster_url=os.getenv("OPENSEARCH_URL", "http://localhost:9200"),
    username=os.getenv("OPENSEARCH_USER"),
    password=os.getenv("OPENSEARCH_PASS"),
    index_name="bible-verses"
)
```

### 5. Performance Optimization

Cache expensive operations:

```python
from functools import lru_cache
from abba.morphology.hebrew_morphology import HebrewMorphology

@lru_cache(maxsize=1000)
def parse_cached_morph(morph_code: str):
    morph = HebrewMorphology()
    return morph.parse_morph_code(morph_code)
```

## Version Compatibility

ABBA supports Python 3.11, 3.12, and 3.13. Some features require additional dependencies:

- ML features: `pip install abba[ml]`
- OpenSearch: `pip install abba[opensearch]`
- Graph export: `pip install abba[graph]`
- All features: `pip install abba[all]`

## API Versioning

The ABBA API follows semantic versioning:

- Major version: Breaking changes
- Minor version: New features, backward compatible
- Patch version: Bug fixes

Current stable API version: 1.0.0