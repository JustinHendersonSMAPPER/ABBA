# ABBA Quick Start Guide

Get started with ABBA in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/jhenderson/ABBA.git
cd ABBA

# Install with Poetry
poetry install
```

## Basic Usage

### 1. Export Complete Bible Data

The ABBA CLI can export complete Bible translations with all enrichments:

```bash
# Export KJV Bible to JSON with all enrichments
python -m abba --output kjv_bible --format json --translations ENG_KJV

# Export multiple translations at once
python -m abba --output multi_bible --format json --translations ENG_KJV ENG_ASV ENG_WEB

# Export to SQLite database
python -m abba --output bible.db --format sqlite --translations ENG_KJV

# Use custom data directory
python -m abba --data-dir /path/to/data --output export --format json
```

### Output Structure

JSON exports are organized by translation:
```
kjv_bible/
└── ENG_KJV/
    ├── manifest.json      # Translation metadata
    ├── Gen.json          # Genesis with all verses
    ├── Exod.json         # Exodus with all verses
    ├── ...               # All 66 books
    └── Rev.json          # Revelation with all verses
```

Each verse includes:
- Complete text
- Cross-references to related verses
- Theological annotations
- Timeline events (where applicable)
- Book metadata (name, testament, canonical order)

Example verse structure:
```json
{
  "verse": 1,
  "text": "In the beginning God created the heaven and the earth.",
  "verse_id": "Gen.1.1",
  "cross_references": [
    {
      "target": "JHN.1.1",
      "type": "thematic_parallel",
      "relationship": "parallels",
      "confidence": 0.95,
      "topic_tags": ["creation", "beginning"],
      "theological_theme": "Creation and Logos"
    }
  ],
  "annotations": [
    {
      "type": "theological_theme",
      "value": "Creation",
      "confidence": 0.9
    }
  ],
  "timeline_events": [
    {
      "id": "creation",
      "name": "Creation",
      "description": "The creation of the world",
      "date": "0996-01-01T00:00:00",
      "confidence": 0.3,
      "categories": ["theological", "cosmological"]
    }
  ],
  "metadata": {
    "book_name": "Genesis",
    "testament": "OT",
    "canonical_order": 1
  }
}
```

### 2. Using the Python API

For custom integrations:

```python
from abba.export.minimal_sqlite import MinimalSQLiteExporter
from abba.export.minimal_json import MinimalJSONExporter

# Export to SQLite
exporter = MinimalSQLiteExporter("verses.db")
exporter.add_verse("GEN.1.1", "GEN", 1, 1, "In the beginning...")
exporter.finalize()

# Export to JSON
exporter = MinimalJSONExporter("verses.json")
exporter.add_verse("GEN.1.1", "GEN", 1, 1, "In the beginning...")
exporter.finalize()
```

### 3. Working with Exported Data

#### Query SQLite Database
```bash
# Count total verses
sqlite3 bible.db "SELECT COUNT(*) FROM verses;"

# Search for specific text
sqlite3 bible.db "SELECT * FROM verses WHERE text LIKE '%love%' LIMIT 5;"

# Get all verses from a book
sqlite3 bible.db "SELECT * FROM verses WHERE book = 'John' ORDER BY chapter, verse;"
```

#### Read JSON Export
```python
import json
from pathlib import Path

# Load a book
with open("kjv_bible/ENG_KJV/Gen.json") as f:
    genesis = json.load(f)
    
# Access first verse
first_verse = genesis["chapters"][0]["verses"][0]
print(f"{first_verse['verse_id']}: {first_verse['text']}")

# Find verses with cross-references
for chapter in genesis["chapters"]:
    for verse in chapter["verses"]:
        if verse["cross_references"]:
            print(f"{verse['verse_id']} has {len(verse['cross_references'])} cross-references")
```

## Available Translations

The system includes many translations. Some popular ones:
- `ENG_KJV` - King James Version
- `ENG_ASV` - American Standard Version
- `ENG_WEB` - World English Bible
- `ENG_BBE` - Bible in Basic English
- `ENG_YLT` - Young's Literal Translation

List all available translations:
```bash
ls data/sources/translations/*.json | wc -l  # Count available translations
```

## Features

### ✅ Currently Working
- Complete Bible export (all 66 books)
- Multiple translation support
- Cross-reference linking
- Theological annotations
- Timeline events with BCE dates
- SQLite and JSON export formats
- Book metadata and canonical ordering

### 🚧 In Development
- Hebrew/Greek morphological data integration
- Advanced ML-powered annotations
- Full OpenSearch export
- Graph database export
- Manuscript variant tracking

## Examples

### Export and Search Example
```bash
# Export KJV to SQLite
python -m abba --output kjv.db --format sqlite --translations ENG_KJV

# Search for verses about love
sqlite3 kjv.db "SELECT verse_id, text FROM verses WHERE text LIKE '%love%' LIMIT 10;"

# Count verses per book
sqlite3 kjv.db "SELECT book, COUNT(*) as verse_count FROM verses GROUP BY book ORDER BY verse_count DESC;"
```

### Multi-Translation Export
```bash
# Export multiple translations for comparison
python -m abba --output compare --format json --translations ENG_KJV ENG_ASV ENG_WEB

# Directory structure:
# compare/
#   ├── ENG_KJV/
#   ├── ENG_ASV/
#   └── ENG_WEB/
```

## Getting Help

- Full documentation: `/docs` folder
- API reference: `/docs/API.md`
- Architecture guide: `/docs/ARCHITECTURE.md`
- Report issues: GitHub Issues

## Next Steps

1. Try exporting your first Bible translation
2. Explore the enriched data structure
3. Build applications using the exported data
4. Contribute to the project on GitHub

Happy coding with ABBA!