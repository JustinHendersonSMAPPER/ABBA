# ABBA Source Code

This directory contains the core source code for the ABBA project, organized in a clean, modular structure.

## Main Entry Point

### `main.py`

The master script that orchestrates the entire ABBA pipeline:

1. **Downloads bible.db** from https://bible.helloao.org/bible.db if needed
2. **Extracts translations** from the database to JSON files
3. **Downloads Strong's lexicons** for Hebrew and Greek
4. **Builds alignment models** with full Strong's concordance
5. **Analyzes translation coverage** and generates reports
6. **Future**: Generate enriched Bible exports

Usage:
```bash
python src/main.py
# or
./src/main.py
```

## Supporting Modules

### `abba_data_downloader.py`
- Downloads bible.db from the server
- Extracts translations to `data/sources/translations/`
- Downloads and converts Strong's lexicons
- Handles different database schemas automatically

### `abba_model_builder.py`
- Builds Hebrew and Greek alignment models
- Extracts translations from Strong's concordance
- Adds manual alignments for high-frequency words
- Creates enhanced models with ~8,674 Hebrew and ~5,624 Greek mappings

### `abba_coverage_analyzer.py`
- Analyzes translation coverage against alignment models
- Works with extracted JSON translation files
- Generates markdown reports showing coverage percentages
- Separates Old Testament (Hebrew) and New Testament (Greek) analysis

## Data Flow

1. **Input**: bible.db (downloaded automatically)
2. **Extraction**: Translations saved as JSON in `data/sources/translations/`
3. **Models**: Built in `models/biblical_alignment/`
4. **Output**: Coverage report in `translation_coverage_report.md`

## Directory Structure

```
data/
├── sources/
│   ├── db/
│   │   └── bible.db          # Downloaded automatically
│   ├── translations/         # Extracted JSON files
│   │   ├── ENG_KJV.json
│   │   ├── ENG_NIV.json
│   │   └── ...
│   └── lexicons/            # Strong's concordance
│       ├── strongs_hebrew.json
│       └── strongs_greek.json

models/
└── biblical_alignment/
    ├── hebrew_english_enhanced.json
    └── greek_english_enhanced.json
```

## Design Principles

1. **Single Entry Point**: `main.py` orchestrates everything
2. **Modular Design**: Each module has a specific responsibility
3. **Automatic Downloads**: No manual setup required
4. **Clear Data Flow**: Database → JSON → Models → Analysis
5. **Extensible**: Easy to add new functionality to main.py

## Future Extensions

As more functionality is implemented, `main.py` will be extended to:
- Generate cross-references
- Add morphological analysis
- Include timeline data
- Export to multiple formats (SQLite, JSON, OpenSearch)
- Create interlinear displays