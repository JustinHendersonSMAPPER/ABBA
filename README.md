# ABBA - Annotated Bible and Background Analysis

A comprehensive Python framework for biblical text analysis that provides instant access to original Hebrew, Aramaic, and Greek texts with morphological analysis, cross-translation verse alignment, and comprehensive linguistic data.

## Quick Start

```bash
# Install dependencies
poetry install

# Initialize database and download biblical texts
poetry run python abba/main.py

# List available translations
poetry run python abba/main.py --list

# View database examples
poetry run python claude/scripts/simple_db_examples.py
```

## Features

### 📚 Current Implementation (Phase 1 - Complete)

- **SQLite Database Architecture**: Fast, local database with structured biblical data
- **Multi-Translation Support**: Import and query 1200+ Bible translations
- **Original Language Integration**: Complete Hebrew (TAHOT) and Greek (TAGNT) texts
- **Morphological Analysis**: Detailed grammatical information for every word
- **Strong's Concordance**: Extended Strong's numbers with lexicon definitions
- **Import Tracking**: Intelligent tracking prevents re-processing of data
- **Cross-Reference System**: Navigate between related passages
- **Verse-Level Alignment**: Pragmatic alignment system across all translations

### 🔧 Core Components

1. **Database Schema**
   - `verses`: All biblical text organized by translation, book, chapter, verse
   - `words`: Original Hebrew/Greek words with morphology and Strong's numbers
   - `lexicon`: Dictionary definitions for Hebrew and Greek terms
   - `morphology`: Grammatical code explanations
   - `books`: Book metadata and ordering information

2. **Data Sources**
   - **bible.db**: Source database with 1200+ translations
   - **STEPBible**: Academic-quality Hebrew/Greek texts with linguistic tagging
   - **Import Tracking**: JSON-based progress tracking at `.import_status.json`

3. **Configuration System**
   - Priority: CLI arguments > Environment variables > Config file > Defaults
   - Supports `.env` files for persistent settings
   - JSON configuration files for complex setups

## Installation

### Prerequisites

- Python 3.8+
- Poetry (recommended) or pip
- ~2GB disk space for full database

### Install with Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ABBA.git
cd ABBA

# Install dependencies
poetry install

# Activate shell
poetry shell
```

### Install with pip

```bash
pip install -e .
```

## Usage

### Basic Commands

```bash
# Initialize database and import default translations
python abba/main.py

# List all available translations
python abba/main.py --list

# Import specific translations only
python abba/main.py --translations KJV ASV ESV

# Force re-download and rebuild database
python abba/main.py --force-download --rebuild-db

# Use custom data directory
python abba/main.py --data-dir /path/to/data

# Quiet mode (minimal output)
python abba/main.py --quiet

# Verbose mode (detailed progress)
python abba/main.py --verbose
```

### Configuration

#### Environment Variables

Create a `.env` file in the project root:

```bash
# Data directory location (default: ./bible_data)
ABBA_DATA_DIR=/path/to/bible/data

# Specific translations to import (comma-separated)
ABBA_TRANSLATIONS=KJV,ASV,ESV,eng_bbe

# Force download even if files exist
ABBA_FORCE_DOWNLOAD=false

# Rebuild database from scratch
ABBA_REBUILD_DB=false

# Output verbosity
ABBA_VERBOSE=false
ABBA_QUIET=false

# Custom bible.db URL (if using alternative source)
ABBA_BIBLE_DB_URL=https://bible.helloao.org/bible.db
```

#### Configuration File

Create a JSON configuration file:

```json
{
  "data_dir": "/custom/path/to/data",
  "translations": ["KJV", "ASV", "ENGWEBP"],
  "download_enabled": true,
  "force_download": false,
  "rebuild_db": false,
  "verbose": true,
  "quiet": false
}
```

Use with: `python abba/main.py --config-file my_config.json`

### Database Queries

Once initialized, the SQLite database (`bible_data/abba.db`) can be queried directly:

```sql
-- Get a specific verse
SELECT * FROM verses 
WHERE translation_id = 'eng_kjv' 
  AND book_id = 43  -- John
  AND chapter = 3 
  AND verse = 16;

-- Find original Greek words for a verse
SELECT w.*, l.gloss, l.definition 
FROM words w
LEFT JOIN lexicon l ON SUBSTR(w.translation, 1, 5) = l.strongs_number
WHERE w.book = 'Jhn' AND w.chapter = 1 AND w.verse = 1
ORDER BY w.word_num;

-- Search for all occurrences of a Strong's number
SELECT book, chapter, verse, greek_text, strongs_primary 
FROM words 
WHERE translation LIKE 'G3056%'  -- logos
LIMIT 10;
```

See `claude/scripts/simple_db_examples.py` for comprehensive query examples.

## Project Structure

```
ABBA/
├── abba/                       # Main package
│   ├── __init__.py
│   ├── main.py                # Entry point
│   ├── bible_extractor.py     # Core extraction logic
│   ├── config.py              # Configuration management
│   ├── cli.py                 # CLI argument parsing
│   ├── env.py                 # Environment variable handling
│   └── database/
│       ├── __init__.py
│       ├── sqlite_manager.py  # Database operations
│       └── import_tracker.py  # Import progress tracking
├── bible_data/                # Data directory (git-ignored)
│   ├── abba.db               # Main SQLite database
│   ├── bible.db              # Source database
│   ├── .import_status.json   # Import tracking
│   └── stepbible/            # STEPBible original language files
├── claude/                    # Claude-specific files
│   ├── checklist.md          # Development checklist
│   └── scripts/              # Example and debug scripts
├── docs/                      # Documentation
│   ├── DATABASE_DESIGN.md    # Database schema details
│   ├── CONFIGURATION.md      # Configuration guide
│   └── QUERYING_DATA.md      # Query examples and guides
├── tests/                     # Test suite
├── pyproject.toml            # Poetry configuration
├── .env.example              # Environment template
└── README.md                 # This file
```

## Data Schema

### Core Tables

1. **verses** - Biblical text by translation
   - `translation_id`: Translation identifier (e.g., 'eng_kjv')
   - `book_id`: Canonical book number (1-66)
   - `chapter`: Chapter number
   - `verse`: Verse number
   - `text`: Verse text

2. **words** - Original language word analysis
   - `word_ref`: Unique reference (e.g., 'Gen.1.1.1')
   - `hebrew_text` / `greek_text`: Original text
   - `transliteration`: Romanized form
   - `translation`: Strong's number + morphology code
   - `strongs_primary`: Primary Strong's number
   - `morphology_code`: Grammatical parsing code

3. **lexicon** - Dictionary definitions
   - `strongs_number`: Strong's concordance number
   - `original_word`: Dictionary form
   - `transliteration`: Standard transliteration
   - `gloss`: Brief definition
   - `definition`: Full definition

4. **morphology** - Grammar code explanations
   - `code`: Morphology code
   - `description`: Human-readable explanation
   - `language`: hebrew or greek

## Development

### Running Tests

```bash
# Run all tests with coverage
nox -s tests

# Run specific test
poetry run pytest tests/test_specific.py

# Run with verbose output
poetry run pytest -v
```

### Code Quality

```bash
# Run all quality checks
nox

# Individual checks
nox -s lint      # Linting
nox -s typing    # Type checking
nox -s security  # Security scan
```

### Adding New Features

1. Check `claude/checklist.md` for development phases
2. Update tests to maintain >95% coverage
3. Run all quality checks before committing
4. Update documentation as needed

## Troubleshooting

### Common Issues

1. **"bible.db not found"**
   - Run with `--force-download` flag
   - Check internet connection
   - Verify `ABBA_BIBLE_DB_URL` if using custom source

2. **"Cannot operate on a closed database"**
   - Database connection issue, restart the import
   - Check disk space
   - Ensure write permissions on data directory

3. **Import seems stuck**
   - Check `.import_status.json` for progress
   - Use `--verbose` flag for detailed output
   - Large translations can take several minutes

4. **Missing STEPBible data**
   - Downloads happen automatically on first run
   - Check `bible_data/stepbible/` directory
   - Internet connection required for initial download

### Debug Mode

```bash
# Maximum verbosity
python abba/main.py --verbose

# Check import status
cat bible_data/.import_status.json | python -m json.tool

# Test database connection
python claude/scripts/check_schema.py
```

## Future Roadmap

- **Phase 2**: Vector database integration for semantic search
- **Phase 3**: Ollama-powered concept extraction and mapping
- **Phase 4**: Advanced linguistic analysis and word studies
- **Phase 5**: Performance optimization and caching
- **Phase 6**: API development and web interface

See `claude/checklist.md` for detailed development phases.

## License

[License information here]

## Contributing

[Contributing guidelines here]

## Acknowledgments

- STEPBible for Hebrew and Greek texts
- bible.helloao.org for the comprehensive translation database