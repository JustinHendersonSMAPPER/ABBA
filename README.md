# ABBA - Annotated Bible and Background Analysis

A comprehensive Python framework for biblical text analysis that provides instant access to original Hebrew, Aramaic, and Greek texts with morphological analysis, semantic search capabilities, and LLM-enhanced concept mapping.

## Quick Start

```bash
# Install dependencies (creates .venv via uv)
uv sync

# Initialize database and download biblical texts
uv run python abba/main.py

# List available translations
uv run python abba/main.py --list

# Validate concept definitions (requires Ollama)
uv run python abba/main.py --validate-concepts

# Map concepts to verses with LLM analysis
uv run python abba/main.py --map-concepts

# View database examples
uv run python claude/scripts/simple_db_examples.py
```

## Features

### 📚 Current Implementation

#### Phase 1: Database Foundation ✅
- **SQLite Database Architecture**: Fast, local database with structured biblical data
- **Multi-Translation Support**: Import and query 1200+ Bible translations
- **Original Language Integration**: Complete Hebrew (TAHOT) and Greek (TAGNT) texts
- **Morphological Analysis**: Detailed grammatical information for every word
- **Strong's Concordance**: Extended Strong's numbers with lexicon definitions
- **Import Tracking**: Intelligent tracking prevents re-processing of data
- **Cross-Reference System**: Navigate between related passages
- **Verse-Level Alignment**: Pragmatic alignment system across all translations

#### Phase 2: Semantic Search ✅
- **Original Language Embeddings**: 29,126 canonical verse embeddings using Hebrew/Greek/Aramaic
- **Universal Semantic Search**: One search works across all 1,204 translations
- **Vector Database**: ChromaDB integration for fast similarity search
- **Embedding Models**: Multilingual-E5-base for cross-linguistic accuracy
- **Performance Optimized**: Sub-10ms semantic queries

#### Phase 4: Concept Mapping ✅
- **LLM-Enhanced Analysis**: Ollama integration for intelligent concept validation
- **User-Defined Concepts**: YAML-based theological concept definitions
- **Three-Phase Mapping**: Traditional → LLM validation → comprehensive scanning
- **Full Traceability**: Every mapping decision logged with reasoning
- **Biblical Accuracy**: Based on Strong's numbers and original language terms

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

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- ~2GB disk space for full database

### Install with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ABBA.git
cd ABBA

# Install dependencies (creates and manages .venv automatically)
uv sync

# Run commands inside the environment with `uv run ...`, e.g.:
uv run python abba/main.py --list
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

# Concept mapping commands (requires Ollama)
python abba/main.py --validate-concepts      # Validate concepts.yaml definitions
python abba/main.py --map-concepts           # Map concepts to verses with LLM
python abba/main.py --concept-report         # Generate detailed mapping report
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

# Ollama Configuration (for concept mapping)
ABBA_OLLAMA_HOST=http://localhost:11434
ABBA_OLLAMA_SEMANTIC_MODELS=llama3
ABBA_OLLAMA_CONSENSUS_THRESHOLD=0.7
ABBA_OLLAMA_TIMEOUT=30

# Concept Mapping Configuration
ABBA_CONCEPTS_FILE=abba/concepts.yaml
ABBA_CONCEPT_VALIDATION_ENABLED=true
ABBA_CONCEPT_VALIDATION_BATCH_SIZE=100

# Vector Database Configuration (for semantic search)
ABBA_VECTOR_DB_TYPE=chromadb
ABBA_VECTOR_DB_PATH=bible_data/vectors
ABBA_VECTOR_DIMENSIONS=768
ABBA_EMBEDDING_MODEL_MULTILINGUAL=intfloat/multilingual-e5-base
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

### Concept Mapping (LLM Integration)

ABBA includes a powerful concept mapping system that uses LLM analysis to map theological concepts to biblical verses with high accuracy.

#### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Start Ollama**: Run `ollama serve`
3. **Download Model**: Run `ollama pull llama3` (or your preferred model)

#### Creating a Concepts File

Create `abba/concepts.yaml` with your theological concepts:

```yaml
concepts:
  - name: "divine_love"
    description: >
      God's love, mercy, compassion, and loving-kindness toward humanity,
      including covenant love (hesed) and unconditional love (agape).
    hebrew_terms:
      - "אהב"      # ahav - to love
      - "חסד"      # hesed - loving-kindness, mercy
      - "רחם"      # racham - to have compassion
    greek_terms:
      - "ἀγάπη"    # agape - divine love
      - "ἀγαπάω"   # agapao - to love
      - "ἔλεος"    # eleos - mercy, compassion
    strongs_numbers:
      - "H157"     # ahav
      - "H2617"    # hesed
      - "G26"      # agape
      - "G25"      # agapao
    keywords:
      - "love"
      - "mercy"
      - "compassion"
      - "loving-kindness"

  - name: "faith_trust" 
    description: >
      Faith, trust, belief, and confidence in God, including faithfulness
      and reliability in human relationships.
    hebrew_terms:
      - "אמן"      # aman - to believe, trust
      - "בטח"      # batach - to trust, rely upon
    greek_terms:
      - "πίστις"   # pistis - faith, trust
      - "πιστεύω"  # pisteuo - to believe
    strongs_numbers:
      - "H539"     # aman
      - "H982"     # batach
      - "G4102"    # pistis
      - "G4100"    # pisteuo
    keywords:
      - "faith"
      - "trust"
      - "believe"
      - "confidence"

# Configuration for concept processing
config:
  validation:
    enabled: true
    confidence_threshold: 0.6
    consensus_threshold: 0.7
  comprehensive_scan:
    enabled: true
    relevance_threshold: 0.7
```

#### Using Concept Mapping

```bash
# Validate your concepts file
python abba/main.py --validate-concepts

# Map all concepts to verses (time-intensive)
python abba/main.py --map-concepts

# Generate detailed report with analysis
python abba/main.py --map-concepts --concept-report

# Use custom concepts file
python abba/main.py --concepts-file my_concepts.yaml --map-concepts

# Customize LLM settings
python abba/main.py --ollama-models llama3 --ollama-consensus 0.8 --map-concepts
```

#### How Concept Mapping Works

1. **Traditional Mapping**: Find verses using Strong's numbers, Hebrew/Greek terms, and keywords
2. **LLM Validation**: Ollama analyzes each match to remove false positives
3. **Comprehensive Scanning**: Ollama scans all 29,126 verses to find additional relevant passages
4. **Database Storage**: Results saved with confidence scores, reasoning, and full traceability

Each concept mapping provides:
- **Validated matches**: Traditional matches confirmed by LLM
- **Discovered verses**: Additional relevant verses found by comprehensive scanning
- **False positives**: Traditional matches rejected by LLM with explanations
- **Confidence scores**: Numerical relevance and confidence ratings
- **Reasoning**: Detailed LLM explanations for each decision

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
uv run pytest tests/test_specific.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

```bash
# Run all quality checks
nox

# Individual checks
nox -s lint      # ruff format --check + ruff check (black/isort/flake8/pylint)
nox -s typing    # pyright (type checking)
nox -s security  # ruff flake8-bandit S rules (security scan)
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

### Completed ✅
- **Phase 1**: Database foundation with SQLite and STEPBible integration
- **Phase 2**: Vector database integration for semantic search
- **Phase 4**: Concept mapping system with LLM validation

### In Progress / Planned
- **Phase 3**: Unified search API combining exact and semantic search
- **Phase 5**: Performance optimization and advanced caching
- **Phase 6**: API development and web interface
- **Advanced Features**: 
  - Real-time concept discovery from user queries
  - Integration with commentary databases
  - Visualization tools for semantic relationships
  - Mobile app API endpoints

See `claude/checklist.md` for detailed development phases.

## License

[License information here]

## Contributing

[Contributing guidelines here]

## Acknowledgments

- STEPBible for Hebrew and Greek texts
- bible.helloao.org for the comprehensive translation database