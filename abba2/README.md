# ABBA 2.0 - Annotated Bible and Background Analysis

Free Biblical Language Analysis System with Academic Rigor

## Vision

ABBA 2.0 provides completely free, transparent, and academically rigorous biblical language resources that rival expensive proprietary tools like BDAG/HALOT. No paywalls, no restrictions - just knowledge for everyone.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/abba
cd abba/abba2

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Basic Usage

```bash
# 1. Download biblical sources (Greek/Hebrew lexicons, texts, treebanks)
abba2 sources download

# 2. Validate downloaded data
abba2 sources validate

# 3. Check system configuration
abba2 info

# List available sources
abba2 sources list
```

## Features

### ✅ Completed (Phase 1)
- **Data Acquisition System**: Downloads and validates 15+ free biblical sources
- **Source Manifest**: YAML-based source definitions with checksums
- **Validation Framework**: Structural and integrity validation for all formats
- **CLI Interface**: Rich terminal UI with progress tracking
- **Configuration System**: Environment variables and config files
- **Clean Architecture**: Modular, testable, maintainable design

### 🚧 In Progress (Phase 2-3)
- **Lexicon Processing**: Parse and normalize multiple lexicon formats
- **Consensus Engine**: Combine multiple sources with confidence scoring
- **Syntactic Analysis**: Treebank integration for grammatical relationships
- **Semantic Search**: Embedding-based concept discovery

### 📋 Planned (Phase 4-6)
- **API Layer**: RESTful JSON API for all functionality
- **Vue.js UI**: Interactive web interface for testing
- **Literary Analysis**: Genre, parallelism, literary devices
- **Historical Context**: Archaeological and cultural background
- **Community Features**: Contributions, corrections, enhancements

## Data Sources

ABBA 2.0 integrates multiple public domain and open-source biblical resources:

### Greek Resources
- **Abbott-Smith** (1922): Classical-focused lexicon
- **Thayer** (1889): Comprehensive Greek lexicon
- **Dodson** (2010): Modern, concise definitions
- **LSJ** (1940): Classical Greek context
- **MorphGNT**: Morphological analysis
- **PROIEL/Lowfat**: Syntactic treebanks

### Hebrew Resources
- **BDB** (1906): Brown-Driver-Briggs lexicon
- **Strong's** (1890): Universal reference system
- **OSHB**: Open Scriptures Hebrew Bible
- **BHSA**: Hebrew syntax and linguistics

## Architecture

```
┌─────────────────────────────────────────────┐
│            CLI / Vue.js UI                  │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│              RESTful API                    │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│        Application Services                 │
│  (Consensus, Analysis, Search, Synthesis)   │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│           Domain Models                     │
│    (Lexicon, Morphology, Syntax, Text)     │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│         Infrastructure Layer                │
│    (SQLite, Embeddings, Cache, Files)      │
└─────────────────────────────────────────────┘
```

## Configuration

ABBA 2.0 uses a hierarchical configuration system:

1. **Environment Variables** (prefix: `ABBA_`)
2. **Config Files** (JSON/YAML)
3. **CLI Arguments**
4. **Defaults**

Common settings:
```bash
export ABBA_DATA_DIR="~/.abba2/data"
export ABBA_PARALLEL_WORKERS=8
export ABBA_LOG_LEVEL="INFO"
```

## Development

### Project Structure
```
abba2/
├── data_acquisition/   # Download and validate sources
├── parsing/           # Parse various formats
├── synthesis/         # Consensus and enhancement
├── analysis/          # Word, verse, passage analysis
├── storage/           # Database and caching
├── api/              # RESTful endpoints
├── tests/            # Comprehensive test suite
└── docs/             # Documentation
```

### Testing
```bash
# Run all tests with coverage
poetry run pytest --cov=abba2

# Run specific test module
poetry run pytest tests/test_downloader.py

# Run with verbose output
poetry run pytest -v
```

### Code Quality
```bash
# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy abba2

# Linting
poetry run pylint abba2
poetry run flake8 abba2

# Security check
poetry run bandit -r abba2
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Ways to Help
- **Scholars**: Review linguistic analyses, contribute expertise
- **Developers**: Add features, fix bugs, optimize performance
- **Users**: Report issues, suggest improvements, provide feedback
- **Writers**: Improve documentation, create tutorials

## Philosophy

1. **Free Forever**: No paywalls, ever
2. **Academically Rigorous**: Multiple sources, transparent methodology
3. **Community-Driven**: Built by and for Bible students
4. **Fully Transparent**: Every algorithm and decision documented
5. **User-Focused**: Designed for actual study needs

## Roadmap

### Phase 1: Foundation (Weeks 1-4) ✅
- [x] Clean project structure
- [x] Data acquisition system
- [x] Source validation
- [x] Basic CLI

### Phase 2: Core Lexicon (Weeks 5-8) 🚧
- [ ] Parse all lexicon sources
- [ ] Build consensus algorithm
- [ ] Implement confidence scoring
- [ ] Create word study API

### Phase 3: Syntactic Analysis (Weeks 9-12)
- [ ] Integrate treebank data
- [ ] Build syntactic analyzer
- [ ] Implement clause detection
- [ ] Add emphasis recognition

### Phase 4: Semantic & Conceptual (Weeks 13-16)
- [ ] Implement semantic domains
- [ ] Build concept search
- [ ] Add statistical analysis
- [ ] Create pattern detection

### Phase 5: Advanced Features (Weeks 17-20)
- [ ] Add literary analysis
- [ ] Implement manuscript variants
- [ ] Build cultural context system
- [ ] Add theological themes

### Phase 6: Polish & Release (Weeks 21-24)
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Public beta release

## License

MIT License - Use freely for any purpose

## Citation

If using ABBA in academic work:
```
ABBA 2.0: Annotated Bible and Background Analysis
Free Biblical Language Analysis System
[https://github.com/yourusername/abba]
Accessed: [Date]
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/abba/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/abba/discussions)
- **Documentation**: [Full Docs](./docs/)

---

*"The Bible belongs to humanity. Understanding it shouldn't require a subscription."*