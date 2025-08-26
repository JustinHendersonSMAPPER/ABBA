# ABBA Documentation

Welcome to the ABBA (Annotated Bible and Background Analysis) documentation. This guide covers all aspects of the system from configuration to advanced features.

## Table of Contents

### Getting Started
- [Configuration Guide](CONFIGURATION.md) - Complete configuration reference including CLI arguments, environment variables, and config files
- [Database Design](DATABASE_DESIGN.md) - Schema and architecture of the ABBA database
- [Database Implementation](DATABASE_IMPLEMENTATION.md) - Technical details of the database layer

### Core Features
- [Canon System](CANON_SYSTEM.md) - **NEW**: Canon-aware import system supporting Protestant, Catholic, Orthodox, and other biblical traditions
- [Validation System](VALIDATION_SYSTEM.md) - Data integrity validation including hash-based verification and canon-aware book validation
- [STEPBible Integration](STEPBible.md) - Working with Hebrew and Greek texts from STEPBible

### Advanced Topics
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md) - Parallel processing and optimization strategies
- [Querying Data](QUERYING_DATA.md) - How to query and analyze biblical data

## Quick Start

### Basic Usage
```bash
# Download and import default translations
python abba/main.py

# Import specific translations
python abba/main.py --translations KJV ESV NIV

# Generate embeddings for semantic search
python abba/main.py --embed-verses --embed-words
```

### Configuration Priority
1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file
4. Default values (lowest priority)

### Key Features

#### Canon-Aware Import
The system intelligently recognizes different biblical canons:
- Protestant (66 books)
- Catholic (73 books) 
- Orthodox (76+ books)
- Ethiopian (81 books)
- Jewish/Hebrew (39 books)

This eliminates false warnings when importing Bibles with deuterocanonical books.

#### Parallel Processing
- Automatic CPU core detection
- Configurable worker count
- Thread/process selection based on task type

#### Data Validation
- MurmurHash3-based content validation
- Automatic cleanup of interrupted operations
- Canon-aware book validation

#### Semantic Search
- Verse embeddings using E5-large-v2 (1024D)
- Word embeddings using multilingual-E5-base (768D)
- ChromaDB vector storage

## Environment Variables

All environment variables use the `ABBA_` prefix:

```bash
ABBA_DATA_DIR=/path/to/data
ABBA_TRANSLATIONS=KJV,ESV,NIV
ABBA_PARALLEL_WORKERS=8
ABBA_VECTOR_DB_PATH=/path/to/vectors
```

See [Configuration Guide](CONFIGURATION.md) for the complete list.

## Common Tasks

### Import Catholic Bible Without Warnings
```bash
python abba/main.py --translations NABRE
```

### Rebuild Everything
```bash
python abba/main.py --rebuild-db --force-reembed
```

### Production Setup
```bash
python abba/main.py --config-file production.json --quiet
```

## Architecture Overview

```
STEPBible Files → SQLite Database → Embeddings → Vector Database
                           ↓                              ↓
                   Direct SQL Queries            Semantic Search
                           ↓                              ↓
                       Unified API ← ← ← ← ← ← ← ← ← ← ← ↓
```

## Contributing

When adding new features:
1. Update relevant documentation
2. Ensure >95% test coverage
3. Run all quality checks (black, pylint, mypy)
4. Update CLAUDE.md if adding development notes

## Support

For issues or questions:
- Check existing documentation
- Review [VALIDATION_SYSTEM.md](VALIDATION_SYSTEM.md) for data integrity issues
- See [CONFIGURATION.md](CONFIGURATION.md) for setup problems