# Scripts Organization

## Production Scripts (in `scripts/`)

### Data Preparation
- **download_sources.py** - Downloads biblical source texts and lexicons
- **create_corpus_files.py** - Creates training corpus files
- **create_manual_alignments.py** - Creates manual alignment files for high-frequency words
- **load_full_strongs_concordance.py** - Loads complete Strong's concordance into alignment model

### Training Scripts
- **train_all_models.py** - Trains Hebrew and Greek alignment models
- **train_enhanced_alignment.py** - Enhanced training with Strong's integration
- **train_modern_alignment.py** - Modern alignment approach
- **train_word_alignment.py** - Word-level alignment training
- **train_hybrid_alignment.py** - Hybrid alignment approach
- **train_full_corpus.py** - Training on full corpus
- **iterative_alignment_pipeline.py** - Multi-stage iterative training pipeline

### Analysis Scripts
- **analyze_all_translations_coverage.py** - Analyzes alignment coverage across translations
- **analyze_enrichment_coverage.py** - Analyzes enrichment data coverage
- **extract_from_sqlite.py** - Extracts data from SQLite databases

### Shell Scripts
- **run_full_coverage_analysis.sh** - Complete workflow for coverage analysis
- **run_enhanced_coverage_analysis.sh** - Coverage analysis with enhanced models

## Development/Test Scripts (moved to `claude_scripts/`)

All scripts used for testing, debugging, demonstrations, and one-time fixes have been moved to `claude_scripts/` including:
- debug_*.py scripts
- demo_*.py scripts
- demonstrate_*.py scripts
- test_*.py scripts (for Claude testing, not pytest)
- validate_*.py scripts
- One-time fix scripts
- Development progress summaries

## Guidelines

1. **Production scripts** should be:
   - Reusable tools for data processing, training, or analysis
   - Well-documented with clear purposes
   - Part of the regular workflow

2. **Claude scripts** should be:
   - One-time test or demonstration scripts
   - Debug utilities for development
   - Temporary fixes or experiments
   - Scripts created to validate Claude's work