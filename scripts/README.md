# ABBA Scripts

This directory contains production scripts for the ABBA project.

## Coverage Analysis

### Main Script: `run_coverage_analysis.py`

This is the unified entry point for running coverage analysis with enhanced models. It:

1. **Checks source data** - Downloads Strong's concordance if needed
2. **Builds enhanced models** - Creates models with full Strong's mappings:
   - Hebrew: 8,673 Strong's entries + 25 manual alignments
   - Greek: 5,472 Strong's entries + 25 manual alignments  
3. **Configures models** - Ensures enhanced models are used (not basic ones)
4. **Runs analysis** - Analyzes all translations and generates report

Usage:
```bash
python scripts/run_coverage_analysis.py
# or
./scripts/run_coverage_analysis.py
```

Output: `translation_coverage_report.md`

### Supporting Scripts

- `load_full_strongs_concordance.py` - Loads complete Strong's concordance
- `create_manual_alignments.py` - Creates high-quality manual alignments
- `create_enhanced_models.py` - Builds enhanced alignment models
- `analyze_all_translations_coverage.py` - Performs the actual coverage analysis

## Other Scripts

- `download_sources.py` - Downloads biblical source data
- `train_*.py` - Various training scripts for alignment models
- `validate_*.py` - Validation and testing scripts
- `demonstrate_*.py` - Demonstration scripts showing features

## Note

Test scripts created during development are in the `claude_scripts/` directory.