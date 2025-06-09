# Coverage Analysis Guide

This guide explains how to analyze alignment coverage across multiple Bible translations and generate comprehensive reports.

## Overview

The coverage analysis system:
- Automatically trains Hebrew and Greek models with all features
- Analyzes each translation to determine coverage percentage
- Generates detailed markdown reports with tables and statistics
- Identifies common uncovered words for improvement

## Quick Start

### One-Command Analysis

```bash
# Run complete analysis workflow
./scripts/run_full_coverage_analysis.sh
```

This script:
1. Downloads source data (if needed)
2. Trains Hebrew and Greek models
3. Analyzes all translations
4. Generates markdown report

### Step-by-Step

```bash
# 1. Train all models (Hebrew and Greek)
python scripts/train_all_models.py

# 2. Analyze all translations
python scripts/analyze_all_translations_coverage.py

# 3. View the report
cat translation_coverage_report.md
```

## Training Configuration

Models are automatically trained with:
- **All features enabled**: morphology, phrases, syntax, semantics, discourse, Strong's
- **Parallel passages**: For improved accuracy
- **Full corpus**: Using all available source texts

### Default Training Command
```bash
abba-align train --source hebrew --target english --corpus-dir data/sources --features all --parallel-passages
abba-align train --source greek --target english --corpus-dir data/sources --features all --parallel-passages
```

## Coverage Report Features

### 1. Summary Statistics
- Total translations analyzed
- Average token coverage (% of word occurrences)
- Average type coverage (% of unique words)

### 2. Main Coverage Table
Shows for each translation:
- Testament (OT/NT) - automatically detected
- Source language (Hebrew/Greek)
- Token coverage with visual indicators
- Type coverage
- Total and covered word counts
- High-frequency word coverage

### 3. Visual Indicators
- 🟢 Excellent: ≥90% coverage
- 🟡 Good: 80-89% coverage
- 🟠 Fair: 70-79% coverage
- 🔴 Poor: <70% coverage

### 4. Testament Comparison
Compares average coverage between Old and New Testament translations

### 5. Common Uncovered Words
Lists the most frequent words without alignments across all translations

### 6. Model Information
Details about the alignment models used:
- Enabled features
- Number of Strong's mappings
- Estimated coverage capability

## Sample Report Output

```markdown
# Translation Coverage Analysis Report

Generated: 2025-06-08 12:00:00

## Summary Statistics

- **Translations Analyzed**: 10
- **Average Token Coverage**: 87.5%
- **Average Type Coverage**: 76.3%

## Coverage by Translation

| Translation | Testament | Source | Token Coverage | Type Coverage | Total Words | Covered Words |
|-------------|-----------|--------|----------------|---------------|-------------|---------------|
| ESV_2016    | OT        | Hebrew | 🟢 92.3%      | 84.5%         | 622,084     | 574,183      |
| NIV_2011    | OT        | Hebrew | 🟢 91.7%      | 83.2%         | 615,923     | 564,801      |
| KJV_1769    | OT        | Hebrew | 🟡 85.4%      | 72.8%         | 647,229     | 552,734      |
...
```

## Understanding Coverage Metrics

### Token Coverage
- **Definition**: Percentage of word occurrences that can be aligned
- **Example**: If "God" appears 100 times and 95 can be aligned, that's 95% coverage
- **Target**: >90% for modern translations, >85% for older translations

### Type Coverage
- **Definition**: Percentage of unique words that have alignments
- **Example**: If there are 1,000 unique words and 800 can be aligned, that's 80% coverage
- **Target**: >80% for comprehensive alignment

### High-Frequency Coverage
- **Definition**: Coverage of words appearing 100+ times
- **Importance**: These words make up a large portion of the text
- **Target**: >95% for good user experience

## Improving Coverage

Based on the analysis results:

### 1. For Low Coverage Translations
- Add manual alignments for high-frequency uncovered words
- Use translation-specific vocabulary mappings
- Consider training specialized models

### 2. Common Uncovered Words
Often include:
- Archaic terms (thee, thou, unto)
- Translation-specific expressions
- Function words not in source
- Proper noun variations

### 3. Enhancement Strategies
```bash
# Add more training data
abba-align train --source hebrew --target english \
    --corpus-dir expanded_corpus \
    --features all

# Create manual alignments for specific words
{
  "manual_alignments": {
    "unto": ["H413", "H5921"],
    "yea": ["H637", "H1571"]
  }
}
```

## Advanced Usage

### Analyze Specific Directory
```bash
python scripts/analyze_all_translations_coverage.py \
    --translations-dir custom_translations/ \
    --output custom_report.md
```

### Train and Analyze
```bash
python scripts/analyze_all_translations_coverage.py \
    --train-first \
    --output fresh_analysis.md
```

### Filter by Testament
```python
# In your own script
analyzer = TranslationCoverageAnalyzer()
results = analyzer.analyze_all_translations()

# Filter OT only
ot_results = [r for r in results if r['testament'] == 'OT']
```

## Interpreting Results

### Good Coverage (>85%)
- Most biblical text can be traced to original
- Suitable for study tools and research
- Minor gaps in archaic/rare terms

### Fair Coverage (70-85%)
- Core vocabulary well-covered
- Some gaps in less common words
- May need enhancement for scholarly use

### Poor Coverage (<70%)
- Significant alignment gaps
- Needs additional training data
- Consider manual alignment additions

## Troubleshooting

### "No models found"
```bash
# Train models first
python scripts/train_all_models.py
```

### "No translations found"
- Ensure translation JSON files are in `translations/` directory
- Check file format matches expected schema

### Low coverage scores
1. Check if using correct source language
2. Verify Strong's concordance loaded properly
3. Consider translation-specific challenges

## Best Practices

1. **Regular Analysis**: Run after training new models
2. **Track Progress**: Save reports to compare improvements
3. **Focus on High-Impact**: Prioritize high-frequency uncovered words
4. **Document Changes**: Note manual alignments added
5. **Version Control**: Keep reports with model versions

## Summary

The coverage analysis system provides comprehensive insights into alignment quality across translations. Use it to:
- Validate model performance
- Identify improvement areas
- Track progress over time
- Ensure quality for end users

Regular analysis ensures your biblical text alignment system maintains high coverage across diverse translations.