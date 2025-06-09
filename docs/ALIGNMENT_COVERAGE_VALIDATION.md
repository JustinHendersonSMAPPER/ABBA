# Alignment Coverage Validation Guide

This guide explains how to validate that your alignment models provide sufficient coverage of a translation's vocabulary.

## Overview

Coverage validation measures what percentage of a translation can be traced back to the original language. This is critical for:
- Bible study tools that show original language connections
- Translation checking and quality assurance
- Academic research on translation techniques
- Identifying gaps in alignment models

## Key Metrics

### 1. Token Coverage
**Definition**: Percentage of word occurrences (tokens) that can be aligned to source language.

**Example**: If "God" appears 100 times and 95 can be aligned to אֱלֹהִים, that's 95% token coverage for that word.

**Targets**:
- Excellent: >90%
- Good: 80-90%
- Acceptable: 70-80%
- Needs improvement: <70%

### 2. Type Coverage  
**Definition**: Percentage of unique words (types) that have at least one alignment.

**Example**: If a text has 1,000 unique words and 800 have alignments, that's 80% type coverage.

**Targets**:
- Excellent: >80%
- Good: 70-80%
- Acceptable: 60-70%
- Needs improvement: <60%

### 3. Coverage by Frequency Band

Different frequency bands should have different coverage expectations:

| Frequency Band | Occurrences | Expected Coverage |
|----------------|-------------|-------------------|
| High frequency | 100+ | >95% |
| Medium frequency | 10-99 | >85% |
| Low frequency | 2-9 | >70% |
| Hapax legomena | 1 | >50% |

## Using ABBA-Align for Coverage Validation

### Basic Usage

```bash
# Analyze coverage of KJV translation against Hebrew
abba-align coverage \
    --translation translations/kjv.json \
    --source-language hebrew \
    --report kjv_coverage_report.txt

# Analyze coverage with specific model
abba-align coverage \
    --translation translations/niv.json \
    --source-language greek \
    --model models/greek_english_biblical.json \
    --min-coverage 85
```

### Programmatic Usage

```python
from abba_align.coverage_analyzer import AlignmentCoverageAnalyzer

# Initialize analyzer
analyzer = AlignmentCoverageAnalyzer(source_lang='hebrew')

# Run analysis
stats = analyzer.analyze_translation_coverage(
    translation_path='translations/esv.json',
    alignment_model_path='models/hebrew_english.json'
)

# Check coverage
token_coverage = stats['summary']['token_coverage']
type_coverage = stats['summary']['type_coverage']

print(f"Token Coverage: {token_coverage:.1f}%")
print(f"Type Coverage: {type_coverage:.1f}%")

# Generate detailed report
report = analyzer.generate_coverage_report(stats, 'coverage_report.txt')
```

### Using the Standalone Script

```bash
# Basic validation
python scripts/validate_alignment_coverage.py \
    --model models/hebrew_english_biblical.json \
    --translation translations/kjv.json \
    --report kjv_coverage.txt \
    --verbose

# With minimum coverage requirement
python scripts/validate_alignment_coverage.py \
    --model models/greek_english_biblical.json \
    --translation translations/niv_nt.json \
    --min-coverage 85 \
    --report niv_nt_coverage.txt
```

## Understanding Coverage Reports

### Report Sections

1. **Overall Coverage**
   - Total tokens and types
   - Aligned tokens and types
   - Coverage percentages

2. **Coverage by Word Frequency**
   - Shows how well different frequency bands are covered
   - Helps identify if model handles common vs. rare words

3. **Coverage by Part of Speech**
   - Breaks down coverage by grammatical category
   - Useful for identifying systematic gaps

4. **Verse-Level Statistics**
   - Distribution of coverage across verses
   - Identifies problematic passages

5. **Uncovered Words**
   - List of most common words without alignments
   - Prioritizes improvements to alignment model

### Sample Report

```
======================================================================
ALIGNMENT COVERAGE ANALYSIS REPORT
======================================================================

OVERALL COVERAGE
----------------------------------------
Total word occurrences (tokens): 789,633
Unique words (types): 12,403
Aligned occurrences: 712,456 (90.2%)
Aligned unique words: 9,823 (79.2%)

COVERAGE BY WORD FREQUENCY
----------------------------------------
High frequency (100+):
  Types: 245/267 (91.8%)
  Tokens: 456,789/478,234 (95.5%)
Medium frequency (10-99):
  Types: 1,234/1,456 (84.7%)
  Tokens: 89,123/98,765 (90.2%)
Low frequency (2-9):
  Types: 3,456/4,567 (75.6%)
  Tokens: 23,456/28,901 (81.1%)
Hapax legomena (1):
  Types: 5,088/6,113 (83.2%)
  Tokens: 5,088/6,113 (83.2%)

VERSE-LEVEL STATISTICS
----------------------------------------
Total verses analyzed: 31,102
Average verse coverage: 89.3%
Standard deviation: 12.4%
Verses with perfect coverage (100%): 8,234
Verses with high coverage (≥90%): 24,567
Verses with medium coverage (70-89%): 5,234
Verses with low coverage (<70%): 1,301

TOP 30 UNCOVERED WORDS
----------------------------------------
 1. unto                 8,234 occurrences
 2. thereof              2,345 occurrences
 3. yea                  1,234 occurrences
[...]
```

## Improving Coverage

### 1. Analyze Uncovered Words

Look at the most common uncovered words to identify patterns:
- Archaic English words (thee, thou, unto)
- Function words not directly translated
- Idiomatic expressions
- Proper nouns

### 2. Enhance Training Data

```bash
# Add more parallel texts
abba-align train \
    --source hebrew \
    --target english \
    --corpus-dir expanded_corpus \
    --parallel-passages \
    --features all

# Use multiple Bible versions
abba-align train \
    --source greek \
    --target english \
    --corpus-dir multi_version_corpus \
    --features morphology phrases discourse
```

### 3. Leverage Strong's Concordance

Ensure Strong's numbers are properly mapped:
```python
# Check Strong's coverage
strongs_stats = stats['strongs_mapping_stats']
print(f"Strong's entries with mappings: {strongs_stats['total_mappings']}")
print(f"Average translations per entry: {strongs_stats['average_translations_per_word']:.1f}")
```

### 4. Add Manual Alignments

For critical high-frequency words without alignments:
```json
{
  "manual_alignments": {
    "unto": ["H413", "H5921"],  // אֶל, עַל
    "thereof": ["H1931", "H0"],  // הוּא, suffix
    "yea": ["H637", "H1571"]     // אַף, גַּם
  }
}
```

## Best Practices

### 1. Set Realistic Targets

- Modern translations: 85-95% token coverage
- Older translations (KJV): 80-90% token coverage  
- Paraphrases (NLT, Message): 70-85% token coverage

### 2. Regular Validation

Run coverage validation:
- After training new models
- When adding new translations
- As part of CI/CD pipeline

### 3. Focus on High-Impact Improvements

Prioritize based on frequency:
- Fix high-frequency uncovered words first
- Ensure theological terms are well-covered
- Accept lower coverage for rare/archaic terms

### 4. Document Known Limitations

Some words legitimately can't be aligned:
- English additions for clarity
- Idioms with no direct equivalent
- Cultural adaptations

## Troubleshooting

### Low Coverage Issues

1. **Missing Strong's numbers**: Ensure source texts have Strong's annotations
2. **Incorrect language**: Verify source language matches (Hebrew for OT, Greek for NT)
3. **Preprocessing issues**: Check text normalization and tokenization
4. **Model not loaded**: Ensure alignment model exists and loads correctly

### Performance Issues

For large translations:
```python
# Process in batches
for book in books:
    stats = analyzer.analyze_translation_coverage(
        translation_path=f'translations/{book}.json'
    )
```

### Validation Errors

Common errors and solutions:
- `FileNotFoundError`: Check file paths
- `KeyError`: Ensure JSON format matches expected schema
- `MemoryError`: Process smaller chunks or increase memory

## Advanced Topics

### Custom Coverage Metrics

Extend the analyzer for specific needs:
```python
class CustomCoverageAnalyzer(AlignmentCoverageAnalyzer):
    def calculate_theological_coverage(self, verse_data):
        """Calculate coverage of theological terms."""
        theological_terms = {'God', 'Lord', 'salvation', 'grace', ...}
        # Custom implementation
```

### Integration with Bible Software

Export coverage data for use in other tools:
```python
# Export to CSV
coverage_df = pd.DataFrame(stats['verse_coverage'])
coverage_df.to_csv('verse_coverage.csv', index=False)

# Export to SQLite
conn = sqlite3.connect('coverage.db')
coverage_df.to_sql('verse_coverage', conn, if_exists='replace')
```

### Comparative Analysis

Compare coverage across translations:
```bash
for translation in kjv niv esv nlt; do
    abba-align coverage \
        --translation translations/${translation}.json \
        --report reports/${translation}_coverage.txt
done
```

## Conclusion

Regular coverage validation ensures your alignment models provide value to end users. By monitoring and improving coverage metrics, you can build robust biblical text alignment systems that preserve the connection between translations and original languages.