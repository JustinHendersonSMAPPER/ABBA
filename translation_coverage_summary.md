# Translation Coverage Analysis Report

## Model Summary

### Hebrew-English Model
- Strong's mappings: 8,673
- Features: strongs, manual_alignments, morphology, phrases, syntax, semantics, discourse

### Greek-English Model
- Strong's mappings: 5,472
- Features: strongs, manual_alignments, morphology, phrases, syntax, semantics, discourse

## Translation Summary

Total translation files found: 1020

### Sample Analysis

| Translation | Language | Books | Verses | Words |
|-------------|----------|-------|--------|-------|
| King James (Authorized) Versio... | en | 66 | 31,102 | 792,612 |
| World English Bible with Deute... | en | 66 | 31,103 | 748,919 |
| American Standard Version (190... | en | 66 | 31,102 | 783,611 |
| Bible in Basic English... | en | 66 | 31,102 | 839,363 |

## Current Limitations

1. **Limited Strong's Mappings**: Models currently have limited word mappings
2. **Training Data**: Models need more parallel text data for comprehensive coverage
3. **Translation Format**: Working with standardized JSON format from eBible.org

## Recommendations

1. **Expand Training Corpus**: Add more parallel texts with Strong's numbers
2. **Import Full Strong's**: Load complete Strong's concordance mappings
3. **Manual Alignments**: Add high-frequency word alignments manually
4. **Iterative Training**: Train models on aligned output to improve coverage

## Next Steps

1. Load full Strong's concordance into models
2. Create manual alignment files for common words
3. Implement progressive alignment training
4. Generate per-translation coverage reports
