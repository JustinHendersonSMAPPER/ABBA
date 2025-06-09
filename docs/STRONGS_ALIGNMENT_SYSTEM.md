# Strong's Concordance Alignment System

## Overview

This comprehensive system loads the full Strong's concordance into the alignment models and implements manual alignments for high-frequency biblical words. The system provides:

1. **Full Strong's Concordance Integration** - Complete Hebrew and Greek lexicon data
2. **Manual Alignment Mappings** - Curated mappings for high-frequency words
3. **Enhanced Training System** - Iterative training using concordance data
4. **Progressive Refinement Pipeline** - Multi-stage alignment improvement

## System Components

### 1. Strong's Concordance Loader
**Script**: `scripts/load_full_strongs_concordance.py`

Loads and processes the complete Strong's Hebrew and Greek concordance data:
- Extracts translation pairs from glosses, definitions, and KJV usage
- Creates enhanced alignment models with Strong's metadata
- Provides manual mappings for the most common biblical words
- Generates comprehensive statistics

### 2. Manual Alignment Mappings
**Files**: 
- `data/manual_alignments/high_frequency_hebrew.json`
- `data/manual_alignments/high_frequency_greek.json`

High-quality manual alignments for the most frequent biblical words:
- Hebrew: 25 most common words (יהוה, אלהים, אמר, etc.)
- Greek: 25 most common words (θεός, Ἰησοῦς, Χριστός, etc.)
- Includes confidence scores and usage notes

### 3. Enhanced Training System
**Script**: `scripts/train_enhanced_alignment.py`

Implements iterative training with Strong's concordance:
- Initializes models with lexicon data
- Uses Strong's numbers for alignment anchoring
- Applies morphological constraints
- Tracks training progress and convergence

### 4. Iterative Pipeline
**Script**: `scripts/iterative_alignment_pipeline.py`

Complete pipeline for progressive alignment improvement:
- 8-stage pipeline with validation
- Automatic refinement based on evaluation
- Comprehensive reporting and metrics

## Usage

### Step 1: Load Strong's Concordance
```bash
python scripts/load_full_strongs_concordance.py
```

This creates:
- `models/alignment/strongs_enhanced_alignment.json` - Full concordance model
- `models/alignment/sample_strongs_enhanced_alignment.json` - Sample for testing

### Step 2: Run Enhanced Training
```bash
python scripts/train_enhanced_alignment.py
```

This creates:
- `models/alignment/hebrew_enhanced_alignment.json`
- `models/alignment/greek_enhanced_alignment.json`
- Training reports for each language

### Step 3: Run Complete Pipeline
```bash
python scripts/iterative_alignment_pipeline.py
```

This runs all stages:
1. Load Strong's concordance
2. Load manual alignments
3. Initial training
4. Evaluation
5. Refinement
6. Phrasal extraction (placeholder)
7. Final training
8. Validation

Output:
- `models/alignment/hebrew_final_alignment.json`
- `models/alignment/greek_final_alignment.json`
- `models/alignment/pipeline_report.txt`

## Manual Alignment Format

Each manual alignment entry contains:
```json
{
  "strongs": "H3068",
  "hebrew": "יהוה",
  "transliteration": "YHWH",
  "primary_translations": ["LORD", "Yahweh"],
  "all_translations": ["LORD", "Yahweh", "Jehovah", "God"],
  "confidence": 0.95,
  "frequency": "very_high",
  "notes": "Tetragrammaton - the divine name"
}
```

## Model Output Format

The enhanced alignment models include:
```json
{
  "trans_probs": {
    "H3068": {
      "lord": 0.8,
      "yahweh": 0.8,
      "jehovah": 0.2,
      "god": 0.2
    }
  },
  "source_vocab": ["H1", "H2", ...],
  "target_vocab": ["father", "god", ...],
  "strongs_metadata": {
    "hebrew": {
      "H1": {
        "original": "אב",
        "lemma": "אָב",
        "translit": "ʼâb",
        "morph": "n-m",
        "gloss": "father",
        "definition": "father of an individual..."
      }
    }
  },
  "manual_mappings": {...},
  "version": "3.0-final",
  "training_history": [...]
}
```

## Key Features

### 1. Comprehensive Coverage
- Full Strong's concordance (8,674 Hebrew + 5,624 Greek entries)
- Manual mappings for highest-frequency words
- Multiple translation variants per word

### 2. Intelligent Initialization
- Lexicon-based probability initialization
- Manual mappings get highest confidence (0.8)
- Morphological constraints applied

### 3. Progressive Refinement
- Initial training with concordance data
- Evaluation against manual mappings
- Refinement to boost manual alignment coverage
- Final validation and metrics

### 4. Quality Metrics
- Coverage ratio of manual mappings
- High-confidence alignment ratio
- Translation pair statistics
- Convergence tracking

## Extending the System

### Adding More Manual Alignments
1. Edit the JSON files in `data/manual_alignments/`
2. Follow the existing format
3. Re-run the pipeline

### Customizing Training Parameters
Edit in `train_enhanced_alignment.py`:
```python
self.iterations = 5  # Number of training iterations
self.confidence_threshold = 0.3  # Minimum alignment confidence
self.min_frequency = 2  # Minimum word frequency
```

### Adding New Languages
1. Add lexicon data to `data/sources/lexicons/`
2. Create manual alignments in `data/manual_alignments/`
3. Update the scripts to include the new language

## Performance Considerations

- Initial concordance loading takes ~30 seconds
- Full training takes 5-10 minutes per language
- Models are large (500MB+) due to comprehensive coverage
- Sample models provided for testing

## Troubleshooting

### Missing Lexicon Files
Ensure `strongs_hebrew.json` and `strongs_greek.json` exist in `data/sources/lexicons/`

### Memory Issues
Use the sample models for testing:
- `sample_strongs_enhanced_alignment.json`
- Reduce vocabulary size in training

### Convergence Issues
- Increase iteration count
- Adjust learning parameters
- Check training data quality