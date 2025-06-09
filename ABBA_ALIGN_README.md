# ABBA-Align: Biblical Text Alignment Toolkit

A specialized command-line tool for training and applying advanced word alignment models for biblical texts, with support for morphological analysis, phrase detection, and theological concept mapping.

## Features

### Core Capabilities
- **Hybrid Alignment Models**: Combines statistical (IBM Model 1) and neural (embeddings) approaches
- **Morphological Analysis**: Deep analysis of Hebrew prefixes/suffixes and Greek case endings
- **Phrase Detection**: Identifies and aligns multi-word expressions and biblical idioms
- **Parallel Passage Analysis**: Leverages synoptic gospels and OT/NT quotations for improved accuracy
- **Semantic Role Labeling**: Tracks who-did-what-to-whom across languages
- **Discourse Analysis**: Handles narrative markers and discourse structure
- **Strong's Concordance Integration**: Automatic XML to JSON conversion during setup for enhanced alignment accuracy

### Supported Languages
- **Source**: Hebrew (BHS), Greek (NA28), Aramaic
- **Target**: Any language with proper training data

## Installation

### As a Standalone Tool
```bash
pip install abba-align

# Or with ML features
pip install abba-align[ml]
```

### From Source
```bash
git clone https://github.com/your-org/abba
cd abba

# Download source data and convert Strong's XML to JSON
python scripts/download_sources.py

# Install ABBA-Align
pip install -e .[dev,ml] -f setup_abba_align.py
```

## Quick Start

### 1. Train an Alignment Model
```bash
# Train Hebrew-English model with all features
abba-align train --source hebrew --target english \
    --corpus-dir data/corpora \
    --features all \
    --parallel-passages \
    --output-dir models/biblical

# Train Greek-English with specific features
abba-align train --source greek --target english \
    --corpus-dir data/corpora \
    --features morphology phrases discourse
```

### 2. Annotate a Translation
```bash
# Annotate a Bible translation with alignment confidence
abba-align annotate \
    --input translations/eng_niv.json \
    --output annotated/eng_niv_aligned.json \
    --include-morphology \
    --include-phrases

# Batch process multiple translations
abba-align annotate \
    --input translations/ \
    --output annotated/ \
    --confidence-threshold 0.4
```

### 3. Extract Biblical Phrases
```bash
# Extract Hebrew phrases and idioms
abba-align phrases \
    --language hebrew \
    --corpus data/corpora/heb_bhs.json \
    --min-frequency 5 \
    --output hebrew_phrases.json

# Extract English theological terms
abba-align phrases \
    --language english \
    --corpus translations/eng_kjv.json \
    --min-frequency 10 \
    --output english_biblical_phrases.json
```

### 4. Analyze Parallel Passages
```bash
# Analyze synoptic gospel parallels
abba-align parallels \
    --type synoptic \
    --output synoptic_analysis.json

# Find OT quotes in NT
abba-align parallels \
    --type ot-quotes \
    --output ot_in_nt.json
```

### 5. Morphological Analysis
```bash
# Analyze Hebrew morphology with decomposition
abba-align morphology \
    --language hebrew \
    --input texts/genesis.json \
    --decompose \
    --output genesis_morphology.json

# Analyze Greek case patterns
abba-align morphology \
    --language greek \
    --input texts/matthew.json \
    --output matthew_morphology.json
```

## Advanced Usage

### Custom Training Pipeline
```python
from abba_align import AlignmentTrainer, MorphologicalAnalyzer, BiblicalPhraseDetector

# Initialize components
trainer = AlignmentTrainer(
    source_lang='hebrew',
    target_lang='english',
    enable_morphology=True,
    enable_phrases=True,
    enable_semantics=True
)

# Add morphological constraints
morph_analyzer = MorphologicalAnalyzer('hebrew')
trainer.add_morphological_constraints(morph_analyzer)

# Add phrase detection
phrase_detector = BiblicalPhraseDetector('hebrew', min_frequency=5)
trainer.add_phrase_detector(phrase_detector)

# Train model
model = trainer.train()
```

### Ensemble Alignment
```python
from abba_align import HybridAlignmentModel

# Create hybrid model
hybrid = HybridAlignmentModel()

# Train both components
hybrid.train_parallel(parallel_corpus)
hybrid.train_monolingual(hebrew_sentences, english_sentences, anchor_pairs)

# Apply ensemble alignment
alignments = hybrid.align_with_ensemble(
    source_words,
    target_text,
    threshold=0.3
)
```

## Output Format

### Annotated Translation Format
```json
{
  "verse_id": "Gen.1.1",
  "translations": {
    "ENG_NIV": "In the beginning God created the heavens and the earth."
  },
  "alignments": [
    {
      "source_phrase": "בְּרֵאשִׁית",
      "target_phrase": "In the beginning",
      "confidence": 0.92,
      "type": "phrase",
      "morphology": {
        "prefixes": [{"ב": "in"}],
        "root": "ראשית",
        "meaning": "beginning"
      }
    },
    {
      "source_word": "אֱלֹהִים",
      "target_word": "God",
      "confidence": 0.98,
      "type": "word",
      "semantic_role": "agent"
    }
  ],
  "phrases_detected": [
    {
      "phrase": "In the beginning",
      "type": "temporal_marker",
      "theological_significance": "creation_narrative"
    }
  ]
}
```

## Configuration

### Model Configuration
```yaml
# config/alignment_config.yaml
alignment:
  model_type: hybrid
  ibm_weight: 0.6
  embedding_weight: 0.4
  
morphology:
  enable_decomposition: true
  use_lexicon: true
  
phrases:
  min_frequency: 3
  max_length: 5
  
training:
  iterations: 10
  batch_size: 1000
  learning_rate: 0.001
```

## Evaluation Metrics

### Alignment Quality Metrics
- **Precision**: How many predicted alignments are correct
- **Recall**: How many gold alignments were found  
- **F1 Score**: Harmonic mean of precision and recall
- **AER**: Alignment Error Rate
- **Phrase Accuracy**: Correct phrase boundary detection

### Run Evaluation
```bash
# Evaluate model performance
abba-align evaluate \
    --model models/hebrew_english.json \
    --test-set test/gold_alignments.json \
    --metrics all

# Output:
# Precision: 0.823
# Recall: 0.756
# F1 Score: 0.788
# AER: 0.212
# Phrase Accuracy: 0.891
```

## Best Practices

### For Translators
1. Use high confidence thresholds (>0.7) for critical decisions
2. Review phrase alignments manually for theological terms
3. Consider morphological decomposition for accurate word studies

### For Researchers  
1. Train separate models for different text genres (narrative vs. poetry)
2. Use parallel passages to improve rare word alignments
3. Combine multiple models for ensemble predictions

### For Developers
1. Implement caching for large-scale processing
2. Use batch processing for multiple books
3. Monitor memory usage with large corpora

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional language support (Coptic, Syriac, Latin)
- Improved phrase detection algorithms
- Integration with existing Bible software
- Performance optimizations
- Additional evaluation metrics

## Citation

If you use ABBA-Align in your research, please cite:
```bibtex
@software{abba-align,
  title = {ABBA-Align: Biblical Text Alignment Toolkit},
  author = {ABBA Project Contributors},
  year = {2024},
  url = {https://github.com/your-org/abba}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Strong's Concordance for lexical data
- CCAT/Tyndale House for morphological databases
- Various open-source Bible digitization projects