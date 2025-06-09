# Automatic Model Selection in ABBA-Align

## Overview

ABBA-Align now features intelligent automatic model selection, eliminating the need to manually specify models in most cases.

## How It Works

### 1. Training Creates Models Automatically

When you train a model:
```bash
python -m abba_align train --source hebrew --target english --corpus-dir data/sources --features all
```

The system automatically:
- Creates the model file: `models/biblical_alignment/hebrew_english_biblical.json`
- Includes Strong's mappings and feature configurations
- Generates a training report with statistics

### 2. Commands Auto-Select Best Model

#### Annotation (No Model Needed)
```bash
# System auto-detects Hebrew content and selects hebrew_english_biblical.json
abba-align annotate --input hebrew_text.json --output annotated.json

# System auto-detects Greek content and selects greek_english_biblical.json  
abba-align annotate --input greek_text.json --output annotated.json
```

#### Coverage Analysis
```bash
# Automatically uses best Hebrew model
abba-align coverage --translation kjv.json --source-language hebrew

# Automatically uses best Greek model
abba-align coverage --translation kjv_nt.json --source-language greek
```

### 3. Model Selection Logic

The system selects models based on:

1. **Content Detection**: Analyzes text for Hebrew/Greek characters
2. **Filename Hints**: Looks for "hebrew", "greek", "heb", "grc" in filenames
3. **Quality Metrics**: Chooses model with:
   - Most features enabled
   - Highest translation mappings
   - Best estimated coverage

### 4. Override When Needed

You can still specify a model explicitly:
```bash
abba-align annotate --input text.json --output annotated.json --model custom_model.json
```

## Model Quality Indicators

When models are auto-selected, the system considers:

| Metric | Weight | Description |
|--------|--------|-------------|
| Features | High | More features = better alignment |
| Mappings | High | More Strong's mappings = better coverage |
| Coverage | Medium | Estimated % of text that can be aligned |
| Recency | Low | Newer models preferred |

## Examples

### Auto-Detection from Content

```json
// hebrew_sample.json
{
  "text": "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
}
```

```bash
# Automatically detects Hebrew and uses hebrew_english_biblical.json
abba-align annotate --input hebrew_sample.json --output result.json
```

### Model Discovery

```bash
# See all available models
abba-align models --list

# Find best model for language pair
abba-align models --find hebrew spanish

# Get model details
abba-align models --info models/biblical_alignment/hebrew_english_biblical.json
```

## Benefits

1. **Simplified Usage**: No need to remember model paths
2. **Intelligent Selection**: Always uses the best available model
3. **Fallback Support**: Clear error messages if no suitable model exists
4. **Quality Assurance**: Automatically picks models with best coverage

## Troubleshooting

### "No model found"
```bash
# Train the needed model
abba-align train --source hebrew --target english --corpus-dir data/sources --features all
```

### "Could not auto-detect language"
```bash
# Specify source language explicitly
abba-align coverage --translation bible.json --source-language hebrew
```

### Wrong model selected
```bash
# Override with specific model
abba-align annotate --input text.json --output result.json --model path/to/specific/model.json
```

## Technical Details

### Model Files Include:
- Trained alignment parameters
- Strong's concordance mappings (2,000-3,000 entries)
- Feature configurations
- Version and training metadata

### Auto-Detection Heuristics:
- Hebrew: Looks for characters like א, ב, ג, בְּרֵאשִׁית
- Greek: Looks for characters like Α, Ω, Ἰησοῦς, Χριστός
- Aramaic: Specific character patterns

### Selection Algorithm:
1. Detect source language from content
2. Find all models for that source language
3. Rank by: features count × mappings × coverage estimate
4. Select highest ranking model

## Summary

With automatic model selection, using ABBA-Align is now as simple as:

```bash
# Train once
abba-align train --source hebrew --target english --corpus-dir data/sources --features all

# Use anywhere - model selected automatically
abba-align annotate --input any_hebrew_text.json --output annotated.json
abba-align coverage --translation kjv.json --source-language hebrew
```

The system intelligently handles model selection, making biblical text alignment accessible without deep technical knowledge of the underlying models.