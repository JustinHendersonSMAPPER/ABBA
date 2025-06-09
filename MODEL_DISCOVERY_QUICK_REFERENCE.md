# Model Discovery Quick Reference

## Finding and Using Models in ABBA-Align

### 1. List All Available Models

```bash
# Show all models with details
abba-align models --list

# Output in JSON format
abba-align models --list --json
```

### 2. Find Model for Specific Languages

```bash
# Find Hebrew to English model
abba-align models --find hebrew english

# Find Greek to Spanish model
abba-align models --find greek spanish
```

### 3. Get Model Information

```bash
# Show detailed info about a specific model
abba-align models --info models/biblical_alignment/hebrew_english_biblical.json

# Get info in JSON format
abba-align models --info models/biblical_alignment/greek_english_biblical.json --json
```

### 4. Validate Model File

```bash
# Check if model file is valid
abba-align models --validate models/biblical_alignment/hebrew_english_biblical.json
```

## Default Model Locations

Models are stored in: `models/biblical_alignment/`

Standard naming: `{source_language}_{target_language}_biblical.json`

## Common Models

| Source | Target | Model File |
|--------|--------|------------|
| Hebrew | English | `hebrew_english_biblical.json` |
| Greek | English | `greek_english_biblical.json` |
| Aramaic | English | `aramaic_english_biblical.json` |

## Auto-Detection

ABBA-Align can auto-detect models based on:
1. Input file content (detects Hebrew/Greek text)
2. Filename patterns (e.g., "hebrew_bible.json")
3. Source language metadata in JSON

## Model Quality Indicators

When listing models, look for:
- **Features**: More enabled features = better quality
- **Mappings**: Higher translation mappings = better coverage
- **Est. Coverage**: Estimated percentage of text that can be aligned

### Example Output

```
SOURCE: HEBREW
----------------------------------------
  → english
    Model: hebrew_english_biblical
    Features: morphology, phrases, strongs, semantics, discourse
    Mappings: 2,848
    Est. Coverage: 85%

SOURCE: GREEK
----------------------------------------
  → english
    Model: greek_english_biblical
    Features: morphology, phrases, strongs
    Mappings: 1,523
    Est. Coverage: 80%
```

## Training New Models

If no model exists for your language pair:

```bash
abba-align train --source hebrew --target spanish --corpus-dir data/corpora --features all
```

## Programmatic Model Discovery

```python
from abba_align.model_info import ModelDiscovery

discovery = ModelDiscovery()

# Find best model
model = discovery.find_model('hebrew', 'english')
if model:
    print(f"Use: {model.path}")
    print(f"Coverage: {model.get_coverage_estimate()}%")

# List all models
for model_info in discovery.list_all_models():
    print(f"{model_info['name']}: {model_info['estimated_coverage']}% coverage")
```

## Tips

1. **Check coverage** before using a model: `abba-align coverage --translation your_bible.json`
2. **Use consistent models** across your project for reproducibility
3. **Document which model version** you used in your research
4. **Validate models** after copying between systems
5. **Train specialized models** for specific domains (e.g., poetry vs. prose)