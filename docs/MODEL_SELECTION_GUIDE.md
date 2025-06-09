# Model Selection Guide for ABBA-Align

This guide helps you choose the right alignment model for your specific use case.

## Quick Decision Tree

```
What is your source text?
├── Old Testament (Hebrew/Aramaic)
│   └── Use: hebrew_english_biblical.json
├── New Testament (Greek)  
│   └── Use: greek_english_biblical.json
└── Deuterocanonical/Apocrypha
    ├── Hebrew portions → hebrew_english_biblical.json
    └── Greek portions → greek_english_biblical.json
```

## Model Naming Convention

ABBA-Align models follow this naming pattern:
```
{source_language}_{target_language}_biblical.json
```

Examples:
- `hebrew_english_biblical.json` - Hebrew to English alignment
- `greek_english_biblical.json` - Greek to English alignment
- `hebrew_spanish_biblical.json` - Hebrew to Spanish alignment
- `aramaic_english_biblical.json` - Aramaic to English alignment

## Finding Available Models

### 1. List Existing Models

```bash
# List all trained models
ls -la models/biblical_alignment/

# Or use find
find . -name "*_biblical.json" -type f

# Check model details
cat models/biblical_alignment/hebrew_english_report.json
```

### 2. Model Discovery Tool

Create this helper script to discover and analyze available models:

```python
#!/usr/bin/env python3
# save as: scripts/list_models.py

import json
from pathlib import Path
from datetime import datetime

def list_available_models():
    """List all available alignment models with details."""
    model_dir = Path('models/biblical_alignment')
    
    if not model_dir.exists():
        print("No models directory found. Train some models first!")
        return
    
    models = list(model_dir.glob('*_biblical.json'))
    reports = list(model_dir.glob('*_report.json'))
    
    print("=" * 70)
    print("AVAILABLE ALIGNMENT MODELS")
    print("=" * 70)
    print()
    
    for model_path in sorted(models):
        # Parse model name
        name_parts = model_path.stem.split('_')
        if len(name_parts) >= 3:
            source_lang = name_parts[0]
            target_lang = name_parts[1]
            
            print(f"Model: {model_path.name}")
            print(f"  Source Language: {source_lang}")
            print(f"  Target Language: {target_lang}")
            
            # Check for report
            report_path = model_path.parent / f"{source_lang}_{target_lang}_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                    
                print(f"  Features:")
                features = report.get('features_enabled', {})
                enabled = [k for k, v in features.items() if v]
                print(f"    - " + "\n    - ".join(enabled))
                
                if 'strongs_summary' in report:
                    strongs = report['strongs_summary']
                    total_entries = strongs.get('hebrew_entries', 0) + strongs.get('greek_entries', 0)
                    print(f"  Strong's Entries: {total_entries:,}")
                    print(f"  Translation Mappings: {strongs.get('translation_mappings', 0):,}")
            
            # Check file stats
            stat = model_path.stat()
            print(f"  Size: {stat.st_size / 1024:.1f} KB")
            print(f"  Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")
            print()

if __name__ == '__main__':
    list_available_models()
```

### 3. Auto-Detection in ABBA-Align

ABBA-Align can auto-detect the appropriate model:

```bash
# Auto-detection based on input file
abba-align annotate \
    --input translations/hebrew_bible.json \
    --output annotated_output.json
    # Model auto-selected based on source language in input
```

## Model Selection by Use Case

### 1. **Bible Study Applications**

For verse-by-verse study showing original language:
```bash
# For Old Testament
--model models/biblical_alignment/hebrew_english_biblical.json

# For New Testament  
--model models/biblical_alignment/greek_english_biblical.json
```

### 2. **Translation Comparison**

When comparing multiple English translations:
```bash
# Use the same model for consistency
--model models/biblical_alignment/hebrew_english_biblical.json
# Apply to KJV, NIV, ESV, etc.
```

### 3. **Academic Research**

For scholarly work requiring high precision:
```bash
# Train with all features for maximum accuracy
abba-align train \
    --source hebrew \
    --target english \
    --features all \
    --parallel-passages \
    --output-dir models/academic

# Use the resulting model
--model models/academic/hebrew_english_biblical.json
```

### 4. **Multilingual Applications**

For non-English targets:
```bash
# Spanish Old Testament
--model models/biblical_alignment/hebrew_spanish_biblical.json

# German New Testament
--model models/biblical_alignment/greek_german_biblical.json
```

## Model Quality Indicators

### Check Model Report

```bash
# View model statistics
cat models/biblical_alignment/hebrew_english_report.json | jq '.'
```

Look for:
- **strongs_summary.translation_mappings**: Higher is better (2000+ is good)
- **features_enabled**: More features = better alignment quality
- **training_complete**: Should be `true`

### Test Model Coverage

```bash
# Quick coverage test
abba-align coverage \
    --translation sample_translation.json \
    --source-language hebrew \
    --model models/biblical_alignment/hebrew_english_biblical.json
```

Good models achieve:
- Token coverage: >85%
- Type coverage: >75%

## Creating Model Catalogs

### 1. Model Metadata File

Create `models/model_catalog.json`:

```json
{
  "models": [
    {
      "id": "hebrew_english_biblical",
      "name": "Hebrew to English (Standard)",
      "source_language": "hebrew",
      "target_language": "english",
      "description": "Standard Hebrew-English alignment with Strong's",
      "recommended_for": ["bible_study", "translation_checking"],
      "coverage": {
        "token_coverage": 89.5,
        "type_coverage": 78.3
      },
      "features": ["morphology", "phrases", "strongs"],
      "file": "models/biblical_alignment/hebrew_english_biblical.json"
    },
    {
      "id": "greek_english_biblical",
      "name": "Greek to English (Standard)",
      "source_language": "greek", 
      "target_language": "english",
      "description": "Standard Greek-English alignment for NT",
      "recommended_for": ["new_testament_study"],
      "coverage": {
        "token_coverage": 91.2,
        "type_coverage": 82.1
      },
      "features": ["morphology", "phrases", "strongs", "discourse"],
      "file": "models/biblical_alignment/greek_english_biblical.json"
    }
  ]
}
```

### 2. Model Selection Script

```python
#!/usr/bin/env python3
# save as: scripts/select_model.py

import json
from pathlib import Path

def select_model(source_text_type, target_language='english', use_case='general'):
    """Smart model selection based on requirements."""
    
    # Load catalog
    catalog_path = Path('models/model_catalog.json')
    if catalog_path.exists():
        with open(catalog_path) as f:
            catalog = json.load(f)
    else:
        catalog = {'models': []}
    
    # Determine source language
    source_map = {
        'old_testament': 'hebrew',
        'new_testament': 'greek',
        'hebrew_bible': 'hebrew',
        'septuagint': 'greek',
        'tanakh': 'hebrew',
        'gospels': 'greek',
        'paul': 'greek',
        'torah': 'hebrew',
        'prophets': 'hebrew'
    }
    
    source_language = source_map.get(source_text_type.lower(), source_text_type)
    
    # Find matching models
    candidates = []
    for model in catalog['models']:
        if (model['source_language'] == source_language and 
            model['target_language'] == target_language):
            candidates.append(model)
    
    # Filter by use case
    if use_case != 'general' and candidates:
        filtered = [m for m in candidates if use_case in m.get('recommended_for', [])]
        if filtered:
            candidates = filtered
    
    # Sort by coverage
    candidates.sort(key=lambda m: m.get('coverage', {}).get('token_coverage', 0), reverse=True)
    
    if candidates:
        best = candidates[0]
        print(f"Recommended model: {best['name']}")
        print(f"File: {best['file']}")
        print(f"Coverage: {best['coverage']['token_coverage']}% tokens, {best['coverage']['type_coverage']}% types")
        return best['file']
    else:
        # Fallback to standard naming
        model_name = f"{source_language}_{target_language}_biblical.json"
        model_path = Path(f"models/biblical_alignment/{model_name}")
        
        if model_path.exists():
            print(f"Using standard model: {model_name}")
            return str(model_path)
        else:
            print(f"No model found for {source_language} to {target_language}")
            print(f"Train one with: abba-align train --source {source_language} --target {target_language}")
            return None

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        select_model(sys.argv[1], *sys.argv[2:])
    else:
        print("Usage: select_model.py <source_text_type> [target_language] [use_case]")
        print("Example: select_model.py old_testament english bible_study")
```

## Model Validation Checklist

Before using a model, verify:

1. **Model exists and loads**
   ```python
   import json
   with open('models/biblical_alignment/hebrew_english_biblical.json') as f:
       model = json.load(f)
   ```

2. **Check training features**
   ```bash
   grep "features_enabled" models/biblical_alignment/*_report.json
   ```

3. **Verify Strong's integration**
   ```bash
   grep "strongs" models/biblical_alignment/*_report.json
   ```

4. **Test on sample verse**
   ```bash
   abba-align annotate \
       --input sample_verse.json \
       --output test_output.json \
       --model models/biblical_alignment/hebrew_english_biblical.json
   ```

## Fallback Strategies

### 1. Missing Model Fallback

```python
def get_model_with_fallback(source_lang, target_lang):
    """Get model with fallback options."""
    # Try specific model
    specific = Path(f"models/biblical_alignment/{source_lang}_{target_lang}_biblical.json")
    if specific.exists():
        return specific
    
    # Try English if target is similar
    if target_lang in ['american_english', 'british_english']:
        english = Path(f"models/biblical_alignment/{source_lang}_english_biblical.json")
        if english.exists():
            return english
    
    # Try base model without features
    base = Path(f"models/{source_lang}_{target_lang}.json")
    if base.exists():
        return base
    
    return None
```

### 2. Multi-Model Ensemble

For critical applications, use multiple models:

```python
def ensemble_alignment(text, models):
    """Combine predictions from multiple models."""
    all_alignments = []
    
    for model_path in models:
        alignments = get_alignments(text, model_path)
        all_alignments.append(alignments)
    
    # Merge with voting or averaging
    return merge_alignments(all_alignments)
```

## Troubleshooting Model Issues

### "Model not found"
1. Check path: `ls -la models/biblical_alignment/`
2. Train model: `abba-align train --source hebrew --target english`
3. Download pre-trained: See project releases

### "Invalid model format"
1. Verify JSON: `python -m json.tool model.json`
2. Check version compatibility
3. Re-train with current version

### "Low coverage with model"
1. Check model report for features
2. Verify source language matches text
3. Consider training with more features
4. Add parallel passages for rare words

## Best Practices

1. **Use consistent models** across your application
2. **Document model versions** in your project
3. **Validate coverage** before deployment
4. **Keep model catalog updated** as you train new models
5. **Version control** model files for reproducibility

## Summary

Model selection in ABBA-Align is straightforward:
- Use naming convention to identify models
- Check model reports for quality metrics  
- Auto-detection handles most cases
- Create catalogs for complex deployments
- Validate coverage before production use

The standard models (`hebrew_english_biblical.json` and `greek_english_biblical.json`) work well for most Bible study and translation applications.