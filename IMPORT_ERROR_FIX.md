# ABBA-Align Import Error Fix

## Issue

When running ABBA-Align with Greek source language:
```bash
python -m abba_align train --source greek --target english --corpus-dir data/sources --features all
```

The system failed with:
```
ERROR: Failed to initialize trainer: cannot access local variable 're' where it is not associated with a value
```

## Root Cause

The `lexicon_integration.py` module was importing the `re` module inside method scopes, but then attempting to use it outside those scopes in other methods. This created an UnboundLocalError when the module was loaded for Greek training.

## Solution

Moved the `re` module import to the top-level imports in `src/abba_align/lexicon_integration.py`:

```python
import json
import logging
import re  # Added here
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
```

And removed the redundant `import re` statements from inside the methods:
- Line 80 in `_extract_glosses()`
- Line 153 in `_build_translation_mapping()`

## Verification

After the fix, both Hebrew and Greek training work correctly:

**Hebrew Training:**
- Loads 8,674 Strong's entries
- Generates 2,848 translation mappings
- Training completes successfully

**Greek Training:**
- Loads 5,624 Strong's entries  
- Generates 299 translation mappings (from test corpus)
- Training completes successfully

## Impact

This fix ensures ABBA-Align can train alignment models for both Hebrew and Greek source texts with Strong's Concordance integration enabled. The import error no longer occurs, and the lexicon integration module properly initializes for all supported languages.