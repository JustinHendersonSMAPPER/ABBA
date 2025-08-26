# Canon-Aware Import System

## Overview

ABBA now includes a canon-aware import system that intelligently recognizes different biblical canons (Protestant, Catholic, Orthodox, Ethiopian, and Jewish) to eliminate false warnings during import while maintaining data integrity checks.

## Biblical Canons Supported

### 1. Hebrew Bible/Tanakh (39 books)
- **Canon Type**: `BOOKS_39 = "hebrew"`
- **Description**: Jewish canonical texts only (Old Testament)
- **Translations**: JPS, TNK, OJB

### 2. Protestant Canon (66 books)
- **Canon Type**: `BOOKS_66 = "protestant"`
- **Description**: Standard Protestant Bible (39 OT + 27 NT)
- **Translations**: KJV, NIV, ESV, NASB, NET, BSB, etc.
- **Default**: Unrecognized translations default to this canon

### 3. Catholic Canon (73 books)
- **Canon Type**: `BOOKS_73 = "catholic"`
- **Description**: Includes 7 deuterocanonical books plus additions to Daniel and Esther
- **Translations**: NABRE, DRC, CPDV, RSV-CE, NRSV-CE, NJB, Vulgate
- **Additional Books**: 
  - Tobit (TOB)
  - Judith (JDT)
  - Wisdom (WIS)
  - Sirach/Ecclesiasticus (SIR)
  - Baruch (BAR)
  - 1-2 Maccabees (1MA, 2MA)
  - Additions to Esther (ESG)
  - Additions to Daniel (LJE, S3Y, SUS, BEL)

### 4. Orthodox Canon (76+ books)
- **Canon Type**: `BOOKS_76_PLUS = "orthodox"`
- **Description**: Includes all Catholic books plus additional texts
- **Translations**: EOB, OSB, LXX, SEPT
- **Additional Books** (beyond Catholic):
  - 1 Esdras (1ES)
  - 3 Maccabees (3MA)
  - Prayer of Manasseh (MAN)
  - Psalm 151 (PS2)
  - Some traditions: 4 Maccabees (4MA), 2 Esdras (2ES)

### 5. Ethiopian Canon (81 books)
- **Canon Type**: `BOOKS_81 = "ethiopian"`
- **Description**: Most extensive biblical canon
- **Translations**: Ethiopian/Amharic versions
- **Additional Books** (beyond Orthodox):
  - 1 Enoch (ENO)
  - Jubilees (JUB)
  - 4-6 Ezra (4ES, 5ES, 6ES)

## Translation Detection

The system automatically detects which canon a translation follows based on patterns in the translation ID:

```python
def get_translation_canon(translation_id: str) -> Canon:
    tid = translation_id.upper()
    
    # Catholic indicators
    if any(indicator in tid for indicator in [
        'NABRE', 'DRC', 'CPDV', 'RSV-CE', 'NRSV-CE', 'CE', 
        'CATHOLIC', 'NJB', 'CCB', 'GNT-CE', 'VULG', 'VUL'
    ]):
        return Canon.BOOKS_73
        
    # Orthodox indicators
    if any(indicator in tid for indicator in [
        'EOB', 'OSB', 'ORTHODOX', 'LXX', 'SEPT', 'SAAS'
    ]):
        return Canon.BOOKS_76_PLUS
    
    # ... etc
```

## Import Behavior

### Before Canon-Aware System
```
WARNING:abba.parallel_import:Unknown book ID: WIS
WARNING:abba.parallel_import:Unknown book ID: SIR
WARNING:abba.parallel_import:Unknown book ID: 1MA
```

### After Canon-Aware System
- Any Bible importing WIS, SIR, 1MA, etc.: **Silent** (known deuterocanonical book)
- Any Bible importing truly unknown book (e.g., 'XYZ'): **Warning** (data integrity issue)
- The system imports exactly what's in bible.db for each translation
- Only the standard 66 books are mapped to our database (1-66)
- All known deuterocanonical/apocryphal books are silently skipped

## Validation Logic

The simplified validation follows this decision tree:

1. **Is the book in the standard 66-book mapping?**
   - Yes → Import normally into database
   - No → Continue to step 2

2. **Is this a recognized deuterocanonical/apocryphal book?**
   - Yes → Skip silently (known extended canon book)
   - No → Warn and skip (truly unknown book - potential data issue)

## Implementation Details

### Key Components

1. **Canon Enum** (`abba/parallel_import.py`):
```python
class Canon(Enum):
    BOOKS_39 = "hebrew"
    BOOKS_66 = "protestant"
    BOOKS_73 = "catholic"
    BOOKS_76_PLUS = "orthodox"
    BOOKS_81 = "ethiopian"
```

2. **Extended Canon Book Sets**:
```python
EXTENDED_CANON_BOOKS: Dict[Canon, Set[str]] = {
    Canon.BOOKS_73: {
        'TOB', 'JDT', 'ESG', 'WIS', 'SIR', 'BAR', '1MA', '2MA',
        'LJE', 'S3Y', 'SUS', 'BEL'
    },
    # ... etc
}
```

3. **Smart Warning Logic**:
```python
# Comprehensive list of known deuterocanonical books
ALL_KNOWN_EXTENDED_BOOKS = {
    'TOB', 'JDT', 'ESG', 'WIS', 'SIR', 'BAR', '1MA', '2MA',
    'LJE', 'S3Y', 'SUS', 'BEL', '1ES', '3ES', '3MA', '4MA',
    'MAN', 'PS2', '2ES', '4ES', '5ES', '6ES', 'ENO', 'JUB',
    'PSS', 'LAO', 'ODE', 'EZA', 'DAG', 'PS3', 'POL', 'EEP', 'ADE'
}

if book_id == 0:
    if book_str in ALL_KNOWN_EXTENDED_BOOKS:
        # Known deuterocanonical book - skip silently
        logger.debug(f"Skipping extended canon book {book_str}")
        continue
    else:
        # Truly unknown - warn
        logger.warning(f"Unknown book ID: {book_str}")
```

## Benefits

1. **Reduced Noise**: Eliminates thousands of false warnings during import
2. **Better Data Integrity**: Still catches truly unknown or corrupted book IDs
3. **Theological Awareness**: Respects different biblical traditions
4. **Extensible**: Easy to add new canons or translation patterns

## Testing

Run the canon validation test:
```bash
poetry run python claude/scripts/test_canon_aware_import.py
```

This verifies:
- Translation → Canon mapping
- Book validation logic
- Extended canon book recognition

## Future Enhancements

1. **User Configuration**: Allow users to specify canon preferences
2. **Book Mapping**: Optionally import Apocrypha books with extended IDs (67+)
3. **Canon Metadata**: Store canon type in translation metadata
4. **Reporting**: Generate canon-specific import statistics