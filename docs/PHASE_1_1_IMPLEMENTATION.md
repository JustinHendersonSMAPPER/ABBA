# Phase 1.1 Implementation: Canonical Data Schema

## Overview

Phase 1.1 establishes the foundational data structures for the ABBA project, including:
- JSON Schema for canonical verse format
- Book ID standardization (3-letter codes)
- Verse ID normalization system
- Split verse handling
- Versification mapping rules

## Completed Components

### 1. JSON Schema for Canonical Verse Format
**File**: `src/abba/schemas/canonical_verse.schema.json`

The canonical verse schema defines the complete structure for storing biblical verses with:
- **Core fields**: canonical_id, book, chapter, verse, translations
- **Original language support**: hebrew_tokens, greek_tokens with full morphological data
- **Annotations**: Hierarchical topic/theme tagging system
- **Cross-references**: Bidirectional reference support with relationship types
- **Timeline data**: Historical context with date ranges and precision
- **Geographic data**: Location information with coordinates
- **Canon support**: Track which traditions include each verse
- **Textual variants**: Manuscript differences with witness support

Key features:
- Strict validation using JSON Schema Draft 7
- Support for verse parts (a, b, c) for split verses
- Extensible annotation system with confidence scores
- Multi-language translation storage with RTL support

### 2. Book ID Standardization
**File**: `src/abba/book_codes.py`

Comprehensive book code system with:
- **66 canonical book codes**: All Protestant canon books with 3-letter codes
- **Book metadata**: Full names, abbreviations, chapter counts, testament
- **Alias mapping**: 400+ variations mapped to canonical codes
  - Multiple abbreviation styles (Gen, Gn, Ge)
  - Full names and common variations
  - Numbered books (1 Samuel, I Samuel, 1Sam)
- **Canon membership**: Track which books belong to which traditions
- **Utility functions**:
  - `normalize_book_name()`: Convert any book reference to canonical code
  - `get_book_info()`: Retrieve metadata for a book
  - `get_books_by_testament()`: List OT or NT books
  - `get_book_order()`: Canonical ordering (1-66)

### 3. Verse ID Normalization System
**File**: `src/abba/verse_id.py`

Robust verse identification system:
- **VerseID class**: Immutable, comparable verse references
  - Supports canonical format: "GEN.1.1"
  - Handles verse parts: "ROM.3.23a"
  - Sortable and hashable for use in collections
- **Flexible parsing**: Handles multiple input formats
  - Canonical: "GEN.1.1"
  - Common: "Genesis 1:1" or "Gen 1:1"
  - Compact: "Gn1:1"
- **VerseRange class**: For passage references
  - Parse ranges like "GEN.1.1-GEN.1.5"
  - Check containment and expand to verse lists
- **Navigation methods**: next_verse(), previous_verse()
- **Validation**: Ensures valid book codes and reasonable chapter/verse numbers

### 4. Split Verse Schema
**File**: `src/abba/schemas/split_verse.schema.json`

Specialized schema for handling verse divisions:
- **Split types**: split, combined, reordered, variant
- **Part tracking**: How verses are divided across translations
- **Mapping rules**: Define relationships between versification systems
- **Translation tracking**: Which versions are affected by splits

Example: Psalm 13:5-6 is one verse in some translations but split in others

### 5. Versification Mapping System
**File**: `src/abba/versification.py`

Complete versification handling:
- **Systems defined**: MT, LXX, Vulgate, KJV, Modern, Orthodox, Catholic
- **VersificationMapper class**: Convert verses between systems
  - Handles offsets (e.g., Psalm superscriptions)
  - Manages split/merged verses
  - Tracks chapter boundary differences
- **Difference documentation**: Common versification issues
  - Psalm titles counted as verse 1 in Hebrew
  - Malachi 3-4 chapter division
  - Romans 16:25-27 placement variations
- **Translation mapping**: Links versions to their versification system
- **Comprehensive rules**: Documentation of all mapping patterns

## Usage Examples

### Basic Verse Operations
```python
from abba import parse_verse_id, normalize_book_name

# Parse various formats
verse1 = parse_verse_id("Genesis 1:1")
verse2 = parse_verse_id("GEN.1.1")
verse3 = parse_verse_id("Gn 1:1")
# All return VerseID(book='GEN', chapter=1, verse=1)

# Normalize book names
assert normalize_book_name("1 Samuel") == "1SA"
assert normalize_book_name("song of songs") == "SNG"
```

### Versification Mapping
```python
from abba import VersificationMapper, VersificationSystem

mapper = VersificationMapper()

# Map Psalm 3:2 from Hebrew to Modern
# (Hebrew counts title as verse 1)
modern_verses = mapper.map_verse(
    "PSA.3.2",
    VersificationSystem.MT,
    VersificationSystem.MODERN
)
# Returns [VerseID('PSA.3.1')]
```

### Working with Ranges
```python
from abba import parse_verse_range

# Parse a range
range = parse_verse_range("Romans 3:21-26")

# Check if verse is in range
verse = parse_verse_id("ROM.3.23")
assert verse in range

# Expand to list
verses = range.to_list()
```

## File Structure Created

```
src/abba/
├── __init__.py              # Package initialization with exports
├── book_codes.py            # Book standardization system
├── verse_id.py              # Verse ID parsing and normalization
├── versification.py         # Versification mapping system
└── schemas/
    ├── canonical_verse.schema.json  # Main verse schema
    └── split_verse.schema.json      # Split verse handling
```

## Next Steps

With Phase 1.1 complete, the foundation is ready for:
1. **Phase 1.2**: Build data import pipeline
   - Hebrew/Greek XML parsers
   - Translation JSON normalizer
   - Strong's lexicon parser
2. **Phase 1.3**: Implement verse alignment system
   - Build versification bridge tables
   - Create the Unified Reference System
3. **Phase 1.4**: Set up data validation
   - Schema validators
   - Integrity checkers

## Technical Notes

- All code follows Python type hints for better IDE support
- Enums used for controlled vocabularies (Testament, Canon, etc.)
- Dataclasses for clean data structures
- Comprehensive docstrings for all public functions
- Regex patterns optimized for performance
- Immutable data structures where appropriate

## Achievement Summary

✅ **Phase 1.1 Complete**: All 5 tasks accomplished
- Created comprehensive JSON schemas
- Implemented book code standardization
- Built verse ID normalization system  
- Designed split verse handling
- Documented versification mapping rules

The foundation is now ready for building the data import pipeline in Phase 1.2.