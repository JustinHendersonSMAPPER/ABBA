# ABBA Validation System Documentation

## Overview

The ABBA project implements a robust validation system to ensure data integrity during import and embedding operations. The system uses fast hash-based validation with MurmurHash3 and hierarchical job tracking to handle interruptions gracefully.

## Architecture

### Components

1. **StateTracker** (`abba/state_tracker.py`)
   - Tracks operation and job states
   - Automatically detects interrupted operations
   - Persists state to JSON for recovery

2. **HashValidator** (`abba/hash_validator.py`)
   - Uses MurmurHash3 for fast content validation
   - Validates imports and embeddings
   - Supports streaming and aggregate validation

3. **OperationManager** (`abba/operation_manager.py`)
   - Orchestrates cleanup and validation
   - Handles job-level granularity
   - Integrates state tracking with validation

## State Management

### Operation States

Operations and jobs can be in one of these states:

- `NEVER_STARTED` - Not yet begun
- `IN_PROGRESS` - Currently running
- `COMPLETED` - Successfully finished with validation
- `FAILED` - Failed with error
- `INTERRUPTED` - Was in progress but system stopped (auto-detected)

### Hierarchical Structure

```json
{
  "operations": {
    "import_translations": {
      "status": "in_progress",
      "started_at": "2025-01-28T10:00:00",
      "jobs": {
        "KJV": {
          "status": "completed",
          "validation": {
            "valid": true,
            "message": "Successfully validated 31,102 verses",
            "timestamp": "2025-01-28T10:05:00"
          }
        },
        "ASV": {
          "status": "interrupted",
          "progress": {
            "verses_imported": 15000,
            "total_verses": 31102
          }
        }
      }
    }
  }
}
```

## Validation Methods

### 1. Hash-Based Validation

Each verse is hashed using MurmurHash3 with a deterministic key:

```python
key = f"{translation_id}:{book_id:03d}:{chapter:03d}:{verse:03d}:{text}"
hash = mmh3.hash(key, seed=42)
```

This provides:
- Fast validation (~3GB/s throughput)
- Deterministic results
- Detection of any content changes

### 2. Validation Approaches

#### Streaming Validation
- Validates verses one by one during import
- Immediately detects mismatches
- Memory efficient for large datasets

#### Aggregate Validation
- Calculates XOR checksum of all verse hashes
- Very fast whole-translation validation
- Good for quick integrity checks

#### Sample Validation
- Validates specific verses or samples
- Useful for spot-checking
- Helps debug specific issues

### 3. Tolerance Levels

The system allows configurable tolerance for real-world data:

- **Verse counts**: 5% variance allowed (empty verses, textual variants)
- **Word counts**: 20% variance allowed (deduplication, morphology)
- **Exact match**: Available for critical operations

## Import Process

### Translation Import Flow

1. **Start Job**
   ```python
   op_manager.start_job("import_translations", "KJV", db_manager)
   ```
   - Checks for interrupted state
   - Runs cleanup if needed (deletes partial data)
   - Marks job as in_progress

2. **Import Data**
   - Extract verses from source bible.db
   - Calculate hash for each verse
   - Store in destination abba.db with hash

3. **Validate Import**
   ```python
   op_manager.complete_job(
       "import_translations", "KJV", 
       db_manager,
       validation_params={"use_checksum": True}
   )
   ```
   - Compares source and destination hashes
   - Validates verse counts match
   - Marks job as completed with validation results

### Cleanup on Failure

If a job is interrupted:

1. On next run, StateTracker detects `INTERRUPTED` status
2. OperationManager runs cleanup:
   - Deletes partial verses for that translation
   - Deletes partial words for that translation
   - Resets job state
3. Job can be restarted cleanly

## Embedding Process

### Verse Embedding Flow

1. **Start Job**
   ```python
   op_manager.start_job("embed_verses", "ARBNAV", db_manager, chroma_manager)
   ```

2. **Generate Embeddings**
   - Read verses from database
   - Generate embeddings with model
   - Store in ChromaDB with source hash

3. **Validate Embeddings**
   - Verify all verses have embeddings
   - Check source hashes match
   - Validate counts within tolerance

### Word Embedding Validation

Word embeddings are validated differently due to deduplication:

- Expected: Unique words from database
- Actual: Embedded words in ChromaDB
- Tolerance: 20% variance allowed

## Usage Examples

### Basic Import with Validation

```python
from abba.operation_manager import OperationManager
from pathlib import Path

# Initialize
op_manager = OperationManager(
    state_file=Path("bible_data/.abba_state.json"),
    source_db_path=Path("bible_data/bible.db")
)

# Import a translation
translation_id = "ESV"

if op_manager.start_job("import_translations", translation_id, db_manager):
    try:
        # Do the import
        verse_count = import_translation(translation_id)
        
        # Validate and complete
        success = op_manager.complete_job(
            "import_translations",
            translation_id,
            db_manager
        )
        
        if success:
            print(f"Successfully imported and validated {translation_id}")
        else:
            print(f"Validation failed for {translation_id}")
            
    except Exception as e:
        op_manager.tracker.fail_job("import_translations", translation_id, str(e))
```

### Check Status

```python
# Get summary of all operations
summary = op_manager.tracker.get_summary()
print(json.dumps(summary, indent=2))

# Check for interrupted jobs
warnings = op_manager.handle_interrupted_operations()
for warning in warnings:
    print(warning)
```

### Manual Validation

```python
from abba.hash_validator import HashValidator

validator = HashValidator()

# Quick validation using checksums
is_valid, message = validator.quick_validate(
    "bible_data/bible.db",
    "bible_data/abba.db",
    "KJV"
)

# Detailed validation with streaming
is_valid, message, details = validator.validate_translation_import(
    "KJV",
    "bible_data/bible.db", 
    "bible_data/abba.db"
)

if not is_valid:
    print(f"Validation failed: {message}")
    print(f"Details: {json.dumps(details, indent=2)}")
```

## Benefits

1. **Data Integrity**: Every verse and embedding is validated
2. **Automatic Recovery**: Interrupted operations are cleaned up
3. **Granular Control**: Per-translation tracking and validation
4. **Performance**: Fast MurmurHash validation (~3GB/s)
5. **Transparency**: Clear status tracking and error reporting

## Configuration

### State File Location
Default: `bible_data/.abba_state.json`

### Validation Options
- `use_checksum`: Enable aggregate checksum validation
- `tolerance`: Set variance tolerance (0.0 - 1.0)
- `sample_size`: Number of verses to sample for spot checks

### Hash Configuration
- Seed: 42 (consistent across runs)
- Algorithm: MurmurHash3 32-bit

## Troubleshooting

### Common Issues

1. **"Validation failed: Hash mismatches"**
   - Check encoding consistency
   - Verify source data hasn't changed
   - Look at specific verse differences

2. **"Cleanup failed"**
   - Ensure database is not locked
   - Check write permissions
   - Verify ChromaDB is accessible

3. **"Too many verses without embeddings"**
   - Check embedding model loaded correctly
   - Verify GPU/CPU resources available
   - Look for empty or corrupted verses

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger('abba').setLevel(logging.DEBUG)
```

This will show:
- Hash calculations
- Cleanup operations
- Validation comparisons
- State transitions