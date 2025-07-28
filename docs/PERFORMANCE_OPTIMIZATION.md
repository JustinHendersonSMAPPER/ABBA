# Performance Optimization Guide

## Overview

The ABBA import process has been optimized to utilize full CPU capacity through parallel processing. The system can achieve 5-10x speedup compared to sequential processing.

## Parallel Import System

### Key Features

1. **Multi-processing/Threading Support**
   - Processes: Better for CPU-bound operations
   - Threads: Better for I/O-bound operations (usually optimal for database imports)
   - Automatic CPU detection for optimal worker count

2. **Batch Processing**
   - Default batch size: 1,000 verses
   - Reduces database round trips
   - Optimizes memory usage

3. **Database Optimizations**
   ```sql
   PRAGMA synchronous = OFF;      -- Faster writes
   PRAGMA journal_mode = WAL;     -- Better concurrency
   PRAGMA cache_size = 10000;     -- Larger cache
   PRAGMA temp_store = MEMORY;    -- Use RAM for temp tables
   ```

4. **Connection Pooling**
   - Pre-created connections
   - Avoids connection overhead
   - Thread-safe pool management

## Usage

### Basic Parallel Import

```python
from abba.parallel_import import ParallelImporter

# Initialize importer
importer = ParallelImporter(
    source_db_path=Path("bible_data/bible.db"),
    dest_db_path=Path("bible_data/abba.db"),
    max_workers=8  # Or None for auto-detect
)

# Import translations in parallel
results = importer.import_translations_parallel(
    translation_ids=["KJV", "ESV", "NIV", "NASB"],
    use_processes=False,  # Use threads (recommended)
    batch_size=1000,
    show_progress=True
)

# Check results
for tid, result in results.items():
    if result.success:
        print(f"{tid}: {result.verse_count} verses in {result.duration:.1f}s")
    else:
        print(f"{tid}: Failed - {result.error}")
```

### With State Tracking and Validation

```python
from abba.operation_manager import OperationManager

# Setup with operation manager
op_manager = OperationManager(
    state_file=Path("bible_data/.abba_state.json"),
    source_db_path=Path("bible_data/bible.db")
)

importer = ParallelImporter(
    source_db_path=Path("bible_data/bible.db"),
    dest_db_path=Path("bible_data/abba.db"),
    operation_manager=op_manager
)

# Import with automatic validation
results = importer.import_with_validation(
    translation_ids=["KJV", "ESV", "NIV"],
    validate_after=True,
    use_processes=False
)
```

## Performance Tuning

### 1. Choose the Right Parallelism

**For Database Imports (I/O Bound):**
```python
# Threads are usually better
results = importer.import_translations_parallel(
    translations,
    use_processes=False,  # Use threads
    max_workers=8        # 2-4x CPU count often optimal
)
```

**For CPU-Intensive Processing:**
```python
# Processes avoid GIL limitations
results = importer.import_translations_parallel(
    translations,
    use_processes=True,   # Use processes
    max_workers=cpu_count()  # Match CPU cores
)
```

### 2. Optimize Batch Size

```python
# Larger batches = fewer round trips but more memory
# Smaller batches = more responsive, less memory

# For fast SSDs
batch_size = 5000

# For slower storage or limited RAM
batch_size = 500

results = importer.import_translations_parallel(
    translations,
    batch_size=batch_size
)
```

### 3. Database Location

- **Same Drive**: Keep source and destination on same fast SSD
- **Different Drives**: Can improve if one is significantly faster
- **RAM Disk**: For ultimate speed (if data fits)

### 4. System Tuning

**Linux:**
```bash
# Increase file descriptor limits
ulimit -n 4096

# Tune kernel parameters
echo 1 > /proc/sys/vm/overcommit_memory
```

**macOS:**
```bash
# Increase file limits
ulimit -n 4096
```

## Benchmarking

### Run Benchmarks

```bash
cd claude/scripts
python benchmark_parallel_import.py --full
```

### Expected Performance

Based on typical hardware:

| Method | Verses/Second | Speedup |
|--------|--------------|---------|
| Sequential | 10,000 | 1.0x |
| Threads-2 | 18,000 | 1.8x |
| Threads-4 | 35,000 | 3.5x |
| Threads-8 | 50,000 | 5.0x |
| Process-4 | 30,000 | 3.0x |

### Performance Monitoring

```python
# Monitor during import
import psutil
import os

process = psutil.Process(os.getpid())

# Before import
cpu_before = process.cpu_percent()
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# Run import
results = importer.import_translations_parallel(translations)

# After import
cpu_avg = process.cpu_percent(interval=1)
mem_after = process.memory_info().rss / 1024 / 1024  # MB

print(f"CPU Usage: {cpu_avg}%")
print(f"Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
```

## Troubleshooting

### Low CPU Usage

1. **Check I/O bottleneck**:
   ```python
   # Use more workers
   max_workers = cpu_count() * 2
   ```

2. **Enable WAL mode**:
   ```sql
   PRAGMA journal_mode = WAL;
   ```

3. **Use threads instead of processes**:
   ```python
   use_processes = False
   ```

### Database Locked Errors

1. **Use single writer**:
   ```python
   # Serialize writes while parallelizing reads
   max_workers = 1  # For writing
   ```

2. **Increase timeout**:
   ```python
   conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
   ```

### Memory Issues

1. **Reduce batch size**:
   ```python
   batch_size = 100  # Smaller batches
   ```

2. **Limit workers**:
   ```python
   max_workers = 2  # Fewer parallel operations
   ```

3. **Use processes**:
   ```python
   use_processes = True  # Each process has separate memory
   ```

## Integration with Existing Code

To integrate parallel import into the existing extractor:

```python
# In bible_extractor.py
def extract_translations_parallel(self, translation_ids=None):
    """Extract translations using parallel processing."""
    
    if not translation_ids:
        translation_ids = self.get_all_translation_ids()
    
    # Use parallel importer
    importer = ParallelImporter(
        source_db_path=self.db_path,
        dest_db_path=self.config.abba_db_path,
        operation_manager=self.operation_manager
    )
    
    # Import with optimal settings
    results = importer.import_with_validation(
        translation_ids=translation_ids,
        validate_after=True,
        use_processes=False  # Threads usually better
    )
    
    # Summary
    successful = sum(1 for r in results.values() if r.success)
    total_verses = sum(r.verse_count for r in results.values())
    total_time = sum(r.duration for r in results.values())
    
    print(f"Imported {successful}/{len(translation_ids)} translations")
    print(f"Total verses: {total_verses:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average rate: {total_verses/total_time:.0f} verses/second")
    
    return results
```

## Best Practices

1. **Start with threads**: Usually optimal for database operations
2. **Monitor CPU usage**: Adjust workers if < 80% utilization
3. **Validate after import**: Ensures data integrity
4. **Use progress bars**: For user feedback on long operations
5. **Handle failures gracefully**: Some translations may fail
6. **Benchmark your system**: Optimal settings vary by hardware

## Example: Full Import Pipeline

```python
import multiprocessing as mp
from pathlib import Path
from abba.parallel_import import ParallelImporter
from abba.operation_manager import OperationManager

# Setup
source_db = Path("bible_data/bible.db")
dest_db = Path("bible_data/abba.db") 
state_file = Path("bible_data/.abba_state.json")

# Initialize components
op_manager = OperationManager(state_file, source_db)
importer = ParallelImporter(
    source_db, 
    dest_db,
    op_manager,
    max_workers=mp.cpu_count() * 2  # Oversubscribe for I/O
)

# Get all translations
with sqlite3.connect(str(source_db)) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT translation_id FROM verses")
    all_translations = [row[0] for row in cursor.fetchall()]

print(f"Found {len(all_translations)} translations to import")

# Import in chunks to avoid overwhelming system
chunk_size = 50
for i in range(0, len(all_translations), chunk_size):
    chunk = all_translations[i:i+chunk_size]
    print(f"\nImporting translations {i+1}-{i+len(chunk)}...")
    
    results = importer.import_with_validation(
        translation_ids=chunk,
        validate_after=True,
        use_processes=False
    )
    
    # Report any failures
    failures = [tid for tid, r in results.items() if not r.success]
    if failures:
        print(f"Failed translations: {failures}")

print("\nImport complete!")
```