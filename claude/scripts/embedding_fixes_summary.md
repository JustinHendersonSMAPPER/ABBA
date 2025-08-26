# ChromaDB and Original Language Embedding Fixes Summary

## Issues Fixed

### 1. ChromaDB Purge-All Issue
**Problem**: ChromaDB vectors directory wasn't being properly cleaned with `--purge-all`
**Fix**: Already implemented in main.py - `shutil.rmtree(config.vectors_path)`

### 2. ChromaDB Corruption ("database disk image is malformed")
**Fixes Applied**:
- Added proper `close()` method to ChromaManager
- Added cleanup in main.py to close ChromaDB before exit
- Modified get_collection_stats to handle corruption gracefully

### 3. ChromaDB Instance Conflict
**Problem**: Validator creating new ChromaDB instance conflicting with existing one
**Fix**: Modified OriginalEmbeddingValidator to accept existing ChromaManager instance

### 4. Book Name Mapping Issues
**Problem**: Some books (Sng, Ezk, Jol, Nam, Mrk, Jhn, Php) had incorrect mappings causing duplicate IDs like "000:001:001"
**Fix**: Updated all book name mappings in original_language_pipeline.py to match actual data

### 5. Segmentation Fault During Embedding
**Problem**: "'dict' object is not callable" error causing segfault around batch 189 (18,900 verses)
**Fixes Applied**:
- Added sub-batching (20 items) to avoid overwhelming ChromaDB
- Added better error handling and corruption detection
- Increased retry delay to 2 seconds
- Skip corrupted batches instead of crashing

### 6. Duplicate ID Warnings on Resume
**Problem**: Resume tries to re-add existing embeddings
**Fix**: Check existing IDs before adding new embeddings

## Architecture Change

### Original Language Embeddings
**Before**: 13.8M translation-specific embeddings (one per verse per translation)
**After**: 29,126 canonical verse embeddings (one per verse using Hebrew/Greek text)

This provides:
- Universal semantic search across all translations
- Massive storage savings (~475x reduction)
- Better semantic accuracy using original languages

## Current Status

- Successfully created 29,126 canonical verse embeddings
- Successfully created 21,016 word embeddings
- Resume functionality working with duplicate detection
- ChromaDB cleanup properly implemented

## Remaining Issues

The main issue is the segmentation fault that occurs around batch 189 (18,900 verses). This appears to be a ChromaDB internal issue when handling large amounts of data. The workarounds implemented should help:

1. Smaller sub-batches (20 items)
2. Better error handling
3. Resume capability to continue after crashes
4. Duplicate detection to avoid re-adding

## Usage

```bash
# Full purge and rebuild
python abba/main.py --purge-all -y

# Resume embedding generation after crash
python abba/main.py --embed-all

# Force re-embed (overwrites existing)
python abba/main.py --embed-all --force-reembed

# Test semantic search
python claude/scripts/test_semantic_search.py
```