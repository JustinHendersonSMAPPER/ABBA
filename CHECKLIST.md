# ABBA Modern Alignment System - Implementation Checklist

## Important: Iterative Development Approach

**After completing each section, run `python src/main.py` to ensure the system still works.**
Each phase builds incrementally while maintaining a working system throughout.

## Phase 1: Remove Strong's and Update Downloads ✅

### Step 1.1: Remove Strong's Concordance Completely ✅
- [x] Delete Strong's-related files:
  - [x] `data/sources/lexicons/strongs_*.xml`
  - [x] `data/sources/lexicons/strongs_*.json`
  - [x] `src/abba_model_builder.py`
  - [x] `src/abba_model_builder_enhanced.py`
  - [x] `models/biblical_alignment/*` (all Strong's models)
- [x] Remove Strong's code from `scripts/download_sources.py`:
  - [x] `download_and_extract_strongs()` function
  - [x] `convert_strongs_to_json()` function
  - [x] Related function calls in `main()`
- [x] Update `src/main.py`:
  - [x] Remove model builder imports
  - [x] Remove model building step
  - [x] Comment out coverage analysis temporarily
- [x] **Run `python src/main.py`** - Should complete data download only ✅

### Step 1.2: Create Unified Download Script ✅
- [x] Update `scripts/download_sources.py` to be main download script:
  - [x] Keep existing Hebrew/Greek Bible downloads
  - [x] Add OSHB morphological data download
  - [x] Add SBLGNT/MorphGNT morphological data download
  - [x] Keep translation download via `abba_data_downloader.py` import
- [x] Ensure all downloads go to organized directories:
  - [x] `data/sources/hebrew/` - Hebrew text files (40 files)
  - [x] `data/sources/greek/` - Greek text files (27 files)
  - [x] `data/sources/morphology/hebrew/` - Hebrew morphological data (40 JSON files)
  - [x] `data/sources/morphology/greek/` - Greek morphological data (27 JSON files)
  - [x] `data/sources/translations/` - All Bible translations (1025 files)
- [x] Convert morphological data to unified JSON format:
  - [x] Convert OSHB XML to JSON with lemma, morph, POS data
  - [x] Convert MorphGNT TXT to JSON with parsed features
  - [x] Both formats now consistent and easy to parse
- [x] **Run `python scripts/download_sources.py`** - Verify all data downloads ✅
- [x] **Run `python src/main.py`** - Should show successful data preparation ✅

## Phase 2: Basic Alignment System ✅

### Step 2.1: Create Simple Position-Based Aligner ✅
- [x] Create `src/abba/alignment/position_aligner.py`:
  - [x] Map word positions proportionally (word 1/10 → word 1/12, etc.)
  - [x] Return low confidence scores (0.3-0.5)
  - [x] Handle different verse lengths gracefully
- [x] Update `src/main.py`:
  - [x] Import position aligner
  - [x] Use it to generate basic alignments
  - [x] Show sample alignments in output
- [x] **Run `python src/main.py`** - Should show position-based alignments ✅

### Step 2.2: Re-enable Coverage Analysis ✅
- [x] Update `src/abba_coverage_analyzer.py`:
  - [x] Remove Strong's model dependencies
  - [x] Accept aligner object instead of models
  - [x] Use aligner to check word mappings
- [x] Update `src/main.py`:
  - [x] Pass position aligner to coverage analyzer
  - [x] Re-enable coverage analysis step
- [x] **Run `python src/main.py`** - Should show ~30-40% coverage with position aligner ✅ (showing 60% average)

## Phase 3: Morphological Analysis Layer ✅

### Step 3.1: Parse Downloaded Morphological Data ✅
- [x] Create `src/abba/morphology/oshb_parser.py`:
  - [x] Parse OSHB JSON format (already converted from XML)
  - [x] Extract lemma, POS, morphology codes
  - [x] Map to verse references
- [x] Create `src/abba/morphology/sblgnt_parser.py`:
  - [x] Parse SBLGNT/MorphGNT JSON format
  - [x] Extract lemma, POS, case, mood, tense, voice
  - [x] Map to verse references
- [x] Test parsers with sample files
- [x] **Run parser tests** - Verify data extraction works ✅

### Step 3.2: Create Morphological Aligner ✅
- [x] Create `src/abba/alignment/morphological_aligner.py`:
  - [x] Load parsed morphological data
  - [x] Match words by lemma similarity
  - [x] Boost score for POS agreement
  - [x] Return confidence based on feature match
- [x] Update `src/main.py`:
  - [x] Create morphological aligner instance
  - [x] Show it as second alignment method
- [x] **Run `python src/main.py`** - Should show morphological alignments ✅

### Step 3.3: Combine Position and Morphological ✅
- [x] Create `src/abba/alignment/ensemble_aligner.py`:
  - [x] Accept list of aligners
  - [x] Weighted combination (morphological=0.7, position=0.3)
  - [x] Return combined confidence scores
- [x] Update `src/main.py` to use ensemble
- [x] **Run `python src/main.py`** - Should show ~50-60% coverage ✅ (showing 38.5%)

## Phase 4: Statistical Alignment ✅

### Step 4.1: Prepare Parallel Corpus ✅
- [x] Create `scripts/prepare_parallel_corpus.py`:
  - [x] Read translations from `data/sources/translations/`
  - [x] Extract parallel verses (same verse across languages)
  - [x] Format for statistical alignment tools
  - [x] Start with 3 high-quality translations (KJV, WEB, ASV)
- [x] **Run preparation script** - Generate `data/parallel_corpus/` ✅
- [x] Verify format is correct for alignment tools ✅ (600 verse pairs)

### Step 4.2: Install Statistical Alignment Tools ✅
- [x] Used existing libraries (numpy, scikit-learn) instead of eflomal/fastalign
- [x] Created simple co-occurrence based statistical aligner
- [x] No additional dependencies needed
- [x] **Run test** - Statistical models trained successfully ✅

### Step 4.3: Train Statistical Models ✅
- [x] Create `scripts/train_statistical_alignment.py`:
  - [x] Load parallel corpus
  - [x] Train Hebrew→English and Greek→English models
  - [x] Save models to `models/statistical/`
- [x] Create `src/abba/alignment/statistical_aligner.py`:
  - [x] Load trained models
  - [x] Provide alignment scores
- [x] Update ensemble to include statistical (weight=0.5)
- [x] **Run `python src/main.py`** - Coverage: 7.6% with full Bible corpus (30,936 verse pairs)
- [x] **Auto-build corpus**: `src/main.py` now automatically builds full Bible corpus if missing

## Phase 5: Confidence and Evaluation

### Step 5.1: Add Confidence Scoring
- [ ] Update `src/abba/alignment/ensemble_aligner.py`:
  - [ ] Calculate agreement between methods
  - [ ] Higher agreement = higher confidence
  - [ ] Return confidence with each alignment
- [ ] Update coverage analyzer to show confidence distribution
- [ ] **Run `python src/main.py`** - Should show confidence breakdown

### Step 5.2: Create Evaluation Framework
- [ ] Create `data/test/gold_alignments.json`:
  - [ ] Start with 20 manually aligned verses
  - [ ] Include Genesis 1:1, John 3:16, Psalm 23:1, etc.
- [ ] Create `scripts/evaluate_alignments.py`:
  - [ ] Compare system output to gold standard
  - [ ] Calculate precision, recall, F1
- [ ] **Run evaluation** - Get baseline metrics

### Step 5.3: Iterative Improvement
- [ ] Identify failure cases from evaluation
- [ ] Adjust ensemble weights based on results
- [ ] Add more test verses incrementally
- [ ] **Run evaluation** after each change
- [ ] Document what improves accuracy

## Phase 6: Neural Models

### Step 6.1: Add Neural Dependencies
- [ ] Add to pyproject.toml: `poetry add sentence-transformers`
- [ ] This will install transformers and torch automatically
- [ ] Create `test_neural_setup.py` to verify installation
- [ ] **Run test** - May take time to download models

### Step 6.2: Create Neural Aligner
- [ ] Create `src/abba/alignment/neural_aligner.py`:
  - [ ] Use multilingual model (e.g., LaBSE)
  - [ ] Compute embeddings for source and target
  - [ ] Calculate cosine similarity
  - [ ] Cache embeddings for performance
- [ ] Add to ensemble with low weight (0.2) initially
- [ ] **Run `python src/main.py`** - First run will be slow

### Step 6.3: Optimize Neural Performance
- [ ] Implement embedding cache in `data/cache/embeddings/`
- [ ] Batch process verses for efficiency
- [ ] Add progress bar for long operations
- [ ] **Run `python src/main.py`** - Should be faster on second run
- [ ] Increase neural weight if it improves accuracy

## Phase 7: Configuration and Output

### Step 7.1: Add Configuration Support
- [ ] Update `src/main.py` to accept:
  - [ ] `--confidence-threshold` from CLI (default: 0.7)
  - [ ] `ABBA_CONFIDENCE_THRESHOLD` from environment variable
  - [ ] CLI overrides environment variable if both present
- [ ] Update ensemble aligner to filter by confidence threshold
- [ ] **Run `python src/main.py --confidence-threshold 0.8`** - Should filter results
- [ ] **Run `ABBA_CONFIDENCE_THRESHOLD=0.9 python src/main.py`** - Should use env var

### Step 7.2: JSON Output Format
- [ ] Create `src/abba/export/json_alignment_exporter.py`:
  - [ ] Export alignments as structured JSON
  - [ ] Include confidence scores for each alignment
  - [ ] Include metadata (source language, target language, methods used)
- [ ] Update `src/main.py`:
  - [ ] Add `--output-format json` option (only JSON supported initially)
  - [ ] Default output directory: `aligned_output/`
- [ ] Validate JSON output structure:
  - [ ] Valid JSON syntax
  - [ ] Contains all required fields
  - [ ] Confidence scores are between 0 and 1
- [ ] **Run `python src/main.py --output-format json`** - Generate JSON files
- [ ] **Validate output** - Use `jq` or Python to verify JSON structure

### Step 7.3: Performance Baseline
- [ ] Time alignment of single book (Genesis)
- [ ] Document memory usage
- [ ] Identify bottlenecks for future optimization
- [ ] **Run with timing** - Record baseline performance

## Dependencies to Add to pyproject.toml

### Phase-by-Phase Dependencies

```toml
# Phase 4 - Statistical alignment
poetry add eflomal
# or if that fails:
# poetry add pyfastalign

# Phase 6 - Neural models
poetry add sentence-transformers
# This automatically installs:
# - transformers
# - torch
# - numpy
# - scikit-learn
```

## Order of Implementation

1. **Phase 1**: Remove Strong's concordance completely
2. **Phase 2**: Create basic position aligner (immediate functionality)
3. **Phase 3**: Add morphological parsing and alignment
4. **Phase 4**: Add statistical alignment (biggest accuracy boost)
5. **Phase 5**: Add confidence scoring and evaluation
6. **Phase 6**: Neural models (optional enhancement)

## Key Principles

- **Test continuously**: Run `python src/main.py` after every change
- **Incremental progress**: Each phase adds value independently
- **No broken states**: System always produces some output
- **Start simple**: Test with 10 verses before full Bible
- **Measure progress**: Track coverage percentage improvements