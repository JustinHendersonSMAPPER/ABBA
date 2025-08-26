# ABBA Implementation Checklist

## Architectural Changes Summary

### Key Modifications from Original Design:

1. **Original Language Embeddings Only**
   - Instead of embedding all 1,204 translations (~13.8M verses)
   - Embed only original Hebrew/Greek texts (~31K unique verses)
   - Maps to translations via canonical verse references
   - Ensures universal semantic search across all languages

2. **User-Defined Concepts with LLM Validation**
   - Concepts are user-supplied, not LLM-generated
   - Traditional Strong's/term mapping as baseline
   - LLM validates each verse for accuracy (removes false positives)
   - LLM scans ALL verses to find missing relevant ones
   - Full traceability of inclusion/exclusion decisions

3. **Build-Time LLM Processing**
   - All LLM processing happens during dataset build
   - No LLM required for actual searching
   - Creates portable, self-contained dataset
   - Trading build time (~65 hours) for search accuracy

4. **Semantic Search Architecture**
   ```
   User Query → Concept Extraction → Original Language Search → Translation Mapping
                                            ↓
                                  Pre-validated Concept Mappings
   ```

## Configuration Architecture Overview

### Environment Variables (.env)
```
# Database Configuration
ABBA_DATABASE_PATH=bible_data/abba.db
ABBA_USE_CACHE=true
ABBA_CACHE_TTL=3600

# Vector Database Configuration  
ABBA_VECTOR_DB_TYPE=chromadb
ABBA_VECTOR_DB_PATH=bible_data/vectors
ABBA_VECTOR_DIMENSIONS=768

# Embedding Model Configuration (Sentence Transformers)
ABBA_EMBEDDING_LIBRARY=sentence-transformers
ABBA_EMBEDDING_MODEL_ENGLISH=intfloat/e5-large-v2
ABBA_EMBEDDING_MODEL_MULTILINGUAL=intfloat/multilingual-e5-base
ABBA_EMBEDDING_CONTEXT_MODE=enhanced  # Include morphology and Strong's

# Ollama Configuration (for semantic analysis only)
ABBA_OLLAMA_HOST=http://localhost:11434
ABBA_OLLAMA_SEMANTIC_MODELS=llama4:scout,command-r-plus:latest
ABBA_OLLAMA_CONSENSUS_THRESHOLD=0.7
ABBA_OLLAMA_TIMEOUT=30
ABBA_OLLAMA_BATCH_SIZE=100

# Search Configuration
ABBA_MAX_RESULTS=50
ABBA_SIMILARITY_THRESHOLD=0.7
ABBA_ENABLE_QUERY_EXPANSION=true

# Performance Configuration
ABBA_PARALLEL_WORKERS=4
ABBA_CONNECTION_POOL_SIZE=10
```

## Phase 1: Database Foundation

### Update Existing Code
- [x] Modify `bible_extractor.py` to read directly from `bible.db` instead of exporting to JSON
- [x] Remove JSON export functionality from main workflow
- [x] Update `main.py` to use new database-first approach

### Configuration Updates
- [x] Update `config.py` to add database settings:
  - [x] `database_path` - SQLite database location
  - [x] `use_cache` - Enable/disable query caching
  - [x] `cache_ttl` - Cache time-to-live
- [x] Add database CLI arguments:
  - [x] `--db-path` - Override database location
  - [x] `--rebuild-db` - Force database rebuild
  - [x] `--no-cache` - Disable caching
- [x] Update `.env.example` with new database variables
- [x] Add database configuration validation

### SQLite Database Setup
- [x] Create `abba/database/` directory structure
- [x] Write `schema.sql` with tables for:
  - [x] `words` table (biblical text with morphology)
  - [x] `lexicon` table (Hebrew/Greek definitions)
  - [x] `morphology` table (grammatical codes)
  - [x] `translations` table (all Bible versions)
  - [x] `books` table (book metadata)
  - [x] Full-text search virtual tables
- [x] Create `sqlite_manager.py` for database operations
- [x] Implement database initialization script

### Data Import Pipeline
- [x] Create STEPBible data parser for TAHOT files (Hebrew OT)
- [x] Create STEPBible data parser for TAGNT files (Greek NT)
- [x] Create lexicon importer (TBESH/TBESG)
- [x] Create morphology code importer (TEHMC/TEGMC)
- [x] Create translation importer from `bible.db`
- [x] Add progress tracking for imports
- [x] Implement data validation checks

### Core API Development
- [x] Create `abba/api/` directory
- [x] Implement `search.py` with methods:
  - [x] `get_verse()` - retrieve by reference
  - [x] `search_strongs()` - find by Strong's number
  - [x] `search_morphology()` - find by grammatical pattern
  - [x] `get_word_analysis()` - complete word breakdown
- [x] Implement `analysis.py` for linguistic analysis
- [x] Add caching layer for frequent queries

## Phase 2: Vector Database Integration

### Vector Database Configuration
- [x] Add vector database settings to `config.py`:
  - [x] `vector_db_type` - ChromaDB/Weaviate/Qdrant selection
  - [x] `vector_db_path` - Storage location for embeddings
  - [x] `vector_dimensions` - Embedding size (384/768/1024)
  - [x] `similarity_metric` - cosine/euclidean/dot product
- [x] Add vector CLI arguments:
  - [x] `--embed-verses` - Generate verse embeddings
  - [x] `--embed-words` - Generate word embeddings
  - [x] `--embed-all` - Generate all embeddings
  - [x] `--force-reembed` - Force regenerate embeddings
  - [x] `--embedding-batch-size` - Batch size for generation
- [x] Update environment variables for vector settings

### Vector Database Setup
- [x] Research and select vector database (ChromaDB/Weaviate/Qdrant)
- [x] Create `abba/embeddings/chroma_manager.py`
- [x] Design vector schema for embeddings
- [x] Implement vector database initialization

### Embedding Model Configuration
- [x] Add embedding settings to `config.py`:
  - [x] `embedding_library` - "sentence-transformers" or "huggingface"
  - [x] `embedding_model_english` - Model for English text
  - [x] `embedding_model_multilingual` - Model for Hebrew/Greek/Aramaic
  - [x] `embedding_context_mode` - "basic" or "enhanced" (with morphology)
  - [x] `embedding_cache_dir` - Local model storage path
- [x] Add embedding CLI arguments (integrated into main flow)
- [x] Create model download verification
- [x] Implement language detection for model selection

### Ollama Configuration (Semantic Analysis)
- [x] Add Ollama settings to `config.py`:
  - [x] `ollama_host` - Ollama API endpoint (default: http://localhost:11434)
  - [x] `ollama_semantic_models` - List of models for analysis
  - [x] `ollama_consensus_threshold` - Agreement threshold for multi-model
  - [x] `ollama_timeout` - API timeout in seconds
  - [x] `ollama_batch_size` - Batch size for processing
- [x] Add Ollama CLI arguments:
  - [x] `--ollama-host` - Override Ollama endpoint
  - [x] `--ollama-models` - Comma-separated model list
  - [x] `--ollama-consensus` - Set consensus threshold
- [x] Create Ollama connection validator
- [x] Add `.env` variables:
  ```
  ABBA_OLLAMA_HOST=http://localhost:11434
  ABBA_OLLAMA_SEMANTIC_MODELS=llama4:scout,command-r-plus:latest
  ABBA_OLLAMA_CONSENSUS_THRESHOLD=0.7
  ```

### Embedding Model Integration
- [x] Create `abba/embeddings/` directory
- [x] Implement `model_manager.py`:
  - [x] Model downloading and caching
  - [x] Language detection for model selection
  - [x] Batch processing with progress tracking
- [x] Create `context_builder.py` for context enrichment
- [x] Add model warmup and validation

### Enhanced Embedding Generation
- [x] Implement contextual embedding enhancement:
  ```python
  # Enhanced context for richer embeddings
  {
    "verse_ref": "Genesis 1:1",
    "english": "In the beginning God created...",
    "hebrew": "בְּרֵאשִׁית בָּרָא אֱלֹהִים",
    "transliteration": "bereshit bara elohim",
    "morphology": "Prep+Noun Verb Noun",
    "strongs": "H7225 H1254 H430"
  }
  ```
- [x] Create language-specific pipelines:
  - [x] English verses → E5-large-v2
  - [x] Hebrew/Greek/Aramaic → Multilingual-E5-base
  - [x] Mixed language verses → Both models with alignment

### Multi-Model Semantic Analysis (Ollama)
- [x] Create `abba/semantic/` directory
- [x] Implement `ollama_analyzer.py` for concept extraction
- [x] Build consensus scoring system:
  - [x] Collect responses from multiple models
  - [x] Calculate agreement scores
  - [x] Weight by model performance/size
- [x] Add semantic analysis caching

### Original Language Embedding Generation
- [ ] Remove translation-specific verse embeddings
- [ ] Implement original language verse embeddings:
  - [ ] Use Hebrew/Greek text from `stepbible_verses` table
  - [ ] Include morphology and Strong's numbers in context
  - [ ] Generate single embedding per canonical verse
  - [ ] Map embeddings to all translations via verse reference
- [ ] Verify embedding deduplication (31K verses, not 13M)
- [ ] Update embedding validator for new structure

### Word-level Embeddings (Original Language)
- [x] Extract unique words with linguistic context:
  - [x] Include lexicon definitions
  - [x] Add morphological variations
  - [x] Preserve semantic domain information
- [x] Generate context-aware word embeddings
- [x] Create multilingual word similarity index

## Phase 3: Semantic Search Implementation

### Unified Search API
- [ ] Extend `search.py` with semantic methods:
  - [ ] `search_similar_verses()` - semantic verse search
  - [ ] `search_related_words()` - semantic word search
  - [ ] `hybrid_search()` - combined exact + semantic
- [ ] Implement result ranking algorithms
- [ ] Add search result explanations

### Search Configuration
- [ ] Add search settings to `config.py`:
  - [ ] `max_results` - Default result limit
  - [ ] `similarity_threshold` - Minimum similarity score
  - [ ] `enable_query_expansion` - Auto-expand search terms
  - [ ] `search_cache_size` - Result cache size
- [ ] Add search CLI arguments:
  - [ ] `--max-results` - Override result limit
  - [ ] `--similarity-threshold` - Set minimum score
  - [ ] `--exact-only` - Disable semantic search

### Search Optimization
- [ ] Create search query parser
- [ ] Implement query expansion for better results
- [ ] Add search filters (book, testament, language)
- [ ] Optimize vector similarity calculations
- [ ] Add search result caching

## Phase 4: Concept Mapping System

### User-Defined Concept Configuration
- [x] Add concept settings to `config.py`:
  - [x] `concepts_file` - Path to user-defined concepts JSON/YAML
  - [x] `concept_validation_model` - Ollama model for validation
  - [x] `concept_validation_batch_size` - Verses per validation batch
  - [x] `concept_validation_cache` - Cache validation results
- [x] Add concept CLI arguments:
  - [x] `--concepts-file` - Override concepts file path
  - [x] `--validate-concepts` - Run LLM validation on concepts
  - [x] `--concept-report` - Generate validation report

### Concept Definition Structure
- [x] Create `concepts.yaml` template with structure:
  ```yaml
  concepts:
    - name: "sexual_sin"
      description: "Sexual immorality including adultery, fornication..."
      hebrew_terms: ["זנה", "נאף", "ערוה"]
      greek_terms: ["πορνεία", "μοιχεία", "ἀσέλγεια"]
      strongs_numbers: ["H2181", "H5003", "G4202", "G3430"]
  ```
- [x] Implement concept loader and validator
- [x] Create concept management utilities

### LLM-Enhanced Concept Validation Pipeline
- [x] Implement traditional mapping phase:
  - [x] Find verses by Strong's numbers
  - [x] Find verses by Hebrew/Greek terms
  - [x] Create initial verse sets per concept
- [x] Implement LLM validation phase:
  - [x] Validate each initial verse for relevance
  - [x] Remove false positives with explanations
  - [x] Log validation decisions for review
- [x] Implement comprehensive scanning phase:
  - [x] Check ALL verses not in initial set
  - [x] Find missing relevant verses
  - [x] Use batching with progress tracking
  - [x] Estimate: ~1.3 hours per concept with RTX 4090

### Concept-to-Verse Mapping Storage
- [x] Create database tables:
  ```sql
  CREATE TABLE concept_definitions (
    concept_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    hebrew_terms TEXT,
    greek_terms TEXT,
    strongs_numbers TEXT
  );
  
  CREATE TABLE concept_verse_mappings (
    concept_id TEXT,
    verse_id TEXT,
    validation_method TEXT, -- 'traditional', 'llm_validated', 'llm_discovered'
    confidence_score REAL,
    validation_reason TEXT,
    PRIMARY KEY (concept_id, verse_id)
  );
  ```
- [x] Store validation metadata and reasons
- [x] Enable traceability of mapping decisions

## Phase 5: Enhanced Features

### Advanced Analysis
- [ ] Implement cross-reference analysis
- [ ] Add translation comparison features
- [ ] Create word frequency analysis
- [ ] Add grammatical pattern detection

### Performance Configuration
- [ ] Add performance settings to `config.py`:
  - [ ] `connection_pool_size` - Database connection pool
  - [ ] `query_timeout` - Maximum query execution time
  - [ ] `parallel_workers` - Number of parallel processors
  - [ ] `memory_limit` - Maximum memory usage
  - [ ] `enable_profiling` - Performance profiling toggle
- [ ] Add performance CLI arguments:
  - [ ] `--workers` - Set parallel worker count
  - [ ] `--profile` - Enable performance profiling
  - [ ] `--benchmark` - Run performance tests

### Performance Optimization
- [ ] Profile database queries
- [ ] Optimize indexes for common queries
- [ ] Implement connection pooling
- [ ] Add query result pagination
- [ ] Create performance benchmarks

### User Interface Enhancements
- [ ] Update CLI with new search capabilities
- [ ] Add interactive mode for exploration
- [ ] Create example scripts for common tasks
- [ ] Add export functionality for results

## Phase 6: Testing and Documentation

### Testing Infrastructure
- [ ] Create unit tests for database operations
- [ ] Add integration tests for search functions
- [ ] Test embedding generation accuracy
- [ ] Validate concept mappings
- [ ] Performance testing for large queries

### Documentation
- [ ] Update API documentation
- [ ] Create user guide for semantic search
- [ ] Document concept taxonomy
- [ ] Add code examples for common use cases
- [ ] Create troubleshooting guide

### Deployment Preparation
- [ ] Create installation scripts
- [ ] Add database migration support
- [ ] Create backup/restore functionality
- [ ] Add configuration validation
- [ ] Prepare distribution package

## Phase 7: Future Enhancements (Post-MVP)

### Extended Capabilities
- [ ] Multi-language semantic search
- [ ] Real-time text analysis API
- [ ] Concept discovery from user queries
- [ ] Integration with commentary databases
- [ ] Visualization tools for semantic relationships
- [ ] Mobile app API endpoints
- [ ] Collaborative concept editing
- [ ] Machine learning for concept refinement