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
- [x] Remove translation-specific verse embeddings
- [x] Implement original language verse embeddings:
  - [x] Use Hebrew/Greek text from `stepbible_verses` table
  - [x] Include morphology and Strong's numbers in context
  - [x] Generate single embedding per canonical verse
  - [x] Map embeddings to all translations via verse reference
- [x] Verify embedding deduplication (31K verses, not 13M)
- [x] Update embedding validator for new structure

### Word-level Embeddings (Original Language)
- [x] Extract unique words with linguistic context:
  - [x] Include lexicon definitions
  - [x] Add morphological variations
  - [x] Preserve semantic domain information
- [x] Generate context-aware word embeddings
- [x] Create multilingual word similarity index

## Phase 3: Semantic Search Implementation

### Unified Search API
- [x] Extend `search.py` with semantic methods:
  - [x] `search_similar_verses()` - semantic verse search
  - [x] `search_related_words()` - semantic word search
  - [x] `hybrid_search()` - combined exact + semantic
- [x] Implement result ranking algorithms
- [x] Add search result explanations

### Search Configuration
- [x] Add search settings to `config.py`:
  - [x] `max_results` - Default result limit
  - [x] `similarity_threshold` - Minimum similarity score
  - [x] `enable_query_expansion` - Auto-expand search terms
  - [x] `search_cache_size` - Result cache size
- [x] Add search CLI arguments:
  - [x] `--max-results` - Override result limit
  - [x] `--similarity-threshold` - Set minimum score
  - [x] `--exact-only` - Disable semantic search

### Search Optimization
- [x] Create search query parser
- [x] Implement query expansion for better results
- [x] Add search filters (book, testament, language)
- [x] Optimize vector similarity calculations
- [x] Add search result caching

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

## Phase 5: FastAPI Foundation + Enrichment Data Layer

> **Rationale:** Expert review (see `claude/EXPERT_REVIEW_SYNTHESIS.md`) identified that
> user-facing accessibility must be prioritized alongside technical completeness. The backend
> has no HTTP layer, no cultural context, no literary metadata, and no progressive disclosure.
> These are prerequisites for making scholar-level knowledge accessible to everyday readers.

### FastAPI Application Setup
- [x] Add `fastapi` and `uvicorn` dependencies via `poetry add`
- [x] Create FastAPI app factory (`abba/api/app.py`) with CORS middleware
- [x] Create Pydantic response models (`abba/api/models.py`):
  - [x] `DepthLevel` enum: basic, standard, deep, scholarly
  - [x] `VerseResponse` with depth-conditional fields
  - [x] `WordDetail`, `RichnessFlag`, `CulturalNote`, `CrossRef` models
  - [x] `TopicalResult`, `ThemeGroup`, `BookInfo`, `PassageInfo` models
- [x] Create FastAPI routes (`abba/api/routes.py`):
  - [x] `GET /api/v1/verses/{translation}/{book}/{chapter}/{verse}?depth=` — depth-aware verse
  - [x] `GET /api/v1/verses/{translation}/{book}/{chapter}?depth=` — chapter endpoint
  - [x] `GET /api/v1/compare/{book}/{chapter}/{verse}?translations=` — translation comparison
  - [x] `GET /api/v1/search/semantic?q=` — semantic search
  - [x] `GET /api/v1/search/text?q=` — full-text search
  - [x] `GET /api/v1/search/strongs/{number}` — Strong's lookup
  - [x] `GET /api/v1/lexicon/{strongs_number}` — lexicon entry
  - [x] `GET /api/v1/words/{book}/{chapter}/{verse}/{word_num}` — word detail
  - [x] `GET /api/v1/topics` — list available topics/concepts
  - [x] `GET /api/v1/topics/search?q=` — natural-language topic search
  - [x] `GET /api/v1/topics/{concept_name}` — concept detail with themed verse groups
  - [x] `GET /api/v1/books/{book_id}` — book metadata with genre/context
  - [x] `GET /api/v1/passages/{book_id}/{chapter}` — pericope boundaries
- [x] Wire existing `SearchAPI` and `AnalysisAPI` into FastAPI route handlers

### Enrichment Schema Additions (all additive — zero changes to existing tables)
- [x] `book_metadata` table — genre, author, audience, date range, literary features, reading context, canonical section per book
- [x] `passages` table — pericope definitions with title, genre, literary type, structural features, parent passage support
- [x] `literary_structures` table — chiasmus, parallelism, acrostic, inclusio annotations with element data
- [x] `cultural_context` table — scope-flexible (book → verse) with type, summary, detailed content, time period, confidence, sources, priority
- [x] `cross_references` table — source/target verse pairs with type (quotation, allusion, parallel, thematic, prophecy_fulfillment, typology, contrast)
- [x] `word_richness` table — precomputed gloss_coverage, morphology_significance, untranslatable_nuances, richness_score per word occurrence
- [x] `life_topics` table — everyday topic names/categories (emotions, relationships, struggles, life stages)
- [x] `life_topic_concepts` table — mapping life topics to existing concept definitions
- [x] `topic_study_steps` table — curated verse sequences per topic with step types (comfort, understanding, guidance, hope)
- [x] Add all tables via migration framework (extend `migrations.py`)

### Enrichment Data Population
- [x] **Book metadata curation**: Genre, author, audience, features for all 66 books (curated data → DB import)
- [x] **Cross-reference import**: Curated cross-references (quotations, allusions, parallels, typology, prophecy fulfillment)
- [x] **Meaning-richness computation**: Build-time comparison of lexicon gloss vs. definition for all entries
- [x] **Passage/pericope boundaries**: 137 curated OT+NT passage units with genre and literary type
- [x] **Initial cultural context**: Book-level introductions for 16 major books (curated; remaining books deferred)
- [x] **Life topic mappings**: Map 12 everyday topics to existing concepts with curated study steps

### Lexicon Expansion (scholarly quality improvement)
- [x] Integrate Strong's Greek Dictionary (CC0, morphgnt) — public domain alternative to Thayer's (no structured Thayer's data exists)
- [x] Integrate full BDB Hebrew Lexicon (1906, public domain) — OpenScriptures BrownDriverBriggs.xml + LexicalIndex.xml
- [x] LEH (Lust-Eynikel-Hauspie) — SKIPPED: copyrighted (Deutsche Bibelgesellschaft, 2003), not free to use
- [x] Add source attribution to all lexicon entries (which lexicon provided each definition)
- [x] Integrate Dodson Greek-English Lexicon (CC0) — compiled from Abbott-Smith, Berry, Souter, Strong

### Concept Definition Quality Review
- [x] Add temporal tags to concept definitions (OT concept / NT concept / post-biblical systematization)
- [x] Add semantic range warnings for high-frequency polysemous words (e.g., H430 elohim, H7307 ruach)
- [x] Review "Trinity" concept — flag as confessional reading, reduce false-positive surface area
- [x] Review high-frequency mapped Strong's numbers for over-matching risk (e.g., H6213 asah appears 2,627x)
- [x] Document LLM validation methodology: model versions used, theological limitations, reproducibility notes

## Phase 6: Literary and Contextual Intelligence

### Literary Genre and Structure
- [x] Literary genre indicators at book and passage level
- [x] Well-established literary structure annotations (13 curated structures):
  - [x] Chiastic structures (Flood narrative Gen 6-9, Psalm 8, Sermon on the Mount, John Prologue, Phil 2 Christ Hymn)
  - [x] Acrostic poems (Psalm 119, Lamentations, Proverbs 31:10-31)
  - [x] Hebrew parallelism (Isaiah 5 Song of the Vineyard, Hebrews 11 Hall of Faith)
  - [x] Inclusio patterns (Amos 1-2 Oracles, Revelation 4-5 Throne Room)
  - [x] NT discourse structures (Sermon on the Mount)
- [x] Genre-shift detection within books (e.g., narrative → poetry in Exodus 15, Judges 5)

### Anti-Proof-Texting Safeguards
- [x] Always return surrounding context with verse results (min: previous and next verse)
- [x] Speaker attribution for quoted speech (God, Satan, Job's friends, Pharisees, etc.)
- [x] Genre tags on all verse results (via passage_info with genre field at deep/scholarly depth)
- [x] Descriptive vs. prescriptive flag for narrative passages
- [x] Passage summary / reading context note for major sections (137 curated passages)

### Translation Insight Features
- [x] Meaning-richness indicator computation using word_richness table (richness flags at standard+ depth)
- [x] Translation divergence detection for compare endpoint
- [x] Plain-English explanations for top 500 Hebrew + top 500 Greek words where meaning is lost
- [x] Frame all indicators as "the original adds richness" — never "your Bible is wrong"

## Phase 7: Performance + Testing

### Performance Configuration
- [x] Add performance settings to `config.py`:
  - [x] `search_cache_size` - Search result LRU cache
  - [x] `search_timeout` - Maximum search execution time
  - [x] `parallel_workers` - Number of parallel processors
  - [x] `connection_pool_size` - Database connection pool
  - [x] `memory_limit` - Maximum memory usage
  - [x] `enable_profiling` - Performance profiling toggle
- [x] Add performance CLI arguments:
  - [x] `--workers` - Set parallel worker count
  - [x] `--profile` - Enable performance profiling
  - [x] `--benchmark` - Run performance tests

### Performance Optimization
- [x] Connection pooling for FastAPI concurrent requests
- [x] Precomputed verse annotation cache (materialized `verse_annotations_cache` table)
- [x] Profile and optimize database queries (slow query logging, composite indexes for range queries)
- [x] Add query result pagination
- [x] Create performance benchmarks (targets: <5ms basic, <30ms standard, <100ms deep, <200ms scholarly)

### Testing
- [x] Create unit tests for database operations (80% min, goal 95%)
- [x] Add integration tests for all FastAPI endpoints (39 tests across 8 user flow classes)
- [x] Test embedding generation accuracy
- [x] Validate concept mappings against known scholarly references
- [x] Performance testing for large queries
- [x] Test progressive depth responses at all four levels

### Documentation
- [x] Update API documentation (OpenAPI/Swagger auto-generated from FastAPI)
- [x] Create user guide for semantic search
- [x] Document concept taxonomy and life topic mappings
- [x] Add code examples for common API use cases
- [x] Create troubleshooting guide

### Deployment Preparation
- [x] Create installation scripts
- [x] Add database migration support
- [x] Create backup/restore functionality
- [x] Add configuration validation
- [x] Prepare distribution package

## Phase 8: User Experience Layer

### Guided Study Features
- [x] Reading plans / guided study paths for new Christians (6 plans, 49+ daily entries)
- [x] Passage summaries for major sections (book intros, pericope summaries)
- [x] "What do I do with this?" reflective application questions per passage
- [x] Beginner onboarding flow with "Start Here" guidance

### Interactive Features
- [x] Note-taking and verse saving/collections
- [x] Sharing functionality (passages, study notes, topic collections)
- [x] Interactive mode for exploration via CLI
- [x] Export functionality for study results (JSON, Markdown)

### Frontend Foundation
- [x] Vue.js project setup with mobile-responsive design
- [x] Clean reading pane (Level 1: just text, no clutter)
- [x] Translation Lens component (subtle richness indicators with progressive disclosure)
- [x] Context Sidebar component (collapsible, scope-aware cultural notes)
- [x] Depth Dial control (Read → Understand → Study → Analyze)
- [x] Life Topic Navigator (problem-first search entry point)
- [x] Literary Mode Indicator (ambient visual genre shifts)
- [x] Word Journey cards (expandable word study with tabs: meaning, occurrences, word family, this verse)

## Phase 9: Future Enhancements (Post-MVP)

### Extended Capabilities
- [ ] Multi-language semantic search
- [ ] MACULA treebank integration for clause-level syntax (discourse analysis)
- [ ] OpenText.org discourse annotation integration
- [ ] Louw-Nida semantic domain classification system
- [ ] STEPBible TFLSJ full LSJ Greek lexicon integration (CC BY 4.0, Tyndale House)
- [ ] Manuscript variant surfacing with explanations
- [ ] Community contribution system for cultural context
- [ ] Concept discovery from natural-language user queries
- [ ] Audio integration for listening
- [ ] Collaborative concept editing
- [ ] Machine learning for concept refinement
- [ ] Visualization tools for semantic relationships
- [ ] Mobile native app API endpoints