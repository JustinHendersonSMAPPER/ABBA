# ABBA Concept Mapping Guide

## Overview

ABBA's concept mapping system uses Large Language Models (LLMs) to intelligently map theological concepts to biblical verses. This provides more accurate results than traditional keyword or Strong's number searches alone.

## How It Works

### Three-Phase Process

1. **Traditional Mapping**: Find verses using Strong's numbers, Hebrew/Greek terms, and keywords
2. **LLM Validation**: Analyze each traditional match to remove false positives
3. **Comprehensive Scanning**: Scan all 29,126 verses to discover additional relevant passages

### Benefits

- **Higher Accuracy**: LLM validation removes false positives from traditional searches
- **Completeness**: Comprehensive scanning finds verses missed by traditional methods
- **Traceability**: Every inclusion/exclusion decision is logged with detailed reasoning
- **Theological Accuracy**: Based on user-defined concepts, not LLM-generated ones

## Creating Concepts

### Basic Structure

Create `abba/concepts.yaml` with this structure:

```yaml
concepts:
  - name: "concept_name"           # Unique identifier (use underscores)
    description: >                 # Detailed theological description
      Full description of the concept including theological context,
      scope, and any important distinctions or nuances.
    hebrew_terms:                  # Optional: Hebrew words (with comments)
      - "Hebrew word 1"            # Brief gloss in comment
      - "Hebrew word 2"
    greek_terms:                   # Optional: Greek words (with comments)  
      - "Greek word 1"             # Brief gloss in comment
      - "Greek word 2"
    strongs_numbers:               # Optional: Strong's concordance numbers
      - "H1234"                    # Hebrew numbers start with H
      - "G5678"                    # Greek numbers start with G
    keywords:                      # Optional: English keywords
      - "keyword1"
      - "keyword2"
      - "phrase with multiple words"
```

### Example: Divine Love

```yaml
concepts:
  - name: "divine_love"
    description: >
      God's love, mercy, compassion, and loving-kindness toward humanity,
      including covenant love (hesed) and unconditional love (agape).
      This encompasses both God's general love for creation and His
      specific covenant love for His people.
    hebrew_terms:
      - "אהב"      # ahav - to love
      - "חסד"      # hesed - loving-kindness, covenant love
      - "רחם"      # racham - to have compassion, show mercy  
      - "חנן"      # chanan - to be gracious, show favor
    greek_terms:
      - "ἀγάπη"    # agape - divine/unconditional love
      - "ἀγαπάω"   # agapao - to love (verb form)
      - "ἔλεος"    # eleos - mercy, compassion
      - "φιλανθρωπία" # philanthropia - love of mankind
    strongs_numbers:
      - "H157"     # ahav
      - "H2617"    # hesed  
      - "H7355"    # racham
      - "H2603"    # chanan
      - "G26"      # agape
      - "G25"      # agapao
      - "G1656"    # eleos
      - "G5363"    # philanthropia
    keywords:
      - "love"
      - "mercy"
      - "compassion"
      - "loving-kindness"
      - "gracious"
      - "tender mercy"
      - "steadfast love"
```

### Advanced Example: Complex Concept

```yaml
concepts:
  - name: "covenant_faithfulness"
    description: >
      The mutual faithfulness expected in biblical covenants, including
      both God's faithfulness to His promises and the required human
      response of loyalty and obedience. This includes both the Hebrew
      concept of hesed (covenant love/loyalty) and the broader theme
      of covenant fidelity throughout Scripture.
    hebrew_terms:
      - "ברית"     # berit - covenant, alliance
      - "חסד"      # hesed - covenant love, loyalty
      - "אמונה"    # emunah - faithfulness, reliability
      - "אמן"      # aman - to be faithful, trustworthy
    greek_terms:
      - "διαθήκη"  # diatheke - covenant, testament
      - "πίστις"   # pistis - faith, faithfulness
      - "πιστός"   # pistos - faithful, trustworthy
    strongs_numbers:
      - "H1285"    # berit
      - "H2617"    # hesed
      - "H530"     # emunah
      - "H539"     # aman
      - "G1242"    # diatheke
      - "G4102"    # pistis
      - "G4103"    # pistos
    keywords:
      - "covenant"
      - "faithfulness"
      - "loyal"
      - "steadfast"
      - "reliable"
      - "trustworthy"
      - "covenant love"
      - "treaty"
```

## Configuration Options

Add configuration to your concepts file:

```yaml
# After defining concepts, add configuration
config:
  # LLM validation settings
  validation:
    enabled: true              # Enable LLM validation
    confidence_threshold: 0.6  # Minimum confidence for inclusion
    consensus_threshold: 0.7   # Minimum agreement between models
    
  # Traditional mapping settings  
  traditional_mapping:
    strongs_weight: 1.0        # Weight for Strong's number matches
    keyword_weight: 0.8        # Weight for keyword matches
    term_weight: 0.9           # Weight for Hebrew/Greek term matches
    
  # Comprehensive scanning settings
  comprehensive_scan:
    enabled: true              # Enable full verse scanning
    relevance_threshold: 0.7   # Higher threshold for discovered verses
    max_verses_per_concept: 500 # Limit results per concept
    
  # Output settings
  output:
    include_analysis_details: true  # Include LLM reasoning in output
    include_false_positives: true   # Track rejected matches
    generate_reports: true          # Generate detailed reports
```

## Using Concept Mapping

### Basic Commands

```bash
# Validate concept definitions
python abba/main.py --validate-concepts

# Map all concepts to verses
python abba/main.py --map-concepts

# Generate detailed report
python abba/main.py --map-concepts --concept-report
```

### Advanced Options

```bash
# Use custom concepts file
python abba/main.py --concepts-file my_concepts.yaml --map-concepts

# Customize LLM settings
python abba/main.py --ollama-models llama3 --ollama-consensus 0.8 --map-concepts

# Use different Ollama server
python abba/main.py --ollama-host http://localhost:11434 --map-concepts
```

### Environment Variables

Set in `.env` file:

```bash
# Ollama Configuration
ABBA_OLLAMA_HOST=http://localhost:11434
ABBA_OLLAMA_SEMANTIC_MODELS=llama3
ABBA_OLLAMA_CONSENSUS_THRESHOLD=0.7
ABBA_OLLAMA_TIMEOUT=30
ABBA_OLLAMA_BATCH_SIZE=100

# Concept Configuration  
ABBA_CONCEPTS_FILE=abba/concepts.yaml
ABBA_CONCEPT_VALIDATION_ENABLED=true
ABBA_CONCEPT_VALIDATION_BATCH_SIZE=100
```

## Understanding Results

### Mapping Categories

Each verse is categorized as:

1. **Traditional Match**: Found by Strong's numbers, terms, or keywords
2. **LLM Validated**: Traditional match confirmed by LLM analysis
3. **False Positive**: Traditional match rejected by LLM
4. **LLM Discovered**: Additional verse found by comprehensive scanning

### Database Schema

Results are stored in these tables:

```sql
-- Concept definitions
CREATE TABLE concept_definitions (
    concept_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    hebrew_terms TEXT,
    greek_terms TEXT,
    strongs_numbers TEXT,
    keywords TEXT
);

-- Verse mappings with metadata
CREATE TABLE concept_verse_mappings (
    concept_id TEXT,
    verse_id TEXT,              -- Format: "book_id:chapter:verse"
    validation_method TEXT,     -- 'traditional', 'llm_validated', 'llm_discovered'
    relevance_score REAL,       -- 0.0-1.0 relevance score from LLM
    confidence_score REAL,      -- 0.0-1.0 confidence score from LLM
    validation_reason TEXT,     -- Detailed reasoning from LLM
    PRIMARY KEY (concept_id, verse_id)
);
```

### Querying Results

```sql
-- Get all verses for a concept
SELECT vm.verse_id, vm.relevance_score, vm.validation_reason
FROM concept_verse_mappings vm
WHERE vm.concept_id = 'divine_love'
ORDER BY vm.relevance_score DESC;

-- Count verses by validation method
SELECT validation_method, COUNT(*) as count
FROM concept_verse_mappings  
WHERE concept_id = 'divine_love'
GROUP BY validation_method;

-- Get high-confidence discovered verses
SELECT verse_id, relevance_score, validation_reason
FROM concept_verse_mappings
WHERE concept_id = 'divine_love' 
  AND validation_method = 'llm_discovered'
  AND relevance_score > 0.8
ORDER BY relevance_score DESC;
```

## Performance Expectations

### Time Estimates (with llama3)

- **Concept validation**: ~2 seconds per verse
- **Single concept mapping**: 30-60 minutes depending on traditional matches
- **All 5 default concepts**: 2-5 hours total
- **Comprehensive scanning**: ~1 hour per concept (scans all 29,126 verses)

### Optimizing Performance

1. **Use fewer models**: Single model is faster than consensus
2. **Adjust thresholds**: Higher thresholds mean fewer verses to analyze
3. **Limit scope**: Define concepts more narrowly to reduce matches
4. **Batch processing**: Built-in batching optimizes LLM calls

### Resource Usage

- **Memory**: ~2GB for embeddings + model memory
- **Storage**: ~500MB for full mapping results
- **Network**: Ollama API calls (local by default)

## Best Practices

### Concept Design

1. **Be Specific**: Narrow, well-defined concepts work better than broad ones
2. **Include Context**: Provide rich theological descriptions
3. **Use Original Languages**: Hebrew/Greek terms improve accuracy
4. **Add Strong's Numbers**: Most reliable traditional mapping method
5. **Test Iteratively**: Start with one concept, refine, then expand

### Validation Workflow

```bash
# 1. Create/edit concepts
vim abba/concepts.yaml

# 2. Validate syntax and definitions  
python abba/main.py --validate-concepts

# 3. Test with one concept first
# (Edit concepts.yaml to include only one concept for testing)
python abba/main.py --map-concepts

# 4. Review results in database
# 5. Refine concept definitions as needed
# 6. Scale up to all concepts
```

### Quality Control

- **Review False Positives**: Check why traditional matches were rejected
- **Examine Discovered Verses**: Verify LLM-found verses are actually relevant  
- **Check Edge Cases**: Look at low-confidence matches for boundary cases
- **Cross-Reference**: Compare with other biblical reference works

## Troubleshooting

### Common Issues

1. **"Ollama server not available"**
   - Start Ollama: `ollama serve`
   - Check host setting: `--ollama-host http://localhost:11434`
   - Verify model is downloaded: `ollama pull llama3`

2. **"No concepts found"**
   - Check concepts file path: `--concepts-file path/to/concepts.yaml`
   - Validate YAML syntax: Use online YAML validator
   - Ensure concepts section exists in file

3. **"Concept validation errors"**
   - Check for duplicate concept names
   - Ensure required fields (name, description) are present
   - Validate Strong's number format (H1234, G5678)

4. **Slow performance**
   - Use single model: `--ollama-models llama3`
   - Increase batch size: `ABBA_OLLAMA_BATCH_SIZE=200`
   - Reduce concept scope with more specific definitions

### Debug Mode

```bash
# Maximum verbosity
python abba/main.py --verbose --map-concepts

# Test LLM connection
python claude/scripts/test_ollama_integration.py

# Validate setup
python claude/scripts/test_concept_mapping.py
```

## Advanced Usage

### Multiple Models for Consensus

```bash
# Use multiple models for higher accuracy
python abba/main.py --ollama-models "llama3,mixtral" --map-concepts
```

### Custom Thresholds

```yaml
config:
  validation:
    confidence_threshold: 0.7    # Higher = more strict
    consensus_threshold: 0.8     # Higher = require more agreement
  comprehensive_scan:
    relevance_threshold: 0.8     # Higher = fewer discovered verses
```

### Integration with Other Tools

The concept mapping results can be:
- Exported to JSON/CSV for analysis
- Integrated with Bible study software
- Used to generate themed verse collections
- Connected to sermon preparation tools

For advanced integration, query the `concept_verse_mappings` table directly or use the Python API to access mapping results programmatically.