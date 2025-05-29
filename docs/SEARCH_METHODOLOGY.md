# ABBA Search Methodology

This document describes how ABBA enables sophisticated topical and conceptual searching across biblical texts, bridging ancient languages with modern queries.

## Overview

ABBA's search system addresses a fundamental challenge: users think in modern concepts while the Bible uses ancient languages and categories. Our methodology enables intuitive searching while maintaining linguistic precision.

## Search Types

### 1. Direct Word Search

Simple searches for specific words in translations.

**Example**: User searches for "love" in English
```json
{
  "query": "love",
  "type": "direct_word",
  "search_in": ["ESV", "NIV", "NASB"],
  "results": [
    {
      "verse": "JOH.3.16",
      "text": "For God so loved the world...",
      "translation": "ESV"
    }
  ]
}
```

### 2. Original Language Search

Searches using Strong's numbers or original language terms.

**Example**: Search for Hebrew "chesed" (H2617)
```json
{
  "query": "H2617",
  "type": "strong_number",
  "results": [
    {
      "verse": "PSA.136.1",
      "hebrew": "חֶסֶד",
      "gloss": "steadfast love",
      "translation": "His steadfast love endures forever"
    }
  ]
}
```

### 3. Semantic Concept Search

Maps modern concepts to multiple original language words.

**Example**: "Love" concept mapping
```json
{
  "concept": "love",
  "mappings": {
    "hebrew": {
      "H157": {"gloss": "love", "usage": "general/romantic"},
      "H2617": {"gloss": "steadfast love", "usage": "covenant loyalty"},
      "H7355": {"gloss": "compassion", "usage": "tender love"},
      "H1730": {"gloss": "beloved", "usage": "romantic/intimate"}
    },
    "greek": {
      "G25": {"gloss": "agapaō", "usage": "divine/selfless love"},
      "G5368": {"gloss": "phileō", "usage": "friendship/affection"},
      "G5360": {"gloss": "philadelphia", "usage": "brotherly love"}
    }
  }
}
```

### 4. Lexical Domain Search

Searches based on semantic domains from lexicons.

**Example**: Search for EMOTION/JOY domain
```json
{
  "domain": "EMOTION",
  "subdomain": "JOY",
  "results": [
    {
      "verse": "PSA.16.11",
      "matching_words": [
        {"strong": "H8057", "word": "simchah", "gloss": "joy"},
        {"strong": "H5273", "word": "na'im", "gloss": "pleasant"}
      ]
    }
  ]
}
```

### 5. Collocation Search

Finds statistically significant word combinations.

**Example**: "fear" + "LORD"
```json
{
  "collocation": ["fear", "LORD"],
  "frequency": 89,
  "confidence": 0.98,
  "verses": ["PRO.9.10", "PSA.111.10", "...]
}
```

## Search Process Flow

### Step 1: Query Analysis

```python
def analyze_query(user_input):
    # Detect query type
    if is_strong_number(user_input):
        return {"type": "strong", "number": user_input}
    
    # Check concept mappings
    if concept := semantic_concepts.get(user_input.lower()):
        return {"type": "concept", "mapping": concept}
    
    # Default to word search
    return {"type": "word", "term": user_input}
```

### Step 2: Query Expansion

For conceptual searches, expand to related terms:

```python
def expand_concept_query(concept):
    expanded = {
        "primary_words": concept.core_vocabulary,
        "related_domains": lexicon.get_related_domains(concept),
        "collocations": find_common_collocations(concept),
        "synonyms": get_synonyms(concept)
    }
    return expanded
```

### Step 3: Ranking Results

Results are ranked by multiple factors:

1. **Exact matches** (weight: 1.0)
2. **Lexical domain matches** (weight: 0.8)
3. **Collocation presence** (weight: 0.7)
4. **Semantic field overlap** (weight: 0.6)
5. **Statistical co-occurrence** (weight: 0.5)

## Bridging Modern and Ancient Concepts

### Challenge: Anachronistic Searches

Users search for modern concepts not directly present in ancient texts.

### Solution: Multi-Layer Mapping

#### Layer 1: Direct Vocabulary
- Modern term → Ancient equivalent
- "Anxiety" → "troubled heart/spirit"

#### Layer 2: Phenomenological Matching
- Modern experience → Ancient description
- "Depression" → Psalms of lament

#### Layer 3: Semantic Domain Clustering
- Modern category → Multiple ancient concepts
- "Mental health" → EMOTION + DISTRESS + SOUL domains

### Example: "Social Justice" Search

```json
{
  "modern_query": "social justice",
  "search_strategy": {
    "vocabulary_mapping": {
      "justice": ["H4941", "H6664", "G1342"],
      "righteousness": ["H6666", "G1343"],
      "judgment": ["H8199", "G2920"]
    },
    "thematic_passages": {
      "care_for_poor": ["LEV.19.15", "DEU.15.7-11"],
      "widow_orphan_stranger": ["EXO.22.21-23", "JAM.1.27"],
      "jubilee_laws": ["LEV.25"],
      "prophetic_critique": ["ISA.58.6-7", "MIC.6.8"]
    },
    "domain_search": {
      "domains": ["JUSTICE", "RIGHTEOUSNESS", "OPPRESSION", "POVERTY"],
      "weight": 0.8
    }
  }
}
```

## Implementation Strategies

### 1. Pre-computed Indices

Build search indices during compilation:

```python
# Word index
word_index = {
    "love": ["GEN.29.20", "DEU.6.5", ...],
    "H2617": ["PSA.136.1", "LAM.3.22", ...]
}

# Domain index
domain_index = {
    "EMOTION/JOY": ["PSA.16.11", "PHI.4.4", ...],
    "TIME/BEGINNING": ["GEN.1.1", "JOH.1.1", ...]
}

# Collocation index
collocation_index = {
    ("fear", "LORD"): ["PRO.9.10", "PSA.111.10", ...],
    ("love", "God"): ["DEU.6.5", "MAT.22.37", ...]
}
```

### 2. Search Refinement UI

Guide users to more specific searches:

```
User: "love"
System: "What aspect of love interests you?"
        
[ ] God's love for humanity (chesed, agape)
[ ] Human love for God (worship, devotion)  
[ ] Romantic love (eros, intimate affection)
[ ] Family love (storge, natural affection)
[ ] Friendship (philia, companionship)
[ ] Love as command (ethical obligation)
```

### 3. Result Presentation

Show why each result matches:

```json
{
  "verse": "1CO.13.4",
  "text": "Love is patient and kind...",
  "match_reasons": [
    {"type": "direct_word", "word": "love", "count": 9},
    {"type": "domain", "domain": "VIRTUE/LOVE", "confidence": 1.0},
    {"type": "keyword", "tf_idf": 0.89}
  ],
  "relevance_score": 0.95
}
```

## Search Optimization

### Caching Strategy

1. **Popular concept mappings** - Pre-compute and cache
2. **Frequent queries** - Store results
3. **Domain traversals** - Cache lexical hierarchies

### Performance Considerations

1. **Use inverted indices** for O(1) lookups
2. **Parallel search** across different indices
3. **Early termination** for large result sets
4. **Lazy loading** of verse content

## Advanced Search Features

### 1. Proximity Search

Find words within N verses of each other:
```
"kingdom" NEAR/5 "heaven"
```

### 2. Morphological Search

Search by grammatical features:
```
verb:imperative AND subject:2nd-person
```

### 3. Discourse Search

Find specific discourse patterns:
```
discourse:question AND speaker:Jesus
```

### 4. Chronological Search

Limit by time period:
```
"temple" AND period:second-temple
```

## Extensibility

### Custom Concept Mappings

Users can define their own concept mappings:

```json
{
  "concept": "environmental_stewardship",
  "user_defined": true,
  "mappings": {
    "vocabulary": ["H8104", "H5647"],  // shamar, abad
    "passages": ["GEN.2.15", "PSA.24.1"],
    "domains": ["CREATION", "RESPONSIBILITY"]
  }
}
```

### Search Plugins

Allow specialized search algorithms:

```python
class TheologicalInferenceSearch(SearchPlugin):
    def search(self, concept):
        # Implement theological reasoning
        # Return relevant passages
        pass
```

## Evaluation Metrics

### Search Quality

1. **Precision**: Relevant results / Total results
2. **Recall**: Found relevant / All relevant
3. **F1 Score**: Harmonic mean of precision and recall

### User Satisfaction

1. **Click-through rate** on results
2. **Search refinement** frequency
3. **Time to successful result**

## Future Enhancements

1. **Machine Learning** for query understanding
2. **Personalized ranking** based on user history
3. **Multilingual concept mapping**
4. **Voice search** with natural language processing
5. **Contextual search** based on study topics