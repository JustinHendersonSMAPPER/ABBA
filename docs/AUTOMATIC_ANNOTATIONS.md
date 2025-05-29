# ABBA Automatic Annotation Generation

This document describes how ABBA automatically generates annotations with extreme confidence by focusing on objectively computable linguistic and statistical features.

## Overview

ABBA's automatic annotation system generates data that can be computed with high confidence (0.95-1.0) from:
- Morphologically tagged texts
- Established lexicons
- Statistical analysis
- Syntactic patterns
- Objective text comparison

## Annotation Categories

### 1. Lexical Domain Mapping

Maps each word to semantic domains using established lexicons.

**Sources**:
- Semantic Dictionary of Biblical Hebrew (SDBH)
- Louw-Nida Greek-English Lexicon
- Traditional lexicons (BDB, HALOT, BDAG)

**Algorithm**:
```python
def map_lexical_domains(strong_number):
    domain = lookup_in_lexicon(strong_number)
    return {
        "domain": domain.primary,
        "subdomain": domain.secondary,
        "confidence": 1.0  # Direct lexicon lookup
    }
```

**Example Output**:
```json
{
  "strong_number": "H7225",
  "domain": "TIME",
  "subdomain": "BEGINNING",
  "confidence": 1.0,
  "source": "SDBH"
}
```

### 2. Statistical Text Analysis

Computes word frequencies and significance across the biblical corpus.

#### Metrics Calculated

**Hapax Legomena**
- Words appearing only once in the entire Bible
- Useful for identifying unique vocabulary
- Confidence: 1.0 (simple counting)

**TF-IDF (Term Frequency-Inverse Document Frequency)**
- Measures word importance in a passage relative to the corpus
- High TF-IDF indicates keywords distinctive to that passage
- Range: 0.0-1.0

**Collocations**
- Statistically significant word pairs/phrases
- Uses PMI (Pointwise Mutual Information)
- Threshold: PMI > 3.0 for significance

**Algorithm**:
```python
def compute_text_statistics(verse_tokens, corpus):
    # Calculate TF-IDF
    tf = count_frequency(verse_tokens) / len(verse_tokens)
    df = count_documents_containing(token, corpus)
    idf = log(total_documents / df)
    tf_idf = tf * idf
    
    # Find collocations
    collocations = []
    for word1, word2 in bigrams(verse_tokens):
        pmi = calculate_pmi(word1, word2, corpus)
        if pmi > 3.0:
            collocations.append((word1, word2, pmi))
    
    return {
        "keywords": filter(lambda w: tf_idf[w] > 0.7, verse_tokens),
        "collocations": collocations,
        "hapax": filter(lambda w: corpus_freq[w] == 1, verse_tokens)
    }
```

### 3. Syntactic Pattern Recognition

Identifies grammatical structures and discourse markers from morphological tagging.

#### Features Detected

**Word Order Patterns**
- VSO (Verb-Subject-Object) - Hebrew narrative
- SVO (Subject-Verb-Object) - Greek standard
- Fronted elements for emphasis
- Confidence: 1.0 with tagged text

**Clause Analysis**
- Main vs. subordinate clauses
- Relative clauses
- Conditional statements
- Purpose/result clauses

**Discourse Markers**
- Vav-consecutive (Hebrew narrative sequence)
- Temporal markers (בְּרֵאשִׁית, ἐν ἀρχῇ)
- Logical connectors (therefore, because, but)
- Transition markers

**Verbal Patterns**
- Stem/voice (Qal, Niphal, Piel, etc.)
- Aspect (perfect, imperfect, participle)
- Special constructions (divine passive, prophetic perfect)

**Implementation**:
```python
def analyze_syntax(verse_morphology):
    patterns = {
        "word_order": detect_word_order(verse_morphology),
        "clause_type": classify_clause(verse_morphology),
        "discourse_markers": find_discourse_markers(verse_morphology),
        "verbal_analysis": analyze_verbs(verse_morphology)
    }
    
    # All based on morphological tags, confidence = 1.0
    return patterns
```

### 4. Intertextual Detection

Finds objective textual relationships between passages.

#### Types of Relationships

**Direct Quotations**
- Exact string matches > 5 words
- Accounts for minor variations (articles, conjunctions)
- Confidence: 1.0 for exact matches

**Allusions**
- Shared rare vocabulary (words appearing < 10 times)
- Same phrase structure with different words
- Confidence: 0.9-0.95 based on similarity

**Parallel Structures**
- Same syntactic patterns
- Often in poetry (parallelism)
- Detected via syntax tree comparison

**Statistical Co-occurrence**
- Passages frequently cited together in ancient sources
- Based on manuscript evidence and early commentaries

**Algorithm**:
```python
def detect_intertextuality(verse, corpus):
    results = {
        "quotations": [],
        "allusions": [],
        "shared_vocabulary": [],
        "parallel_structures": []
    }
    
    # Find exact quotations
    for other_verse in corpus:
        common_substring = longest_common_substring(verse.text, other_verse.text)
        if len(common_substring.split()) >= 5:
            results["quotations"].append({
                "target": other_verse.id,
                "text": common_substring,
                "confidence": 1.0
            })
    
    # Find shared rare words
    rare_words = [w for w in verse.words if corpus_frequency[w] < 10]
    for other_verse in corpus:
        shared_rare = set(rare_words) & set(other_verse.words)
        if shared_rare:
            results["shared_vocabulary"].append({
                "target": other_verse.id,
                "words": list(shared_rare),
                "confidence": 0.95
            })
    
    return results
```

### 5. Semantic Field Construction

Maps words to concepts based purely on lexical data.

#### Process

1. **Extract Strong's Numbers**
   - From morphologically tagged text
   - Each word → lemma → Strong's number

2. **Map to Lexical Domains**
   - Using SDBH for Hebrew
   - Using Louw-Nida for Greek
   - Hierarchical domain structure

3. **Group by Semantic Field**
   - Time words together
   - Emotion words together
   - Movement words together

4. **No Interpretation**
   - Only what's explicit in lexicons
   - No theological inference
   - No cultural commentary

**Example**:
```python
def build_semantic_fields(verse_morphology):
    fields = defaultdict(list)
    
    for token in verse_morphology:
        strong = token.strong_number
        domain = lexicon.get_domain(strong)
        
        fields[domain.primary].append({
            "word": strong,
            "subdomain": domain.secondary,
            "gloss": token.gloss
        })
    
    return {
        "explicit_concepts": [
            {
                "concept": field_name,
                "words": words,
                "source": "lexicon",
                "confidence": 1.0
            }
            for field_name, words in fields.items()
        ]
    }
```

### 6. Text Critical Analysis

Documents manuscript variations and ancient translations.

#### Components

**Manuscript Variants**
- Based on critical apparatus (NA28, BHS)
- Weighted by manuscript evidence
- Age, geographic distribution, text families

**Ancient Versions**
- LXX (Septuagint) - Greek OT
- Targums - Aramaic paraphrases
- Vulgate - Latin
- Peshitta - Syriac

**Translation Decisions**
- Ambiguous grammatical constructions
- Multiple valid renderings
- Documented scholarly debates

## Quality Assurance

### Confidence Scoring

All automatic annotations include confidence scores:
- **1.0**: Direct data (lexicon lookup, morphology)
- **0.95-0.99**: Statistical significance (collocations, rare words)
- **0.90-0.94**: Pattern matching (syntactic parallels)
- **0.85-0.89**: Derived data (semantic clustering)

### Validation

1. **Cross-reference with multiple sources**
   - Multiple lexicons must agree
   - Statistical significance thresholds
   - Peer-reviewed linguistic data

2. **Reproducibility**
   - Same input → same output
   - Deterministic algorithms
   - Version-controlled reference data

3. **Transparency**
   - Source attribution for all data
   - Algorithm documentation
   - No "black box" decisions

## What Is NOT Automatically Generated

The following require human annotation and are excluded from automatic generation:

1. **Theological Interpretations**
   - Doctrinal significance
   - Systematic theology connections
   - Denominational perspectives

2. **Cultural Commentary**
   - Historical background
   - Social context
   - Archaeological insights

3. **Modern Applications**
   - Contemporary relevance
   - Pastoral guidance
   - Ethical implications

4. **Phenomenological Parallels**
   - Modern experience mapping
   - Psychological insights
   - Sociological analysis

These can be added through manual annotation systems that extend the base format.

## Implementation Notes

### Performance Considerations

- Pre-compute all statistics during build phase
- Cache lexicon lookups
- Use efficient string algorithms for text matching
- Parallelize where possible

### Data Sources Required

1. **Morphologically tagged texts**
   - Hebrew: Westminster Hebrew Morphology
   - Greek: MorphGNT, SBLGNT

2. **Lexicons**
   - SDBH (Semantic Dictionary of Biblical Hebrew)
   - Louw-Nida Greek Lexicon
   - Traditional: BDB, HALOT, BDAG

3. **Critical Apparatus**
   - NA28 for NT
   - BHS for OT
   - Ancient version alignments

### Output Format

All automatic annotations follow the structure defined in [CANONICAL_FORMAT.md](./CANONICAL_FORMAT.md) under the `annotations` object.