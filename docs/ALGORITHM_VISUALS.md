# ABBA Algorithm Visual Guide

This document provides visual representations of ABBA's core algorithms and data structures.

## 1. Text Alignment Algorithm

### Overview
The alignment system matches original language tokens (Hebrew/Greek) with modern translation words.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Alignment Algorithm Flow                          │
└───────────────────────────────────────────────────────────────────────────┘

Source (Hebrew):    בְּרֵאשִׁית    בָּרָא    אֱלֹהִים    אֵת    הַשָּׁמַיִם
                        │            │          │         │         │
                        ▼            ▼          ▼         ▼         ▼
Strong's Numbers:    H7225       H1254      H430     H853     H8064
                        │            │          │         │         │
                     ┌──┴────┬───────┴──┬───────┴─────────┴─────────┘
                     │       │          │              
Target (English):    In the beginning  created   God        the heavens

Confidence Matrix:
┌─────────────┬──────┬───────────┬─────────┬─────┬─────────┐
│   Hebrew    │  In  │    the    │beginning│ God │ created │
├─────────────┼──────┼───────────┼─────────┼─────┼─────────┤
│ בְּרֵאשִׁית │ 0.3  │    0.3    │   0.9   │ 0.1 │   0.1   │
│ בָּרָא      │ 0.1  │    0.1    │   0.1   │ 0.1 │   0.95  │
│ אֱלֹהִים    │ 0.05 │    0.05   │   0.05  │ 0.9 │   0.05  │
└─────────────┴──────┴───────────┴─────────┴─────┴─────────┘
```

### Alignment Score Calculation

```python
# Ensemble Confidence Scoring
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Final Score = w₁×S_strongs + w₂×S_stat + w₃×S_neural     │
│                                                             │
│  Where:                                                     │
│    w₁ = 0.35 (Strong's weight)                            │
│    w₂ = 0.25 (Statistical weight)                         │
│    w₃ = 0.25 (Neural weight)                              │
│    w₄ = 0.15 (Syntactic weight)                           │
│                                                             │
│  S_strongs = Direct Strong's number match (0 or 1)        │
│  S_stat = IBM Model 2 probability                         │
│  S_neural = BERT cosine similarity                        │
│  S_syntax = Pattern match score                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. Morphological Analysis Pipeline

### Hebrew Morphology Decoding

```
Input: "Ncfsa" (Morphology Code)
         │
         ▼
┌─────────────────────────────────────────────┐
│ N - Part of Speech → Noun                   │
│ c - Gender → Common                         │
│ f - Gender Detail → Feminine                │
│ s - Number → Singular                       │
│ a - State → Absolute                        │
└─────────────────────────────────────────────┘
         │
         ▼
Output: {
    "pos": "noun",
    "gender": "feminine",
    "number": "singular", 
    "state": "absolute",
    "person": null,
    "stem": null
}
```

### Greek Morphology Parsing

```
Input: "V-PAI-3S" (Morphology Code)
         │
         ▼
┌─────────────────────────────────────────────┐
│        Parse Greek Morphology Code          │
├─────────────────────────────────────────────┤
│ Position 1: V = Verb                        │
│ Position 2: - = Separator                   │
│ Position 3: P = Present Tense               │
│ Position 4: A = Active Voice                │
│ Position 5: I = Indicative Mood             │
│ Position 6: - = Separator                   │
│ Position 7: 3 = Third Person                │
│ Position 8: S = Singular Number             │
└─────────────────────────────────────────────┘
         │
         ▼
Output: {
    "pos": "verb",
    "tense": "present",
    "voice": "active",
    "mood": "indicative",
    "person": "3rd",
    "number": "singular",
    "case": null,
    "gender": null
}
```

## 3. ML Annotation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Annotation Generation Flow                      │
└─────────────────────────────────────────────────────────────────────┘

Input Text: "For God so loved the world..."
                    │
                    ▼
        ┌───────────────────────┐
        │  1. Preprocessing     │
        │  - Tokenization       │
        │  - Normalization      │
        └───────────┬───────────┘
                    │
    ┌───────────────┴───────────────┬───────────────────────┐
    ▼                               ▼                       ▼
┌─────────────────┐        ┌─────────────────┐    ┌─────────────────┐
│  Zero-Shot      │        │   Few-Shot      │    │  BERT Context   │
│  Classifier     │        │   Matching      │    │   Analysis      │
├─────────────────┤        ├─────────────────┤    ├─────────────────┤
│ Model: BART     │        │ Examples DB     │    │ Model: BERT     │
│ Labels:         │        │ Similarity:     │    │ Embeddings:     │
│ - Love          │        │ - Cosine        │    │ - 768 dims      │
│ - Salvation     │        │ - Jaccard       │    │ - Contextual    │
│ - Grace         │        │ - Edit distance │    │ - Clustering    │
└────────┬────────┘        └────────┬────────┘    └────────┬────────┘
         │                          │                       │
         │         Confidence       │      Confidence       │
         │         0.85            │         0.72          │
         └──────────────────┬───────┴───────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Ensemble   │
                    │   Scoring    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Quality    │
                    │   Control    │
                    │ Threshold>0.7│
                    └──────┬──────┘
                           │
                           ▼
                Output Annotations:
                - Theological Theme: "Love of God" (0.92)
                - Key Term: "κόσμος/world" (0.88)
                - Cross Reference: "1 John 4:9" (0.75)
```

## 4. Cross-Reference Detection

### Quote Detection Algorithm

```
Source: "All have sinned and fall short" (ROM 3:23)
Target Corpus: Old Testament
                    │
                    ▼
        ┌───────────────────────┐
        │  Tokenize & Normalize │
        └───────────┬───────────┘
                    │
                    ▼
        Tokens: ["all", "have", "sinned", "fall", "short"]
                    │
                    ▼
        ┌───────────────────────────────────┐
        │  Sliding Window Search (n=3)      │
        │  Window 1: ["all", "have", "sinned"]│
        │  Window 2: ["have", "sinned", "fall"]│
        │  Window 3: ["sinned", "fall", "short"]│
        └───────────┬───────────────────────┘
                    │
                    ▼
        Search each window in target verses
                    │
                    ▼
        Match Found: PSA 14:3 "they have all turned aside"
        Confidence = (matched_tokens / source_tokens) × weight
                   = (3/5) × 1.2 = 0.72
```

### Semantic Similarity Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                    Semantic Similarity Flow                      │
└─────────────────────────────────────────────────────────────────┘

Source Verse                              Target Verse
"The LORD is my shepherd"                 "He makes me lie down"
      │                                         │
      ▼                                         ▼
 BERT Encoder                              BERT Encoder
      │                                         │
      ▼                                         ▼
 [0.23, -0.45, 0.67, ...]                 [0.19, -0.41, 0.71, ...]
      │                                         │
      └──────────────┬──────────────────────────┘
                     │
                     ▼
              Cosine Similarity
              cos(θ) = A·B / (||A||×||B||)
                     │
                     ▼
              Similarity: 0.89
                     │
                     ▼
         If similarity > 0.8 AND
         shared_keywords >= 2
                     │
                     ▼
         Create Cross-Reference
         Type: THEMATIC_PARALLEL
         Confidence: 0.89 × 0.9 = 0.80
```

## 5. Timeline Processing

### BCE Date Encoding

```
Historical Date: 1446 BCE (The Exodus)
                    │
                    ▼
┌─────────────────────────────────────────────┐
│          BCE Date Encoding Logic            │
├─────────────────────────────────────────────┤
│                                             │
│  Python datetime cannot handle BCE dates    │
│  Solution: Offset encoding                  │
│                                             │
│  Encoded_Year = 5000 - BCE_Year            │
│  1446 BCE → 5000 - 1446 = 3554             │
│                                             │
│  Store: datetime(3554, 1, 1)               │
│  Metadata: {"bce": true, "original": 1446} │
│                                             │
└─────────────────────────────────────────────┘
                    │
                    ▼
            Stored in Database
```

### Temporal Uncertainty Modeling

```
┌─────────────────────────────────────────────────────────────────┐
│                    Uncertainty Confidence Scale                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Exact Date Known       ████████████████████████  1.00         │
│  (e.g., "March 15, 44 BCE")                                    │
│                                                                 │
│  Year Known            ██████████████████████     0.90         │
│  (e.g., "30 CE")                                               │
│                                                                 │
│  Decade Known          ████████████████           0.70         │
│  (e.g., "40s CE")                                              │
│                                                                 │
│  Century Known         ██████████                 0.50         │
│  (e.g., "1st century")                                         │
│                                                                 │
│  Disputed/Unknown      ████                       0.20         │
│  (e.g., "sometime in antiquity")                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 6. Export Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Export Pipeline Flow                        │
└─────────────────────────────────────────────────────────────────┘

                    Canonical Data Model
                           │
                           ▼
                 ┌─────────────────────┐
                 │  Validation Layer   │
                 │  - Schema checking  │
                 │  - Data integrity   │
                 └─────────┬───────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │         Transform Layer           │
         │   Format-specific conversions     │
         └─────────────────┬─────────────────┘
                           │
     ┌──────────┬──────────┴──────────┬──────────┐
     ▼          ▼                     ▼          ▼
┌─────────┐┌─────────┐          ┌─────────┐┌─────────┐
│ SQLite  ││  JSON   │          │  Open   ││  Graph  │
│         ││         │          │ Search  ││   DB    │
├─────────┤├─────────┤          ├─────────┤├─────────┤
│ Tables: ││ Files:  │          │Indices: ││ Nodes:  │
│ -verses ││ -books/ │          │ -verses ││ -Verse  │
│ -tokens ││ -search/│          │ -topics ││ -Person │
│ -annot. ││ -xrefs/ │          │ -xrefs  ││ -Place  │
└─────────┘└─────────┘          └─────────┘└─────────┘
     │          │                     │          │
     ▼          ▼                     ▼          ▼
  bible.db   /json/              ES Cluster   Neo4j
```

## 7. Performance Optimization Strategies

### Caching Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Cache Hierarchy                            │
└─────────────────────────────────────────────────────────────────┘

Level 1: In-Memory Cache (LRU)
├── Model Predictions (5 min TTL)
├── Alignment Results (10 min TTL)
└── Parsed Morphology (30 min TTL)
                │
                ▼
Level 2: Redis Cache
├── Cross-References (1 hour TTL)
├── Annotation Results (2 hour TTL)
└── Search Results (4 hour TTL)
                │
                ▼
Level 3: Database Cache
├── Pre-computed Indices
├── Materialized Views
└── Common Query Results
```

### Parallel Processing Flow

```
                Input: 1000 Verses
                        │
                        ▼
                ┌───────────────┐
                │   Chunking    │
                │  (size=100)   │
                └───────┬───────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
    Worker 1        Worker 2        Worker 3        Worker 4
    [1-100]        [101-200]      [201-300]      [301-400]
        │               │               │               │
        │               │               │               │
        ▼               ▼               ▼               ▼
    Process         Process         Process         Process
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                                │
                        ┌───────▼───────┐
                        │   Merge       │
                        │   Results     │
                        └───────────────┘
```

## Summary

These visual representations show how ABBA's algorithms work together to process biblical texts:

1. **Alignment** matches original language to translations using statistical and neural methods
2. **Morphology** decodes grammatical information from standardized codes
3. **Annotations** use ML ensemble methods for accurate classification
4. **Cross-references** detect quotes and thematic parallels
5. **Timeline** handles historical dates with uncertainty modeling
6. **Export** transforms data for multiple output formats
7. **Optimization** uses caching and parallelization for performance

For implementation details, see [Data Flow & Algorithms](DATA_FLOW_AND_ALGORITHMS.md).