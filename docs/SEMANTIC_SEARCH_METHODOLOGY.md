# Semantic Search Methodology

## Overview

ABBA implements a **Strong's-Centric Semantic Mapping** approach for biblical concept searching. This methodology prioritizes lexicographic accuracy and scholarly defensibility over algorithmic complexity.

## Core Principles

1. **Strong's Numbers as Semantic Anchors**: Every concept is defined primarily through Strong's Concordance numbers, which provide lemma-level semantic identification
2. **Traceable Authority**: All matches can be traced back to established lexicographic sources (BDAG, BDB, Strong's)
3. **No Semantic Inference**: The system does not infer meaning beyond what is explicitly defined in authoritative lexicons
4. **Transparent Confidence Scoring**: Each match type has a clearly defined confidence level

## Architecture

### 1. Concept Definition Structure

```yaml
concept:
  name: love
  primary_strongs: [G25, G26]      # Core semantic representation
  extended_strongs: [G5368, G5360]  # Related terms in semantic field
  hebrew_strongs: [H157, H160]      # Hebrew equivalents
  phrase_patterns:                  # Multi-word expressions
    - name: "love one another"
      strongs: [G25, G240]
  validation_source: "BDAG"         # Lexicographic authority
```

### 2. Database Schema

The system relies on normalized data for efficient searching:

```sql
-- Main verse data with normalized columns
stepbible_verses:
  - original_word: Greek/Hebrew with diacritics
  - normalized_word: Stripped of vowel points/accents
  - strongs_lexical: Clean Strong's numbers (extracted from complex formats)
  - morphology: Grammatical information

-- Lexicon for authoritative definitions
lexicon:
  - strongs_number: Primary key
  - lemma: Dictionary form (if available)
  - definition: Authoritative gloss
```

### 3. Search Algorithm

The search process follows a layered approach with decreasing confidence:

```python
def search_concept(concept):
    matches = []
    
    # Layer 1: Primary Strong's (confidence: 1.0)
    # Direct semantic matches - highest precision
    matches += find_strongs_matches(concept.primary_strongs, confidence=1.0)
    
    # Layer 2: Extended Strong's (confidence: 0.8)
    # Related terms in the semantic field
    matches += find_strongs_matches(concept.extended_strongs, confidence=0.8)
    
    # Layer 3: Phrase Patterns (confidence: 0.9)
    # Multi-word expressions with specific meaning
    matches += find_phrase_patterns(concept.phrase_patterns, confidence=0.9)
    
    # Layer 4: Lemma Variants (confidence: 0.7)
    # Same lemma, different Strong's number
    matches += find_lemma_variants(concept.all_strongs, confidence=0.7)
    
    return deduplicate_and_rank(matches)
```

## Match Types and Confidence Levels

| Match Type | Confidence | Description | Example |
|------------|------------|-------------|---------|
| Primary | 1.0 | Direct Strong's match | G26 (ἀγάπη) for "love" |
| Phrase | 0.9 | Multi-word pattern | G932 + G2316 for "kingdom of God" |
| Extended | 0.8 | Semantic field match | G5368 (φιλέω) for "love" |
| Lemma | 0.7 | Same lemma, different Strong's | Variants sharing lexicon entry |
| Potential | 0.3 | Requires manual review | Embedding similarity only |

## Why This Approach?

### 1. **Scholarly Defensibility**
- Every match can be verified against published lexicons
- No "black box" semantic inference
- Respects the linguistic work of biblical scholars

### 2. **Accuracy Over Coverage**
- Prefers missing some matches to including false positives
- Users can trust that returned results are legitimate
- Clear about what the system does NOT do

### 3. **Transparency**
- Users can see exactly why each match was included
- Confidence scores have concrete meaning
- Evidence trail for academic citation

### 4. **Extensibility**
- New concepts can be added by defining Strong's numbers
- System can grow without algorithmic changes
- Community can contribute validated concept definitions

## What This System Does NOT Do

1. **Semantic Inference**: Does not guess at meaning beyond lexicon entries
2. **Metaphorical Extension**: Does not automatically handle figurative language
3. **Discourse Analysis**: Operates at word/phrase level, not paragraph/discourse
4. **Theological Interpretation**: Returns linguistic matches, not doctrinal conclusions
5. **Textual Criticism**: Uses the text as provided, doesn't handle variants

## Handling Greek Morphology

The system addresses Greek inflected forms through:

1. **Strong's Number Consistency**: All forms of a word share the same Strong's number
2. **Normalized Text Matching**: Strips accents for basic form matching
3. **Lexicon Lemmatization**: Uses lexicon entries to group related forms
4. **No Algorithmic Stemming**: Avoids linguistic assumptions

Example:
- ἀγαπάω (I love) - G25
- ἀγαπήσεις (you will love) - G25
- ἀγαπῶμεν (we might love) - G25

All mapped through Strong's G25, no complex morphological analysis required.

## Integration with Embeddings

While the primary search is Strong's-based, embeddings serve a supplementary role:

1. **Discovery Tool**: Find potential concepts for manual validation
2. **Similarity Scoring**: Rank results within same confidence tier
3. **Cross-Lingual Bridge**: Connect Hebrew and Greek concepts
4. **Never Primary**: Embedding matches alone are marked "potential" (0.3 confidence)

## Example: Searching for "Love"

```yaml
Input: concept "love"
Process:
  1. Primary Strong's G25, G26 → 1,234 verses (confidence: 1.0)
  2. Extended Strong's G5368 → 456 verses (confidence: 0.8)
  3. Hebrew equivalents H157 → 789 verses (confidence: 1.0)
  4. Phrase "love one another" → 12 verses (confidence: 0.9)
  
Output: 2,491 verses ranked by confidence and reference
Evidence: Each match includes Strong's number and lexicon definition
```

## Future Enhancements

Potential improvements that maintain the scholarly approach:

1. **Syntactic Patterns**: Use morphology codes for grammatical constructions
2. **Semantic Domains**: Group concepts by Louw-Nida domains
3. **Cross-Reference Integration**: Include scripture cross-references
4. **Manuscript Variants**: Optional layer for textual criticism

## Contributing Concepts

To add a new biblical concept:

1. Identify primary Strong's numbers from respected concordance
2. Verify with lexicon (BDAG for Greek, BDB for Hebrew)
3. Add extended Strong's for semantic field
4. Document validation source
5. Submit with example verses showing usage

## API Usage

```python
from abba.semantic import StrongsConcordance

# Initialize
concordance = StrongsConcordance(db_path)

# Define concept
love_concept = concordance.define_concept(
    name="love",
    primary_strongs=["G25", "G26"],
    extended_strongs=["G5368"]
)

# Build concordance
matches = concordance.build_concordance(love_concept)

# Generate report
report = concordance.generate_report(love_concept, matches)
```

## Validation and Testing

Every concept definition should be validated against:

1. **Lexicon entries**: Strong's numbers exist and definitions match
2. **Sample verses**: Manual review of top matches confirms accuracy
3. **Negative cases**: Ensures excluded terms don't appear
4. **Cross-lingual consistency**: Hebrew and Greek concepts align

## Conclusion

This methodology provides a robust, transparent, and academically sound approach to biblical concept searching. By anchoring on Strong's Concordance and established lexicons, we ensure that results are both accurate and defensible in scholarly contexts.