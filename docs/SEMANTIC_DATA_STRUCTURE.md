# Semantic Data Storage Structure

## Overview
Semantic concept data should be stored in a structured, educational-friendly format that supports Bible study, personal research, and quick lookups. The data is organized under `ABBA_DATA_DIR/concepts/` with both database storage and file-based exports for different use cases.

## Directory Structure
```
ABBA_DATA_DIR/
├── concepts/               # All concept-related data
│   ├── mappings/           # Individual concept mapping files
│   │   ├── love.json       # Complete data for "love" concept
│   │   ├── faith.json      # Complete data for "faith" concept
│   │   └── ...
│   ├── indexes/            # Quick lookup indexes
│   │   ├── verse_to_concepts.json  # Which concepts each verse relates to
│   │   ├── strongs_to_concepts.json # Map Strong's numbers to concepts
│   │   └── semantic_clusters.json   # Semantically related concept groups
│   ├── reports/            # Human-readable reports
│   │   ├── love_study.md   # Bible study format for "love"
│   │   └── ...
│   └── metadata.json       # Concept processing metadata
```

## JSON Format for Concept Mappings

Each concept file (e.g., `love.json`) contains:

```json
{
  "concept": {
    "name": "love",
    "description": "Biblical concept of love including agape, phileo, and hesed",
    "primary_strongs": ["G25", "G26", "H157", "H160"],
    "extended_strongs": ["G5368", "H2245", "H1730"],
    "semantic_keywords": ["affection", "charity", "compassion", "devotion"]
  },
  "statistics": {
    "total_matches": 750,
    "lexical_matches": 690,
    "semantic_matches": 60,
    "books_covered": 55,
    "confidence_distribution": {
      "high": 690,
      "medium": 45,
      "low": 15
    },
    "last_updated": "2025-01-18T10:30:00Z",
    "processing_version": "1.0.0"
  },
  "verses": {
    "lexical": [
      {
        "verse_id": "1Co:13:1",
        "book": "1 Corinthians",
        "chapter": 13,
        "verse": 1,
        "text": "Though I speak with the tongues of men and of angels...",
        "original_text": "Ἐὰν ταῖς γλώσσαις τῶν ἀνθρώπων λαλῶ...",
        "strongs_matched": ["G26"],
        "word_occurrences": [
          {
            "word": "ἀγάπην",
            "strongs": "G26",
            "position": 15,
            "morphology": "N-ASF",
            "gloss": "love"
          }
        ],
        "confidence": 1.0,
        "context": {
          "previous": "1Co:12:31",
          "next": "1Co:13:2",
          "chapter_theme": "The supremacy of love"
        }
      }
    ],
    "semantic": [
      {
        "verse_id": "Gal:5:22",
        "book": "Galatians",
        "chapter": 5,
        "verse": 22,
        "text": "But the fruit of the Spirit is love, joy, peace...",
        "semantic_score": 0.89,
        "ollama_validation": {
          "validated": true,
          "confidence": 0.85,
          "reasoning": "Describes love as divine fruit, core aspect of the concept"
        },
        "thematic_connection": "Spiritual manifestation of love",
        "context": {
          "section": "Fruit of the Spirit",
          "related_concepts": ["joy", "peace", "patience"]
        }
      }
    ]
  },
  "themes": {
    "God's_love": {
      "description": "Verses about God's love for humanity",
      "verse_ids": ["Jhn:3:16", "Rom:5:8", "1Jn:4:8"],
      "key_insights": "God's love is sacrificial and unconditional"
    },
    "brotherly_love": {
      "description": "Love between believers",
      "verse_ids": ["1Jn:4:7", "Rom:12:10", "1Pe:1:22"],
      "key_insights": "Believers are commanded to love one another"
    },
    "love_commands": {
      "description": "Commandments about love",
      "verse_ids": ["Mat:22:37-39", "Mar:12:30-31", "Jhn:13:34"],
      "key_insights": "Love God and love neighbor summarize the law"
    }
  },
  "cross_references": {
    "related_concepts": ["grace", "mercy", "compassion", "kindness"],
    "contrasting_concepts": ["hate", "indifference", "selfishness"],
    "progression": {
      "foundation": ["faith"],
      "expression": ["service", "sacrifice"],
      "result": ["unity", "peace"]
    }
  },
  "study_notes": {
    "hebrew_perspective": "Hesed (H2617) emphasizes covenant loyalty...",
    "greek_perspective": "Agape (G26) represents unconditional, divine love...",
    "practical_application": "Love is not merely emotion but action...",
    "historical_context": "In ancient Near Eastern culture..."
  }
}
```

## Index Files

### verse_to_concepts.json
Maps each verse to all concepts it relates to:
```json
{
  "1Co:13:1": {
    "concepts": ["love", "faith", "spiritual_gifts"],
    "primary": ["love"],
    "semantic": ["spiritual_gifts"]
  }
}
```

### strongs_to_concepts.json
Maps Strong's numbers to concepts:
```json
{
  "G26": ["love", "charity"],
  "G4102": ["faith", "belief", "trust"],
  "H2617": ["mercy", "kindness", "love"]
}
```

### semantic_clusters.json
Groups related concepts:
```json
{
  "virtues": ["love", "faith", "hope", "patience"],
  "emotions": ["love", "joy", "peace", "anger"],
  "relationships": ["love", "unity", "fellowship"]
}
```

## Bible Study Report Format (Markdown)

`reports/love_study.md`:
```markdown
# Bible Study: Love

## Overview
- **Total Verses**: 750
- **Key Strong's Numbers**: G25 (agapaō), G26 (agapē), H157 (ahab)
- **Main Books**: 1 Corinthians, 1 John, John

## Key Themes

### 1. God's Love
**Foundation Verse**: John 3:16
> "For God so loved the world..."

**Supporting Verses**:
- Romans 5:8 - Love demonstrated through Christ's death
- 1 John 4:8 - God is love

### 2. Love Commands
**Foundation Verse**: Matthew 22:37-39
> "Love the Lord your God... Love your neighbor as yourself"

## Word Studies

### Greek: ἀγάπη (agapē) - G26
- **Occurrences**: 116 times
- **Definition**: Unconditional, divine love
- **Key Usage**: 1 Corinthians 13 (Love chapter)

### Hebrew: אָהַב (ahab) - H157
- **Occurrences**: 208 times
- **Definition**: To love, showing affection
- **Key Usage**: Deuteronomy 6:5 (Love the Lord)

## Semantic Connections
Verses that embody the concept without using the specific words:
- Galatians 5:22 - Fruit of the Spirit
- Philippians 2:3-4 - Considering others

## Application Questions
1. How does understanding agape vs phileo change your view of God's love?
2. What practical steps can you take to show this love?

## Cross-References
- See also: Grace, Mercy, Compassion
- Contrasts with: Hate, Selfishness
```

## Database Schema Extensions

Add tables for efficient semantic lookups:

```sql
-- Concept definitions
CREATE TABLE concepts (
    concept_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    primary_strongs TEXT,  -- JSON array
    extended_strongs TEXT, -- JSON array
    semantic_keywords TEXT, -- JSON array
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Verse-concept relationships with scores
CREATE TABLE verse_concepts (
    verse_id TEXT,
    concept_id INTEGER,
    match_type TEXT CHECK(match_type IN ('lexical', 'semantic')),
    confidence REAL,
    semantic_score REAL,
    strongs_matched TEXT, -- JSON array
    ollama_validated BOOLEAN,
    ollama_confidence REAL,
    PRIMARY KEY (verse_id, concept_id),
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

-- Concept themes for study organization
CREATE TABLE concept_themes (
    theme_id INTEGER PRIMARY KEY,
    concept_id INTEGER,
    theme_name TEXT,
    description TEXT,
    verse_ids TEXT, -- JSON array
    key_insights TEXT,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

-- Cross-concept relationships
CREATE TABLE concept_relationships (
    concept_id INTEGER,
    related_concept_id INTEGER,
    relationship_type TEXT CHECK(relationship_type IN 
        ('related', 'contrasting', 'foundation', 'expression', 'result')),
    strength REAL,
    PRIMARY KEY (concept_id, related_concept_id, relationship_type),
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id),
    FOREIGN KEY (related_concept_id) REFERENCES concepts(concept_id)
);

-- Indexes for fast lookups
CREATE INDEX idx_verse_concepts_verse ON verse_concepts(verse_id);
CREATE INDEX idx_verse_concepts_concept ON verse_concepts(concept_id);
CREATE INDEX idx_verse_concepts_type ON verse_concepts(match_type);
CREATE INDEX idx_verse_concepts_confidence ON verse_concepts(confidence DESC);
```

## Benefits of This Structure

1. **Educational Focus**: Organized by themes and study topics
2. **Quick Lookups**: Indexed for fast verse-to-concept and concept-to-verse queries
3. **Context-Rich**: Includes surrounding verses and thematic connections
4. **Cross-Referenced**: Links related concepts for deeper study
5. **Bilingual**: Preserves original language alongside translations
6. **Confidence Scoring**: Users can filter by reliability
7. **Study-Ready**: Markdown reports can be directly used for teaching
8. **Extensible**: JSON format allows adding new fields without breaking existing code
9. **Version Controlled**: Metadata tracks when and how data was generated
10. **API-Friendly**: JSON structure works well with web services

## Usage Examples

### Finding all verses about love with high confidence:
```python
with open('concepts/mappings/love.json') as f:
    data = json.load(f)
    high_confidence = [v for v in data['verses']['lexical'] 
                      if v['confidence'] >= 0.9]
```

### Getting verses that relate to multiple concepts:
```python
with open('concepts/indexes/verse_to_concepts.json') as f:
    index = json.load(f)
    multi_concept = {v: c for v, c in index.items() 
                    if len(c['concepts']) > 2}
```

### Building a thematic Bible study:
```python
def build_study(concept_name, theme_name):
    with open(f'concepts/mappings/{concept_name}.json') as f:
        data = json.load(f)
    theme = data['themes'][theme_name]
    verses = [v for v in data['verses']['lexical'] 
             if v['verse_id'] in theme['verse_ids']]
    return {
        'theme': theme['description'],
        'insights': theme['key_insights'],
        'verses': verses
    }
```

This structure provides a comprehensive, educational-focused storage system for semantic biblical data that serves both technical and ministry needs.