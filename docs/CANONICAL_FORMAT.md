# ABBA Canonical Data Format Specification

## Overview

The ABBA Canonical Format is the authoritative source from which all other data formats are generated. It prioritizes completeness, correctness, and clarity over performance or size optimization.

## Format Specification

### Top-Level Structure

```json
{
  "format_version": "1.0.0",
  "metadata": {
    "generated_at": "2024-01-01T00:00:00Z",
    "generator_version": "abba-tools-1.0.0",
    "sources": [
      {
        "type": "translation",
        "id": "ESV",
        "version": "2016",
        "copyright": "...",
        "license": "..."
      }
    ],
    "canonical_books": {
      "protestant": ["GEN", "EXO", ...],
      "catholic": ["GEN", "EXO", ..., "TOB", ...],
      "orthodox": ["GEN", "EXO", ..., "1MA", "2MA", ...]
    }
  },
  "books": [...],
  "cross_references": [...],
  "annotations": [...],
  "timeline": [...],
  "people": [...],
  "places": [...],
  "word_definitions": [...],
  "semantic_concepts": [...]
}
```

### Book Structure

```json
{
  "id": "GEN",
  "names": {
    "en": {
      "full": "Genesis",
      "short": "Gen",
      "alternate": ["1 Moses", "Bereshit"]
    },
    "he": {
      "full": "בְּרֵאשִׁית",
      "short": "בר׳"
    }
  },
  "metadata": {
    "author": {
      "traditional": "Moses",
      "critical": "Multiple sources (J, E, P, D)",
      "confidence": 0.4,
      "references": [
        {
          "type": "scholarly_article",
          "citation": "Friedman, Richard Elliott. 'Who Wrote the Bible?' Harper & Row, 1987, pp. 50-70.",
          "perspective": "critical",
          "url": "https://doi.org/..."
        },
        {
          "type": "commentary",
          "citation": "Sailhamer, John H. 'The Pentateuch as Narrative.' Zondervan, 1992, pp. 1-79.",
          "perspective": "traditional"
        }
      ]
    },
    "date_written": {
      "traditional": "1440-1400 BCE",
      "critical": "950-500 BCE",
      "confidence": 0.5,
      "references": [
        {
          "type": "archaeological_study",
          "citation": "Kitchen, K.A. 'On the Reliability of the Old Testament.' Eerdmans, 2003, pp. 294-300.",
          "perspective": "traditional"
        }
      ]
    },
    "original_language": "Hebrew",
    "genre": ["narrative", "law", "poetry"],
    "canonical_order": {
      "protestant": 1,
      "catholic": 1,
      "hebrew": 1
    }
  },
  "chapters": [...]
}
```

#### Traditional vs Critical Perspectives

The ABBA format includes both **traditional** and **critical** scholarly perspectives throughout the data:

- **Traditional**: Represents views held by religious traditions, often based on internal biblical testimony, ancient Jewish and Christian sources, and conservative scholarship. These views typically accept biblical claims at face value.

- **Critical**: Represents modern scholarly consensus based on historical-critical methods, archaeological evidence, linguistic analysis, and comparative ancient Near Eastern studies. These views often challenge traditional authorship and dating.

Examples:
- **Authorship**: Traditional view attributes Pentateuch to Moses; critical view suggests multiple sources (J, E, P, D)
- **Dating**: Traditional dates often align with biblical chronology; critical dates are based on archaeological and linguistic evidence
- **Historical Events**: Traditional view accepts biblical narratives as historical; critical view may question historicity

When these perspectives differ significantly, the `confidence` field indicates the level of scholarly agreement.

### Chapter Structure

```json
{
  "number": 1,
  "metadata": {
    "theme": "Creation",
    "outline": [
      {"verses": "1-2", "title": "The Initial State"},
      {"verses": "3-31", "title": "The Six Days of Creation"}
    ],
    "literary_structure": "Narrative with poetic elements"
  },
  "verses": [...]
}
```

### Verse Structure

```json
{
  "canonical_id": "GEN.1.1",
  "number": 1,
  "part": null,  // or "a", "b" for split verses
  
  "translations": {
    "ESV": {
      "text": "In the beginning, God created the heavens and the earth.",
      "footnotes": [
        {
          "marker": "a",
          "text": "Or In the beginning when God created or When God began to create",
          "type": "translation_alternative"
        }
      ]
    },
    "NIV": {
      "text": "In the beginning God created the heavens and the earth."
    },
    "HEBREW": {
      "text": "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃",
      "transliteration": "bereshit bara elohim et hashamayim ve'et ha'aretz"
    }
  },
  
  "original_language": {
    "language": "Hebrew",
    "tokens": [
      {
        "position": 1,
        "surface": "בְּרֵאשִׁ֖ית",
        "transliteration": "bereshit",
        "lemma": "רֵאשִׁית",
        "strongs": ["H7225"],
        "morphology": {
          "pos": "noun",
          "gender": "feminine",
          "number": "singular",
          "state": "construct",
          "suffix": "none"
        },
        "gloss": "beginning",
        "semantic_domain": ["time", "creation"],
        "etymology": {
          "root": "ראש",
          "meaning": "head, first"
        }
      },
      // ... more tokens
    ]
  },
  
  "annotations": {
    "lexical_domains": [
      {
        "strong_number": "H7225",
        "domain": "TIME",
        "subdomain": "BEGINNING",
        "confidence": 1.0,
        "source": "SDBH"  // Semantic Dictionary of Biblical Hebrew
      },
      {
        "strong_number": "H1254", 
        "domain": "CREATION",
        "subdomain": "DIVINE_ACTION",
        "confidence": 1.0,
        "source": "SDBH"
      }
    ],
    "word_frequencies": {
      "hapax_legomena": [],  // Words appearing only once in Bible
      "rare_words": [],      // Words appearing < 10 times
      "keywords": [          // Statistically significant for this passage
        {
          "lemma": "H1254",  // bara
          "tf_idf": 0.92,    // Term frequency-inverse document frequency
          "significance": "Creation verb used only with God as subject"
        }
      ]
    },
    "linguistic_features": {
      "verbal_patterns": {
        "main_verb": "H1254",  // bara
        "stem": "qal",
        "aspect": "perfect",
        "subject_always_divine": true,
        "confidence": 1.0
      },
      "syntactic_structure": {
        "clause_type": "verbal",
        "word_order": "V-S-O",  // Verb-Subject-Object
        "clause_relations": []   // Standalone opening
      },
      "discourse_markers": {
        "beginning_marker": "בְּרֵאשִׁית",
        "narrative_sequence": false,  // No vav-consecutive
        "temporal_indicator": "absolute_beginning"
      }
    },
    "semantic_fields": {
      "explicit_concepts": [  // Directly from morphology
        {
          "concept": "beginning/time",
          "words": ["H7225"],
          "confidence": 1.0
        },
        {
          "concept": "creation/making",
          "words": ["H1254"],
          "confidence": 1.0  
        },
        {
          "concept": "deity",
          "words": ["H430"],  // Elohim
          "confidence": 1.0
        },
        {
          "concept": "cosmos/space",
          "words": ["H8064", "H776"],  // heavens, earth
          "confidence": 1.0
        }
      ],
      "collocations": [  // Words that frequently appear together
        {
          "phrase": "heavens_and_earth",
          "frequency": 42,
          "significance": "Merism for totality of creation",
          "confidence": 1.0
        }
      ]
    },
    "intertextuality": {
      "quotations": [],  // Direct quotes (none for Gen 1:1)
      "allusions": [],   // Clear references (none for Gen 1:1)
      "shared_vocabulary": [  // Same rare words/phrases
        {
          "phrase": "בְּרֵאשִׁית",
          "other_occurrences": ["JER.26.1", "JER.27.1", "JER.28.1"],
          "pattern": "Temporal markers for significant beginnings",
          "confidence": 1.0
        }
      ],
      "parallel_structures": [  // Similar syntactic patterns
        {
          "verse": "JOH.1.1",
          "pattern": "ἐν ἀρχῇ",  // Greek parallel to בְּרֵאשִׁית
          "lxx_influence": true,
          "confidence": 0.95
        }
      ]
    },
    "text_critical": {
      "manuscript_variants": [],  // None for Gen 1:1
      "ancient_versions": {
        "lxx": "ἐν ἀρχῇ ἐποίησεν ὁ θεὸς τὸν οὐρανὸν καὶ τὴν γῆν",
        "targum": "בְּקַדְמִין בְּרָא יְיָ יָת שְׁמַיָּא וְיָת אַרְעָא",
        "vulgate": "in principio creavit Deus caelum et terram"
      },
      "translation_decisions": {
        "definite_article": false,  // "In beginning" vs "In the beginning"
        "temporal_vs_dependent": "temporal",  // Not "When God began..."
        "confidence": 0.85
      }
    },
    "x_manual_annotations": {
      // Extension point for manual theological/thematic annotations
      // These would NOT be generated by ABBA automatically
    }
  },
  
  "cross_references": [
    {
      "target": "JOH.1.1-3",
      "type": "parallel",
      "confidence": 0.95,
      "note": "Creation through the Word"
    },
    {
      "target": "HEB.11.3",
      "type": "commentary",
      "confidence": 0.85,
      "note": "Creation by faith",
      "references": [
        {
          "type": "commentary",
          "citation": "Bruce, F.F. *The Epistle to the Hebrews*. NICNT. Eerdmans, 1990, pp. 279-280.",
          "perspective": "traditional",
          "quote": "The author sees Genesis 1:1 as the foundational assertion of creation ex nihilo."
        }
      ]
    }
  ],
  
  "timeline": {
    "event_id": "creation",
    "date": "beginning",
    "precision": "theological",
    "order": 1
  },
  
  "entities": {
    "people": [],
    "places": [
      {
        "id": "heavens",
        "type": "cosmic"
      },
      {
        "id": "earth",
        "type": "cosmic"
      }
    ],
    "objects": []
  }
}
```

#### Confidence Scores and Documentation

The `confidence` field appears throughout the format and indicates the certainty or scholarly agreement level for various assertions. **Confidence scores below 0.9 SHOULD include supporting references.**

**Scale**: 0.0 to 1.0
- **1.0**: Universal agreement, indisputable (e.g., "creation" as a topic for Gen 1:1)
- **0.9-0.99**: Strong consensus with minor variations
- **0.7-0.89**: General agreement with some scholarly debate (references required)
- **0.5-0.69**: Disputed, with significant scholarly disagreement (references required)
- **Below 0.5**: Highly speculative or minority views (references required)

**Reference Requirements**:
When confidence < 0.9, try to include a `references` array with:
```json
{
  "confidence": 0.7,
  "references": [
    {
      "type": "scholarly_article|commentary|monograph|archaeological_study|manuscript_evidence",
      "citation": "Full bibliographic citation in Chicago/Turabian format",
      "perspective": "traditional|critical|neutral",
      "url": "DOI or stable URL if available",
      "pages": "Specific page numbers",
      "quote": "Relevant quote supporting this position (optional)"
    }
  ]
}
```

**Citation Format Requirements**:
- Author Last, First. "Article Title." *Journal Name* vol. # (Year): pages.
- Author Last, First. *Book Title*. Publisher, Year, pp. #-#.
- Must include specific page numbers, not just the work as a whole
- DOI preferred over regular URLs when available

**Applications**:
1. **Topic/Theme Assignment**: How certainly a verse relates to a topic
   - 1.0 = Explicit and undeniable connection (no references needed)
   - 0.8 = Strong implicit connection (1-2 references)
   - 0.5 = Possible but debated connection (3+ references from different perspectives)

2. **Cross-References**: Strength of connection between passages
   - 0.95 = Direct quotation or explicit reference
   - 0.85 = Clear allusion or parallel
   - 0.7 = Thematic connection
   - 0.5 = Possible but uncertain connection

3. **Authorship/Dating**: Level of scholarly consensus
   - 0.9+ = Near universal agreement
   - 0.7-0.89 = Majority view with notable dissent
   - 0.3-0.69 = Significantly disputed
   - Below 0.3 = Minority or fringe position

**Calculation Methods**:
- For scholarly matters: Based on systematic review of peer-reviewed literature from the last 50 years or with public consensus
- For textual connections: Based on linguistic, thematic, and contextual analysis with citation
- For manuscript variants: Based on manuscript evidence weight (age, number, geographic distribution)

### Cross-Reference Structure

```json
{
  "id": "xref_001",
  "source": {
    "verses": ["GEN.1.1"],
    "context": "Creation account"
  },
  "target": {
    "verses": ["JOH.1.1-3"],
    "context": "Word as creator"
  },
  "relationship": {
    "type": "theological_parallel",
    "subtype": "creation_theology",
    "direction": "forward",  // OT -> NT
    "confidence": 0.95
  },
  "metadata": {
    "discovered_by": "early_church_fathers",
    "scholarly_consensus": "strong",
    "notes": "Both passages emphasize pre-existence and creative power"
  }
}
```

### Annotation Structure

```json
{
  "id": "topic_creation",
  "type": "topic",
  "hierarchy": "/theology/god/works/creation",
  "names": {
    "en": "Creation",
    "es": "Creación",
    "he": "בריאה"
  },
  "definition": "The act of God bringing the universe into existence",
  "related_topics": ["providence", "sovereignty", "cosmology"],
  "key_verses": ["GEN.1.1", "JOH.1.3", "COL.1.16"],
  "scholarly_notes": "..."
}
```

### Timeline Structure

```json
{
  "id": "timeline_creation",
  "type": "theological_event",
  "names": {
    "en": "Creation",
    "he": "בריאת העולם"
  },
  "date": {
    "display": "In the beginning",
    "canonical": "beginning",
    "alternatives": {
      "young_earth": "~4000 BCE",
      "old_earth": "~13.8 billion years ago"
    },
    "precision": "theological_marker"
  },
  "duration": {
    "biblical": "6 days",
    "interpretation": "disputed"
  },
  "verses": ["GEN.1.1-2.3"],
  "significance": "Foundation of biblical worldview"
}
```

### Person Structure

```json
{
  "id": "person_abraham",
  "names": {
    "primary": "Abraham",
    "alternate": ["Abram"],
    "meaning": "Father of many",
    "languages": {
      "he": "אַבְרָהָם",
      "ar": "إبراهيم"
    }
  },
  "genealogy": {
    "father": "person_terah",
    "mother": null,
    "spouse": ["person_sarah", "person_hagar", "person_keturah"],
    "children": ["person_ishmael", "person_isaac", "..."]
  },
  "chronology": {
    "birth": "~2000 BCE",
    "death": "~1825 BCE",
    "age_at_death": 175
  },
  "biography": {
    "birthplace": "place_ur",
    "occupation": ["nomad", "herder"],
    "key_events": [
      {
        "event": "call",
        "verses": ["GEN.12.1-3"],
        "age": 75
      }
    ]
  },
  "significance": {
    "roles": ["patriarch", "prophet"],
    "covenant": "abrahamic",
    "titles": ["Friend of God"]
  }
}
```

### Place Structure

```json
{
  "id": "place_jerusalem",
  "names": {
    "primary": "Jerusalem",
    "alternate": ["Jebus", "City of David", "Zion"],
    "languages": {
      "he": "יְרוּשָׁלַיִם",
      "ar": "القدس"
    }
  },
  "geography": {
    "coordinates": {
      "lat": 31.7683,
      "lon": 35.2137,
      "elevation_m": 760
    },
    "region": "Judean Mountains",
    "ancient_territory": "Judah"
  },
  "history": [
    {
      "period": "Jebusite",
      "dates": "~2000-1000 BCE",
      "verses": ["JOS.15.63"]
    },
    {
      "period": "Davidic",
      "dates": "~1000-960 BCE",
      "verses": ["2SA.5.6-10"]
    }
  ],
  "significance": {
    "religious": ["temple_site", "holy_city"],
    "political": ["capital_united_kingdom", "capital_judah"]
  }
}
```

### Word Definition Structure

```json
{
  "id": "H7225",
  "lemma": "רֵאשִׁית",
  "transliteration": "reshit",
  "pronunciation": {
    "ipa": "/reːˈʃiːt/",
    "simplified": "ray-SHEET"
  },
  "definitions": [
    {
      "gloss": "beginning, first",
      "expanded": "The first or chief part; temporal beginning",
      "contexts": ["temporal", "primacy", "firstfruits"]
    }
  ],
  "etymology": {
    "root": "ראש",
    "root_meaning": "head",
    "development": "head -> first -> beginning"
  },
  "usage": {
    "frequency": 51,
    "books": ["GEN", "EXO", "LEV", "..."],
    "collocations": ["בְּרֵאשִׁית", "רֵאשִׁית חָכְמָה"]
  },
  "morphology": {
    "part_of_speech": "noun",
    "gender": "feminine",
    "patterns": ["קְטִילָה"]
  },
  "semantic": {
    "domains": ["time", "order", "priority"],
    "concepts": ["beginning", "first", "origin", "start"],
    "lxx_equivalents": ["ἀρχή", "ἀπαρχή"]
  }
}
```

### Semantic Concept Structure

To enable topical searching across languages, the format includes semantic concept mappings:

```json
{
  "id": "concept_love",
  "names": {
    "en": ["love", "loving-kindness", "charity", "affection"],
    "es": ["amor", "caridad", "afecto"],
    "fr": ["amour", "charité", "affection"]
  },
  "types": {
    "divine_love": {
      "description": "God's love for humanity",
      "words": {
        "hebrew": ["H2617", "H157", "H160"],  // hesed, ahab, ahabah
        "greek": ["G26", "G5368"]              // agape, phileo
      }
    },
    "covenant_love": {
      "description": "Steadfast, loyal love",
      "words": {
        "hebrew": ["H2617"],  // hesed
        "greek": ["G26"]      // agape
      }
    },
    "romantic_love": {
      "description": "Romantic or passionate love",
      "words": {
        "hebrew": ["H1730", "H5690"],  // dod, agab
        "greek": ["G2037"]              // eros (in LXX)
      }
    },
    "friendship_love": {
      "description": "Brotherly love, friendship",
      "words": {
        "hebrew": ["H7474"],   // rea
        "greek": ["G5368"]     // phileo
      }
    }
  },
  "search_guidance": {
    "en": "The concept of 'love' in the Bible encompasses multiple distinct words. Consider searching for specific types: God's love (hesed), covenant faithfulness (hesed), romantic love (Song of Songs), or brotherly love (philadelphia).",
    "prompts": [
      "Are you looking for God's love for people?",
      "Are you interested in commands to love others?",
      "Are you studying marital/romantic love?",
      "Are you researching covenant faithfulness?"
    ]
  },
  "key_verses": {
    "divine_love": ["JOH.3.16", "ROM.5.8", "1JO.4.8"],
    "covenant_love": ["PSA.136.1", "LAM.3.22-23"],
    "love_commands": ["MAT.22.37-39", "JOH.13.34", "1CO.13.1-13"]
  }
}
```

#### Key Annotation Fields

The `annotations` object contains automatically generated linguistic and statistical data:

- **lexical_domains**: Semantic domains from standard lexicons
- **word_frequencies**: Statistical measures including hapax legomena and TF-IDF
- **linguistic_features**: Syntactic patterns and discourse markers
- **semantic_fields**: Concept groupings based on lexical data
- **intertextuality**: Quotations, allusions, and shared vocabulary
- **text_critical**: Manuscript variants and ancient versions

For detailed information on how these annotations are generated, see [AUTOMATIC_ANNOTATIONS.md](./AUTOMATIC_ANNOTATIONS.md).


### Automatic Annotation Generation

ABBA automatically generates high-confidence annotations from objectively computable data. See [AUTOMATIC_ANNOTATIONS.md](./AUTOMATIC_ANNOTATIONS.md) for detailed methodology.

### Search Implementation

The canonical format enables sophisticated searching across languages and concepts. See [SEARCH_METHODOLOGY.md](./SEARCH_METHODOLOGY.md) for search strategies and implementation.

## Validation Rules

### Required Fields
- Every verse must have a canonical_id
- Every verse must have at least one translation
- Cross-references must have valid source and target verses
- All IDs must be unique within their type

### ID Formats
- Verse: `BOOK.CHAPTER.VERSE[.PART]` (e.g., "GEN.1.1", "ROM.3.23a")
- Person: `person_[name]` (e.g., "person_abraham")
- Place: `place_[name]` (e.g., "place_jerusalem")
- Topic: `topic_[hierarchy]` (e.g., "topic_theology_salvation")

### Data Integrity
- All cross-references must be bidirectional
- Timeline events must have associated verses
- Morphology must match the source language
- Confidence scores must be between 0.0 and 1.0

## Extensibility

### Custom Fields
Extensions can add fields prefixed with `x_`:
```json
{
  "canonical_id": "GEN.1.1",
  "x_denomination_notes": {
    "catholic": "...",
    "orthodox": "..."
  }
}
```

### Version Management
- Format version follows semantic versioning
- Backward compatibility maintained within major versions
- Migration tools provided for major version upgrades

## Best Practices

1. **Completeness over Compression**: Include all available data
2. **Source Attribution**: Always indicate data sources
3. **Confidence Levels**: Use confidence scores for disputed data
4. **Multiple Perspectives**: Include traditional and critical views
5. **Language Neutrality**: Store language-specific content separately
6. **Scholarly Integrity**: Maintain academic standards

## Tools and Utilities

```bash
# Validate canonical format
abba validate canonical-data.json

# Convert to other formats
abba convert canonical-data.json --format sqlite --output abba.db
abba convert canonical-data.json --format static --output static/

# Merge multiple sources
abba merge source1.json source2.json --output merged.json

# Generate diff between versions
abba diff v1.0.json v1.1.json
```