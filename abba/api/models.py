"""Pydantic response models for the ABBA FastAPI layer."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DepthLevel(str, Enum):
    """Controls how much data is returned in verse responses."""

    BASIC = "basic"  # Just the translated text
    STANDARD = "standard"  # Text + original language words + meaning-richness flags
    DEEP = "deep"  # + cultural context, cross-references, literary structure, concepts
    SCHOLARLY = "scholarly"  # + parallel passages, full lexical data, manuscript notes


# --- Word and Lexicon Models ---


class WordDetail(BaseModel):
    """A single word from the original language text."""

    word_num: int
    original_text: Optional[str] = None
    transliteration: Optional[str] = None
    english_gloss: Optional[str] = None
    strongs_number: Optional[str] = None
    morphology_code: Optional[str] = None
    morphology_description: Optional[str] = None
    part_of_speech: Optional[str] = None
    language: Optional[str] = None


class LexiconEntry(BaseModel):
    """Full lexicon entry for a Strong's number."""

    strongs_number: str
    original_word: Optional[str] = None
    transliteration: Optional[str] = None
    part_of_speech: Optional[str] = None
    gloss: Optional[str] = None
    definition: Optional[str] = None
    language: Optional[str] = None


class MorphologyInfo(BaseModel):
    """Morphology code description."""

    code: str
    description: Optional[str] = None
    components: Optional[str] = None


class WordAnalysis(BaseModel):
    """Complete analysis for a specific word in a verse."""

    word: WordDetail
    lexicon: Optional[LexiconEntry] = None
    morphology: Optional[MorphologyInfo] = None


# --- Meaning Richness Models ---


class RichnessFlag(BaseModel):
    """Indicates where an English translation loses meaning from the original."""

    word_num: int
    strongs_number: str
    original_word: Optional[str] = None
    english_gloss: Optional[str] = None
    richness_score: float = Field(ge=0.0, le=1.0)
    untranslatable_nuances: List[str] = Field(default_factory=list)
    full_definition: Optional[str] = None
    morphology_significance: Optional[str] = None


# --- Cultural Context Models ---


class CulturalNote(BaseModel):
    """Cultural or historical context for a passage."""

    context_id: int
    context_type: str
    title: str
    summary: str
    detailed_content: Optional[str] = None
    time_period: Optional[str] = None
    geographic_region: Optional[str] = None
    confidence: Optional[str] = None


# --- Cross-Reference Models ---


class CrossRef(BaseModel):
    """A cross-reference between two passages."""

    target_reference: str
    ref_type: str
    confidence: float = 0.8
    notes: Optional[str] = None


# --- Literary Structure Models ---


class LiteraryStructure(BaseModel):
    """A literary structure annotation (chiasmus, parallelism, etc.)."""

    structure_type: str
    description: Optional[str] = None
    significance: Optional[str] = None
    elements: List[Dict[str, Any]] = Field(default_factory=list)


class PassageInfo(BaseModel):
    """A coherent textual unit (pericope)."""

    passage_id: int
    title: str
    genre: Optional[str] = None
    literary_type: Optional[str] = None
    structural_features: List[str] = Field(default_factory=list)
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int


# --- Verse Response Models ---


class VerseResponse(BaseModel):
    """Response for a single verse, with progressive depth."""

    # Always returned (basic)
    reference: str
    book_name: str
    chapter: int
    verse: int
    text: str
    translation_id: str

    # Standard depth
    words: Optional[List[WordDetail]] = None
    richness_flags: Optional[List[RichnessFlag]] = None

    # Deep depth
    cultural_context: Optional[List[CulturalNote]] = None
    cross_references: Optional[List[CrossRef]] = None
    passage_info: Optional[PassageInfo] = None
    literary_structures: Optional[List[LiteraryStructure]] = None
    concepts: Optional[List[Dict[str, Any]]] = None

    # Scholarly depth
    parallel_passages: Optional[List[Dict[str, Any]]] = None


# --- Translation Comparison Models ---


class TranslationComparison(BaseModel):
    """Comparison of a verse across multiple translations."""

    reference: str
    translations: Dict[str, str]
    original_words: List[WordDetail] = Field(default_factory=list)


# --- Topical Study Models ---


class ThemeGroup(BaseModel):
    """A thematic grouping of verses within a concept."""

    theme_name: str
    description: Optional[str] = None
    verses: List[Dict[str, Any]] = Field(default_factory=list)
    key_insights: Optional[str] = None


class TopicSummary(BaseModel):
    """Summary of an available topic/concept."""

    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    verse_count: int = 0


class TopicalResult(BaseModel):
    """Full topical study result with themed verse groups."""

    concept_name: str
    concept_description: Optional[str] = None
    total_verses: int = 0
    hebrew_terms: List[Dict[str, str]] = Field(default_factory=list)
    greek_terms: List[Dict[str, str]] = Field(default_factory=list)
    theme_groups: List[ThemeGroup] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)


# --- Book and Passage Metadata Models ---


class BookInfo(BaseModel):
    """Metadata for a biblical book."""

    book_id: int
    name: str
    common_name: Optional[str] = None
    testament: Optional[str] = None
    chapter_count: int = 0
    primary_genre: Optional[str] = None
    secondary_genres: List[str] = Field(default_factory=list)
    author_traditional: Optional[str] = None
    date_range: Optional[str] = None
    original_audience: Optional[str] = None
    literary_features: List[str] = Field(default_factory=list)
    reading_context: Optional[str] = None
    canonical_section: Optional[str] = None
    passages: Optional[List[PassageInfo]] = None


# --- Search Result Models ---


class StrongsResult(BaseModel):
    """Result from a Strong's number search."""

    book: str
    chapter: int
    verse: int
    word_num: int
    original_text: Optional[str] = None
    transliteration: Optional[str] = None
    english_gloss: Optional[str] = None
    strongs_number: Optional[str] = None
    morphology_code: Optional[str] = None
    language: str


class TextSearchResult(BaseModel):
    """Result from a full-text search."""

    translation_id: str
    book_id: int
    chapter: int
    verse: int
    text: str
    book_name: Optional[str] = None


# --- API Info Model ---


class APIInfo(BaseModel):
    """API metadata returned at the root endpoint."""

    name: str = "ABBA Bible Study API"
    version: str = "0.1.0"
    description: str = "Annotated Bible and Background Analysis"
    docs_url: str = "/docs"
