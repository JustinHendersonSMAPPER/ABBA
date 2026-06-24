"""Pydantic response models for the ABBA FastAPI layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .constants import DEFAULT_TRANSLATION_ID


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
    source: Optional[str] = None


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


class WordExplanation(BaseModel):
    """Plain-English explanation of what the original adds beyond translation."""

    strongs_number: str
    language: str
    explanation: str


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


class ProvenanceRecord(BaseModel):
    """Auditable attribution record exposed for public scrutiny."""

    entity_type: str
    entity_id: str
    source: str
    source_detail: Optional[str] = None
    trust_tier: str
    trust_rationale: str
    generated_by: Optional[str] = None
    grounding: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    pipeline_version: str


# --- Speaker Attribution Models ---


class SpeakerAttribution(BaseModel):
    """Who is speaking in a quoted passage."""

    speaker: str
    context_note: Optional[str] = None


# --- Genre Shift Models ---


class GenreShift(BaseModel):
    """A genre transition within a book."""

    chapter: int
    verse: int
    from_genre: str
    to_genre: str
    description: Optional[str] = None


# --- Verse Response Models ---


class VerseContext(BaseModel):
    """Surrounding verses for anti-proof-texting context."""

    previous_verse: Optional[str] = None
    next_verse: Optional[str] = None


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
    surrounding_context: Optional[VerseContext] = None
    speaker: Optional[SpeakerAttribution] = None
    genre: Optional[str] = None
    is_descriptive: Optional[bool] = None

    # Scholarly depth
    parallel_passages: Optional[List[Dict[str, Any]]] = None
    manuscript_variants: Optional[List[ManuscriptVariant]] = None
    syntax_tree: Optional[VerseSyntaxTree] = None
    discourse_units: Optional[List[DiscourseUnit]] = None
    semantic_domains: Optional[List[SemanticDomainMapping]] = None


# --- Translation Comparison Models ---


class TranslationComparison(BaseModel):
    """Comparison of a verse across multiple translations."""

    reference: str
    translations: Dict[str, str]
    original_words: List[WordDetail] = Field(default_factory=list)
    divergences: Optional[List[Dict[str, Any]]] = None


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


class SemanticSearchResult(BaseModel):
    """Result from a semantic or hybrid search."""

    book_id: int
    chapter: int
    verse: int
    text: str = ""
    book_name: str = ""
    score: float = 0.0
    match_type: str = ""
    explanation: str = ""
    translation_id: str = ""


# --- Life Topic Models ---


class LifeTopicDetail(BaseModel):
    """Full life topic with study steps."""

    slug: str
    name: str
    category: str
    description: Optional[str] = None
    icon: Optional[str] = None
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    study_steps: List[Dict[str, Any]] = Field(default_factory=list)


class LifeTopicSummary(BaseModel):
    """Summary of a life topic for listings."""

    slug: str
    name: str
    category: str
    description: Optional[str] = None
    icon: Optional[str] = None


# --- Reading Plan Models ---


class ReadingPlanSummary(BaseModel):
    """Summary of a reading plan."""

    slug: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    estimated_days: int = 0


class ReadingPlanEntry(BaseModel):
    """A single day's entry in a reading plan."""

    day_number: int
    book_id: int
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int
    title: Optional[str] = None
    reflection_question: Optional[str] = None


class ReadingPlanDetail(BaseModel):
    """Full reading plan with entries."""

    slug: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    estimated_days: int = 0
    entries: List[ReadingPlanEntry] = Field(default_factory=list)


# --- User Annotation Models ---


class NoteCreate(BaseModel):
    """Request body for creating a note."""

    content: str
    note_type: str = "personal"


class NoteResponse(BaseModel):
    """A verse note."""

    note_id: int
    book_id: int
    chapter: int
    verse: int
    content: str
    note_type: str = "personal"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CollectionCreate(BaseModel):
    """Request body for creating a collection."""

    name: str
    description: str = ""


class CollectionResponse(BaseModel):
    """A user collection."""

    collection_id: int
    name: str
    description: str = ""
    created_at: Optional[str] = None
    verse_count: int = 0


class CollectionItemAdd(BaseModel):
    """Request body for adding a verse to a collection."""

    book_id: int
    chapter: int
    verse: int
    note: str = ""


class ShareCreate(BaseModel):
    """Request body for creating a shared item."""

    share_type: str
    title: str = ""
    content: Dict[str, Any] = Field(default_factory=dict)


class ShareResponse(BaseModel):
    """A shared item."""

    share_token: str
    share_type: str
    title: str = ""
    content: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


# --- Phase 9: Semantic Domain Models ---


class SemanticDomain(BaseModel):
    """A Louw-Nida semantic domain classification."""

    domain_code: str
    domain_name: str
    parent_domain: Optional[str] = None
    description: Optional[str] = None
    level: int = 1


class SemanticDomainMapping(BaseModel):
    """Mapping of a Strong's number to a semantic domain."""

    strongs_number: str
    domain_code: str
    domain_name: str
    confidence: float = 0.9


class WordDomainResult(BaseModel):
    """A word with its semantic domain info for word study."""

    strongs_number: str
    original_word: Optional[str] = None
    gloss: Optional[str] = None
    domains: List[SemanticDomain] = Field(default_factory=list)
    related_words: List[Dict[str, Any]] = Field(default_factory=list)


# --- Phase 9: Syntax Tree Models ---


class SyntaxNode(BaseModel):
    """A node in a clause-level syntax tree."""

    node_id: str
    node_type: str  # sentence, clause, phrase, word
    role: Optional[str] = None  # subject, predicate, object, modifier
    clause_type: Optional[str] = None  # main, temporal, relative, causal
    relation: Optional[str] = None
    depth: int = 0
    text_content: Optional[str] = None
    children: List["SyntaxNode"] = Field(default_factory=list)
    word_num: Optional[int] = None


class VerseSyntaxTree(BaseModel):
    """Complete syntax tree for a verse."""

    book_id: int
    chapter: int
    verse: int
    root_nodes: List[SyntaxNode] = Field(default_factory=list)


# --- Phase 9: Discourse Annotation Models ---


class DiscourseUnit(BaseModel):
    """A discourse annotation from OpenText.org analysis."""

    discourse_id: int
    discourse_type: str  # narrative, argument, exposition, hymn, dialogue
    function_label: Optional[str] = None
    relation_to_context: Optional[str] = None  # continuation, contrast, cause, result
    description: Optional[str] = None
    prominence: int = 0
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int


# --- Phase 9: Manuscript Variant Models ---


class ManuscriptVariant(BaseModel):
    """A textual variant from manuscript tradition."""

    variant_id: int
    variant_type: str  # addition, omission, substitution, transposition
    base_text: Optional[str] = None
    variant_text: Optional[str] = None
    manuscripts: Optional[str] = None
    explanation: Optional[str] = None
    significance: str = "minor"  # major, minor, orthographic
    confidence: float = 0.8


# --- Phase 9: Community Contribution Models ---


class ContributionCreate(BaseModel):
    """Request body for creating a community contribution."""

    book_id: int
    chapter: Optional[int] = None
    verse: Optional[int] = None
    contribution_type: str  # cultural_context, historical_note, translation_note
    title: str
    content: str


class ContributionResponse(BaseModel):
    """A community contribution."""

    id: int
    book_id: int
    chapter: Optional[int] = None
    verse: Optional[int] = None
    contribution_type: str
    title: str
    content: str
    author_id: str = "anonymous"
    status: str = "pending"
    created_at: Optional[str] = None


class ContributionReviewCreate(BaseModel):
    """Request body for reviewing a contribution."""

    decision: str  # approve, reject, request_changes
    review_note: Optional[str] = None


# --- Phase 9: Concept Proposal Models ---


class ConceptProposalCreate(BaseModel):
    """Request body for proposing a concept change."""

    concept_name: str
    proposal_type: str  # new, edit, merge, delete
    description: str
    hebrew_terms: List[Dict[str, str]] = Field(default_factory=list)
    greek_terms: List[Dict[str, str]] = Field(default_factory=list)
    verse_mappings: List[str] = Field(default_factory=list)


class ConceptProposalResponse(BaseModel):
    """A concept proposal."""

    id: int
    concept_name: str
    proposed_by: str = "anonymous"
    proposal_type: str
    description: str
    hebrew_terms: List[Dict[str, str]] = Field(default_factory=list)
    greek_terms: List[Dict[str, str]] = Field(default_factory=list)
    verse_mappings: List[str] = Field(default_factory=list)
    status: str = "pending"
    created_at: Optional[str] = None


# --- Phase 9: Semantic Relationship Graph Models ---


class SemanticRelationship(BaseModel):
    """A relationship between two biblical concepts."""

    source_concept: str
    target_concept: str
    relationship_type: str  # synonym, antithetical, causal, enables, contrast
    weight: float = 1.0
    evidence_count: int = 0
    shared_strongs: List[str] = Field(default_factory=list)


class ConceptGraph(BaseModel):
    """A graph of related concepts for visualization."""

    center_concept: str
    relationships: List[SemanticRelationship] = Field(default_factory=list)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)


# --- Phase 9: Concept Discovery Models ---


class ConceptDiscoveryResult(BaseModel):
    """Result from natural-language concept discovery."""

    query: str
    matched_concepts: List[TopicSummary] = Field(default_factory=list)
    matched_life_topics: List[LifeTopicSummary] = Field(default_factory=list)
    suggested_searches: List[str] = Field(default_factory=list)


# --- Phase 9: Audio Integration Models ---


class AudioResource(BaseModel):
    """An audio resource for a passage."""

    book_id: int
    chapter: int
    verse_start: int = 1
    verse_end: Optional[int] = None
    audio_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    narrator: Optional[str] = None
    translation_id: str = DEFAULT_TRANSLATION_ID


# --- Phase 9: Mobile API Models ---


class MobileVerseResponse(BaseModel):
    """Optimized verse response for mobile clients."""

    ref: str
    text: str
    tid: str
    words: Optional[List[Dict[str, Any]]] = None
    flags: Optional[List[Dict[str, Any]]] = None


class MobileSyncRequest(BaseModel):
    """Request for syncing offline data."""

    last_sync: Optional[str] = None
    book_ids: List[int] = Field(default_factory=list)
    include_words: bool = True
    include_annotations: bool = False


class MobileSyncResponse(BaseModel):
    """Response with offline-ready data."""

    sync_timestamp: str
    verses: List[MobileVerseResponse] = Field(default_factory=list)
    total_verses: int = 0


# --- Export Models ---


class ExportRequest(BaseModel):
    """Request for exporting study data."""

    format: str = "json"  # "json" or "markdown"
    include_original_language: bool = True
    include_cross_references: bool = True
    include_cultural_context: bool = False


# --- Pagination Models ---


class PaginatedResponse(BaseModel):
    """Wrapper for paginated results."""

    items: List[Any] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50
    has_next: bool = False
    has_previous: bool = False


# --- API Info Model ---


class APIInfo(BaseModel):
    """API metadata returned at the root endpoint."""

    name: str = "ABBA Bible Study API"
    version: str = "0.1.0"
    description: str = "Annotated Bible and Background Analysis"
    docs_url: str = "/docs"


# --- Translation List Model ---


class TranslationInfo(BaseModel):
    """Basic metadata for a Bible translation."""

    id: str
    name: str
    language: str
    english_name: Optional[str] = None
