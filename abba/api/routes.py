"""FastAPI route definitions for the ABBA Bible Study API."""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..config import ABBAConfig, ConfigManager
from ..database import SQLiteManager
from ..provenance import ProvenanceStore
from .analysis import AnalysisAPI
from .models import (
    APIInfo,
    AudioResource,
    BookInfo,
    CollectionCreate,
    CollectionItemAdd,
    CollectionResponse,
    ConceptDiscoveryResult,
    ConceptGraph,
    ConceptProposalCreate,
    ConceptProposalResponse,
    ContributionCreate,
    ContributionResponse,
    ContributionReviewCreate,
    CrossRef,
    CulturalNote,
    DepthLevel,
    DiscourseUnit,
    GenreShift,
    LexiconEntry,
    LifeTopicDetail,
    LifeTopicSummary,
    ManuscriptVariant,
    MobileSyncRequest,
    MobileSyncResponse,
    MobileVerseResponse,
    MorphologyInfo,
    NoteCreate,
    NoteResponse,
    PassageInfo,
    ProvenanceRecord,
    ReadingPlanDetail,
    ReadingPlanEntry,
    ReadingPlanSummary,
    RichnessFlag,
    SemanticDomain,
    SemanticDomainMapping,
    SemanticRelationship,
    SemanticSearchResult,
    ShareCreate,
    ShareResponse,
    SpeakerAttribution,
    StrongsResult,
    SyntaxNode,
    TextSearchResult,
    ThemeGroup,
    TopicalResult,
    TopicSummary,
    TranslationComparison,
    VerseContext,
    VerseResponse,
    VerseSyntaxTree,
    WordAnalysis,
    WordDetail,
    WordDomainResult,
    WordExplanation,
)
from .query_parser import parse_query
from .search import SearchAPI
from .semantic_search import SemanticSearchAPI

router = APIRouter(prefix="/api/v1", tags=["bible"])


# --- Singleton state container (avoids pylint global-statement) ---


class _AppState:
    """Holds lazily-initialized singleton instances for the API layer."""

    db_manager: Optional[SQLiteManager] = None
    search_api: Optional[SearchAPI] = None
    analysis_api: Optional[AnalysisAPI] = None
    provenance_store: Optional[ProvenanceStore] = None
    semantic_api: Optional[SemanticSearchAPI] = None
    semantic_unavailable: bool = False


_state = _AppState()


def _get_db() -> SQLiteManager:
    """Get or create the database manager singleton."""
    if _state.db_manager is None:
        config_mgr = ConfigManager()
        config: ABBAConfig = config_mgr.get_config()
        db_path = config.abba_db_path
        if not db_path.exists():
            raise HTTPException(status_code=503, detail="Database not initialized. Run the import pipeline first.")
        _state.db_manager = SQLiteManager(db_path)
    return _state.db_manager


def _get_search() -> SearchAPI:
    """Get or create the SearchAPI singleton."""
    if _state.search_api is None:
        _state.search_api = SearchAPI(_get_db())
    return _state.search_api


def _get_analysis() -> AnalysisAPI:
    """Get or create the AnalysisAPI singleton."""
    if _state.analysis_api is None:
        _state.analysis_api = AnalysisAPI(_get_db())
    return _state.analysis_api


def _get_provenance() -> ProvenanceStore:
    """Get or create the ProvenanceStore singleton."""
    if _state.provenance_store is None:
        _state.provenance_store = ProvenanceStore(_get_db())
    return _state.provenance_store


def _get_semantic() -> Optional[SemanticSearchAPI]:
    """Build the semantic search API lazily.

    Returns None (and remembers the failure) if ChromaDB vectors or the
    embedding models are unavailable, so callers degrade gracefully to FTS.
    """
    if _state.semantic_api is None and not _state.semantic_unavailable:
        try:
            from ..embeddings.chroma_manager import ChromaManager
            from ..embeddings.model_manager import EmbeddingModelManager

            config = ConfigManager().get_config()
            chroma = ChromaManager(persist_path=str(config.vectors_path))
            models = EmbeddingModelManager(cache_dir=str(config.data_dir / "models"))
            _state.semantic_api = SemanticSearchAPI(_get_db(), chroma, models)
        except Exception:  # noqa: BLE001 - any failure means "degrade to FTS"
            _state.semantic_unavailable = True
            return None
    return _state.semantic_api


def _fts_only_semantic_results(search_text: str, translation_id: str, limit: int) -> List[SemanticSearchResult]:
    """Full-text fallback when semantic search is unavailable."""
    db = _get_db()
    results: List[SemanticSearchResult] = []
    try:
        fts_rows = db.search_verses(translation_id, search_text, limit * 2)
        for rank, row in enumerate(fts_rows):
            row_keys = row.keys() if hasattr(row, "keys") else []
            results.append(
                SemanticSearchResult(
                    book_id=row["book_id"],
                    chapter=row["chapter"],
                    verse=row["verse"],
                    text=row["text"],
                    book_name=row["book_name"] if "book_name" in row_keys else "",
                    score=round(1.0 - (rank / max(len(fts_rows), 1)), 3),
                    match_type="exact",
                    explanation=f"Text match (rank {rank + 1})",
                    translation_id=translation_id,
                )
            )
    except Exception:  # noqa: BLE001, S110 - best-effort fallback; failure is non-fatal
        pass
    return results


def configure_db(db_manager: SQLiteManager) -> None:
    """Configure the routes with an existing database manager.

    Args:
        db_manager: Pre-initialized SQLiteManager instance.
    """
    _state.db_manager = db_manager
    _state.search_api = SearchAPI(db_manager)
    _state.analysis_api = AnalysisAPI(db_manager)
    _state.provenance_store = ProvenanceStore(db_manager)


# --- Root ---


@router.get("/", response_model=APIInfo, tags=["info"])
async def api_root() -> APIInfo:
    """Return API metadata."""
    return APIInfo()


# --- Provenance Endpoints ---


@router.get("/provenance/export", response_model=List[ProvenanceRecord], tags=["provenance"])
async def export_provenance() -> List[ProvenanceRecord]:
    """Export every provenance record for public scrutiny."""
    store = _get_provenance()
    return [ProvenanceRecord(**rec) for rec in store.export_all()]


@router.get("/provenance/{entity_type}/{entity_id}", response_model=ProvenanceRecord, tags=["provenance"])
async def get_provenance(entity_type: str, entity_id: str) -> ProvenanceRecord:
    """Return the audit record for one enrichment element."""
    prov = _get_provenance().get(entity_type, entity_id)
    if prov is None:
        raise HTTPException(status_code=404, detail="No provenance record found")
    return ProvenanceRecord(
        entity_type=prov.entity_type,
        entity_id=prov.entity_id,
        source=prov.source,
        source_detail=prov.source_detail,
        trust_tier=prov.trust_tier.value,
        trust_rationale=prov.trust_rationale,
        generated_by=prov.generated_by,
        grounding=prov.grounding,
        confidence=prov.confidence,
        pipeline_version=prov.pipeline_version,
    )


# --- Verse Endpoints ---


@router.get("/verses/{translation_id}/{book_id}/{chapter}/{verse}", response_model=VerseResponse)
async def get_verse(
    translation_id: str,
    book_id: int,
    chapter: int,
    verse: int,
    depth: DepthLevel = Query(DepthLevel.BASIC, description="Level of detail to return"),
) -> VerseResponse:
    """Get a single verse with progressive data depth.

    - **basic**: just the translated text
    - **standard**: text + original language words + meaning-richness flags
    - **deep**: + cultural context, cross-references, literary structure, concepts
    - **scholarly**: + parallel passages, full lexical data
    """
    search = _get_search()
    result = search.get_verse(translation_id, book_id, chapter, verse)
    if not result:
        raise HTTPException(status_code=404, detail="Verse not found")

    # Resolve the book name from the books table (words table uses names like "Gen", not IDs)
    book_name = _resolve_book_name(book_id, translation_id)

    response = VerseResponse(
        reference=f"{book_name or book_id} {chapter}:{verse}",
        book_name=book_name or str(book_id),
        chapter=chapter,
        verse=verse,
        text=result.text,
        translation_id=translation_id,
    )

    if depth in (DepthLevel.STANDARD, DepthLevel.DEEP, DepthLevel.SCHOLARLY):
        cached = _try_annotation_cache(book_id, chapter, verse, depth, response, translation_id, book_name)
        if not cached:
            response.words = _get_words_for_verse(book_name or str(book_id), chapter, verse)
            response.richness_flags = _get_richness_flags(book_name or str(book_id), chapter, verse)

            if depth in (DepthLevel.DEEP, DepthLevel.SCHOLARLY):
                response.cross_references = _get_cross_refs(book_id, chapter, verse)
                response.cultural_context = _get_cultural_context(book_id, chapter, verse)
                response.passage_info = _get_passage_info(book_id, chapter, verse)
                response.literary_structures = _get_literary_structures(book_id, chapter, verse)
                response.concepts = []
                response.surrounding_context = _get_surrounding_context(translation_id, book_id, chapter, verse)
                response.speaker = _get_speaker(book_id, chapter, verse)
                response.genre = _get_active_genre(book_id, chapter, verse)
                if response.genre in ("narrative", "unknown"):
                    response.is_descriptive = True

    if depth == DepthLevel.SCHOLARLY:
        analysis = _get_analysis()
        parallels = analysis.parallel_passage_detection(book_name or str(book_id), chapter, verse)
        response.parallel_passages = parallels
        response.manuscript_variants = _get_manuscript_variants(book_id, chapter, verse)
        response.syntax_tree = _get_syntax_tree(book_id, chapter, verse)
        response.discourse_units = _get_discourse_units(book_id, chapter, verse)
        response.semantic_domains = _get_semantic_domains_for_verse(book_name or str(book_id), chapter, verse)

    return response


@router.get("/verses/{translation_id}/{book_id}/{chapter}", response_model=List[VerseResponse])
async def get_chapter(
    translation_id: str,
    book_id: int,
    chapter: int,
    depth: DepthLevel = Query(DepthLevel.BASIC, description="Level of detail to return"),
) -> List[VerseResponse]:
    """Get all verses in a chapter."""
    _ = depth  # Reserved for future per-verse enrichment at standard/deep/scholarly levels
    db = _get_db()
    rows = db.execute_query(
        "SELECT verse, text FROM verses WHERE translation_id = ? AND book_id = ? AND chapter = ? ORDER BY verse",
        (translation_id, book_id, chapter),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Chapter not found")

    results = []
    for row in rows:
        v_num, text = row[0], row[1]
        resp = VerseResponse(
            reference=f"{book_id} {chapter}:{v_num}",
            book_name=str(book_id),
            chapter=chapter,
            verse=v_num,
            text=text,
            translation_id=translation_id,
        )
        results.append(resp)
    return results


# --- Translation Comparison ---


@router.get("/compare/{book}/{chapter}/{verse}", response_model=TranslationComparison)
async def compare_translations(
    book: str,
    chapter: int,
    verse: int,
    translations: List[str] = Query(..., description="Translation IDs to compare", min_length=2),
) -> TranslationComparison:
    """Compare a verse across multiple translations with original language data."""
    analysis = _get_analysis()
    result = analysis.compare_translations(book, chapter, verse, translations)
    words = [
        WordDetail(
            word_num=w["position"],
            original_text=w.get("text"),
            transliteration=w.get("transliteration"),
            english_gloss=w.get("translation"),
            strongs_number=w.get("strongs"),
            morphology_code=w.get("morphology"),
            language=w.get("language"),
        )
        for w in result.get("original_words", [])
    ]
    divergences = _detect_translation_divergences(result.get("translations", {}))
    return TranslationComparison(
        reference=result["reference"],
        translations=result.get("translations", {}),
        original_words=words,
        divergences=divergences,
    )


# --- Search Endpoints ---


@router.get("/search/text", response_model=List[TextSearchResult])
async def text_search(
    q: str = Query(..., description="Text search query"),
    translation_id: str = Query("engbsb", description="Translation ID"),
    limit: int = Query(50, ge=1, le=200),
    page: int = Query(1, ge=1, description="Page number for pagination"),
) -> List[TextSearchResult]:
    """Full-text search within a specific translation. Supports pagination."""
    search = _get_search()
    offset = (page - 1) * limit
    results = search.search_verses(translation_id, q, limit + offset)
    paginated = results[offset : offset + limit]
    return [
        TextSearchResult(
            translation_id=r.translation_id,
            book_id=r.book_id,
            chapter=r.chapter,
            verse=r.verse,
            text=r.text,
            book_name=r.book_name,
        )
        for r in paginated
    ]


@router.get("/search/strongs/{strongs_number}", response_model=List[StrongsResult])
async def search_by_strongs(
    strongs_number: str,
    limit: int = Query(100, ge=1, le=500),
) -> List[StrongsResult]:
    """Find all occurrences of a specific Strong's number."""
    search = _get_search()
    results = search.search_strongs(strongs_number)
    return [
        StrongsResult(
            book=r.book,
            chapter=r.chapter,
            verse=r.verse,
            word_num=r.word_num,
            original_text=r.hebrew_text or r.greek_text,
            transliteration=r.transliteration,
            english_gloss=r.translation,
            strongs_number=r.strongs_primary,
            morphology_code=r.morphology_code,
            language=r.language,
        )
        for r in results[:limit]
    ]


# --- Lexicon Endpoints ---


@router.get("/lexicon/{strongs_number}", response_model=LexiconEntry)
async def get_lexicon_entry(strongs_number: str) -> LexiconEntry:
    """Get full lexicon entry for a Strong's number."""
    db = _get_db()
    row = db.get_lexicon_entry(strongs_number)
    if not row:
        raise HTTPException(status_code=404, detail=f"Lexicon entry not found for {strongs_number}")
    return LexiconEntry(
        strongs_number=row["strongs_number"],
        original_word=row["original_word"],
        transliteration=row["transliteration"],
        part_of_speech=row["part_of_speech"],
        gloss=row["gloss"],
        definition=row["definition"],
        language=row["language"],
    )


# --- Word Analysis Endpoints ---


@router.get("/words/{book}/{chapter}/{verse}/{word_num}", response_model=WordAnalysis)
async def get_word_detail(
    book: str,
    chapter: int,
    verse: int,
    word_num: int,
) -> WordAnalysis:
    """Get complete analysis for a specific word in a specific verse."""
    search = _get_search()
    result = search.get_word_analysis(book, chapter, verse, word_num)
    if not result:
        raise HTTPException(status_code=404, detail="Word not found")

    word_data = result["word"]
    word = WordDetail(
        word_num=word_data["word_num"],
        original_text=word_data.get("hebrew_text") or word_data.get("greek_text"),
        transliteration=word_data.get("transliteration"),
        english_gloss=word_data.get("translation"),
        language=word_data.get("language"),
    )

    lexicon = None
    if result.get("lexicon"):
        lex = result["lexicon"]
        lexicon = LexiconEntry(
            strongs_number=lex["strongs_number"],
            original_word=lex.get("original_word"),
            transliteration=lex.get("transliteration"),
            part_of_speech=lex.get("part_of_speech"),
            gloss=lex.get("gloss"),
            definition=lex.get("definition"),
        )

    morph = None
    if result.get("morphology"):
        m = result["morphology"]
        morph = MorphologyInfo(
            code=m["code"],
            description=m.get("description"),
            components=m.get("components"),
        )

    return WordAnalysis(word=word, lexicon=lexicon, morphology=morph)


# --- Topic / Concept Endpoints ---


@router.get("/topics", response_model=List[TopicSummary])
async def list_topics() -> List[TopicSummary]:
    """List all available concepts with summary info."""
    db = _get_db()
    rows = db.execute_query(
        """
        SELECT cd.concept_id, cd.name, cd.description,
               COUNT(cvm.verse_id) as verse_count
        FROM concept_definitions cd
        LEFT JOIN concept_verse_mappings cvm ON cd.concept_id = cvm.concept_id
        GROUP BY cd.concept_id
        ORDER BY cd.name
        """,
    )
    return [
        TopicSummary(
            name=row[1] or row[0],
            description=row[2],
            verse_count=row[3],
        )
        for row in rows
    ]


@router.get("/topics/{concept_name}", response_model=TopicalResult)
async def get_concept(
    concept_name: str,
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=500),
) -> TopicalResult:
    """Get a concept with all its associated verses."""
    db = _get_db()

    # Get concept definition
    concept_rows = db.execute_query(
        "SELECT concept_id, name, description, hebrew_terms, greek_terms FROM concept_definitions WHERE name = ?",
        (concept_name,),
    )
    if not concept_rows:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")

    row = concept_rows[0]
    concept_id = row[0]

    # Get mapped verses
    verse_rows = db.execute_query(
        """
        SELECT verse_id, validation_method, confidence_score, validation_reason
        FROM concept_verse_mappings
        WHERE concept_id = ? AND confidence_score >= ?
        ORDER BY confidence_score DESC
        LIMIT ?
        """,
        (concept_id, min_confidence, limit),
    )

    verses = [
        {
            "verse_id": vr[0],
            "validation_method": vr[1],
            "confidence_score": vr[2],
            "validation_reason": vr[3],
        }
        for vr in verse_rows
    ]

    return TopicalResult(
        concept_name=row[1] or concept_name,
        concept_description=row[2],
        total_verses=len(verses),
        theme_groups=(
            [
                ThemeGroup(
                    theme_name="All verses",
                    verses=verses,
                )
            ]
            if verses
            else []
        ),
    )


# --- Book Metadata Endpoints ---


@router.get("/books", response_model=List[BookInfo])
async def list_books() -> List[BookInfo]:
    """List all biblical books with metadata."""
    db = _get_db()
    rows = db.execute_query(
        "SELECT book_id, name, common_name, number_of_chapters, testament FROM books ORDER BY book_order",
    )
    results = []
    for row in rows:
        book = BookInfo(
            book_id=row[0],
            name=row[1],
            common_name=row[2],
            chapter_count=row[3] or 0,
            testament=row[4],
        )
        # Enrich with book_metadata if table exists
        _enrich_book_metadata(db, book)
        results.append(book)
    return results


@router.get("/books/{book_id}", response_model=BookInfo)
async def get_book_info(book_id: int) -> BookInfo:
    """Get metadata for a specific biblical book."""
    db = _get_db()
    rows = db.execute_query(
        "SELECT book_id, name, common_name, number_of_chapters, testament FROM books WHERE book_id = ?",
        (book_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Book not found")

    row = rows[0]
    book = BookInfo(
        book_id=row[0],
        name=row[1],
        common_name=row[2],
        chapter_count=row[3] or 0,
        testament=row[4],
    )
    _enrich_book_metadata(db, book)
    return book


# --- Analysis Endpoints ---


@router.get("/analysis/morphology", response_model=List[Dict[str, Any]])
async def analyze_morphology(
    language: str = Query("hebrew", description="Language: hebrew or greek"),
    pattern: Optional[str] = Query(None, description="Morphology pattern filter"),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """Analyze morphological patterns in biblical texts."""
    analysis = _get_analysis()
    results = analysis.analyze_morphology_patterns(language, pattern, limit)
    return [
        {
            "pattern": r.pattern,
            "description": r.description,
            "count": r.count,
            "examples": r.examples,
        }
        for r in results
    ]


@router.get("/analysis/frequency", response_model=List[Dict[str, Any]])
async def word_frequency(
    strongs_pattern: Optional[str] = Query(None, description="Strong's pattern filter, e.g. H% for Hebrew"),
    min_frequency: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Analyze word frequency across the biblical corpus."""
    analysis = _get_analysis()
    results = analysis.word_frequency_analysis(strongs_pattern, min_frequency, limit)
    return [
        {
            "word": r.word,
            "strongs_number": r.strongs_number,
            "frequency": r.frequency,
            "books": sorted(r.books),
            "first_occurrence": r.first_occurrence,
            "last_occurrence": r.last_occurrence,
        }
        for r in results
    ]


@router.get("/analysis/semantic-domain/{domain}", response_model=List[Dict[str, Any]])
async def semantic_domain(domain: str) -> List[Dict[str, Any]]:
    """Analyze words belonging to a semantic domain."""
    analysis = _get_analysis()
    return analysis.semantic_domain_analysis(domain)


# --- Semantic Search ---


@router.get("/search/semantic", response_model=List[SemanticSearchResult])
async def semantic_search(
    q: str = Query(..., description="Natural language search query"),
    translation_id: str = Query("engbsb", description="Translation for text display"),
    limit: int = Query(20, ge=1, le=100),
    testament: Optional[str] = Query(None, description="Filter: 'old' or 'new'"),
    book_id: Optional[int] = Query(None, description="Filter by book ID"),
) -> List[SemanticSearchResult]:
    """Search using natural language — combines exact text matching with semantic similarity.

    Supports structured query syntax:
    - ``love in:john`` — filter by book
    - ``grace testament:new`` — filter by testament
    - ``"living water"`` — exact phrase
    """
    parsed = parse_query(q)
    testament_override = testament or parsed.testament_filter
    book_override = book_id or parsed.book_filter
    search_text = parsed.text or q

    semantic = _get_semantic()
    if semantic is not None:
        hybrid = semantic.hybrid_search(search_text, translation_id=translation_id, n_results=limit * 2)
        results = [
            SemanticSearchResult(
                book_id=h.book_id,
                chapter=h.chapter,
                verse=h.verse,
                text=h.text,
                book_name=h.book_name,
                score=round(h.score, 3),
                match_type=h.match_type,
                explanation=h.explanation,
                translation_id=h.translation_id,
            )
            for h in hybrid
        ]
    else:
        results = _fts_only_semantic_results(search_text, translation_id, limit)

    # Apply filters
    if testament_override:
        t = "old" if testament_override in ("old", "ot") else "new"
        allowed = set(range(1, 40)) if t == "old" else set(range(40, 67))
        results = [r for r in results if r.book_id in allowed]

    if book_override:
        results = [r for r in results if r.book_id == book_override]

    return results[:limit]


# --- Life Topics Endpoints ---


@router.get("/life-topics", response_model=List[LifeTopicSummary])
async def list_life_topics() -> List[LifeTopicSummary]:
    """List all life topics for everyday topical access to Scripture."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT slug, name, category, description, icon FROM life_topics ORDER BY display_order"
        )
        return [LifeTopicSummary(slug=r[0], name=r[1], category=r[2], description=r[3], icon=r[4]) for r in rows]
    except sqlite3.OperationalError:
        return []


@router.get("/life-topics/search", response_model=List[LifeTopicSummary])
async def search_life_topics(
    q: str = Query(..., description="Search query for topics"),
) -> List[LifeTopicSummary]:
    """Search life topics by name, category, or description."""
    db = _get_db()
    try:
        pattern = f"%{q}%"
        rows = db.execute_query(
            "SELECT slug, name, category, description, icon FROM life_topics "
            "WHERE name LIKE ? OR description LIKE ? OR category LIKE ? "
            "ORDER BY display_order",
            (pattern, pattern, pattern),
        )
        return [LifeTopicSummary(slug=r[0], name=r[1], category=r[2], description=r[3], icon=r[4]) for r in rows]
    except sqlite3.OperationalError:
        return []


@router.get("/life-topics/{slug}", response_model=LifeTopicDetail)
async def get_life_topic(slug: str) -> LifeTopicDetail:
    """Get a life topic with its study steps and concept links."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT id, slug, name, category, description, icon FROM life_topics WHERE slug = ?",
            (slug,),
        )
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail="Life topics not available") from exc

    if not rows:
        raise HTTPException(status_code=404, detail=f"Topic '{slug}' not found")

    row = rows[0]
    topic_id = row[0]

    concepts: List[Dict[str, Any]] = []
    try:
        concept_rows = db.execute_query(
            "SELECT concept_name, relevance_aspect FROM life_topic_concepts WHERE topic_id = ? ORDER BY display_order",
            (topic_id,),
        )
        concepts = [{"concept_name": c[0], "relevance_aspect": c[1]} for c in concept_rows]
    except sqlite3.OperationalError:
        pass

    steps: List[Dict[str, Any]] = []
    try:
        step_rows = db.execute_query(
            "SELECT step_order, step_type, verse_reference, insight FROM topic_study_steps "
            "WHERE topic_id = ? ORDER BY step_order",
            (topic_id,),
        )
        steps = [{"step_order": s[0], "step_type": s[1], "verse_reference": s[2], "insight": s[3]} for s in step_rows]
    except sqlite3.OperationalError:
        pass

    return LifeTopicDetail(
        slug=row[1],
        name=row[2],
        category=row[3],
        description=row[4],
        icon=row[5],
        concepts=concepts,
        study_steps=steps,
    )


# --- Passages / Pericope Endpoints ---


@router.get("/passages/{book_id}/{chapter}", response_model=List[PassageInfo])
async def get_passages(book_id: int, chapter: int) -> List[PassageInfo]:
    """Get passage/pericope boundaries for a chapter."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT passage_id, title, genre, literary_type, structural_features, "
            "start_chapter, start_verse, end_chapter, end_verse "
            "FROM passages "
            "WHERE book_id = ? AND start_chapter <= ? AND end_chapter >= ? "
            "ORDER BY display_order",
            (book_id, chapter, chapter),
        )
    except sqlite3.OperationalError:
        return []

    return [
        PassageInfo(
            passage_id=r[0],
            title=r[1],
            genre=r[2],
            literary_type=r[3],
            structural_features=_parse_json_list(r[4]),
            start_chapter=r[5],
            start_verse=r[6],
            end_chapter=r[7],
            end_verse=r[8],
        )
        for r in rows
    ]


# --- Reading Plan Endpoints ---


@router.get("/reading-plans", response_model=List[ReadingPlanSummary])
async def list_reading_plans() -> List[ReadingPlanSummary]:
    """List all available reading plans."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT slug, name, description, category, estimated_days FROM reading_plans ORDER BY slug"
        )
        return [
            ReadingPlanSummary(
                slug=r[0],
                name=r[1],
                description=r[2],
                category=r[3],
                estimated_days=r[4] or 0,
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


@router.get("/reading-plans/{slug}", response_model=ReadingPlanDetail)
async def get_reading_plan(slug: str) -> ReadingPlanDetail:
    """Get a reading plan with all daily entries."""
    db = _get_db()
    try:
        plan_rows = db.execute_query(
            "SELECT slug, name, description, category, estimated_days FROM reading_plans WHERE slug = ?",
            (slug,),
        )
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail="Reading plans not available") from exc

    if not plan_rows:
        raise HTTPException(status_code=404, detail=f"Reading plan '{slug}' not found")

    plan = plan_rows[0]
    entries: List[ReadingPlanEntry] = []
    try:
        entry_rows = db.execute_query(
            "SELECT day_number, book_id, start_chapter, start_verse, "
            "end_chapter, end_verse, title, reflection_question "
            "FROM reading_plan_entries WHERE plan_slug = ? ORDER BY day_number",
            (slug,),
        )
        entries = [
            ReadingPlanEntry(
                day_number=e[0],
                book_id=e[1],
                start_chapter=e[2],
                start_verse=e[3],
                end_chapter=e[4],
                end_verse=e[5],
                title=e[6],
                reflection_question=e[7],
            )
            for e in entry_rows
        ]
    except sqlite3.OperationalError:
        pass

    return ReadingPlanDetail(
        slug=plan[0],
        name=plan[1],
        description=plan[2],
        category=plan[3],
        estimated_days=plan[4] or 0,
        entries=entries,
    )


# --- Word Explanations Endpoint ---


@router.get("/word-explanations/{strongs_number}", response_model=WordExplanation)
async def get_word_explanation(strongs_number: str) -> WordExplanation:
    """Get a plain-English explanation of what the original word adds beyond translation."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT strongs_number, language, explanation FROM word_explanations WHERE strongs_number = ?",
            (strongs_number,),
        )
        if not rows:
            raise HTTPException(status_code=404, detail=f"No explanation for {strongs_number}")
        r = rows[0]
        return WordExplanation(strongs_number=r[0], language=r[1], explanation=r[2])
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail="Word explanations not available") from exc


# --- Genre Shifts Endpoint ---


@router.get("/genre-shifts/{book_id}", response_model=List[GenreShift])
async def get_genre_shifts(book_id: int) -> List[GenreShift]:
    """Get all genre transitions within a book."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT chapter, verse, from_genre, to_genre, description "
            "FROM genre_shifts WHERE book_id = ? ORDER BY chapter, verse",
            (book_id,),
        )
        return [GenreShift(chapter=r[0], verse=r[1], from_genre=r[2], to_genre=r[3], description=r[4]) for r in rows]
    except sqlite3.OperationalError:
        return []


# --- Notes Endpoints ---


@router.post("/notes/{book_id}/{chapter}/{verse}", response_model=NoteResponse)
async def create_note(book_id: int, chapter: int, verse: int, body: NoteCreate) -> NoteResponse:
    """Create a note on a verse."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    note_id = mgr.create_note(book_id, chapter, verse, body.content, body.note_type)
    return NoteResponse(
        note_id=note_id,
        book_id=book_id,
        chapter=chapter,
        verse=verse,
        content=body.content,
        note_type=body.note_type,
    )


@router.get("/notes/{book_id}/{chapter}/{verse}", response_model=List[NoteResponse])
async def get_notes(book_id: int, chapter: int, verse: int) -> List[NoteResponse]:
    """Get all notes for a verse."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    notes = mgr.get_notes_for_verse(book_id, chapter, verse)
    return [
        NoteResponse(
            note_id=n["note_id"],
            book_id=book_id,
            chapter=chapter,
            verse=verse,
            content=n["content"],
            note_type=n["note_type"],
            created_at=n.get("created_at"),
            updated_at=n.get("updated_at"),
        )
        for n in notes
    ]


@router.delete("/notes/{note_id}")
async def delete_note(note_id: int) -> Dict[str, Any]:
    """Delete a note."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    deleted = mgr.delete_note(note_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"deleted": True}


# --- Collections Endpoints ---


@router.post("/collections", response_model=CollectionResponse)
async def create_collection(body: CollectionCreate) -> CollectionResponse:
    """Create a new verse collection."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    cid = mgr.create_collection(body.name, body.description)
    return CollectionResponse(collection_id=cid, name=body.name, description=body.description)


@router.get("/collections", response_model=List[CollectionResponse])
async def list_collections() -> List[CollectionResponse]:
    """List all user collections."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    return [
        CollectionResponse(
            collection_id=c["collection_id"],
            name=c["name"],
            description=c["description"],
            created_at=c.get("created_at"),
            verse_count=c["verse_count"],
        )
        for c in mgr.list_collections()
    ]


@router.post("/collections/{collection_id}/items")
async def add_to_collection(collection_id: int, body: CollectionItemAdd) -> Dict[str, Any]:
    """Add a verse to a collection."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    added = mgr.add_to_collection(collection_id, body.book_id, body.chapter, body.verse, body.note)
    if not added:
        raise HTTPException(status_code=409, detail="Verse already in collection")
    return {"added": True}


@router.get("/collections/{collection_id}/items", response_model=List[Dict[str, Any]])
async def get_collection_items(collection_id: int) -> List[Dict[str, Any]]:
    """Get all items in a collection."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    return mgr.get_collection_items(collection_id)


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: int) -> Dict[str, Any]:
    """Delete a collection."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    deleted = mgr.delete_collection(collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"deleted": True}


# --- Sharing Endpoints ---


@router.post("/share", response_model=ShareResponse)
async def create_share(body: ShareCreate) -> ShareResponse:
    """Create a shareable link for content."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    token = mgr.create_share(body.share_type, body.content, body.title)
    return ShareResponse(share_token=token, share_type=body.share_type, title=body.title, content=body.content)


@router.get("/share/{token}", response_model=ShareResponse)
async def get_shared_item(token: str) -> ShareResponse:
    """Retrieve a shared item."""
    from ..enrichment.user_annotations import UserAnnotationManager

    db = _get_db()
    mgr = UserAnnotationManager(db.db_path)
    item = mgr.get_shared_item(token)
    if not item:
        raise HTTPException(status_code=404, detail="Shared item not found")
    return ShareResponse(
        share_token=token,
        share_type=item["share_type"],
        title=item["title"],
        content=item["content"],
        created_at=item.get("created_at"),
    )


# --- Export Endpoint ---


@router.get("/export/verse/{translation_id}/{book_id}/{chapter}/{verse}")
async def export_verse(
    translation_id: str,
    book_id: int,
    chapter: int,
    verse: int,
    export_format: str = Query("json", alias="format", description="Export format: json or markdown"),
) -> Dict[str, Any]:
    """Export a verse with all available enrichment data."""
    search = _get_search()
    result = search.get_verse(translation_id, book_id, chapter, verse)
    if not result:
        raise HTTPException(status_code=404, detail="Verse not found")

    book_name = _resolve_book_name(book_id, translation_id)
    ref = f"{book_name or book_id} {chapter}:{verse}"

    words = _get_words_for_verse(book_name or str(book_id), chapter, verse)
    xrefs = _get_cross_refs(book_id, chapter, verse)

    data: Dict[str, Any] = {
        "reference": ref,
        "text": result.text,
        "translation_id": translation_id,
    }
    if words:
        data["original_words"] = [w.model_dump() for w in words]
    if xrefs:
        data["cross_references"] = [x.model_dump() for x in xrefs]
    if export_format == "markdown":
        data["markdown"] = _build_export_markdown(ref, result.text, words, xrefs)

    return data


# --- Internal helpers ---


def _build_export_markdown(ref: str, text: str, words: List["WordDetail"], xrefs: List["CrossRef"]) -> str:
    """Build markdown representation of a verse export."""
    md = f"# {ref}\n\n> {text}\n\n"
    if words:
        md += "## Original Language Words\n\n"
        for w in words:
            md += f"- **{w.original_text}** ({w.transliteration}) — {w.english_gloss}"
            if w.strongs_number:
                md += f" [{w.strongs_number}]"
            md += "\n"
    if xrefs:
        md += "\n## Cross References\n\n"
        for x in xrefs:
            md += f"- {x.target_reference} ({x.ref_type})"
            if x.notes:
                md += f" — {x.notes}"
            md += "\n"
    return md


def _deserialize_json_list(json_str: Optional[str], model_cls: type) -> Optional[List[Any]]:
    """Deserialize a JSON string into a list of Pydantic model instances."""
    if not json_str:
        return None
    try:
        return [model_cls(**item) for item in json.loads(json_str)]
    except (json.JSONDecodeError, TypeError):
        return None


def _deserialize_json_obj(json_str: Optional[str], model_cls: type) -> Optional[Any]:
    """Deserialize a JSON string into a single Pydantic model instance."""
    if not json_str:
        return None
    try:
        return model_cls(**json.loads(json_str))
    except (json.JSONDecodeError, TypeError):
        return None


def _apply_cache_standard(row: sqlite3.Row, response: VerseResponse, book_name: Optional[str], verse: int) -> None:
    """Apply STANDARD-depth cached fields to response, falling back to live queries."""
    book_key = book_name or str(response.chapter)
    chapter = response.chapter

    cached_words = _deserialize_json_list(row["words_json"], WordDetail)
    response.words = cached_words if cached_words is not None else _get_words_for_verse(book_key, chapter, verse)

    cached_flags = _deserialize_json_list(row["richness_flags_json"], RichnessFlag)
    if cached_flags is not None:
        response.richness_flags = cached_flags


def _apply_cache_deep(
    row: sqlite3.Row,
    response: VerseResponse,
    book_id: int,
    chapter: int,
    verse: int,
    translation_id: str,
) -> None:
    """Apply DEEP-depth cached fields to response, falling back to live queries."""
    from .models import LiteraryStructure as LS

    response.cross_references = _deserialize_json_list(row["cross_references_json"], CrossRef) or _get_cross_refs(
        book_id, chapter, verse
    )
    response.cultural_context = _deserialize_json_list(
        row["cultural_context_json"], CulturalNote
    ) or _get_cultural_context(book_id, chapter, verse)
    response.passage_info = _deserialize_json_obj(row["passage_info_json"], PassageInfo) or _get_passage_info(
        book_id, chapter, verse
    )
    response.literary_structures = _deserialize_json_list(
        row["literary_structures_json"], LS
    ) or _get_literary_structures(book_id, chapter, verse)
    response.speaker = _deserialize_json_obj(row["speaker_json"], SpeakerAttribution) or _get_speaker(
        book_id, chapter, verse
    )
    response.genre = row["active_genre"]
    if response.genre in ("narrative", "unknown"):
        response.is_descriptive = True
    response.concepts = []
    response.surrounding_context = _get_surrounding_context(translation_id, book_id, chapter, verse)


def _try_annotation_cache(
    book_id: int,
    chapter: int,
    verse: int,
    depth: DepthLevel,
    response: VerseResponse,
    translation_id: str,
    book_name: Optional[str],
) -> bool:
    """Try to populate a VerseResponse from the annotation cache.

    Returns True if the cache was used, False if the caller should fall back
    to individual queries.
    """
    db = _get_db()
    row = db.get_annotation_cache(book_id, chapter, verse)
    if row is None:
        return False

    _apply_cache_standard(row, response, book_name, verse)

    if depth in (DepthLevel.DEEP, DepthLevel.SCHOLARLY):
        _apply_cache_deep(row, response, book_id, chapter, verse, translation_id)

    return True


def _resolve_book_name(book_id: int, translation_id: str) -> Optional[str]:
    """Look up the book name used in the words table for a given numeric book_id.

    The words table stores abbreviated book names from STEPBible (e.g. 'Gen', 'John'),
    while the books table stores full names (e.g. 'Genesis', 'John').  This function
    first checks the words table for the actual abbreviation used, then falls back to
    the books table full name.
    """
    db = _get_db()

    # First: check what abbreviation the words table actually uses for this book.
    # The books table maps book_id -> book_order which corresponds to the canonical number,
    # but the words table uses the STEPBible short name.  We look up via the books table
    # name and also try the words table directly.
    book_rows = db.execute_query(
        "SELECT name FROM books WHERE book_id = ? AND translation_id = ? LIMIT 1",
        (book_id, translation_id),
    )
    full_name: Optional[str] = str(book_rows[0][0]) if book_rows else None

    # Check if the words table uses this full name
    if full_name:
        word_check = db.execute_query(
            "SELECT 1 FROM words WHERE book = ? LIMIT 1",
            (full_name,),
        )
        if word_check:
            return full_name

    # Try the words table for common abbreviation patterns (STEPBible uses 3-char codes)
    # Look up any word at the expected chapter/verse to find the actual book abbreviation
    word_rows = db.execute_query(
        "SELECT DISTINCT book FROM words ORDER BY book LIMIT 100",
    )
    if full_name and word_rows:
        full_lower = full_name.lower()
        for row in word_rows:
            candidate: str = str(row[0])
            # Match if the full name starts with the abbreviation
            if full_lower.startswith(candidate.lower()) or candidate.lower().startswith(full_lower[:3]):
                return candidate

    return full_name


def _get_words_for_verse(book: str, chapter: int, verse: int) -> List[WordDetail]:
    """Get original language words for a verse as WordDetail models."""
    search = _get_search()
    results = search.get_words_for_verse(book, chapter, verse)
    return [
        WordDetail(
            word_num=w.word_num,
            original_text=w.hebrew_text or w.greek_text,
            transliteration=w.transliteration,
            english_gloss=w.translation,
            strongs_number=w.strongs_primary,
            morphology_code=w.morphology_code,
            language=w.language,
        )
        for w in results
    ]


def _enrich_book_metadata(db: SQLiteManager, book: BookInfo) -> None:
    """Enrich a BookInfo with data from book_metadata table if it exists."""
    try:
        rows = db.execute_query(
            """SELECT primary_genre, secondary_genres, author_traditional,
                      date_range_start, date_range_end, original_audience,
                      literary_features, reading_context, canonical_section
               FROM book_metadata WHERE book_id = ?""",
            (book.book_id,),
        )
        if rows:
            row = rows[0]
            book.primary_genre = row[0]
            book.secondary_genres = _parse_json_list(row[1])
            book.author_traditional = row[2]
            start, end = row[3], row[4]
            if start and end:
                era_s = "BCE" if start < 0 else "CE"
                era_e = "BCE" if end < 0 else "CE"
                book.date_range = f"{abs(start)} {era_s} - {abs(end)} {era_e}"
            book.original_audience = row[5]
            book.literary_features = _parse_json_list(row[6])
            book.reading_context = row[7]
            book.canonical_section = row[8]
    except sqlite3.OperationalError:
        pass  # Table may not exist yet


def _get_richness_flags(book: str, chapter: int, verse: int) -> List[RichnessFlag]:
    """Get word richness flags for a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT wr.word_num, wr.strongs_number, wr.richness_score, "
            "wr.untranslatable_nuances, wr.morphology_significance, "
            "l.original_word, l.gloss, l.definition "
            "FROM word_richness wr "
            "LEFT JOIN lexicon l ON wr.strongs_number = l.strongs_number "
            "WHERE wr.book = ? AND wr.chapter = ? AND wr.verse = ? "
            "AND wr.richness_score > 0.3 "
            "ORDER BY wr.richness_score DESC",
            (book, chapter, verse),
        )
        flags = []
        for r in rows:
            nuances: List[str] = []
            if r[3]:
                try:
                    nuances = json.loads(r[3])
                except (json.JSONDecodeError, TypeError):
                    pass
            flags.append(
                RichnessFlag(
                    word_num=r[0],
                    strongs_number=r[1],
                    richness_score=r[2],
                    untranslatable_nuances=nuances,
                    morphology_significance=r[4],
                    original_word=r[5],
                    english_gloss=r[6],
                    full_definition=r[7],
                )
            )
        return flags
    except sqlite3.OperationalError:
        return []


def _get_cross_refs(book_id: int, chapter: int, verse: int) -> List[CrossRef]:
    """Get cross-references for a verse."""
    db = _get_db()
    book_names = {
        1: "Gen",
        2: "Exo",
        3: "Lev",
        4: "Num",
        5: "Deu",
        6: "Jos",
        7: "Jdg",
        8: "Rut",
        9: "1Sa",
        10: "2Sa",
        11: "1Ki",
        12: "2Ki",
        13: "1Ch",
        14: "2Ch",
        15: "Ezr",
        16: "Neh",
        17: "Est",
        18: "Job",
        19: "Psa",
        20: "Pro",
        21: "Ecc",
        22: "Sng",
        23: "Isa",
        24: "Jer",
        25: "Lam",
        26: "Ezk",
        27: "Dan",
        28: "Hos",
        29: "Jol",
        30: "Amo",
        31: "Oba",
        32: "Jon",
        33: "Mic",
        34: "Nam",
        35: "Hab",
        36: "Zep",
        37: "Hag",
        38: "Zec",
        39: "Mal",
        40: "Mat",
        41: "Mrk",
        42: "Luk",
        43: "Jhn",
        44: "Act",
        45: "Rom",
        46: "1Co",
        47: "2Co",
        48: "Gal",
        49: "Eph",
        50: "Php",
        51: "Col",
        52: "1Th",
        53: "2Th",
        54: "1Ti",
        55: "2Ti",
        56: "Tit",
        57: "Phm",
        58: "Heb",
        59: "Jas",
        60: "1Pe",
        61: "2Pe",
        62: "1Jn",
        63: "2Jn",
        64: "3Jn",
        65: "Jud",
        66: "Rev",
    }
    try:
        rows = db.execute_query(
            "SELECT target_book_id, target_chapter, target_verse, ref_type, confidence, notes "
            "FROM cross_references "
            "WHERE source_book_id = ? AND source_chapter = ? AND source_verse = ?",
            (book_id, chapter, verse),
        )
        refs = []
        for r in rows:
            tgt_name = book_names.get(r[0], str(r[0]))
            refs.append(
                CrossRef(
                    target_reference=f"{tgt_name} {r[1]}:{r[2]}",
                    ref_type=r[3],
                    confidence=r[4] or 0.8,
                    notes=r[5],
                )
            )
        # Also include incoming references
        rows2 = db.execute_query(
            "SELECT source_book_id, source_chapter, source_verse, ref_type, confidence, notes "
            "FROM cross_references "
            "WHERE target_book_id = ? AND target_chapter = ? AND target_verse = ?",
            (book_id, chapter, verse),
        )
        for r in rows2:
            src_name = book_names.get(r[0], str(r[0]))
            refs.append(
                CrossRef(
                    target_reference=f"{src_name} {r[1]}:{r[2]}",
                    ref_type=r[3],
                    confidence=r[4] or 0.8,
                    notes=r[5],
                )
            )
        return refs
    except sqlite3.OperationalError:
        return []


def _get_cultural_context(book_id: int, _chapter: int = 0, _verse: int = 0) -> List[CulturalNote]:
    """Get cultural context for a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT context_id, context_type, title, summary, detailed_content, "
            "time_period, geographic_region, confidence "
            "FROM cultural_context "
            "WHERE book_id = ? AND start_chapter IS NULL "
            "ORDER BY display_priority",
            (book_id,),
        )
        return [
            CulturalNote(
                context_id=r[0],
                context_type=r[1],
                title=r[2],
                summary=r[3],
                detailed_content=r[4],
                time_period=r[5],
                geographic_region=r[6],
                confidence=r[7],
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


def _get_passage_info(book_id: int, chapter: int, verse: int) -> Optional[PassageInfo]:
    """Get the innermost passage containing a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT passage_id, title, genre, literary_type, structural_features, "
            "start_chapter, start_verse, end_chapter, end_verse "
            "FROM passages "
            "WHERE book_id = ? "
            "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
            "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) "
            "ORDER BY (end_chapter - start_chapter) ASC, (end_verse - start_verse) ASC "
            "LIMIT 1",
            (book_id, chapter, chapter, verse, chapter, chapter, verse),
        )
        if rows:
            r = rows[0]
            return PassageInfo(
                passage_id=r[0],
                title=r[1],
                genre=r[2],
                literary_type=r[3],
                structural_features=_parse_json_list(r[4]),
                start_chapter=r[5],
                start_verse=r[6],
                end_chapter=r[7],
                end_verse=r[8],
            )
    except sqlite3.OperationalError:
        pass
    return None


def _get_literary_structures(book_id: int, chapter: int, verse: int) -> List[Any]:
    """Get literary structures containing a verse."""
    from .models import LiteraryStructure as LS

    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT structure_type, description, significance, elements "
            "FROM literary_structures "
            "WHERE book_id = ? "
            "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
            "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) ",
            (book_id, chapter, chapter, verse, chapter, chapter, verse),
        )
        structures: List[LS] = []
        for r in rows:
            elements = _parse_json_list(r[3])
            structures.append(
                LS(
                    structure_type=r[0],
                    description=r[1],
                    significance=r[2],
                    elements=elements,
                )
            )
        return structures
    except sqlite3.OperationalError:
        return []


def _get_surrounding_context(translation_id: str, book_id: int, chapter: int, verse: int) -> VerseContext:
    """Get previous and next verse text for anti-proof-texting context."""
    db = _get_db()
    prev_text = None
    next_text = None
    try:
        if verse > 1:
            prev_rows = db.execute_query(
                "SELECT text FROM verses WHERE translation_id = ? AND book_id = ? AND chapter = ? AND verse = ?",
                (translation_id, book_id, chapter, verse - 1),
            )
            if prev_rows:
                prev_text = str(prev_rows[0][0])
        next_rows = db.execute_query(
            "SELECT text FROM verses WHERE translation_id = ? AND book_id = ? AND chapter = ? AND verse = ?",
            (translation_id, book_id, chapter, verse + 1),
        )
        if next_rows:
            next_text = str(next_rows[0][0])
    except sqlite3.OperationalError:
        pass
    return VerseContext(previous_verse=prev_text, next_verse=next_text)


def _get_speaker(book_id: int, chapter: int, verse: int) -> Optional[SpeakerAttribution]:
    """Get speaker attribution for a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT speaker, context_note FROM speaker_attributions "
            "WHERE book_id = ? "
            "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
            "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) "
            "LIMIT 1",
            (book_id, chapter, chapter, verse, chapter, chapter, verse),
        )
        if rows:
            return SpeakerAttribution(speaker=rows[0][0], context_note=rows[0][1])
    except sqlite3.OperationalError:
        pass
    return None


def _get_active_genre(book_id: int, chapter: int, verse: int) -> Optional[str]:
    """Determine the active literary genre at a verse based on genre shifts."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT to_genre FROM genre_shifts "
            "WHERE book_id = ? AND (chapter < ? OR (chapter = ? AND verse <= ?)) "
            "ORDER BY chapter DESC, verse DESC LIMIT 1",
            (book_id, chapter, chapter, verse),
        )
        if rows:
            return str(rows[0][0])
    except sqlite3.OperationalError:
        pass
    # Fall back to book's primary genre from book_metadata
    try:
        rows = db.execute_query(
            "SELECT primary_genre FROM book_metadata WHERE book_id = ?",
            (book_id,),
        )
        if rows:
            return str(rows[0][0])
    except sqlite3.OperationalError:
        pass
    return None


def _detect_translation_divergences(translations: Dict[str, str]) -> List[Dict[str, Any]]:
    """Detect significant differences between translations."""
    if len(translations) < 2:
        return []
    divergences: List[Dict[str, Any]] = []
    items = list(translations.items())
    for i, (tid1, text1) in enumerate(items):
        for tid2, text2 in items[i + 1 :]:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            unique_to_1 = words1 - words2
            unique_to_2 = words2 - words1
            overlap = words1 & words2
            total_unique = len(words1 | words2)
            similarity = len(overlap) / max(total_unique, 1)
            if similarity < 0.85:
                divergences.append(
                    {
                        "translations": [tid1, tid2],
                        "similarity": round(similarity, 3),
                        "unique_to_first": sorted(unique_to_1)[:5],
                        "unique_to_second": sorted(unique_to_2)[:5],
                    }
                )
    return divergences


def _parse_json_list(value: Optional[str]) -> List[Any]:
    """Parse a JSON array string into a list, or return empty list."""
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# --- Phase 9 Helper Functions ---


def _get_manuscript_variants(book_id: int, chapter: int, verse: int) -> List[ManuscriptVariant]:
    """Get manuscript variants for a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT id, variant_type, base_text, variant_text, manuscripts, "
            "explanation, significance, confidence "
            "FROM manuscript_variants "
            "WHERE book_id = ? AND chapter = ? AND verse = ?",
            (book_id, chapter, verse),
        )
        return [
            ManuscriptVariant(
                variant_id=r[0],
                variant_type=r[1],
                base_text=r[2],
                variant_text=r[3],
                manuscripts=r[4],
                explanation=r[5],
                significance=r[6] or "minor",
                confidence=r[7] or 0.8,
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


def _get_syntax_tree(book_id: int, chapter: int, verse: int) -> Optional[VerseSyntaxTree]:
    """Get syntax tree for a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT node_id, node_type, role, parent_id, clause_type, relation, depth, text_content, word_num "
            "FROM syntax_trees "
            "WHERE book_id = ? AND chapter = ? AND verse = ? "
            "ORDER BY depth, id",
            (book_id, chapter, verse),
        )
        if not rows:
            return None

        nodes_by_id: Dict[str, SyntaxNode] = {}
        root_nodes: List[SyntaxNode] = []

        for r in rows:
            node = SyntaxNode(
                node_id=r[0],
                node_type=r[1],
                role=r[2],
                clause_type=r[4],
                relation=r[5],
                depth=r[6] or 0,
                text_content=r[7],
                word_num=r[8],
            )
            nodes_by_id[r[0]] = node
            parent_id = r[3]
            if parent_id and parent_id in nodes_by_id:
                nodes_by_id[parent_id].children.append(node)
            else:
                root_nodes.append(node)

        return VerseSyntaxTree(book_id=book_id, chapter=chapter, verse=verse, root_nodes=root_nodes)
    except sqlite3.OperationalError:
        return None


def _get_discourse_units(book_id: int, chapter: int, verse: int) -> List[DiscourseUnit]:
    """Get discourse annotations covering a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT id, discourse_type, function_label, relation_to_context, description, "
            "prominence, start_chapter, start_verse, end_chapter, end_verse "
            "FROM discourse_annotations "
            "WHERE book_id = ? "
            "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
            "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) ",
            (book_id, chapter, chapter, verse, chapter, chapter, verse),
        )
        return [
            DiscourseUnit(
                discourse_id=r[0],
                discourse_type=r[1],
                function_label=r[2],
                relation_to_context=r[3],
                description=r[4],
                prominence=r[5] or 0,
                start_chapter=r[6],
                start_verse=r[7],
                end_chapter=r[8],
                end_verse=r[9],
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


def _get_semantic_domains_for_verse(book: str, chapter: int, verse: int) -> List[SemanticDomainMapping]:
    """Get semantic domain mappings for words in a verse."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT DISTINCT sdm.strongs_number, sdm.domain_code, sd.domain_name, sdm.confidence "
            "FROM strongs_domain_mappings sdm "
            "JOIN semantic_domains sd ON sdm.domain_code = sd.domain_code "
            "JOIN words w ON w.strongs_primary = sdm.strongs_number "
            "WHERE w.book = ? AND w.chapter = ? AND w.verse = ?",
            (book, chapter, verse),
        )
        return [
            SemanticDomainMapping(
                strongs_number=r[0],
                domain_code=r[1],
                domain_name=r[2],
                confidence=r[3] or 0.9,
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


# --- Phase 9A: Semantic Domain Endpoints ---


@router.get("/semantic-domains", response_model=List[SemanticDomain], tags=["phase9"])
async def list_semantic_domains(
    parent: Optional[str] = Query(None, description="Filter by parent domain code"),
) -> List[SemanticDomain]:
    """List Louw-Nida semantic domains."""
    db = _get_db()
    try:
        if parent:
            rows = db.execute_query(
                "SELECT domain_code, domain_name, parent_domain, description, level "
                "FROM semantic_domains WHERE parent_domain = ? ORDER BY domain_code",
                (parent,),
            )
        else:
            rows = db.execute_query(
                "SELECT domain_code, domain_name, parent_domain, description, level "
                "FROM semantic_domains ORDER BY domain_code",
            )
        return [
            SemanticDomain(domain_code=r[0], domain_name=r[1], parent_domain=r[2], description=r[3], level=r[4] or 1)
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


@router.get("/semantic-domains/{domain_code}/words", response_model=List[WordDomainResult], tags=["phase9"])
async def get_domain_words(domain_code: str) -> List[WordDomainResult]:
    """Get all words mapped to a semantic domain."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT sdm.strongs_number, l.original_word, l.gloss, sdm.confidence "
            "FROM strongs_domain_mappings sdm "
            "LEFT JOIN lexicon l ON sdm.strongs_number = l.strongs_number "
            "WHERE sdm.domain_code = ? "
            "ORDER BY sdm.confidence DESC",
            (domain_code,),
        )
        return [WordDomainResult(strongs_number=r[0], original_word=r[1], gloss=r[2]) for r in rows]
    except sqlite3.OperationalError:
        return []


@router.get("/words/{strongs_number}/domains", response_model=WordDomainResult, tags=["phase9"])
async def get_word_domains(strongs_number: str) -> WordDomainResult:
    """Get semantic domains for a specific word (by Strong's number)."""
    db = _get_db()
    lexicon_row = db.get_lexicon_entry(strongs_number)
    try:
        rows = db.execute_query(
            "SELECT sd.domain_code, sd.domain_name, sd.parent_domain, sd.description, sd.level, sdm.confidence "
            "FROM strongs_domain_mappings sdm "
            "JOIN semantic_domains sd ON sdm.domain_code = sd.domain_code "
            "WHERE sdm.strongs_number = ?",
            (strongs_number,),
        )
        domains = [
            SemanticDomain(domain_code=r[0], domain_name=r[1], parent_domain=r[2], description=r[3], level=r[4] or 1)
            for r in rows
        ]
        # Get related words in same domains
        related: List[Dict[str, Any]] = []
        for domain in domains[:3]:
            rel_rows = db.execute_query(
                "SELECT sdm.strongs_number, l.original_word, l.gloss "
                "FROM strongs_domain_mappings sdm "
                "LEFT JOIN lexicon l ON sdm.strongs_number = l.strongs_number "
                "WHERE sdm.domain_code = ? AND sdm.strongs_number != ? "
                "LIMIT 5",
                (domain.domain_code, strongs_number),
            )
            for rr in rel_rows:
                related.append({"strongs_number": rr[0], "original_word": rr[1], "gloss": rr[2]})
    except sqlite3.OperationalError:
        domains = []
        related = []
    return WordDomainResult(
        strongs_number=strongs_number,
        original_word=lexicon_row["original_word"] if lexicon_row else None,
        gloss=lexicon_row["gloss"] if lexicon_row else None,
        domains=domains,
        related_words=related,
    )


# --- Phase 9A: Concept Discovery Endpoint ---

_SYNONYM_MAP: Dict[str, List[str]] = {
    "worry": ["anxiety", "fear", "peace"],
    "anxious": ["anxiety", "fear", "peace"],
    "angry": ["anger", "wrath", "patience"],
    "sad": ["grief", "mourning", "comfort"],
    "money": ["wealth", "generosity", "contentment"],
    "forgive": ["forgiveness", "mercy", "grace"],
    "love": ["love", "agape", "compassion"],
    "death": ["grief", "resurrection", "eternal life"],
    "sin": ["sin", "repentance", "forgiveness"],
    "pray": ["prayer", "intercession", "worship"],
}


def _discover_matched_concepts(db: Any, query_lower: str, limit: int) -> List[TopicSummary]:
    """Search concepts by name/description match."""
    try:
        rows = db.execute_query(
            "SELECT cd.concept_id, cd.name, cd.description, "
            "COUNT(cvm.verse_id) as verse_count "
            "FROM concept_definitions cd "
            "LEFT JOIN concept_verse_mappings cvm ON cd.concept_id = cvm.concept_id "
            "WHERE LOWER(cd.name) LIKE ? OR LOWER(cd.description) LIKE ? "
            "GROUP BY cd.concept_id ORDER BY verse_count DESC LIMIT ?",
            (f"%{query_lower}%", f"%{query_lower}%", limit),
        )
        return [TopicSummary(name=r[1] or r[0], description=r[2], verse_count=r[3]) for r in rows]
    except sqlite3.OperationalError:
        return []


def _discover_matched_topics(db: Any, query_lower: str, limit: int) -> List[LifeTopicSummary]:
    """Search life topics by name/description match."""
    try:
        rows = db.execute_query(
            "SELECT slug, name, category, description, icon FROM life_topics "
            "WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ? LIMIT ?",
            (f"%{query_lower}%", f"%{query_lower}%", limit),
        )
        return [LifeTopicSummary(slug=r[0], name=r[1], category=r[2], description=r[3], icon=r[4]) for r in rows]
    except sqlite3.OperationalError:
        return []


def _discover_suggestions(db: Any, query_lower: str, query_words: set) -> List[str]:
    """Generate suggested searches from semantic domains and synonym expansion."""
    suggestions: List[str] = []
    try:
        domain_rows = db.execute_query(
            "SELECT domain_name FROM semantic_domains WHERE LOWER(domain_name) LIKE ? LIMIT 5",
            (f"%{query_lower}%",),
        )
        for r in domain_rows:
            suggestions.append(f"words in domain: {r[0]}")
    except sqlite3.OperationalError:
        pass

    for word in query_words:
        if word in _SYNONYM_MAP:
            for synonym in _SYNONYM_MAP[word]:
                if synonym not in [s.replace("words in domain: ", "") for s in suggestions]:
                    suggestions.append(f"topic: {synonym}")
    return suggestions[:10]


@router.get("/discover", response_model=ConceptDiscoveryResult, tags=["phase9"])
async def discover_concepts(
    q: str = Query(..., description="Natural language query"),
    limit: int = Query(10, ge=1, le=50),
) -> ConceptDiscoveryResult:
    """Discover biblical concepts from a natural-language query.

    Maps everyday language ('worried about money', 'how to forgive')
    to biblical concepts and life topics.
    """
    db = _get_db()
    query_lower = q.lower()
    query_words = set(query_lower.split())

    matched_concepts = _discover_matched_concepts(db, query_lower, limit)
    matched_topics = _discover_matched_topics(db, query_lower, limit)
    suggestions = _discover_suggestions(db, query_lower, query_words)

    return ConceptDiscoveryResult(
        query=q,
        matched_concepts=matched_concepts[:limit],
        matched_life_topics=matched_topics[:limit],
        suggested_searches=suggestions,
    )


# --- Phase 9B: Syntax Tree Endpoints ---


@router.get(
    "/syntax/{book_id}/{chapter}/{verse}",
    response_model=VerseSyntaxTree,
    tags=["phase9"],
)
async def get_verse_syntax(book_id: int, chapter: int, verse: int) -> VerseSyntaxTree:
    """Get clause-level syntax tree for a verse (MACULA treebank)."""
    tree = _get_syntax_tree(book_id, chapter, verse)
    if not tree:
        raise HTTPException(status_code=404, detail="Syntax tree not found for this verse")
    return tree


# --- Phase 9B: Discourse Annotation Endpoints ---


@router.get(
    "/discourse/{book_id}/{chapter}/{verse}",
    response_model=List[DiscourseUnit],
    tags=["phase9"],
)
async def get_verse_discourse(book_id: int, chapter: int, verse: int) -> List[DiscourseUnit]:
    """Get discourse annotations covering a verse (OpenText.org)."""
    return _get_discourse_units(book_id, chapter, verse)


@router.get(
    "/discourse/{book_id}",
    response_model=List[DiscourseUnit],
    tags=["phase9"],
)
async def get_book_discourse(book_id: int) -> List[DiscourseUnit]:
    """Get all discourse annotations for a book."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT id, discourse_type, function_label, relation_to_context, description, "
            "prominence, start_chapter, start_verse, end_chapter, end_verse "
            "FROM discourse_annotations WHERE book_id = ? ORDER BY start_chapter, start_verse",
            (book_id,),
        )
        return [
            DiscourseUnit(
                discourse_id=r[0],
                discourse_type=r[1],
                function_label=r[2],
                relation_to_context=r[3],
                description=r[4],
                prominence=r[5] or 0,
                start_chapter=r[6],
                start_verse=r[7],
                end_chapter=r[8],
                end_verse=r[9],
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


# --- Phase 9B: Manuscript Variant Endpoints ---


@router.get(
    "/variants/{book_id}/{chapter}/{verse}",
    response_model=List[ManuscriptVariant],
    tags=["phase9"],
)
async def get_verse_variants(book_id: int, chapter: int, verse: int) -> List[ManuscriptVariant]:
    """Get manuscript variants for a verse."""
    return _get_manuscript_variants(book_id, chapter, verse)


@router.get(
    "/variants/significant",
    response_model=List[ManuscriptVariant],
    tags=["phase9"],
)
async def get_significant_variants(
    limit: int = Query(50, ge=1, le=200),
) -> List[ManuscriptVariant]:
    """Get all significant (major) manuscript variants."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT id, variant_type, base_text, variant_text, manuscripts, "
            "explanation, significance, confidence "
            "FROM manuscript_variants WHERE significance = 'major' "
            "ORDER BY book_id, chapter, verse LIMIT ?",
            (limit,),
        )
        return [
            ManuscriptVariant(
                variant_id=r[0],
                variant_type=r[1],
                base_text=r[2],
                variant_text=r[3],
                manuscripts=r[4],
                explanation=r[5],
                significance=r[6] or "major",
                confidence=r[7] or 0.8,
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


# --- Phase 9C: Multi-language Semantic Search Endpoint ---


def _multilingual_word_search(db: Any, query_lower: str, limit: int) -> list:
    """Find matching original-language words/concepts."""
    try:
        rows = db.execute_query(
            "SELECT DISTINCT strongs_primary, book, chapter, verse "
            "FROM words WHERE LOWER(translation) LIKE ? "
            "ORDER BY book, chapter, verse LIMIT ?",
            (f"%{query_lower}%", limit * 3),
        )
        return list(rows)
    except sqlite3.OperationalError:
        return []


def _multilingual_lexicon_fallback(db: Any, query_lower: str, limit: int) -> list:
    """Fall back to lexicon gloss search when word search yields nothing."""
    word_rows: list = []
    try:
        lex_rows = db.execute_query(
            "SELECT strongs_number FROM lexicon WHERE LOWER(gloss) LIKE ? OR LOWER(definition) LIKE ? LIMIT 10",
            (f"%{query_lower}%", f"%{query_lower}%"),
        )
        for sn in [str(r[0]) for r in lex_rows][:5]:
            word_matches = db.execute_query(
                "SELECT DISTINCT book, chapter, verse FROM words WHERE strongs_primary = ? LIMIT ?",
                (sn, limit),
            )
            word_rows.extend([(sn, r[0], r[1], r[2]) for r in word_matches])
    except sqlite3.OperationalError:
        pass
    return word_rows


def _resolve_book_id(db: Any, book_name: str) -> Optional[int]:
    """Resolve a book name to its book_id."""
    try:
        rows = db.execute_query(
            "SELECT DISTINCT book_id FROM books WHERE name = ? OR common_name = ? LIMIT 1",
            (book_name, book_name),
        )
        return int(rows[0][0]) if rows else None
    except sqlite3.OperationalError:
        return None


def _fetch_verse_result(
    db: Any, tid: str, bid: int, ch: int, vs: int, book_name: str, strongs: str
) -> Optional[SemanticSearchResult]:
    """Fetch a single verse text and build a search result."""
    try:
        rows = db.execute_query(
            "SELECT text FROM verses WHERE translation_id = ? AND book_id = ? AND chapter = ? AND verse = ?",
            (tid, bid, ch, vs),
        )
        if rows:
            return SemanticSearchResult(
                book_id=bid,
                chapter=ch,
                verse=vs,
                text=str(rows[0][0]),
                book_name=book_name,
                score=0.8,
                match_type="multilingual",
                explanation=f"Matched via original language ({strongs})",
                translation_id=tid,
            )
    except sqlite3.OperationalError:
        pass
    return None


def _multilingual_resolve_verses(
    db: Any, word_rows: list, target_list: List[str], limit: int
) -> List[SemanticSearchResult]:
    """Resolve word matches to verse texts in requested translations."""
    results: List[SemanticSearchResult] = []
    seen: set = set()
    for row in word_rows:
        book_name, ch, vs = str(row[1]), int(row[2]), int(row[3])
        bid = _resolve_book_id(db, book_name)
        if bid is None:
            continue
        for tid in target_list:
            key = (tid, bid, ch, vs)
            if key in seen:
                continue
            seen.add(key)
            result = _fetch_verse_result(db, tid, bid, ch, vs, book_name, str(row[0]))
            if result:
                results.append(result)
            if len(results) >= limit:
                return results
        if len(results) >= limit:
            return results
    return results


@router.get("/search/multilingual", response_model=List[SemanticSearchResult], tags=["phase9"])
async def multilingual_search(
    q: str = Query(..., description="Search query in any language"),
    source_lang: str = Query("en", description="Source language of the query"),  # noqa: ARG001  # pylint: disable=unused-argument
    target_translations: Optional[str] = Query(None, description="Comma-separated translation IDs"),
    limit: int = Query(20, ge=1, le=100),
) -> List[SemanticSearchResult]:
    """Search across translations in any language via original-language alignment.

    The query is matched against original Hebrew/Greek concepts, then results
    are returned from the requested translations regardless of language.
    """
    db = _get_db()
    query_lower = q.lower()
    word_rows = _multilingual_word_search(db, query_lower, limit)
    if not word_rows:
        word_rows = _multilingual_lexicon_fallback(db, query_lower, limit)

    target_list = target_translations.split(",") if target_translations else ["engbsb"]
    return _multilingual_resolve_verses(db, word_rows, target_list, limit)[:limit]


# --- Phase 9C: Community Contribution Endpoints ---


@router.post("/community/contributions", response_model=ContributionResponse, tags=["phase9"])
async def create_contribution(body: ContributionCreate) -> ContributionResponse:
    """Submit a community contribution for review."""
    db = _get_db()
    try:
        db.execute_update(
            "INSERT INTO community_contributions (book_id, chapter, verse, contribution_type, title, content) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (body.book_id, body.chapter, body.verse, body.contribution_type, body.title, body.content),
        )
        rows = db.execute_query(
            "SELECT id, book_id, chapter, verse, contribution_type, title, content, "
            "author_id, status, created_at "
            "FROM community_contributions ORDER BY id DESC LIMIT 1",
        )
        r = rows[0]
        return ContributionResponse(
            id=r[0],
            book_id=r[1],
            chapter=r[2],
            verse=r[3],
            contribution_type=r[4],
            title=r[5],
            content=r[6],
            author_id=r[7] or "anonymous",
            status=r[8] or "pending",
            created_at=str(r[9]) if r[9] else None,
        )
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create contribution: {e}") from e


@router.get("/community/contributions", response_model=List[ContributionResponse], tags=["phase9"])
async def list_contributions(
    status: Optional[str] = Query(None, description="Filter by status"),
    book_id: Optional[int] = Query(None, description="Filter by book"),
    limit: int = Query(50, ge=1, le=200),
) -> List[ContributionResponse]:
    """List community contributions."""
    db = _get_db()
    try:
        query = (
            "SELECT id, book_id, chapter, verse, contribution_type, title, content, "
            "author_id, status, created_at FROM community_contributions WHERE 1=1"
        )
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if book_id is not None:
            query += " AND book_id = ?"
            params.append(book_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = db.execute_query(query, tuple(params))
        return [
            ContributionResponse(
                id=r[0],
                book_id=r[1],
                chapter=r[2],
                verse=r[3],
                contribution_type=r[4],
                title=r[5],
                content=r[6],
                author_id=r[7] or "anonymous",
                status=r[8] or "pending",
                created_at=str(r[9]) if r[9] else None,
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


@router.post("/community/contributions/{contribution_id}/review", tags=["phase9"])
async def review_contribution(contribution_id: int, body: ContributionReviewCreate) -> Dict[str, str]:
    """Review a community contribution."""
    db = _get_db()
    try:
        db.execute_update(
            "INSERT INTO contribution_reviews (contribution_id, reviewer_id, decision, review_note) "
            "VALUES (?, ?, ?, ?)",
            (contribution_id, "reviewer", body.decision, body.review_note),
        )
        new_status = "approved" if body.decision == "approve" else "rejected"
        if body.decision == "request_changes":
            new_status = "pending"
        db.execute_update(
            "UPDATE community_contributions SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_status, contribution_id),
        )
        return {"status": "ok", "new_status": new_status}
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Review failed: {e}") from e


# --- Phase 9C: Concept Proposal Endpoints ---


@router.post("/concepts/proposals", response_model=ConceptProposalResponse, tags=["phase9"])
async def create_concept_proposal(body: ConceptProposalCreate) -> ConceptProposalResponse:
    """Propose a new concept or edit to an existing one."""
    db = _get_db()
    try:
        db.execute_update(
            "INSERT INTO concept_proposals "
            "(concept_name, proposal_type, description, hebrew_terms_json, greek_terms_json, verse_mappings_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                body.concept_name,
                body.proposal_type,
                body.description,
                json.dumps(body.hebrew_terms),
                json.dumps(body.greek_terms),
                json.dumps(body.verse_mappings),
            ),
        )
        rows = db.execute_query(
            "SELECT id, concept_name, proposed_by, proposal_type, description, "
            "hebrew_terms_json, greek_terms_json, verse_mappings_json, status, created_at "
            "FROM concept_proposals ORDER BY id DESC LIMIT 1",
        )
        r = rows[0]
        return ConceptProposalResponse(
            id=r[0],
            concept_name=r[1],
            proposed_by=r[2] or "anonymous",
            proposal_type=r[3],
            description=r[4],
            hebrew_terms=json.loads(r[5]) if r[5] else [],
            greek_terms=json.loads(r[6]) if r[6] else [],
            verse_mappings=json.loads(r[7]) if r[7] else [],
            status=r[8] or "pending",
            created_at=str(r[9]) if r[9] else None,
        )
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create proposal: {e}") from e


@router.get("/concepts/proposals", response_model=List[ConceptProposalResponse], tags=["phase9"])
async def list_concept_proposals(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> List[ConceptProposalResponse]:
    """List concept proposals."""
    db = _get_db()
    try:
        query = (
            "SELECT id, concept_name, proposed_by, proposal_type, description, "
            "hebrew_terms_json, greek_terms_json, verse_mappings_json, status, created_at "
            "FROM concept_proposals WHERE 1=1"
        )
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = db.execute_query(query, tuple(params))
        return [
            ConceptProposalResponse(
                id=r[0],
                concept_name=r[1],
                proposed_by=r[2] or "anonymous",
                proposal_type=r[3],
                description=r[4],
                hebrew_terms=json.loads(r[5]) if r[5] else [],
                greek_terms=json.loads(r[6]) if r[6] else [],
                verse_mappings=json.loads(r[7]) if r[7] else [],
                status=r[8] or "pending",
                created_at=str(r[9]) if r[9] else None,
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []


# --- Phase 9D: Audio Integration Endpoints ---


@router.get(
    "/audio/{book_id}/{chapter}",
    response_model=AudioResource,
    tags=["phase9"],
)
async def get_audio_resource(
    book_id: int,
    chapter: int,
    translation_id: str = Query("engbsb"),
) -> AudioResource:
    """Get audio resource URL for a chapter.

    Returns metadata for audio playback. Actual audio files are served
    externally; this endpoint provides the URL and timing metadata.
    """
    return AudioResource(
        book_id=book_id,
        chapter=chapter,
        verse_start=1,
        audio_url=f"/audio/{translation_id}/{book_id}/{chapter}.mp3",
        translation_id=translation_id,
        narrator="ABBA Default",
    )


# --- Phase 9D: Semantic Relationship Graph Endpoints ---


@router.get("/graph/{concept_name}", response_model=ConceptGraph, tags=["phase9"])
async def get_concept_graph(
    concept_name: str,
    depth: int = Query(1, ge=1, le=3, description="How many hops from the center concept"),
) -> ConceptGraph:
    """Get a semantic relationship graph for a concept.

    Returns nodes and edges for visualization (e.g., force-directed graph).
    """
    db = _get_db()
    relationships: List[SemanticRelationship] = []
    visited: set = {concept_name}
    frontier = [concept_name]

    for _ in range(depth):
        next_frontier: List[str] = []
        for concept in frontier:
            try:
                rows = db.execute_query(
                    "SELECT source_concept, target_concept, relationship_type, weight, "
                    "evidence_count, shared_strongs_json "
                    "FROM semantic_relationship_graph "
                    "WHERE source_concept = ? OR target_concept = ?",
                    (concept, concept),
                )
                for r in rows:
                    shared = []
                    if r[5]:
                        try:
                            shared = json.loads(r[5])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    relationships.append(
                        SemanticRelationship(
                            source_concept=r[0],
                            target_concept=r[1],
                            relationship_type=r[2],
                            weight=r[3] or 1.0,
                            evidence_count=r[4] or 0,
                            shared_strongs=shared,
                        )
                    )
                    neighbor = r[1] if r[0] == concept else r[0]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            except sqlite3.OperationalError:
                break
        frontier = next_frontier

    nodes = [{"name": n, "is_center": n == concept_name} for n in visited]
    return ConceptGraph(
        center_concept=concept_name,
        relationships=relationships,
        nodes=nodes,
    )


# --- Phase 9D: ML Concept Feedback Endpoints ---


@router.post("/concepts/{concept_name}/feedback", tags=["phase9"])
async def submit_concept_feedback(
    concept_name: str,
    verse_id: str = Query(..., description="Verse identifier"),
    feedback_type: str = Query(..., description="relevant, irrelevant, or partial"),
) -> Dict[str, str]:
    """Submit feedback on a concept-verse mapping for ML refinement."""
    db = _get_db()
    if feedback_type not in ("relevant", "irrelevant", "partial"):
        raise HTTPException(status_code=400, detail="feedback_type must be relevant, irrelevant, or partial")
    try:
        db.execute_update(
            "INSERT INTO concept_feedback (concept_name, verse_id, feedback_type) VALUES (?, ?, ?)",
            (concept_name, verse_id, feedback_type),
        )
        return {"status": "ok", "message": f"Feedback recorded for {concept_name}"}
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}") from e


@router.get("/concepts/{concept_name}/feedback/summary", tags=["phase9"])
async def get_concept_feedback_summary(concept_name: str) -> Dict[str, Any]:
    """Get aggregated feedback for a concept's verse mappings."""
    db = _get_db()
    try:
        rows = db.execute_query(
            "SELECT feedback_type, COUNT(*) FROM concept_feedback WHERE concept_name = ? GROUP BY feedback_type",
            (concept_name,),
        )
        summary: Dict[str, int] = {}
        for r in rows:
            summary[str(r[0])] = int(r[1])
        return {"concept_name": concept_name, "feedback": summary}
    except sqlite3.OperationalError:
        return {"concept_name": concept_name, "feedback": {}}


# --- Phase 9D: Mobile Native App Endpoints ---


@router.post("/mobile/sync", response_model=MobileSyncResponse, tags=["phase9"])
async def mobile_sync(body: MobileSyncRequest) -> MobileSyncResponse:
    """Sync optimized verse data for mobile offline use.

    Returns compact verse objects with optional word data for
    the requested books.
    """
    db = _get_db()
    import datetime

    sync_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    verses: List[MobileVerseResponse] = []

    book_ids = body.book_ids if body.book_ids else [1]  # Default to Genesis
    for bid in book_ids[:5]:  # Limit to 5 books per sync
        try:
            rows = db.execute_query(
                "SELECT b.name, v.chapter, v.verse, v.text "
                "FROM verses v "
                "JOIN books b ON v.book_id = b.book_id AND v.translation_id = b.translation_id "
                "WHERE v.translation_id = 'engbsb' AND v.book_id = ? "
                "ORDER BY v.chapter, v.verse",
                (bid,),
            )
            for r in rows:
                mv = MobileVerseResponse(
                    ref=f"{r[0]} {r[1]}:{r[2]}",
                    text=str(r[3]),
                    tid="engbsb",
                )
                if body.include_words:
                    try:
                        word_rows = db.execute_query(
                            "SELECT word_num, transliteration, translation, strongs_primary "
                            "FROM words WHERE book = ? AND chapter = ? AND verse = ? ORDER BY word_num",
                            (str(r[0]), int(r[1]), int(r[2])),
                        )
                        mv.words = [{"n": w[0], "t": w[1], "g": w[2], "s": w[3]} for w in word_rows]
                    except sqlite3.OperationalError:
                        pass
                verses.append(mv)
        except sqlite3.OperationalError:
            continue

    return MobileSyncResponse(
        sync_timestamp=sync_ts,
        verses=verses,
        total_verses=len(verses),
    )
