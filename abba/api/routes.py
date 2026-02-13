"""FastAPI route definitions for the ABBA Bible Study API."""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..config import ABBAConfig, ConfigManager
from ..database import SQLiteManager
from .analysis import AnalysisAPI
from .models import (
    APIInfo,
    BookInfo,
    CrossRef,
    CulturalNote,
    DepthLevel,
    LexiconEntry,
    LifeTopicDetail,
    LifeTopicSummary,
    MorphologyInfo,
    PassageInfo,
    ReadingPlanDetail,
    ReadingPlanEntry,
    ReadingPlanSummary,
    RichnessFlag,
    SemanticSearchResult,
    StrongsResult,
    TextSearchResult,
    ThemeGroup,
    TopicalResult,
    TopicSummary,
    TranslationComparison,
    VerseResponse,
    WordAnalysis,
    WordDetail,
)
from .query_parser import parse_query
from .search import SearchAPI

router = APIRouter(prefix="/api/v1", tags=["bible"])


# --- Singleton state container (avoids pylint global-statement) ---


class _AppState:
    """Holds lazily-initialized singleton instances for the API layer."""

    db_manager: Optional[SQLiteManager] = None
    search_api: Optional[SearchAPI] = None
    analysis_api: Optional[AnalysisAPI] = None


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


def configure_db(db_manager: SQLiteManager) -> None:
    """Configure the routes with an existing database manager.

    Args:
        db_manager: Pre-initialized SQLiteManager instance.
    """
    _state.db_manager = db_manager
    _state.search_api = SearchAPI(db_manager)
    _state.analysis_api = AnalysisAPI(db_manager)


# --- Root ---


@router.get("/", response_model=APIInfo, tags=["info"])
async def api_root() -> APIInfo:
    """Return API metadata."""
    return APIInfo()


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
        response.words = _get_words_for_verse(book_name or str(book_id), chapter, verse)
        response.richness_flags = _get_richness_flags(book_name or str(book_id), chapter, verse)

    if depth in (DepthLevel.DEEP, DepthLevel.SCHOLARLY):
        response.cross_references = _get_cross_refs(book_id, chapter, verse)
        response.cultural_context = _get_cultural_context(book_id, chapter, verse)
        response.passage_info = _get_passage_info(book_id, chapter, verse)
        response.literary_structures = _get_literary_structures(book_id, chapter, verse)
        response.concepts = []

    if depth == DepthLevel.SCHOLARLY:
        analysis = _get_analysis()
        parallels = analysis.parallel_passage_detection(book_name or str(book_id), chapter, verse)
        response.parallel_passages = parallels

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
    return TranslationComparison(
        reference=result["reference"],
        translations=result.get("translations", {}),
        original_words=words,
    )


# --- Search Endpoints ---


@router.get("/search/text", response_model=List[TextSearchResult])
async def text_search(
    q: str = Query(..., description="Text search query"),
    translation_id: str = Query("engbsb", description="Translation ID"),
    limit: int = Query(50, ge=1, le=200),
) -> List[TextSearchResult]:
    """Full-text search within a specific translation."""
    search = _get_search()
    results = search.search_verses(translation_id, q, limit)
    return [
        TextSearchResult(
            translation_id=r.translation_id,
            book_id=r.book_id,
            chapter=r.chapter,
            verse=r.verse,
            text=r.text,
            book_name=r.book_name,
        )
        for r in results
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

    db = _get_db()
    results: List[SemanticSearchResult] = []

    # FTS search
    try:
        fts_rows = db.search_verses(translation_id, search_text, limit * 2)
        for rank, row in enumerate(fts_rows):
            results.append(
                SemanticSearchResult(
                    book_id=row["book_id"],
                    chapter=row["chapter"],
                    verse=row["verse"],
                    text=row["text"],
                    book_name=row["book_name"] if "book_name" in (row.keys() if hasattr(row, "keys") else []) else "",
                    score=round(1.0 - (rank / max(len(fts_rows), 1)), 3),
                    match_type="exact",
                    explanation=f"Text match (rank {rank + 1})",
                    translation_id=translation_id,
                )
            )
    except Exception:  # noqa: BLE001
        pass

    # Apply filters
    if testament_override:
        t = "old" if testament_override in ("old", "ot") else "new"
        old_books = set(range(1, 40))
        new_books = set(range(40, 67))
        allowed = old_books if t == "old" else new_books
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
            "SELECT concept_name, relevance_aspect FROM life_topic_concepts "
            "WHERE topic_id = ? ORDER BY display_order",
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


def _parse_json_list(value: Optional[str]) -> List[Any]:
    """Parse a JSON array string into a list, or return empty list."""
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
