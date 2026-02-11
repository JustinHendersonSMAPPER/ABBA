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
    DepthLevel,
    LexiconEntry,
    MorphologyInfo,
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

    response = VerseResponse(
        reference=f"{result.book_name or book_id} {chapter}:{verse}",
        book_name=result.book_name or str(book_id),
        chapter=chapter,
        verse=verse,
        text=result.text,
        translation_id=translation_id,
    )

    if depth in (DepthLevel.STANDARD, DepthLevel.DEEP, DepthLevel.SCHOLARLY):
        response.words = _get_words_for_verse(str(book_id), chapter, verse)

    if depth in (DepthLevel.DEEP, DepthLevel.SCHOLARLY):
        # Placeholder for enrichment data — populated once enrichment tables exist
        response.cultural_context = []
        response.cross_references = []
        response.concepts = []

    if depth == DepthLevel.SCHOLARLY:
        analysis = _get_analysis()
        parallels = analysis.parallel_passage_detection(str(book_id), chapter, verse)
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


# --- Internal helpers ---


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


def _parse_json_list(value: Optional[str]) -> List[str]:
    """Parse a JSON array string into a list, or return empty list."""
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
