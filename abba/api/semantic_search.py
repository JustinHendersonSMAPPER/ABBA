"""Semantic search API for ABBA biblical analysis.

Extends the base SearchAPI with embedding-based semantic search,
related word discovery, and hybrid (exact + semantic) search.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..database import SQLiteManager
from ..embeddings.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for search results."""

    def __init__(self, max_size: int = 256) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache, moving it to front."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Store a value, evicting LRU entry if full."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)


@dataclass
class SemanticVerseResult:
    """A verse returned from semantic search."""

    book_id: int
    chapter: int
    verse: int
    similarity: float
    testament: str = ""
    language: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelatedWordResult:
    """A word returned from semantic word search."""

    strongs_number: str
    similarity: float
    word: str = ""
    transliteration: str = ""
    gloss: str = ""
    language: str = ""
    part_of_speech: str = ""


@dataclass
class HybridSearchResult:
    """A result combining exact and semantic search."""

    book_id: int
    chapter: int
    verse: int
    text: str = ""
    translation_id: str = ""
    book_name: str = ""
    score: float = 0.0
    match_type: str = ""  # "exact", "semantic", "both"
    semantic_similarity: float = 0.0
    exact_rank: int = 0
    explanation: str = ""


class SemanticSearchAPI:
    """Provides semantic search functionality using ChromaDB embeddings."""

    # Book ID to name mapping for verse ID conversion
    BOOK_ID_TO_NAME: Dict[int, str] = {
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

    def __init__(
        self,
        db_manager: SQLiteManager,
        chroma_manager: ChromaManager,
        model_manager: Optional[Any] = None,
    ) -> None:
        """Initialize semantic search.

        Args:
            db_manager: SQLite database manager.
            chroma_manager: ChromaDB vector manager.
            model_manager: Optional EmbeddingModelManager for query encoding.
        """
        self.db = db_manager
        self.chroma = chroma_manager
        self.models = model_manager
        self._cache = LRUCache(max_size=256)

    # ------------------------------------------------------------------ #
    #  search_similar_verses                                               #
    # ------------------------------------------------------------------ #

    def search_similar_verses(
        self,
        query_text: str,
        n_results: int = 20,
        similarity_threshold: float = 0.3,
        testament_filter: Optional[str] = None,
        book_filter: Optional[int] = None,
    ) -> List[SemanticVerseResult]:
        """Find verses semantically similar to a query using original-language embeddings.

        Args:
            query_text: Natural language query (English).
            n_results: Maximum results to return.
            similarity_threshold: Minimum cosine similarity (0-1).
            testament_filter: Optional "old" or "new".
            book_filter: Optional book ID to restrict search.

        Returns:
            Sorted list of SemanticVerseResult.
        """
        if self.models is None:
            logger.warning("No model manager — cannot encode query for semantic search")
            return []

        collection = self.chroma.get_collection("original_verses")
        if collection is None:
            logger.warning("original_verses collection not found")
            return []

        # Encode query with the multilingual model (same model used for original verses)
        query_embedding = self.models.encode_single(query_text, model_type="multilingual")

        # Build metadata filter
        where_filter = self._build_where_filter(testament_filter, book_filter)

        # Query ChromaDB
        try:
            raw = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results * 2, 200),  # over-fetch then filter
                where=where_filter if where_filter else None,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        if not raw["ids"] or not raw["ids"][0]:
            return []

        results: List[SemanticVerseResult] = []
        for i, _verse_id in enumerate(raw["ids"][0]):
            distance = raw["distances"][0][i] if raw["distances"] else 0.0
            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue

            metadata = raw["metadatas"][0][i] if raw["metadatas"] else {}
            results.append(
                SemanticVerseResult(
                    book_id=int(metadata.get("book_id", 0)),
                    chapter=int(metadata.get("chapter", 0)),
                    verse=int(metadata.get("verse", 0)),
                    similarity=round(similarity, 4),
                    testament=str(metadata.get("testament", "")),
                    language=str(metadata.get("language", "")),
                    metadata=dict(metadata),
                )
            )

        results.sort(key=lambda r: -r.similarity)
        return results[:n_results]

    # ------------------------------------------------------------------ #
    #  search_related_words                                                #
    # ------------------------------------------------------------------ #

    def search_related_words(
        self,
        query_text: str,
        n_results: int = 20,
        similarity_threshold: float = 0.3,
        language_filter: Optional[str] = None,
    ) -> List[RelatedWordResult]:
        """Find words semantically related to a query.

        Args:
            query_text: Natural language query.
            n_results: Maximum results to return.
            similarity_threshold: Minimum cosine similarity.
            language_filter: Optional "hebrew", "greek", or "aramaic".

        Returns:
            Sorted list of RelatedWordResult.
        """
        if self.models is None:
            logger.warning("No model manager — cannot encode query for word search")
            return []

        collection = self.chroma.get_collection("words")
        if collection is None:
            logger.warning("words collection not found")
            return []

        query_embedding = self.models.encode_single(query_text, model_type="multilingual")

        where_filter: Optional[Dict[str, Any]] = None
        if language_filter:
            where_filter = {"language": language_filter}

        try:
            raw = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results * 2, 200),
                where=where_filter,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB word query failed: %s", e)
            return []

        if not raw["ids"] or not raw["ids"][0]:
            return []

        results: List[RelatedWordResult] = []
        for i, word_id in enumerate(raw["ids"][0]):
            distance = raw["distances"][0][i] if raw["distances"] else 0.0
            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue

            metadata = raw["metadatas"][0][i] if raw["metadatas"] else {}
            results.append(
                RelatedWordResult(
                    strongs_number=str(metadata.get("strongs", word_id.split(":")[0])),
                    similarity=round(similarity, 4),
                    word=str(metadata.get("word", "")),
                    transliteration=str(metadata.get("transliteration", "")),
                    gloss=str(metadata.get("gloss", "")),
                    language=str(metadata.get("language", "")),
                    part_of_speech=str(metadata.get("part_of_speech", "")),
                )
            )

        results.sort(key=lambda r: -r.similarity)
        return results[:n_results]

    # ------------------------------------------------------------------ #
    #  hybrid_search                                                       #
    # ------------------------------------------------------------------ #

    def hybrid_search(
        self,
        query_text: str,
        translation_id: str = "engbsb",
        n_results: int = 20,
        similarity_threshold: float = 0.3,
        exact_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[HybridSearchResult]:
        """Combine full-text search with semantic search for best results.

        Args:
            query_text: Search query (used for both FTS and semantic).
            translation_id: Translation for exact search.
            n_results: Maximum results.
            similarity_threshold: Minimum semantic similarity.
            exact_weight: Weight for exact match score (0-1).
            semantic_weight: Weight for semantic similarity (0-1).

        Returns:
            Sorted list of HybridSearchResult combining both search types.
        """
        combined: Dict[str, HybridSearchResult] = {}

        # --- Exact (FTS) search ---
        fts_results = self._fts_search(query_text, translation_id, limit=n_results * 2)
        for rank, fts in enumerate(fts_results):
            key = f"{fts['book_id']}:{fts['chapter']}:{fts['verse']}"
            combined[key] = HybridSearchResult(
                book_id=fts["book_id"],
                chapter=fts["chapter"],
                verse=fts["verse"],
                text=fts.get("text", ""),
                translation_id=translation_id,
                book_name=fts.get("book_name", ""),
                score=0.0,
                match_type="exact",
                exact_rank=rank + 1,
            )

        # --- Semantic search ---
        semantic_results = self.search_similar_verses(
            query_text, n_results=n_results * 2, similarity_threshold=similarity_threshold
        )

        for sem in semantic_results:
            key = f"{sem.book_id}:{sem.chapter}:{sem.verse}"
            if key in combined:
                combined[key].match_type = "both"
                combined[key].semantic_similarity = sem.similarity
            else:
                # Look up verse text from database
                text = self._get_verse_text(translation_id, sem.book_id, sem.chapter, sem.verse)
                combined[key] = HybridSearchResult(
                    book_id=sem.book_id,
                    chapter=sem.chapter,
                    verse=sem.verse,
                    text=text,
                    translation_id=translation_id,
                    book_name=self.BOOK_ID_TO_NAME.get(sem.book_id, ""),
                    score=0.0,
                    match_type="semantic",
                    semantic_similarity=sem.similarity,
                )

        # --- Score, rank, and explain ---
        max_fts_rank = len(fts_results) or 1
        for result in combined.values():
            exact_score = (1.0 - (result.exact_rank / max_fts_rank)) if result.exact_rank > 0 else 0.0
            sem_score = result.semantic_similarity

            if result.match_type == "both":
                result.score = exact_weight * exact_score + semantic_weight * sem_score
                result.explanation = (
                    f"Matched both text (rank {result.exact_rank}) and meaning (similarity {sem_score:.0%})"
                )
            elif result.match_type == "exact":
                result.score = exact_weight * exact_score
                result.explanation = f"Text match (rank {result.exact_rank} of {max_fts_rank})"
            else:
                result.score = semantic_weight * sem_score
                result.explanation = f"Semantic similarity {sem_score:.0%} to query meaning"

        ranked = sorted(combined.values(), key=lambda r: -r.score)
        return ranked[:n_results]

    # ------------------------------------------------------------------ #
    #  Query expansion                                                     #
    # ------------------------------------------------------------------ #

    def expand_query_with_strongs(self, strongs_number: str, n_related: int = 5) -> List[str]:
        """Expand a search by finding words related to a Strong's number.

        Args:
            strongs_number: Starting Strong's number (e.g., "H0430").
            n_related: Number of related words to find.

        Returns:
            List of related Strong's numbers.
        """
        collection = self.chroma.get_collection("words")
        if collection is None:
            return []

        # Get the embedding for this Strong's number
        try:
            existing = collection.get(ids=[strongs_number], include=["embeddings"])
            if not existing["embeddings"]:
                return []

            embedding = existing["embeddings"][0]
            raw = collection.query(
                query_embeddings=[embedding],
                n_results=n_related + 1,  # +1 to exclude self
                include=["metadatas", "distances"],
            )

            if not raw["ids"] or not raw["ids"][0]:
                return []

            related = []
            for i, word_id in enumerate(raw["ids"][0]):
                if word_id == strongs_number:
                    continue
                metadata = raw["metadatas"][0][i] if raw["metadatas"] else {}
                strongs = str(metadata.get("strongs", word_id.split(":")[0]))
                if strongs and strongs not in related:
                    related.append(strongs)

            return related[:n_related]

        except Exception as e:
            logger.error("Query expansion failed: %s", e)
            return []

    def get_cache_stats(self) -> Dict[str, int]:
        """Return search cache statistics."""
        return {"hits": self._cache.hits, "misses": self._cache.misses, "size": self._cache.size}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_where_filter(
        testament_filter: Optional[str] = None,
        book_filter: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB metadata filter."""
        conditions: List[Dict[str, Any]] = []

        if testament_filter:
            conditions.append({"testament": testament_filter})
        if book_filter is not None:
            conditions.append({"book_id": book_filter})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _fts_search(self, query_text: str, translation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Run full-text search against the verses table."""
        results: List[Dict[str, Any]] = []
        try:
            rows = self.db.search_verses(translation_id, query_text, limit)
            for row in rows:
                row_keys = row.keys() if hasattr(row, "keys") else []
                results.append(
                    {
                        "book_id": row["book_id"],
                        "chapter": row["chapter"],
                        "verse": row["verse"],
                        "text": row["text"],
                        "book_name": row["book_name"] if "book_name" in row_keys else "",
                    }
                )
        except Exception as e:
            logger.error("FTS search failed: %s", e)
        return results

    def _get_verse_text(self, translation_id: str, book_id: int, chapter: int, verse: int) -> str:
        """Look up verse text from the database."""
        try:
            result = self.db.get_verse(translation_id, book_id, chapter, verse)
            if result:
                return str(result["text"])
        except Exception as e:
            logger.debug("Could not look up verse text: %s", e)
        return ""
