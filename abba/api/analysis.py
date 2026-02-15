"""Linguistic analysis API for ABBA biblical analysis."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from ..database import SQLiteManager


@dataclass
class MorphologyPattern:
    """Represents a morphological pattern analysis result."""

    pattern: str
    description: str
    count: int
    examples: List[Dict[str, Any]]


@dataclass
class WordFrequency:
    """Represents word frequency analysis result."""

    word: str
    strongs_number: Optional[str]
    frequency: int
    books: Set[str]
    first_occurrence: Dict[str, Any]
    last_occurrence: Dict[str, Any]


@dataclass
class LexicalCluster:
    """Represents a group of related words."""

    root: str
    strongs_numbers: List[str]
    words: List[Dict[str, Any]]
    semantic_domain: Optional[str]


class AnalysisAPI:
    """Provides linguistic analysis functionality for biblical texts."""

    def __init__(self, db_manager: SQLiteManager):
        """Initialize analysis API.

        Args:
            db_manager: SQLite database manager
        """
        self.db_manager = db_manager

    def analyze_morphology_patterns(
        self, language: str = "hebrew", pattern: Optional[str] = None, limit: int = 50
    ) -> List[MorphologyPattern]:
        """Analyze morphological patterns in the text.

        Args:
            language: Language to analyze ('hebrew' or 'greek')
            pattern: Specific pattern to search for (e.g., 'V%' for verbs)
            limit: Maximum number of results

        Returns:
            List of morphology patterns with counts and examples
        """
        query = """
            SELECT
                w.morphology_code,
                m.description,
                COUNT(*) as count,
                GROUP_CONCAT(w.word_ref || ':' ||
                    COALESCE(w.hebrew_text, w.greek_text), '|') as examples
            FROM words w
            LEFT JOIN morphology m ON w.morphology_code = m.code
            WHERE w.language = ?
        """

        params: List[Union[str, int]] = [language]
        if pattern:
            query += " AND w.morphology_code LIKE ?"
            params.append(pattern)

        query += """
            GROUP BY w.morphology_code
            ORDER BY count DESC
            LIMIT ?
        """
        params.append(limit)

        results = []
        rows = self.db_manager.execute_query(query, tuple(params))

        for row in rows:
            morph_code, description, count, examples_str = row
            # Parse examples
            examples = []
            if examples_str:
                for example in examples_str.split("|")[:5]:  # Limit to 5 examples
                    if ":" in example:
                        ref, text = example.split(":", 1)
                        examples.append({"reference": ref, "text": text})

            results.append(
                MorphologyPattern(
                    pattern=morph_code or "Unknown",
                    description=description or "No description",
                    count=count,
                    examples=examples,
                )
            )

        return results

    def word_frequency_analysis(
        self, strongs_pattern: Optional[str] = None, min_frequency: int = 1, limit: int = 100
    ) -> List[WordFrequency]:
        """Analyze word frequency across the biblical corpus.

        Args:
            strongs_pattern: Pattern to filter Strong's numbers (e.g., 'H%' for Hebrew)
            min_frequency: Minimum frequency threshold
            limit: Maximum number of results

        Returns:
            List of word frequency data
        """
        query = """
            SELECT
                w.strongs_primary,
                COALESCE(l.original_word, w.hebrew_text, w.greek_text) as word,
                COUNT(*) as frequency,
                GROUP_CONCAT(DISTINCT w.book) as books,
                MIN(w.book || '.' || w.chapter || '.' || w.verse) as first_ref,
                MAX(w.book || '.' || w.chapter || '.' || w.verse) as last_ref
            FROM words w
            LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
            WHERE w.strongs_primary IS NOT NULL
        """

        params: List[Union[str, int]] = []
        if strongs_pattern:
            query += " AND w.strongs_primary LIKE ?"
            params.append(strongs_pattern)

        query += """
            GROUP BY w.strongs_primary
            HAVING frequency >= ?
            ORDER BY frequency DESC
            LIMIT ?
        """
        params.extend([min_frequency, limit])

        results = []
        rows = self.db_manager.execute_query(query, tuple(params))

        for row in rows:
            strongs, word, freq, books_str, first_ref, last_ref = row
            books = set(books_str.split(",")) if books_str else set()

            results.append(
                WordFrequency(
                    word=word or strongs,
                    strongs_number=strongs,
                    frequency=freq,
                    books=books,
                    first_occurrence={"reference": first_ref},
                    last_occurrence={"reference": last_ref},
                )
            )

        return results

    def find_hapax_legomena(self, language: str = "hebrew") -> List[Dict[str, Any]]:
        """Find hapax legomena (words appearing only once).

        Args:
            language: Language to analyze

        Returns:
            List of words appearing only once with their context
        """
        query = """
            SELECT
                w.strongs_primary,
                COALESCE(w.hebrew_text, w.greek_text) as word,
                w.transliteration,
                w.translation,
                w.book || '.' || w.chapter || '.' || w.verse as reference,
                l.gloss,
                l.definition
            FROM words w
            LEFT JOIN lexicon l ON w.strongs_primary = l.strongs_number
            WHERE w.language = ? AND w.strongs_primary IS NOT NULL
            GROUP BY w.strongs_primary
            HAVING COUNT(*) = 1
            ORDER BY w.book, w.chapter, w.verse
        """

        results = []
        rows = self.db_manager.execute_query(query, (language,))

        for row in rows:
            strongs, word, translit, trans, ref, gloss, definition = row
            results.append(
                {
                    "strongs_number": strongs,
                    "word": word,
                    "transliteration": translit,
                    "translation": trans,
                    "reference": ref,
                    "gloss": gloss,
                    "definition": definition,
                }
            )

        return results

    def analyze_word_clusters(self, root_pattern: str) -> List[LexicalCluster]:
        """Analyze clusters of related words based on root patterns.

        Args:
            root_pattern: Root pattern to search for in Strong's numbers

        Returns:
            List of lexical clusters
        """
        # First, find all Strong's numbers matching the pattern
        query = """
            SELECT DISTINCT
                l.strongs_number,
                l.original_word,
                l.gloss,
                l.part_of_speech
            FROM lexicon l
            WHERE l.strongs_number LIKE ?
            ORDER BY l.strongs_number
        """

        clusters = defaultdict(list)
        rows = self.db_manager.execute_query(query, (root_pattern,))

        for row in rows:
            strongs, word, gloss, pos = row
            # Extract root (first 4 characters of Strong's number)
            root = strongs[:5] if len(strongs) >= 5 else strongs
            clusters[root].append(
                {
                    "strongs_number": strongs,
                    "word": word,
                    "gloss": gloss,
                    "part_of_speech": pos,
                }
            )

        # Convert to LexicalCluster objects
        results = []
        for root, words in clusters.items():
            if len(words) > 1:  # Only include actual clusters
                results.append(
                    LexicalCluster(
                        root=root,
                        strongs_numbers=[w["strongs_number"] for w in words],
                        words=words,
                        semantic_domain=None,  # Could be enhanced with domain detection
                    )
                )

        return results

    def compare_translations(self, book: str, chapter: int, verse: int, translation_ids: List[str]) -> Dict[str, Any]:
        """Compare verse across multiple translations with linguistic analysis.

        Args:
            book: Book name
            chapter: Chapter number
            verse: Verse number
            translation_ids: List of translation IDs to compare

        Returns:
            Comparison data including original languages
        """
        result: Dict[str, Any] = {
            "reference": f"{book} {chapter}:{verse}",
            "translations": {},
            "original_words": [],
        }

        # Get original language words
        word_query = """
            SELECT
                word_num,
                COALESCE(hebrew_text, greek_text) as original,
                transliteration,
                translation,
                strongs_primary,
                morphology_code,
                language
            FROM words
            WHERE book = ? AND chapter = ? AND verse = ?
            ORDER BY word_num
        """

        word_rows = self.db_manager.execute_query(word_query, (book, chapter, verse))
        for row in word_rows:
            word_num, original, translit, trans, strongs, morph, lang = row
            result["original_words"].append(
                {
                    "position": word_num,
                    "text": original,
                    "transliteration": translit,
                    "translation": trans,
                    "strongs": strongs,
                    "morphology": morph,
                    "language": lang,
                }
            )

        # Get translations
        placeholders = ",".join(["?"] * len(translation_ids))
        verse_query = f"""
            SELECT translation_id, text
            FROM verses
            WHERE book_id = (SELECT book_id FROM books WHERE name = ? LIMIT 1)
            AND chapter = ? AND verse = ?
            AND translation_id IN ({placeholders})
        """

        params = [book, chapter, verse] + translation_ids
        verse_rows = self.db_manager.execute_query(verse_query, tuple(params))

        for trans_id, text in verse_rows:
            result["translations"][trans_id] = text

        return result

    def analyze_grammatical_constructions(
        self, construction_type: str, language: str = "hebrew"
    ) -> List[Dict[str, Any]]:
        """Analyze specific grammatical constructions.

        Args:
            construction_type: Type of construction ('construct', 'infinitive', etc.)
            language: Language to analyze

        Returns:
            List of grammatical construction examples
        """
        # Map construction types to morphology patterns
        construction_patterns = {
            "construct": "%c%",  # Construct state
            "infinitive": "%i%",  # Infinitive
            "participle": "%p%",  # Participle
            "imperative": "%m%",  # Imperative
            "perfect": "%qp%",  # Perfect tense
            "imperfect": "%qi%",  # Imperfect tense
        }

        pattern = construction_patterns.get(construction_type, f"%{construction_type}%")

        query = """
            SELECT
                w.word_ref,
                COALESCE(w.hebrew_text, w.greek_text) as text,
                w.transliteration,
                w.translation,
                w.morphology_code,
                m.description
            FROM words w
            LEFT JOIN morphology m ON w.morphology_code = m.code
            WHERE w.language = ?
            AND LOWER(w.morphology_code) LIKE LOWER(?)
            LIMIT 100
        """

        results = []
        rows = self.db_manager.execute_query(query, (language, pattern))

        for row in rows:
            ref, text, translit, trans, morph, desc = row
            results.append(
                {
                    "reference": ref,
                    "text": text,
                    "transliteration": translit,
                    "translation": trans,
                    "morphology": morph,
                    "description": desc,
                }
            )

        return results

    def semantic_domain_analysis(self, domain: str) -> List[Dict[str, Any]]:
        """Analyze words belonging to a semantic domain.

        Args:
            domain: Semantic domain to analyze (e.g., 'love', 'covenant', 'temple')

        Returns:
            List of words in the semantic domain
        """
        # Search for words related to the domain in definitions and glosses
        query = """
            SELECT DISTINCT
                l.strongs_number,
                l.original_word,
                l.transliteration,
                l.gloss,
                l.definition,
                l.language,
                COUNT(w.id) as usage_count
            FROM lexicon l
            LEFT JOIN words w ON l.strongs_number = w.strongs_primary
            WHERE LOWER(l.gloss) LIKE LOWER(?)
               OR LOWER(l.definition) LIKE LOWER(?)
            GROUP BY l.strongs_number
            ORDER BY usage_count DESC
            LIMIT 50
        """

        search_pattern = f"%{domain}%"
        results = []
        rows = self.db_manager.execute_query(query, (search_pattern, search_pattern))

        for row in rows:
            strongs, word, translit, gloss, definition, lang, count = row
            results.append(
                {
                    "strongs_number": strongs,
                    "word": word,
                    "transliteration": translit,
                    "gloss": gloss,
                    "definition": definition,
                    "language": lang,
                    "usage_count": count,
                }
            )

        return results

    def parallel_passage_detection(
        self, book: str, chapter: int, verse: int, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Detect parallel passages based on shared vocabulary.

        Args:
            book: Book name
            chapter: Chapter number
            verse: Verse number
            threshold: Similarity threshold (0-1)

        Returns:
            List of potentially parallel passages
        """
        # Get Strong's numbers from the source verse
        source_query = """
            SELECT DISTINCT strongs_primary
            FROM words
            WHERE book = ? AND chapter = ? AND verse = ?
            AND strongs_primary IS NOT NULL
        """

        source_strongs = set()
        rows = self.db_manager.execute_query(source_query, (book, chapter, verse))
        for (strongs,) in rows:
            source_strongs.add(strongs)

        if not source_strongs:
            return []

        # Find verses with similar vocabulary
        strongs_placeholders = ",".join(["?"] * len(source_strongs))
        parallel_query = f"""
            SELECT
                book, chapter, verse,
                COUNT(DISTINCT strongs_primary) as shared_count,
                GROUP_CONCAT(DISTINCT strongs_primary) as shared_strongs
            FROM words
            WHERE strongs_primary IN ({strongs_placeholders})
            AND NOT (book = ? AND chapter = ? AND verse = ?)
            GROUP BY book, chapter, verse
            HAVING shared_count >= ?
            ORDER BY shared_count DESC
            LIMIT 20
        """

        min_shared = int(len(source_strongs) * threshold)
        params = list(source_strongs) + [book, chapter, verse, min_shared]

        results = []
        rows = self.db_manager.execute_query(parallel_query, tuple(params))

        for row in rows:
            p_book, p_chapter, p_verse, shared_count, shared_strongs_str = row
            similarity = shared_count / len(source_strongs)

            results.append(
                {
                    "reference": f"{p_book} {p_chapter}:{p_verse}",
                    "shared_words": shared_count,
                    "similarity": round(similarity, 3),
                    "shared_strongs": shared_strongs_str.split(",") if shared_strongs_str else [],
                }
            )

        return results
