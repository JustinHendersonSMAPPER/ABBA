"""Context builder for creating rich embedding contexts from biblical data."""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds enhanced contexts for embeddings using linguistic and semantic data."""
    
    def __init__(self, db_manager):
        """Initialize with database manager.
        
        Args:
            db_manager: SQLiteManager instance for data access
        """
        self.db = db_manager
    
    def build_verse_context(
        self,
        translation_id: str,
        book_id: int,
        chapter: int,
        verse: int
    ) -> str:
        """Build enhanced context for a verse embedding.
        
        Combines verse text with original language, morphology, and key terms
        to create a rich context for embedding generation.
        
        Args:
            translation_id: Translation identifier
            book_id: Canonical book number (1-66)
            chapter: Chapter number
            verse: Verse number
            
        Returns:
            Enhanced context string for embedding
        """
        context_parts = []
        
        # Get verse text
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get verse text
            cursor.execute("""
                SELECT text FROM verses
                WHERE translation_id = ? AND book_id = ? 
                AND chapter = ? AND verse = ?
            """, (translation_id, book_id, chapter, verse))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Verse not found: {translation_id} {book_id}:{chapter}:{verse}")
                return ""
            
            verse_text = result[0]
            
            # Get book name for context
            cursor.execute("""
                SELECT name FROM books
                WHERE translation_id = ? AND book_id = ?
            """, (translation_id, book_id))
            
            book_result = cursor.fetchone()
            book_name = book_result[0] if book_result else f"Book{book_id}"
            
            # Add reference as context
            context_parts.append(f"{book_name} {chapter}:{verse}")
            
            # Add verse text
            context_parts.append(verse_text)
            
            # Get original language data
            # Map book_id to three-letter book code
            book_code = self._get_book_code(book_id)
            
            if book_code:
                cursor.execute("""
                    SELECT 
                        word_num,
                        hebrew_text,
                        greek_text,
                        transliteration,
                        strongs_primary,
                        translation,
                        morphology_code
                    FROM words
                    WHERE book = ? AND chapter = ? AND verse = ?
                    ORDER BY word_num
                """, (book_code, chapter, verse))
                
                words = cursor.fetchall()
                
                if words:
                    # Build original text
                    original_texts = []
                    for word in words:
                        original = word[1] or word[2]  # hebrew_text or greek_text
                        if original:
                            original_texts.append(original)
                    
                    if original_texts:
                        context_parts.append(f"Original: {' '.join(original_texts)}")
                    
                    # Get key terms with meanings
                    key_terms = self._extract_key_terms(cursor, words)
                    if key_terms:
                        context_parts.append(f"Keywords: {key_terms}")
                    
                    # Add grammatical summary
                    grammar_summary = self._get_grammar_summary(words)
                    if grammar_summary:
                        context_parts.append(f"Grammar: {grammar_summary}")
        
        return " | ".join(context_parts)
    
    def build_word_context(self, word_data: Dict[str, Any]) -> str:
        """Build enhanced context for a word embedding.
        
        Combines the word with its lexicon definition, morphology,
        and usage information.
        
        Args:
            word_data: Dictionary with word information
            
        Returns:
            Enhanced context string for embedding
        """
        context_parts = []
        
        # Get the actual word
        word_text = word_data.get('greek_text') or word_data.get('hebrew_text', '')
        if word_text:
            context_parts.append(f"Word: {word_text}")
        
        # Add transliteration
        if word_data.get('transliteration'):
            context_parts.append(f"({word_data['transliteration']})")
        
        # Add Strong's number
        if word_data.get('strongs_primary'):
            context_parts.append(word_data['strongs_primary'])
            
            # Get lexicon entry
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT gloss, definition, part_of_speech
                    FROM lexicon
                    WHERE strongs_number = ?
                """, (word_data['strongs_primary'],))
                
                lexicon = cursor.fetchone()
                if lexicon:
                    gloss, definition, pos = lexicon
                    
                    if gloss:
                        context_parts.append(f"Meaning: {gloss}")
                    
                    if pos:
                        context_parts.append(f"Part of speech: {pos}")
                    
                    if definition:
                        # Truncate long definitions
                        def_text = definition[:200] + "..." if len(definition) > 200 else definition
                        context_parts.append(f"Definition: {def_text}")
                
                # Add morphological information
                if word_data.get('morphology_code'):
                    morph_desc = self._get_morphology_description(
                        cursor, 
                        word_data['morphology_code'],
                        word_data.get('language', 'greek')
                    )
                    if morph_desc:
                        context_parts.append(f"Form: {morph_desc}")
                
                # Add usage context (sample verses where it appears)
                usage_samples = self._get_usage_samples(
                    cursor,
                    word_data['strongs_primary'],
                    limit=3
                )
                if usage_samples:
                    context_parts.append(f"Used in: {', '.join(usage_samples)}")
        
        return " | ".join(context_parts)
    
    def build_batch_contexts(
        self,
        items: List[Dict[str, Any]],
        context_type: str = "verse"
    ) -> List[str]:
        """Build contexts for a batch of items.
        
        Args:
            items: List of items (verses or words)
            context_type: Type of context to build ("verse" or "word")
            
        Returns:
            List of context strings
        """
        contexts = []
        
        for item in items:
            if context_type == "verse":
                context = self.build_verse_context(
                    item['translation_id'],
                    item['book_id'],
                    item['chapter'],
                    item['verse']
                )
            elif context_type == "word":
                context = self.build_word_context(item)
            else:
                raise ValueError(f"Unknown context type: {context_type}")
            
            contexts.append(context)
        
        return contexts
    
    def _get_book_code(self, book_id: int) -> Optional[str]:
        """Convert book_id to three-letter book code.
        
        Args:
            book_id: Canonical book number (1-66)
            
        Returns:
            Three-letter book code or None
        """
        # Standard book codes mapping
        book_codes = {
            1: "Gen", 2: "Exo", 3: "Lev", 4: "Num", 5: "Deu",
            6: "Jos", 7: "Jdg", 8: "Rut", 9: "1Sa", 10: "2Sa",
            11: "1Ki", 12: "2Ki", 13: "1Ch", 14: "2Ch", 15: "Ezr",
            16: "Neh", 17: "Est", 18: "Job", 19: "Psa", 20: "Pro",
            21: "Ecc", 22: "Sng", 23: "Isa", 24: "Jer", 25: "Lam",
            26: "Eze", 27: "Dan", 28: "Hos", 29: "Joe", 30: "Amo",
            31: "Oba", 32: "Jon", 33: "Mic", 34: "Nah", 35: "Hab",
            36: "Zep", 37: "Hag", 38: "Zec", 39: "Mal",
            40: "Mat", 41: "Mar", 42: "Luk", 43: "Jhn", 44: "Act",
            45: "Rom", 46: "1Co", 47: "2Co", 48: "Gal", 49: "Eph",
            50: "Php", 51: "Col", 52: "1Th", 53: "2Th", 54: "1Ti",
            55: "2Ti", 56: "Tit", 57: "Phm", 58: "Heb", 59: "Jam",
            60: "1Pe", 61: "2Pe", 62: "1Jn", 63: "2Jn", 64: "3Jn",
            65: "Jud", 66: "Rev"
        }
        
        return book_codes.get(book_id)
    
    def _extract_key_terms(self, cursor, words: List[tuple]) -> str:
        """Extract key terms with glosses from word list.
        
        Args:
            cursor: Database cursor
            words: List of word tuples from database
            
        Returns:
            Formatted string of key terms
        """
        key_terms = []
        seen_strongs = set()
        
        for word in words[:15]:  # Limit to first 15 words
            strongs = word[4]  # strongs_primary
            
            # Skip if already seen or if it's a common particle
            if (not strongs or 
                strongs in seen_strongs or 
                strongs.startswith('H9') or  # Hebrew particles
                strongs in ['G3588', 'G2532', 'G1161']):  # Common Greek articles/conjunctions
                continue
            
            seen_strongs.add(strongs)
            
            # Get gloss from lexicon
            cursor.execute("""
                SELECT gloss FROM lexicon
                WHERE strongs_number = ?
            """, (strongs,))
            
            result = cursor.fetchone()
            if result and result[0]:
                key_terms.append(f"{strongs}:{result[0]}")
            
            if len(key_terms) >= 8:  # Maximum 8 key terms
                break
        
        return ", ".join(key_terms)
    
    def _get_grammar_summary(self, words: List[tuple]) -> str:
        """Create grammatical summary from word list.
        
        Args:
            words: List of word tuples from database
            
        Returns:
            Grammar summary string
        """
        features = []
        
        # Count grammatical features
        verb_count = 0
        noun_count = 0
        tenses = set()
        moods = set()
        
        for word in words:
            translation = word[5]  # translation field
            if not translation:
                continue
            
            # Parse morphology from translation field
            if '=' in translation:
                morph_part = translation.split('=')[1]
                
                if morph_part.startswith('V-'):
                    verb_count += 1
                    # Extract tense
                    if 'P' in morph_part:
                        tenses.add('present')
                    elif 'A' in morph_part:
                        tenses.add('aorist')
                    elif 'I' in morph_part:
                        tenses.add('imperfect')
                    elif 'F' in morph_part:
                        tenses.add('future')
                    
                    # Extract mood
                    if 'M' in morph_part:
                        moods.add('imperative')
                    elif 'S' in morph_part:
                        moods.add('subjunctive')
                
                elif morph_part.startswith('N-'):
                    noun_count += 1
        
        # Build summary
        if verb_count > 0:
            verb_info = f"{verb_count} verb{'s' if verb_count > 1 else ''}"
            if tenses:
                verb_info += f" ({', '.join(sorted(tenses))})"
            features.append(verb_info)
        
        if noun_count > 0:
            features.append(f"{noun_count} noun{'s' if noun_count > 1 else ''}")
        
        if moods:
            features.append(f"mood: {', '.join(sorted(moods))}")
        
        return ", ".join(features) if features else ""
    
    def _get_morphology_description(
        self, 
        cursor, 
        morph_code: str, 
        language: str
    ) -> Optional[str]:
        """Get human-readable morphology description.
        
        Args:
            cursor: Database cursor
            morph_code: Morphology code
            language: Language (hebrew/greek)
            
        Returns:
            Description or None
        """
        cursor.execute("""
            SELECT description FROM morphology
            WHERE code = ? AND language = ?
        """, (morph_code, language))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _get_usage_samples(
        self, 
        cursor, 
        strongs_number: str, 
        limit: int = 3
    ) -> List[str]:
        """Get sample verse references where a word is used.
        
        Args:
            cursor: Database cursor
            strongs_number: Strong's number
            limit: Maximum samples to return
            
        Returns:
            List of verse references
        """
        cursor.execute("""
            SELECT DISTINCT book, chapter, verse
            FROM words
            WHERE strongs_primary = ?
            ORDER BY id
            LIMIT ?
        """, (strongs_number, limit))
        
        results = cursor.fetchall()
        references = []
        
        for book, chapter, verse in results:
            references.append(f"{book} {chapter}:{verse}")
        
        return references