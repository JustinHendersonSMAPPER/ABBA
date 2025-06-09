"""
OSHB (Open Scriptures Hebrew Bible) morphological data parser.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OSHBParser:
    """Parser for OSHB morphological data in JSON format."""
    
    def __init__(self, morphology_dir: Path = Path('data/sources/morphology/hebrew')):
        """
        Initialize OSHB parser.
        
        Args:
            morphology_dir: Directory containing Hebrew morphological JSON files
        """
        self.morphology_dir = morphology_dir
        self._morph_code_map = self._build_morph_code_map()
    
    def _build_morph_code_map(self) -> Dict[str, Dict[str, str]]:
        """Build mapping of morphology codes to features."""
        return {
            # Part of speech codes
            'N': {'pos': 'noun'},
            'V': {'pos': 'verb'},
            'A': {'pos': 'adjective'},
            'P': {'pos': 'pronoun'},
            'D': {'pos': 'adverb'},
            'T': {'pos': 'particle'},
            'C': {'pos': 'conjunction'},
            'R': {'pos': 'preposition'},
            
            # Gender codes
            'm': {'gender': 'masculine'},
            'f': {'gender': 'feminine'},
            'c': {'gender': 'common'},
            
            # Number codes
            's': {'number': 'singular'},
            'p': {'number': 'plural'},
            'd': {'number': 'dual'},
            
            # State codes
            'a': {'state': 'absolute'},
            'c': {'state': 'construct'},
            'e': {'state': 'emphatic'},
            
            # Stem codes (for verbs)
            'q': {'stem': 'qal'},
            'N': {'stem': 'niphal'},
            'p': {'stem': 'piel'},
            'P': {'stem': 'pual'},
            'h': {'stem': 'hiphil'},
            'H': {'stem': 'hophal'},
            't': {'stem': 'hithpael'},
            
            # Aspect codes (for verbs)
            'p': {'aspect': 'perfect'},
            'i': {'aspect': 'imperfect'},
            'w': {'aspect': 'wayyiqtol'},
            'a': {'aspect': 'imperative'},
            'j': {'aspect': 'jussive'},
            'c': {'aspect': 'cohortative'},
            
            # Person codes
            '1': {'person': '1'},
            '2': {'person': '2'},
            '3': {'person': '3'},
        }
    
    def parse_morph_code(self, morph_code: str) -> Dict[str, str]:
        """
        Parse OSHB morphology code into features.
        
        Args:
            morph_code: OSHB morphology code (e.g., "HNcmpa", "HVqp3ms")
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        if not morph_code:
            return features
        
        # Remove language prefix (H for Hebrew)
        if morph_code.startswith('H'):
            morph_code = morph_code[1:]
        
        # Handle compound codes with /
        if '/' in morph_code:
            parts = morph_code.split('/')
            # Parse each part and combine features
            for part in parts:
                part_features = self._parse_single_morph(part)
                # For compound words, mark the first part
                if parts.index(part) == 0:
                    part_features['compound'] = 'prefix'
                features.update(part_features)
        else:
            features = self._parse_single_morph(morph_code)
        
        return features
    
    def _parse_single_morph(self, code: str) -> Dict[str, str]:
        """Parse a single morphology code."""
        features = {}
        
        if not code:
            return features
        
        # First character is usually POS
        if code[0] in self._morph_code_map:
            features.update(self._morph_code_map[code[0]])
        
        # Special handling for different POS types
        if features.get('pos') == 'verb' and len(code) > 1:
            # Verb pattern: Vstem+aspect+person+gender+number
            pos = 1
            # Stem
            if pos < len(code) and code[pos] in ['q', 'N', 'p', 'P', 'h', 'H', 't']:
                features['stem'] = self._morph_code_map.get(code[pos], {}).get('stem', code[pos])
                pos += 1
            
            # Aspect/tense
            if pos < len(code) and code[pos] in ['p', 'i', 'w', 'a', 'j', 'c']:
                features['aspect'] = self._morph_code_map.get(code[pos], {}).get('aspect', code[pos])
                pos += 1
            
            # Person, gender, number
            while pos < len(code):
                if code[pos] in '123':
                    features['person'] = code[pos]
                elif code[pos] in self._morph_code_map:
                    features.update(self._morph_code_map[code[pos]])
                pos += 1
                
        elif features.get('pos') == 'noun' and len(code) > 1:
            # Noun pattern: Ntype+gender+number+state
            pos = 1
            # Common/proper
            if pos < len(code) and code[pos] in ['c', 'p']:
                features['noun_type'] = 'common' if code[pos] == 'c' else 'proper'
                pos += 1
            
            # Gender, number, state
            while pos < len(code):
                if code[pos] in self._morph_code_map:
                    features.update(self._morph_code_map[code[pos]])
                pos += 1
        else:
            # For other POS, parse remaining characters
            for char in code[1:]:
                if char in self._morph_code_map:
                    features.update(self._morph_code_map[char])
        
        return features
    
    def load_book(self, book_code: str) -> Optional[Dict]:
        """
        Load morphological data for a book.
        
        Args:
            book_code: Book code (e.g., "Gen", "Exod")
            
        Returns:
            Book data with parsed morphology or None if not found
        """
        file_path = self.morphology_dir / f"{book_code}.json"
        
        if not file_path.exists():
            logger.warning(f"Morphological data not found for {book_code}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load morphological data for {book_code}: {e}")
            return None
    
    def get_verse_words(self, book_code: str, chapter: int, verse: int) -> List[Dict[str, any]]:
        """
        Get words with parsed morphology for a specific verse.
        
        Args:
            book_code: Book code
            chapter: Chapter number
            verse: Verse number
            
        Returns:
            List of word dictionaries with parsed features
        """
        book_data = self.load_book(book_code)
        if not book_data:
            return []
        
        # Find the verse
        target_id = f"{book_code}.{chapter}.{verse}"
        
        for verse_data in book_data.get('verses', []):
            if verse_data.get('osisID') == target_id:
                words = []
                for word in verse_data.get('words', []):
                    # Parse morphology code
                    morph_features = self.parse_morph_code(word.get('morph', ''))
                    
                    word_info = {
                        'text': word.get('text', ''),
                        'lemma': word.get('lemma', ''),
                        'morph_code': word.get('morph', ''),
                        'features': morph_features
                    }
                    words.append(word_info)
                
                return words
        
        return []
    
    def extract_lemmas(self, book_code: str) -> Dict[str, int]:
        """
        Extract all lemmas from a book with frequency counts.
        
        Args:
            book_code: Book code
            
        Returns:
            Dictionary mapping lemmas to frequency counts
        """
        book_data = self.load_book(book_code)
        if not book_data:
            return {}
        
        lemma_counts = {}
        
        for verse_data in book_data.get('verses', []):
            for word in verse_data.get('words', []):
                lemma = word.get('lemma', '')
                if lemma:
                    # Remove Strong's numbers if present
                    clean_lemma = lemma.split('/')[-1] if '/' in lemma else lemma
                    lemma_counts[clean_lemma] = lemma_counts.get(clean_lemma, 0) + 1
        
        return lemma_counts