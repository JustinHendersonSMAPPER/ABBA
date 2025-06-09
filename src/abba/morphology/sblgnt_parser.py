"""
SBLGNT/MorphGNT Greek morphological data parser.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SBLGNTParser:
    """Parser for SBLGNT/MorphGNT morphological data in JSON format."""
    
    def __init__(self, morphology_dir: Path = Path('data/sources/morphology/greek')):
        """
        Initialize SBLGNT parser.
        
        Args:
            morphology_dir: Directory containing Greek morphological JSON files
        """
        self.morphology_dir = morphology_dir
        self._book_name_map = self._build_book_name_map()
    
    def _build_book_name_map(self) -> Dict[str, str]:
        """Build mapping from book codes to morphology filenames."""
        return {
            # Gospels and Acts
            'Matt': 'matthew.json',
            'Mark': 'mark.json',
            'Luke': 'luke.json',
            'John': 'john.json',
            'Acts': 'acts.json',
            
            # Pauline Epistles
            'Rom': 'romans.json',
            '1Cor': '1corinthians.json',
            '2Cor': '2corinthians.json',
            'Gal': 'galatians.json',
            'Eph': 'ephesians.json',
            'Phil': 'philippians.json',
            'Col': 'colossians.json',
            '1Thess': '1thessalonians.json',
            '2Thess': '2thessalonians.json',
            '1Tim': '1timothy.json',
            '2Tim': '2timothy.json',
            'Titus': 'titus.json',
            'Phlm': 'philemon.json',
            
            # General Epistles
            'Heb': 'hebrews.json',
            'Jas': 'james.json',
            '1Pet': '1peter.json',
            '2Pet': '2peter.json',
            '1John': '1john.json',
            '2John': '2john.json',
            '3John': '3john.json',
            'Jude': 'jude.json',
            
            # Revelation
            'Rev': 'revelation.json'
        }
    
    def parse_pos_code(self, pos_code: str) -> Dict[str, str]:
        """
        Parse part-of-speech code.
        
        Args:
            pos_code: POS code (e.g., "N-", "V-", "P-")
            
        Returns:
            Dictionary with part of speech
        """
        pos_map = {
            'N': 'noun',
            'V': 'verb',
            'P': 'preposition',
            'A': 'adjective',
            'D': 'adverb',
            'C': 'conjunction',
            'R': 'article',
            'I': 'interjection',
            'X': 'particle',
            'M': 'numeral',
            'F': 'foreign'
        }
        
        if pos_code and pos_code[0] in pos_map:
            return {'pos': pos_map[pos_code[0]]}
        
        return {'pos': 'unknown'}
    
    def parse_features(self, morph_features: Dict[str, str]) -> Dict[str, str]:
        """
        Parse morphological features from the JSON structure.
        
        Args:
            morph_features: Dictionary of morphological features
            
        Returns:
            Standardized feature dictionary
        """
        features = {}
        
        # Case mapping
        case_map = {
            'N': 'nominative',
            'G': 'genitive',
            'D': 'dative',
            'A': 'accusative',
            'V': 'vocative'
        }
        
        # Number mapping
        number_map = {
            'S': 'singular',
            'P': 'plural'
        }
        
        # Gender mapping
        gender_map = {
            'M': 'masculine',
            'F': 'feminine',
            'N': 'neuter'
        }
        
        # Map features
        if 'case' in morph_features:
            features['case'] = case_map.get(morph_features['case'], morph_features['case'])
        
        if 'number' in morph_features:
            features['number'] = number_map.get(morph_features['number'], morph_features['number'])
        
        if 'gender' in morph_features:
            features['gender'] = gender_map.get(morph_features['gender'], morph_features['gender'])
        
        # For verbs, parse additional features from parse code if available
        if 'tense' in morph_features:
            features['tense'] = morph_features['tense']
        
        if 'voice' in morph_features:
            features['voice'] = morph_features['voice']
        
        if 'mood' in morph_features:
            features['mood'] = morph_features['mood']
        
        if 'person' in morph_features:
            features['person'] = morph_features['person']
        
        return features
    
    def parse_verb_code(self, parse_code: str) -> Dict[str, str]:
        """
        Parse verb-specific features from parse code.
        
        Args:
            parse_code: Full parse code (e.g., "V-PAI-3S")
            
        Returns:
            Dictionary of verb features
        """
        features = {}
        
        if len(parse_code) < 8:
            return features
        
        # Parse code format: ----TVMN
        # T = Tense, V = Voice, M = Mood, N = Number + Person
        
        # Tense (position 4)
        tense_map = {
            'P': 'present',
            'I': 'imperfect',
            'F': 'future',
            'A': 'aorist',
            'X': 'perfect',
            'Y': 'pluperfect'
        }
        
        # Voice (position 5)
        voice_map = {
            'A': 'active',
            'M': 'middle',
            'P': 'passive'
        }
        
        # Mood (position 6)
        mood_map = {
            'I': 'indicative',
            'S': 'subjunctive',
            'O': 'optative',
            'M': 'imperative',
            'N': 'infinitive',
            'P': 'participle'
        }
        
        if len(parse_code) > 4 and parse_code[4] != '-':
            features['tense'] = tense_map.get(parse_code[4], parse_code[4])
        
        if len(parse_code) > 5 and parse_code[5] != '-':
            features['voice'] = voice_map.get(parse_code[5], parse_code[5])
        
        if len(parse_code) > 6 and parse_code[6] != '-':
            features['mood'] = mood_map.get(parse_code[6], parse_code[6])
        
        # Person and Number (position 7-8)
        if len(parse_code) > 7 and parse_code[7] != '-':
            features['person'] = parse_code[7]
        
        if len(parse_code) > 8 and parse_code[8] != '-':
            number_map = {'S': 'singular', 'P': 'plural'}
            features['number'] = number_map.get(parse_code[8], parse_code[8])
        
        return features
    
    def load_book(self, book_code: str) -> Optional[Dict]:
        """
        Load morphological data for a book.
        
        Args:
            book_code: Book code (e.g., "Matt", "Rom")
            
        Returns:
            Book data with parsed morphology or None if not found
        """
        # Try direct file first
        file_path = self.morphology_dir / f"{book_code}.json"
        
        # If not found, try mapped filename
        if not file_path.exists() and book_code in self._book_name_map:
            file_path = self.morphology_dir / self._book_name_map[book_code]
        
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
        for verse_data in book_data.get('verses', []):
            if (verse_data.get('chapter') == chapter and 
                verse_data.get('verse') == verse):
                
                words = []
                for word in verse_data.get('words', []):
                    # Get POS features
                    pos_features = self.parse_pos_code(word.get('pos', ''))
                    
                    # Get morphological features
                    morph_features = self.parse_features(word.get('morph_features', {}))
                    
                    # For verbs, also parse the parse code
                    if pos_features.get('pos') == 'verb' and 'parse' in word:
                        verb_features = self.parse_verb_code(word['parse'])
                        morph_features.update(verb_features)
                    
                    # Combine all features
                    all_features = {}
                    all_features.update(pos_features)
                    all_features.update(morph_features)
                    
                    word_info = {
                        'text': word.get('text', ''),
                        'normalized': word.get('normalized', word.get('text', '')),
                        'lemma': word.get('lemma', ''),
                        'pos_code': word.get('pos', ''),
                        'parse_code': word.get('parse', ''),
                        'features': all_features
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
                    lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1
        
        return lemma_counts