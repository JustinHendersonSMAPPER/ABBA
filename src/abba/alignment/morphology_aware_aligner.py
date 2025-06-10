"""
Morphology-Aware Aligner for Biblical Texts

Uses morphological analysis from OSHB (Hebrew) and MorphGNT (Greek) 
to improve alignment accuracy by understanding grammatical relationships.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger('ABBA.MorphologyAwareAligner')


class BiblicalMorphologyAnalyzer:
    """
    Analyzes biblical Hebrew and Greek morphology using OSHB and MorphGNT data.
    """
    
    def __init__(self):
        self.hebrew_morphology = {}
        self.greek_morphology = {}
        self._load_morphological_data()
        
    def _load_morphological_data(self):
        """Load morphological data from OSHB and MorphGNT files."""
        # Load Hebrew morphology from OSHB
        hebrew_morph_dir = Path("data/sources/morphology/hebrew")
        if hebrew_morph_dir.exists():
            for file_path in hebrew_morph_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        book_name = data.get('book', file_path.stem)
                        self.hebrew_morphology[book_name] = data
                except Exception as e:
                    logger.warning(f"Error loading Hebrew morphology from {file_path}: {e}")
        
        # Load Greek morphology from MorphGNT
        greek_morph_dir = Path("data/sources/morphology/greek")
        if greek_morph_dir.exists():
            for file_path in greek_morph_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        book_name = data.get('book', file_path.stem)
                        self.greek_morphology[book_name] = data
                except Exception as e:
                    logger.warning(f"Error loading Greek morphology from {file_path}: {e}")
        
        logger.info(f"Loaded morphology for {len(self.hebrew_morphology)} Hebrew books, {len(self.greek_morphology)} Greek books")
    
    def get_morphological_features(self, word: str, book: str, chapter: int, verse: int, language: str) -> Optional[Dict]:
        """Get morphological features for a word at a specific location."""
        morphology_data = self.hebrew_morphology if language == 'hebrew' else self.greek_morphology
        
        if book not in morphology_data:
            return None
        
        book_data = morphology_data[book]
        
        # Find the verse
        for verse_data in book_data.get('verses', []):
            verse_ref = verse_data.get('verse_id', '')
            if f"{chapter}.{verse}" in verse_ref:
                # Find the word in this verse
                for word_data in verse_data.get('words', []):
                    word_text = word_data.get('text', '').strip()
                    normalized = word_data.get('normalized', '').strip()
                    
                    # Try exact match or normalized match
                    if word == word_text or word == normalized:
                        return self._extract_morphological_features(word_data, language)
                    
                    # Try partial match for prefixed words
                    if language == 'hebrew' and (word in word_text or word_text in word):
                        return self._extract_morphological_features(word_data, language)
        
        return None
    
    def _extract_morphological_features(self, word_data: Dict, language: str) -> Dict:
        """Extract morphological features from word data."""
        features = {
            'lemma': word_data.get('lemma', ''),
            'pos': word_data.get('pos', word_data.get('POS', '')),
            'language': language
        }
        
        if language == 'hebrew':
            # OSHB Hebrew features
            features.update({
                'morph_code': word_data.get('morph', ''),
                'person': word_data.get('person', ''),
                'gender': word_data.get('gender', ''),
                'number': word_data.get('number', ''),
                'state': word_data.get('state', ''),
                'type': word_data.get('type', '')
            })
        
        elif language == 'greek':
            # MorphGNT Greek features
            morph_features = word_data.get('morph_features', {})
            features.update({
                'parse_code': word_data.get('parse', ''),
                'person': morph_features.get('person'),
                'tense': morph_features.get('tense'),
                'voice': morph_features.get('voice'),
                'mood': morph_features.get('mood'),
                'case': morph_features.get('case'),
                'number': morph_features.get('number'),
                'gender': morph_features.get('gender'),
                'degree': morph_features.get('degree')
            })
        
        # Remove None values
        return {k: v for k, v in features.items() if v is not None and v != ''}


class MorphologyAwareAligner:
    """
    Aligner that uses morphological analysis to improve alignment accuracy.
    """
    
    def __init__(self):
        self.morphology_analyzer = BiblicalMorphologyAnalyzer()
        self.pos_alignment_weights = {
            # POS tag similarities for alignment
            'noun': {'noun': 1.0, 'pronoun': 0.7, 'adjective': 0.5},
            'verb': {'verb': 1.0, 'participle': 0.8, 'auxiliary': 0.6},
            'adjective': {'adjective': 1.0, 'noun': 0.5, 'adverb': 0.7},
            'adverb': {'adverb': 1.0, 'adjective': 0.7, 'preposition': 0.4},
            'preposition': {'preposition': 1.0, 'adverb': 0.4},
            'conjunction': {'conjunction': 1.0, 'adverb': 0.3},
            'article': {'article': 1.0, 'determiner': 0.9, 'pronoun': 0.4},
            'particle': {'particle': 1.0, 'adverb': 0.6},
            'pronoun': {'pronoun': 1.0, 'noun': 0.7, 'article': 0.4}
        }
    
    def align_verse(self, source_words: List[str], target_words: List[str],
                   source_lang: str = 'hebrew', target_lang: str = 'english', **kwargs) -> List[Dict]:
        """
        Align words using morphological analysis.
        """
        book_code = kwargs.get('book_code', 'Gen')
        chapter = kwargs.get('chapter', 1)
        verse = kwargs.get('verse', 1)
        
        logger.debug(f"Morphological alignment: {len(source_words)} {source_lang} → {len(target_words)} {target_lang}")
        
        alignments = []
        
        # Get morphological analysis for source words
        source_morphology = []
        for word in source_words:
            morph = self.morphology_analyzer.get_morphological_features(
                word, book_code, chapter, verse, source_lang
            )
            source_morphology.append(morph)
        
        # Align based on morphological compatibility
        for src_idx, (src_word, src_morph) in enumerate(zip(source_words, source_morphology)):
            if src_morph is None:
                continue
                
            best_alignments = []
            
            for tgt_idx, tgt_word in enumerate(target_words):
                confidence = self._calculate_morphological_alignment_confidence(
                    src_word, src_morph, tgt_word, target_lang
                )
                
                if confidence > 0.4:  # Threshold for morphological alignment
                    best_alignments.append((tgt_idx, confidence))
            
            # Sort by confidence and take best match
            best_alignments.sort(key=lambda x: x[1], reverse=True)
            
            for tgt_idx, confidence in best_alignments[:1]:  # Take best match only
                alignment = {
                    'source_index': src_idx,
                    'target_index': tgt_idx,
                    'source_word': src_word,
                    'target_word': target_words[tgt_idx],
                    'confidence': round(confidence, 3),
                    'method': 'morphological',
                    'morphological_features': {
                        'source_pos': src_morph.get('pos', ''),
                        'source_lemma': src_morph.get('lemma', ''),
                        'morphological_match': True,
                        'alignment_type': 'morphological'
                    }
                }
                alignments.append(alignment)
        
        logger.debug(f"Morphological alignment found {len(alignments)} alignments")
        return alignments
    
    def _calculate_morphological_alignment_confidence(self, source_word: str, source_morph: Dict, 
                                                    target_word: str, target_lang: str) -> float:
        """Calculate alignment confidence based on morphological features."""
        confidence = 0.0
        
        # Base confidence from POS matching
        source_pos = source_morph.get('pos', '').lower()
        target_pos = self._guess_english_pos(target_word)
        
        pos_confidence = self._get_pos_alignment_confidence(source_pos, target_pos)
        confidence += pos_confidence * 0.6
        
        # Boost confidence for function words
        if source_pos in ['article', 'preposition', 'conjunction', 'particle']:
            function_word_confidence = self._get_function_word_confidence(source_word, target_word, source_morph)
            confidence += function_word_confidence * 0.4
        
        # Boost confidence for content words based on semantic similarity
        elif source_pos in ['noun', 'verb', 'adjective']:
            semantic_confidence = self._get_semantic_confidence(source_morph.get('lemma', ''), target_word)
            confidence += semantic_confidence * 0.4
        
        return min(confidence, 1.0)
    
    def _get_pos_alignment_confidence(self, source_pos: str, target_pos: str) -> float:
        """Get confidence based on POS tag compatibility."""
        if source_pos in self.pos_alignment_weights:
            return self.pos_alignment_weights[source_pos].get(target_pos, 0.0)
        return 0.0
    
    def _guess_english_pos(self, word: str) -> str:
        """Simple English POS guessing based on word patterns."""
        word_lower = word.lower()
        
        # Articles
        if word_lower in ['the', 'a', 'an']:
            return 'article'
        
        # Prepositions
        if word_lower in ['in', 'on', 'at', 'by', 'with', 'from', 'to', 'of', 'for', 'through', 'over', 'under']:
            return 'preposition'
        
        # Conjunctions
        if word_lower in ['and', 'or', 'but', 'so', 'yet', 'nor', 'for']:
            return 'conjunction'
        
        # Pronouns
        if word_lower in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their']:
            return 'pronoun'
        
        # Common verbs (simplified)
        if word_lower in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'would', 'should', 'could', 'may', 'might', 'must']:
            return 'verb'
        
        # Check for verb endings
        if word_lower.endswith(('ed', 'ing', 's')):
            return 'verb'
        
        # Check for adverb endings
        if word_lower.endswith('ly'):
            return 'adverb'
        
        # Default to noun for content words
        return 'noun'
    
    def _get_function_word_confidence(self, source_word: str, target_word: str, source_morph: Dict) -> float:
        """Get confidence for function word alignments."""
        source_pos = source_morph.get('pos', '').lower()
        target_word_lower = target_word.lower()
        
        # Hebrew function words
        if source_morph.get('language') == 'hebrew':
            if source_word.startswith('ה') and target_word_lower == 'the':
                return 0.9
            elif source_word.startswith('ו') and target_word_lower in ['and', 'but', 'or']:
                return 0.85
            elif source_word.startswith('ב') and target_word_lower in ['in', 'on', 'with', 'by']:
                return 0.8
            elif source_word.startswith('ל') and target_word_lower in ['to', 'for']:
                return 0.8
            elif source_word.startswith('מ') and target_word_lower in ['from', 'of']:
                return 0.8
        
        # Greek function words
        elif source_morph.get('language') == 'greek':
            lemma = source_morph.get('lemma', '').lower()
            if lemma.startswith('ὁ') and target_word_lower == 'the':
                return 0.9
            elif lemma == 'καί' and target_word_lower in ['and', 'also', 'but']:
                return 0.85
            elif lemma == 'ἐν' and target_word_lower in ['in', 'on', 'among']:
                return 0.8
            elif lemma == 'πρός' and target_word_lower in ['to', 'toward', 'with']:
                return 0.8
        
        return 0.0
    
    def _get_semantic_confidence(self, source_lemma: str, target_word: str) -> float:
        """Get confidence based on semantic similarity (simplified)."""
        # This would ideally use the modern semantic aligner
        # For now, simple lemma-based matching
        if not source_lemma:
            return 0.0
        
        # Basic semantic matches (could be expanded)
        semantic_pairs = {
            'אלהים': ['god', 'divine'],
            'ארץ': ['earth', 'land', 'ground'],
            'שמים': ['heaven', 'heavens', 'sky'],
            'ברא': ['create', 'created', 'make', 'made'],
            'ראשית': ['beginning', 'first', 'start'],
            'θεός': ['god', 'divine'],
            'κόσμος': ['world', 'universe'],
            'λόγος': ['word', 'message', 'speech'],
            'ἀρχή': ['beginning', 'origin', 'principle']
        }
        
        target_lower = target_word.lower()
        for lemma, translations in semantic_pairs.items():
            if source_lemma == lemma and target_lower in translations:
                return 0.8
        
        return 0.1  # Small default for content words