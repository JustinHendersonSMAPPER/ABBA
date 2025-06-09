"""
Morphological aligner that uses lemma and POS information for alignment.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

from ..morphology.oshb_parser import OSHBParser
from ..morphology.sblgnt_parser import SBLGNTParser

logger = logging.getLogger(__name__)


class MorphologicalAligner:
    """Aligner that uses morphological features for word alignment."""
    
    def __init__(self, base_confidence: float = 0.6, morphology_dir: Optional[Path] = None):
        """
        Initialize morphological aligner.
        
        Args:
            base_confidence: Base confidence for morphological alignments
            morphology_dir: Directory containing morphological data
        """
        self.base_confidence = base_confidence
        
        # Initialize parsers
        if morphology_dir:
            self.hebrew_parser = OSHBParser(morphology_dir / 'hebrew')
            self.greek_parser = SBLGNTParser(morphology_dir / 'greek')
        else:
            self.hebrew_parser = OSHBParser()
            self.greek_parser = SBLGNTParser()
        
        # Cache for loaded morphological data
        self._morph_cache = {}
    
    def _get_morph_words(self, book_code: str, chapter: int, verse: int, 
                        language: str) -> List[Dict[str, any]]:
        """Get morphological words for a verse."""
        cache_key = f"{language}:{book_code}:{chapter}:{verse}"
        
        if cache_key in self._morph_cache:
            return self._morph_cache[cache_key]
        
        if language == 'hebrew':
            words = self.hebrew_parser.get_verse_words(book_code, chapter, verse)
        elif language == 'greek':
            words = self.greek_parser.get_verse_words(book_code, chapter, verse)
        else:
            words = []
        
        self._morph_cache[cache_key] = words
        return words
    
    def _calculate_lemma_similarity(self, source_lemma: str, target_word: str) -> float:
        """
        Calculate similarity between lemma and target word.
        
        This is a simplified approach - in practice, would use:
        - Translation lexicons
        - Bilingual embeddings
        - Statistical translation models
        """
        # For now, use string similarity as placeholder
        # Real implementation would look up translation probabilities
        return SequenceMatcher(None, source_lemma.lower(), target_word.lower()).ratio()
    
    def _calculate_pos_bonus(self, source_features: Dict, target_word: str) -> float:
        """
        Calculate bonus score based on POS matching.
        
        In real implementation, would use:
        - POS tagging for target language
        - Cross-lingual POS mappings
        """
        pos = source_features.get('pos', '')
        
        # Simple heuristics for demonstration
        pos_patterns = {
            'noun': ['the', 'a', 'an'],  # Articles often precede nouns
            'verb': ['ed', 'ing', 's'],   # Verb endings
            'preposition': ['in', 'on', 'at', 'to', 'from', 'with', 'by'],
            'conjunction': ['and', 'or', 'but', 'for', 'yet'],
        }
        
        # Check if target word matches expected patterns
        bonus = 0.0
        
        if pos in pos_patterns:
            # Check if target is in pattern list
            if target_word.lower() in pos_patterns[pos]:
                bonus = 0.2
            # Check endings for verbs
            elif pos == 'verb':
                for ending in pos_patterns['verb']:
                    if target_word.lower().endswith(ending):
                        bonus = 0.1
                        break
        
        return bonus
    
    def align_verse(
        self,
        source_words: List[str],
        target_words: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        book_code: Optional[str] = None,
        chapter: Optional[int] = None,
        verse: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Align words using morphological information.
        
        Args:
            source_words: List of source language words
            target_words: List of target language words
            source_lang: Source language ('hebrew' or 'greek')
            target_lang: Target language
            book_code: Book code for looking up morphology
            chapter: Chapter number
            verse: Verse number
            
        Returns:
            List of alignment dictionaries
        """
        if not source_words or not target_words:
            return []
        
        alignments = []
        
        # Get morphological data if available
        morph_words = []
        if source_lang in ['hebrew', 'greek'] and all([book_code, chapter, verse]):
            morph_words = self._get_morph_words(book_code, chapter, verse, source_lang)
        
        # Create alignment matrix
        scores = {}
        
        for src_idx, src_word in enumerate(source_words):
            # Get morphological info for source word
            morph_info = None
            if src_idx < len(morph_words):
                morph_info = morph_words[src_idx]
            
            for tgt_idx, tgt_word in enumerate(target_words):
                score = 0.0
                
                if morph_info:
                    # Use lemma for similarity
                    lemma = morph_info.get('lemma', '')
                    if lemma:
                        # Clean lemma (remove Strong's numbers if present)
                        clean_lemma = lemma.split('/')[-1] if '/' in lemma else lemma
                        clean_lemma = ''.join(c for c in clean_lemma if not c.isdigit())
                        
                        if clean_lemma:
                            score = self._calculate_lemma_similarity(clean_lemma, tgt_word)
                    
                    # Add POS bonus
                    features = morph_info.get('features', {})
                    if features:
                        score += self._calculate_pos_bonus(features, tgt_word)
                else:
                    # Fallback to simple string similarity
                    score = SequenceMatcher(None, src_word.lower(), tgt_word.lower()).ratio()
                
                # Apply base confidence
                score *= self.base_confidence
                
                scores[(src_idx, tgt_idx)] = score
        
        # Find best alignment for each source word
        # Simple greedy approach - in practice would use optimal assignment
        used_targets = set()
        
        for src_idx, src_word in enumerate(source_words):
            best_tgt_idx = -1
            best_score = 0.0
            
            for tgt_idx in range(len(target_words)):
                if tgt_idx not in used_targets:
                    score = scores.get((src_idx, tgt_idx), 0.0)
                    if score > best_score:
                        best_score = score
                        best_tgt_idx = tgt_idx
            
            if best_tgt_idx >= 0:
                used_targets.add(best_tgt_idx)
                
                # Get morphological features for alignment
                features = {}
                if src_idx < len(morph_words):
                    features = morph_words[src_idx].get('features', {})
                
                alignment = {
                    'source_index': src_idx,
                    'target_index': best_tgt_idx,
                    'source_word': src_word,
                    'target_word': target_words[best_tgt_idx],
                    'confidence': round(best_score, 3),
                    'method': 'morphological',
                    'features': features
                }
                alignments.append(alignment)
        
        return alignments
    
    def align_batch(
        self,
        verse_pairs: List[Tuple[List[str], List[str]]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        verse_refs: Optional[List[Tuple[str, int, int]]] = None
    ) -> List[List[Dict[str, any]]]:
        """
        Align multiple verse pairs.
        
        Args:
            verse_pairs: List of (source_words, target_words) tuples
            source_lang: Source language
            target_lang: Target language  
            verse_refs: List of (book_code, chapter, verse) tuples
            
        Returns:
            List of alignment lists
        """
        results = []
        
        for i, (source_words, target_words) in enumerate(verse_pairs):
            # Get verse reference if available
            book_code = chapter = verse = None
            if verse_refs and i < len(verse_refs):
                book_code, chapter, verse = verse_refs[i]
            
            alignments = self.align_verse(
                source_words, target_words,
                source_lang, target_lang,
                book_code, chapter, verse
            )
            results.append(alignments)
        
        return results