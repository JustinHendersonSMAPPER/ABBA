"""
Position-based word aligner for basic alignment.

This aligner maps words based on their relative position in verses,
providing a baseline alignment method with low confidence scores.
"""

from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PositionAligner:
    """Simple position-based word aligner."""
    
    def __init__(self, base_confidence: float = 0.4):
        """
        Initialize the position aligner.
        
        Args:
            base_confidence: Base confidence score for position alignments (default: 0.4)
        """
        self.base_confidence = base_confidence
    
    def align_verse(
        self,
        source_words: List[str],
        target_words: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Align words based on their relative position in the verse.
        
        Args:
            source_words: List of words in source language
            target_words: List of words in target language
            source_lang: Source language code (optional)
            target_lang: Target language code (optional)
            
        Returns:
            List of alignment dictionaries with format:
            {
                'source_index': int,
                'target_index': int,
                'source_word': str,
                'target_word': str,
                'confidence': float,
                'method': 'position'
            }
        """
        if not source_words or not target_words:
            return []
        
        alignments = []
        source_len = len(source_words)
        target_len = len(target_words)
        
        # Calculate position ratios and align proportionally
        for src_idx, src_word in enumerate(source_words):
            # Calculate relative position (0.0 to 1.0)
            src_position = src_idx / max(source_len - 1, 1)
            
            # Find closest target position
            target_idx = int(src_position * (target_len - 1) + 0.5)
            target_idx = min(target_idx, target_len - 1)
            
            # Calculate confidence based on verse length difference
            length_ratio = min(source_len, target_len) / max(source_len, target_len)
            confidence = self.base_confidence * length_ratio
            
            # Adjust confidence based on position
            # Words at beginning and end tend to align better
            if src_position <= 0.1 or src_position >= 0.9:
                confidence *= 1.2
            
            # Cap confidence at reasonable levels
            confidence = min(confidence, 0.5)
            
            alignment = {
                'source_index': src_idx,
                'target_index': target_idx,
                'source_word': src_word,
                'target_word': target_words[target_idx],
                'confidence': round(confidence, 3),
                'method': 'position'
            }
            alignments.append(alignment)
        
        return alignments
    
    def align_batch(
        self,
        verse_pairs: List[Tuple[List[str], List[str]]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> List[List[Dict[str, any]]]:
        """
        Align multiple verse pairs.
        
        Args:
            verse_pairs: List of (source_words, target_words) tuples
            source_lang: Source language code (optional)
            target_lang: Target language code (optional)
            
        Returns:
            List of alignment lists, one per verse pair
        """
        results = []
        for source_words, target_words in verse_pairs:
            alignments = self.align_verse(
                source_words, target_words, source_lang, target_lang
            )
            results.append(alignments)
        
        return results
    
    def get_alignment_scores(
        self,
        source_words: List[str],
        target_words: List[str]
    ) -> Dict[Tuple[int, int], float]:
        """
        Get alignment scores as a matrix.
        
        Args:
            source_words: List of words in source language
            target_words: List of words in target language
            
        Returns:
            Dictionary mapping (source_idx, target_idx) to confidence scores
        """
        scores = {}
        alignments = self.align_verse(source_words, target_words)
        
        for alignment in alignments:
            key = (alignment['source_index'], alignment['target_index'])
            scores[key] = alignment['confidence']
        
        return scores