"""
Ensemble aligner that combines multiple alignment methods.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnsembleAligner:
    """Combines multiple aligners with weighted voting."""
    
    def __init__(self, aligners: List[Tuple[Any, float]]):
        """
        Initialize ensemble aligner.
        
        Args:
            aligners: List of (aligner, weight) tuples
        """
        self.aligners = aligners
        
        # Normalize weights
        total_weight = sum(weight for _, weight in aligners)
        self.aligners = [(aligner, weight / total_weight) for aligner, weight in aligners]
        
        logger.info(f"Initialized ensemble with {len(aligners)} aligners")
        for aligner, weight in self.aligners:
            logger.info(f"  - {aligner.__class__.__name__}: weight={weight:.2f}")
    
    def align_verse(
        self,
        source_words: List[str],
        target_words: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, any]]:
        """
        Align words using ensemble of methods.
        
        Args:
            source_words: Source language words
            target_words: Target language words
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments passed to aligners
            
        Returns:
            List of alignment dictionaries
        """
        if not source_words or not target_words:
            return []
        
        # Collect alignments from all methods
        all_alignments = []
        alignment_scores = defaultdict(lambda: defaultdict(float))
        alignment_features = defaultdict(dict)
        
        for aligner, weight in self.aligners:
            try:
                # Call aligner with appropriate arguments
                if hasattr(aligner, 'align_verse'):
                    # Check if aligner accepts additional arguments
                    import inspect
                    sig = inspect.signature(aligner.align_verse)
                    params = sig.parameters
                    
                    # Build kwargs based on what the aligner accepts
                    aligner_kwargs = {}
                    for key, value in kwargs.items():
                        if key in params:
                            aligner_kwargs[key] = value
                    
                    alignments = aligner.align_verse(
                        source_words, target_words,
                        source_lang, target_lang,
                        **aligner_kwargs
                    )
                    
                    # Accumulate weighted scores
                    for align in alignments:
                        src_idx = align['source_index']
                        tgt_idx = align['target_index']
                        score = align['confidence'] * weight
                        
                        alignment_scores[src_idx][tgt_idx] += score
                        
                        # Keep features from morphological aligner
                        if 'features' in align and align['features']:
                            alignment_features[(src_idx, tgt_idx)] = align['features']
                        
                        # Track method contributions
                        method_key = f"method_{aligner.__class__.__name__}"
                        if (src_idx, tgt_idx) not in alignment_features:
                            alignment_features[(src_idx, tgt_idx)] = {}
                        alignment_features[(src_idx, tgt_idx)][method_key] = align['confidence']
                    
            except Exception as e:
                logger.warning(f"Aligner {aligner.__class__.__name__} failed: {e}")
        
        # Find best alignments using Hungarian algorithm approximation
        alignments = self._extract_best_alignments(
            source_words, target_words, alignment_scores, alignment_features
        )
        
        return alignments
    
    def _extract_best_alignments(
        self,
        source_words: List[str],
        target_words: List[str],
        scores: Dict[int, Dict[int, float]],
        features: Dict[Tuple[int, int], Dict]
    ) -> List[Dict[str, any]]:
        """
        Extract best alignments from score matrix.
        
        Uses greedy approximation of optimal assignment.
        """
        alignments = []
        used_targets = set()
        
        # Sort source indices by best available score
        source_indices = []
        for src_idx in range(len(source_words)):
            if src_idx in scores:
                best_score = max(scores[src_idx].values()) if scores[src_idx] else 0
                source_indices.append((best_score, src_idx))
        
        source_indices.sort(reverse=True)
        
        # Assign each source to best available target
        for _, src_idx in source_indices:
            best_tgt_idx = -1
            best_score = 0.0
            
            for tgt_idx, score in scores[src_idx].items():
                if tgt_idx not in used_targets and score > best_score:
                    best_score = score
                    best_tgt_idx = tgt_idx
            
            if best_tgt_idx >= 0 and best_score > 0.05:  # Minimum threshold (lowered for better coverage)
                used_targets.add(best_tgt_idx)
                
                # Enhanced confidence calculation with multi-method agreement
                pair_features = features.get((src_idx, best_tgt_idx), {})
                method_scores = [v for k, v in pair_features.items() if k.startswith('method_')]
                method_names = [k.replace('method_', '') for k in pair_features.keys() if k.startswith('method_')]
                
                # Base confidence from weighted average
                confidence = best_score
                
                if len(method_scores) > 1:
                    # Multi-method agreement bonus
                    score_range = max(method_scores) - min(method_scores)
                    agreement_factor = 1.0 - score_range  # Higher when scores are similar
                    
                    # Consensus bonus: exponential increase with more agreeing methods
                    consensus_bonus = 0.15 * (len(method_scores) - 1) * agreement_factor
                    confidence *= (1.0 + consensus_bonus)
                    
                    # Additional bonuses for specific method combinations
                    if 'StatisticalAligner' in method_names and 'MorphologicalAligner' in method_names:
                        # Statistical + Morphological agreement is very reliable
                        confidence *= 1.1
                    
                    if len(method_scores) >= 3:
                        # All three methods agree - highest confidence boost
                        confidence *= 1.15
                
                # Confidence capping and method-specific adjustments
                if 'StatisticalAligner' in method_names:
                    # Statistical alignments tend to be more reliable
                    confidence *= 1.05
                
                # Final confidence normalization
                confidence = min(confidence, 1.0)
                
                alignment = {
                    'source_index': src_idx,
                    'target_index': best_tgt_idx,
                    'source_word': source_words[src_idx],
                    'target_word': target_words[best_tgt_idx],
                    'confidence': round(confidence, 3),
                    'method': 'ensemble',
                    'num_methods': len(method_scores),
                    'contributing_methods': method_names,
                    'method_agreement': round(1.0 - score_range, 3) if len(method_scores) > 1 else 1.0,
                    'base_score': round(best_score, 3)
                }
                
                # Include morphological features if available
                if pair_features and not any(k.startswith('method_') for k in pair_features):
                    alignment['features'] = {k: v for k, v in pair_features.items() 
                                           if not k.startswith('method_')}
                
                alignments.append(alignment)
        
        # Sort by source index for display
        alignments.sort(key=lambda x: x['source_index'])
        
        return alignments
    
    def align_batch(
        self,
        verse_pairs: List[Tuple[List[str], List[str]]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> List[List[Dict[str, any]]]:
        """
        Align multiple verse pairs.
        
        Args:
            verse_pairs: List of (source_words, target_words) tuples
            source_lang: Source language
            target_lang: Target language
            **kwargs: Additional arguments passed to aligners
            
        Returns:
            List of alignment lists
        """
        results = []
        
        for i, (source_words, target_words) in enumerate(verse_pairs):
            # Update kwargs with verse reference if available
            verse_kwargs = kwargs.copy()
            if 'verse_refs' in kwargs and i < len(kwargs['verse_refs']):
                book_code, chapter, verse = kwargs['verse_refs'][i]
                verse_kwargs.update({
                    'book_code': book_code,
                    'chapter': chapter,
                    'verse': verse
                })
            
            alignments = self.align_verse(
                source_words, target_words,
                source_lang, target_lang,
                **verse_kwargs
            )
            results.append(alignments)
        
        return results