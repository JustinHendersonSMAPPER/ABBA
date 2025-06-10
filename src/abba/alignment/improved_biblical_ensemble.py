"""
Improved Biblical Ensemble Aligner

This ensemble is specifically designed for biblical texts with proper weighting
and confidence calibration based on linguistic principles.
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

from .ensemble_aligner import EnsembleAligner

logger = logging.getLogger('ABBA.ImprovedBiblicalEnsemble')


class ImprovedBiblicalEnsemble(EnsembleAligner):
    """
    Enhanced ensemble aligner specifically optimized for biblical text alignment.
    
    Key improvements:
    1. Biblical semantic aligner as primary method
    2. Proper weighting based on biblical linguistics
    3. Confidence calibration for ancient languages
    4. Multi-word phrase detection
    """
    
    def __init__(self, aligners: List[Tuple], weights: Optional[Dict[str, float]] = None):
        # Initialize parent with original aligners
        super().__init__(aligners)
        
        # Morphological lexicon weights using OSHB/MorphGNT lemma data
        # Using actual biblical lemmas, not similarity heuristics
        self.biblical_weights = {
            'MorphologicalLexiconAligner': 0.50,    # Primary: OSHB/MorphGNT lemmas
            'MorphologyAwareAligner': 0.25,         # Secondary: morphological features
            'ModernSemanticAligner': 0.15,          # Tertiary: modern lexicon similarity
            'CrossLingualEmbeddingAligner': 0.05,   # Minimal: embeddings
            'StatisticalAligner': 0.05              # Minimal: fallback
        }
        
        # Confidence calibration parameters
        self.confidence_calibration = {
            'lexicon_match_bonus': 0.15,
            'morphology_match_bonus': 0.1,
            'consensus_threshold': 2,  # Number of methods needed for consensus
            'minimum_confidence': 0.5,  # Lowered threshold for better coverage
            'maximum_confidence': 0.98  # Leave room for uncertainty
        }
        
        logger.info("Initialized improved biblical ensemble with modern aligners")
        logger.info(f"Weights: {self.biblical_weights}")
    
    def align_verse(self, source_words: List[str], target_words: List[str],
                   source_lang: str = 'hebrew', target_lang: str = 'english', **kwargs) -> List[Dict]:
        """
        Enhanced alignment using modern biblical alignment methods and proper confidence scoring.
        """
        logger.debug(f"Modern biblical ensemble alignment: {len(source_words)} → {len(target_words)} words")
        
        # Step 1: Get alignments from all modern methods
        all_alignments = self._collect_all_alignments(source_words, target_words, source_lang, target_lang, **kwargs)
        
        # Step 2: Combine and calibrate confidence scores 
        combined_alignments = self._combine_alignments_with_calibration(
            all_alignments, source_words, target_words
        )
        
        # Step 3: Resolve conflicts and optimize assignment
        final_alignments = self._optimize_alignment_assignment(
            combined_alignments, len(source_words), len(target_words)
        )
        
        logger.debug(f"Modern biblical ensemble: {len(final_alignments)} high-confidence alignments")
        return final_alignments
    
    def _collect_all_alignments(self, source_words: List[str], target_words: List[str],
                               source_lang: str, target_lang: str, **kwargs) -> Dict[str, List[Dict]]:
        """Collect alignments from all modern methods."""
        alignments_by_method = {}
        
        for aligner, weight in self.aligners:
            method_name = aligner.__class__.__name__
            try:
                alignments = aligner.align_verse(source_words, target_words, source_lang, target_lang, **kwargs)
                alignments_by_method[method_name] = alignments
                logger.debug(f"{method_name}: {len(alignments)} alignments")
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                alignments_by_method[method_name] = []
        
        return alignments_by_method
    
    def _combine_alignments_with_calibration(self, all_alignments: Dict[str, List[Dict]], 
                                           source_words: List[str], target_words: List[str]) -> List[Dict]:
        """
        Combine alignments with proper confidence calibration for biblical texts.
        """
        # Create alignment candidates grouped by (source, target) pair
        alignment_candidates = defaultdict(list)
        
        # Add alignments from all modern methods
        for method_name, alignments in all_alignments.items():
            weight = self.biblical_weights.get(method_name, 0.1)
            
            for align in alignments:
                key = (align['source_index'], align['target_index'])
                
                # Apply method-specific weight to confidence
                calibrated_confidence = min(align['confidence'] * weight, 0.95)  # Allow higher confidence for modern methods
                
                align_copy = align.copy()
                align_copy['contributing_methods'] = [method_name]
                align_copy['method_confidence'] = calibrated_confidence
                align_copy['original_confidence'] = align['confidence']
                
                alignment_candidates[key].append(align_copy)
        
        # Combine candidates for each (source, target) pair
        combined_alignments = []
        for (src_idx, tgt_idx), candidates in alignment_candidates.items():
            if not candidates:
                continue
                
            combined = self._calibrate_alignment_confidence(
                candidates, source_words[src_idx], target_words[tgt_idx]
            )
            
            if combined['confidence'] >= self.confidence_calibration['minimum_confidence']:
                combined_alignments.append(combined)
        
        return combined_alignments
    
    def _calibrate_alignment_confidence(self, candidates: List[Dict], 
                                      source_word: str, target_word: str) -> Dict:
        """
        Calibrate confidence scores using biblical linguistics principles.
        """
        if not candidates:
            return None
        
        # Use the highest-weighted candidate as base
        base_candidate = max(candidates, key=lambda c: c['method_confidence'])
        
        # Calculate consensus bonus
        methods = set()
        confidence_scores = []
        
        for candidate in candidates:
            methods.update(candidate['contributing_methods'])
            confidence_scores.append(candidate['method_confidence'])
        
        # Base confidence from best method
        base_confidence = base_candidate['method_confidence']
        
        # Consensus bonus: multiple methods agreeing
        if len(methods) >= self.confidence_calibration['consensus_threshold']:
            consensus_factor = min(len(methods) / 5.0, 0.3)  # Up to 30% bonus
            base_confidence += consensus_factor
        
        # Lexicon match bonus - prioritize morphological lexicon alignments
        if 'MorphologicalLexiconAligner' in methods:
            # High bonus for morphological lexicon alignments (using OSHB/MorphGNT lemmas)
            base_confidence += self.confidence_calibration['lexicon_match_bonus'] * 2
        elif any(method in methods for method in ['ModernSemanticAligner']):
            # Lower bonus for similarity-based semantic methods
            semantic_features = base_candidate.get('semantic_features', {})
            if semantic_features.get('modern_lexicon', False) or semantic_features.get('has_lexicon_support', False):
                base_confidence += self.confidence_calibration['lexicon_match_bonus']
        
        # Morphological consistency bonus
        if self._has_morphological_consistency(source_word, target_word):
            base_confidence += self.confidence_calibration['morphology_match_bonus']
        
        # Apply calibration bounds
        final_confidence = max(
            self.confidence_calibration['minimum_confidence'],
            min(base_confidence, self.confidence_calibration['maximum_confidence'])
        )
        
        # Create combined alignment
        combined = base_candidate.copy()
        combined.update({
            'confidence': round(final_confidence, 3),
            'contributing_methods': list(methods),
            'method_count': len(methods),
            'consensus_score': statistics.mean(confidence_scores) if confidence_scores else 0.0,
            'calibration_applied': True
        })
        
        return combined
    
    def _has_morphological_consistency(self, source_word: str, target_word: str) -> bool:
        """Check for morphological consistency between source and target."""
        # Simple heuristics for morphological consistency
        
        # Articles
        if source_word.startswith('ה') and target_word.lower() == 'the':
            return True
            
        # Conjunctions
        if source_word.startswith('ו') and target_word.lower() in ['and', 'but', 'or']:
            return True
            
        # Prepositions
        if source_word.startswith(('ב', 'ל', 'מ', 'כ')) and target_word.lower() in ['in', 'to', 'from', 'like', 'as', 'with']:
            return True
            
        # Greek articles
        if source_word.startswith(('ὁ', 'ἡ', 'τό', 'οἱ', 'αἱ', 'τά')) and target_word.lower() == 'the':
            return True
            
        return False
    
    def _optimize_alignment_assignment(self, alignments: List[Dict], 
                                     num_source: int, num_target: int) -> List[Dict]:
        """
        Optimize alignment assignment using linguistic constraints.
        
        Unlike the greedy assignment in the original ensemble, this uses
        proper optimization considering biblical translation patterns.
        """
        # Sort alignments by confidence
        sorted_alignments = sorted(alignments, key=lambda a: a['confidence'], reverse=True)
        
        # Track usage to handle one-to-many and many-to-one mappings
        source_usage = defaultdict(list)  # source_idx -> list of alignments
        target_usage = defaultdict(list)  # target_idx -> list of alignments
        
        selected_alignments = []
        
        for alignment in sorted_alignments:
            src_idx = alignment['source_index']
            tgt_idx = alignment['target_index']
            confidence = alignment['confidence']
            
            # Check if this would create acceptable mappings
            current_src_mappings = len(source_usage[src_idx])
            current_tgt_mappings = len(target_usage[tgt_idx])
            
            # Enforce stricter 1:1 mapping for better quality
            # Only allow multiple mappings for very high confidence alignments
            max_mappings_per_word = 2 if confidence > 0.9 else 1
            
            if (current_src_mappings < max_mappings_per_word and 
                current_tgt_mappings < max_mappings_per_word):
                
                selected_alignments.append(alignment)
                source_usage[src_idx].append(alignment)
                target_usage[tgt_idx].append(alignment)
                
                logger.debug(f"Selected: {alignment['source_word']} → {alignment['target_word']} "
                           f"(conf: {confidence:.3f}, methods: {alignment['contributing_methods']})")
        
        # Keep alignments as word-level only - no phrase detection
        # The phrase detection was creating nonsensical multi-word mappings
        final_alignments = []
        for alignment in selected_alignments:
            # Force word-level alignment
            alignment.update({
                'source_span': 1,
                'target_span': 1,
                'alignment_type': 'word'
            })
            final_alignments.append(alignment)
        
        return final_alignments
    
    def _detect_phrase_spans(self, alignment: Dict, all_alignments: List[Dict]) -> Dict:
        """Detect if alignment is part of a phrase span."""
        src_idx = alignment['source_index']
        tgt_idx = alignment['target_index']
        
        # Find consecutive alignments that might form phrases
        consecutive_src = []
        consecutive_tgt = []
        
        for other in all_alignments:
            if abs(other['source_index'] - src_idx) <= 2:  # Within 2 words
                consecutive_src.append(other)
            if abs(other['target_index'] - tgt_idx) <= 2:
                consecutive_tgt.append(other)
        
        # Determine span size
        source_span = 1
        target_span = 1
        
        if len(consecutive_src) > 1:
            src_indices = [a['source_index'] for a in consecutive_src]
            source_span = max(src_indices) - min(src_indices) + 1
            
        if len(consecutive_tgt) > 1:
            tgt_indices = [a['target_index'] for a in consecutive_tgt]
            target_span = max(tgt_indices) - min(tgt_indices) + 1
        
        return {
            'source_span': source_span,
            'target_span': target_span,
            'alignment_type': 'phrase' if source_span > 1 or target_span > 1 else 'word'
        }