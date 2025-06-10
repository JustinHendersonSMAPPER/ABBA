"""
Enhanced Ensemble Aligner with N:M Mapping Support

This aligner removes the 1:1 constraint and supports phrase-level alignments
for better biblical text alignment quality.
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import re

from .ensemble_aligner import EnsembleAligner

logger = logging.getLogger('ABBA.EnhancedEnsembleAligner')


@dataclass
class AlignmentCandidate:
    """Represents a potential alignment between source and target spans."""
    source_start: int
    source_end: int  # exclusive
    target_start: int
    target_end: int  # exclusive
    confidence: float
    method: str
    features: Dict


class EnhancedEnsembleAligner(EnsembleAligner):
    """Enhanced ensemble aligner supporting N:M mappings and phrase-level alignment."""
    
    def __init__(self, aligners: List, weights: Dict[str, float] = None):
        super().__init__(aligners)
        self.phrase_patterns = self._initialize_phrase_patterns()
        self.confidence_threshold = 0.2  # Higher threshold for better quality
    
    def _initialize_phrase_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for common biblical phrase translations."""
        return {
            # Hebrew construct chains that translate to English phrases
            'hebrew_constructs': [
                r'בן.*אדם',  # "son of man"
                r'מלך.*ישראל',  # "king of Israel"
                r'ארץ.*מצרים',  # "land of Egypt"
                r'בית.*יהוה',  # "house of the LORD"
                r'רוח.*אלהים',  # "Spirit of God"
            ],
            # Greek participial phrases
            'greek_participles': [
                r'λέγων.*αὐτοῖς',  # "saying to them"
                r'ἀποκριθεὶς.*εἶπεν',  # "answering said"
                r'προσελθὼν.*αὐτῷ',  # "coming to him"
            ],
            # English phrases that come from single original words
            'english_expansions': [
                r'in\s+the\s+beginning',  # בראשית
                r'and\s+it\s+came\s+to\s+pass',  # ויהי
                r'thus\s+says\s+the\s+lord',  # כה אמר יהוה
            ]
        }
    
    def align_verse(self, source_words: List[str], target_words: List[str], 
                    source_lang: str = 'hebrew', target_lang: str = 'english', **kwargs) -> List[Dict]:
        """
        Enhanced alignment supporting N:M mappings and phrase detection.
        """
        logger.info(f"Enhanced alignment: {len(source_words)} source → {len(target_words)} target words")
        
        # First get all potential alignments from sub-aligners
        all_candidates = self._collect_alignment_candidates(
            source_words, target_words, source_lang, target_lang
        )
        
        # Detect phrase-level patterns
        phrase_candidates = self._detect_phrase_alignments(
            source_words, target_words, source_lang, target_lang
        )
        all_candidates.extend(phrase_candidates)
        
        # Score and filter candidates
        scored_candidates = self._score_candidates(all_candidates, source_words, target_words)
        
        # Select best non-conflicting alignments using beam search
        final_alignments = self._select_best_alignments(
            scored_candidates, len(source_words), len(target_words)
        )
        
        logger.info(f"Enhanced alignment completed: {len(final_alignments)} alignments found")
        return final_alignments
    
    def _collect_alignment_candidates(self, source_words: List[str], target_words: List[str],
                                    source_lang: str, target_lang: str) -> List[AlignmentCandidate]:
        """Collect alignment candidates from all sub-aligners."""
        candidates = []
        
        for aligner, weight in self.aligners:
            try:
                alignments = aligner.align_verse(source_words, target_words, source_lang, target_lang)
                
                for alignment in alignments:
                    candidate = AlignmentCandidate(
                        source_start=alignment['source_index'],
                        source_end=alignment['source_index'] + 1,
                        target_start=alignment['target_index'],
                        target_end=alignment['target_index'] + 1,
                        confidence=alignment['confidence'],
                        method=alignment.get('method', aligner.__class__.__name__),
                        features={'alignment': alignment}
                    )
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.warning(f"Aligner {aligner.__class__.__name__} failed: {e}")
        
        return candidates
    
    def _detect_phrase_alignments(self, source_words: List[str], target_words: List[str],
                                source_lang: str, target_lang: str) -> List[AlignmentCandidate]:
        """Detect phrase-level alignment patterns."""
        candidates = []
        
        # Detect Hebrew construct chains
        if source_lang == 'hebrew':
            candidates.extend(self._detect_hebrew_constructs(source_words, target_words))
        
        # Detect Greek participial phrases  
        elif source_lang == 'greek':
            candidates.extend(self._detect_greek_participles(source_words, target_words))
        
        # Detect English phrase expansions
        candidates.extend(self._detect_english_expansions(source_words, target_words))
        
        return candidates
    
    def _detect_hebrew_constructs(self, source_words: List[str], target_words: List[str]) -> List[AlignmentCandidate]:
        """Detect Hebrew construct chain patterns."""
        candidates = []
        source_text = ' '.join(source_words)
        target_text = ' '.join(target_words)
        
        for pattern in self.phrase_patterns['hebrew_constructs']:
            matches = list(re.finditer(pattern, source_text))
            for match in matches:
                # Find word boundaries for the match
                source_start, source_end = self._find_word_boundaries(
                    source_words, match.start(), match.end()
                )
                
                # Look for corresponding English phrase
                if 'בן.*אדם' in pattern and ('son' in target_text.lower() and 'man' in target_text.lower()):
                    target_indices = self._find_phrase_in_target(target_words, ['son', 'man'])
                    if target_indices:
                        candidates.append(AlignmentCandidate(
                            source_start=source_start,
                            source_end=source_end,
                            target_start=target_indices[0],
                            target_end=target_indices[-1] + 1,
                            confidence=0.8,
                            method='hebrew_construct',
                            features={'pattern': pattern, 'phrase_type': 'construct_chain'}
                        ))
        
        return candidates
    
    def _detect_english_expansions(self, source_words: List[str], target_words: List[str]) -> List[AlignmentCandidate]:
        """Detect cases where single original words expand to English phrases."""
        candidates = []
        target_text = ' '.join(target_words).lower()
        
        # Common expansions
        expansions = {
            'בראשית': ['in', 'the', 'beginning'],
            'ויהי': ['and', 'it', 'came', 'to', 'pass'],
            'והיה': ['and', 'it', 'shall', 'be'],
        }
        
        for i, source_word in enumerate(source_words):
            # Remove vowel points and prefixes for matching
            clean_source = re.sub(r'[^\u05D0-\u05EA]', '', source_word)
            
            if clean_source in expansions:
                english_phrase = expansions[clean_source]
                target_indices = self._find_phrase_in_target(target_words, english_phrase)
                
                if target_indices:
                    candidates.append(AlignmentCandidate(
                        source_start=i,
                        source_end=i + 1,
                        target_start=target_indices[0],
                        target_end=target_indices[-1] + 1,
                        confidence=0.9,
                        method='phrase_expansion',
                        features={'expansion_type': 'single_to_phrase', 'phrase': english_phrase}
                    ))
        
        return candidates
    
    def _find_phrase_in_target(self, target_words: List[str], phrase_words: List[str]) -> Optional[List[int]]:
        """Find a phrase in target words, allowing for some flexibility."""
        target_lower = [w.lower().strip('.,!?;:') for w in target_words]
        phrase_lower = [w.lower() for w in phrase_words]
        
        # Exact sequence match
        for i in range(len(target_lower) - len(phrase_lower) + 1):
            if target_lower[i:i+len(phrase_lower)] == phrase_lower:
                return list(range(i, i + len(phrase_lower)))
        
        # Flexible match allowing intervening words
        phrase_indices = []
        phrase_idx = 0
        
        for i, word in enumerate(target_lower):
            if phrase_idx < len(phrase_lower) and word == phrase_lower[phrase_idx]:
                phrase_indices.append(i)
                phrase_idx += 1
                
                if phrase_idx == len(phrase_lower):
                    # Found complete phrase
                    if len(phrase_indices) == len(phrase_lower):
                        return phrase_indices
        
        return None
    
    def _find_word_boundaries(self, words: List[str], char_start: int, char_end: int) -> Tuple[int, int]:
        """Find word indices corresponding to character positions."""
        current_pos = 0
        start_word = 0
        end_word = len(words)
        
        for i, word in enumerate(words):
            word_start = current_pos
            word_end = current_pos + len(word)
            
            if word_start <= char_start < word_end:
                start_word = i
            if word_start < char_end <= word_end:
                end_word = i + 1
                break
                
            current_pos += len(word) + 1  # +1 for space
        
        return start_word, end_word
    
    def _score_candidates(self, candidates: List[AlignmentCandidate], 
                         source_words: List[str], target_words: List[str]) -> List[AlignmentCandidate]:
        """Enhanced scoring for alignment candidates."""
        for candidate in candidates:
            # Base confidence from the alignment method
            base_confidence = candidate.confidence
            
            # Span length bonus/penalty
            source_span = candidate.source_end - candidate.source_start
            target_span = candidate.target_end - candidate.target_start
            
            # Prefer balanced span lengths
            span_ratio = min(source_span, target_span) / max(source_span, target_span)
            span_bonus = 0.1 * span_ratio
            
            # Method-specific bonuses
            method_bonus = 0.0
            if candidate.method in ['hebrew_construct', 'greek_participle', 'phrase_expansion']:
                method_bonus = 0.2  # Phrase methods get higher confidence
            elif candidate.method == 'StatisticalAligner':
                method_bonus = 0.1
            elif candidate.method == 'MorphologicalAligner':
                method_bonus = 0.05
            
            # Length appropriateness
            if source_span > 1 and target_span > 1:
                # Multi-word alignments are good if they're phrase patterns
                if candidate.method in ['hebrew_construct', 'phrase_expansion']:
                    method_bonus += 0.15
            
            # Final confidence calculation
            candidate.confidence = min(1.0, base_confidence + span_bonus + method_bonus)
        
        # Filter by confidence threshold
        return [c for c in candidates if c.confidence >= self.confidence_threshold]
    
    def _select_best_alignments(self, candidates: List[AlignmentCandidate], 
                              num_source: int, num_target: int) -> List[Dict]:
        """Select best non-overlapping alignments using beam search."""
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        selected = []
        source_covered = set()
        target_covered = set()
        
        for candidate in candidates:
            # Check for conflicts with already selected alignments
            source_range = set(range(candidate.source_start, candidate.source_end))
            target_range = set(range(candidate.target_start, candidate.target_end))
            
            # Allow some overlap for phrase alignments but prevent total overlap
            source_overlap = len(source_range & source_covered) / len(source_range)
            target_overlap = len(target_range & target_covered) / len(target_range)
            
            # Accept if overlap is minimal (< 50%) or if it's a high-confidence phrase alignment
            if (source_overlap < 0.5 and target_overlap < 0.5) or \
               (candidate.confidence > 0.8 and candidate.method in ['hebrew_construct', 'phrase_expansion']):
                
                selected.append(self._candidate_to_alignment(candidate))
                source_covered.update(source_range)
                target_covered.update(target_range)
        
        return selected
    
    def _candidate_to_alignment(self, candidate: AlignmentCandidate) -> Dict:
        """Convert alignment candidate to final alignment format."""
        return {
            'source_index': candidate.source_start,
            'target_index': candidate.target_start,
            'source_span': candidate.source_end - candidate.source_start,
            'target_span': candidate.target_end - candidate.target_start,
            'source_word': f"span[{candidate.source_start}:{candidate.source_end}]",
            'target_word': f"span[{candidate.target_start}:{candidate.target_end}]",
            'confidence': round(candidate.confidence, 3),
            'method': candidate.method,
            'alignment_type': 'phrase' if candidate.source_end - candidate.source_start > 1 or 
                                        candidate.target_end - candidate.target_start > 1 else 'word',
            'features': candidate.features
        }
    
    def _detect_greek_participles(self, source_words: List[str], target_words: List[str]) -> List[AlignmentCandidate]:
        """Detect Greek participial phrase patterns."""
        candidates = []
        # Implementation for Greek participle detection would go here
        # This is a placeholder for now
        return candidates