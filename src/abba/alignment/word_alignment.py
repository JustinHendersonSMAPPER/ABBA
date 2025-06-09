"""
Word-level alignment between original languages and translations.

This module implements statistical word alignment using IBM Model 1 with
enhancements for biblical texts including morphological constraints and
Strong's number anchoring.
"""

import json
import logging
import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignmentPair:
    """Represents an alignment between source and target positions."""
    source_idx: int
    target_idx: int
    confidence: float
    source_word: str = ""
    target_word: str = ""
    
    
@dataclass 
class WordAlignment:
    """Complete alignment for a verse."""
    verse_id: str
    source_words: List[str]
    target_words: List[str]
    alignments: List[AlignmentPair]
    alignment_score: float = 0.0


class IBMModel1:
    """
    IBM Model 1 word alignment with biblical text enhancements.
    
    Enhancements include:
    - Strong's number constraints
    - Morphological decomposition
    - Lexicon-based initialization
    - Function word handling
    """
    
    def __init__(self, 
                 use_morphology: bool = True,
                 use_strongs: bool = True,
                 use_lexicon: bool = True):
        """Initialize the alignment model."""
        self.use_morphology = use_morphology
        self.use_strongs = use_strongs
        self.use_lexicon = use_lexicon
        
        # Translation probabilities: P(target|source)
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        
        # Vocabulary
        self.source_vocab = set()
        self.target_vocab = set()
        
        # Special tokens
        self.NULL_TOKEN = "<NULL>"
        
        # Morphology and Strong's data
        self.source_morphology = {}  # word -> morphological data
        self.source_strongs = {}     # word -> Strong's number
        self.lexicon_data = {}       # Strong's -> gloss
        
        # Training parameters
        self.num_iterations = 10
        self.convergence_threshold = 0.001
        
    def initialize_from_lexicon(self, lexicon_data: Dict[str, Any]):
        """Initialize translation probabilities from lexicon."""
        self.lexicon_data = lexicon_data
        
        # Use lexicon glosses to initialize probabilities
        for strongs, entry in lexicon_data.items():
            if 'gloss' in entry:
                glosses = entry['gloss'].lower().split(';')
                for gloss in glosses:
                    words = gloss.strip().split()
                    for word in words:
                        # Higher initial probability for lexicon matches
                        self.trans_probs[strongs][word] = 0.5
                        
    def add_morphological_data(self, word: str, morph_data: Dict[str, Any], 
                              strongs: Optional[str] = None):
        """Add morphological information for a source word."""
        self.source_morphology[word] = morph_data
        if strongs:
            self.source_strongs[word] = strongs
            
    def tokenize_source(self, hebrew_words: List[Dict]) -> List[str]:
        """Tokenize Hebrew/Greek source text with morphological awareness."""
        tokens = []
        
        for word_data in hebrew_words:
            word = word_data.get('text', '')
            if not word:
                continue
                
            # Store morphological data
            if 'morphology' in word_data:
                self.add_morphological_data(
                    word, 
                    word_data['morphology'],
                    word_data.get('lemma', '').split('/')[-1]
                )
            
            if self.use_morphology and '/' in word:
                # Split compound forms (e.g., "בְּ/רֵאשִׁית" -> ["בְּ", "רֵאשִׁית"])
                parts = word.split('/')
                tokens.extend(parts)
            else:
                tokens.append(word)
                
        return tokens
        
    def tokenize_target(self, text: str) -> List[str]:
        """Tokenize English target text."""
        # Simple tokenization - could be enhanced
        import re
        text = text.lower()
        # Keep punctuation separate
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
        
    def train(self, parallel_corpus: List[Tuple[List[Dict], str]], 
              verbose: bool = True):
        """
        Train IBM Model 1 on parallel corpus.
        
        Args:
            parallel_corpus: List of (source_words, target_text) pairs
            verbose: Print training progress
        """
        if verbose:
            logger.info(f"Training IBM Model 1 on {len(parallel_corpus)} verses")
            
        # Prepare training data
        training_pairs = []
        for source_words, target_text in parallel_corpus:
            source_tokens = self.tokenize_source(source_words)
            target_tokens = self.tokenize_target(target_text)
            
            if source_tokens and target_tokens:
                training_pairs.append((source_tokens, target_tokens))
                self.source_vocab.update(source_tokens)
                self.target_vocab.update(target_tokens)
                
        # Add NULL token for unaligned words
        self.source_vocab.add(self.NULL_TOKEN)
        
        # Initialize uniform probabilities
        self._initialize_probabilities()
        
        # EM training
        prev_likelihood = float('-inf')
        
        for iteration in range(self.num_iterations):
            # E-step: collect expected counts
            counts = defaultdict(lambda: defaultdict(float))
            total_counts = defaultdict(float)
            
            for source_tokens, target_tokens in training_pairs:
                # Add NULL token for unaligned words
                source_with_null = [self.NULL_TOKEN] + source_tokens
                
                for t_idx, t_word in enumerate(target_tokens):
                    # Compute alignment probabilities
                    norm = 0.0
                    probs = []
                    
                    for s_idx, s_word in enumerate(source_with_null):
                        prob = self.trans_probs[s_word][t_word]
                        probs.append(prob)
                        norm += prob
                        
                    # Normalize and update counts
                    if norm > 0:
                        for s_idx, s_word in enumerate(source_with_null):
                            prob = probs[s_idx] / norm
                            counts[s_word][t_word] += prob
                            total_counts[s_word] += prob
                            
            # M-step: update translation probabilities
            likelihood = 0.0
            for s_word in self.source_vocab:
                for t_word in self.target_vocab:
                    if total_counts[s_word] > 0:
                        self.trans_probs[s_word][t_word] = (
                            counts[s_word][t_word] / total_counts[s_word]
                        )
                        if counts[s_word][t_word] > 0:
                            likelihood += counts[s_word][t_word] * math.log(
                                self.trans_probs[s_word][t_word]
                            )
                            
            if verbose:
                logger.info(f"Iteration {iteration + 1}: likelihood = {likelihood:.2f}")
                
            # Check convergence
            if abs(likelihood - prev_likelihood) < self.convergence_threshold:
                if verbose:
                    logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            prev_likelihood = likelihood
            
    def _initialize_probabilities(self):
        """Initialize translation probabilities."""
        # Uniform initialization
        init_prob = 1.0 / len(self.target_vocab)
        
        for s_word in self.source_vocab:
            for t_word in self.target_vocab:
                if self.use_lexicon and s_word in self.lexicon_data:
                    # Use lexicon for initialization if available
                    gloss = self.lexicon_data[s_word].get('gloss', '').lower()
                    if t_word in gloss:
                        self.trans_probs[s_word][t_word] = 0.5
                    else:
                        self.trans_probs[s_word][t_word] = init_prob / 2
                else:
                    self.trans_probs[s_word][t_word] = init_prob
                    
    def align_verse(self, source_words: List[Dict], target_text: str,
                   threshold: float = 0.1) -> WordAlignment:
        """
        Align a single verse.
        
        Args:
            source_words: Hebrew/Greek words with morphology
            target_text: English translation
            threshold: Minimum probability threshold for alignments
            
        Returns:
            WordAlignment object with alignments
        """
        source_tokens = self.tokenize_source(source_words)
        target_tokens = self.tokenize_target(target_text)
        
        alignments = []
        alignment_score = 0.0
        
        # Add NULL token
        source_with_null = [self.NULL_TOKEN] + source_tokens
        
        for t_idx, t_word in enumerate(target_tokens):
            # Find best source word
            best_prob = 0.0
            best_s_idx = -1
            
            for s_idx, s_word in enumerate(source_with_null):
                prob = self.trans_probs[s_word][t_word]
                
                # Apply Strong's number constraint
                if self.use_strongs and s_word in self.source_strongs:
                    strongs = self.source_strongs[s_word]
                    if strongs in self.lexicon_data:
                        gloss = self.lexicon_data[strongs].get('gloss', '').lower()
                        if t_word in gloss:
                            prob *= 2.0  # Boost probability
                            
                if prob > best_prob:
                    best_prob = prob
                    best_s_idx = s_idx
                    
            # Create alignment if above threshold
            if best_prob > threshold and best_s_idx > 0:  # Skip NULL alignments
                alignment = AlignmentPair(
                    source_idx=best_s_idx - 1,  # Adjust for NULL token
                    target_idx=t_idx,
                    confidence=best_prob,
                    source_word=source_tokens[best_s_idx - 1],
                    target_word=t_word
                )
                alignments.append(alignment)
                alignment_score += best_prob
                
        # Normalize alignment score
        if alignments:
            alignment_score /= len(alignments)
            
        return WordAlignment(
            verse_id="",
            source_words=source_tokens,
            target_words=target_tokens,
            alignments=alignments,
            alignment_score=alignment_score
        )
        
    def save_model(self, path: Path):
        """Save trained model to disk."""
        model_data = {
            'trans_probs': dict(self.trans_probs),
            'source_vocab': list(self.source_vocab),
            'target_vocab': list(self.target_vocab),
            'source_morphology': self.source_morphology,
            'source_strongs': self.source_strongs,
            'lexicon_data': self.lexicon_data,
            'parameters': {
                'use_morphology': self.use_morphology,
                'use_strongs': self.use_strongs,
                'use_lexicon': self.use_lexicon,
                'num_iterations': self.num_iterations
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved alignment model to {path}")
        
    def load_model(self, path: Path):
        """Load trained model from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        # Restore translation probabilities
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        for s_word, targets in model_data['trans_probs'].items():
            for t_word, prob in targets.items():
                self.trans_probs[s_word][t_word] = prob
                
        self.source_vocab = set(model_data['source_vocab'])
        self.target_vocab = set(model_data['target_vocab'])
        self.source_morphology = model_data.get('source_morphology', {})
        self.source_strongs = model_data.get('source_strongs', {})
        self.lexicon_data = model_data.get('lexicon_data', {})
        
        # Restore parameters
        params = model_data.get('parameters', {})
        self.use_morphology = params.get('use_morphology', True)
        self.use_strongs = params.get('use_strongs', True)
        self.use_lexicon = params.get('use_lexicon', True)
        self.num_iterations = params.get('num_iterations', 10)
        
        logger.info(f"Loaded alignment model from {path}")


class PhrasalAligner:
    """
    Enhanced aligner that handles phrase-level alignments.
    
    This builds on IBM Model 1 to identify multi-word expressions
    and idiomatic phrases that should be aligned as units.
    """
    
    def __init__(self, base_model: IBMModel1):
        self.base_model = base_model
        self.phrase_table = defaultdict(lambda: defaultdict(float))
        
    def extract_phrases(self, alignments: List[WordAlignment], 
                       max_phrase_length: int = 4):
        """Extract phrase pairs from word alignments."""
        phrase_counts = defaultdict(int)
        
        for alignment in alignments:
            # Find consecutive alignment patterns
            source_phrases = self._extract_consecutive_phrases(
                alignment.alignments, 'source', max_phrase_length
            )
            target_phrases = self._extract_consecutive_phrases(
                alignment.alignments, 'target', max_phrase_length
            )
            
            # Count co-occurring phrases
            for s_phrase in source_phrases:
                for t_phrase in target_phrases:
                    if self._phrases_aligned(s_phrase, t_phrase, alignment.alignments):
                        s_text = ' '.join(alignment.source_words[i] for i in s_phrase)
                        t_text = ' '.join(alignment.target_words[i] for i in t_phrase)
                        phrase_counts[(s_text, t_text)] += 1
                        
        # Convert counts to probabilities
        for (s_phrase, t_phrase), count in phrase_counts.items():
            if count > 2:  # Minimum frequency threshold
                self.phrase_table[s_phrase][t_phrase] = count
                
    def _extract_consecutive_phrases(self, alignments: List[AlignmentPair],
                                   side: str, max_length: int) -> List[List[int]]:
        """Extract consecutive index sequences from alignments."""
        indices = sorted(set(
            a.source_idx if side == 'source' else a.target_idx 
            for a in alignments
        ))
        
        phrases = []
        for length in range(1, min(max_length + 1, len(indices) + 1)):
            for i in range(len(indices) - length + 1):
                phrase = indices[i:i + length]
                if all(phrase[j] + 1 == phrase[j + 1] for j in range(len(phrase) - 1)):
                    phrases.append(phrase)
                    
        return phrases
        
    def _phrases_aligned(self, source_phrase: List[int], 
                        target_phrase: List[int],
                        alignments: List[AlignmentPair]) -> bool:
        """Check if source and target phrases are aligned."""
        aligned_pairs = {(a.source_idx, a.target_idx) for a in alignments}
        
        # Check if any word in source phrase aligns to any word in target phrase
        for s_idx in source_phrase:
            for t_idx in target_phrase:
                if (s_idx, t_idx) in aligned_pairs:
                    return True
        return False
        
    def align_with_phrases(self, source_words: List[Dict], 
                          target_text: str) -> WordAlignment:
        """Align verse considering phrasal alignments."""
        # Get base word alignments
        alignment = self.base_model.align_verse(source_words, target_text)
        
        # Enhance with phrasal alignments
        # TODO: Implement phrase-based alignment enhancement
        
        return alignment


def format_alignment_output(alignment: WordAlignment, 
                          translations: Dict[str, str]) -> Dict[str, Any]:
    """Format alignment data for export."""
    aligned_data = {
        'source_words': alignment.source_words,
        'alignments': {}
    }
    
    for trans_code, trans_text in translations.items():
        # Group alignments by source word
        source_alignments = defaultdict(list)
        
        for align in alignment.alignments:
            source_alignments[align.source_idx].append({
                'target_idx': align.target_idx,
                'target_word': align.target_word,
                'confidence': round(align.confidence, 3)
            })
            
        # Create alignment entries
        alignment_entries = []
        for s_idx, s_word in enumerate(alignment.source_words):
            if s_idx in source_alignments:
                target_indices = [a['target_idx'] for a in source_alignments[s_idx]]
                target_words = [a['target_word'] for a in source_alignments[s_idx]]
                avg_confidence = sum(a['confidence'] for a in source_alignments[s_idx]) / len(source_alignments[s_idx])
                
                alignment_entries.append({
                    'source_idx': s_idx,
                    'source_word': s_word,
                    'target_indices': target_indices,
                    'target_phrase': ' '.join(target_words),
                    'confidence': round(avg_confidence, 3)
                })
                
        aligned_data['alignments'][trans_code] = alignment_entries
        
    return aligned_data