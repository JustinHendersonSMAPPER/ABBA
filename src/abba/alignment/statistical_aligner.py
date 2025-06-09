"""
Statistical word aligner using trained translation probabilities.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class StatisticalAligner:
    """Aligner using statistical translation models."""
    
    def __init__(self, models_dir: Path = Path('models/statistical'), 
                 base_confidence: float = 0.8):
        """
        Initialize statistical aligner.
        
        Args:
            models_dir: Directory containing trained models
            base_confidence: Base confidence multiplier
        """
        self.models_dir = models_dir
        self.base_confidence = base_confidence
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained statistical models."""
        # Hebrew-English model
        hebrew_model_path = self.models_dir / 'hebrew_english_alignment.json'
        if hebrew_model_path.exists():
            with open(hebrew_model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                self.models['hebrew'] = model_data['translation_probs']
                logger.info(f"Loaded Hebrew model with {len(self.models['hebrew'])} words")
        
        # Greek-English model
        greek_model_path = self.models_dir / 'greek_english_alignment.json'
        if greek_model_path.exists():
            with open(greek_model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                self.models['greek'] = model_data['translation_probs']
                logger.info(f"Loaded Greek model with {len(self.models['greek'])} words")
    
    def align_verse(
        self,
        source_words: List[str],
        target_words: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, any]]:
        """
        Align words using statistical translation probabilities.
        
        Args:
            source_words: Source language words
            target_words: Target language words
            source_lang: Source language ('hebrew' or 'greek')
            target_lang: Target language (ignored, assumes English)
            
        Returns:
            List of alignment dictionaries
        """
        if not source_words or not target_words:
            return []
        
        if source_lang not in self.models:
            logger.warning(f"No statistical model for {source_lang}")
            return []
        
        model = self.models[source_lang]
        alignments = []
        
        # Build score matrix
        scores = {}
        for src_idx, src_word in enumerate(source_words):
            if src_word in model:
                translations = model[src_word]
                for tgt_idx, tgt_word in enumerate(target_words):
                    # Direct match
                    if tgt_word in translations:
                        score = translations[tgt_word] * self.base_confidence
                        scores[(src_idx, tgt_idx)] = score
                    # Case-insensitive match
                    elif tgt_word.lower() in translations:
                        score = translations[tgt_word.lower()] * self.base_confidence * 0.9
                        scores[(src_idx, tgt_idx)] = score
                    else:
                        # Check if any translation matches with lowercased target
                        for trans_word, trans_score in translations.items():
                            if trans_word.lower() == tgt_word.lower():
                                score = trans_score * self.base_confidence * 0.8
                                scores[(src_idx, tgt_idx)] = score
                                break
        
        # Extract best alignments using greedy approach
        used_targets = set()
        
        # Sort by score
        sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for (src_idx, tgt_idx), score in sorted_pairs:
            if tgt_idx not in used_targets and score > 0.3:  # Minimum threshold
                used_targets.add(tgt_idx)
                
                alignment = {
                    'source_index': src_idx,
                    'target_index': tgt_idx,
                    'source_word': source_words[src_idx],
                    'target_word': target_words[tgt_idx],
                    'confidence': round(score, 3),
                    'method': 'statistical'
                }
                alignments.append(alignment)
        
        # Sort by source index
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
            
        Returns:
            List of alignment lists
        """
        results = []
        
        for source_words, target_words in verse_pairs:
            alignments = self.align_verse(
                source_words, target_words, source_lang, target_lang
            )
            results.append(alignments)
        
        return results
    
    def get_model_stats(self) -> Dict[str, Dict]:
        """Get statistics about loaded models."""
        stats = {}
        
        for lang, model in self.models.items():
            # Find most confident translations
            top_pairs = []
            for src_word, translations in model.items():
                for tgt_word, score in translations.items():
                    if score > 0.9:  # High confidence
                        top_pairs.append((src_word, tgt_word, score))
            
            top_pairs.sort(key=lambda x: x[2], reverse=True)
            
            stats[lang] = {
                'vocab_size': len(model),
                'total_translations': sum(len(t) for t in model.values()),
                'top_translations': top_pairs[:10]
            }
        
        return stats