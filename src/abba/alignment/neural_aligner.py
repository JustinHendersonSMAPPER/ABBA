"""
Neural word alignment using multilingual sentence transformers.

This module provides neural-based word alignment using pre-trained
multilingual models that can understand semantic relationships
across Hebrew, Greek, and English.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class NeuralAligner:
    """Neural word aligner using multilingual sentence transformers."""
    
    def __init__(self, 
                 base_confidence: float = 0.7,
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 cache_dir: Optional[str] = None,
                 similarity_threshold: float = 0.4):
        """
        Initialize neural aligner.
        
        Args:
            base_confidence: Base confidence score for neural alignments
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory to cache embeddings (None to disable caching)
            similarity_threshold: Minimum cosine similarity for alignment
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. Run: poetry add sentence-transformers"
            )
        
        self.base_confidence = base_confidence
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        
        # Initialize model
        logger.info(f"Loading neural model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("✓ Neural model loaded successfully")
        
        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache/embeddings')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_dir is not None
        
        if self.cache_enabled:
            logger.info(f"Neural embedding cache: {self.cache_dir}")
    
    def _get_cache_key(self, words: List[str], language: str) -> str:
        """Generate cache key for a list of words."""
        content = f"{language}:{','.join(words)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache."""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to cache."""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embeddings)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def _get_embeddings(self, words: List[str], language: str) -> np.ndarray:
        """Get embeddings for words, using cache if available."""
        cache_key = self._get_cache_key(words, language)
        
        # Try to load from cache
        embeddings = self._load_from_cache(cache_key)
        if embeddings is not None:
            logger.debug(f"Loaded embeddings from cache for {len(words)} {language} words")
            return embeddings
        
        # Generate embeddings
        logger.debug(f"Generating embeddings for {len(words)} {language} words")
        embeddings = self.model.encode(words)
        
        # Save to cache
        self._save_to_cache(cache_key, embeddings)
        
        return embeddings
    
    def align_verse(self, 
                   source_words: List[str], 
                   target_words: List[str], 
                   source_lang: str = 'hebrew', 
                   target_lang: str = 'english') -> List[Dict]:
        """
        Align words using neural embeddings.
        
        Args:
            source_words: Words in source language (Hebrew/Greek)
            target_words: Words in target language (usually English)
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of alignment dictionaries with neural confidence scores
        """
        if not source_words or not target_words:
            return []
        
        try:
            # Get embeddings for both sets of words
            source_embeddings = self._get_embeddings(source_words, source_lang)
            target_embeddings = self._get_embeddings(target_words, target_lang)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
            
            # Find best alignments
            alignments = []
            
            for i, source_word in enumerate(source_words):
                # Find best target word for this source word
                similarities = similarity_matrix[i]
                best_target_idx = np.argmax(similarities)
                best_similarity = similarities[best_target_idx]
                
                # Only create alignment if similarity is above threshold
                if best_similarity >= self.similarity_threshold:
                    target_word = target_words[best_target_idx]
                    
                    # Calculate confidence based on similarity and base confidence
                    confidence = self.base_confidence * best_similarity
                    
                    alignment = {
                        'source_index': i,
                        'target_index': best_target_idx,
                        'source_word': source_word,
                        'target_word': target_word,
                        'confidence': float(confidence),
                        'method': 'neural',
                        'similarity': float(best_similarity),
                        'features': {
                            'neural_similarity': float(best_similarity),
                            'model': self.model_name
                        }
                    }
                    
                    alignments.append(alignment)
            
            logger.debug(f"Neural aligner: {len(alignments)}/{len(source_words)} words aligned "
                        f"(threshold: {self.similarity_threshold})")
            
            return alignments
            
        except Exception as e:
            logger.error(f"Neural alignment failed: {e}")
            return []
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about cached embeddings."""
        if not self.cache_enabled:
            return {'cache_enabled': False}
        
        cache_files = list(self.cache_dir.glob('*.npy'))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_enabled': True,
            'cache_dir': str(self.cache_dir),
            'cached_files': len(cache_files),
            'total_cache_size_mb': total_size / (1024 * 1024),
            'model_name': self.model_name
        }
    
    def clear_cache(self) -> int:
        """Clear embedding cache and return number of files removed."""
        if not self.cache_enabled:
            return 0
        
        cache_files = list(self.cache_dir.glob('*.npy'))
        removed_count = 0
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {removed_count} embedding cache files")
        return removed_count