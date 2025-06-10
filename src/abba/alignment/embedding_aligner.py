"""
Cross-Lingual Embedding Aligner for Biblical Texts

Uses modern multilingual embeddings trained on diverse corpora,
not biased toward archaic translations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('ABBA.EmbeddingAligner')


class CrossLingualEmbeddingAligner:
    """
    Aligner using cross-lingual word embeddings for semantic similarity.
    
    This approach avoids the Strong's concordance bias by using:
    1. Modern multilingual embeddings (mBERT, XLM-R, LaBSE)
    2. Contextual understanding
    3. Training on diverse modern corpora
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding_model = embedding_model
        self.embeddings_cache = {}
        self.use_real_embeddings = False
        
        # Initialize embedding model - try real first, fallback to mock
        if embedding_model == "mock":
            self._create_mock_embeddings()
        else:
            self._load_real_embeddings(embedding_model)
    
    def _create_mock_embeddings(self):
        """
        Create mock embeddings that demonstrate the concept.
        In production, this would use actual multilingual models.
        """
        # Mock embeddings based on semantic similarity
        self.mock_embeddings = {
            # Hebrew words
            "אלהים": np.array([0.9, 0.1, 0.0, 0.8, 0.2]),  # Divine concept
            "ארץ": np.array([0.1, 0.9, 0.8, 0.1, 0.1]),    # Earth/land concept  
            "שמים": np.array([0.2, 0.8, 0.1, 0.9, 0.3]),   # Sky/heaven concept
            "ברא": np.array([0.3, 0.2, 0.9, 0.4, 0.8]),    # Creation concept
            "ראשית": np.array([0.7, 0.1, 0.2, 0.3, 0.9]),  # Beginning concept
            "את": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),     # Function word
            "ה": np.array([0.2, 0.1, 0.0, 0.1, 0.0]),      # Article
            "ו": np.array([0.1, 0.0, 0.1, 0.2, 0.0]),      # Conjunction
            "ב": np.array([0.0, 0.2, 0.1, 0.0, 0.1]),      # Preposition
            
            # Greek words  
            "θεός": np.array([0.9, 0.1, 0.0, 0.8, 0.2]),   # Divine concept
            "κόσμος": np.array([0.1, 0.9, 0.8, 0.1, 0.1]), # World concept
            "λόγος": np.array([0.4, 0.3, 0.7, 0.6, 0.5]),  # Word/reason concept
            "ἀρχή": np.array([0.7, 0.1, 0.2, 0.3, 0.9]),   # Beginning concept
            "ὁ": np.array([0.2, 0.1, 0.0, 0.1, 0.0]),      # Article
            "καί": np.array([0.1, 0.0, 0.1, 0.2, 0.0]),    # Conjunction
            
            # English words
            "God": np.array([0.9, 0.1, 0.0, 0.8, 0.2]),    # Divine concept
            "earth": np.array([0.1, 0.9, 0.8, 0.1, 0.1]),  # Earth concept
            "land": np.array([0.1, 0.9, 0.7, 0.1, 0.2]),   # Land concept (similar to earth)
            "heaven": np.array([0.2, 0.8, 0.1, 0.9, 0.3]), # Heaven concept
            "heavens": np.array([0.2, 0.8, 0.1, 0.9, 0.3]), # Heaven concept (plural)
            "sky": np.array([0.2, 0.7, 0.2, 0.8, 0.2]),    # Sky concept (similar to heaven)
            "create": np.array([0.3, 0.2, 0.9, 0.4, 0.8]), # Creation concept
            "created": np.array([0.3, 0.2, 0.9, 0.4, 0.8]), # Creation concept (past)
            "make": np.array([0.2, 0.3, 0.8, 0.3, 0.7]),   # Similar to create
            "beginning": np.array([0.7, 0.1, 0.2, 0.3, 0.9]), # Beginning concept
            "start": np.array([0.6, 0.2, 0.3, 0.2, 0.8]),  # Similar to beginning
            "first": np.array([0.5, 0.1, 0.1, 0.2, 0.8]),  # Similar to beginning
            "the": np.array([0.2, 0.1, 0.0, 0.1, 0.0]),    # Article
            "and": np.array([0.1, 0.0, 0.1, 0.2, 0.0]),    # Conjunction
            "in": np.array([0.0, 0.2, 0.1, 0.0, 0.1]),     # Preposition
            "on": np.array([0.0, 0.3, 0.1, 0.0, 0.1]),     # Preposition (similar to in)
            "with": np.array([0.1, 0.2, 0.0, 0.1, 0.1]),   # Preposition
        }
        
        logger.info("Created mock cross-lingual embeddings for demonstration")
    
    def _load_real_embeddings(self, model_name: str):
        """
        Load real multilingual embeddings.
        Uses models like:
        - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        - sentence-transformers/LaBSE
        - microsoft/Multilingual-MiniLM-L12-H384
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading real embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.use_real_embeddings = True
            logger.info(f"✓ Successfully loaded {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self._create_mock_embeddings()
            self.use_real_embeddings = False
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}, using mock embeddings")
            self._create_mock_embeddings()
            self.use_real_embeddings = False
    
    def get_embedding(self, word: str, language: str = None) -> np.ndarray:
        """Get embedding vector for a word."""
        # Clean the word
        clean_word = self._clean_word(word, language)
        
        # Check cache first
        if clean_word in self.embeddings_cache:
            return self.embeddings_cache[clean_word]
        
        # Get embedding
        if self.use_real_embeddings:
            embedding = self._get_real_embedding(clean_word)
        else:
            embedding = self._get_mock_embedding(clean_word, word, language)
        
        # Cache it
        self.embeddings_cache[clean_word] = embedding
        return embedding
    
    def _clean_word(self, word: str, language: str) -> str:
        """Clean word for embedding lookup."""
        if language == 'hebrew':
            # Remove vowel points and cantillation marks
            cleaned = re.sub(r'[\u0591-\u05C7]', '', word)
            return cleaned
        elif language == 'greek':
            # Remove diacritics
            cleaned = re.sub(r'[\u0300-\u036F]', '', word)
            return cleaned.lower()
        else:
            # English - just lowercase and remove punctuation
            return word.lower().strip('.,!?;:')
    
    def _get_mock_embedding(self, clean_word: str, original_word: str, language: str) -> np.ndarray:
        """Get mock embedding with fallback logic."""
        # Try exact match first
        if clean_word in self.mock_embeddings:
            return self.mock_embeddings[clean_word]
        
        # Try original word
        if original_word in self.mock_embeddings:
            return self.mock_embeddings[original_word]
        
        # Try partial matching for Hebrew/Greek with prefixes
        if language in ['hebrew', 'greek']:
            for key in self.mock_embeddings:
                if clean_word in key or key in clean_word:
                    return self.mock_embeddings[key]
        
        # Default embedding for unknown words
        return np.random.normal(0, 0.1, 5)
    
    def _get_real_embedding(self, word: str) -> np.ndarray:
        """Get real embedding from transformer model."""
        try:
            embedding = self.model.encode([word])[0]
            return np.array(embedding)
        except Exception as e:
            logger.warning(f"Error getting embedding for '{word}': {e}")
            # Fallback to random vector
            return np.random.normal(0, 0.1, 384)  # MiniLM has 384 dimensions
    
    def align_verse(self, source_words: List[str], target_words: List[str],
                   source_lang: str = 'hebrew', target_lang: str = 'english', **kwargs) -> List[Dict]:
        """
        Align words using cross-lingual embeddings (optimized with batching).
        """
        logger.debug(f"Cross-lingual embedding alignment: {len(source_words)} {source_lang} → {len(target_words)} {target_lang}")
        
        # Get embeddings for all words (batch processing for real embeddings)
        if self.use_real_embeddings:
            source_embeddings = self._get_batch_embeddings(source_words, source_lang)
            target_embeddings = self._get_batch_embeddings(target_words, target_lang)
        else:
            source_embeddings = [self.get_embedding(word, source_lang) for word in source_words]
            target_embeddings = [self.get_embedding(word, target_lang) for word in target_words]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
        
        alignments = []
        
        for src_idx, src_word in enumerate(source_words):
            # Get similarities for this source word
            similarities = similarity_matrix[src_idx]
            
            # Find best matches above threshold
            threshold = 0.75  # Higher threshold for better precision
            
            # Only keep top 2 matches to reduce noise
            top_indices = np.argsort(similarities)[-2:][::-1]  # Top 2 in descending order
            
            for tgt_idx in top_indices:
                similarity = similarities[tgt_idx]
                if similarity >= threshold:
                    alignment = {
                        'source_index': src_idx,
                        'target_index': tgt_idx,
                        'source_word': src_word,
                        'target_word': target_words[tgt_idx],
                        'confidence': round(float(similarity), 3),
                        'method': 'cross_lingual_embedding',
                        'semantic_features': {
                            'embedding_similarity': float(similarity),
                            'model': self.embedding_model,
                            'alignment_type': 'semantic'
                        }
                    }
                    alignments.append(alignment)
        
        logger.debug(f"Cross-lingual embedding alignment found {len(alignments)} alignments")
        return alignments
    
    def _get_batch_embeddings(self, words: List[str], language: str) -> List[np.ndarray]:
        """Get embeddings for multiple words efficiently using batching."""
        # Clean all words first
        cleaned_words = [self._clean_word(word, language) for word in words]
        
        # Check cache for all words
        cached_embeddings = []
        uncached_words = []
        uncached_indices = []
        
        for i, clean_word in enumerate(cleaned_words):
            if clean_word in self.embeddings_cache:
                cached_embeddings.append((i, self.embeddings_cache[clean_word]))
            else:
                uncached_words.append(clean_word)
                uncached_indices.append(i)
        
        # Get embeddings for uncached words in batch
        if uncached_words:
            try:
                batch_embeddings = self.model.encode(uncached_words)
                for i, (word_idx, embedding) in enumerate(zip(uncached_indices, batch_embeddings)):
                    self.embeddings_cache[uncached_words[i]] = np.array(embedding)
                    cached_embeddings.append((word_idx, np.array(embedding)))
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}, falling back to individual")
                for word_idx, word in zip(uncached_indices, uncached_words):
                    embedding = self._get_real_embedding(word)
                    cached_embeddings.append((word_idx, embedding))
        
        # Sort by original order and return
        cached_embeddings.sort(key=lambda x: x[0])
        return [embedding for _, embedding in cached_embeddings]
    
    def get_semantic_similarity(self, word1: str, word2: str, 
                               lang1: str = None, lang2: str = None) -> float:
        """Calculate semantic similarity between two words."""
        embedding1 = self.get_embedding(word1, lang1)
        embedding2 = self.get_embedding(word2, lang2)
        
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return float(similarity)


class HybridSemanticAligner:
    """
    Combines modern lexicon and embedding approaches for best results.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from .modern_semantic_aligner import ModernSemanticAligner
        
        self.lexicon_aligner = ModernSemanticAligner()
        self.embedding_aligner = CrossLingualEmbeddingAligner(embedding_model)
        
    def align_verse(self, source_words: List[str], target_words: List[str],
                   source_lang: str = 'hebrew', target_lang: str = 'english', **kwargs) -> List[Dict]:
        """
        Hybrid alignment combining lexicon and embeddings.
        """
        logger.debug(f"Hybrid semantic alignment: {len(source_words)} {source_lang} → {len(target_words)} {target_lang}")
        
        # Get alignments from both methods
        lexicon_alignments = self.lexicon_aligner.align_verse(
            source_words, target_words, source_lang, target_lang, **kwargs
        )
        
        embedding_alignments = self.embedding_aligner.align_verse(
            source_words, target_words, source_lang, target_lang, **kwargs
        )
        
        # Combine and deduplicate
        combined_alignments = self._combine_alignments(lexicon_alignments, embedding_alignments)
        
        logger.debug(f"Hybrid alignment found {len(combined_alignments)} combined alignments")
        return combined_alignments
    
    def _combine_alignments(self, lexicon_alignments: List[Dict], 
                           embedding_alignments: List[Dict]) -> List[Dict]:
        """Combine alignments from different methods."""
        alignment_map = {}
        
        # Add lexicon alignments (higher priority)
        for align in lexicon_alignments:
            key = (align['source_index'], align['target_index'])
            alignment_map[key] = align
            alignment_map[key]['contributing_methods'] = ['modern_lexicon']
        
        # Add embedding alignments
        for align in embedding_alignments:
            key = (align['source_index'], align['target_index'])
            
            if key in alignment_map:
                # Combine confidences
                existing = alignment_map[key]
                combined_confidence = (existing['confidence'] + align['confidence']) / 2
                existing['confidence'] = round(combined_confidence * 1.1, 3)  # Boost for agreement
                existing['contributing_methods'].append('cross_lingual_embedding')
                existing['method'] = 'hybrid_semantic'
            else:
                # New alignment
                align['contributing_methods'] = ['cross_lingual_embedding']
                alignment_map[key] = align
        
        return list(alignment_map.values())