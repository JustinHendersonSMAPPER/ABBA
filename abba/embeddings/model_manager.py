"""Sentence Transformer model management for biblical text embeddings."""

from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingModelManager:
    """Manages Sentence Transformer models for embedding generation."""
    
    def __init__(self, cache_dir: str = "bible_data/models"):
        """Initialize model manager with cache directory.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "english": {
                "name": "intfloat/e5-large-v2",
                "dimensions": 1024,
                "max_length": 512,
                "instruction": "passage: ",  # E5 models require this prefix
                "description": "Best for English biblical text"
            },
            "multilingual": {
                "name": "intfloat/multilingual-e5-base",
                "dimensions": 768,
                "max_length": 512,
                "instruction": "passage: ",
                "description": "Best for Hebrew, Greek, and other languages"
            }
        }
        
        # Lazy-loaded models
        self._models: Dict[str, SentenceTransformer] = {}
        
        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("Using CPU for embeddings (GPU not available)")
    
    def get_model(self, model_type: str = "english") -> SentenceTransformer:
        """Get or load a model.
        
        Args:
            model_type: Type of model ("english" or "multilingual")
            
        Returns:
            SentenceTransformer model instance
        """
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(self.model_configs.keys())}")
        
        # Load model if not cached
        if model_type not in self._models:
            config = self.model_configs[model_type]
            logger.info(f"Loading {model_type} model: {config['name']}")
            
            self._models[model_type] = SentenceTransformer(
                config["name"],
                cache_folder=str(self.cache_dir),
                device=self.device
            )
            
            # Set max sequence length
            self._models[model_type].max_seq_length = config["max_length"]
            
            logger.info(f"Loaded {model_type} model with {config['dimensions']} dimensions")
        
        return self._models[model_type]
    
    def encode_texts(
        self,
        texts: List[str],
        model_type: str = "english",
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            model_type: Model to use for encoding
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings (for cosine similarity)
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        model = self.get_model(model_type)
        config = self.model_configs[model_type]
        
        # Add instruction prefix for E5 models
        if config.get("instruction"):
            prefixed_texts = [config["instruction"] + text for text in texts]
        else:
            prefixed_texts = texts
        
        # Log encoding details
        logger.info(f"Encoding {len(texts)} texts with {model_type} model")
        
        # Generate embeddings
        embeddings = model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            device=self.device,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(
        self,
        text: str,
        model_type: str = "english",
        normalize: bool = True
    ) -> np.ndarray:
        """Encode a single text.
        
        Args:
            text: Text to encode
            model_type: Model to use
            normalize: Whether to normalize embedding
            
        Returns:
            Embedding vector
        """
        embeddings = self.encode_texts(
            [text],
            model_type=model_type,
            batch_size=1,
            show_progress=False,
            normalize=normalize
        )
        return embeddings[0]
    
    def encode_with_context(
        self,
        texts: List[str],
        contexts: List[Dict[str, str]],
        model_type: str = "english",
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Encode texts with additional context information.
        
        Args:
            texts: Primary texts to encode
            contexts: List of context dictionaries for each text
            model_type: Model to use
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings array
        """
        if len(texts) != len(contexts):
            raise ValueError("Number of texts must match number of contexts")
        
        # Combine texts with contexts
        enriched_texts = []
        for text, context in zip(texts, contexts):
            parts = [text]
            
            # Add context elements in a structured way
            if context.get("original"):
                parts.append(f"Original: {context['original']}")
            if context.get("keywords"):
                parts.append(f"Keywords: {context['keywords']}")
            if context.get("grammar"):
                parts.append(f"Grammar: {context['grammar']}")
            if context.get("reference"):
                parts.insert(0, context["reference"])
            
            enriched_text = " | ".join(parts)
            enriched_texts.append(enriched_text)
        
        return self.encode_texts(
            enriched_texts,
            model_type=model_type,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    def select_model_for_text(self, text: str) -> str:
        """Automatically select best model based on text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Model type to use ("english" or "multilingual")
        """
        # Simple heuristic: check for non-Latin scripts
        # Hebrew, Greek, Arabic, etc. have Unicode ranges outside Latin
        non_latin_chars = 0
        total_chars = len(text)
        
        for char in text:
            code_point = ord(char)
            # Check for Hebrew, Greek, Arabic, etc.
            if (0x0590 <= code_point <= 0x05FF or  # Hebrew
                0x0600 <= code_point <= 0x06FF or  # Arabic
                0x0370 <= code_point <= 0x03FF or  # Greek
                0x1F00 <= code_point <= 0x1FFF):   # Greek Extended
                non_latin_chars += 1
        
        # If more than 20% non-Latin, use multilingual
        if total_chars > 0 and non_latin_chars / total_chars > 0.2:
            return "multilingual"
        
        return "english"
    
    def get_model_info(self, model_type: str = "english") -> Dict[str, any]:
        """Get information about a model.
        
        Args:
            model_type: Model type to get info for
            
        Returns:
            Dictionary with model information
        """
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.model_configs[model_type].copy()
        
        # Add runtime info if model is loaded
        if model_type in self._models:
            config["loaded"] = True
            config["device"] = str(self.device)
        else:
            config["loaded"] = False
        
        return config
    
    def preload_models(self, model_types: Optional[List[str]] = None):
        """Preload models into memory.
        
        Args:
            model_types: List of model types to preload, or None for all
        """
        if model_types is None:
            model_types = list(self.model_configs.keys())
        
        for model_type in model_types:
            logger.info(f"Preloading {model_type} model...")
            self.get_model(model_type)
        
        logger.info("All models preloaded")
    
    def clear_cache(self):
        """Clear loaded models from memory."""
        self._models.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")