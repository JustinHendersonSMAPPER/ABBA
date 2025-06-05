"""
Biblical BERT adapter for domain-specific text understanding.

Provides a wrapper around BERT models with biblical text preprocessing
and domain adaptation capabilities.
"""

import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline


@dataclass
class BERTEmbedding:
    """Container for BERT embeddings with metadata."""

    text: str
    embedding: np.ndarray
    tokens: List[str]
    attention_mask: Optional[np.ndarray] = None
    special_tokens_mask: Optional[np.ndarray] = None


class BiblicalBERTAdapter:
    """
    Adapter for using BERT models with biblical text.

    Handles domain-specific preprocessing and provides various
    text understanding capabilities.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        """
        Initialize the BERT adapter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Biblical text preprocessing patterns
        self.biblical_terms = self._load_biblical_vocabulary()

    def _load_biblical_vocabulary(self) -> Dict[str, List[str]]:
        """Load biblical-specific vocabulary mappings."""
        return {
            # Old English to modern mappings
            "thee": ["you"],
            "thou": ["you"],
            "thy": ["your"],
            "thine": ["yours"],
            "ye": ["you"],
            "hath": ["has"],
            "doth": ["does"],
            "saith": ["says"],
            # Biblical names variations
            "jesus": ["jesus", "christ", "messiah", "lord"],
            "god": ["god", "lord", "yahweh", "jehovah", "father"],
            "holy spirit": ["holy spirit", "spirit", "comforter", "helper"],
            # Theological terms
            "salvation": ["salvation", "saved", "redemption"],
            "sin": ["sin", "transgression", "iniquity"],
            "grace": ["grace", "favor", "mercy"],
        }

    def preprocess_biblical_text(self, text: str) -> str:
        """
        Preprocess biblical text for better model understanding.

        Args:
            text: Raw biblical text

        Returns:
            Preprocessed text
        """
        # Lowercase for processing
        processed = text.lower()

        # Normalize verse references
        import re

        processed = re.sub(r"(\d+):(\d+)", r"verse \1 \2", processed)

        # Replace archaic words with modern equivalents
        archaic_replacements = {
            "thou": "you",
            "thee": "you",
            "thy": "your",
            "thine": "your",
            "ye": "you",
            "hath": "has",
            "doth": "does",
            "shalt": "shall",
            "art": "are",
            "wilt": "will",
            "saith": "says",
            "sayeth": "says",
        }
        for archaic, modern in archaic_replacements.items():
            # Use word boundaries to avoid partial replacements
            processed = re.sub(r'\b' + archaic + r'\b', modern, processed)

        # Expand contractions
        contractions = {
            "n't": " not",
            "'s": " is",
            "'ll": " will",
            "'ve": " have",
            "'re": " are",
            "'d": " would",
        }
        for contraction, expansion in contractions.items():
            processed = processed.replace(contraction, expansion)

        return processed

    def get_embeddings(
        self, texts: List[str], batch_size: int = 32, layer: int = -1
    ) -> List[BERTEmbedding]:
        """
        Get BERT embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            layer: Which layer to extract embeddings from (-1 for last)

        Returns:
            List of BERTEmbedding objects
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Preprocess texts
            processed_texts = [self.preprocess_biblical_text(t) for t in batch_texts]

            # Tokenize
            encoded = self.tokenizer(
                processed_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
                )

                # Extract embeddings from specified layer
                hidden_states = outputs.hidden_states[layer]

                # Mean pooling (accounting for attention mask)
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

            # Convert to numpy
            batch_embeddings = mean_embeddings.cpu().numpy()

            # Create embedding objects
            for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[j])
                embeddings.append(
                    BERTEmbedding(
                        text=texts[i + j],  # Original text
                        embedding=embedding,
                        tokens=tokens,
                        attention_mask=attention_mask[j].cpu().numpy(),
                    )
                )

        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        embeddings = self.get_embeddings([text1, text2])

        # Cosine similarity
        emb1 = embeddings[0].embedding
        emb2 = embeddings[1].embedding

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Convert to 0-1 range
        return (similarity + 1) / 2

    def find_key_phrases(self, text: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
        """
        Extract key phrases from text using attention patterns.

        Args:
            text: Input text
            max_phrases: Maximum number of phrases to return

        Returns:
            List of (phrase, importance_score) tuples
        """
        # Preprocess
        processed = self.preprocess_biblical_text(text)

        # Tokenize
        encoded = self.tokenizer(processed, return_tensors="pt", truncation=True, max_length=512)

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
            )

            # Average attention across all layers and heads
            attentions = outputs.attentions
            avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))

        # Convert tokens to words
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Group subword tokens
        word_scores = []
        current_word = ""
        current_score = 0.0

        for token, score in zip(tokens[1:-1], avg_attention[1:-1]):  # Skip [CLS] and [SEP]
            if token.startswith("##"):
                current_word += token[2:]
                current_score += score.item()
            else:
                if current_word:
                    word_scores.append((current_word, current_score))
                current_word = token
                current_score = score.item()

        if current_word:
            word_scores.append((current_word, current_score))

        # Sort by importance
        word_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract phrases (simple n-gram approach)
        phrases = []
        words = text.split()

        for n in [3, 2, 1]:  # Try trigrams, bigrams, then unigrams
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                # Calculate phrase score
                phrase_score = sum(
                    score for word, score in word_scores if word.lower() in phrase.lower()
                )
                if phrase_score > 0:
                    phrases.append((phrase, phrase_score))

        # Remove duplicates and sort
        seen = set()
        unique_phrases = []
        for phrase, score in sorted(phrases, key=lambda x: x[1], reverse=True):
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                unique_phrases.append((phrase, score))

        return unique_phrases[:max_phrases]

    def adapt_to_biblical_domain(
        self, biblical_texts: List[str], labels: Optional[List[int]] = None, num_epochs: int = 3
    ):
        """
        Fine-tune the model on biblical texts (if labels provided).

        Args:
            biblical_texts: List of biblical texts
            labels: Optional labels for supervised fine-tuning
            num_epochs: Number of training epochs
        """
        if labels is None:
            # Unsupervised domain adaptation using masked language modeling
            print("Performing unsupervised domain adaptation...")
            # This would implement MLM fine-tuning
            # For now, this is a placeholder
            pass
        else:
            # Supervised fine-tuning
            print("Performing supervised fine-tuning...")
            # This would implement classification fine-tuning
            # For now, this is a placeholder
            pass

    def create_topic_classifier(
        self, topics: List[str], examples_per_topic: Dict[str, List[str]]
    ) -> Any:
        """
        Create a topic classifier for the given topics.

        Args:
            topics: List of topic names
            examples_per_topic: Example texts for each topic

        Returns:
            Classifier pipeline
        """
        # For now, return a zero-shot classifier
        # In practice, this could fine-tune on the examples
        classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
        )

        return classifier

    def extract_theological_concepts(
        self, text: str, concept_embeddings: Dict[str, np.ndarray], threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Extract theological concepts from text using pre-computed embeddings.

        Args:
            text: Input text
            concept_embeddings: Pre-computed embeddings for concepts
            threshold: Minimum similarity threshold

        Returns:
            List of (concept, similarity) tuples
        """
        # Get text embedding
        text_embedding = self.get_embeddings([text])[0].embedding

        # Compare with concept embeddings
        matches = []

        for concept, concept_emb in concept_embeddings.items():
            similarity = np.dot(text_embedding, concept_emb) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(concept_emb)
            )

            # Convert to 0-1 range
            similarity = (similarity + 1) / 2

            if similarity >= threshold:
                matches.append((concept, similarity))

        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches
