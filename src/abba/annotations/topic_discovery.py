"""
BERTopic implementation for automatic topic discovery in biblical texts.

Provides unsupervised topic modeling to discover themes and patterns
not explicitly defined in the taxonomy.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import pandas as pd


@dataclass
class DiscoveredTopic:
    """Represents a topic discovered through unsupervised learning."""

    topic_id: int
    words: List[str]
    word_scores: List[float]
    size: int  # Number of documents
    representative_docs: List[str]
    coherence_score: float
    embedding_center: Optional[np.ndarray] = None


class BERTopicDiscovery:
    """
    Discovers topics in biblical texts using BERTopic methodology.

    Uses sentence embeddings, dimensionality reduction, clustering,
    and c-TF-IDF to find coherent topics.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_topic_size: int = 10,
        n_gram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
    ):
        """
        Initialize the topic discovery system.

        Args:
            embedding_model: Sentence transformer model name
            min_topic_size: Minimum cluster size for HDBSCAN
            n_gram_range: N-gram range for topic representation
            min_df: Minimum document frequency for terms
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.min_topic_size = min_topic_size
        self.n_gram_range = n_gram_range
        self.min_df = min_df

        # Components
        self.umap_model = None
        self.cluster_model = None
        self.vectorizer = None

        # Results
        self.topics = {}
        self.topic_embeddings = {}
        self.document_topics = []

    def fit_transform(
        self, documents: List[str], embeddings: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Discover topics in the documents.

        Args:
            documents: List of text documents
            embeddings: Pre-computed embeddings (optional)

        Returns:
            List of topic IDs for each document
        """
        # Step 1: Create embeddings if not provided
        if embeddings is None:
            print("Creating document embeddings...")
            embeddings = self._create_embeddings(documents)

        # Step 2: Reduce dimensionality
        print("Reducing dimensionality...")
        reduced_embeddings = self._reduce_dimensionality(embeddings)

        # Step 3: Cluster documents
        print("Clustering documents...")
        clusters = self._cluster_documents(reduced_embeddings)

        # Step 4: Create topic representations
        print("Creating topic representations...")
        self._create_topics(documents, clusters)

        # Store results
        self.document_topics = clusters

        return clusters

    def _create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Create sentence embeddings for documents."""
        # Preprocess for biblical text
        processed_docs = [self._preprocess_biblical_text(doc) for doc in documents]

        # Create embeddings
        embeddings = self.embedding_model.encode(
            processed_docs, show_progress_bar=True, batch_size=32
        )

        return embeddings

    def _preprocess_biblical_text(self, text: str) -> str:
        """Preprocess biblical text for better embeddings."""
        # Remove verse numbers
        import re

        processed = re.sub(r"\b\d+:\d+\b", "", text)

        # Normalize archaic language
        replacements = {
            "thou": "you",
            "thee": "you",
            "thy": "your",
            "thine": "yours",
            "ye": "you",
            "hath": "has",
            "doth": "does",
            "saith": "says",
        }

        for old, new in replacements.items():
            processed = re.sub(r"\b" + old + r"\b", new, processed, flags=re.IGNORECASE)

        return processed.strip()

    def _reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensionality using UMAP."""
        self.umap_model = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
        )

        return self.umap_model.fit_transform(embeddings)

    def _cluster_documents(self, embeddings: np.ndarray) -> List[int]:
        """Cluster documents using HDBSCAN."""
        self.cluster_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        clusters = self.cluster_model.fit_predict(embeddings)

        return clusters

    def _create_topics(self, documents: List[str], clusters: List[int]):
        """Create topic representations using c-TF-IDF."""
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            ngram_range=self.n_gram_range,
            stop_words="english",
            min_df=self.min_df,
            max_features=10000,
        )

        # Fit vectorizer on all documents
        doc_term_matrix = self.vectorizer.fit_transform(documents)

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Calculate c-TF-IDF per topic
        unique_topics = set(clusters)

        for topic_id in unique_topics:
            if topic_id == -1:  # Skip noise cluster
                continue

            # Get documents in this topic
            topic_docs = [doc for doc, cluster in zip(documents, clusters) if cluster == topic_id]

            # Calculate c-TF-IDF
            topic_words, word_scores = self._c_tf_idf(
                doc_term_matrix, clusters, topic_id, feature_names
            )

            # Calculate coherence
            coherence = self._calculate_coherence(topic_words[:10], documents)

            # Create topic object
            topic = DiscoveredTopic(
                topic_id=topic_id,
                words=topic_words[:30],  # Top 30 words
                word_scores=word_scores[:30],
                size=len(topic_docs),
                representative_docs=topic_docs[:5],  # Top 5 examples
                coherence_score=coherence,
            )

            self.topics[topic_id] = topic

    def _c_tf_idf(
        self, doc_term_matrix: Any, clusters: List[int], topic_id: int, feature_names: np.ndarray
    ) -> Tuple[List[str], List[float]]:
        """
        Calculate class-based TF-IDF for topic representation.

        This gives high scores to words that are frequent in a topic
        but rare in other topics.
        """
        # Get document indices for this topic
        topic_indices = [i for i, c in enumerate(clusters) if c == topic_id]

        # Calculate term frequency for the topic
        topic_term_freq = np.asarray(doc_term_matrix[topic_indices].sum(axis=0)).squeeze()

        # Calculate document frequency across all documents
        doc_freq = np.asarray((doc_term_matrix > 0).sum(axis=0)).squeeze()

        # Calculate IDF
        n_docs = doc_term_matrix.shape[0]
        idf = np.log(n_docs / (doc_freq + 1))

        # Calculate c-TF-IDF
        c_tf_idf = topic_term_freq * idf

        # Get top words
        top_indices = c_tf_idf.argsort()[::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [c_tf_idf[i] for i in top_indices]

        return top_words, top_scores

    def _calculate_coherence(self, topic_words: List[str], documents: List[str]) -> float:
        """
        Calculate topic coherence using pointwise mutual information.

        Higher coherence indicates more semantically related words.
        """
        # Simple coherence: co-occurrence based
        word_doc_freq = {}
        pair_doc_freq = {}

        for doc in documents:
            doc_lower = doc.lower()

            # Count single word occurrences
            for word in topic_words:
                if word in doc_lower:
                    word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

            # Count pair co-occurrences
            for i in range(len(topic_words)):
                for j in range(i + 1, len(topic_words)):
                    if topic_words[i] in doc_lower and topic_words[j] in doc_lower:
                        pair = (topic_words[i], topic_words[j])
                        pair_doc_freq[pair] = pair_doc_freq.get(pair, 0) + 1

        # Calculate PMI
        n_docs = len(documents)
        coherence_scores = []

        for i in range(len(topic_words)):
            for j in range(i + 1, len(topic_words)):
                w1, w2 = topic_words[i], topic_words[j]

                p_w1 = word_doc_freq.get(w1, 0) / n_docs
                p_w2 = word_doc_freq.get(w2, 0) / n_docs
                p_w1_w2 = pair_doc_freq.get((w1, w2), 0) / n_docs

                if p_w1 > 0 and p_w2 > 0 and p_w1_w2 > 0:
                    pmi = np.log(p_w1_w2 / (p_w1 * p_w2))
                    coherence_scores.append(pmi)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def get_topic_info(self) -> pd.DataFrame:
        """Get information about all discovered topics."""
        topic_data = []

        for topic_id, topic in self.topics.items():
            topic_data.append(
                {
                    "Topic": topic_id,
                    "Size": topic.size,
                    "Top_Words": ", ".join(topic.words[:10]),
                    "Coherence": topic.coherence_score,
                    "Representative_Doc": (
                        topic.representative_docs[0][:100] + "..."
                        if topic.representative_docs
                        else ""
                    ),
                }
            )

        return pd.DataFrame(topic_data).sort_values("Size", ascending=False)

    def get_topic_words(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        """Get top words for a specific topic."""
        if topic_id not in self.topics:
            return []

        topic = self.topics[topic_id]
        return list(zip(topic.words[:n_words], topic.word_scores[:n_words]))

    def find_topics(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Find topics most similar to a query.

        Args:
            query: Query text
            top_n: Number of topics to return

        Returns:
            List of (topic_id, similarity) tuples
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate similarities to topic embeddings
        similarities = []

        for topic_id, topic in self.topics.items():
            if topic.embedding_center is not None:
                sim = np.dot(query_embedding, topic.embedding_center) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(topic.embedding_center)
                )
                similarities.append((topic_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]

    def merge_similar_topics(self, similarity_threshold: float = 0.8):
        """Merge topics that are very similar."""
        # Calculate pairwise similarities between topics
        topic_ids = list(self.topics.keys())
        merge_groups = []
        merged = set()

        for i, tid1 in enumerate(topic_ids):
            if tid1 in merged:
                continue

            group = [tid1]

            for j, tid2 in enumerate(topic_ids[i + 1 :], i + 1):
                if tid2 in merged:
                    continue

                # Compare top words
                words1 = set(self.topics[tid1].words[:20])
                words2 = set(self.topics[tid2].words[:20])

                jaccard = len(words1 & words2) / len(words1 | words2)

                if jaccard >= similarity_threshold:
                    group.append(tid2)
                    merged.add(tid2)

            if len(group) > 1:
                merge_groups.append(group)
                merged.add(tid1)

        # Perform merges
        for group in merge_groups:
            self._merge_topics(group)

    def _merge_topics(self, topic_ids: List[int]):
        """Merge multiple topics into one."""
        # Use the largest topic as the base
        topic_ids.sort(key=lambda x: self.topics[x].size, reverse=True)
        base_id = topic_ids[0]

        # Merge other topics into base
        for tid in topic_ids[1:]:
            if tid in self.topics:
                # Update document assignments
                for i, topic in enumerate(self.document_topics):
                    if topic == tid:
                        self.document_topics[i] = base_id

                # Remove merged topic
                del self.topics[tid]

        # Recalculate base topic representation
        # (Would need to recalculate c-TF-IDF here)

    def visualize_topics(self) -> Dict[str, Any]:
        """
        Create visualization data for topics.

        Returns:
            Dictionary with visualization data
        """
        # Topic sizes
        sizes = [topic.size for topic in self.topics.values()]

        # Topic words (for word clouds)
        word_clouds = {}
        for tid, topic in self.topics.items():
            word_clouds[tid] = dict(zip(topic.words[:20], topic.word_scores[:20]))

        # Inter-topic distances (simplified)
        # Would calculate actual distances between topic centers

        return {
            "topic_sizes": sizes,
            "word_clouds": word_clouds,
            "n_topics": len(self.topics),
            "n_outliers": sum(1 for t in self.document_topics if t == -1),
        }
