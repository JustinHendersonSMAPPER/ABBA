"""
Multi-level annotation engine that combines all ML approaches.

Orchestrates BERT, BERTopic, SetFit, and zero-shot classification
to provide comprehensive biblical text annotation.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .models import (
    Annotation,
    AnnotationType,
    AnnotationLevel,
    Topic,
    AnnotationConfidence,
    AnnotationCollection,
)
from .taxonomy import TheologicalTaxonomy
from .bert_adapter import BiblicalBERTAdapter
from .topic_discovery import BERTopicDiscovery
from .few_shot_classifier import SetFitClassifier, FewShotExample
from .zero_shot_classifier import ZeroShotTheologyClassifier
from ..verse_id import VerseID, parse_verse_id


@dataclass
class AnnotationRequest:
    """Request for text annotation."""

    text: str
    verse_id: VerseID
    level: AnnotationLevel = AnnotationLevel.VERSE
    context: Optional[Dict[str, Any]] = None
    methods: List[str] = None  # Which methods to use

    def __post_init__(self):
        if self.methods is None:
            self.methods = ["zero_shot", "bert_similarity", "few_shot"]


@dataclass
class AnnotationResult:
    """Result from annotation engine."""

    annotations: List[Annotation]
    discovered_topics: List[Any]  # From BERTopic
    key_phrases: List[Tuple[str, float]]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]


class AnnotationEngine:
    """
    Main annotation engine that orchestrates multiple ML approaches.

    Combines:
    - Zero-shot classification for known theological concepts
    - BERT embeddings for semantic similarity
    - BERTopic for discovering new themes
    - SetFit for few-shot learning on specific domains
    """

    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        zero_shot_model: str = "facebook/bart-large-mnli",
        sentence_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the annotation engine.

        Args:
            bert_model: BERT model for embeddings
            zero_shot_model: Model for zero-shot classification
            sentence_model: Sentence transformer model
        """
        # Initialize components
        print("Initializing annotation engine components...")

        self.taxonomy = TheologicalTaxonomy()
        self.bert_adapter = BiblicalBERTAdapter(bert_model)
        self.zero_shot = ZeroShotTheologyClassifier(zero_shot_model)
        self.topic_discovery = BERTopicDiscovery(sentence_model)
        self.setfit = SetFitClassifier(sentence_model)

        # Cache for efficiency
        self.embedding_cache = {}
        self.annotation_cache = {}

        # Configuration
        self.config = {
            "min_confidence": 0.5,
            "max_annotations_per_verse": 5,
            "enable_topic_discovery": True,
            "enable_few_shot": True,
            "ensemble_weights": {"zero_shot": 0.4, "bert_similarity": 0.3, "few_shot": 0.3},
        }

        # Few-shot training examples (would be loaded from data)
        self._initialize_few_shot_examples()

    def _initialize_few_shot_examples(self):
        """Initialize few-shot classifier with biblical examples."""
        # Sample training examples for major categories
        examples = [
            # Salvation examples
            FewShotExample(
                text="For by grace you have been saved through faith", label="salvation"
            ),
            FewShotExample(
                text="Believe in the Lord Jesus, and you will be saved", label="salvation"
            ),
            # Trinity examples
            FewShotExample(
                text="In the name of the Father and of the Son and of the Holy Spirit",
                label="trinity",
            ),
            FewShotExample(
                text="The grace of the Lord Jesus Christ and the love of God and the fellowship of the Holy Spirit",
                label="trinity",
            ),
            # Prayer examples
            FewShotExample(text="Our Father in heaven, hallowed be your name", label="prayer"),
            FewShotExample(
                text="Do not be anxious about anything, but in everything by prayer", label="prayer"
            ),
            # Sin examples
            FewShotExample(
                text="For all have sinned and fall short of the glory of God", label="sin"
            ),
            FewShotExample(
                text="If we confess our sins, he is faithful and just to forgive", label="sin"
            ),
            # Faith examples
            FewShotExample(text="Now faith is the assurance of things hoped for", label="faith"),
            FewShotExample(text="The righteous shall live by faith", label="faith"),
        ]

        # Train SetFit classifier
        if examples:
            print("Training few-shot classifier...")
            self.setfit.train(examples)

    def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """
        Annotate text using multiple ML approaches.

        Args:
            request: Annotation request

        Returns:
            AnnotationResult with all annotations
        """
        # Get cached result if available
        cache_key = f"{request.verse_id}:{request.text[:50]}"
        if cache_key in self.annotation_cache:
            return self.annotation_cache[cache_key]

        annotations = []
        confidence_scores = {}

        # 1. Zero-shot classification
        if "zero_shot" in request.methods:
            zs_annotations = self._zero_shot_annotate(request)
            annotations.extend(zs_annotations)
            confidence_scores["zero_shot"] = (
                np.mean([a.confidence.overall_score for a in zs_annotations if a.confidence])
                if zs_annotations
                else 0.0
            )

        # 2. BERT similarity-based annotation
        if "bert_similarity" in request.methods:
            bert_annotations = self._bert_similarity_annotate(request)
            annotations.extend(bert_annotations)
            confidence_scores["bert_similarity"] = (
                np.mean([a.confidence.overall_score for a in bert_annotations if a.confidence])
                if bert_annotations
                else 0.0
            )

        # 3. Few-shot classification
        if "few_shot" in request.methods and self.setfit.is_trained:
            fs_annotations = self._few_shot_annotate(request)
            annotations.extend(fs_annotations)
            confidence_scores["few_shot"] = (
                np.mean([a.confidence.overall_score for a in fs_annotations if a.confidence])
                if fs_annotations
                else 0.0
            )

        # 4. Extract key phrases
        key_phrases = self.bert_adapter.find_key_phrases(request.text)

        # 5. Topic discovery (for collections of verses)
        discovered_topics = []
        if self.config["enable_topic_discovery"] and request.level != AnnotationLevel.WORD:
            # Would run topic discovery on a collection of texts
            pass

        # 6. Merge and rank annotations
        merged_annotations = self._merge_annotations(annotations)

        # 7. Apply confidence threshold
        filtered_annotations = [
            a
            for a in merged_annotations
            if a.confidence and a.confidence.overall_score >= self.config["min_confidence"]
        ]

        # 8. Limit number of annotations
        filtered_annotations = filtered_annotations[: self.config["max_annotations_per_verse"]]

        # Create result
        result = AnnotationResult(
            annotations=filtered_annotations,
            discovered_topics=discovered_topics,
            key_phrases=key_phrases,
            confidence_scores=confidence_scores,
            metadata={
                "methods_used": request.methods,
                "total_candidates": len(annotations),
                "filtered_count": len(filtered_annotations),
            },
        )

        # Cache result
        self.annotation_cache[cache_key] = result

        return result

    def _zero_shot_annotate(self, request: AnnotationRequest) -> List[Annotation]:
        """Annotate using zero-shot classification."""
        annotations = []

        # Get predictions
        prediction = self.zero_shot.classify_with_similarity(
            request.text, multi_label=True, threshold=0.6
        )

        # Create annotations for top predictions
        for label, score in zip(prediction.labels, prediction.scores):
            if label in self.taxonomy.taxonomy.topics:
                topic = self.taxonomy.taxonomy.topics[label]

                annotation = Annotation(
                    id=f"zs_{request.verse_id}_{label}",
                    annotation_type=AnnotationType.THEOLOGICAL_THEME,
                    level=request.level,
                    start_verse=request.verse_id,
                    topic_id=topic.id,
                    topic_name=topic.name,
                    content=topic.description,
                    confidence=AnnotationConfidence(
                        overall_score=score,
                        model_confidence=score,
                        contextual_relevance=0.8,  # Would calculate
                        semantic_similarity=0.85,
                        supporting_phrases=[],
                    ),
                    source="zero_shot",
                    tags=["automatic", "zero_shot"],
                )

                annotations.append(annotation)

        return annotations

    def _bert_similarity_annotate(self, request: AnnotationRequest) -> List[Annotation]:
        """Annotate using BERT embeddings and similarity."""
        annotations = []

        # Get text embedding
        text_embedding = self._get_cached_embedding(request.text)

        # Compare with topic embeddings
        topic_similarities = []

        for topic_id, topic in self.taxonomy.taxonomy.topics.items():
            # Get topic embedding (from description)
            topic_text = f"{topic.name}: {topic.description}"
            topic_embedding = self._get_cached_embedding(topic_text)

            # Calculate similarity
            similarity = np.dot(text_embedding, topic_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(topic_embedding)
            )

            # Convert to 0-1 range
            similarity = (similarity + 1) / 2

            if similarity >= 0.7:  # Threshold
                topic_similarities.append((topic, similarity))

        # Sort by similarity
        topic_similarities.sort(key=lambda x: x[1], reverse=True)

        # Create annotations for top matches
        for topic, similarity in topic_similarities[:3]:
            annotation = Annotation(
                id=f"bert_{request.verse_id}_{topic.id}",
                annotation_type=AnnotationType.THEOLOGICAL_THEME,
                level=request.level,
                start_verse=request.verse_id,
                topic_id=topic.id,
                topic_name=topic.name,
                content=topic.description,
                confidence=AnnotationConfidence(
                    overall_score=similarity,
                    model_confidence=similarity,
                    contextual_relevance=0.7,
                    semantic_similarity=similarity,
                    supporting_phrases=[],
                ),
                source="bert_similarity",
                tags=["automatic", "bert"],
            )

            annotations.append(annotation)

        return annotations

    def _few_shot_annotate(self, request: AnnotationRequest) -> List[Annotation]:
        """Annotate using few-shot classifier."""
        annotations = []

        # Get predictions
        prediction = self.setfit.predict(request.text, return_all_scores=True)

        # Create annotations for confident predictions
        for label, score in prediction.all_scores.items():
            if score >= 0.6 and label in self.taxonomy.taxonomy.topics:
                topic = self.taxonomy.taxonomy.topics[label]

                annotation = Annotation(
                    id=f"fs_{request.verse_id}_{label}",
                    annotation_type=AnnotationType.THEOLOGICAL_THEME,
                    level=request.level,
                    start_verse=request.verse_id,
                    topic_id=topic.id,
                    topic_name=topic.name,
                    content=topic.description,
                    confidence=AnnotationConfidence(
                        overall_score=score,
                        model_confidence=score,
                        contextual_relevance=0.75,
                        semantic_similarity=0.8,
                        supporting_phrases=[],
                    ),
                    source="few_shot",
                    tags=["automatic", "few_shot"],
                )

                annotations.append(annotation)

        return annotations

    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get cached or compute embedding for text."""
        if text not in self.embedding_cache:
            embeddings = self.bert_adapter.get_embeddings([text])
            self.embedding_cache[text] = embeddings[0].embedding

        return self.embedding_cache[text]

    def _merge_annotations(self, annotations: List[Annotation]) -> List[Annotation]:
        """Merge annotations from different methods."""
        # Group by topic
        topic_groups = defaultdict(list)

        for ann in annotations:
            if ann.topic_id:
                topic_groups[ann.topic_id].append(ann)

        # Merge annotations for same topic
        merged = []

        for topic_id, group in topic_groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Ensemble the confidence scores
                ensemble_score = self._ensemble_confidence(group)

                # Use the annotation with highest individual score as base
                best_ann = max(
                    group, key=lambda a: a.confidence.overall_score if a.confidence else 0
                )

                # Update confidence with ensemble score
                best_ann.confidence.overall_score = ensemble_score
                best_ann.source = "ensemble"
                best_ann.tags.append("ensemble")

                merged.append(best_ann)

        # Sort by confidence
        merged.sort(key=lambda a: a.confidence.overall_score if a.confidence else 0, reverse=True)

        return merged

    def _ensemble_confidence(self, annotations: List[Annotation]) -> float:
        """Ensemble confidence scores from multiple methods."""
        method_scores = {}

        for ann in annotations:
            if ann.confidence:
                method = ann.source
                score = ann.confidence.overall_score
                method_scores[method] = score

        # Weighted average based on method
        total_score = 0.0
        total_weight = 0.0

        for method, score in method_scores.items():
            weight = self.config["ensemble_weights"].get(method, 0.25)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def annotate_collection(
        self, texts: List[str], verse_ids: List[VerseID], discover_topics: bool = True
    ) -> AnnotationCollection:
        """
        Annotate a collection of texts and discover topics.

        Args:
            texts: List of texts to annotate
            verse_ids: Corresponding verse IDs
            discover_topics: Whether to run topic discovery

        Returns:
            AnnotationCollection with all results
        """
        all_annotations = []

        # Annotate individual texts
        for text, verse_id in zip(texts, verse_ids):
            request = AnnotationRequest(text=text, verse_id=verse_id, level=AnnotationLevel.VERSE)

            result = self.annotate(request)
            all_annotations.extend(result.annotations)

        # Topic discovery on the collection
        if discover_topics and len(texts) >= 10:
            print("Discovering topics in collection...")

            # Run BERTopic
            topics = self.topic_discovery.fit_transform(texts)

            # Create annotations for discovered topics
            topic_info = self.topic_discovery.get_topic_info()

            for idx, (text, verse_id, topic_id) in enumerate(zip(texts, verse_ids, topics)):
                if topic_id != -1:  # Not noise
                    topic_words = self.topic_discovery.get_topic_words(topic_id, n_words=5)
                    topic_desc = ", ".join([word for word, _ in topic_words])

                    annotation = Annotation(
                        id=f"discovered_{verse_id}_{topic_id}",
                        annotation_type=AnnotationType.THEOLOGICAL_THEME,
                        level=AnnotationLevel.VERSE,
                        start_verse=verse_id,
                        topic_name=f"Discovered Topic {topic_id}",
                        content=f"Automatically discovered topic: {topic_desc}",
                        confidence=AnnotationConfidence(
                            overall_score=0.7,
                            model_confidence=0.7,
                            contextual_relevance=0.8,
                            semantic_similarity=0.75,
                        ),
                        source="topic_discovery",
                        tags=["automatic", "discovered", f"topic_{topic_id}"],
                    )

                    all_annotations.append(annotation)

        # Create collection
        collection = AnnotationCollection(
            annotations=all_annotations,
            metadata={
                "total_texts": len(texts),
                "annotation_methods": list(self.config["ensemble_weights"].keys()),
                "topic_discovery_enabled": discover_topics,
                "discovered_topics": len(self.topic_discovery.topics) if discover_topics else 0,
            },
        )

        return collection

    def explain_annotation(self, annotation: Annotation) -> Dict[str, Any]:
        """
        Explain why an annotation was made.

        Args:
            annotation: Annotation to explain

        Returns:
            Explanation dictionary
        """
        explanation = {
            "annotation_id": annotation.id,
            "topic": annotation.topic_name,
            "confidence": annotation.confidence.overall_score if annotation.confidence else 0,
            "source_method": annotation.source,
            "explanation_text": "",
        }

        # Get method-specific explanation
        if annotation.source == "zero_shot" and annotation.topic_id:
            zs_explain = self.zero_shot.explain_classification(
                annotation.content, annotation.topic_id
            )
            explanation["zero_shot_details"] = zs_explain
            explanation["explanation_text"] = zs_explain.get("explanation", "")

        elif annotation.source == "few_shot":
            fs_explain = self.setfit.explain_prediction(annotation.content)
            explanation["few_shot_details"] = fs_explain
            explanation["explanation_text"] = fs_explain.get("explanation", "")

        elif annotation.source == "bert_similarity":
            explanation["explanation_text"] = (
                f"Text has high semantic similarity ({annotation.confidence.semantic_similarity:.1%}) "
                f"to the concept of {annotation.topic_name}"
            )

        return explanation

    def update_config(self, config_updates: Dict[str, Any]):
        """Update engine configuration."""
        self.config.update(config_updates)

        # Clear caches if needed
        if "ensemble_weights" in config_updates:
            self.annotation_cache.clear()
