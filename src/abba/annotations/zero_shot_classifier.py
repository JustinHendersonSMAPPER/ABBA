"""
Zero-shot classification for theological concepts.

Uses natural language descriptions of theological concepts to classify
text without requiring training examples.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch


@dataclass
class ZeroShotPrediction:
    """Container for zero-shot classification results."""

    text: str
    labels: List[str]
    scores: List[float]
    predicted_label: str
    confidence: float


@dataclass
class TheologicalConcept:
    """Theological concept with description for zero-shot classification."""

    name: str
    description: str
    synonyms: List[str] = None
    key_phrases: List[str] = None
    biblical_examples: List[str] = None

    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if self.key_phrases is None:
            self.key_phrases = []
        if self.biblical_examples is None:
            self.biblical_examples = []


class ZeroShotTheologyClassifier:
    """
    Zero-shot classifier for theological concepts using natural language descriptions.

    Combines transformer-based zero-shot classification with semantic similarity
    for robust theological concept identification.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the zero-shot classifier.

        Args:
            model_name: HuggingFace model for zero-shot classification
            embedding_model: Sentence transformer for semantic similarity
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or (0 if torch.cuda.is_available() else -1)

        # Initialize zero-shot classifier
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=self.device)

        # Initialize embedding model for semantic similarity
        self.embedder = SentenceTransformer(embedding_model)

        # Theological concepts
        self.concepts = self._build_theological_concepts()

        # Precomputed embeddings
        self.concept_embeddings = {}
        self._precompute_embeddings()

    def _build_theological_concepts(self) -> Dict[str, TheologicalConcept]:
        """Build theological concept descriptions for zero-shot classification."""
        concepts = {}

        # Core theological concepts with rich descriptions
        concepts["trinity"] = TheologicalConcept(
            name="Trinity",
            description="The Christian doctrine that God exists as three persons - Father, Son, and Holy Spirit - in one divine essence, coequal and coeternal",
            synonyms=["triune god", "godhead", "three persons"],
            key_phrases=["father son holy spirit", "three in one", "triune"],
            biblical_examples=["In the name of the Father and of the Son and of the Holy Spirit"],
        )

        concepts["salvation"] = TheologicalConcept(
            name="Salvation",
            description="God's deliverance of humanity from sin and its consequences through faith in Jesus Christ, resulting in eternal life",
            synonyms=["saved", "redemption", "deliverance", "eternal life"],
            key_phrases=["saved by grace", "forgiveness of sins", "born again"],
            biblical_examples=["For by grace you have been saved through faith"],
        )

        concepts["justification"] = TheologicalConcept(
            name="Justification",
            description="God's act of declaring sinners righteous through faith in Christ, not by their own works or merit",
            synonyms=["declared righteous", "justified by faith"],
            key_phrases=["justified by faith", "counted as righteous", "not by works"],
            biblical_examples=[
                "Therefore, since we have been justified by faith, we have peace with God"
            ],
        )

        concepts["sanctification"] = TheologicalConcept(
            name="Sanctification",
            description="The process of being made holy and set apart for God through the work of the Holy Spirit",
            synonyms=["holiness", "consecration", "spiritual growth"],
            key_phrases=["made holy", "set apart", "grow in grace", "spiritual maturity"],
            biblical_examples=[
                "But now that you have been set free from sin and have become slaves of God"
            ],
        )

        concepts["atonement"] = TheologicalConcept(
            name="Atonement",
            description="Christ's sacrificial death that reconciles humanity to God by paying the penalty for sin",
            synonyms=["propitiation", "expiation", "reconciliation"],
            key_phrases=["blood of christ", "died for our sins", "sacrifice for sin"],
            biblical_examples=["He is the propitiation for our sins"],
        )

        concepts["grace"] = TheologicalConcept(
            name="Grace",
            description="God's unmerited favor and love toward humanity, given freely without being earned",
            synonyms=["unmerited favor", "divine favor", "god's kindness"],
            key_phrases=["undeserved favor", "gift of god", "not by works"],
            biblical_examples=["But by the grace of God I am what I am"],
        )

        concepts["faith"] = TheologicalConcept(
            name="Faith",
            description="Trust and belief in God and His promises, the means by which we receive salvation",
            synonyms=["belief", "trust", "confidence in god"],
            key_phrases=["believe in", "trust in the lord", "faith in christ"],
            biblical_examples=["Now faith is the assurance of things hoped for"],
        )

        concepts["repentance"] = TheologicalConcept(
            name="Repentance",
            description="Turning away from sin and turning toward God with genuine sorrow and commitment to change",
            synonyms=["turn from sin", "conversion", "change of heart"],
            key_phrases=["turn away from", "confess sins", "change your mind"],
            biblical_examples=["Repent and be baptized every one of you"],
        )

        concepts["holy_spirit"] = TheologicalConcept(
            name="Holy Spirit",
            description="The third person of the Trinity who indwells believers, guides, comforts, and empowers them",
            synonyms=["spirit of god", "comforter", "helper", "paraclete"],
            key_phrases=["filled with the spirit", "gifts of the spirit", "fruit of the spirit"],
            biblical_examples=["But the Helper, the Holy Spirit, whom the Father will send"],
        )

        concepts["church"] = TheologicalConcept(
            name="Church",
            description="The universal body of believers in Christ, as well as local assemblies of Christians",
            synonyms=["body of christ", "assembly", "congregation", "ekklesia"],
            key_phrases=["body of believers", "gathered together", "fellowship"],
            biblical_examples=[
                "And I tell you that you are Peter, and on this rock I will build my church"
            ],
        )

        concepts["baptism"] = TheologicalConcept(
            name="Baptism",
            description="The sacrament of initiation into the Christian faith, symbolizing death to sin and new life in Christ",
            synonyms=["water baptism", "immersion", "christening"],
            key_phrases=["baptized in water", "buried with christ", "baptized into christ"],
            biblical_examples=["Go therefore and make disciples of all nations, baptizing them"],
        )

        concepts["communion"] = TheologicalConcept(
            name="Communion",
            description="The sacrament commemorating Christ's death through bread and wine, representing His body and blood",
            synonyms=["lord's supper", "eucharist", "breaking of bread"],
            key_phrases=["body and blood", "do this in remembrance", "bread and wine"],
            biblical_examples=["This is my body, which is given for you"],
        )

        concepts["prayer"] = TheologicalConcept(
            name="Prayer",
            description="Communication with God through praise, thanksgiving, confession, and petition",
            synonyms=["supplication", "intercession", "petition"],
            key_phrases=["talk to god", "pray without ceasing", "ask in prayer"],
            biblical_examples=["Ask, and it will be given to you; seek, and you will find"],
        )

        concepts["worship"] = TheologicalConcept(
            name="Worship",
            description="Reverent honor and homage paid to God through praise, adoration, and service",
            synonyms=["praise", "adoration", "glorify"],
            key_phrases=["praise god", "bow down", "sing to the lord", "glorify god"],
            biblical_examples=[
                "God is spirit, and those who worship him must worship in spirit and truth"
            ],
        )

        concepts["sin"] = TheologicalConcept(
            name="Sin",
            description="Any thought, word, or action that violates God's law and separates humanity from God",
            synonyms=["transgression", "iniquity", "wrongdoing", "disobedience"],
            key_phrases=["fall short", "miss the mark", "break god's law"],
            biblical_examples=["For all have sinned and fall short of the glory of God"],
        )

        concepts["redemption"] = TheologicalConcept(
            name="Redemption",
            description="Being freed from slavery to sin through the payment of Christ's sacrifice",
            synonyms=["ransom", "deliverance", "liberation"],
            key_phrases=["set free", "bought with a price", "redeemed by blood"],
            biblical_examples=["In him we have redemption through his blood"],
        )

        concepts["gospel"] = TheologicalConcept(
            name="Gospel",
            description="The good news of salvation through Jesus Christ's life, death, and resurrection",
            synonyms=["good news", "glad tidings", "message of salvation"],
            key_phrases=["good news of christ", "preach the gospel", "salvation message"],
            biblical_examples=["For I am not ashamed of the gospel, for it is the power of God"],
        )

        concepts["kingdom_of_god"] = TheologicalConcept(
            name="Kingdom of God",
            description="God's sovereign rule and reign, both present in believers' hearts and future in its fullness",
            synonyms=["kingdom of heaven", "god's reign", "divine rule"],
            key_phrases=["kingdom come", "reign of god", "kingdom is near"],
            biblical_examples=["The kingdom of God is at hand; repent and believe in the gospel"],
        )

        concepts["eternal_life"] = TheologicalConcept(
            name="Eternal Life",
            description="The everlasting life with God that begins at salvation and continues forever",
            synonyms=["everlasting life", "life eternal", "immortality"],
            key_phrases=["live forever", "never perish", "life everlasting"],
            biblical_examples=["And this is eternal life, that they know you, the only true God"],
        )

        concepts["judgment"] = TheologicalConcept(
            name="Judgment",
            description="God's evaluation of humanity's deeds and hearts, resulting in eternal reward or punishment",
            synonyms=["divine judgment", "day of judgment", "final judgment"],
            key_phrases=["judge the world", "give account", "judgment seat"],
            biblical_examples=["For we must all appear before the judgment seat of Christ"],
        )

        return concepts

    def _precompute_embeddings(self):
        """Precompute embeddings for concept descriptions."""
        for concept_id, concept in self.concepts.items():
            # Combine description with examples for richer embedding
            texts = [concept.description]

            if concept.biblical_examples:
                texts.extend(concept.biblical_examples)

            # Compute embeddings
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)

            # Average embeddings
            self.concept_embeddings[concept_id] = np.mean(embeddings, axis=0)

    def classify(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
        multi_label: bool = True,
        threshold: float = 0.5,
    ) -> ZeroShotPrediction:
        """
        Classify text into theological concepts.

        Args:
            text: Text to classify
            candidate_labels: Specific labels to consider (None for all)
            multi_label: Whether to allow multiple labels
            threshold: Minimum score threshold for multi-label

        Returns:
            ZeroShotPrediction object
        """
        # Prepare candidate labels
        if candidate_labels is None:
            candidate_labels = list(self.concepts.keys())

        # Create hypothesis template for theological context
        hypothesis_template = "This text is about {}."

        # Get concept descriptions for labels
        label_descriptions = []
        label_names = []

        for label in candidate_labels:
            if label in self.concepts:
                concept = self.concepts[label]
                # Use both name and description
                label_descriptions.append(f"{concept.name}: {concept.description}")
                label_names.append(label)

        # Perform zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=label_descriptions,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )

        # Extract results
        if multi_label:
            # Filter by threshold
            labels = []
            scores = []

            for desc, score in zip(result["labels"], result["scores"]):
                if score >= threshold:
                    # Map back to concept ID
                    try:
                        idx = label_descriptions.index(desc)
                        labels.append(label_names[idx])
                        scores.append(score)
                    except ValueError:
                        # Handle case where result label doesn't match exactly
                        # Try to find a partial match
                        matched = False
                        for i, label_desc in enumerate(label_descriptions):
                            # Check if the desc starts with the same text as label_desc
                            # This handles truncation with "..."
                            desc_clean = desc.rstrip('.')
                            label_desc_clean = label_desc.rstrip('.')
                            if (desc in label_desc or label_desc in desc or 
                                desc_clean in label_desc_clean or label_desc_clean.startswith(desc_clean) or
                                desc_clean.startswith(label_desc_clean.split(':')[0])):
                                labels.append(label_names[i])
                                scores.append(score)
                                matched = True
                                break

            predicted_label = labels[0] if labels else None
            confidence = scores[0] if scores else 0.0
        else:
            # Single label
            try:
                idx = label_descriptions.index(result["labels"][0])
                predicted_label = label_names[idx]
            except ValueError:
                # Handle case where result label doesn't match exactly
                # Try to find a partial match
                result_label = result["labels"][0]
                matched = False
                for i, desc in enumerate(label_descriptions):
                    if result_label in desc or desc in result_label:
                        idx = i
                        predicted_label = label_names[idx]
                        matched = True
                        break
                
                if not matched:
                    # Default to first candidate or None
                    predicted_label = label_names[0] if label_names else None
                    
            confidence = result["scores"][0]
            labels = [predicted_label] if predicted_label else []
            scores = [confidence] if predicted_label else []

        return ZeroShotPrediction(
            text=text,
            labels=labels,
            scores=scores,
            predicted_label=predicted_label,
            confidence=confidence,
        )

    def classify_with_similarity(
        self, text: str, candidate_labels: Optional[List[str]] = None, alpha: float = 0.7
    ) -> ZeroShotPrediction:
        """
        Classify using both zero-shot and semantic similarity.

        Args:
            text: Text to classify
            candidate_labels: Specific labels to consider
            alpha: Weight for zero-shot scores (1-alpha for similarity)

        Returns:
            Combined prediction
        """
        # Get zero-shot predictions
        zs_result = self.classify(text, candidate_labels, multi_label=False)

        # Get semantic similarity scores
        text_embedding = self.embedder.encode(text, convert_to_numpy=True)

        similarity_scores = {}
        for label in candidate_labels or self.concepts.keys():
            if label in self.concept_embeddings:
                sim = util.cos_sim(text_embedding, self.concept_embeddings[label]).item()
                similarity_scores[label] = sim

        # Combine scores
        combined_scores = {}

        for label in zs_result.labels:
            zs_score = zs_result.scores[zs_result.labels.index(label)]
            sim_score = similarity_scores.get(label, 0.0)

            # Weighted combination
            combined_scores[label] = alpha * zs_score + (1 - alpha) * sim_score

        # Add any labels only in similarity scores
        for label, sim_score in similarity_scores.items():
            if label not in combined_scores:
                combined_scores[label] = (1 - alpha) * sim_score

        # Sort by combined score
        sorted_labels = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        labels = [label for label, _ in sorted_labels]
        scores = [score for _, score in sorted_labels]

        return ZeroShotPrediction(
            text=text,
            labels=labels,
            scores=scores,
            predicted_label=labels[0] if labels else None,
            confidence=scores[0] if scores else 0.0,
        )

    def explain_classification(self, text: str, label: str) -> Dict[str, Any]:
        """
        Explain why a text was classified with a particular label.

        Args:
            text: Classified text
            label: Label to explain

        Returns:
            Explanation dictionary
        """
        if label not in self.concepts:
            return {"error": f"Unknown label: {label}"}

        concept = self.concepts[label]

        # Find matching key phrases
        text_lower = text.lower()
        matching_phrases = []

        for phrase in concept.key_phrases:
            if phrase in text_lower:
                matching_phrases.append(phrase)

        # Find synonym matches
        matching_synonyms = []
        for synonym in concept.synonyms:
            if synonym in text_lower:
                matching_synonyms.append(synonym)

        # Calculate semantic similarity to examples
        if concept.biblical_examples:
            text_emb = self.embedder.encode(text, convert_to_numpy=True)
            example_sims = []

            for example in concept.biblical_examples:
                ex_emb = self.embedder.encode(example, convert_to_numpy=True)
                sim = util.cos_sim(text_emb, ex_emb).item()
                example_sims.append((example, sim))

            example_sims.sort(key=lambda x: x[1], reverse=True)
            best_example = example_sims[0] if example_sims else None
        else:
            best_example = None

        explanation = {
            "text": text,
            "label": label,
            "concept_name": concept.name,
            "concept_description": concept.description,
            "matching_key_phrases": matching_phrases,
            "matching_synonyms": matching_synonyms,
            "most_similar_example": best_example,
            "explanation": self._generate_explanation(
                concept, matching_phrases, matching_synonyms, best_example
            ),
        }

        return explanation

    def _generate_explanation(
        self,
        concept: TheologicalConcept,
        matching_phrases: List[str],
        matching_synonyms: List[str],
        best_example: Optional[Tuple[str, float]],
    ) -> str:
        """Generate human-readable explanation."""
        parts = [f"This text relates to {concept.name} because:"]

        if matching_phrases:
            parts.append(f"- It contains key phrases: {', '.join(matching_phrases)}")

        if matching_synonyms:
            parts.append(f"- It mentions related terms: {', '.join(matching_synonyms)}")

        if best_example and best_example[1] > 0.7:
            parts.append(f"- It's similar to the biblical example: '{best_example[0][:50]}...'")

        parts.append(f"- {concept.name} is defined as: {concept.description}")

        return "\n".join(parts)

    def batch_classify(
        self, texts: List[str], candidate_labels: Optional[List[str]] = None, batch_size: int = 8
    ) -> List[ZeroShotPrediction]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify
            candidate_labels: Labels to consider
            batch_size: Batch size for processing

        Returns:
            List of predictions
        """
        predictions = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Get predictions for batch
            for text in batch:
                pred = self.classify_with_similarity(text, candidate_labels)
                predictions.append(pred)

        return predictions
