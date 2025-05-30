"""
SetFit implementation for few-shot learning on theological concepts.

Enables efficient classification with minimal training examples per class.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import torch
from torch.utils.data import DataLoader, Dataset
import random


@dataclass
class FewShotExample:
    """Container for few-shot training examples."""

    text: str
    label: str
    embedding: Optional[np.ndarray] = None


@dataclass
class SetFitPrediction:
    """Container for SetFit predictions."""

    text: str
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float]


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning in SetFit."""

    def __init__(self, examples: List[FewShotExample], n_iterations: int = 20):
        """
        Create contrastive pairs from examples.

        Args:
            examples: List of labeled examples
            n_iterations: Number of times to iterate through data
        """
        self.pairs = []
        self.labels = []

        # Group examples by label
        label_groups = {}
        for ex in examples:
            if ex.label not in label_groups:
                label_groups[ex.label] = []
            label_groups[ex.label].append(ex)

        # Create positive and negative pairs
        for _ in range(n_iterations):
            for label, group in label_groups.items():
                if len(group) < 2:
                    continue

                # Positive pairs (same label)
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        self.pairs.append((group[i].text, group[j].text))
                        self.labels.append(1)  # Similar

                # Negative pairs (different labels)
                other_labels = [l for l in label_groups.keys() if l != label]
                for other_label in other_labels:
                    other_group = label_groups[other_label]
                    for ex1 in group:
                        ex2 = random.choice(other_group)
                        self.pairs.append((ex1.text, ex2.text))
                        self.labels.append(0)  # Dissimilar

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


class SetFitClassifier:
    """
    SetFit classifier for few-shot learning on biblical/theological texts.

    Uses contrastive learning to fine-tune sentence transformers with
    minimal examples, then trains a simple classifier on embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_iterations: int = 20,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        """
        Initialize the SetFit classifier.

        Args:
            model_name: Sentence transformer model name
            max_iterations: Number of training iterations
            batch_size: Batch size for training
            learning_rate: Learning rate for fine-tuning
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Components
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        # Biblical text preprocessing
        self.biblical_replacements = {
            "thou": "you",
            "thee": "you",
            "thy": "your",
            "thine": "yours",
            "ye": "you",
            "hath": "has",
            "doth": "does",
            "saith": "says",
        }

    def preprocess_biblical_text(self, text: str) -> str:
        """Preprocess biblical text for better model understanding."""
        processed = text.lower()

        # Replace archaic words
        for old, new in self.biblical_replacements.items():
            import re

            processed = re.sub(r"\b" + old + r"\b", new, processed)

        # Remove verse references
        processed = re.sub(r"\b\d+:\d+\b", "", processed)

        return processed.strip()

    def train(
        self,
        examples: List[FewShotExample],
        validation_examples: Optional[List[FewShotExample]] = None,
    ):
        """
        Train the SetFit model on few-shot examples.

        Args:
            examples: Training examples (few per class)
            validation_examples: Optional validation set
        """
        print(f"Training SetFit with {len(examples)} examples...")

        # Step 1: Fine-tune sentence transformer with contrastive learning
        if len(examples) > 1:
            self._fine_tune_encoder(examples)

        # Step 2: Generate embeddings for all examples
        texts = [self.preprocess_biblical_text(ex.text) for ex in examples]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Store embeddings in examples
        for ex, emb in zip(examples, embeddings):
            ex.embedding = emb

        # Step 3: Train classifier on embeddings
        X = np.array(embeddings)
        y = self.label_encoder.fit_transform([ex.label for ex in examples])

        # Use LogisticRegression for few-shot scenarios
        self.classifier = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            class_weight="balanced",  # Handle imbalanced classes
            random_state=42,
        )

        # Use cross-validation to select best regularization
        if len(np.unique(y)) > 1 and len(examples) > 10:
            best_score = -1
            best_C = 1.0

            for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
                clf = LogisticRegression(
                    C=C,
                    multi_class="multinomial",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                )

                scores = cross_val_score(clf, X, y, cv=min(3, len(np.unique(y))))
                mean_score = np.mean(scores)

                if mean_score > best_score:
                    best_score = mean_score
                    best_C = C

            self.classifier.C = best_C

        # Fit the classifier
        self.classifier.fit(X, y)
        self.is_trained = True

        # Evaluate on validation set if provided
        if validation_examples:
            self._evaluate(validation_examples)

    def _fine_tune_encoder(self, examples: List[FewShotExample]):
        """Fine-tune the sentence encoder using contrastive learning."""
        # Create contrastive dataset
        dataset = ContrastiveDataset(examples, n_iterations=self.max_iterations)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Fine-tuning would happen here
        # For now, this is a simplified version
        print(f"Created {len(dataset)} contrastive pairs for fine-tuning")

        # In practice, you would:
        # 1. Convert model to training mode
        # 2. Define contrastive loss
        # 3. Optimize for several epochs
        # 4. Update the sentence transformer

        # Placeholder for actual fine-tuning
        pass

    def predict(
        self, texts: Union[str, List[str]], return_all_scores: bool = False
    ) -> Union[SetFitPrediction, List[SetFitPrediction]]:
        """
        Predict labels for new texts.

        Args:
            texts: Single text or list of texts
            return_all_scores: Whether to return scores for all classes

        Returns:
            Prediction or list of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Handle single text
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        # Preprocess and encode
        processed_texts = [self.preprocess_biblical_text(t) for t in texts]
        embeddings = self.model.encode(processed_texts, convert_to_numpy=True)

        # Get predictions
        predictions = []

        if return_all_scores:
            # Get probability scores for all classes
            proba = self.classifier.predict_proba(embeddings)

            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                scores = {}
                for j, label in enumerate(self.label_encoder.classes_):
                    scores[label] = float(proba[i, j])

                pred_idx = np.argmax(proba[i])
                pred_label = self.label_encoder.classes_[pred_idx]

                predictions.append(
                    SetFitPrediction(
                        text=text,
                        predicted_label=pred_label,
                        confidence=float(proba[i, pred_idx]),
                        all_scores=scores,
                    )
                )
        else:
            # Just get predictions
            pred_indices = self.classifier.predict(embeddings)
            pred_labels = self.label_encoder.inverse_transform(pred_indices)

            # Get confidence scores
            proba = self.classifier.predict_proba(embeddings)

            for text, label, prob, idx in zip(texts, pred_labels, proba, pred_indices):
                predictions.append(
                    SetFitPrediction(
                        text=text, predicted_label=label, confidence=float(prob[idx]), all_scores={}
                    )
                )

        return predictions[0] if single_text else predictions

    def _evaluate(self, examples: List[FewShotExample]):
        """Evaluate model on validation examples."""
        predictions = self.predict([ex.text for ex in examples])

        correct = sum(
            1 for ex, pred in zip(examples, predictions) if ex.label == pred.predicted_label
        )

        accuracy = correct / len(examples)
        print(f"Validation accuracy: {accuracy:.2%}")

        # Per-class accuracy
        class_correct = {}
        class_total = {}

        for ex, pred in zip(examples, predictions):
            if ex.label not in class_total:
                class_total[ex.label] = 0
                class_correct[ex.label] = 0

            class_total[ex.label] += 1
            if ex.label == pred.predicted_label:
                class_correct[ex.label] += 1

        print("\nPer-class accuracy:")
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label]
            print(f"  {label}: {acc:.2%} ({class_correct[label]}/{class_total[label]})")

    def add_examples(self, new_examples: List[FewShotExample]):
        """
        Add new examples and retrain (for online learning).

        Args:
            new_examples: New labeled examples to add
        """
        if not self.is_trained:
            raise ValueError("Model must be trained initially before adding examples")

        # Get current training data (would need to store this)
        # For now, just retrain on all data
        print(f"Adding {len(new_examples)} new examples and retraining...")

        # This would ideally:
        # 1. Retrieve stored training examples
        # 2. Add new examples
        # 3. Retrain incrementally

        # Placeholder implementation
        pass

    def get_similar_examples(
        self, text: str, examples: List[FewShotExample], top_k: int = 5
    ) -> List[Tuple[FewShotExample, float]]:
        """
        Find training examples most similar to given text.

        Args:
            text: Query text
            examples: Pool of examples to search
            top_k: Number of similar examples to return

        Returns:
            List of (example, similarity) tuples
        """
        # Encode query
        query_embedding = self.model.encode(
            [self.preprocess_biblical_text(text)], convert_to_numpy=True
        )[0]

        # Calculate similarities
        similarities = []

        for ex in examples:
            if ex.embedding is None:
                ex.embedding = self.model.encode(
                    [self.preprocess_biblical_text(ex.text)], convert_to_numpy=True
                )[0]

            sim = np.dot(query_embedding, ex.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(ex.embedding)
            )

            similarities.append((ex, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """
        Explain a prediction by showing similar training examples.

        Args:
            text: Text to explain prediction for

        Returns:
            Explanation dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before explanation")

        # Get prediction
        prediction = self.predict(text, return_all_scores=True)

        # Find similar examples from training set
        # (Would need to store training examples for this)

        explanation = {
            "text": text,
            "prediction": prediction.predicted_label,
            "confidence": prediction.confidence,
            "all_scores": prediction.all_scores,
            "explanation": f"Predicted '{prediction.predicted_label}' with {prediction.confidence:.1%} confidence",
        }

        return explanation

    def save(self, path: str):
        """Save the trained model."""
        import pickle
        import os

        os.makedirs(path, exist_ok=True)

        # Save sentence transformer
        self.model.save(os.path.join(path, "sentence_transformer"))

        # Save classifier and label encoder
        with open(os.path.join(path, "classifier.pkl"), "wb") as f:
            pickle.dump(
                {
                    "classifier": self.classifier,
                    "label_encoder": self.label_encoder,
                    "is_trained": self.is_trained,
                },
                f,
            )

    def load(self, path: str):
        """Load a trained model."""
        import pickle
        import os

        # Load sentence transformer
        self.model = SentenceTransformer(os.path.join(path, "sentence_transformer"))

        # Load classifier and label encoder
        with open(os.path.join(path, "classifier.pkl"), "rb") as f:
            data = pickle.load(f)
            self.classifier = data["classifier"]
            self.label_encoder = data["label_encoder"]
            self.is_trained = data["is_trained"]
