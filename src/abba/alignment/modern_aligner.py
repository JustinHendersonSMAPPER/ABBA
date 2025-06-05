"""
Modern alignment system using current NLP libraries.

This implementation replaces fast_align with modern, well-maintained libraries
while adding ML/statistical controls for improved quality assurance.
"""

import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import re

# Modern NLP libraries (well-maintained)
try:
    import spacy
    from spacy.tokens import Doc

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..verse_id import VerseID
from ..parsers.translation_parser import TranslationVerse
from ..parsers.hebrew_parser import HebrewVerse
from ..parsers.greek_parser import GreekVerse
from ..interlinear.token_extractor import ExtractedToken


@dataclass
class ModernAlignment:
    """Enhanced alignment using modern NLP techniques."""

    source_tokens: List[ExtractedToken]
    target_words: List[str]
    strong_numbers: List[str]

    # Confidence scoring
    confidence_score: float  # 0.0-1.0 overall confidence
    confidence_breakdown: Dict[str, float]  # Per-method confidence

    # Alignment methods used
    alignment_methods: List[str]  # ['strongs', 'neural', 'semantic']

    # Quality metrics
    cross_translation_consistency: float  # How consistent across translations
    semantic_similarity: float  # Contextual semantic similarity
    syntactic_compatibility: float  # Grammar compatibility

    # Semantic analysis
    semantic_losses: List[Dict]  # Detected semantic losses
    cultural_context: Optional[str]  # Cultural/historical context
    morphological_notes: List[str]  # Grammatical explanations

    # Alternative options
    alternative_alignments: List[Dict]  # Other possible alignments
    uncertainty_flags: List[str]  # What makes this uncertain

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_text": " ".join(token.text for token in self.source_tokens),
            "target_text": " ".join(self.target_words),
            "strong_numbers": self.strong_numbers,
            "confidence_score": self.confidence_score,
            "confidence_breakdown": self.confidence_breakdown,
            "alignment_methods": self.alignment_methods,
            "quality_metrics": {
                "cross_translation_consistency": self.cross_translation_consistency,
                "semantic_similarity": self.semantic_similarity,
                "syntactic_compatibility": self.syntactic_compatibility,
            },
            "semantic_losses": self.semantic_losses,
            "cultural_context": self.cultural_context,
            "morphological_notes": self.morphological_notes,
            "alternative_alignments": self.alternative_alignments,
            "uncertainty_flags": self.uncertainty_flags,
        }


class ConfidenceEnsemble:
    """Ensemble method for computing alignment confidence."""

    def __init__(self):
        self.weights = {
            "strongs": 0.4,  # Strong's number exact match
            "semantic": 0.25,  # Semantic similarity
            "neural": 0.25,  # Neural alignment confidence
            "syntactic": 0.1,  # Syntactic compatibility
        }

        # Load semantic model if available
        self.semantic_model = None
        if ML_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e:
                logging.warning(f"Could not load semantic model: {e}")

    def compute_confidence(
        self, alignment: ModernAlignment, context: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute ensemble confidence score."""

        scores = {}

        # Strong's number confidence
        scores["strongs"] = self._compute_strongs_confidence(alignment)

        # Semantic similarity confidence
        scores["semantic"] = self._compute_semantic_confidence(alignment, context)

        # Neural alignment confidence (simulated for now)
        scores["neural"] = self._compute_neural_confidence(alignment, context)

        # Syntactic compatibility
        scores["syntactic"] = self._compute_syntactic_confidence(alignment)

        # Weighted ensemble
        overall_confidence = sum(
            self.weights[method] * scores[method] for method in scores if scores[method] is not None
        )

        return overall_confidence, scores

    def _compute_strongs_confidence(self, alignment: ModernAlignment) -> float:
        """Confidence based on Strong's number availability and frequency."""
        if not alignment.strong_numbers:
            return 0.0

        # High confidence if Strong's numbers are available
        return 0.9 if alignment.strong_numbers else 0.0

    def _compute_semantic_confidence(
        self, alignment: ModernAlignment, context: Dict
    ) -> Optional[float]:
        """Confidence based on semantic similarity."""
        if not self.semantic_model or not ML_AVAILABLE:
            return None

        try:
            source_text = " ".join(token.text for token in alignment.source_tokens)
            target_text = " ".join(alignment.target_words)

            # Get embeddings
            source_embedding = self.semantic_model.encode([source_text])
            target_embedding = self.semantic_model.encode([target_text])

            # Compute similarity
            similarity = cosine_similarity(source_embedding, target_embedding)[0][0]

            return float(similarity)

        except Exception as e:
            logging.warning(f"Error computing semantic similarity: {e}")
            return None

    def _compute_neural_confidence(self, alignment: ModernAlignment, context: Dict) -> float:
        """Simulated neural alignment confidence."""
        # In real implementation, this would use awesome-align or similar
        # For now, simulate based on word length and frequency

        source_length = len(alignment.source_tokens)
        target_length = len(alignment.target_words)

        # Handle empty alignments
        if source_length == 0 or target_length == 0:
            return 0.0

        # Prefer 1:1 alignments, penalize many:many
        length_ratio = min(source_length, target_length) / max(source_length, target_length)

        return length_ratio * 0.8  # Simulated confidence

    def _compute_syntactic_confidence(self, alignment: ModernAlignment) -> float:
        """Confidence based on syntactic compatibility."""
        # Check if parts of speech are compatible

        # Get morphological information from tokens
        source_pos = []
        for token in alignment.source_tokens:
            if token.morphology:
                pos = getattr(token.morphology, "part_of_speech", None)
                if pos:
                    source_pos.append(pos)

        if not source_pos:
            return 0.5  # Neutral if no morphological data

        # Simple heuristic: noun-noun, verb-verb alignments are good
        # This would be more sophisticated in practice
        if len(set(source_pos)) == 1:  # All same POS
            return 0.8
        else:
            return 0.6


class CrossTranslationValidator:
    """Validates alignment consistency across multiple translations."""

    def __init__(self):
        self.translation_alignments: Dict[str, Dict] = defaultdict(dict)

    def add_translation_alignment(
        self, translation_id: str, verse_id: str, alignments: List[ModernAlignment]
    ):
        """Add alignment data for a specific translation."""
        self.translation_alignments[translation_id][verse_id] = alignments

    def compute_consistency_score(self, verse_id: str, strong_number: str) -> float:
        """Compute how consistently a Strong's number aligns across translations."""

        aligned_words = []

        for translation_id in self.translation_alignments:
            if verse_id in self.translation_alignments[translation_id]:
                verse_alignments = self.translation_alignments[translation_id][verse_id]

                for alignment in verse_alignments:
                    if strong_number in alignment.strong_numbers:
                        aligned_words.extend(alignment.target_words)

        if not aligned_words:
            return 0.0

        # Measure consistency using most common word frequency
        word_counts = Counter(aligned_words)
        most_common_count = word_counts.most_common(1)[0][1]
        total_count = len(aligned_words)

        consistency = most_common_count / total_count
        return consistency

    def validate_alignment_set(
        self, verse_id: str, alignments: List[ModernAlignment]
    ) -> Dict[str, float]:
        """Validate a set of alignments for consistency."""

        consistency_scores = {}

        for alignment in alignments:
            for strong_num in alignment.strong_numbers:
                consistency = self.compute_consistency_score(verse_id, strong_num)
                consistency_scores[strong_num] = consistency

        return consistency_scores


class ModernAlignmentPipeline:
    """Complete modern alignment pipeline."""

    def __init__(self):
        self.confidence_ensemble = ConfidenceEnsemble()
        self.cross_validator = CrossTranslationValidator()

        # Strong's mapping database (would be built from training data)
        self.strongs_mappings: Dict[str, Dict[str, float]] = {}

        # Phrase pattern database
        self.phrase_patterns: Dict[str, List[str]] = {}

        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                # Try to load multilingual model
                self.nlp = spacy.load("xx_core_web_sm")
            except OSError:
                logging.warning("spaCy multilingual model not available")

    def build_training_data(
        self,
        hebrew_verses: List[HebrewVerse],
        greek_verses: List[GreekVerse],
        translation_verses: List[TranslationVerse],
    ):
        """Build training data from parallel corpus."""

        print("Building Strong's mappings from training data...")

        # Group translations by verse
        translation_map = defaultdict(list)
        for tv in translation_verses:
            translation_map[str(tv.verse_id)].append(tv)

        # Build Strong's to English mappings
        all_original_verses = hebrew_verses + greek_verses

        for orig_verse in all_original_verses:
            verse_key = str(orig_verse.verse_id)

            if verse_key in translation_map:
                for trans_verse in translation_map[verse_key]:
                    self._update_strongs_mappings(orig_verse, trans_verse)

        # Normalize probabilities
        self._normalize_strongs_mappings()

        print(f"Built mappings for {len(self.strongs_mappings)} Strong's numbers")

    def _update_strongs_mappings(
        self, orig_verse: Union[HebrewVerse, GreekVerse], trans_verse: TranslationVerse
    ):
        """Update Strong's to English word mappings."""

        trans_words = self._tokenize_english(trans_verse.text)
        strong_words = [
            (word.strong_number, word.text) for word in orig_verse.words if word.strong_number
        ]

        if not strong_words or not trans_words:
            return

        # Simple distribution (in practice, would use more sophisticated alignment)
        words_per_strong = max(1, len(trans_words) // len(strong_words))

        for i, (strong_num, orig_text) in enumerate(strong_words):
            start_idx = i * words_per_strong
            end_idx = min(start_idx + words_per_strong, len(trans_words))
            aligned_words = trans_words[start_idx:end_idx]

            if strong_num not in self.strongs_mappings:
                self.strongs_mappings[strong_num] = defaultdict(float)

            for word in aligned_words:
                self.strongs_mappings[strong_num][word] += 1.0

    def _normalize_strongs_mappings(self):
        """Normalize Strong's mappings to probabilities."""
        for strong_num in self.strongs_mappings:
            total = sum(self.strongs_mappings[strong_num].values())
            if total > 0:
                for word in self.strongs_mappings[strong_num]:
                    self.strongs_mappings[strong_num][word] /= total

    def align_verse(
        self, orig_verse: Union[HebrewVerse, GreekVerse], trans_verse: TranslationVerse
    ) -> List[ModernAlignment]:
        """Align a single verse using modern pipeline."""

        alignments = []
        trans_words = self._tokenize_english(trans_verse.text)

        # Stage 1: Strong's-based alignment
        strong_alignments = self._align_by_strongs(orig_verse, trans_words)

        # Stage 2: Add confidence scoring
        for alignment in strong_alignments:
            context = {"original_verse": orig_verse, "translation_verse": trans_verse}

            confidence, breakdown = self.confidence_ensemble.compute_confidence(alignment, context)
            alignment.confidence_score = confidence
            alignment.confidence_breakdown = breakdown

            # Add cross-translation consistency if available
            alignment.cross_translation_consistency = self._get_consistency_score(
                str(orig_verse.verse_id), alignment.strong_numbers
            )

            alignments.append(alignment)

        return alignments

    def _align_by_strongs(
        self, orig_verse: Union[HebrewVerse, GreekVerse], trans_words: List[str]
    ) -> List[ModernAlignment]:
        """Align using Strong's numbers."""

        alignments = []

        for word in orig_verse.words:
            if not word.strong_number or word.strong_number not in self.strongs_mappings:
                continue

            # Find best matching English words
            strong_probs = self.strongs_mappings[word.strong_number]
            matching_words = []

            for trans_word in trans_words:
                if trans_word.lower() in strong_probs:
                    matching_words.append(trans_word)

            if matching_words:
                token = ExtractedToken(
                    text=word.text, strong_number=word.strong_number, lemma=word.lemma
                )

                alignment = ModernAlignment(
                    source_tokens=[token],
                    target_words=matching_words,
                    strong_numbers=[word.strong_number],
                    confidence_score=0.0,  # Will be computed later
                    confidence_breakdown={},
                    alignment_methods=["strongs"],
                    cross_translation_consistency=0.0,
                    semantic_similarity=0.0,
                    syntactic_compatibility=0.0,
                    semantic_losses=[],
                    cultural_context=None,
                    morphological_notes=[],
                    alternative_alignments=[],
                    uncertainty_flags=[],
                )

                alignments.append(alignment)

        return alignments

    def _get_consistency_score(self, verse_id: str, strong_numbers: List[str]) -> float:
        """Get cross-translation consistency score."""
        if not strong_numbers:
            return 0.0

        scores = []
        for strong_num in strong_numbers:
            score = self.cross_validator.compute_consistency_score(verse_id, strong_num)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _tokenize_english(self, text: str) -> List[str]:
        """Tokenize English text."""
        # Remove punctuation and split
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        return [word for word in clean_text.split() if word]

    def generate_quality_report(self, alignments: Dict[str, List[ModernAlignment]]) -> Dict:
        """Generate comprehensive quality report."""

        total_alignments = sum(len(verse_alignments) for verse_alignments in alignments.values())

        # Confidence distribution
        confidence_scores = []
        method_usage = defaultdict(int)

        for verse_alignments in alignments.values():
            for alignment in verse_alignments:
                confidence_scores.append(alignment.confidence_score)
                for method in alignment.alignment_methods:
                    method_usage[method] += 1

        # Quality metrics
        high_confidence = sum(1 for score in confidence_scores if score > 0.8)
        medium_confidence = sum(1 for score in confidence_scores if 0.5 <= score <= 0.8)
        low_confidence = sum(1 for score in confidence_scores if score < 0.5)

        report = {
            "total_alignments": total_alignments,
            "confidence_distribution": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence,
                "percentages": {
                    "high": high_confidence / total_alignments * 100,
                    "medium": medium_confidence / total_alignments * 100,
                    "low": low_confidence / total_alignments * 100,
                },
            },
            "method_usage": dict(method_usage),
            "average_confidence": (
                sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            ),
            "strongs_coverage": len(self.strongs_mappings),
            "recommendations": self._generate_recommendations(confidence_scores, method_usage),
        }

        return report

    def _generate_recommendations(
        self, confidence_scores: List[float], method_usage: Dict
    ) -> List[str]:
        """Generate recommendations for improving alignment quality."""

        recommendations = []

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        if avg_confidence < 0.7:
            recommendations.append(
                "Consider adding more training data to improve alignment confidence"
            )

        low_confidence_ratio = sum(1 for score in confidence_scores if score < 0.5) / len(
            confidence_scores
        )
        if low_confidence_ratio > 0.2:
            recommendations.append(
                "High number of low-confidence alignments - review uncertain cases"
            )

        if method_usage.get("strongs", 0) < method_usage.get("neural", 0):
            recommendations.append("Strong's coverage could be improved")

        return recommendations


# Usage example
def demonstrate_modern_pipeline():
    """Demonstrate the modern alignment pipeline."""

    pipeline = ModernAlignmentPipeline()

    # This would be called with real data
    print("Modern Alignment Pipeline Demo")
    print("=" * 50)

    if ML_AVAILABLE:
        print("✓ Machine Learning libraries available")
    else:
        print("⚠ ML libraries not available - using fallback methods")

    if SPACY_AVAILABLE:
        print("✓ spaCy available for advanced NLP")
    else:
        print("⚠ spaCy not available - using basic tokenization")

    print("\nPipeline components:")
    print("- Confidence ensemble scoring")
    print("- Cross-translation validation")
    print("- Semantic similarity analysis")
    print("- Quality assurance reporting")

    return pipeline


if __name__ == "__main__":
    demonstrate_modern_pipeline()
