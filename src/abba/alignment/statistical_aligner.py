"""
Advanced statistical alignment system for biblical texts.

This module implements a multi-stage alignment pipeline that combines
Strong's anchoring, statistical models, syntactic constraints, and
semantic loss detection to achieve maximum alignment accuracy.
"""

import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum

# import numpy as np  # Would be used for advanced statistical methods

from ..verse_id import VerseID
from ..parsers.translation_parser import TranslationVerse
from ..parsers.hebrew_parser import HebrewVerse
from ..parsers.greek_parser import GreekVerse
from ..interlinear.token_extractor import ExtractedToken


class AlignmentConfidence(Enum):
    """Confidence levels for alignments."""

    HIGH = "high"  # 0.8-1.0 (Strong's exact match)
    MEDIUM = "medium"  # 0.5-0.8 (Statistical + syntactic)
    LOW = "low"  # 0.2-0.5 (Statistical only)
    UNCERTAIN = "uncertain"  # 0.0-0.2 (No clear alignment)


class SemanticLossType(Enum):
    """Types of semantic information that may be lost in translation."""

    LEXICAL_RICHNESS = "lexical_richness"  # Original has richer meaning
    ASPECTUAL_DETAIL = "aspectual_detail"  # Greek aspect flattened
    CULTURAL_CONTEXT = "cultural_context"  # Ancient cultural concepts
    WORDPLAY = "wordplay"  # Puns, alliteration lost
    GRAMMATICAL_NUANCE = "grammatical_nuance"  # Hebrew construct chains, etc.


@dataclass
class SemanticLoss:
    """Represents semantic information lost in translation."""

    loss_type: SemanticLossType
    description: str
    original_concept: str
    translation_concept: str
    explanation: str
    severity: float  # 0.0-1.0, how much meaning is lost


@dataclass
class EnhancedAlignment:
    """Enhanced alignment with confidence and semantic loss detection."""

    source_tokens: List[ExtractedToken]
    target_words: List[str]
    strong_numbers: List[str]
    confidence: AlignmentConfidence
    confidence_score: float  # 0.0-1.0
    alignment_method: str  # How this alignment was determined
    semantic_losses: List[SemanticLoss]
    alternative_translations: List[str]  # Other possible translations
    morphological_notes: List[str]  # Grammatical explanations
    phrase_id: Optional[str] = None  # If part of a larger phrase

    def to_dict(self) -> Dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "source_text": " ".join(token.text for token in self.source_tokens),
            "target_text": " ".join(self.target_words),
            "strong_numbers": self.strong_numbers,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "alignment_method": self.alignment_method,
            "semantic_losses": [loss.__dict__ for loss in self.semantic_losses],
            "alternative_translations": self.alternative_translations,
            "morphological_notes": self.morphological_notes,
            "phrase_id": self.phrase_id,
        }


class StatisticalAligner:
    """Advanced statistical alignment system."""

    def __init__(self):
        # Strong's to English word mappings (learned from data)
        self.strongs_to_english: Dict[str, Dict[str, float]] = {}

        # Word co-occurrence probabilities
        self.translation_probs: Dict[Tuple[str, str], float] = {}

        # Phrase patterns
        self.phrase_patterns: Dict[str, List[str]] = {}

        # Semantic loss database
        self.semantic_loss_db: Dict[str, List[SemanticLoss]] = {}

        # Alignment statistics for confidence scoring
        self.alignment_stats: Dict[str, Dict] = defaultdict(dict)

    def build_alignment_model(
        self,
        original_verses: List[Union[HebrewVerse, GreekVerse]],
        translation_verses: List[TranslationVerse],
    ) -> None:
        """
        Build statistical alignment model from parallel corpus.

        This implements a simplified IBM Model 1 approach with Strong's anchoring.
        """
        print("Building statistical alignment model...")

        # Stage 1: Build Strong's-to-English mapping
        self._build_strongs_mapping(original_verses, translation_verses)

        # Stage 2: Learn translation probabilities
        self._learn_translation_probabilities(original_verses, translation_verses)

        # Stage 3: Identify phrase patterns
        self._identify_phrase_patterns(original_verses, translation_verses)

        # Stage 4: Build semantic loss database
        self._build_semantic_loss_database()

    def _build_strongs_mapping(
        self,
        original_verses: List[Union[HebrewVerse, GreekVerse]],
        translation_verses: List[TranslationVerse],
    ) -> None:
        """Build Strong's number to English word mappings."""
        # Create verse alignment by ID
        verse_map = {str(tv.verse_id): tv for tv in translation_verses}

        for orig_verse in original_verses:
            verse_key = str(orig_verse.verse_id)
            if verse_key not in verse_map:
                continue

            trans_verse = verse_map[verse_key]
            trans_words = self._tokenize_english(trans_verse.text)

            # Simple heuristic: distribute English words among Strong's numbers
            strong_words = [
                (word.strong_number, word.text) for word in orig_verse.words if word.strong_number
            ]

            if strong_words and trans_words:
                # Simple equal distribution (in reality, we'd use alignment algorithms)
                words_per_strong = max(1, len(trans_words) // len(strong_words))

                for i, (strong_num, orig_text) in enumerate(strong_words):
                    start_idx = i * words_per_strong
                    end_idx = min(start_idx + words_per_strong, len(trans_words))
                    aligned_words = trans_words[start_idx:end_idx]

                    if strong_num not in self.strongs_to_english:
                        self.strongs_to_english[strong_num] = defaultdict(float)

                    for word in aligned_words:
                        self.strongs_to_english[strong_num][word] += 1.0

        # Normalize to probabilities
        for strong_num in self.strongs_to_english:
            total = sum(self.strongs_to_english[strong_num].values())
            if total > 0:
                for word in self.strongs_to_english[strong_num]:
                    self.strongs_to_english[strong_num][word] /= total

    def _learn_translation_probabilities(
        self,
        original_verses: List[Union[HebrewVerse, GreekVerse]],
        translation_verses: List[TranslationVerse],
    ) -> None:
        """Learn word-to-word translation probabilities using EM algorithm."""
        # Simplified implementation - in practice, use fast_align or similar
        verse_map = {str(tv.verse_id): tv for tv in translation_verses}

        for orig_verse in original_verses:
            verse_key = str(orig_verse.verse_id)
            if verse_key not in verse_map:
                continue

            trans_verse = verse_map[verse_key]
            orig_words = [word.text for word in orig_verse.words]
            trans_words = self._tokenize_english(trans_verse.text)

            # Simple co-occurrence counting
            for orig_word in orig_words:
                for trans_word in trans_words:
                    self.translation_probs[(orig_word, trans_word)] = (
                        self.translation_probs.get((orig_word, trans_word), 0) + 1
                    )

    def _identify_phrase_patterns(
        self,
        original_verses: List[Union[HebrewVerse, GreekVerse]],
        translation_verses: List[TranslationVerse],
    ) -> None:
        """Identify common phrase patterns for multi-word alignments."""
        # Hebrew construct chains, Greek participial phrases, etc.
        # This would require syntactic parsing
        pass

    def _build_semantic_loss_database(self) -> None:
        """Build database of known semantic losses."""
        # Database of known semantic richness cases
        self.semantic_loss_db = {
            "H2617": [  # chesed
                SemanticLoss(
                    loss_type=SemanticLossType.LEXICAL_RICHNESS,
                    description="Hebrew 'chesed' encompasses covenant love, mercy, kindness, and faithfulness",
                    original_concept="chesed (חֶסֶד)",
                    translation_concept="mercy/kindness/love",
                    explanation="English lacks a single word for covenant faithfulness combined with mercy",
                    severity=0.7,
                )
            ],
            "G26": [  # agape
                SemanticLoss(
                    loss_type=SemanticLossType.LEXICAL_RICHNESS,
                    description="Greek 'agape' represents unconditional, self-sacrificial love",
                    original_concept="agape (ἀγάπη)",
                    translation_concept="love",
                    explanation="English 'love' doesn't distinguish agape from eros, phileo, or storge",
                    severity=0.6,
                )
            ],
            # Greek aspect markers
            "aorist": [
                SemanticLoss(
                    loss_type=SemanticLossType.ASPECTUAL_DETAIL,
                    description="Greek aorist indicates completed action",
                    original_concept="aorist aspect",
                    translation_concept="past tense",
                    explanation="English past tense doesn't capture the completed aspect nuance",
                    severity=0.4,
                )
            ],
        }

    def align_verse(
        self, original_verse: Union[HebrewVerse, GreekVerse], translation_verse: TranslationVerse
    ) -> List[EnhancedAlignment]:
        """
        Perform multi-stage alignment on a single verse.

        Returns list of EnhancedAlignment objects with confidence scores
        and semantic loss indicators.
        """
        alignments = []

        # Stage 1: Strong's anchoring
        strong_alignments = self._align_by_strongs(original_verse, translation_verse)
        alignments.extend(strong_alignments)

        # Stage 2: Statistical alignment for remaining tokens
        remaining_alignments = self._align_statistically(
            original_verse, translation_verse, strong_alignments
        )
        alignments.extend(remaining_alignments)

        # Stage 3: Phrase detection
        phrase_alignments = self._detect_phrases(alignments)

        # Stage 4: Semantic loss annotation
        self._annotate_semantic_losses(alignments)

        return alignments

    def _align_by_strongs(
        self, original_verse: Union[HebrewVerse, GreekVerse], translation_verse: TranslationVerse
    ) -> List[EnhancedAlignment]:
        """Stage 1: Align using Strong's numbers."""
        alignments = []
        trans_words = self._tokenize_english(translation_verse.text)

        for word in original_verse.words:
            if not word.strong_number:
                continue

            # Find most probable English words for this Strong's number
            if word.strong_number in self.strongs_to_english:
                strong_probs = self.strongs_to_english[word.strong_number]

                # Find matching words in translation
                matching_words = []
                for trans_word in trans_words:
                    if trans_word.lower() in strong_probs:
                        matching_words.append(trans_word)

                if matching_words:
                    # Create high-confidence alignment
                    token = ExtractedToken(
                        text=word.text, strong_number=word.strong_number, lemma=word.lemma
                    )

                    alignment = EnhancedAlignment(
                        source_tokens=[token],
                        target_words=matching_words,
                        strong_numbers=[word.strong_number],
                        confidence=AlignmentConfidence.HIGH,
                        confidence_score=max(strong_probs[w.lower()] for w in matching_words),
                        alignment_method="strongs_exact",
                        semantic_losses=[],
                        alternative_translations=list(strong_probs.keys())[:3],
                        morphological_notes=[],
                    )

                    alignments.append(alignment)

        return alignments

    def _align_statistically(
        self,
        original_verse: Union[HebrewVerse, GreekVerse],
        translation_verse: TranslationVerse,
        existing_alignments: List[EnhancedAlignment],
    ) -> List[EnhancedAlignment]:
        """Stage 2: Statistical alignment for remaining words."""
        # Find unaligned words
        aligned_strong_numbers = set()
        for alignment in existing_alignments:
            aligned_strong_numbers.update(alignment.strong_numbers)

        remaining_words = [
            word
            for word in original_verse.words
            if not word.strong_number or word.strong_number not in aligned_strong_numbers
        ]

        alignments = []
        trans_words = self._tokenize_english(translation_verse.text)

        # Simple statistical alignment (in practice, use more sophisticated algorithms)
        for word in remaining_words:
            best_matches = []
            best_score = 0.0

            for trans_word in trans_words:
                score = self.translation_probs.get((word.text, trans_word), 0.0)
                if score > best_score:
                    best_score = score
                    best_matches = [trans_word]
                elif score == best_score and score > 0:
                    best_matches.append(trans_word)

            if best_matches and best_score > 0.1:  # Threshold
                token = ExtractedToken(
                    text=word.text, strong_number=word.strong_number, lemma=word.lemma
                )

                confidence = (
                    AlignmentConfidence.MEDIUM if best_score > 0.5 else AlignmentConfidence.LOW
                )

                alignment = EnhancedAlignment(
                    source_tokens=[token],
                    target_words=best_matches,
                    strong_numbers=[word.strong_number] if word.strong_number else [],
                    confidence=confidence,
                    confidence_score=best_score,
                    alignment_method="statistical",
                    semantic_losses=[],
                    alternative_translations=[],
                    morphological_notes=[],
                )

                alignments.append(alignment)

        return alignments

    def _detect_phrases(self, alignments: List[EnhancedAlignment]) -> List[EnhancedAlignment]:
        """Stage 3: Detect and align phrase-level constructions."""
        # Identify Hebrew construct chains, Greek participial phrases, etc.
        # For now, return original alignments
        return alignments

    def _annotate_semantic_losses(self, alignments: List[EnhancedAlignment]) -> None:
        """Stage 4: Annotate semantic losses."""
        for alignment in alignments:
            for strong_num in alignment.strong_numbers:
                if strong_num in self.semantic_loss_db:
                    alignment.semantic_losses.extend(self.semantic_loss_db[strong_num])

    def _tokenize_english(self, text: str) -> List[str]:
        """Tokenize English text."""
        # Remove punctuation and split
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        return [word for word in clean_text.split() if word]

    def generate_search_indices(
        self, alignments: Dict[str, List[EnhancedAlignment]]
    ) -> Dict[str, Dict]:
        """
        Generate search indices for cross-language search.

        Returns indices for:
        - English word -> verses containing related original language words
        - Strong's number -> verses
        - Semantic concept -> verses
        - Morphological feature -> verses
        """
        indices = {
            "english_to_verses": defaultdict(set),
            "strongs_to_verses": defaultdict(set),
            "concept_to_verses": defaultdict(set),
            "morphology_to_verses": defaultdict(set),
        }

        for verse_id, verse_alignments in alignments.items():
            for alignment in verse_alignments:
                # English word index
                for word in alignment.target_words:
                    indices["english_to_verses"][word.lower()].add(verse_id)

                # Strong's index
                for strong_num in alignment.strong_numbers:
                    indices["strongs_to_verses"][strong_num].add(verse_id)

                # Concept index (from semantic loss data)
                for loss in alignment.semantic_losses:
                    indices["concept_to_verses"][loss.original_concept].add(verse_id)

                # Morphological index
                for token in alignment.source_tokens:
                    if token.morphology:
                        indices["morphology_to_verses"][str(token.morphology)].add(verse_id)

        return indices


# Usage example showing the complete pipeline
class AlignmentPipeline:
    """Complete alignment pipeline for ABBA project."""

    def __init__(self):
        self.aligner = StatisticalAligner()
        self.search_indices = {}

    def process_corpus(
        self,
        hebrew_verses: List[HebrewVerse],
        greek_verses: List[GreekVerse],
        translation_verses: List[TranslationVerse],
    ) -> Dict[str, List[EnhancedAlignment]]:
        """Process entire corpus and return alignments."""

        # Build model from all available data
        all_original = hebrew_verses + greek_verses
        self.aligner.build_alignment_model(all_original, translation_verses)

        # Process each translation version
        translation_map = defaultdict(list)
        for tv in translation_verses:
            translation_map[str(tv.verse_id)].append(tv)

        all_alignments = {}

        # Align each original verse with its translations
        for orig_verse in all_original:
            verse_key = str(orig_verse.verse_id)
            if verse_key in translation_map:
                verse_alignments = []

                for trans_verse in translation_map[verse_key]:
                    alignments = self.aligner.align_verse(orig_verse, trans_verse)
                    verse_alignments.extend(alignments)

                all_alignments[verse_key] = verse_alignments

        # Build search indices
        self.search_indices = self.aligner.generate_search_indices(all_alignments)

        return all_alignments

    def search_cross_language(self, query: str, search_type: str = "english") -> List[str]:
        """
        Search across languages using the alignment data.

        Args:
            query: Search term
            search_type: "english", "strongs", "concept", or "morphology"

        Returns:
            List of verse IDs matching the query
        """
        if search_type == "english":
            return list(self.search_indices["english_to_verses"].get(query.lower(), set()))
        elif search_type == "strongs":
            return list(self.search_indices["strongs_to_verses"].get(query, set()))
        elif search_type == "concept":
            return list(self.search_indices["concept_to_verses"].get(query, set()))
        elif search_type == "morphology":
            return list(self.search_indices["morphology_to_verses"].get(query, set()))
        else:
            return []
