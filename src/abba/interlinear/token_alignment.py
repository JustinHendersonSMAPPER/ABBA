"""
Token-level alignment between original languages and translations.

This module provides functionality to align tokens from Hebrew/Greek
with their corresponding words in translations using Strong's numbers
and linguistic analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .token_extractor import ExtractedToken


class AlignmentType(Enum):
    """Type of token alignment."""

    ONE_TO_ONE = "one_to_one"  # Single token to single word
    ONE_TO_MANY = "one_to_many"  # Single token to multiple words
    MANY_TO_ONE = "many_to_one"  # Multiple tokens to single word
    MANY_TO_MANY = "many_to_many"  # Multiple tokens to multiple words
    PHRASE = "phrase"  # Phrase-level alignment
    UNALIGNED = "unaligned"  # No alignment found


@dataclass
class AlignedToken:
    """Represents an aligned token with its translation(s)."""

    source_tokens: List[ExtractedToken]  # Original language token(s)
    target_words: List[str]  # Translation word(s)
    alignment_type: AlignmentType
    confidence: float = 1.0  # Alignment confidence (0-1)
    strong_numbers: List[str] = None  # Strong's numbers involved
    notes: Optional[str] = None  # Alignment notes

    def __post_init__(self) -> None:
        """Initialize derived fields."""
        if self.strong_numbers is None:
            self.strong_numbers = []
            for token in self.source_tokens:
                if token.strong_number:
                    self.strong_numbers.append(token.strong_number)

    def get_source_text(self) -> str:
        """Get concatenated source text."""
        return " ".join(token.text for token in self.source_tokens)

    def get_target_text(self) -> str:
        """Get concatenated target text."""
        return " ".join(self.target_words)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format."""
        return {
            "source_text": self.get_source_text(),
            "target_text": self.get_target_text(),
            "alignment_type": self.alignment_type.value,
            "confidence": self.confidence,
            "strong_numbers": self.strong_numbers,
            "source_tokens": [token.to_dict() for token in self.source_tokens],
            "target_words": self.target_words,
            "notes": self.notes,
        }


@dataclass
class TokenAlignment:
    """Complete token alignment for a verse."""

    verse_id: str
    source_language: str
    target_language: str
    alignments: List[AlignedToken]
    unaligned_source: List[ExtractedToken]  # Source tokens with no alignment
    unaligned_target: List[str]  # Target words with no alignment
    alignment_score: float = 0.0  # Overall alignment quality (0-1)

    def get_coverage_stats(self) -> Dict[str, float]:
        """Calculate alignment coverage statistics."""
        total_source = len(self.alignments) + len(self.unaligned_source)
        total_target = len(self.alignments) + len(self.unaligned_target)

        source_coverage = len(self.alignments) / total_source if total_source > 0 else 0
        target_coverage = len(self.alignments) / total_target if total_target > 0 else 0

        return {
            "source_coverage": source_coverage,
            "target_coverage": target_coverage,
            "overall_coverage": (source_coverage + target_coverage) / 2,
        }

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format."""
        return {
            "verse_id": self.verse_id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "alignments": [align.to_dict() for align in self.alignments],
            "unaligned_source": [token.to_dict() for token in self.unaligned_source],
            "unaligned_target": self.unaligned_target,
            "alignment_score": self.alignment_score,
            "coverage_stats": self.get_coverage_stats(),
        }


class TokenAligner:
    """Aligns tokens between original languages and translations."""

    def __init__(self, strongs_mapping: Optional[Dict[str, List[str]]] = None) -> None:
        """
        Initialize the token aligner.

        Args:
            strongs_mapping: Optional mapping of Strong's numbers to English words
        """
        self.strongs_mapping = strongs_mapping or {}
        self._alignment_cache: Dict[str, TokenAlignment] = {}

    def align_tokens(
        self,
        source_tokens: List[ExtractedToken],
        target_text: str,
        verse_id: str,
        source_language: str = "hebrew",
        target_language: str = "english",
    ) -> TokenAlignment:
        """
        Align source tokens with target translation text.

        Args:
            source_tokens: List of extracted tokens from original language
            target_text: Translation text
            verse_id: Verse identifier
            source_language: Source language name
            target_language: Target language name

        Returns:
            TokenAlignment object with alignment results
        """
        # Check cache
        cache_key = f"{verse_id}:{source_language}:{target_language}"
        if cache_key in self._alignment_cache:
            return self._alignment_cache[cache_key]

        # Split target text into words
        target_words = self._tokenize_target(target_text)

        # Track which tokens/words have been aligned
        aligned_source_indices = set()
        aligned_target_indices = set()
        alignments = []

        # First pass: Strong's number based alignment
        if self.strongs_mapping:
            strongs_alignments = self._align_by_strongs(
                source_tokens, target_words, aligned_source_indices, aligned_target_indices
            )
            alignments.extend(strongs_alignments)

        # Second pass: Morphological and semantic alignment
        morph_alignments = self._align_by_morphology(
            source_tokens, target_words, aligned_source_indices, aligned_target_indices
        )
        alignments.extend(morph_alignments)

        # Third pass: Positional heuristics for remaining tokens
        position_alignments = self._align_by_position(
            source_tokens, target_words, aligned_source_indices, aligned_target_indices
        )
        alignments.extend(position_alignments)

        # Collect unaligned tokens and words
        unaligned_source = [
            token for i, token in enumerate(source_tokens) if i not in aligned_source_indices
        ]
        unaligned_target = [
            word for i, word in enumerate(target_words) if i not in aligned_target_indices
        ]

        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(
            alignments, len(source_tokens), len(target_words)
        )

        # Create alignment result
        result = TokenAlignment(
            verse_id=verse_id,
            source_language=source_language,
            target_language=target_language,
            alignments=alignments,
            unaligned_source=unaligned_source,
            unaligned_target=unaligned_target,
            alignment_score=alignment_score,
        )

        # Cache result
        self._alignment_cache[cache_key] = result

        return result

    def _tokenize_target(self, text: str) -> List[str]:
        """Tokenize target language text."""
        import re

        # Split on whitespace and punctuation, but keep punctuation
        words = re.findall(r"\w+|[^\w\s]", text)
        return [w for w in words if w.strip()]

    def _align_by_strongs(
        self,
        source_tokens: List[ExtractedToken],
        target_words: List[str],
        aligned_source: Set[int],
        aligned_target: Set[int],
    ) -> List[AlignedToken]:
        """Align tokens using Strong's numbers."""
        alignments = []

        for i, token in enumerate(source_tokens):
            if i in aligned_source or not token.strong_number:
                continue

            # Look up possible English words for this Strong's number
            possible_words = self.strongs_mapping.get(token.strong_number, [])
            if not possible_words:
                continue

            # Find matching words in target
            matches = []
            for j, target_word in enumerate(target_words):
                if j in aligned_target:
                    continue

                # Case-insensitive matching
                if any(target_word.lower() == word.lower() for word in possible_words):
                    matches.append(j)

            if matches:
                # Create alignment
                alignment = AlignedToken(
                    source_tokens=[token],
                    target_words=[target_words[j] for j in matches],
                    alignment_type=(
                        AlignmentType.ONE_TO_MANY if len(matches) > 1 else AlignmentType.ONE_TO_ONE
                    ),
                    confidence=0.9,
                    notes="Strong's number match",
                )
                alignments.append(alignment)

                # Mark as aligned
                aligned_source.add(i)
                aligned_target.update(matches)

        return alignments

    def _align_by_morphology(
        self,
        source_tokens: List[ExtractedToken],
        target_words: List[str],
        aligned_source: Set[int],
        aligned_target: Set[int],
    ) -> List[AlignedToken]:
        """Align tokens using morphological analysis."""
        alignments = []

        # Look for proper nouns, pronouns, and other morphologically identifiable words
        for i, token in enumerate(source_tokens):
            if i in aligned_source or not token.morphology:
                continue

            # Proper nouns often transliterate
            if token.morphology.features.part_of_speech == "noun" and token.transliteration:
                # Look for similar sounding words in target
                for j, target_word in enumerate(target_words):
                    if j in aligned_target:
                        continue

                    # Simple similarity check
                    if self._is_similar_transliteration(token.transliteration, target_word):
                        alignment = AlignedToken(
                            source_tokens=[token],
                            target_words=[target_word],
                            alignment_type=AlignmentType.ONE_TO_ONE,
                            confidence=0.7,
                            notes="Transliteration match",
                        )
                        alignments.append(alignment)
                        aligned_source.add(i)
                        aligned_target.add(j)
                        break

        return alignments

    def _align_by_position(
        self,
        source_tokens: List[ExtractedToken],
        target_words: List[str],
        aligned_source: Set[int],
        aligned_target: Set[int],
    ) -> List[AlignedToken]:
        """Align remaining tokens by position heuristics."""
        alignments = []

        # Get unaligned indices
        unaligned_source_indices = [i for i in range(len(source_tokens)) if i not in aligned_source]
        unaligned_target_indices = [i for i in range(len(target_words)) if i not in aligned_target]

        if not unaligned_source_indices or not unaligned_target_indices:
            return alignments

        # Simple proportional alignment
        source_count = len(unaligned_source_indices)
        target_count = len(unaligned_target_indices)

        if source_count == target_count:
            # One-to-one alignment
            for src_idx, tgt_idx in zip(unaligned_source_indices, unaligned_target_indices):
                alignment = AlignedToken(
                    source_tokens=[source_tokens[src_idx]],
                    target_words=[target_words[tgt_idx]],
                    alignment_type=AlignmentType.ONE_TO_ONE,
                    confidence=0.5,
                    notes="Positional alignment",
                )
                alignments.append(alignment)
        elif source_count < target_count:
            # One source to many targets
            ratio = target_count / source_count
            for i, src_idx in enumerate(unaligned_source_indices):
                start_tgt = int(i * ratio)
                end_tgt = int((i + 1) * ratio)
                tgt_indices = unaligned_target_indices[start_tgt:end_tgt]

                if tgt_indices:
                    alignment = AlignedToken(
                        source_tokens=[source_tokens[src_idx]],
                        target_words=[target_words[idx] for idx in tgt_indices],
                        alignment_type=AlignmentType.ONE_TO_MANY,
                        confidence=0.4,
                        notes="Positional alignment (one-to-many)",
                    )
                    alignments.append(alignment)

        return alignments

    def _is_similar_transliteration(self, translit1: str, translit2: str) -> bool:
        """Check if two transliterations are similar."""
        # Simple edit distance check
        if not translit1 or not translit2:
            return False

        # Normalize
        t1 = translit1.lower().replace("'", "").replace("-", "")
        t2 = translit2.lower().replace("'", "").replace("-", "")

        # Exact match
        if t1 == t2:
            return True

        # Check if one starts with the other
        if t1.startswith(t2) or t2.startswith(t1):
            return True

        # Simple edit distance (Levenshtein)
        if len(t1) < 3 or len(t2) < 3:
            return False

        distance = self._edit_distance(t1, t2)
        max_len = max(len(t1), len(t2))

        # Allow up to 30% difference
        return distance / max_len <= 0.3

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _calculate_alignment_score(
        self, alignments: List[AlignedToken], total_source: int, total_target: int
    ) -> float:
        """Calculate overall alignment quality score."""
        if not alignments:
            return 0.0

        # Weighted average of confidence scores
        total_confidence = sum(align.confidence for align in alignments)
        avg_confidence = total_confidence / len(alignments)

        # Coverage factor
        aligned_source = sum(len(align.source_tokens) for align in alignments)
        aligned_target = sum(len(align.target_words) for align in alignments)

        source_coverage = aligned_source / total_source if total_source > 0 else 0
        target_coverage = aligned_target / total_target if total_target > 0 else 0
        coverage_factor = (source_coverage + target_coverage) / 2

        # Combined score
        return avg_confidence * coverage_factor
