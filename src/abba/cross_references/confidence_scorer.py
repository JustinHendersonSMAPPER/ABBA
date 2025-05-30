"""
Confidence scoring system for cross-references.

Provides sophisticated algorithms for calculating confidence scores
based on multiple factors including textual similarity, scholarly consensus,
and contextual evidence.
"""

import math
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

from .models import CrossReference, ReferenceType, ReferenceRelationship, ReferenceConfidence
from ..verse_id import VerseID


class ConfidenceScorer:
    """Calculates and validates confidence scores for cross-references."""

    def __init__(self):
        """Initialize the confidence scorer."""
        self.type_base_scores = self._build_type_base_scores()
        self.scholarly_weights = self._build_scholarly_weights()
        self.uncertainty_penalties = self._build_uncertainty_penalties()

    def _build_type_base_scores(self) -> Dict[ReferenceType, float]:
        """Build base confidence scores by reference type."""
        return {
            ReferenceType.DIRECT_QUOTE: 0.95,
            ReferenceType.PARTIAL_QUOTE: 0.85,
            ReferenceType.PARAPHRASE: 0.75,
            ReferenceType.PROPHECY_FULFILLMENT: 0.88,
            ReferenceType.TYPE_ANTITYPE: 0.75,
            ReferenceType.ALLUSION: 0.65,
            ReferenceType.THEMATIC_PARALLEL: 0.70,
            ReferenceType.STRUCTURAL_PARALLEL: 0.60,
            ReferenceType.HISTORICAL_PARALLEL: 0.65,
            ReferenceType.LITERARY_PARALLEL: 0.60,
            ReferenceType.DOCTRINAL_PARALLEL: 0.70,
            ReferenceType.EXPLANATION: 0.75,
            ReferenceType.ILLUSTRATION: 0.65,
            ReferenceType.CONTRAST: 0.60,
        }

    def _build_scholarly_weights(self) -> Dict[str, float]:
        """Build weights for different types of scholarly evidence."""
        return {
            "direct_quotation_marker": 0.20,  # "It is written", etc.
            "attribution": 0.15,  # "David says", etc.
            "lexical_overlap": 0.25,  # Shared vocabulary
            "semantic_similarity": 0.20,  # Conceptual overlap
            "contextual_support": 0.15,  # Surrounding context
            "structural_similarity": 0.05,  # Literary patterns
        }

    def _build_uncertainty_penalties(self) -> Dict[str, float]:
        """Build penalty factors for uncertainty indicators."""
        return {
            "low_text_similarity": 0.15,
            "no_lexical_overlap": 0.20,
            "weak_thematic_connection": 0.10,
            "controversial_reference": 0.25,
            "modern_scholarship_disputed": 0.20,
            "textual_variant_issues": 0.15,
            "chronological_problems": 0.30,
            "contextual_mismatch": 0.20,
            "forced_interpretation": 0.25,
        }

    def calculate_confidence(
        self,
        source_text: str,
        target_text: str,
        ref_type: ReferenceType,
        relationship: ReferenceRelationship,
        source_verse: VerseID,
        target_verse: VerseID,
        context: Optional[Dict] = None,
    ) -> ReferenceConfidence:
        """
        Calculate comprehensive confidence score for a cross-reference.

        Args:
            source_text: Text of the source verse
            target_text: Text of the target verse
            ref_type: Type of reference
            relationship: Directional relationship
            source_verse: Source verse ID
            target_verse: Target verse ID
            context: Optional context information

        Returns:
            Complete confidence assessment
        """
        context = context or {}

        # Calculate component scores
        textual_similarity = self._calculate_textual_similarity(source_text, target_text)
        thematic_similarity = self._calculate_thematic_similarity(source_text, target_text)
        structural_similarity = self._calculate_structural_similarity(source_text, target_text)
        scholarly_consensus = self._calculate_scholarly_consensus(
            ref_type, relationship, source_verse, target_verse, context
        )

        # Calculate supporting evidence metrics
        lexical_links = self._count_lexical_links(source_text, target_text)
        semantic_links = self._count_semantic_links(source_text, target_text)
        contextual_support = self._assess_contextual_support(context)

        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(
            textual_similarity, thematic_similarity, ref_type, source_verse, target_verse, context
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            textual_similarity,
            thematic_similarity,
            structural_similarity,
            scholarly_consensus,
            ref_type,
            uncertainty_factors,
        )

        return ReferenceConfidence(
            overall_score=overall_score,
            textual_similarity=textual_similarity,
            thematic_similarity=thematic_similarity,
            structural_similarity=structural_similarity,
            scholarly_consensus=scholarly_consensus,
            uncertainty_factors=uncertainty_factors,
            lexical_links=lexical_links,
            semantic_links=semantic_links,
            contextual_support=contextual_support,
        )

    def _calculate_textual_similarity(self, text1: str, text2: str) -> float:
        """Calculate textual similarity between passages."""
        # Normalize texts
        text1_clean = self._normalize_text(text1)
        text2_clean = self._normalize_text(text2)

        # Word-level similarity
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0

        # Character-level similarity using longest common subsequence
        lcs_ratio = self._lcs_similarity(text1_clean, text2_clean)

        # Combine metrics
        return (jaccard * 0.6) + (lcs_ratio * 0.4)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _lcs_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity using longest common subsequence."""
        if not s1 or not s2:
            return 0.0

        # Simple LCS implementation
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return (2 * lcs_length) / (m + n)

    def _calculate_thematic_similarity(self, text1: str, text2: str) -> float:
        """Calculate thematic/conceptual similarity."""
        # Define thematic word groups
        thematic_groups = {
            "salvation": ["save", "salvation", "savior", "rescue", "redeem", "deliver"],
            "judgment": ["judge", "judgment", "wrath", "condemn", "punish", "destroy"],
            "love": ["love", "beloved", "dear", "cherish", "compassion", "mercy"],
            "covenant": ["covenant", "promise", "oath", "agreement", "bond"],
            "worship": ["worship", "praise", "honor", "glorify", "exalt", "magnify"],
            "sin": ["sin", "iniquity", "transgression", "wickedness", "evil"],
            "holy": ["holy", "sacred", "pure", "righteous", "sanctify"],
            "kingdom": ["kingdom", "king", "reign", "rule", "throne", "dominion"],
            "peace": ["peace", "rest", "calm", "tranquil", "still"],
            "light": ["light", "brightness", "shine", "illuminate", "lamp"],
        }

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        shared_themes = 0
        total_themes = 0

        for theme, words in thematic_groups.items():
            has_theme1 = any(word in text1_lower for word in words)
            has_theme2 = any(word in text2_lower for word in words)

            if has_theme1 or has_theme2:
                total_themes += 1
                if has_theme1 and has_theme2:
                    shared_themes += 1

        return shared_themes / total_themes if total_themes > 0 else 0.0

    def _calculate_structural_similarity(self, text1: str, text2: str) -> float:
        """Calculate structural/literary similarity."""
        # Check for similar structural patterns
        patterns = [
            (r"blessed (?:is|are)", "beatitude"),
            (r"woe (?:to|unto)", "woe_saying"),
            (r"the lord is", "divine_declaration"),
            (r"in the beginning", "creation_formula"),
            (r"it came to pass", "narrative_formula"),
            (r"thus says the lord", "prophetic_formula"),
            (r"hear (?:o|the)", "call_to_attention"),
            (r"behold", "attention_getter"),
            (r"verily|truly", "truth_formula"),
        ]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        matches = 0
        total_patterns = 0

        for pattern, pattern_type in patterns:
            import re

            has_pattern1 = bool(re.search(pattern, text1_lower))
            has_pattern2 = bool(re.search(pattern, text2_lower))

            if has_pattern1 or has_pattern2:
                total_patterns += 1
                if has_pattern1 and has_pattern2:
                    matches += 1

        return matches / total_patterns if total_patterns > 0 else 0.0

    def _calculate_scholarly_consensus(
        self,
        ref_type: ReferenceType,
        relationship: ReferenceRelationship,
        source_verse: VerseID,
        target_verse: VerseID,
        context: Dict,
    ) -> float:
        """Calculate scholarly consensus score."""
        # Base score by reference type
        base_score = self.type_base_scores.get(ref_type, 0.5)

        # Adjustments based on reference characteristics

        # Direct quotations have highest consensus
        if ref_type in [ReferenceType.DIRECT_QUOTE, ReferenceType.PARTIAL_QUOTE]:
            base_score = min(0.95, base_score + 0.1)

        # Well-established patterns
        if self._is_well_established_pattern(source_verse, target_verse):
            base_score = min(0.98, base_score + 0.15)

        # NT quoting OT is generally well-accepted
        if self._is_nt_quoting_ot(source_verse, target_verse):
            base_score = min(0.95, base_score + 0.05)

        # Controversial references get penalty
        if context.get("controversial", False):
            base_score *= 0.8

        return base_score

    def _count_lexical_links(self, text1: str, text2: str) -> int:
        """Count significant lexical connections."""
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())

        # Filter out very common words
        stop_words = {
            "the",
            "and",
            "of",
            "to",
            "a",
            "in",
            "is",
            "it",
            "you",
            "that",
            "he",
            "was",
            "for",
            "on",
            "are",
            "as",
            "with",
            "his",
            "they",
            "i",
            "at",
            "be",
            "this",
            "have",
            "from",
            "or",
            "one",
            "had",
            "by",
            "word",
            "but",
            "not",
            "what",
            "all",
            "were",
            "we",
            "when",
        }

        words1 = words1 - stop_words
        words2 = words2 - stop_words

        return len(words1 & words2)

    def _count_semantic_links(self, text1: str, text2: str) -> int:
        """Count semantic/conceptual links."""
        # This is a simplified version - in practice would use
        # semantic embeddings or more sophisticated NLP

        semantic_pairs = [
            ("save", "salvation"),
            ("king", "kingdom"),
            ("holy", "sanctify"),
            ("love", "beloved"),
            ("light", "lamp"),
            ("shepherd", "flock"),
            ("judge", "judgment"),
            ("sin", "iniquity"),
            ("peace", "rest"),
            ("praise", "worship"),
            ("promise", "covenant"),
            ("word", "speak"),
        ]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        links = 0
        for word1, word2 in semantic_pairs:
            if (word1 in text1_lower and word2 in text2_lower) or (
                word2 in text1_lower and word1 in text2_lower
            ):
                links += 1

        return links

    def _assess_contextual_support(self, context: Dict) -> int:
        """Assess strength of contextual support (0-5 scale)."""
        support = 0

        # Quote markers boost confidence
        if context.get("has_quote_marker", False):
            support += 2

        # Attribution present
        if context.get("has_attribution", False):
            support += 1

        # Thematic consistency with surrounding context
        if context.get("thematic_consistency", 0) > 0.7:
            support += 1

        # Multiple cross-references in vicinity
        if context.get("nearby_references", 0) > 2:
            support += 1

        return min(5, support)

    def _identify_uncertainty_factors(
        self,
        textual_sim: float,
        thematic_sim: float,
        ref_type: ReferenceType,
        source_verse: VerseID,
        target_verse: VerseID,
        context: Dict,
    ) -> List[str]:
        """Identify factors that reduce confidence."""
        factors = []

        # Low similarity scores
        if textual_sim < 0.3:
            factors.append("low_text_similarity")

        if thematic_sim < 0.4:
            factors.append("weak_thematic_connection")

        # Type-specific uncertainty
        if ref_type == ReferenceType.ALLUSION and textual_sim < 0.5:
            factors.append("weak_allusion_evidence")

        # Chronological issues
        if self._has_chronological_problems(source_verse, target_verse):
            factors.append("chronological_problems")

        # Contextual mismatch
        if context.get("contextual_mismatch", False):
            factors.append("contextual_mismatch")

        # Scholarly disputes
        if context.get("scholarly_disputed", False):
            factors.append("modern_scholarship_disputed")

        # Textual variants
        if context.get("textual_variants", False):
            factors.append("textual_variant_issues")

        return factors

    def _calculate_overall_score(
        self,
        textual_sim: float,
        thematic_sim: float,
        structural_sim: float,
        scholarly_consensus: float,
        ref_type: ReferenceType,
        uncertainty_factors: List[str],
    ) -> float:
        """Calculate overall confidence score."""
        # Component weights
        weights = {"textual": 0.35, "thematic": 0.25, "structural": 0.15, "scholarly": 0.25}

        # Calculate weighted score
        base_score = (
            textual_sim * weights["textual"]
            + thematic_sim * weights["thematic"]
            + structural_sim * weights["structural"]
            + scholarly_consensus * weights["scholarly"]
        )

        # Apply uncertainty penalties
        penalty = 0.0
        for factor in uncertainty_factors:
            penalty += self.uncertainty_penalties.get(factor, 0.05)

        # Cap total penalty
        penalty = min(penalty, 0.4)

        final_score = max(0.0, base_score - penalty)

        return min(1.0, final_score)

    def _is_well_established_pattern(self, source_verse: VerseID, target_verse: VerseID) -> bool:
        """Check if this is a well-established reference pattern."""
        # Some well-known reference patterns
        established_patterns = [
            ("MAT", "ISA"),  # Matthew quoting Isaiah
            ("ROM", "PSA"),  # Romans quoting Psalms
            ("HEB", "PSA"),  # Hebrews quoting Psalms
            ("ACT", "PSA"),  # Acts quoting Psalms
            ("1PE", "ISA"),  # 1 Peter quoting Isaiah
        ]

        source_target = (source_verse.book, target_verse.book)
        return source_target in established_patterns

    def _is_nt_quoting_ot(self, source_verse: VerseID, target_verse: VerseID) -> bool:
        """Check if NT verse is quoting OT verse."""
        ot_books = {
            "GEN",
            "EXO",
            "LEV",
            "NUM",
            "DEU",
            "JOS",
            "JDG",
            "RUT",
            "1SA",
            "2SA",
            "1KI",
            "2KI",
            "1CH",
            "2CH",
            "EZR",
            "NEH",
            "EST",
            "JOB",
            "PSA",
            "PRO",
            "ECC",
            "SNG",
            "ISA",
            "JER",
            "LAM",
            "EZK",
            "DAN",
            "HOS",
            "JOL",
            "AMO",
            "OBA",
            "JON",
            "MIC",
            "NAM",
            "HAB",
            "ZEP",
            "HAG",
            "ZEC",
            "MAL",
        }

        nt_books = {
            "MAT",
            "MRK",
            "LUK",
            "JHN",
            "ACT",
            "ROM",
            "1CO",
            "2CO",
            "GAL",
            "EPH",
            "PHP",
            "COL",
            "1TH",
            "2TH",
            "1TI",
            "2TI",
            "TIT",
            "PHM",
            "HEB",
            "JAS",
            "1PE",
            "2PE",
            "1JN",
            "2JN",
            "3JN",
            "JUD",
            "REV",
        }

        return source_verse.book in nt_books and target_verse.book in ot_books

    def _has_chronological_problems(self, source_verse: VerseID, target_verse: VerseID) -> bool:
        """Check for chronological inconsistencies."""
        # Simplified check - in practice would need more sophisticated dating

        # Check if later OT book seems to quote earlier OT book
        late_ot_books = ["1CH", "2CH", "EZR", "NEH", "EST", "DAN"]
        early_ot_books = ["GEN", "EXO", "LEV", "NUM", "DEU"]

        if (
            source_verse.book in late_ot_books
            and target_verse.book in early_ot_books
            and source_verse.book != target_verse.book
        ):
            # This might be fine (later books can quote earlier)
            return False

        # More sophisticated chronological analysis would go here
        return False

    def validate_confidence_score(self, confidence: ReferenceConfidence) -> Tuple[bool, List[str]]:
        """Validate a confidence score for consistency and reasonableness."""
        issues = []

        # Check score ranges
        if not (0.0 <= confidence.overall_score <= 1.0):
            issues.append("Overall score out of range [0, 1]")

        for attr in [
            "textual_similarity",
            "thematic_similarity",
            "structural_similarity",
            "scholarly_consensus",
        ]:
            value = getattr(confidence, attr)
            if not (0.0 <= value <= 1.0):
                issues.append(f"{attr} out of range [0, 1]")

        # Check consistency
        component_avg = (
            confidence.textual_similarity
            + confidence.thematic_similarity
            + confidence.structural_similarity
            + confidence.scholarly_consensus
        ) / 4

        if abs(confidence.overall_score - component_avg) > 0.3:
            issues.append("Overall score inconsistent with component scores")

        # Check supporting evidence reasonableness
        if confidence.lexical_links < 0 or confidence.lexical_links > 50:
            issues.append("Lexical links count unreasonable")

        if confidence.semantic_links < 0 or confidence.semantic_links > 20:
            issues.append("Semantic links count unreasonable")

        if not (0 <= confidence.contextual_support <= 5):
            issues.append("Contextual support out of expected range [0, 5]")

        return len(issues) == 0, issues
