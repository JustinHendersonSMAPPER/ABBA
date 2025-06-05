"""
Reference type classifier for biblical cross-references.

Automatically classifies cross-references based on textual analysis,
semantic similarity, and contextual factors.
"""

import re
import difflib
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter

from .models import ReferenceType, ReferenceRelationship
from ..verse_id import VerseID


class ReferenceTypeClassifier:
    """Classifies cross-references into appropriate types and relationships."""

    def __init__(self):
        """Initialize the classifier with linguistic patterns."""
        self.quotation_markers = self._build_quotation_markers()
        self.prophetic_phrases = self._build_prophetic_phrases()
        self.thematic_keywords = self._build_thematic_keywords()
        self.stop_words = self._build_stop_words()

    def _build_quotation_markers(self) -> List[str]:
        """Build list of phrases that indicate quotations."""
        return [
            "it is written",
            "the scripture says",
            "as it says",
            "the word of the lord",
            "thus says the lord",
            "the lord says",
            "scripture",
            "written",
            "spoke",
            "said",
        ]

    def _build_prophetic_phrases(self) -> List[str]:
        """Build list of phrases indicating prophetic fulfillment."""
        return [
            "that it might be fulfilled",
            "fulfilled",
            "fulfilling",
            "accomplish",
            "came to pass",
            "according to",
            "as foretold",
            "predicted",
            "prophesied",
        ]

    def _build_thematic_keywords(self) -> Dict[str, List[str]]:
        """Build thematic keyword groups."""
        return {
            "salvation": ["save", "salvation", "savior", "rescue", "redeem", "redemption"],
            "judgment": ["judge", "judgment", "wrath", "punish", "condemn", "destruction"],
            "covenant": ["covenant", "promise", "oath", "agreement", "testament"],
            "temple": ["temple", "sanctuary", "tabernacle", "holy place", "house of god"],
            "sacrifice": ["sacrifice", "offering", "blood", "altar", "priest"],
            "kingship": ["king", "kingdom", "throne", "crown", "reign", "rule"],
            "shepherd": ["shepherd", "sheep", "flock", "pasture", "fold"],
            "light": ["light", "darkness", "lamp", "shine", "illuminate"],
            "water": ["water", "river", "fountain", "well", "spring", "stream"],
            "bread": ["bread", "food", "eat", "feast", "hunger", "feed"],
            "love": ["love", "loved", "loving", "beloved", "charity"],
            "neighbor": ["neighbor", "neighbour", "fellow", "brother", "sister"],
        }

    def _build_stop_words(self) -> Set[str]:
        """Build set of stop words to ignore in similarity calculations."""
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "shall",
            "have",
            "had",
            "his",
            "her",
            "him",
            "them",
            "they",
            "their",
            "i",
            "me",
            "my",
            "you",
            "your",
            "do",
            "unto",
            "would",
        }
    
    def classify(
        self,
        source_text: str,
        target_text: str,
        source_verse: VerseID,
        target_verse: VerseID
    ) -> ReferenceType:
        """
        Simplified classify method for compatibility.
        
        Args:
            source_text: Text of the source verse
            target_text: Text of the target verse
            source_verse: VerseID of source
            target_verse: VerseID of target
            
        Returns:
            ReferenceType classification
        """
        ref_type, _ = self.classify_reference(
            source_text, target_text, source_verse, target_verse
        )
        return ref_type

    def classify_reference(
        self,
        source_text: str,
        target_text: str,
        source_verse: VerseID,
        target_verse: VerseID,
        context: Optional[str] = None,
    ) -> Tuple[ReferenceType, ReferenceRelationship]:
        """
        Classify a cross-reference based on textual analysis.

        Args:
            source_text: Text of the source verse
            target_text: Text of the target verse
            source_verse: VerseID of source
            target_verse: VerseID of target
            context: Optional surrounding context

        Returns:
            Tuple of (ReferenceType, ReferenceRelationship)
        """
        # Calculate text similarity
        similarity_score = self._calculate_text_similarity(source_text, target_text)

        # Check for direct quotations
        if self._is_direct_quotation(source_text, target_text, similarity_score):
            relationship = self._determine_quotation_direction(source_verse, target_verse)
            if similarity_score > 0.9:
                return ReferenceType.DIRECT_QUOTE, relationship
            elif similarity_score > 0.7:
                return ReferenceType.PARTIAL_QUOTE, relationship
            else:
                return ReferenceType.PARAPHRASE, relationship

        # Check for prophetic fulfillment
        if self._is_prophetic_fulfillment(
            source_text, target_text, context, source_verse, target_verse
        ):
            if self._is_ot_to_nt(target_verse, source_verse):
                return ReferenceType.PROPHECY_FULFILLMENT, ReferenceRelationship.FULFILLED_BY
            else:
                return ReferenceType.PROPHECY_FULFILLMENT, ReferenceRelationship.FULFILLS

        # Check for typological relationships
        if self._is_typological(source_text, target_text, source_verse, target_verse):
            if self._is_ot_to_nt(target_verse, source_verse):
                return ReferenceType.TYPE_ANTITYPE, ReferenceRelationship.FULFILLED_BY
            else:
                return ReferenceType.TYPE_ANTITYPE, ReferenceRelationship.FULFILLS

        # Check for allusions
        if self._is_allusion(source_text, target_text, similarity_score):
            relationship = self._determine_allusion_direction(source_verse, target_verse)
            return ReferenceType.ALLUSION, relationship

        # Check for thematic parallels
        thematic_strength = self._calculate_thematic_similarity(source_text, target_text)
        if thematic_strength > 0.4:
            return ReferenceType.THEMATIC_PARALLEL, ReferenceRelationship.PARALLELS

        # Check for structural parallels
        if self._is_structural_parallel(source_text, target_text):
            return ReferenceType.STRUCTURAL_PARALLEL, ReferenceRelationship.PARALLELS

        # Check for historical parallels
        if self._is_historical_parallel(source_verse, target_verse):
            return ReferenceType.HISTORICAL_PARALLEL, ReferenceRelationship.PARALLELS

        # Default to thematic parallel
        return ReferenceType.THEMATIC_PARALLEL, ReferenceRelationship.PARALLELS

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Normalize texts
        text1_clean = self._normalize_text(text1)
        text2_clean = self._normalize_text(text2)

        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(None, text1_clean, text2_clean).ratio()

        # Also check word overlap
        words1 = set(text1_clean.split()) - self.stop_words
        words2 = set(text2_clean.split()) - self.stop_words

        if len(words1) == 0 or len(words2) == 0:
            return similarity

        word_overlap = len(words1 & words2) / len(words1 | words2)

        # Combine similarity measures
        return (similarity * 0.6) + (word_overlap * 0.4)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and extra whitespace
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _is_direct_quotation(self, source_text: str, target_text: str, similarity: float) -> bool:
        """Check if source directly quotes target."""
        # High similarity indicates quotation
        if similarity > 0.6:
            return True

        # Check for quotation markers in context
        combined_text = f"{source_text} {target_text}".lower()
        for marker in self.quotation_markers:
            if marker in combined_text:
                return True

        return False

    def _determine_quotation_direction(
        self, source_verse: VerseID, target_verse: VerseID
    ) -> ReferenceRelationship:
        """Determine direction of quotation relationship."""
        # NT typically quotes OT
        if self._is_ot_to_nt(target_verse, source_verse):
            return ReferenceRelationship.QUOTES
        elif self._is_ot_to_nt(source_verse, target_verse):
            return ReferenceRelationship.QUOTED_BY
        else:
            # Same testament - use chronological order heuristic
            if self._is_chronologically_later(source_verse, target_verse):
                return ReferenceRelationship.QUOTES
            else:
                return ReferenceRelationship.QUOTED_BY

    def _is_ot_to_nt(self, ot_verse: VerseID, nt_verse: VerseID) -> bool:
        """Check if one verse is OT and other is NT."""
        ot_books = [
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
        ]

        return ot_verse.book in ot_books and nt_verse.book not in ot_books

    def _is_chronologically_later(self, verse1: VerseID, verse2: VerseID) -> bool:
        """Rough chronological ordering heuristic."""
        book_order = [
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
        ]

        try:
            pos1 = book_order.index(verse1.book)
            pos2 = book_order.index(verse2.book)
            return pos1 > pos2
        except ValueError:
            return False

    def _is_prophetic_fulfillment(
        self,
        source_text: str,
        target_text: str,
        context: Optional[str],
        source_verse: VerseID,
        target_verse: VerseID,
    ) -> bool:
        """Check if this represents prophetic fulfillment."""
        combined_text = f"{source_text} {target_text}"
        if context:
            combined_text += f" {context}"

        combined_text = combined_text.lower()

        # Check for fulfillment phrases
        for phrase in self.prophetic_phrases:
            if phrase in combined_text:
                return True

        # Check if OT prophetic book to NT
        prophetic_books = [
            "ISA",
            "JER",
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
        ]

        if (
            target_verse.book in prophetic_books
            and source_verse.book not in prophetic_books
            and self._is_ot_to_nt(target_verse, source_verse)
        ):
            return True

        return False

    def _is_typological(
        self, source_text: str, target_text: str, source_verse: VerseID, target_verse: VerseID
    ) -> bool:
        """Check for typological relationships."""
        # Check for type/antitype keywords
        typological_terms = [
            "shadow",
            "type",
            "pattern",
            "copy",
            "example",
            "figure",
            "prefigure",
            "foreshadow",
            "picture",
            "symbol",
        ]

        combined_text = f"{source_text} {target_text}".lower()
        for term in typological_terms:
            if term in combined_text:
                return True

        # Certain OT to NT patterns are typically typological
        typological_patterns = [
            ("GEN", ["HEB", "ROM"]),  # Abraham, Isaac typology
            ("EXO", ["1CO", "HEB"]),  # Exodus typology
            ("LEV", ["HEB"]),  # Levitical typology
            ("PSA", ["HEB"]),  # Psalmic typology
        ]

        for ot_book, nt_books in typological_patterns:
            if (target_verse.book == ot_book and source_verse.book in nt_books) or (
                source_verse.book == ot_book and target_verse.book in nt_books
            ):
                return True

        return False

    def _is_allusion(self, source_text: str, target_text: str, similarity: float) -> bool:
        """Check if this is an allusion (indirect reference)."""
        # Moderate similarity suggests allusion
        if 0.3 <= similarity <= 0.6:
            return True

        # Check for subtle word echoes
        words1 = set(self._normalize_text(source_text).split()) - self.stop_words
        words2 = set(self._normalize_text(target_text).split()) - self.stop_words

        if len(words1) > 0 and len(words2) > 0:
            overlap_ratio = len(words1 & words2) / min(len(words1), len(words2))
            if 0.2 <= overlap_ratio <= 0.5:
                return True

        return False

    def _determine_allusion_direction(
        self, source_verse: VerseID, target_verse: VerseID
    ) -> ReferenceRelationship:
        """Determine direction of allusion."""
        if self._is_ot_to_nt(target_verse, source_verse):
            return ReferenceRelationship.ALLUDES_TO
        elif self._is_ot_to_nt(source_verse, target_verse):
            return ReferenceRelationship.ALLUDED_BY
        else:
            return ReferenceRelationship.PARALLELS

    def _calculate_thematic_similarity(self, text1: str, text2: str) -> float:
        """Calculate thematic similarity based on keyword groups."""
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        theme_scores = []

        for theme, keywords in self.thematic_keywords.items():
            count1 = sum(1 for keyword in keywords if keyword in text1_lower)
            count2 = sum(1 for keyword in keywords if keyword in text2_lower)

            if count1 > 0 and count2 > 0:
                # Both texts have this theme
                theme_score = min(count1, count2) / max(count1, count2)
                theme_scores.append(theme_score)

        if not theme_scores:
            return 0.0

        return sum(theme_scores) / len(theme_scores)

    def _is_structural_parallel(self, text1: str, text2: str) -> bool:
        """Check for structural/literary parallels."""
        # Check for similar sentence structures
        structures = [
            (r"blessed is|are", "beatitude"),
            (r"woe to|unto", "woe_oracle"),
            (r"the lord is my", "possession_statement"),
            (r"in the beginning", "creation_formula"),
            (r"it came to pass", "narrative_formula"),
            (r"thus says the lord", "prophetic_formula"),
        ]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        for pattern, structure_type in structures:
            if re.search(pattern, text1_lower) and re.search(pattern, text2_lower):
                return True

        return False

    def _is_historical_parallel(self, verse1: VerseID, verse2: VerseID) -> bool:
        """Check for historical parallel events."""
        # Historical books that might have parallel accounts
        historical_parallels = [
            (["1SA", "2SA", "1KI", "2KI"], ["1CH", "2CH"]),  # Chronicles parallels
            (["1KI", "2KI"], ["2CH"]),  # Kings/Chronicles
            (["MAT", "MRK", "LUK"], ["MAT", "MRK", "LUK"]),  # Synoptic Gospels
        ]

        for group1, group2 in historical_parallels:
            if verse1.book in group1 and verse2.book in group2:
                return True
            if verse1.book in group2 and verse2.book in group1:
                return True

        return False

    def analyze_reference_confidence(
        self,
        source_text: str,
        target_text: str,
        ref_type: ReferenceType,
        relationship: ReferenceRelationship,
    ) -> Dict[str, float]:
        """Analyze confidence factors for a classified reference."""
        metrics = {}

        # Text similarity metrics
        metrics["textual_similarity"] = self._calculate_text_similarity(source_text, target_text)
        metrics["thematic_similarity"] = self._calculate_thematic_similarity(
            source_text, target_text
        )

        # Type-specific confidence adjustments
        if ref_type == ReferenceType.DIRECT_QUOTE:
            metrics["scholarly_consensus"] = 0.95
            metrics["structural_similarity"] = metrics["textual_similarity"]
        elif ref_type == ReferenceType.PROPHECY_FULFILLMENT:
            metrics["scholarly_consensus"] = 0.85
            metrics["structural_similarity"] = 0.7
        elif ref_type == ReferenceType.ALLUSION:
            metrics["scholarly_consensus"] = 0.7
            metrics["structural_similarity"] = 0.5
        else:
            metrics["scholarly_consensus"] = 0.8
            metrics["structural_similarity"] = 0.6

        # Overall confidence
        weights = {
            "textual_similarity": 0.3,
            "thematic_similarity": 0.3,
            "structural_similarity": 0.2,
            "scholarly_consensus": 0.2,
        }

        metrics["overall_score"] = sum(metrics[key] * weight for key, weight in weights.items())

        return metrics