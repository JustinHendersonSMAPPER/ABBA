"""
Citation tracker for identifying Old Testament quotes in the New Testament.

Provides sophisticated algorithms for detecting quotations, allusions,
and paraphrases of Old Testament texts in New Testament writings.
"""

import re
import difflib
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass

from .models import (
    CrossReference,
    ReferenceType,
    ReferenceRelationship,
    CitationMatch,
    ReferenceConfidence,
)
from ..verse_id import VerseID, parse_verse_id
from .classifier import ReferenceTypeClassifier


@dataclass
class QuoteAnalysis:
    """Analysis of a potential quotation."""

    similarity_score: float
    word_matches: List[str]
    phrase_matches: List[str]
    missing_words: List[str]
    added_words: List[str]
    word_order_changes: int

    # Quotation characteristics
    is_exact_quote: bool
    is_partial_quote: bool
    is_paraphrase: bool
    has_insertions: bool
    has_omissions: bool

    # Context factors
    introduces_quote: bool = False  # "It is written", etc.
    attribution_present: bool = False  # "David says", etc.

    def get_confidence_score(self) -> float:
        """Calculate overall confidence this is a quotation."""
        base_score = self.similarity_score

        # Boost for explicit quote markers
        if self.introduces_quote:
            base_score += 0.15
        if self.attribution_present:
            base_score += 0.1

        # Adjust for quotation characteristics
        if self.is_exact_quote:
            base_score += 0.1
        elif self.has_insertions and self.has_omissions:
            base_score -= 0.05  # Mixed changes reduce confidence

        return min(1.0, base_score)


class CitationTracker:
    """Tracks and identifies biblical citations and quotations."""

    def __init__(self):
        """Initialize the citation tracker."""
        self.classifier = ReferenceTypeClassifier()
        self.quote_introducers = self._build_quote_introducers()
        self.attribution_patterns = self._build_attribution_patterns()
        self.lxx_variants = self._build_lxx_variants()

    def _build_quote_introducers(self) -> List[str]:
        """Build list of phrases that introduce quotations."""
        return [
            "it is written",
            "the scripture says",
            "the scriptures say",
            "as it is written",
            "for it is written",
            "scripture says",
            "the word of the lord",
            "the lord says",
            "says the lord",
            "thus says the lord",
            "the holy spirit says",
            "god says",
            "the scripture",
            "according to the scriptures",
            "fulfilling what was spoken",
            "that it might be fulfilled",
            "to fulfill",
        ]

    def _build_attribution_patterns(self) -> List[str]:
        """Build patterns for explicit attributions."""
        return [
            r"david says?",
            r"moses says?",
            r"isaiah says?",
            r"jeremiah says?",
            r"the prophet says?",
            r"through the prophet",
            r"by the prophet",
            r"as .+ spoke",
            r"as .+ said",
            r".+ testified",
            r"in .+ we read",
        ]

    def _build_lxx_variants(self) -> Dict[str, List[str]]:
        """Build common Septuagint variant readings."""
        # NT often quotes LXX which differs from Hebrew
        return {
            "lord": ["lord", "god", "yahweh", "jehovah"],
            "mercy": ["mercy", "lovingkindness", "steadfast love", "chesed"],
            "righteousness": ["righteousness", "justice"],
            "salvation": ["salvation", "deliverance", "victory"],
            "glory": ["glory", "honor", "weight"],
        }

    def find_ot_quotes_in_nt(
        self, ot_verses: Dict[str, str], nt_verses: Dict[str, str]
    ) -> List[CitationMatch]:
        """
        Find Old Testament quotations in New Testament verses.

        Args:
            ot_verses: Dictionary mapping verse IDs to text
            nt_verses: Dictionary mapping verse IDs to text

        Returns:
            List of identified citation matches
        """
        citations = []

        for nt_id, nt_text in nt_verses.items():
            nt_verse = parse_verse_id(nt_id)

            # Skip if parsing failed or not NT
            if not nt_verse or not self._is_nt_book(nt_verse.book):
                continue

            potential_quotes = self._extract_potential_quotes(nt_text)

            for quote_text, context in potential_quotes:
                # Search for matching OT passages
                matches = self._find_ot_matches(quote_text, ot_verses)

                for ot_id, ot_text, analysis in matches:
                    ot_verse = parse_verse_id(ot_id)

                    if (
                        ot_verse and analysis.get_confidence_score() > 0.6
                    ):  # Threshold for citations
                        citation = CitationMatch(
                            source_verse=nt_verse,
                            target_verse=ot_verse,
                            source_text=quote_text,
                            target_text=ot_text,
                            match_type=self._determine_quote_type(analysis),
                            text_similarity=analysis.similarity_score,
                            word_matches=analysis.word_matches,
                            source_context=context,
                            discovered_by="citation_tracker",
                        )
                        citations.append(citation)

        return citations

    def _is_nt_book(self, book_code: str) -> bool:
        """Check if book code is New Testament."""
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
        return book_code in nt_books

    def _extract_potential_quotes(self, text: str) -> List[Tuple[str, str]]:
        """Extract potential quotations from NT text."""
        quotes = []
        text_lower = text.lower()

        # Look for explicit quote introducers
        for introducer in self.quote_introducers:
            if introducer in text_lower:
                # Extract text after the introducer
                start_pos = text_lower.find(introducer) + len(introducer)
                quote_part = text[start_pos:].strip()

                # Clean up punctuation and get substantial quotes
                quote_part = re.sub(r'^[\\s,:"]+', "", quote_part)
                if len(quote_part.split()) >= 4:  # At least 4 words
                    quotes.append((quote_part, text))

        # Look for quotation marks or other indicators
        quote_patterns = [
            r'"([^"]{20,})"',  # Text in quotes (at least 20 chars)
            r"'([^']{20,})'",  # Text in single quotes
        ]

        for pattern in quote_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                quote_text = match.group(1)
                if len(quote_text.split()) >= 4:
                    quotes.append((quote_text, text))

        # If no explicit markers, check for biblical language patterns
        if not quotes:
            biblical_patterns = [
                r"(blessed (?:is|are) .{20,})",
                r"(the lord .{20,})",
                r"(thus says .{20,})",
                r"(behold .{20,})",
                r"(verily .{20,})",
                r"(woe (?:to|unto) .{20,})",
            ]

            for pattern in biblical_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    quote_text = match.group(1)
                    if len(quote_text.split()) >= 4:
                        quotes.append((quote_text, text))

        # If still no quotes, treat the whole verse as potential quote
        if not quotes and len(text.split()) >= 6:
            quotes.append((text, text))

        return quotes

    def _find_ot_matches(
        self, quote_text: str, ot_verses: Dict[str, str]
    ) -> List[Tuple[str, str, QuoteAnalysis]]:
        """Find Old Testament verses that match the quote."""
        matches = []
        quote_clean = self._normalize_for_comparison(quote_text)
        quote_words = set(quote_clean.split())

        for ot_id, ot_text in ot_verses.items():
            ot_clean = self._normalize_for_comparison(ot_text)
            ot_words = set(ot_clean.split())

            # Quick word overlap filter
            word_overlap = len(quote_words & ot_words)
            if word_overlap < 3:  # At least 3 common words
                continue

            # Detailed analysis
            analysis = self._analyze_quote_similarity(quote_text, ot_text)

            if analysis.similarity_score > 0.3:  # Minimum threshold
                matches.append((ot_id, ot_text, analysis))

        # Sort by confidence score
        matches.sort(key=lambda x: x[2].get_confidence_score(), reverse=True)

        # Return top matches
        return matches[:10]

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for quotation comparison."""
        # Convert to lowercase
        text = text.lower()

        # Handle common spelling variations
        text = re.sub(r"\bsaith\b", "says", text)
        text = re.sub(r"\byea\b", "yes", text)
        text = re.sub(r"\bverily\b", "truly", text)
        text = re.sub(r"\bbehold\b", "see", text)

        # Remove punctuation but keep word boundaries
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _analyze_quote_similarity(self, quote_text: str, ot_text: str) -> QuoteAnalysis:
        """Perform detailed analysis of quotation similarity."""
        # Normalize texts
        quote_norm = self._normalize_for_comparison(quote_text)
        ot_norm = self._normalize_for_comparison(ot_text)

        quote_words = quote_norm.split()
        ot_words = ot_norm.split()

        # Calculate basic similarity
        similarity = difflib.SequenceMatcher(None, quote_norm, ot_norm).ratio()

        # Find word matches
        quote_word_set = set(quote_words)
        ot_word_set = set(ot_words)
        word_matches = list(quote_word_set & ot_word_set)

        # Find phrase matches (2+ consecutive words)
        phrase_matches = self._find_phrase_matches(quote_words, ot_words)

        # Identify missing and added words
        missing_words = list(ot_word_set - quote_word_set)
        added_words = list(quote_word_set - ot_word_set)

        # Check word order changes
        word_order_changes = self._count_word_order_changes(quote_words, ot_words)

        # Determine quotation characteristics
        is_exact_quote = similarity > 0.9 and len(missing_words) == 0 and len(added_words) == 0
        is_partial_quote = similarity > 0.7 and len(phrase_matches) > 0
        is_paraphrase = 0.4 < similarity <= 0.7
        has_insertions = len(added_words) > 0
        has_omissions = len(missing_words) > 0

        # Check for quote introducers in context
        quote_lower = quote_text.lower()
        introduces_quote = any(intro in quote_lower for intro in self.quote_introducers)

        # Check for attributions
        attribution_present = any(
            re.search(pattern, quote_lower) for pattern in self.attribution_patterns
        )

        return QuoteAnalysis(
            similarity_score=similarity,
            word_matches=word_matches,
            phrase_matches=phrase_matches,
            missing_words=missing_words,
            added_words=added_words,
            word_order_changes=word_order_changes,
            is_exact_quote=is_exact_quote,
            is_partial_quote=is_partial_quote,
            is_paraphrase=is_paraphrase,
            has_insertions=has_insertions,
            has_omissions=has_omissions,
            introduces_quote=introduces_quote,
            attribution_present=attribution_present,
        )

    def _find_phrase_matches(self, words1: List[str], words2: List[str]) -> List[str]:
        """Find matching phrases (2+ consecutive words)."""
        phrases = []

        for i in range(len(words1) - 1):
            for length in range(2, min(6, len(words1) - i + 1)):  # Up to 5-word phrases
                phrase1 = " ".join(words1[i : i + length])

                for j in range(len(words2) - length + 1):
                    phrase2 = " ".join(words2[j : j + length])

                    if phrase1 == phrase2:
                        phrases.append(phrase1)
                        break

        return phrases

    def _count_word_order_changes(self, words1: List[str], words2: List[str]) -> int:
        """Count how many words appear in different order."""
        # Simple heuristic: count words that appear but in wrong position
        changes = 0

        for i, word in enumerate(words1):
            if word in words2:
                expected_pos = words2.index(word)
                if abs(i - expected_pos) > 1:  # Allow for minor position differences
                    changes += 1

        return changes

    def _determine_quote_type(self, analysis: QuoteAnalysis) -> ReferenceType:
        """Determine the type of quotation based on analysis."""
        if analysis.is_exact_quote:
            return ReferenceType.DIRECT_QUOTE
        elif analysis.is_partial_quote:
            return ReferenceType.PARTIAL_QUOTE
        elif analysis.is_paraphrase:
            return ReferenceType.PARAPHRASE
        else:
            return ReferenceType.ALLUSION

    def create_citation_cross_references(
        self, citations: List[CitationMatch]
    ) -> List[CrossReference]:
        """Convert citation matches to cross-references."""
        cross_refs = []

        for citation in citations:
            # Calculate confidence metrics
            confidence = ReferenceConfidence(
                overall_score=citation.text_similarity,
                textual_similarity=citation.text_similarity,
                thematic_similarity=0.8,  # Citations typically thematically relevant
                structural_similarity=0.7,
                scholarly_consensus=0.9,  # Quotations have high scholarly consensus
                uncertainty_factors=[],
                lexical_links=len(citation.word_matches),
                semantic_links=3,  # Assume moderate semantic linking
                contextual_support=5 if citation.source_context else 3,
            )

            # Determine relationship
            relationship = ReferenceRelationship.QUOTES

            # Create cross-reference
            cross_ref = CrossReference(
                source_verse=citation.source_verse,
                target_verse=citation.target_verse,
                reference_type=citation.match_type,
                relationship=relationship,
                confidence=confidence,
                citation_match=citation,
                source="citation_tracker",
                theological_theme="quotation",
            )

            cross_refs.append(cross_ref)

            # Create reverse reference
            reverse_ref = cross_ref.get_reverse_reference()
            cross_refs.append(reverse_ref)

        return cross_refs

    def analyze_quotation_patterns(self, citations: List[CitationMatch]) -> Dict[str, any]:
        """Analyze patterns in the quotations found."""
        if not citations:
            return {"error": "No citations to analyze"}

        # Count by NT book
        nt_book_counts = Counter()
        ot_book_counts = Counter()
        quote_type_counts = Counter()

        for citation in citations:
            nt_book_counts[citation.source_verse.book] += 1
            ot_book_counts[citation.target_verse.book] += 1
            quote_type_counts[citation.match_type.value] += 1

        # Find most quoted OT books
        most_quoted_ot = ot_book_counts.most_common(10)

        # Find NT books with most quotes
        most_quoting_nt = nt_book_counts.most_common(10)

        # Average similarity scores
        similarity_scores = [c.text_similarity for c in citations]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        return {
            "total_citations": len(citations),
            "average_similarity": avg_similarity,
            "most_quoted_ot_books": most_quoted_ot,
            "most_quoting_nt_books": most_quoting_nt,
            "quote_types": dict(quote_type_counts),
            "similarity_distribution": {
                "high": len([s for s in similarity_scores if s > 0.8]),
                "medium": len([s for s in similarity_scores if 0.5 <= s <= 0.8]),
                "low": len([s for s in similarity_scores if s < 0.5]),
            },
        }

    def generate_citation_report(self, citations: List[CitationMatch]) -> str:
        """Generate a human-readable report of citation analysis."""
        if not citations:
            return "No citations found."

        analysis = self.analyze_quotation_patterns(citations)

        report = []
        report.append("OT CITATIONS IN NT - ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")

        report.append(f"Total Citations Found: {analysis['total_citations']}")
        report.append(f"Average Text Similarity: {analysis['average_similarity']:.2f}")
        report.append("")

        report.append("Most Quoted OT Books:")
        for book, count in analysis["most_quoted_ot_books"][:5]:
            report.append(f"  {book}: {count} citations")
        report.append("")

        report.append("NT Books with Most Citations:")
        for book, count in analysis["most_quoting_nt_books"][:5]:
            report.append(f"  {book}: {count} citations")
        report.append("")

        report.append("Citation Types:")
        for quote_type, count in analysis["quote_types"].items():
            report.append(f"  {quote_type}: {count}")
        report.append("")

        report.append("Sample High-Confidence Citations:")
        high_conf_citations = sorted(citations, key=lambda c: c.text_similarity, reverse=True)[:5]
        for citation in high_conf_citations:
            report.append(
                f"  {citation.source_verse} → {citation.target_verse} ({citation.text_similarity:.2f})"
            )
            report.append(f"    '{citation.source_text[:60]}...'")
            report.append("")

        return "\\n".join(report)
