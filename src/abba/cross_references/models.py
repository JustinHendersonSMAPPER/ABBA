"""
Data models for cross-reference system.

Defines the core data structures for representing biblical cross-references,
including reference types, relationships, and confidence metrics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..verse_id import VerseID


class ReferenceType(Enum):
    """Types of biblical cross-references."""

    # Direct quotations
    DIRECT_QUOTE = "direct_quote"  # Exact or near-exact quotation
    PARTIAL_QUOTE = "partial_quote"  # Partial quotation with omissions
    PARAPHRASE = "paraphrase"  # Reworded quotation

    # Allusions and parallels
    ALLUSION = "allusion"  # Indirect reference or echo
    THEMATIC_PARALLEL = "thematic_parallel"  # Similar theme or concept
    STRUCTURAL_PARALLEL = "structural_parallel"  # Similar narrative structure

    # Typological relationships
    TYPE_ANTITYPE = "type_antitype"  # Old Testament type, New Testament fulfillment
    PROPHECY_FULFILLMENT = "prophecy_fulfillment"  # Prophetic prediction and fulfillment

    # Contextual relationships
    HISTORICAL_PARALLEL = "historical_parallel"  # Similar historical events
    LITERARY_PARALLEL = "literary_parallel"  # Similar literary forms or genres
    DOCTRINAL_PARALLEL = "doctrinal_parallel"  # Similar theological teaching

    # Cross-reference categories
    EXPLANATION = "explanation"  # One passage explains another
    ILLUSTRATION = "illustration"  # One passage illustrates another
    CONTRAST = "contrast"  # Contrasting or opposing ideas


class ReferenceRelationship(Enum):
    """Directional relationships between references."""

    QUOTES = "quotes"  # Source quotes target
    QUOTED_BY = "quoted_by"  # Source is quoted by target
    ALLUDES_TO = "alludes_to"  # Source alludes to target
    ALLUDED_BY = "alluded_by"  # Source is alluded to by target
    PARALLELS = "parallels"  # Bidirectional parallel
    FULFILLS = "fulfills"  # Source fulfills target prophecy
    FULFILLED_BY = "fulfilled_by"  # Source is fulfilled by target
    EXPLAINS = "explains"  # Source explains target
    EXPLAINED_BY = "explained_by"  # Source is explained by target
    CONTRASTS = "contrasts"  # Source contrasts with target


@dataclass
class ReferenceConfidence:
    """Confidence metrics for cross-references."""

    overall_score: float  # 0.0 to 1.0
    textual_similarity: float  # Similarity of actual text
    thematic_similarity: float  # Similarity of themes/concepts
    structural_similarity: float  # Similarity of literary structure
    scholarly_consensus: float  # Level of scholarly agreement

    # Uncertainty factors
    uncertainty_factors: List[str]  # Factors reducing confidence

    # Supporting evidence
    lexical_links: int = 0  # Number of shared key words
    semantic_links: int = 0  # Number of shared concepts
    contextual_support: int = 0  # Contextual evidence strength

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "overall_score": self.overall_score,
            "textual_similarity": self.textual_similarity,
            "thematic_similarity": self.thematic_similarity,
            "structural_similarity": self.structural_similarity,
            "scholarly_consensus": self.scholarly_consensus,
            "uncertainty_factors": self.uncertainty_factors,
            "supporting_evidence": {
                "lexical_links": self.lexical_links,
                "semantic_links": self.semantic_links,
                "contextual_support": self.contextual_support,
            },
        }


@dataclass
class CitationMatch:
    """Represents a specific citation match between passages."""

    source_verse: VerseID  # Verse containing the citation
    target_verse: VerseID  # Verse being cited
    source_text: str  # Text of the citation
    target_text: str  # Text being cited

    # Match characteristics
    match_type: ReferenceType  # Type of citation
    text_similarity: float  # 0.0 to 1.0 similarity score
    word_matches: List[str]  # Specific matching words

    # Context information
    source_context: Optional[str] = None  # Surrounding context of citation
    target_context: Optional[str] = None  # Surrounding context of target

    # Metadata
    discovered_by: str = "automatic"  # How this match was discovered
    verified: bool = False  # Has been manually verified
    notes: Optional[str] = None  # Additional notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "source_verse": str(self.source_verse),
            "target_verse": str(self.target_verse),
            "source_text": self.source_text,
            "target_text": self.target_text,
            "match_type": self.match_type.value,
            "text_similarity": self.text_similarity,
            "word_matches": self.word_matches,
            "source_context": self.source_context,
            "target_context": self.target_context,
            "metadata": {
                "discovered_by": self.discovered_by,
                "verified": self.verified,
                "notes": self.notes,
            },
        }


@dataclass
class CrossReference:
    """Complete cross-reference between biblical passages."""

    source_verse: VerseID  # Primary verse
    target_verse: VerseID  # Referenced verse
    reference_type: ReferenceType  # Type of reference
    relationship: ReferenceRelationship  # Directional relationship
    confidence: ReferenceConfidence  # Confidence metrics

    # Optional detailed citation data
    citation_match: Optional[CitationMatch] = None

    # Reference metadata
    source: str = "automatic"  # Source of this reference
    created_date: Optional[datetime] = None
    verified: bool = False  # Manual verification status

    # Additional context
    topic_tags: List[str] = None  # Associated topics
    historical_period: Optional[str] = None  # Historical context
    theological_theme: Optional[str] = None  # Theological significance

    def __post_init__(self):
        """Initialize default values."""
        if self.topic_tags is None:
            self.topic_tags = []
        if self.created_date is None:
            self.created_date = datetime.now()

    def get_reverse_reference(self) -> "CrossReference":
        """Create the reverse directional reference."""
        reverse_relationships = {
            ReferenceRelationship.QUOTES: ReferenceRelationship.QUOTED_BY,
            ReferenceRelationship.QUOTED_BY: ReferenceRelationship.QUOTES,
            ReferenceRelationship.ALLUDES_TO: ReferenceRelationship.ALLUDED_BY,
            ReferenceRelationship.ALLUDED_BY: ReferenceRelationship.ALLUDES_TO,
            ReferenceRelationship.FULFILLS: ReferenceRelationship.FULFILLED_BY,
            ReferenceRelationship.FULFILLED_BY: ReferenceRelationship.FULFILLS,
            ReferenceRelationship.EXPLAINS: ReferenceRelationship.EXPLAINED_BY,
            ReferenceRelationship.EXPLAINED_BY: ReferenceRelationship.EXPLAINS,
            ReferenceRelationship.PARALLELS: ReferenceRelationship.PARALLELS,
            ReferenceRelationship.CONTRASTS: ReferenceRelationship.CONTRASTS,
        }

        return CrossReference(
            source_verse=self.target_verse,
            target_verse=self.source_verse,
            reference_type=self.reference_type,
            relationship=reverse_relationships[self.relationship],
            confidence=self.confidence,
            citation_match=self.citation_match,
            source=self.source,
            created_date=self.created_date,
            verified=self.verified,
            topic_tags=self.topic_tags.copy(),
            historical_period=self.historical_period,
            theological_theme=self.theological_theme,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "source_verse": str(self.source_verse),
            "target_verse": str(self.target_verse),
            "reference_type": self.reference_type.value,
            "relationship": self.relationship.value,
            "confidence": self.confidence.to_dict(),
            "metadata": {
                "source": self.source,
                "created_date": self.created_date.isoformat() if self.created_date else None,
                "verified": self.verified,
            },
            "context": {
                "topic_tags": self.topic_tags,
                "historical_period": self.historical_period,
                "theological_theme": self.theological_theme,
            },
        }

        if self.citation_match:
            result["citation_match"] = self.citation_match.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossReference":
        """Create CrossReference from dictionary."""
        # Parse confidence
        conf_data = data["confidence"]
        confidence = ReferenceConfidence(
            overall_score=conf_data["overall_score"],
            textual_similarity=conf_data["textual_similarity"],
            thematic_similarity=conf_data["thematic_similarity"],
            structural_similarity=conf_data["structural_similarity"],
            scholarly_consensus=conf_data["scholarly_consensus"],
            uncertainty_factors=conf_data["uncertainty_factors"],
            lexical_links=conf_data["supporting_evidence"]["lexical_links"],
            semantic_links=conf_data["supporting_evidence"]["semantic_links"],
            contextual_support=conf_data["supporting_evidence"]["contextual_support"],
        )

        # Parse citation match if present
        citation_match = None
        if "citation_match" in data:
            from ..verse_id import parse_verse_id

            cm_data = data["citation_match"]
            citation_match = CitationMatch(
                source_verse=parse_verse_id(cm_data["source_verse"]),
                target_verse=parse_verse_id(cm_data["target_verse"]),
                source_text=cm_data["source_text"],
                target_text=cm_data["target_text"],
                match_type=ReferenceType(cm_data["match_type"]),
                text_similarity=cm_data["text_similarity"],
                word_matches=cm_data["word_matches"],
                source_context=cm_data.get("source_context"),
                target_context=cm_data.get("target_context"),
                discovered_by=cm_data["metadata"]["discovered_by"],
                verified=cm_data["metadata"]["verified"],
                notes=cm_data["metadata"].get("notes"),
            )

        # Parse dates
        created_date = None
        if data["metadata"]["created_date"]:
            created_date = datetime.fromisoformat(data["metadata"]["created_date"])

        return cls(
            source_verse=parse_verse_id(data["source_verse"]),
            target_verse=parse_verse_id(data["target_verse"]),
            reference_type=ReferenceType(data["reference_type"]),
            relationship=ReferenceRelationship(data["relationship"]),
            confidence=confidence,
            citation_match=citation_match,
            source=data["metadata"]["source"],
            created_date=created_date,
            verified=data["metadata"]["verified"],
            topic_tags=data["context"]["topic_tags"],
            historical_period=data["context"]["historical_period"],
            theological_theme=data["context"]["theological_theme"],
        )


@dataclass
class ReferenceCollection:
    """Collection of cross-references with indexing and search capabilities."""

    references: List[CrossReference]
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Initialize indices."""
        self._source_index: Dict[str, List[int]] = {}
        self._target_index: Dict[str, List[int]] = {}
        self._type_index: Dict[ReferenceType, List[int]] = {}
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild all indices."""
        self._source_index.clear()
        self._target_index.clear()
        self._type_index.clear()

        for i, ref in enumerate(self.references):
            # Source index
            source_key = str(ref.source_verse)
            if source_key not in self._source_index:
                self._source_index[source_key] = []
            self._source_index[source_key].append(i)

            # Target index
            target_key = str(ref.target_verse)
            if target_key not in self._target_index:
                self._target_index[target_key] = []
            self._target_index[target_key].append(i)

            # Type index
            if ref.reference_type not in self._type_index:
                self._type_index[ref.reference_type] = []
            self._type_index[ref.reference_type].append(i)

    def add_reference(self, reference: CrossReference):
        """Add a new reference to the collection."""
        self.references.append(reference)
        self._rebuild_indices()  # Simple rebuild for now

    def get_references_from(self, verse: VerseID) -> List[CrossReference]:
        """Get all references originating from a verse."""
        source_key = str(verse)
        indices = self._source_index.get(source_key, [])
        return [self.references[i] for i in indices]

    def get_references_to(self, verse: VerseID) -> List[CrossReference]:
        """Get all references pointing to a verse."""
        target_key = str(verse)
        indices = self._target_index.get(target_key, [])
        return [self.references[i] for i in indices]

    def get_references_by_type(self, ref_type: ReferenceType) -> List[CrossReference]:
        """Get all references of a specific type."""
        indices = self._type_index.get(ref_type, [])
        return [self.references[i] for i in indices]

    def get_bidirectional_references(self, verse: VerseID) -> List[CrossReference]:
        """Get all references involving a verse (both directions)."""
        return self.get_references_from(verse) + self.get_references_to(verse)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metadata": self.metadata,
            "references": [ref.to_dict() for ref in self.references],
            "statistics": {
                "total_references": len(self.references),
                "types_distribution": {
                    ref_type.value: len(indices) for ref_type, indices in self._type_index.items()
                },
                "unique_source_verses": len(self._source_index),
                "unique_target_verses": len(self._target_index),
            },
        }
