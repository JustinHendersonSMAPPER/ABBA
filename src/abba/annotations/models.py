"""
Data models for the annotation and tagging system.

Defines core data structures for representing annotations, topics,
taxonomies, and confidence metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from ..verse_id import VerseID


class AnnotationType(Enum):
    """Types of annotations."""

    # Thematic annotations
    THEOLOGICAL_THEME = "theological_theme"
    DOCTRINAL_CONCEPT = "doctrinal_concept"
    SPIRITUAL_PRACTICE = "spiritual_practice"
    ETHICAL_TEACHING = "ethical_teaching"

    # Literary annotations
    LITERARY_DEVICE = "literary_device"
    NARRATIVE_ELEMENT = "narrative_element"
    POETIC_FORM = "poetic_form"

    # Historical annotations
    HISTORICAL_CONTEXT = "historical_context"
    CULTURAL_BACKGROUND = "cultural_background"
    GEOGRAPHICAL_REFERENCE = "geographical_reference"

    # Linguistic annotations
    KEY_TERM = "key_term"
    WORD_STUDY = "word_study"
    TRANSLATION_NOTE = "translation_note"

    # Application annotations
    PRACTICAL_APPLICATION = "practical_application"
    DEVOTIONAL_THOUGHT = "devotional_thought"
    SERMON_TOPIC = "sermon_topic"


class AnnotationLevel(Enum):
    """Levels at which annotations can be applied."""

    WORD = "word"  # Individual word or phrase
    VERSE = "verse"  # Single verse
    PASSAGE = "passage"  # Multiple verses
    CHAPTER = "chapter"  # Entire chapter
    BOOK = "book"  # Entire book
    COLLECTION = "collection"  # Multiple books (e.g., Pauline epistles)


class TopicCategory(Enum):
    """High-level categories for theological topics."""

    # Core theological categories
    GOD = "god"  # Theology proper
    CHRIST = "christ"  # Christology
    HOLY_SPIRIT = "holy_spirit"  # Pneumatology
    TRINITY = "trinity"  # Trinitarian doctrine

    # Salvation and spiritual life
    SALVATION = "salvation"  # Soteriology
    SANCTIFICATION = "sanctification"  # Christian growth
    CHURCH = "church"  # Ecclesiology
    SACRAMENTS = "sacraments"  # Baptism, communion, etc.

    # Scripture and revelation
    SCRIPTURE = "scripture"  # Bibliology
    REVELATION = "revelation"  # How God reveals Himself
    PROPHECY = "prophecy"  # Prophetic literature

    # Christian living
    ETHICS = "ethics"  # Christian morality
    WORSHIP = "worship"  # Worship and liturgy
    PRAYER = "prayer"  # Prayer and devotion
    DISCIPLESHIP = "discipleship"  # Following Christ

    # Eschatology
    END_TIMES = "end_times"  # Last things
    HEAVEN_HELL = "heaven_hell"  # Eternal destinies
    JUDGMENT = "judgment"  # Divine judgment

    # Other theological topics
    ANGELS_DEMONS = "angels_demons"  # Angelology/Demonology
    HUMANITY = "humanity"  # Anthropology
    SIN = "sin"  # Hamartiology
    CREATION = "creation"  # Creation theology


@dataclass
class Topic:
    """Represents a theological or thematic topic."""

    id: str  # Unique identifier
    name: str  # Display name
    category: TopicCategory  # High-level category
    description: str  # Detailed description

    # Hierarchical relationships
    parent_id: Optional[str] = None  # Parent topic ID
    child_ids: List[str] = field(default_factory=list)

    # Related concepts
    related_ids: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)

    # Biblical references
    key_verses: List[VerseID] = field(default_factory=list)
    scripture_refs: List[str] = field(default_factory=list)

    # Metadata
    importance_score: float = 0.5  # 0.0 to 1.0
    frequency_score: float = 0.0  # How often it appears
    scholarly_consensus: float = 0.8  # Agreement level

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "related_ids": self.related_ids,
            "synonyms": self.synonyms,
            "key_verses": [str(v) for v in self.key_verses],
            "scripture_refs": self.scripture_refs,
            "metadata": {
                "importance_score": self.importance_score,
                "frequency_score": self.frequency_score,
                "scholarly_consensus": self.scholarly_consensus,
            },
        }


@dataclass
class AnnotationConfidence:
    """Confidence metrics for an annotation."""

    overall_score: float  # 0.0 to 1.0

    # Component scores
    model_confidence: float  # ML model's confidence
    contextual_relevance: float  # How well it fits context
    semantic_similarity: float  # Similarity to topic definition

    # Supporting evidence
    supporting_phrases: List[str] = field(default_factory=list)
    evidence_strength: float = 0.5

    # Uncertainty factors
    ambiguity_level: float = 0.0  # Level of ambiguity
    competing_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "overall_score": self.overall_score,
            "components": {
                "model_confidence": self.model_confidence,
                "contextual_relevance": self.contextual_relevance,
                "semantic_similarity": self.semantic_similarity,
            },
            "evidence": {
                "supporting_phrases": self.supporting_phrases,
                "evidence_strength": self.evidence_strength,
            },
            "uncertainty": {
                "ambiguity_level": self.ambiguity_level,
                "competing_topics": self.competing_topics,
            },
        }


@dataclass
class Annotation:
    """Represents an annotation on biblical text."""

    id: str  # Unique identifier
    annotation_type: AnnotationType  # Type of annotation
    level: AnnotationLevel  # Level of annotation

    # Location information
    start_verse: VerseID  # Starting verse
    end_verse: Optional[VerseID] = None  # Ending verse (for passages)
    word_positions: Optional[List[int]] = None  # For word-level

    # Content
    topic_id: Optional[str] = None  # Associated topic ID
    topic_name: Optional[str] = None  # Topic display name
    content: str = ""  # Annotation text/description

    # Confidence and quality
    confidence: Optional[AnnotationConfidence] = None
    quality_score: float = 0.0  # Manual quality assessment

    # Metadata
    source: str = "automatic"  # How it was created
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    created_by: Optional[str] = None
    verified: bool = False

    # Additional properties
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    references: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize dates if not provided."""
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.modified_date is None:
            self.modified_date = self.created_date

    def get_verse_range(self) -> List[VerseID]:
        """Get all verses covered by this annotation."""
        if self.end_verse is None:
            return [self.start_verse]

        # Would need verse enumeration logic here
        # For now, return endpoints
        return [self.start_verse, self.end_verse]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "id": self.id,
            "type": self.annotation_type.value,
            "level": self.level.value,
            "location": {
                "start_verse": str(self.start_verse),
                "end_verse": str(self.end_verse) if self.end_verse else None,
                "word_positions": self.word_positions,
            },
            "content": {
                "topic_id": self.topic_id,
                "topic_name": self.topic_name,
                "text": self.content,
            },
            "metadata": {
                "source": self.source,
                "created_date": self.created_date.isoformat() if self.created_date else None,
                "modified_date": self.modified_date.isoformat() if self.modified_date else None,
                "created_by": self.created_by,
                "verified": self.verified,
                "quality_score": self.quality_score,
            },
            "properties": {"tags": self.tags, "notes": self.notes, "references": self.references},
        }

        if self.confidence:
            result["confidence"] = self.confidence.to_dict()

        return result


@dataclass
class TopicTaxonomy:
    """Hierarchical taxonomy of theological topics."""

    topics: Dict[str, Topic]  # All topics by ID
    root_topics: List[str]  # Top-level topic IDs

    # Indices for efficient lookup
    _category_index: Dict[TopicCategory, List[str]] = field(default_factory=dict)
    _name_index: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Build indices after initialization."""
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild all indices."""
        self._category_index.clear()
        self._name_index.clear()

        for topic_id, topic in self.topics.items():
            # Category index
            if topic.category not in self._category_index:
                self._category_index[topic.category] = []
            self._category_index[topic.category].append(topic_id)

            # Name index (lowercase for case-insensitive lookup)
            self._name_index[topic.name.lower()] = topic_id

            # Also index synonyms
            for synonym in topic.synonyms:
                self._name_index[synonym.lower()] = topic_id

    def add_topic(self, topic: Topic):
        """Add a new topic to the taxonomy."""
        self.topics[topic.id] = topic

        # Update parent's child list
        if topic.parent_id and topic.parent_id in self.topics:
            parent = self.topics[topic.parent_id]
            if topic.id not in parent.child_ids:
                parent.child_ids.append(topic.id)

        # Add to root if no parent
        if topic.parent_id is None and topic.id not in self.root_topics:
            self.root_topics.append(topic.id)

        self._rebuild_indices()

    def get_topic_by_name(self, name: str) -> Optional[Topic]:
        """Look up topic by name (case-insensitive)."""
        topic_id = self._name_index.get(name.lower())
        return self.topics.get(topic_id) if topic_id else None

    def get_topics_by_category(self, category: TopicCategory) -> List[Topic]:
        """Get all topics in a category."""
        topic_ids = self._category_index.get(category, [])
        return [self.topics[tid] for tid in topic_ids]

    def get_topic_hierarchy(self, topic_id: str) -> List[Topic]:
        """Get the full hierarchy from root to topic."""
        hierarchy = []
        current_id = topic_id

        while current_id:
            if current_id not in self.topics:
                break
            topic = self.topics[current_id]
            hierarchy.insert(0, topic)  # Insert at beginning
            current_id = topic.parent_id

        return hierarchy

    def get_all_descendants(self, topic_id: str) -> List[Topic]:
        """Get all descendant topics recursively."""
        descendants = []

        def collect_descendants(tid: str):
            if tid in self.topics:
                topic = self.topics[tid]
                for child_id in topic.child_ids:
                    descendants.append(self.topics[child_id])
                    collect_descendants(child_id)

        collect_descendants(topic_id)
        return descendants

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "topics": {tid: t.to_dict() for tid, t in self.topics.items()},
            "root_topics": self.root_topics,
            "statistics": {
                "total_topics": len(self.topics),
                "categories": {cat.value: len(ids) for cat, ids in self._category_index.items()},
                "max_depth": self._calculate_max_depth(),
            },
        }

    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the taxonomy tree."""
        max_depth = 0

        for topic_id in self.topics:
            hierarchy = self.get_topic_hierarchy(topic_id)
            max_depth = max(max_depth, len(hierarchy))

        return max_depth


@dataclass
class AnnotationCollection:
    """Collection of annotations with indexing capabilities."""

    annotations: List[Annotation]
    metadata: Dict[str, Any]

    # Indices for efficient lookup
    _verse_index: Dict[str, List[int]] = field(default_factory=dict)
    _type_index: Dict[AnnotationType, List[int]] = field(default_factory=dict)
    _topic_index: Dict[str, List[int]] = field(default_factory=dict)
    _level_index: Dict[AnnotationLevel, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        """Build indices after initialization."""
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild all indices."""
        self._verse_index.clear()
        self._type_index.clear()
        self._topic_index.clear()
        self._level_index.clear()

        for i, ann in enumerate(self.annotations):
            # Verse index
            for verse in ann.get_verse_range():
                verse_key = str(verse)
                if verse_key not in self._verse_index:
                    self._verse_index[verse_key] = []
                self._verse_index[verse_key].append(i)

            # Type index
            if ann.annotation_type not in self._type_index:
                self._type_index[ann.annotation_type] = []
            self._type_index[ann.annotation_type].append(i)

            # Topic index
            if ann.topic_id:
                if ann.topic_id not in self._topic_index:
                    self._topic_index[ann.topic_id] = []
                self._topic_index[ann.topic_id].append(i)

            # Level index
            if ann.level not in self._level_index:
                self._level_index[ann.level] = []
            self._level_index[ann.level].append(i)

    def add_annotation(self, annotation: Annotation):
        """Add a new annotation to the collection."""
        self.annotations.append(annotation)
        self._rebuild_indices()  # Simple approach for now

    def get_verse_annotations(self, verse: VerseID) -> List[Annotation]:
        """Get all annotations for a specific verse."""
        verse_key = str(verse)
        indices = self._verse_index.get(verse_key, [])
        return [self.annotations[i] for i in indices]

    def get_annotations_by_type(self, ann_type: AnnotationType) -> List[Annotation]:
        """Get all annotations of a specific type."""
        indices = self._type_index.get(ann_type, [])
        return [self.annotations[i] for i in indices]

    def get_annotations_by_topic(self, topic_id: str) -> List[Annotation]:
        """Get all annotations for a specific topic."""
        indices = self._topic_index.get(topic_id, [])
        return [self.annotations[i] for i in indices]

    def get_annotations_by_level(self, level: AnnotationLevel) -> List[Annotation]:
        """Get all annotations at a specific level."""
        indices = self._level_index.get(level, [])
        return [self.annotations[i] for i in indices]

    def filter_annotations(
        self,
        verse: Optional[VerseID] = None,
        ann_type: Optional[AnnotationType] = None,
        topic_id: Optional[str] = None,
        level: Optional[AnnotationLevel] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Annotation]:
        """Filter annotations by multiple criteria."""
        # Start with all annotations
        candidates = set(range(len(self.annotations)))

        # Apply filters
        if verse:
            verse_indices = set(self._verse_index.get(str(verse), []))
            candidates &= verse_indices

        if ann_type:
            type_indices = set(self._type_index.get(ann_type, []))
            candidates &= type_indices

        if topic_id:
            topic_indices = set(self._topic_index.get(topic_id, []))
            candidates &= topic_indices

        if level:
            level_indices = set(self._level_index.get(level, []))
            candidates &= level_indices

        # Get annotations
        annotations = [self.annotations[i] for i in sorted(candidates)]

        # Apply confidence filter
        if min_confidence is not None:
            annotations = [
                a
                for a in annotations
                if a.confidence and a.confidence.overall_score >= min_confidence
            ]

        return annotations

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "total_annotations": len(self.annotations),
            "by_type": {t.value: len(indices) for t, indices in self._type_index.items()},
            "by_level": {l.value: len(indices) for l, indices in self._level_index.items()},
            "unique_verses": len(self._verse_index),
            "unique_topics": len(self._topic_index),
            "verified_count": sum(1 for a in self.annotations if a.verified),
            "average_confidence": self._calculate_avg_confidence(),
        }

    def _calculate_avg_confidence(self) -> Optional[float]:
        """Calculate average confidence score."""
        scores = [a.confidence.overall_score for a in self.annotations if a.confidence]

        return sum(scores) / len(scores) if scores else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metadata": self.metadata,
            "annotations": [a.to_dict() for a in self.annotations],
            "statistics": self.get_statistics(),
        }
