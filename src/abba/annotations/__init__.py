"""
Annotation and tagging system for biblical texts.

This package provides sophisticated annotation functionality including:
- Hierarchical topic taxonomy with theological concepts
- Automatic annotation using modern NLP (BERT, BERTopic, SetFit)
- Multi-level tagging (verse, passage, chapter, book)
- Manual annotation interface and quality control
- Topic propagation and relationship mapping
"""

from .models import (
    Annotation,
    AnnotationType,
    AnnotationLevel,
    Topic,
    TopicCategory,
    TopicTaxonomy,
    AnnotationCollection,
    AnnotationConfidence,
)
from .taxonomy import TheologicalTaxonomy

# Optional imports that require ML dependencies
try:
    from .bert_adapter import BiblicalBERTAdapter
    from .topic_discovery import BERTopicDiscovery
    from .few_shot_classifier import SetFitClassifier
    from .zero_shot_classifier import ZeroShotTheologyClassifier
    from .annotation_engine import AnnotationEngine
    from .quality_control import AnnotationQualityController
except ImportError:
    # These components require torch and other ML libraries
    BiblicalBERTAdapter = None
    BERTopicDiscovery = None
    SetFitClassifier = None
    ZeroShotTheologyClassifier = None
    AnnotationEngine = None
    AnnotationQualityController = None

__all__ = [
    # Core models
    "Annotation",
    "AnnotationType",
    "AnnotationLevel",
    "Topic",
    "TopicCategory",
    "TopicTaxonomy",
    "AnnotationCollection",
    "AnnotationConfidence",
    # Processing components
    "TheologicalTaxonomy",
    "BiblicalBERTAdapter",
    "BERTopicDiscovery",
    "SetFitClassifier",
    "ZeroShotTheologyClassifier",
    "AnnotationEngine",
    "AnnotationQualityController",
]
