"""Enrichment data population for ABBA biblical analysis."""

from .book_metadata import BookMetadataPopulator
from .concept_quality import ConceptQualityPopulator
from .cross_references import CrossReferencePopulator
from .cultural_context import CulturalContextPopulator
from .discourse_annotations import DiscourseAnnotationPopulator
from .genre_shifts import GenreShiftPopulator
from .life_topics import LifeTopicPopulator
from .literary_structures import LiteraryStructurePopulator
from .manuscript_variants import ManuscriptVariantPopulator
from .passages import PassagePopulator
from .reading_plans import ReadingPlanPopulator
from .semantic_domains import SemanticDomainPopulator
from .semantic_graph import SemanticGraphPopulator
from .speaker_attributions import SpeakerAttributionPopulator
from .syntax_trees import SyntaxTreePopulator
from .user_annotations import UserAnnotationManager
from .word_explanations import WordExplanationPopulator
from .word_richness import WordRichnessComputer

__all__ = [
    "BookMetadataPopulator",
    "ConceptQualityPopulator",
    "CrossReferencePopulator",
    "CulturalContextPopulator",
    "DiscourseAnnotationPopulator",
    "GenreShiftPopulator",
    "LifeTopicPopulator",
    "LiteraryStructurePopulator",
    "ManuscriptVariantPopulator",
    "PassagePopulator",
    "ReadingPlanPopulator",
    "SemanticDomainPopulator",
    "SemanticGraphPopulator",
    "SpeakerAttributionPopulator",
    "SyntaxTreePopulator",
    "UserAnnotationManager",
    "WordExplanationPopulator",
    "WordRichnessComputer",
]
