"""Enrichment data population for ABBA biblical analysis."""

from .book_metadata import BookMetadataPopulator
from .concept_quality import ConceptQualityPopulator
from .cross_references import CrossReferencePopulator
from .cultural_context import CulturalContextPopulator
from .genre_shifts import GenreShiftPopulator
from .life_topics import LifeTopicPopulator
from .literary_structures import LiteraryStructurePopulator
from .passages import PassagePopulator
from .reading_plans import ReadingPlanPopulator
from .speaker_attributions import SpeakerAttributionPopulator
from .user_annotations import UserAnnotationManager
from .word_explanations import WordExplanationPopulator
from .word_richness import WordRichnessComputer

__all__ = [
    "BookMetadataPopulator",
    "ConceptQualityPopulator",
    "CrossReferencePopulator",
    "CulturalContextPopulator",
    "GenreShiftPopulator",
    "LifeTopicPopulator",
    "LiteraryStructurePopulator",
    "PassagePopulator",
    "ReadingPlanPopulator",
    "SpeakerAttributionPopulator",
    "UserAnnotationManager",
    "WordExplanationPopulator",
    "WordRichnessComputer",
]
