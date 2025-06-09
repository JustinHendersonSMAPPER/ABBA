"""
ABBA-Align: Biblical Text Alignment Toolkit

A specialized tool for training and applying word alignment models
for biblical texts, with support for morphological analysis,
phrase detection, and theological concept mapping.
"""

__version__ = "1.0.0"

from .morphological_analyzer import MorphologicalAnalyzer
from .phrase_detector import BiblicalPhraseDetector
from .parallel_passage_aligner import ParallelPassageAligner
from .semantic_role_labeler import BiblicalSRL
from .discourse_analyzer import DiscourseAnalyzer
from .alignment_trainer import AlignmentTrainer
from .alignment_annotator import AlignmentAnnotator

__all__ = [
    'MorphologicalAnalyzer',
    'BiblicalPhraseDetector', 
    'ParallelPassageAligner',
    'BiblicalSRL',
    'DiscourseAnalyzer',
    'AlignmentTrainer',
    'AlignmentAnnotator'
]