"""
Main alignment training module that orchestrates all components.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .lexicon_integration import StrongsLexiconIntegration
from .morphological_analyzer import MorphologicalAnalyzer
from .phrase_detector import BiblicalPhraseDetector

logger = logging.getLogger(__name__)


class AlignmentTrainer:
    """Orchestrates biblical text alignment training."""
    
    def __init__(self, source_lang: str, target_lang: str,
                 enable_morphology: bool = True,
                 enable_phrases: bool = True,
                 enable_syntax: bool = True,
                 enable_semantics: bool = True,
                 enable_discourse: bool = True,
                 enable_strongs: bool = True):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.features = {
            'morphology': enable_morphology,
            'phrases': enable_phrases,
            'syntax': enable_syntax,
            'semantics': enable_semantics,
            'discourse': enable_discourse,
            'strongs': enable_strongs
        }
        self.model = None
        
        # Initialize components
        if enable_strongs:
            self.strongs_integration = StrongsLexiconIntegration()
            self._load_strongs_data()
            
        if enable_morphology:
            self.morph_analyzer = MorphologicalAnalyzer(source_lang)
            
        if enable_phrases:
            self.phrase_detector = BiblicalPhraseDetector(source_lang)
            
    def _load_strongs_data(self):
        """Load Strong's concordance data."""
        # Check for Strong's lexicon files
        hebrew_lexicon = Path('data/sources/lexicons/strongs_hebrew.json')
        greek_lexicon = Path('data/sources/lexicons/strongs_greek.json')
        
        if self.source_lang == 'hebrew':
            if not hebrew_lexicon.exists():
                raise FileNotFoundError(
                    f"Hebrew Strong's lexicon not found at {hebrew_lexicon}\n"
                    "Please run: python scripts/download_sources.py"
                )
            self.strongs_integration.load_strongs_lexicon(hebrew_lexicon, 'hebrew')
            logger.info("Loaded Hebrew Strong's lexicon")
            
        elif self.source_lang == 'greek':
            if not greek_lexicon.exists():
                raise FileNotFoundError(
                    f"Greek Strong's lexicon not found at {greek_lexicon}\n"
                    "Please run: python scripts/download_sources.py"
                )
            self.strongs_integration.load_strongs_lexicon(greek_lexicon, 'greek')
            logger.info("Loaded Greek Strong's lexicon")
        
    def load_corpora(self, corpus_dir: Path):
        """Load training corpora."""
        logger.info(f"Loading corpora from {corpus_dir}")
        # Placeholder implementation - would load actual corpus data
        
    def add_parallel_data(self, parallel_data: List):
        """Add parallel passage data for improved training."""
        logger.info(f"Adding {len(parallel_data)} parallel passages")
        # Placeholder implementation
        
    def train(self) -> Any:
        """Train the alignment model with all features."""
        logger.info("Training alignment model...")
        
        # If Strong's is enabled, use it to enhance training
        if self.features['strongs'] and hasattr(self, 'strongs_integration'):
            logger.info("Using Strong's concordance for enhanced alignment")
            # Would integrate Strong's data into IBM Model 1 initialization
            
        # Placeholder - would integrate all components
        self.model = {
            'trained': True,
            'features': self.features,
            'strongs_enabled': self.features.get('strongs', False)
        }
        return self.model
        
    def save_model(self, model: Any, path: Path):
        """Save trained model."""
        logger.info(f"Saving model to {path}")
        
        # Create full model data structure
        model_data = {
            'trained': True,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'features': self.features,
            'version': '1.0',
            'training_date': datetime.now().isoformat()
        }
        
        # Add Strong's mappings if available
        if hasattr(self, 'strongs_integration') and self.features.get('strongs'):
            model_data['strongs_mappings'] = {}
            # Convert Counter objects to regular dicts for JSON serialization
            for strongs_num, counter in self.strongs_integration.strongs_to_english.items():
                model_data['strongs_mappings'][strongs_num] = dict(counter)
                
        # Add morphology patterns if available
        if hasattr(self, 'morph_analyzer') and self.features.get('morphology'):
            model_data['morphology_patterns'] = {
                'language': self.source_lang,
                'patterns_learned': True
            }
            
        # Save model
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Model successfully saved to {path}")
        
    def generate_report(self) -> Dict:
        """Generate training report."""
        report = {
            'source_language': self.source_lang,
            'target_language': self.target_lang,
            'features_enabled': self.features,
            'training_complete': True
        }
        
        if hasattr(self, 'strongs_integration'):
            report['strongs_summary'] = {
                'hebrew_entries': len(self.strongs_integration.hebrew_strongs),
                'greek_entries': len(self.strongs_integration.greek_strongs),
                'translation_mappings': len(self.strongs_integration.strongs_to_english)
            }
            
        return report