"""
Model information and discovery utilities for ABBA-Align.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelInfo:
    """Information about an alignment model."""
    
    def __init__(self, model_path: Path):
        self.path = model_path
        self.name = model_path.stem
        self.source_lang = None
        self.target_lang = None
        self.features = {}
        self.statistics = {}
        self.metadata = {}
        
        self._parse_name()
        self._load_report()
        
    def _parse_name(self):
        """Parse language info from model filename."""
        # Expected format: source_target_biblical.json
        parts = self.name.split('_')
        if len(parts) >= 3:
            self.source_lang = parts[0]
            self.target_lang = parts[1]
            
    def _load_report(self):
        """Load associated report file if it exists."""
        report_name = f"{self.source_lang}_{self.target_lang}_report.json"
        report_path = self.path.parent / report_name
        
        if report_path.exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    
                self.features = report.get('features_enabled', {})
                self.statistics = report.get('strongs_summary', {})
                self.metadata = {
                    'training_complete': report.get('training_complete', False),
                    'created': self._get_creation_time()
                }
            except Exception as e:
                logger.warning(f"Could not load report for {self.name}: {e}")
                
    def _get_creation_time(self):
        """Get model file creation time."""
        try:
            stat = self.path.stat()
            return datetime.fromtimestamp(stat.st_mtime).isoformat()
        except:
            return None
            
    def get_coverage_estimate(self) -> float:
        """Estimate coverage based on Strong's mappings."""
        mappings = self.statistics.get('translation_mappings', 0)
        
        # Rough estimate based on mappings
        if mappings > 3000:
            return 90.0
        elif mappings > 2000:
            return 85.0
        elif mappings > 1000:
            return 75.0
        else:
            return 65.0
            
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'path': str(self.path),
            'source_language': self.source_lang,
            'target_language': self.target_lang,
            'features': self.features,
            'statistics': self.statistics,
            'metadata': self.metadata,
            'estimated_coverage': self.get_coverage_estimate()
        }


class ModelDiscovery:
    """Discover and manage alignment models."""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path('models/biblical_alignment')
        self.models = []
        self._discover_models()
        
    def _discover_models(self):
        """Find all available models."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return
            
        # Find all model files
        model_files = list(self.model_dir.glob('*_biblical.json'))
        
        for model_path in model_files:
            try:
                model_info = ModelInfo(model_path)
                self.models.append(model_info)
            except Exception as e:
                logger.warning(f"Could not load model {model_path}: {e}")
                
    def find_model(self, source_lang: str, target_lang: str = 'english') -> Optional[ModelInfo]:
        """Find best model for language pair."""
        candidates = []
        
        for model in self.models:
            if (model.source_lang == source_lang and 
                model.target_lang == target_lang):
                candidates.append(model)
                
        if not candidates:
            return None
            
        # Sort by features and coverage
        candidates.sort(
            key=lambda m: (
                len([f for f in m.features.values() if f]),  # Number of features
                m.get_coverage_estimate(),  # Estimated coverage
                m.statistics.get('translation_mappings', 0)  # Mappings
            ),
            reverse=True
        )
        
        return candidates[0]
        
    def find_models_for_source(self, source_lang: str) -> List[ModelInfo]:
        """Find all models for a source language."""
        return [m for m in self.models if m.source_lang == source_lang]
        
    def find_models_for_target(self, target_lang: str) -> List[ModelInfo]:
        """Find all models for a target language."""
        return [m for m in self.models if m.target_lang == target_lang]
        
    def list_all_models(self) -> List[Dict]:
        """List all discovered models with details."""
        return [model.to_dict() for model in self.models]
        
    def get_language_pairs(self) -> List[Tuple[str, str]]:
        """Get all available language pairs."""
        pairs = set()
        for model in self.models:
            if model.source_lang and model.target_lang:
                pairs.add((model.source_lang, model.target_lang))
        return sorted(list(pairs))
        
    def generate_report(self) -> str:
        """Generate a human-readable report of available models."""
        lines = []
        lines.append("=" * 70)
        lines.append("AVAILABLE ABBA-ALIGN MODELS")
        lines.append("=" * 70)
        lines.append("")
        
        if not self.models:
            lines.append("No models found in " + str(self.model_dir))
            lines.append("Train models using: abba-align train --source <lang> --target <lang>")
            return "\n".join(lines)
            
        # Group by source language
        by_source = defaultdict(list)
        for model in self.models:
            by_source[model.source_lang].append(model)
            
        for source_lang in sorted(by_source.keys()):
            lines.append(f"SOURCE: {source_lang.upper()}")
            lines.append("-" * 40)
            
            for model in sorted(by_source[source_lang], key=lambda m: m.target_lang):
                lines.append(f"  → {model.target_lang}")
                lines.append(f"    Model: {model.name}")
                
                # Features
                enabled_features = [k for k, v in model.features.items() if v]
                if enabled_features:
                    lines.append(f"    Features: {', '.join(enabled_features)}")
                    
                # Statistics
                if model.statistics:
                    mappings = model.statistics.get('translation_mappings', 0)
                    lines.append(f"    Mappings: {mappings:,}")
                    
                # Coverage estimate
                coverage = model.get_coverage_estimate()
                lines.append(f"    Est. Coverage: {coverage:.0f}%")
                lines.append("")
                
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total models: {len(self.models)}")
        
        pairs = self.get_language_pairs()
        lines.append(f"Language pairs: {len(pairs)}")
        
        source_langs = set(m.source_lang for m in self.models if m.source_lang)
        target_langs = set(m.target_lang for m in self.models if m.target_lang)
        lines.append(f"Source languages: {', '.join(sorted(source_langs))}")
        lines.append(f"Target languages: {', '.join(sorted(target_langs))}")
        
        return "\n".join(lines)


def auto_select_model(source_text_path: Path) -> Optional[Path]:
    """Automatically select best model based on source text."""
    # Try to detect language from file
    source_lang = None
    
    # Check filename patterns
    if 'hebrew' in str(source_text_path).lower() or 'heb' in str(source_text_path).lower():
        source_lang = 'hebrew'
    elif 'greek' in str(source_text_path).lower() or 'grc' in str(source_text_path).lower():
        source_lang = 'greek'
    elif 'aramaic' in str(source_text_path).lower():
        source_lang = 'aramaic'
        
    # Try to detect from content
    if not source_lang and source_text_path.exists():
        try:
            with open(source_text_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1KB
                
            # Simple heuristics
            if 'בְּרֵאשִׁית' in content or 'אֱלֹהִים' in content:
                source_lang = 'hebrew'
            elif 'Ἰησοῦς' in content or 'Χριστός' in content:
                source_lang = 'greek'
        except:
            pass
            
    if not source_lang:
        logger.warning("Could not auto-detect source language")
        return None
        
    # Find best model
    discovery = ModelDiscovery()
    model = discovery.find_model(source_lang)
    
    if model:
        logger.info(f"Auto-selected model: {model.name}")
        return model.path
    else:
        logger.warning(f"No model found for {source_lang}")
        return None


def main():
    """CLI entry point for model discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover and analyze ABBA-Align models")
    parser.add_argument('--find', nargs=2, metavar=('SOURCE', 'TARGET'),
                       help='Find model for language pair')
    parser.add_argument('--list', action='store_true',
                       help='List all available models')
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format')
    parser.add_argument('--model-dir', type=Path,
                       default=Path('models/biblical_alignment'),
                       help='Model directory path')
    
    args = parser.parse_args()
    
    discovery = ModelDiscovery(args.model_dir)
    
    if args.find:
        source, target = args.find
        model = discovery.find_model(source, target)
        
        if model:
            if args.json:
                print(json.dumps(model.to_dict(), indent=2))
            else:
                print(f"Found model: {model.name}")
                print(f"Path: {model.path}")
                print(f"Features: {', '.join(k for k, v in model.features.items() if v)}")
                print(f"Estimated coverage: {model.get_coverage_estimate():.0f}%")
        else:
            print(f"No model found for {source} → {target}")
            
    else:  # List all
        if args.json:
            models = discovery.list_all_models()
            print(json.dumps(models, indent=2))
        else:
            print(discovery.generate_report())


if __name__ == '__main__':
    main()