#!/usr/bin/env python3
"""
Load Full Strong's Concordance into Alignment Models.

This script loads the complete Strong's Hebrew and Greek concordance data
and integrates it with the existing alignment models to provide comprehensive
biblical word mappings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from abba.parsers.lexicon_parser import LexiconParser, StrongsLexicon
from abba.alignment.word_alignment import IBMModel1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrongsAlignmentLoader:
    """Load and integrate Strong's concordance data with alignment models."""
    
    def __init__(self, data_dir: Path):
        """Initialize the loader with data directory."""
        self.data_dir = Path(data_dir)
        self.lexicon_dir = self.data_dir / "sources" / "lexicons"
        self.alignment_dir = self.data_dir.parent / "models" / "alignment"
        
        # Ensure directories exist
        self.alignment_dir.mkdir(parents=True, exist_ok=True)
        
        # High-frequency word manual mappings
        self.manual_mappings = self._create_manual_mappings()
        
    def _create_manual_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Create manual alignment mappings for high-frequency biblical words."""
        return {
            "hebrew": {
                # Most common Hebrew words with their typical English translations
                "H3068": ["LORD", "Yahweh", "Jehovah", "God"],  # יהוה
                "H430": ["God", "gods", "judges", "mighty"],     # אלהים
                "H559": ["said", "say", "saying", "speak"],       # אמר
                "H1961": ["was", "be", "became", "come"],        # היה
                "H3605": ["all", "every", "whole", "any"],       # כל
                "H834": ["which", "who", "that", "what"],        # אשר
                "H413": ["to", "unto", "into", "toward"],        # אל
                "H5921": ["upon", "on", "over", "above"],        # על
                "H3808": ["not", "no", "nor", "neither"],        # לא
                "H1": ["father", "fathers", "patriarch"],         # אב
                "H120": ["man", "Adam", "mankind", "human"],     # אדם
                "H776": ["earth", "land", "ground", "country"],  # ארץ
                "H1121": ["son", "sons", "children", "child"],   # בן
                "H5414": ["give", "gave", "given", "put"],       # נתן
                "H6213": ["do", "make", "did", "made"],          # עשה
                "H7200": ["see", "saw", "seen", "look"],         # ראה
                "H8085": ["hear", "heard", "listen", "obey"],    # שמע
                "H3045": ["know", "knew", "known", "acknowledge"], # ידע
                "H1697": ["word", "words", "thing", "matter"],   # דבר
                "H3117": ["day", "days", "time", "year"],        # יום
            },
            "greek": {
                # Most common Greek words with their typical English translations
                "G2316": ["God", "god", "godly", "divine"],      # θεός
                "G2424": ["Jesus", "Joshua"],                     # Ἰησοῦς
                "G5547": ["Christ", "Messiah", "anointed"],      # Χριστός
                "G2962": ["Lord", "lord", "master", "sir"],      # κύριος
                "G3588": ["the", "this", "that", "who"],         # ὁ (article)
                "G846": ["he", "she", "it", "they", "him"],     # αὐτός
                "G1722": ["in", "on", "among", "by"],            # ἐν
                "G1519": ["into", "to", "unto", "for"],          # εἰς
                "G1537": ["from", "out of", "of", "by"],         # ἐκ
                "G3754": ["that", "because", "for", "since"],    # ὅτι
                "G3956": ["all", "every", "whole", "any"],       # πᾶς
                "G3361": ["not", "no", "lest"],                  # μή
                "G3756": ["not", "no", "none"],                  # οὐ
                "G444": ["man", "men", "person", "people"],      # ἄνθρωπος
                "G3004": ["say", "said", "speak", "tell"],       # λέγω
                "G1096": ["be", "become", "come", "happen"],     # γίνομαι
                "G2192": ["have", "has", "had", "hold"],         # ἔχω
                "G4160": ["do", "make", "did", "made"],          # ποιέω
                "G3708": ["see", "saw", "seen", "behold"],       # ὁράω
                "G191": ["hear", "heard", "listen"],             # ἀκούω
            }
        }
    
    def load_strongs_data(self) -> Tuple[Dict[str, any], Dict[str, any]]:
        """Load Strong's Hebrew and Greek concordance data."""
        logger.info("Loading Strong's concordance data...")
        
        # Try JSON format first (faster)
        hebrew_json = self.lexicon_dir / "strongs_hebrew.json"
        greek_json = self.lexicon_dir / "strongs_greek.json"
        
        if hebrew_json.exists() and greek_json.exists():
            logger.info("Loading from JSON files...")
            with open(hebrew_json, 'r', encoding='utf-8') as f:
                hebrew_data = json.load(f)
            with open(greek_json, 'r', encoding='utf-8') as f:
                greek_data = json.load(f)
            return hebrew_data, greek_data
        
        # Fall back to XML parsing
        logger.info("Loading from XML files...")
        parser = LexiconParser()
        lexicon = StrongsLexicon.from_directory(self.lexicon_dir)
        
        # Convert to dictionary format
        hebrew_data = {k: v.to_dict() for k, v in lexicon.hebrew_entries.items()}
        greek_data = {k: v.to_dict() for k, v in lexicon.greek_entries.items()}
        
        # Save as JSON for faster future loading
        with open(hebrew_json, 'w', encoding='utf-8') as f:
            json.dump(hebrew_data, f, ensure_ascii=False, indent=2)
        with open(greek_json, 'w', encoding='utf-8') as f:
            json.dump(greek_data, f, ensure_ascii=False, indent=2)
        
        return hebrew_data, greek_data
    
    def extract_translation_pairs(self, strongs_data: Dict[str, any], 
                                 language: str) -> Dict[str, Set[str]]:
        """Extract translation pairs from Strong's data."""
        logger.info(f"Extracting translation pairs for {language}...")
        
        translation_pairs = defaultdict(set)
        
        for strongs_num, entry in strongs_data.items():
            # Get the original word
            original = entry.get('original', entry.get('word', ''))
            if not original:
                continue
            
            # Add manual mappings if available
            if strongs_num in self.manual_mappings.get(language, {}):
                for translation in self.manual_mappings[language][strongs_num]:
                    translation_pairs[original].add(translation.lower())
                    translation_pairs[strongs_num].add(translation.lower())
            
            # Extract from glosses
            gloss = entry.get('gloss', '')
            if gloss:
                # Handle numeric glosses (skip them)
                if not gloss.replace('.', '').isdigit():
                    gloss_words = gloss.lower().replace(',', ' ').replace(';', ' ').split()
                    for word in gloss_words:
                        if len(word) > 2 and not word.startswith('('):
                            translation_pairs[original].add(word)
                            translation_pairs[strongs_num].add(word)
            
            # Extract from definition
            definition = entry.get('definition', '')
            if definition:
                # Extract main words from definition (first few words typically)
                def_words = definition.lower().split()[:10]
                for word in def_words:
                    word = word.strip('(),;:.1234567890')
                    if len(word) > 2 and word.isalpha():
                        translation_pairs[original].add(word)
                        translation_pairs[strongs_num].add(word)
            
            # Extract from KJV usage if available
            kjv_usage = entry.get('kjv_usage', '')
            if kjv_usage:
                kjv_words = kjv_usage.lower().replace(',', ' ').replace(';', ' ').split()
                for word in kjv_words:
                    word = word.strip('()[].-')
                    if len(word) > 2 and word.isalpha() and not word.startswith('idiom'):
                        translation_pairs[original].add(word)
                        translation_pairs[strongs_num].add(word)
        
        logger.info(f"Extracted {len(translation_pairs)} unique source terms")
        return translation_pairs
    
    def create_enhanced_alignment_model(self, hebrew_data: Dict, greek_data: Dict) -> Dict:
        """Create enhanced alignment model with Strong's data."""
        logger.info("Creating enhanced alignment model...")
        
        # Extract translation pairs
        hebrew_pairs = self.extract_translation_pairs(hebrew_data, 'hebrew')
        greek_pairs = self.extract_translation_pairs(greek_data, 'greek')
        
        # Combine into unified translation probability model
        trans_probs = defaultdict(lambda: defaultdict(float))
        
        # Process Hebrew pairs
        for source, targets in hebrew_pairs.items():
            total = len(targets)
            for target in targets:
                # Give higher probability to manual mappings
                if source.startswith('H') and source in self.manual_mappings['hebrew']:
                    if target in [t.lower() for t in self.manual_mappings['hebrew'][source]]:
                        trans_probs[source][target] = 0.8
                    else:
                        trans_probs[source][target] = 0.2 / total
                else:
                    trans_probs[source][target] = 1.0 / total
        
        # Process Greek pairs
        for source, targets in greek_pairs.items():
            total = len(targets)
            for target in targets:
                # Give higher probability to manual mappings
                if source.startswith('G') and source in self.manual_mappings['greek']:
                    if target in [t.lower() for t in self.manual_mappings['greek'][source]]:
                        trans_probs[source][target] = 0.8
                    else:
                        trans_probs[source][target] = 0.2 / total
                else:
                    trans_probs[source][target] = 1.0 / total
        
        # Create vocabulary sets
        source_vocab = set()
        target_vocab = set()
        
        for source, targets in trans_probs.items():
            source_vocab.add(source)
            target_vocab.update(targets.keys())
        
        # Add Strong's metadata
        strongs_metadata = {
            'hebrew': {k: {
                'original': v.get('original', ''),
                'lemma': v.get('lemma', ''),
                'translit': v.get('translit', ''),
                'morph': v.get('morph', ''),
                'gloss': v.get('gloss', ''),
                'definition': v.get('definition', '')[:200]  # Truncate long definitions
            } for k, v in hebrew_data.items()},
            'greek': {k: {
                'original': v.get('original', ''),
                'lemma': v.get('lemma', ''),
                'translit': v.get('translit', ''),
                'morph': v.get('morph', ''),
                'gloss': v.get('gloss', ''),
                'definition': v.get('definition', '')[:200]
            } for k, v in greek_data.items()}
        }
        
        # Create enhanced model
        model_data = {
            'trans_probs': dict(trans_probs),
            'source_vocab': list(source_vocab),
            'target_vocab': list(target_vocab),
            'strongs_metadata': strongs_metadata,
            'manual_mappings': self.manual_mappings,
            'version': '2.0',
            'description': 'Enhanced alignment model with full Strong\'s concordance'
        }
        
        logger.info(f"Created model with {len(source_vocab)} source terms and {len(target_vocab)} target terms")
        return model_data
    
    def save_enhanced_model(self, model_data: Dict, output_name: str = "strongs_enhanced_alignment.json"):
        """Save the enhanced alignment model."""
        output_path = self.alignment_dir / output_name
        
        logger.info(f"Saving enhanced model to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        # Also save a smaller sample version for testing
        sample_data = {
            'trans_probs': dict(list(model_data['trans_probs'].items())[:1000]),
            'source_vocab': model_data['source_vocab'][:1000],
            'target_vocab': model_data['target_vocab'][:1000],
            'strongs_metadata': {
                'hebrew': dict(list(model_data['strongs_metadata']['hebrew'].items())[:500]),
                'greek': dict(list(model_data['strongs_metadata']['greek'].items())[:500])
            },
            'manual_mappings': model_data['manual_mappings'],
            'version': model_data['version'],
            'description': model_data['description'] + ' (sample)'
        }
        
        sample_path = self.alignment_dir / f"sample_{output_name}"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved sample model to {sample_path}")
        
        return output_path
    
    def generate_statistics(self, model_data: Dict) -> Dict:
        """Generate statistics about the loaded data."""
        stats = {
            'total_source_terms': len(model_data['source_vocab']),
            'total_target_terms': len(model_data['target_vocab']),
            'total_translation_pairs': sum(len(targets) for targets in model_data['trans_probs'].values()),
            'hebrew_entries': len(model_data['strongs_metadata']['hebrew']),
            'greek_entries': len(model_data['strongs_metadata']['greek']),
            'manual_hebrew_mappings': len(self.manual_mappings['hebrew']),
            'manual_greek_mappings': len(self.manual_mappings['greek']),
            'top_source_terms': Counter({
                source: len(targets) 
                for source, targets in model_data['trans_probs'].items()
            }).most_common(20)
        }
        
        return stats


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Initialize loader
    loader = StrongsAlignmentLoader(data_dir)
    
    # Load Strong's data
    hebrew_data, greek_data = loader.load_strongs_data()
    logger.info(f"Loaded {len(hebrew_data)} Hebrew entries and {len(greek_data)} Greek entries")
    
    # Create enhanced model
    model_data = loader.create_enhanced_alignment_model(hebrew_data, greek_data)
    
    # Save the model
    output_path = loader.save_enhanced_model(model_data)
    
    # Generate and display statistics
    stats = loader.generate_statistics(model_data)
    
    logger.info("\n=== Strong's Concordance Loading Statistics ===")
    for key, value in stats.items():
        if key == 'top_source_terms':
            logger.info(f"\n{key}:")
            for term, count in value:
                logger.info(f"  {term}: {count} translations")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info(f"\nEnhanced alignment model saved to: {output_path}")
    

if __name__ == "__main__":
    main()