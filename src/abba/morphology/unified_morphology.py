"""
Unified morphology parsing system.

This module provides a unified interface for parsing both Greek and Hebrew
morphology codes, with automatic language detection and formatting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from .base import Language, MorphologyFeatures
from .greek_morphology import GreekMorphology, GreekMorphologyParser
from .hebrew_morphology import HebrewMorphology, HebrewMorphologyParser


@dataclass
class UnifiedMorphology:
    """Unified morphology result with language information."""
    language: Language
    features: Union[GreekMorphology, HebrewMorphology, MorphologyFeatures]
    original_code: str
    
    def is_verb(self) -> bool:
        """Check if this is a verb."""
        return getattr(self.features, 'part_of_speech', None) == 'verb'
    
    def is_participle(self) -> bool:
        """Check if this is a participle."""
        # Check if part_of_speech is 'participle' OR if mood is PARTICIPLE
        if getattr(self.features, 'part_of_speech', None) == 'participle':
            return True
        mood = getattr(self.features, 'mood', None)
        from .base import Mood
        return mood == Mood.PARTICIPLE if mood else False
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the morphology."""
        if hasattr(self.features, 'get_summary'):
            return self.features.get_summary()
        return str(self.features)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "language": self.language.value,
            "original_code": self.original_code,
            "features": self.features.to_dict() if hasattr(self.features, 'to_dict') else {},
            "summary": self.features.get_summary() if hasattr(self.features, 'get_summary') else ""
        }


class UnifiedMorphologyParser:
    """Unified parser for both Greek and Hebrew morphology."""
    
    def __init__(self):
        """Initialize the unified parser."""
        self.greek_parser = GreekMorphologyParser()
        self.hebrew_parser = HebrewMorphologyParser()
    
    def parse(
        self, 
        morph_code: str, 
        language: Optional[Union[str, Language]] = None
    ) -> UnifiedMorphology:
        """
        Parse morphology code with optional language hint.
        
        Args:
            morph_code: Morphology code to parse
            language: Optional language hint ("greek", "hebrew", or None for auto-detect)
            
        Returns:
            UnifiedMorphology object with language and features
        """
        if not morph_code:
            return UnifiedMorphology(
                language=Language.HEBREW,  # Default
                features=MorphologyFeatures(),
                original_code=morph_code or ""
            )
        
        # Use language hint if provided
        if language:
            # Handle Language enum
            if isinstance(language, Language):
                lang_enum = language
                lang_str = language.value.lower()
            elif hasattr(language, 'value'):
                lang_enum = language
                lang_str = language.value.lower()
            else:
                lang_str = str(language).lower()
                lang_enum = Language.GREEK if lang_str in ["greek", "grc", "grk"] else Language.HEBREW
                
            if lang_str in ["greek", "grc", "grk"]:
                return UnifiedMorphology(
                    language=Language.GREEK,
                    features=self.greek_parser.parse(morph_code),
                    original_code=morph_code
                )
            elif lang_str in ["hebrew", "heb", "hbo"]:
                return UnifiedMorphology(
                    language=Language.HEBREW,
                    features=self.hebrew_parser.parse(morph_code),
                    original_code=morph_code
                )
        
        # Auto-detect language based on morphology code pattern
        if self._is_greek_code(morph_code):
            return UnifiedMorphology(
                language=Language.GREEK,
                features=self.greek_parser.parse(morph_code),
                original_code=morph_code
            )
        elif self._is_hebrew_code(morph_code):
            return UnifiedMorphology(
                language=Language.HEBREW,
                features=self.hebrew_parser.parse(morph_code),
                original_code=morph_code
            )
        
        # Try both parsers and return the one with more features
        greek_result = self.greek_parser.parse(morph_code)
        hebrew_result = self.hebrew_parser.parse(morph_code)
        
        # Count populated features
        greek_features = sum(1 for v in greek_result.to_dict().values() if v is not None)
        hebrew_features = sum(1 for v in hebrew_result.to_dict().values() if v is not None)
        
        if greek_features > hebrew_features:
            return UnifiedMorphology(
                language=Language.GREEK,
                features=greek_result,
                original_code=morph_code
            )
        elif hebrew_features > greek_features:
            return UnifiedMorphology(
                language=Language.HEBREW,
                features=hebrew_result,
                original_code=morph_code
            )
        
        # Default to basic features
        return UnifiedMorphology(
            language=Language.HEBREW,  # Default
            features=MorphologyFeatures(),
            original_code=morph_code
        )
    
    def _is_greek_code(self, morph_code: str) -> bool:
        """Check if morphology code looks like Greek."""
        # Greek codes often have dashes: V-PAI-3S, N-NSM
        if "-" in morph_code and morph_code[0] in "VNADCPXIRT":
            return True
        
        # Greek compact format with uppercase letters
        if morph_code and morph_code[0] in "VNADCPXIRT" and morph_code.isupper():
            return True
        
        return False
    
    def _is_hebrew_code(self, morph_code: str) -> bool:
        """Check if morphology code looks like Hebrew."""
        # Hebrew codes often have prefixes: HNcmsa, VQP3MS
        if morph_code and morph_code[0] in "HCR":
            return True
        
        # Hebrew verb codes: VQP3MS (Verb Qal Perfect 3rd Masc Sing)
        if len(morph_code) >= 3 and morph_code[0] == "V" and morph_code[1] in "qQNphHt":
            return True
        
        # Hebrew noun codes: Ncmsa (Noun common masc sing abs)
        if len(morph_code) >= 2 and morph_code[0] == "N" and morph_code[1] in "cp":
            return True
        
        return False
    
    def format_for_display(
        self, 
        morphology: MorphologyFeatures,
        style: str = "full"
    ) -> str:
        """
        Format morphology features for display.
        
        Args:
            morphology: Morphology features to format
            style: Display style ("full", "abbreviated", "code")
            
        Returns:
            Formatted string representation
        """
        if isinstance(morphology, GreekMorphology):
            return self._format_greek_display(morphology, style)
        elif isinstance(morphology, HebrewMorphology):
            return self._format_hebrew_display(morphology, style)
        else:
            return morphology.get_summary()
    
    def _format_greek_display(self, morph: GreekMorphology, style: str) -> str:
        """Format Greek morphology for display."""
        if style == "code":
            # Reconstruct morphology code
            parts = []
            
            # Part of speech
            pos_code = {v: k for k, v in self.greek_parser.PART_OF_SPEECH_MAP.items()}
            if morph.part_of_speech in pos_code:
                parts.append(pos_code[morph.part_of_speech])
            
            # For verbs
            if morph.part_of_speech == "verb":
                tvm = ""
                if morph.tense:
                    tense_code = {v: k for k, v in self.greek_parser.TENSE_MAP.items()}
                    tvm += tense_code.get(morph.tense, "?")
                if morph.voice:
                    voice_code = {v: k for k, v in self.greek_parser.VOICE_MAP.items()}
                    tvm += voice_code.get(morph.voice, "?")
                if morph.mood:
                    mood_code = {v: k for k, v in self.greek_parser.MOOD_MAP.items()}
                    tvm += mood_code.get(morph.mood, "?")
                parts.append(tvm)
                
                # Person/Number or Case/Number/Gender
                if morph.mood == "participle":
                    cng = ""
                    if morph.case:
                        case_code = {v: k for k, v in self.greek_parser.CASE_MAP.items()}
                        cng += case_code.get(morph.case, "?")
                    if morph.number:
                        number_code = {v: k for k, v in self.greek_parser.NUMBER_MAP.items()}
                        cng += number_code.get(morph.number, "?")
                    if morph.gender:
                        gender_code = {v: k for k, v in self.greek_parser.GENDER_MAP.items()}
                        cng += gender_code.get(morph.gender, "?")
                    parts.append(cng)
                elif morph.person or morph.number:
                    pn = ""
                    if morph.person:
                        person_code = {v: k for k, v in self.greek_parser.PERSON_MAP.items()}
                        pn += person_code.get(morph.person, "?")
                    if morph.number:
                        number_code = {v: k for k, v in self.greek_parser.NUMBER_MAP.items()}
                        pn += number_code.get(morph.number, "?")
                    parts.append(pn)
            
            return "-".join(parts)
        
        elif style == "abbreviated":
            return morph.get_greek_summary()
        
        else:  # full
            parts = []
            
            if morph.part_of_speech:
                parts.append(morph.part_of_speech.title())
            
            if morph.tense:
                parts.append(morph.tense.value)
            if morph.voice:
                parts.append(morph.voice.value)
            if morph.mood:
                parts.append(morph.mood.value)
            
            if morph.person:
                parts.append(f"{morph.person.value} person")
            if morph.case:
                parts.append(morph.case.value)
            if morph.gender:
                parts.append(morph.gender.value)
            if morph.number:
                parts.append(morph.number.value)
            
            return ", ".join(parts)
    
    def _format_hebrew_display(self, morph: HebrewMorphology, style: str) -> str:
        """Format Hebrew morphology for display."""
        if style == "code":
            # Reconstruct morphology code
            parts = []
            
            # Prefixes
            if morph.has_article:
                parts.append("H")
            if morph.has_conjunction:
                parts.append("C")
            if morph.has_preposition:
                parts.append("R")
            
            # Part of speech
            pos_code = {v: k for k, v in self.hebrew_parser.PART_OF_SPEECH_MAP.items()}
            if morph.part_of_speech in pos_code:
                parts.append(pos_code[morph.part_of_speech])
            
            # Additional features based on POS
            if morph.part_of_speech == "verb":
                if morph.stem:
                    stem_code = {v: k for k, v in self.hebrew_parser.STEM_MAP.items()}
                    parts.append(stem_code.get(morph.stem, "?"))
                if morph.tense:
                    tense_code = {v: k for k, v in self.hebrew_parser.TENSE_MAP.items()}
                    parts.append(tense_code.get(morph.tense, "?"))
                if morph.person:
                    person_code = {v: k for k, v in self.hebrew_parser.PERSON_MAP.items()}
                    parts.append(person_code.get(morph.person, "?"))
                if morph.gender:
                    gender_code = {v: k for k, v in self.hebrew_parser.GENDER_MAP.items()}
                    parts.append(gender_code.get(morph.gender, "?"))
                if morph.number:
                    number_code = {v: k for k, v in self.hebrew_parser.NUMBER_MAP.items()}
                    parts.append(number_code.get(morph.number, "?"))
            
            return "".join(parts)
        
        elif style == "abbreviated":
            return morph.get_hebrew_summary()
        
        else:  # full
            return morph.get_hebrew_summary()
    
    def get_language(self, morphology: MorphologyFeatures) -> str:
        """Determine the language of a morphology object."""
        if isinstance(morphology, GreekMorphology):
            return "greek"
        elif isinstance(morphology, HebrewMorphology):
            return "hebrew"
        else:
            return "unknown"
    
    def parse_auto_detect(self, morph_code: str) -> UnifiedMorphology:
        """
        Parse morphology code with automatic language detection.
        
        Args:
            morph_code: Morphology code to parse
            
        Returns:
            UnifiedMorphology object with detected language
        """
        return self.parse(morph_code, language=None)
    
    def compare_morphologies(self, morph1: UnifiedMorphology, morph2: UnifiedMorphology) -> Dict[str, Any]:
        """
        Compare two morphology objects.
        
        Args:
            morph1: First morphology to compare
            morph2: Second morphology to compare
            
        Returns:
            Dictionary with comparison results
        """
        
        # Get feature dictionaries
        features1 = morph1.features.to_dict() if hasattr(morph1.features, 'to_dict') else {}
        features2 = morph2.features.to_dict() if hasattr(morph2.features, 'to_dict') else {}
        
        # Find agreements and differences
        agreements = {}
        differences = {}
        
        all_keys = set(features1.keys()) | set(features2.keys())
        
        for key in all_keys:
            val1 = features1.get(key)
            val2 = features2.get(key)
            
            if val1 == val2 and val1 is not None:
                agreements[key] = val1
            elif val1 != val2:
                differences[key] = {"morph1": val1, "morph2": val2}
        
        return {
            "languages": [morph1.language.value, morph2.language.value],
            "agreements": agreements,
            "differences": differences
        }
    
    def get_morphology_statistics(self, morphologies: List[UnifiedMorphology]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of morphologies.
        
        Args:
            morphologies: List of UnifiedMorphology objects
            
        Returns:
            Dictionary with statistics
        """
        from collections import defaultdict
        
        stats = {
            "total": len(morphologies),
            "by_language": defaultdict(int),
            "by_part_of_speech": defaultdict(int)
        }
        
        for morph in morphologies:
            # Count by language
            stats["by_language"][morph.language.value] += 1
            
            # Count by part of speech
            pos = getattr(morph.features, 'part_of_speech', None)
            if pos:
                stats["by_part_of_speech"][pos] += 1
        
        # Convert defaultdicts to regular dicts
        stats["by_language"] = dict(stats["by_language"])
        stats["by_part_of_speech"] = dict(stats["by_part_of_speech"])
        
        return stats