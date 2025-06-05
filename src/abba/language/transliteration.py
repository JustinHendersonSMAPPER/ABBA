"""
Transliteration engine for biblical languages.

Provides reversible transliteration between scripts, supporting multiple
transliteration schemes and academic standards.
"""

import re
import unicodedata
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod


class TransliterationScheme(Enum):
    """Available transliteration schemes."""

    # Hebrew schemes
    SBL_HEBREW = "sbl_hebrew"  # Society of Biblical Literature
    ACADEMIC_HEBREW = "academic_hebrew"  # Academic/scholarly
    SIMPLE_HEBREW = "simple_hebrew"  # Simplified/popular
    ISO_259 = "iso_259"  # ISO standard

    # Greek schemes
    SBL_GREEK = "sbl_greek"  # Society of Biblical Literature
    ACADEMIC_GREEK = "academic_greek"  # Academic/scholarly
    SIMPLE_GREEK = "simple_greek"  # Simplified
    BETA_CODE = "beta_code"  # Beta code

    # Other
    ARABIC_DIN = "arabic_din"  # DIN 31635
    SYRIAC_ACADEMIC = "syriac_academic"  # Academic Syriac


@dataclass
class TransliterationRule:
    """A transliteration rule."""

    source: str  # Source character(s)
    target: str  # Target character(s)
    context: Optional[str] = None  # Regex context
    priority: int = 0  # Higher priority rules apply first
    reversible: bool = True  # Whether rule can be reversed


class TransliterationEngine(ABC):
    """Base transliteration engine."""

    def __init__(self, scheme: TransliterationScheme):
        """Initialize the engine.

        Args:
            scheme: Transliteration scheme to use
        """
        self.scheme = scheme
        self.rules: List[TransliterationRule] = []
        self.reverse_rules: List[TransliterationRule] = []
        self.setup_rules()
        self.build_reverse_rules()

    @abstractmethod
    def setup_rules(self):
        """Setup transliteration rules for the scheme."""
        pass

    def build_reverse_rules(self):
        """Build reverse transliteration rules."""
        self.reverse_rules = []

        for rule in self.rules:
            if rule.reversible:
                reverse_rule = TransliterationRule(
                    source=rule.target,
                    target=rule.source,
                    context=None,  # Context may not reverse cleanly
                    priority=rule.priority,
                    reversible=True,
                )
                self.reverse_rules.append(reverse_rule)

        # Sort by priority and length (longer matches first)
        self.reverse_rules.sort(key=lambda r: (-r.priority, -len(r.source)))

    def transliterate(self, text: str, preserve_unknown: bool = True) -> str:
        """Transliterate text.

        Args:
            text: Text to transliterate
            preserve_unknown: Whether to preserve unknown characters

        Returns:
            Transliterated text
        """
        return self._apply_rules(text, self.rules, preserve_unknown)

    def reverse_transliterate(self, text: str, preserve_unknown: bool = True) -> str:
        """Reverse transliterate text back to original script.

        Args:
            text: Transliterated text
            preserve_unknown: Whether to preserve unknown characters

        Returns:
            Original script text
        """
        return self._apply_rules(text, self.reverse_rules, preserve_unknown)

    def _apply_rules(
        self, text: str, rules: List[TransliterationRule], preserve_unknown: bool
    ) -> str:
        """Apply transliteration rules to text."""
        result = text

        # Track positions that have been transliterated
        transliterated = set()

        # Apply rules in priority order
        for rule in sorted(rules, key=lambda r: (-r.priority, -len(r.source))):
            if rule.context:
                # Context-sensitive rule
                pattern = rule.context.replace("SOURCE", re.escape(rule.source))
                matches = list(re.finditer(pattern, result))

                # Apply in reverse order to preserve positions
                for match in reversed(matches):
                    start, end = match.span()
                    if not any(pos in transliterated for pos in range(start, end)):
                        result = result[:start] + rule.target + result[end:]
                        transliterated.update(range(start, start + len(rule.target)))
            else:
                # Simple replacement
                pos = 0
                while pos < len(result):
                    if result[pos:].startswith(rule.source) and pos not in transliterated:
                        result = result[:pos] + rule.target + result[pos + len(rule.source) :]
                        transliterated.update(range(pos, pos + len(rule.target)))
                        pos += len(rule.target)
                    else:
                        pos += 1

        return result


class HebrewTransliterator(TransliterationEngine):
    """Hebrew transliteration engine."""

    def setup_rules(self):
        """Setup Hebrew transliteration rules."""
        if self.scheme == TransliterationScheme.SBL_HEBREW:
            self._setup_sbl_rules()
        elif self.scheme == TransliterationScheme.ACADEMIC_HEBREW:
            self._setup_academic_rules()
        elif self.scheme == TransliterationScheme.SIMPLE_HEBREW:
            self._setup_simple_rules()
        else:
            self._setup_simple_rules()  # Default

    def _setup_sbl_rules(self):
        """Setup SBL Hebrew transliteration rules."""
        # Consonants
        consonants = [
            # Letters
            TransliterationRule("א", "ʾ", priority=10),
            TransliterationRule("ב", "b", priority=10),
            TransliterationRule("בּ", "b", priority=11),  # With dagesh
            TransliterationRule("ב", "v", context=r"[^ּ]SOURCE", priority=10),  # Without dagesh
            TransliterationRule("ג", "g", priority=10),
            TransliterationRule("גּ", "g", priority=11),
            TransliterationRule("ד", "d", priority=10),
            TransliterationRule("דּ", "d", priority=11),
            TransliterationRule("ה", "h", priority=10),
            TransliterationRule("ו", "w", priority=10),
            TransliterationRule("וּ", "w", priority=11),
            TransliterationRule("ז", "z", priority=10),
            TransliterationRule("ח", "ḥ", priority=10),
            TransliterationRule("ט", "ṭ", priority=10),
            TransliterationRule("י", "y", priority=10),
            TransliterationRule("כ", "k", priority=10),
            TransliterationRule("כּ", "k", priority=11),
            TransliterationRule("כ", "ḵ", context=r"[^ּ]SOURCE", priority=10),
            TransliterationRule("ך", "ḵ", priority=10),  # Final kaf
            TransliterationRule("ל", "l", priority=10),
            TransliterationRule("מ", "m", priority=10),
            TransliterationRule("ם", "m", priority=10),  # Final mem
            TransliterationRule("נ", "n", priority=10),
            TransliterationRule("ן", "n", priority=10),  # Final nun
            TransliterationRule("ס", "s", priority=10),
            TransliterationRule("ע", "ʿ", priority=10),
            TransliterationRule("פ", "p", priority=10),
            TransliterationRule("פּ", "p", priority=11),
            TransliterationRule("פ", "p̄", context=r"[^ּ]SOURCE", priority=10),
            TransliterationRule("ף", "p̄", priority=10),  # Final pe
            TransliterationRule("צ", "ṣ", priority=10),
            TransliterationRule("ץ", "ṣ", priority=10),  # Final tsade
            TransliterationRule("ק", "q", priority=10),
            TransliterationRule("ר", "r", priority=10),
            TransliterationRule("שׁ", "š", priority=11),  # Shin
            TransliterationRule("שׂ", "ś", priority=11),  # Sin
            TransliterationRule("ש", "š", priority=10),  # Default shin
            TransliterationRule("ת", "t", priority=10),
            TransliterationRule("תּ", "t", priority=11),
        ]

        # Vowels
        vowels = [
            # Short vowels
            TransliterationRule("ַ", "a", priority=5),  # Patah
            TransliterationRule("ָ", "ā", priority=5),  # Qamats
            TransliterationRule("ֶ", "e", priority=5),  # Segol
            TransliterationRule("ֵ", "ē", priority=5),  # Tsere
            TransliterationRule("ִ", "i", priority=5),  # Hiriq
            TransliterationRule("ֹ", "ō", priority=5),  # Holam
            TransliterationRule("ֻ", "u", priority=5),  # Qubuts
            TransliterationRule("ְ", "ĕ", priority=5),  # Sheva
            # Hataf vowels
            TransliterationRule("ֲ", "ă", priority=5),  # Hataf Patah
            TransliterationRule("ֱ", "ĕ", priority=5),  # Hataf Segol
            TransliterationRule("ֳ", "ŏ", priority=5),  # Hataf Qamats
            # Full vowels (matres lectionis)
            TransliterationRule("וֹ", "ô", priority=6),  # Vav with holam
            TransliterationRule("וּ", "û", priority=6),  # Vav with dagesh
            TransliterationRule("יִ", "î", priority=6),  # Yod with hiriq
        ]

        # Special characters
        special = [
            TransliterationRule("־", "-", priority=1),  # Maqaf
            TransliterationRule("׃", ".", priority=1),  # Sof pasuq
        ]

        self.rules = consonants + vowels + special

    def _setup_academic_rules(self):
        """Setup academic Hebrew transliteration rules."""
        # Similar to SBL but with some differences
        # Using more standard academic conventions
        self._setup_sbl_rules()  # Start with SBL

        # Override some rules
        overrides = [
            TransliterationRule("צ", "ṣ", priority=12),  # Sometimes 'ẓ'
            TransliterationRule("ץ", "ṣ", priority=12),
            TransliterationRule("ק", "q", priority=12),  # Sometimes 'ḳ'
        ]

        self.rules.extend(overrides)

    def _setup_simple_rules(self):
        """Setup simple Hebrew transliteration rules."""
        # No diacritics, simple Latin letters
        simple_rules = [
            # Consonants
            TransliterationRule("א", "'", priority=10),
            TransliterationRule("ב", "v", priority=10),
            TransliterationRule("בּ", "b", priority=11),
            TransliterationRule("ג", "g", priority=10),
            TransliterationRule("ד", "d", priority=10),
            TransliterationRule("ה", "h", priority=10),
            TransliterationRule("ו", "v", priority=10),
            TransliterationRule("ז", "z", priority=10),
            TransliterationRule("ח", "ch", priority=10),
            TransliterationRule("ט", "t", priority=10),
            TransliterationRule("י", "y", priority=10),
            TransliterationRule("כ", "kh", priority=10),
            TransliterationRule("כּ", "k", priority=11),
            TransliterationRule("ך", "kh", priority=10),
            TransliterationRule("ל", "l", priority=10),
            TransliterationRule("מ", "m", priority=10),
            TransliterationRule("ם", "m", priority=10),
            TransliterationRule("נ", "n", priority=10),
            TransliterationRule("ן", "n", priority=10),
            TransliterationRule("ס", "s", priority=10),
            TransliterationRule("ע", "'", priority=10),
            TransliterationRule("פ", "f", priority=10),
            TransliterationRule("פּ", "p", priority=11),
            TransliterationRule("ף", "f", priority=10),
            TransliterationRule("צ", "ts", priority=10),
            TransliterationRule("ץ", "ts", priority=10),
            TransliterationRule("ק", "k", priority=10),
            TransliterationRule("ר", "r", priority=10),
            TransliterationRule("שׁ", "sh", priority=11),
            TransliterationRule("שׂ", "s", priority=11),
            TransliterationRule("ש", "sh", priority=10),
            TransliterationRule("ת", "t", priority=10),
            # Basic vowels
            TransliterationRule("ַ", "a", priority=5),
            TransliterationRule("ָ", "a", priority=5),
            TransliterationRule("ֶ", "e", priority=5),
            TransliterationRule("ֵ", "e", priority=5),
            TransliterationRule("ִ", "i", priority=5),
            TransliterationRule("ֹ", "o", priority=5),
            TransliterationRule("ֻ", "u", priority=5),
            TransliterationRule("ְ", "e", priority=5),
        ]

        self.rules = simple_rules


class GreekTransliterator(TransliterationEngine):
    """Greek transliteration engine."""

    def setup_rules(self):
        """Setup Greek transliteration rules."""
        if self.scheme == TransliterationScheme.SBL_GREEK:
            self._setup_sbl_rules()
        elif self.scheme == TransliterationScheme.ACADEMIC_GREEK:
            self._setup_academic_rules()
        elif self.scheme == TransliterationScheme.SIMPLE_GREEK:
            self._setup_simple_rules()
        elif self.scheme == TransliterationScheme.BETA_CODE:
            self._setup_beta_code_rules()
        else:
            self._setup_simple_rules()

    def _setup_sbl_rules(self):
        """Setup SBL Greek transliteration rules."""
        # Basic letters
        letters = [
            TransliterationRule("α", "a", priority=10),
            TransliterationRule("β", "b", priority=10),
            TransliterationRule("γ", "g", priority=10),
            TransliterationRule("δ", "d", priority=10),
            TransliterationRule("ε", "e", priority=10),
            TransliterationRule("ζ", "z", priority=10),
            TransliterationRule("η", "ē", priority=10),
            TransliterationRule("θ", "th", priority=10),
            TransliterationRule("ι", "i", priority=10),
            TransliterationRule("κ", "k", priority=10),
            TransliterationRule("λ", "l", priority=10),
            TransliterationRule("μ", "m", priority=10),
            TransliterationRule("ν", "n", priority=10),
            TransliterationRule("ξ", "x", priority=10),
            TransliterationRule("ο", "o", priority=10),
            TransliterationRule("π", "p", priority=10),
            TransliterationRule("ρ", "r", priority=10),
            TransliterationRule("σ", "s", priority=10),
            TransliterationRule("ς", "s", priority=10),  # Final sigma
            TransliterationRule("τ", "t", priority=10),
            TransliterationRule("υ", "y", priority=10),
            TransliterationRule("φ", "ph", priority=10),
            TransliterationRule("χ", "ch", priority=10),
            TransliterationRule("ψ", "ps", priority=10),
            TransliterationRule("ω", "ō", priority=10),
        ]

        # Diphthongs
        diphthongs = [
            TransliterationRule("αι", "ai", priority=11),
            TransliterationRule("ει", "ei", priority=11),
            TransliterationRule("οι", "oi", priority=11),
            TransliterationRule("υι", "yi", priority=11),
            TransliterationRule("αυ", "au", priority=11),
            TransliterationRule("ευ", "eu", priority=11),
            TransliterationRule("ου", "ou", priority=11),
        ]

        # Breathing marks and accents (when preserved)
        diacritics = [
            TransliterationRule("ἀ", "a", priority=12),  # Smooth breathing
            TransliterationRule("ἁ", "ha", priority=12),  # Rough breathing
            TransliterationRule("ά", "á", priority=11),  # Acute
            TransliterationRule("ὰ", "à", priority=11),  # Grave
            TransliterationRule("ᾶ", "â", priority=11),  # Circumflex
        ]

        # Special combinations
        special = [
            TransliterationRule("γγ", "ng", priority=12),
            TransliterationRule("γκ", "nk", priority=12),
            TransliterationRule("γξ", "nx", priority=12),
            TransliterationRule("γχ", "nch", priority=12),
        ]

        self.rules = letters + diphthongs + diacritics + special

    def _setup_academic_rules(self):
        """Setup academic Greek transliteration rules."""
        # Start with SBL
        self._setup_sbl_rules()

        # Add more precise diacritic handling
        additional = [
            TransliterationRule("ῥ", "rh", priority=13),  # Rough breathing on rho
            TransliterationRule("Ῥ", "Rh", priority=13),
        ]

        self.rules.extend(additional)

    def _setup_simple_rules(self):
        """Setup simple Greek transliteration rules."""
        # No diacritics, simple mapping
        simple = [
            TransliterationRule("α", "a", priority=10),
            TransliterationRule("β", "b", priority=10),
            TransliterationRule("γ", "g", priority=10),
            TransliterationRule("δ", "d", priority=10),
            TransliterationRule("ε", "e", priority=10),
            TransliterationRule("ζ", "z", priority=10),
            TransliterationRule("η", "e", priority=10),  # Simple 'e'
            TransliterationRule("θ", "th", priority=10),
            TransliterationRule("ι", "i", priority=10),
            TransliterationRule("κ", "k", priority=10),
            TransliterationRule("λ", "l", priority=10),
            TransliterationRule("μ", "m", priority=10),
            TransliterationRule("ν", "n", priority=10),
            TransliterationRule("ξ", "x", priority=10),
            TransliterationRule("ο", "o", priority=10),
            TransliterationRule("π", "p", priority=10),
            TransliterationRule("ρ", "r", priority=10),
            TransliterationRule("σ", "s", priority=10),
            TransliterationRule("ς", "s", priority=10),
            TransliterationRule("τ", "t", priority=10),
            TransliterationRule("υ", "u", priority=10),  # Simple 'u'
            TransliterationRule("φ", "f", priority=10),  # Simple 'f'
            TransliterationRule("χ", "ch", priority=10),
            TransliterationRule("ψ", "ps", priority=10),
            TransliterationRule("ω", "o", priority=10),  # Simple 'o'
        ]

        self.rules = simple

    def _setup_beta_code_rules(self):
        """Setup Beta Code transliteration rules."""
        # Beta Code uses ASCII representation
        beta_rules = [
            # Letters
            TransliterationRule("α", "a", priority=10),
            TransliterationRule("β", "b", priority=10),
            TransliterationRule("γ", "g", priority=10),
            TransliterationRule("δ", "d", priority=10),
            TransliterationRule("ε", "e", priority=10),
            TransliterationRule("ζ", "z", priority=10),
            TransliterationRule("η", "h", priority=10),
            TransliterationRule("θ", "q", priority=10),
            TransliterationRule("ι", "i", priority=10),
            TransliterationRule("κ", "k", priority=10),
            TransliterationRule("λ", "l", priority=10),
            TransliterationRule("μ", "m", priority=10),
            TransliterationRule("ν", "n", priority=10),
            TransliterationRule("ξ", "c", priority=10),
            TransliterationRule("ο", "o", priority=10),
            TransliterationRule("π", "p", priority=10),
            TransliterationRule("ρ", "r", priority=10),
            TransliterationRule("σ", "s", priority=10),
            TransliterationRule("ς", "s", priority=10),
            TransliterationRule("τ", "t", priority=10),
            TransliterationRule("υ", "u", priority=10),
            TransliterationRule("φ", "f", priority=10),
            TransliterationRule("χ", "x", priority=10),
            TransliterationRule("ψ", "y", priority=10),
            TransliterationRule("ω", "w", priority=10),
            # Uppercase indicated by *
            TransliterationRule("Α", "*a", priority=11),
            TransliterationRule("Β", "*b", priority=11),
            # ... etc
        ]

        self.rules = beta_rules


class ArabicTransliterator(TransliterationEngine):
    """Arabic transliteration engine."""

    def setup_rules(self):
        """Setup Arabic transliteration rules."""
        if self.scheme == TransliterationScheme.ARABIC_DIN:
            self._setup_din_rules()
        else:
            self._setup_din_rules()  # Default

    def _setup_din_rules(self):
        """Setup DIN 31635 Arabic transliteration rules."""
        rules = [
            # Letters
            TransliterationRule("ا", "ā", priority=10),  # Alif
            TransliterationRule("ب", "b", priority=10),  # Ba
            TransliterationRule("ت", "t", priority=10),  # Ta
            TransliterationRule("ث", "ṯ", priority=10),  # Tha
            TransliterationRule("ج", "ǧ", priority=10),  # Jim
            TransliterationRule("ح", "ḥ", priority=10),  # Ha
            TransliterationRule("خ", "ḫ", priority=10),  # Kha
            TransliterationRule("د", "d", priority=10),  # Dal
            TransliterationRule("ذ", "ḏ", priority=10),  # Dhal
            TransliterationRule("ر", "r", priority=10),  # Ra
            TransliterationRule("ز", "z", priority=10),  # Zay
            TransliterationRule("س", "s", priority=10),  # Sin
            TransliterationRule("ش", "š", priority=10),  # Shin
            TransliterationRule("ص", "ṣ", priority=10),  # Sad
            TransliterationRule("ض", "ḍ", priority=10),  # Dad
            TransliterationRule("ط", "ṭ", priority=10),  # Ta
            TransliterationRule("ظ", "ẓ", priority=10),  # Za
            TransliterationRule("ع", "ʿ", priority=10),  # Ayn
            TransliterationRule("غ", "ġ", priority=10),  # Ghayn
            TransliterationRule("ف", "f", priority=10),  # Fa
            TransliterationRule("ق", "q", priority=10),  # Qaf
            TransliterationRule("ك", "k", priority=10),  # Kaf
            TransliterationRule("ل", "l", priority=10),  # Lam
            TransliterationRule("م", "m", priority=10),  # Mim
            TransliterationRule("ن", "n", priority=10),  # Nun
            TransliterationRule("ه", "h", priority=10),  # Ha
            TransliterationRule("و", "w", priority=10),  # Waw
            TransliterationRule("ي", "y", priority=10),  # Ya
            # Special characters
            TransliterationRule("ء", "ʾ", priority=10),  # Hamza
            TransliterationRule("ة", "h", priority=10),  # Ta marbuta
            TransliterationRule("ى", "ā", priority=10),  # Alif maqsura
            # Article
            TransliterationRule("ال", "al-", priority=11),
        ]

        self.rules = rules


class SyriacTransliterator(TransliterationEngine):
    """Syriac transliteration engine."""

    def setup_rules(self):
        """Setup Syriac transliteration rules."""
        # Basic Syriac letters
        rules = [
            TransliterationRule("ܐ", "ʾ", priority=10),  # Alaph
            TransliterationRule("ܒ", "b", priority=10),  # Beth
            TransliterationRule("ܓ", "g", priority=10),  # Gamal
            TransliterationRule("ܕ", "d", priority=10),  # Dalath
            TransliterationRule("ܗ", "h", priority=10),  # He
            TransliterationRule("ܘ", "w", priority=10),  # Waw
            TransliterationRule("ܙ", "z", priority=10),  # Zayn
            TransliterationRule("ܚ", "ḥ", priority=10),  # Heth
            TransliterationRule("ܛ", "ṭ", priority=10),  # Teth
            TransliterationRule("ܝ", "y", priority=10),  # Yudh
            TransliterationRule("ܟ", "k", priority=10),  # Kaph
            TransliterationRule("ܠ", "l", priority=10),  # Lamadh
            TransliterationRule("ܡ", "m", priority=10),  # Mim
            TransliterationRule("ܢ", "n", priority=10),  # Nun
            TransliterationRule("ܣ", "s", priority=10),  # Semkath
            TransliterationRule("ܥ", "ʿ", priority=10),  # E
            TransliterationRule("ܦ", "p", priority=10),  # Pe
            TransliterationRule("ܨ", "ṣ", priority=10),  # Sadhe
            TransliterationRule("ܩ", "q", priority=10),  # Qaph
            TransliterationRule("ܪ", "r", priority=10),  # Resh
            TransliterationRule("ܫ", "š", priority=10),  # Shin
            TransliterationRule("ܬ", "t", priority=10),  # Taw
        ]

        self.rules = rules


class TransliterationValidator:
    """Validator for transliteration quality."""
    
    def __init__(self):
        """Initialize the validator."""
        self.known_schemes = set(TransliterationScheme)
        
    def validate_scheme(self, scheme: TransliterationScheme) -> bool:
        """Validate that scheme is supported.
        
        Args:
            scheme: Transliteration scheme
            
        Returns:
            True if valid scheme
        """
        return scheme in self.known_schemes
        
    def validate_transliteration(self, original: str, transliterated: str, scheme: TransliterationScheme) -> List[str]:
        """Validate transliteration quality.
        
        Args:
            original: Original text
            transliterated: Transliterated text
            scheme: Scheme used
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for empty result
        if original and not transliterated:
            errors.append("Empty transliteration result")
            
        # Check for unconverted characters
        if scheme == TransliterationScheme.SBL_HEBREW:
            # Check for remaining Hebrew characters
            if any('\u0590' <= c <= '\u05FF' for c in transliterated):
                errors.append("Unconverted Hebrew characters remain")
                
        elif scheme == TransliterationScheme.SBL_GREEK:
            # Check for remaining Greek characters
            if any('\u0370' <= c <= '\u03FF' for c in transliterated):
                errors.append("Unconverted Greek characters remain")
                
        # Check for invalid characters in output
        if '\x00' in transliterated:
            errors.append("Null characters in output")
            
        return errors
        
    def validate_reversibility(self, original: str, transliterated: str, reverse_engine: Optional['TransliterationEngine'] = None) -> bool:
        """Check if transliteration is reversible.
        
        Args:
            original: Original text
            transliterated: Transliterated text
            reverse_engine: Engine for reverse transliteration
            
        Returns:
            True if reversible
        """
        if not reverse_engine:
            return False
            
        reversed_text = reverse_engine.transliterate(transliterated)
        
        # Allow for some normalization differences
        original_normalized = unicodedata.normalize('NFC', original)
        reversed_normalized = unicodedata.normalize('NFC', reversed_text)
        
        return original_normalized == reversed_normalized


def create_transliterator(language: str, scheme: Optional[TransliterationScheme] = None) -> TransliterationEngine:
    """Create a transliterator for a language.
    
    Args:
        language: Language code (hebrew, greek, arabic, syriac)
        scheme: Optional specific scheme
        
    Returns:
        Appropriate transliterator
        
    Raises:
        ValueError: If language not supported
    """
    language = language.lower()
    
    if language == "hebrew":
        scheme = scheme or TransliterationScheme.SBL_HEBREW
        return HebrewTransliterator(scheme)
    elif language == "greek":
        scheme = scheme or TransliterationScheme.SBL_GREEK
        return GreekTransliterator(scheme)
    elif language == "arabic":
        scheme = scheme or TransliterationScheme.ARABIC_DIN
        return ArabicTransliterator(scheme)
    elif language == "syriac":
        scheme = scheme or TransliterationScheme.SBL_HEBREW  # Default scheme
        return SyriacTransliterator(scheme)
    else:
        raise ValueError(f"Unsupported language: {language}")
