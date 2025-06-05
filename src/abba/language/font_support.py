"""
Font support management for biblical texts.

Provides font requirement detection, fallback chain management, and rendering
hints for proper display of biblical scripts with special characters.
"""

from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
import logging
import unicodedata

from .script_detector import Script, ScriptDetector


class FontFeature(Enum):
    """OpenType font features for biblical texts."""

    # Hebrew features
    HEBREW_VOWELS = "hebr"  # Hebrew vowel positioning
    HEBREW_CANTILLATION = "cant"  # Cantillation marks
    HEBREW_LIGATURES = "liga"  # Hebrew ligatures

    # Greek features
    GREEK_ACCENTS = "mark"  # Greek accent positioning
    GREEK_BREATHING = "mkmk"  # Breathing marks
    GREEK_LIGATURES = "liga"  # Greek ligatures

    # Arabic features
    ARABIC_INIT = "init"  # Initial forms
    ARABIC_MEDI = "medi"  # Medial forms
    ARABIC_FINA = "fina"  # Final forms
    ARABIC_ISOL = "isol"  # Isolated forms
    ARABIC_LIGA = "liga"  # Arabic ligatures

    # General features
    KERNING = "kern"  # Kerning
    CONTEXTUAL = "calt"  # Contextual alternates
    STYLISTIC_SET = "ss01"  # Stylistic sets
    OLD_STYLE_NUMS = "onum"  # Old style numerals


@dataclass
class FontRequirements:
    """Requirements for displaying text properly."""

    scripts: Set[Script] = field(default_factory=set)
    features: Set[FontFeature] = field(default_factory=set)
    unicode_blocks: List[Tuple[int, int]] = field(default_factory=list)
    combining_marks: bool = False
    rtl_support: bool = False
    vertical_text: bool = False
    minimum_size: int = 12  # Minimum readable size

    def merge(self, other: "FontRequirements") -> "FontRequirements":
        """Merge with another requirements object."""
        return FontRequirements(
            scripts=self.scripts | other.scripts,
            features=self.features | other.features,
            unicode_blocks=list(set(self.unicode_blocks + other.unicode_blocks)),
            combining_marks=self.combining_marks or other.combining_marks,
            rtl_support=self.rtl_support or other.rtl_support,
            vertical_text=self.vertical_text or other.vertical_text,
            minimum_size=max(self.minimum_size, other.minimum_size),
        )


@dataclass
class Font:
    """Font information."""

    name: str
    family: str
    scripts: Set[Script]
    features: Set[FontFeature]
    unicode_coverage: List[Tuple[int, int]]
    is_free: bool = True
    license: str = ""
    url: str = ""

    def supports_requirements(self, reqs: FontRequirements) -> bool:
        """Check if font supports requirements."""
        # Check script support
        if not reqs.scripts.issubset(self.scripts):
            return False

        # Check feature support
        if not reqs.features.issubset(self.features):
            return False

        # Check Unicode coverage
        for req_start, req_end in reqs.unicode_blocks:
            covered = False
            for font_start, font_end in self.unicode_coverage:
                if font_start <= req_start and font_end >= req_end:
                    covered = True
                    break
            if not covered:
                return False

        return True


@dataclass
class FontFallback:
    """Font fallback configuration."""

    primary: Font
    fallbacks: List[Font] = field(default_factory=list)
    script_specific: Dict[Script, Font] = field(default_factory=dict)

    def get_font_for_script(self, script: Script) -> Font:
        """Get best font for script."""
        # Check script-specific override
        if script in self.script_specific:
            return self.script_specific[script]

        # Check primary
        if script in self.primary.scripts:
            return self.primary

        # Check fallbacks
        for font in self.fallbacks:
            if script in font.scripts:
                return font

        # Return primary as last resort
        return self.primary


class FontManager:
    """Manage font requirements and recommendations."""

    # Well-known biblical fonts
    BIBLICAL_FONTS = [
        Font(
            name="SBL Hebrew",
            family="SBL Hebrew",
            scripts={Script.HEBREW},
            features={
                FontFeature.HEBREW_VOWELS,
                FontFeature.HEBREW_CANTILLATION,
                FontFeature.HEBREW_LIGATURES,
                FontFeature.KERNING,
            },
            unicode_coverage=[(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
            is_free=True,
            license="OFL",
            url="https://www.sbl-site.org/educational/BiblicalFonts_SBLHebrew.aspx",
        ),
        Font(
            name="SBL Greek",
            family="SBL Greek",
            scripts={Script.GREEK},
            features={
                FontFeature.GREEK_ACCENTS,
                FontFeature.GREEK_BREATHING,
                FontFeature.GREEK_LIGATURES,
                FontFeature.KERNING,
            },
            unicode_coverage=[(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
            is_free=True,
            license="OFL",
            url="https://www.sbl-site.org/educational/BiblicalFonts_SBLGreek.aspx",
        ),
        Font(
            name="Ezra SIL",
            family="Ezra SIL",
            scripts={Script.HEBREW},
            features={
                FontFeature.HEBREW_VOWELS,
                FontFeature.HEBREW_CANTILLATION,
                FontFeature.KERNING,
                FontFeature.CONTEXTUAL,
            },
            unicode_coverage=[(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
            is_free=True,
            license="OFL",
            url="https://software.sil.org/ezra/",
        ),
        Font(
            name="Cardo",
            family="Cardo",
            scripts={Script.HEBREW, Script.GREEK, Script.LATIN},
            features={
                FontFeature.HEBREW_VOWELS,
                FontFeature.GREEK_ACCENTS,
                FontFeature.KERNING,
                FontFeature.CONTEXTUAL,
            },
            unicode_coverage=[
                (0x0000, 0x024F),
                (0x0370, 0x03FF),
                (0x0590, 0x05FF),
                (0x1F00, 0x1FFF),
            ],
            is_free=True,
            license="OFL",
            url="http://scholarsfonts.net/cardofnt.html",
        ),
        Font(
            name="Noto Sans Hebrew",
            family="Noto Sans",
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS, FontFeature.KERNING},
            unicode_coverage=[(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
            is_free=True,
            license="OFL",
            url="https://fonts.google.com/noto",
        ),
        Font(
            name="Noto Sans Arabic",
            family="Noto Sans",
            scripts={Script.ARABIC},
            features={
                FontFeature.ARABIC_INIT,
                FontFeature.ARABIC_MEDI,
                FontFeature.ARABIC_FINA,
                FontFeature.ARABIC_ISOL,
                FontFeature.ARABIC_LIGA,
                FontFeature.KERNING,
            },
            unicode_coverage=[(0x0600, 0x06FF), (0x0750, 0x077F)],
            is_free=True,
            license="OFL",
            url="https://fonts.google.com/noto",
        ),
        Font(
            name="Noto Sans Syriac",
            family="Noto Sans",
            scripts={Script.SYRIAC},
            features={
                FontFeature.ARABIC_INIT,
                FontFeature.ARABIC_MEDI,
                FontFeature.ARABIC_FINA,
                FontFeature.ARABIC_ISOL,
            },
            unicode_coverage=[(0x0700, 0x074F)],
            is_free=True,
            license="OFL",
            url="https://fonts.google.com/noto",
        ),
    ]

    # Default font stacks by platform
    PLATFORM_DEFAULTS = {
        "windows": {
            Script.HEBREW: ["Ezra SIL", "SBL Hebrew", "Times New Roman", "Arial"],
            Script.GREEK: ["SBL Greek", "Cardo", "Times New Roman", "Arial"],
            Script.LATIN: ["Times New Roman", "Georgia", "Arial"],
        },
        "macos": {
            Script.HEBREW: ["SBL Hebrew", "Ezra SIL", "Times", "Helvetica"],
            Script.GREEK: ["SBL Greek", "Cardo", "Times", "Helvetica"],
            Script.LATIN: ["Times", "Georgia", "Helvetica"],
        },
        "linux": {
            Script.HEBREW: ["SBL Hebrew", "Ezra SIL", "Liberation Serif", "DejaVu Sans"],
            Script.GREEK: ["SBL Greek", "Cardo", "Liberation Serif", "DejaVu Sans"],
            Script.LATIN: ["Liberation Serif", "DejaVu Serif", "DejaVu Sans"],
        },
        "web": {
            Script.HEBREW: ["SBL Hebrew", "Ezra SIL", "serif"],
            Script.GREEK: ["SBL Greek", "Cardo", "serif"],
            Script.LATIN: ["Georgia", "Times New Roman", "serif"],
        },
    }

    def __init__(self):
        """Initialize the font manager."""
        self.logger = logging.getLogger(__name__)
        self.script_detector = ScriptDetector()
        self._font_db = {font.name: font for font in self.BIBLICAL_FONTS}

    def analyze_requirements(self, text: str) -> FontRequirements:
        """Analyze text to determine font requirements.

        Args:
            text: Text to analyze

        Returns:
            Font requirements
        """
        reqs = FontRequirements()

        # Detect scripts
        scripts = self.script_detector.detect_scripts(text, min_confidence=0.01)
        reqs.scripts = set(scripts)

        # Check for RTL scripts
        if any(s in {Script.HEBREW, Script.ARABIC, Script.SYRIAC} for s in scripts):
            reqs.rtl_support = True

        # Analyze Unicode blocks needed
        unicode_blocks = set()
        for char in text:
            cp = ord(char)
            # Group into blocks of 256 characters
            block_start = (cp // 256) * 256
            block_end = block_start + 255
            unicode_blocks.add((block_start, block_end))
        reqs.unicode_blocks = sorted(list(unicode_blocks))

        # Check for combining marks
        if any(unicodedata.combining(char) > 0 for char in text):
            reqs.combining_marks = True

        # Determine required features
        if Script.HEBREW in scripts:
            reqs.features.update({FontFeature.HEBREW_VOWELS, FontFeature.KERNING})
            # Check for cantillation
            if any(0x0591 <= ord(c) <= 0x05AF for c in text):
                reqs.features.add(FontFeature.HEBREW_CANTILLATION)

        if Script.GREEK in scripts:
            reqs.features.update({FontFeature.GREEK_ACCENTS, FontFeature.KERNING})
            # Check for polytonic Greek
            if any(0x1F00 <= ord(c) <= 0x1FFF for c in text):
                reqs.features.add(FontFeature.GREEK_BREATHING)

        if Script.ARABIC in scripts or Script.SYRIAC in scripts:
            reqs.features.update(
                {
                    FontFeature.ARABIC_INIT,
                    FontFeature.ARABIC_MEDI,
                    FontFeature.ARABIC_FINA,
                    FontFeature.ARABIC_ISOL,
                    FontFeature.KERNING,
                }
            )

        # Set minimum size based on complexity
        if reqs.combining_marks:
            reqs.minimum_size = 14

        return reqs

    def recommend_fonts(
        self, requirements: FontRequirements, platform: str = "web"
    ) -> FontFallback:
        """Recommend fonts for requirements.

        Args:
            requirements: Font requirements
            platform: Target platform

        Returns:
            Font fallback configuration
        """
        suitable_fonts = []

        # Find suitable fonts
        for font in self.BIBLICAL_FONTS:
            if font.supports_requirements(requirements):
                suitable_fonts.append(font)

        if not suitable_fonts:
            # Fallback to platform defaults
            return self._get_platform_defaults(requirements, platform)

        # Sort by coverage
        suitable_fonts.sort(key=lambda f: len(f.scripts & requirements.scripts), reverse=True)

        # Build fallback chain
        fallback = FontFallback(primary=suitable_fonts[0])

        # Add additional fonts for uncovered scripts
        covered_scripts = suitable_fonts[0].scripts
        for font in suitable_fonts[1:]:
            new_scripts = font.scripts - covered_scripts
            if new_scripts:
                fallback.fallbacks.append(font)
                covered_scripts |= font.scripts

        # Add script-specific overrides
        for script in requirements.scripts:
            if script == Script.HEBREW:
                if "SBL Hebrew" in self._font_db:
                    fallback.script_specific[Script.HEBREW] = self._font_db["SBL Hebrew"]
            elif script == Script.GREEK:
                if "SBL Greek" in self._font_db:
                    fallback.script_specific[Script.GREEK] = self._font_db["SBL Greek"]

        return fallback

    def _get_platform_defaults(self, requirements: FontRequirements, platform: str) -> FontFallback:
        """Get platform-specific default fonts."""
        if platform not in self.PLATFORM_DEFAULTS:
            platform = "web"

        defaults = self.PLATFORM_DEFAULTS[platform]

        # Find primary script
        primary_script = None
        for script in [Script.HEBREW, Script.GREEK, Script.LATIN]:
            if script in requirements.scripts:
                primary_script = script
                break

        if not primary_script:
            primary_script = Script.LATIN

        # Build fallback from platform defaults
        font_names = defaults.get(primary_script, ["serif"])

        # Create pseudo-fonts for platform fonts
        primary = Font(
            name=font_names[0],
            family=font_names[0],
            scripts=requirements.scripts,
            features=set(),
            unicode_coverage=requirements.unicode_blocks,
        )

        fallback = FontFallback(primary=primary)

        for font_name in font_names[1:]:
            fallback_font = Font(
                name=font_name,
                family=font_name,
                scripts=requirements.scripts,
                features=set(),
                unicode_coverage=requirements.unicode_blocks,
            )
            fallback.fallbacks.append(fallback_font)

        return fallback

    def generate_css_font_stack(
        self, fallback: FontFallback, include_web_fonts: bool = True
    ) -> Dict[str, str]:
        """Generate CSS font declarations.

        Args:
            fallback: Font fallback configuration
            include_web_fonts: Whether to include @font-face declarations

        Returns:
            CSS declarations
        """
        css = {}

        # Build font-family list
        font_names = [f'"{fallback.primary.name}"']
        for font in fallback.fallbacks:
            font_names.append(f'"{font.name}"')
        font_names.append("serif")  # Generic fallback

        css["font-family"] = ", ".join(font_names)

        # Add @font-face declarations if requested
        if include_web_fonts:
            font_faces = []

            for font in [fallback.primary] + fallback.fallbacks:
                if font.is_free and font.url:
                    font_face = f"""@font-face {{
    font-family: "{font.name}";
    src: url("{font.url}");
    font-display: swap;
}}"""
                    font_faces.append(font_face)

            if font_faces:
                css["font-faces"] = "\n\n".join(font_faces)

        # Add feature settings
        features = []
        if FontFeature.KERNING in fallback.primary.features:
            features.append('"kern" 1')
        if FontFeature.CONTEXTUAL in fallback.primary.features:
            features.append('"calt" 1')

        if features:
            css["font-feature-settings"] = ", ".join(features)

        return css

    def check_font_availability(self, font_name: str) -> bool:
        """Check if font is available in our database.

        Args:
            font_name: Font name to check

        Returns:
            True if available
        """
        return font_name in self._font_db

    def get_font_info(self, font_name: str) -> Optional[Font]:
        """Get information about a font.

        Args:
            font_name: Font name

        Returns:
            Font information or None
        """
        return self._font_db.get(font_name)

    def list_fonts_for_script(self, script: Script) -> List[Font]:
        """List all fonts supporting a script.

        Args:
            script: Script to check

        Returns:
            List of supporting fonts
        """
        fonts = []
        for font in self.BIBLICAL_FONTS:
            if script in font.scripts:
                fonts.append(font)

        return fonts


@dataclass
class FontChain:
    """Font chain for text rendering."""
    
    fonts: List[Font]
    features: Set[FontFeature] = field(default_factory=set)
    
    def to_css(self) -> str:
        """Convert to CSS font-family."""
        names = [f'"{font.name}"' for font in self.fonts]
        names.append("serif")  # Generic fallback
        return ", ".join(names)
        

@dataclass 
class RenderingHints:
    """Rendering hints for proper text display."""
    
    font_chain: FontChain
    direction: str = "ltr"  # ltr or rtl
    line_height: float = 1.5
    letter_spacing: float = 0.0
    features: Dict[str, str] = field(default_factory=dict)
    
    def to_css(self) -> Dict[str, str]:
        """Convert to CSS properties."""
        css = {
            "font-family": self.font_chain.to_css(),
            "direction": self.direction,
            "line-height": str(self.line_height),
        }
        
        if self.letter_spacing != 0:
            css["letter-spacing"] = f"{self.letter_spacing}em"
            
        if self.features:
            feature_settings = []
            for feature, value in self.features.items():
                feature_settings.append(f'"{feature}" {value}')
            css["font-feature-settings"] = ", ".join(feature_settings)
            
        return css


def detect_font_requirements(text: str) -> FontRequirements:
    """Detect font requirements for text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Font requirements
    """
    manager = FontManager()
    return manager.analyze_requirements(text)
