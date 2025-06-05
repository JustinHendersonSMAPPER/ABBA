"""Tests for font support management."""

import pytest

from abba.language.font_support import (
    FontManager,
    FontRequirements,
    FontFallback,
    Font,
    FontFeature,
)
from abba.language.script_detector import Script


class TestFontRequirements:
    """Test font requirements."""

    def test_font_requirements_creation(self):
        """Test creating font requirements."""
        reqs = FontRequirements(
            scripts={Script.HEBREW, Script.GREEK},
            features={FontFeature.HEBREW_VOWELS, FontFeature.KERNING},
            unicode_blocks=[(0x0590, 0x05FF)],
            combining_marks=True,
            rtl_support=True,
            minimum_size=14,
        )

        assert Script.HEBREW in reqs.scripts
        assert FontFeature.HEBREW_VOWELS in reqs.features
        assert reqs.combining_marks
        assert reqs.rtl_support
        assert reqs.minimum_size == 14

    def test_font_requirements_merge(self):
        """Test merging font requirements."""
        reqs1 = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS},
            unicode_blocks=[(0x0590, 0x05FF)],
        )

        reqs2 = FontRequirements(
            scripts={Script.GREEK},
            features={FontFeature.GREEK_ACCENTS},
            unicode_blocks=[(0x0370, 0x03FF)],
        )

        merged = reqs1.merge(reqs2)

        assert Script.HEBREW in merged.scripts
        assert Script.GREEK in merged.scripts
        assert FontFeature.HEBREW_VOWELS in merged.features
        assert FontFeature.GREEK_ACCENTS in merged.features
        assert len(merged.unicode_blocks) == 2


class TestFont:
    """Test font model."""

    def test_font_creation(self):
        """Test creating font."""
        font = Font(
            name="SBL Hebrew",
            family="SBL Hebrew",
            scripts={Script.HEBREW},
            features={FontFeature.HEBREW_VOWELS, FontFeature.KERNING},
            unicode_coverage=[(0x0590, 0x05FF)],
            is_free=True,
            license="OFL",
        )

        assert font.name == "SBL Hebrew"
        assert Script.HEBREW in font.scripts
        assert font.is_free

    def test_font_supports_requirements(self):
        """Test checking font support for requirements."""
        font = Font(
            name="Test Font",
            family="Test",
            scripts={Script.HEBREW, Script.LATIN},
            features={FontFeature.HEBREW_VOWELS, FontFeature.KERNING},
            unicode_coverage=[(0x0000, 0x00FF), (0x0590, 0x05FF)],
        )

        # Supported requirements
        reqs1 = FontRequirements(
            scripts={Script.HEBREW},
            features={FontFeature.KERNING},
            unicode_blocks=[(0x0590, 0x05FF)],
        )
        assert font.supports_requirements(reqs1)

        # Unsupported script
        reqs2 = FontRequirements(scripts={Script.GREEK}, features={FontFeature.KERNING})
        assert not font.supports_requirements(reqs2)

        # Unsupported feature
        reqs3 = FontRequirements(
            scripts={Script.HEBREW}, features={FontFeature.HEBREW_CANTILLATION}
        )
        assert not font.supports_requirements(reqs3)


class TestFontFallback:
    """Test font fallback configuration."""

    def test_font_fallback_creation(self):
        """Test creating font fallback."""
        primary = Font(name="Primary", family="Primary", scripts={Script.LATIN})

        fallback = FontFallback(primary=primary)

        assert fallback.primary == primary
        assert fallback.fallbacks == []

    def test_get_font_for_script(self):
        """Test getting font for specific script."""
        primary = Font(name="Primary", family="Primary", scripts={Script.LATIN, Script.GREEK})

        hebrew_font = Font(name="Hebrew Font", family="Hebrew", scripts={Script.HEBREW})

        fallback = FontFallback(primary=primary, fallbacks=[hebrew_font])

        # Primary supports Greek
        assert fallback.get_font_for_script(Script.GREEK) == primary

        # Fallback supports Hebrew
        assert fallback.get_font_for_script(Script.HEBREW) == hebrew_font

        # Script-specific override
        special_hebrew = Font(name="Special Hebrew", family="Special", scripts={Script.HEBREW})
        fallback.script_specific[Script.HEBREW] = special_hebrew
        assert fallback.get_font_for_script(Script.HEBREW) == special_hebrew


class TestFontManager:
    """Test font manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = FontManager()

    def test_analyze_hebrew_requirements(self):
        """Test analyzing Hebrew text requirements."""
        hebrew_text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        reqs = self.manager.analyze_requirements(hebrew_text)

        assert Script.HEBREW in reqs.scripts
        assert reqs.rtl_support
        assert reqs.combining_marks
        assert FontFeature.HEBREW_VOWELS in reqs.features

    def test_analyze_greek_requirements(self):
        """Test analyzing Greek text requirements."""
        greek_text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
        reqs = self.manager.analyze_requirements(greek_text)

        assert Script.GREEK in reqs.scripts
        assert not reqs.rtl_support
        assert reqs.combining_marks
        assert FontFeature.GREEK_ACCENTS in reqs.features

    def test_analyze_mixed_requirements(self):
        """Test analyzing mixed script requirements."""
        mixed_text = "The Hebrew word בְּרֵאשִׁית and Greek λόγος"
        reqs = self.manager.analyze_requirements(mixed_text)

        assert Script.HEBREW in reqs.scripts
        assert Script.GREEK in reqs.scripts
        assert Script.LATIN in reqs.scripts
        assert reqs.rtl_support  # Because of Hebrew

    def test_analyze_cantillation_requirements(self):
        """Test detecting cantillation mark requirements."""
        # Text with cantillation marks
        cantillation_text = "וַיֹּ֣אמֶר אֱלֹהִ֔ים"  # Has accent marks
        reqs = self.manager.analyze_requirements(cantillation_text)

        assert FontFeature.HEBREW_CANTILLATION in reqs.features

    def test_recommend_fonts_hebrew(self):
        """Test font recommendations for Hebrew."""
        reqs = FontRequirements(scripts={Script.HEBREW}, features={FontFeature.HEBREW_VOWELS})

        fallback = self.manager.recommend_fonts(reqs, platform="web")

        assert fallback.primary is not None
        assert Script.HEBREW in fallback.primary.scripts

    def test_recommend_fonts_multiple_scripts(self):
        """Test font recommendations for multiple scripts."""
        reqs = FontRequirements(scripts={Script.HEBREW, Script.GREEK, Script.LATIN})

        fallback = self.manager.recommend_fonts(reqs, platform="web")

        # Should have fonts covering all scripts
        all_scripts = set()
        all_scripts.update(fallback.primary.scripts)
        for font in fallback.fallbacks:
            all_scripts.update(font.scripts)

        assert Script.HEBREW in all_scripts
        assert Script.GREEK in all_scripts
        assert Script.LATIN in all_scripts

    def test_platform_defaults(self):
        """Test platform-specific defaults."""
        reqs = FontRequirements(scripts={Script.HEBREW})

        # Test different platforms
        web_fallback = self.manager.recommend_fonts(reqs, platform="web")
        windows_fallback = self.manager.recommend_fonts(reqs, platform="windows")
        macos_fallback = self.manager.recommend_fonts(reqs, platform="macos")

        assert web_fallback is not None
        assert windows_fallback is not None
        assert macos_fallback is not None

    def test_generate_css_font_stack(self):
        """Test CSS font stack generation."""
        fallback = FontFallback(
            primary=Font(
                name="SBL Hebrew",
                family="SBL Hebrew",
                scripts={Script.HEBREW},
                features={FontFeature.KERNING},
            ),
            fallbacks=[Font(name="Ezra SIL", family="Ezra SIL", scripts={Script.HEBREW})],
        )

        css = self.manager.generate_css_font_stack(fallback)

        assert "font-family" in css
        assert '"SBL Hebrew"' in css["font-family"]
        assert '"Ezra SIL"' in css["font-family"]
        assert "serif" in css["font-family"]

        # Check feature settings
        assert "font-feature-settings" in css
        assert '"kern" 1' in css["font-feature-settings"]

    def test_generate_css_with_webfonts(self):
        """Test CSS generation with @font-face declarations."""
        fallback = FontFallback(primary=self.manager._font_db.get("SBL Hebrew"))

        css = self.manager.generate_css_font_stack(fallback, include_web_fonts=True)

        if "font_faces" in css:
            assert "@font-face" in css["font_faces"]
            assert "SBL Hebrew" in css["font_faces"]

    def test_check_font_availability(self):
        """Test checking font availability."""
        assert self.manager.check_font_availability("SBL Hebrew")
        assert self.manager.check_font_availability("SBL Greek")
        assert not self.manager.check_font_availability("Nonexistent Font")

    def test_get_font_info(self):
        """Test getting font information."""
        info = self.manager.get_font_info("SBL Hebrew")

        assert info is not None
        assert info.name == "SBL Hebrew"
        assert Script.HEBREW in info.scripts
        assert info.is_free

    def test_list_fonts_for_script(self):
        """Test listing fonts for a script."""
        hebrew_fonts = self.manager.list_fonts_for_script(Script.HEBREW)

        assert len(hebrew_fonts) > 0
        assert all(Script.HEBREW in f.scripts for f in hebrew_fonts)

        # Check known fonts are included
        font_names = [f.name for f in hebrew_fonts]
        assert "SBL Hebrew" in font_names or "Ezra SIL" in font_names
