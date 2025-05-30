"""Tests for translation repository."""

import pytest
import json
from pathlib import Path
import tempfile
from abba.canon.translation import TranslationRepository
from abba.canon.models import Translation, TranslationPhilosophy, LicenseType


class TestTranslationRepository:
    """Test TranslationRepository functionality."""

    @pytest.fixture
    def repository(self):
        """Create a translation repository for testing."""
        return TranslationRepository()

    def test_repository_initialization(self, repository):
        """Test repository initializes with default translations."""
        # Should have translations loaded
        translations = repository.list_translations()
        assert len(translations) > 0

        # Check some key translations exist
        assert repository.get_translation("kjv") is not None
        assert repository.get_translation("niv") is not None
        assert repository.get_translation("esv") is not None
        assert repository.get_translation("web") is not None  # Public domain

    def test_get_translation(self, repository):
        """Test retrieving translations."""
        kjv = repository.get_translation("kjv")
        assert kjv is not None
        assert kjv.name == "King James Version"
        assert kjv.abbreviation == "KJV"
        assert kjv.language_code == "en"
        assert kjv.license_type == LicenseType.PUBLIC_DOMAIN

        # Test non-existent translation
        assert repository.get_translation("invalid") is None

    def test_get_translation_by_abbreviation(self, repository):
        """Test retrieving translations by abbreviation."""
        esv = repository.get_translation_by_abbreviation("ESV")
        assert esv is not None
        assert esv.id == "esv"

        # Test case insensitivity
        esv_lower = repository.get_translation_by_abbreviation("esv")
        assert esv_lower is not None
        assert esv_lower.id == "esv"

        # Test non-existent abbreviation
        assert repository.get_translation_by_abbreviation("XXX") is None

    def test_get_translations_by_language(self, repository):
        """Test filtering translations by language."""
        # English translations
        english = repository.get_translations_by_language("en")
        assert len(english) > 5  # Should have multiple English translations
        assert all(t.language_code == "en" for t in english)

        # Spanish translations
        spanish = repository.get_translations_by_language("es")
        assert len(spanish) > 0
        assert all(t.language_code == "es" for t in spanish)

        # German translations
        german = repository.get_translations_by_language("de")
        assert len(german) > 0
        assert any(t.id == "lut2017" for t in german)

    def test_get_translations_by_canon(self, repository):
        """Test filtering translations by canon."""
        protestant = repository.get_translations_by_canon("protestant")
        assert len(protestant) > 0

        catholic = repository.get_translations_by_canon("catholic")
        assert len(catholic) > 0

        # Catholic translations should include specific ones
        catholic_ids = [t.id for t in catholic]
        assert "drb" in catholic_ids  # Douay-Rheims
        assert "nabre" in catholic_ids  # NABRE

    def test_get_translations_by_license(self, repository):
        """Test filtering translations by license type."""
        public_domain = repository.get_translations_by_license(LicenseType.PUBLIC_DOMAIN)
        assert len(public_domain) > 0
        assert all(t.is_public_domain() for t in public_domain)

        # KJV should be public domain
        pd_ids = [t.id for t in public_domain]
        assert "kjv" in pd_ids
        assert "web" in pd_ids

        restricted = repository.get_translations_by_license(LicenseType.RESTRICTED)
        assert len(restricted) > 0
        assert all(t.license_type == LicenseType.RESTRICTED for t in restricted)

    def test_get_public_domain_translations(self, repository):
        """Test getting all public domain translations."""
        pd_translations = repository.get_public_domain_translations()
        assert len(pd_translations) > 0
        assert all(t.is_public_domain() for t in pd_translations)

    def test_get_digital_translations(self, repository):
        """Test getting digitally available translations."""
        digital = repository.get_digital_translations()
        assert len(digital) > 0
        assert all(t.allows_digital_use() for t in digital)

    def test_get_api_accessible_translations(self, repository):
        """Test getting API-accessible translations."""
        api_accessible = repository.get_api_accessible_translations()
        assert len(api_accessible) > 0

        # Public domain should be API accessible
        for trans in api_accessible:
            assert trans.api_access or trans.is_public_domain()

    def test_search_translations(self, repository):
        """Test searching translations with multiple criteria."""
        # English formal equivalence translations
        results = repository.search_translations(
            language="en", philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE
        )
        assert len(results) > 0
        assert all(t.language_code == "en" for t in results)
        assert all(t.philosophy == TranslationPhilosophy.FORMAL_EQUIVALENCE for t in results)

        # Digital-only Protestant translations
        results = repository.search_translations(canon="protestant", digital_only=True)
        assert len(results) > 0
        assert all(t.allows_digital_use() for t in results)

        # Translations by year range
        results = repository.search_translations(year_range=(1900, 2000))
        assert len(results) > 0
        for trans in results:
            assert trans.year_published is not None
            assert 1900 <= trans.year_published <= 2000

    def test_get_translation_families(self, repository):
        """Test grouping translations by base text."""
        families = repository.get_translation_families()

        # Should have major text families
        assert "Textus Receptus" in families
        assert "NA28" in families or "NA27/28" in families
        assert "BHS" in families
        assert "Masoretic Text" in families

        # Each family should have translations
        for base_text, translations in families.items():
            assert len(translations) > 0

    def test_add_custom_translation(self, repository):
        """Test adding a custom translation."""
        custom = Translation(
            id="custom",
            name="Custom Translation",
            abbreviation="CST",
            language_code="en",
            canon_id="protestant",
            versification_scheme_id="standard",
            philosophy=TranslationPhilosophy.DYNAMIC_EQUIVALENCE,
            year_published=2024,
            license_type=LicenseType.OPEN_LICENSE,
        )

        repository.add_translation(custom)

        # Should be retrievable
        retrieved = repository.get_translation("custom")
        assert retrieved is not None
        assert retrieved.name == "Custom Translation"

        # Should appear in searches
        assert "custom" in repository.list_translations()

        # Should be in language index
        english = repository.get_translations_by_language("en")
        assert any(t.id == "custom" for t in english)

    def test_export_metadata(self, repository):
        """Test exporting translation metadata."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            repository.export_metadata(temp_path)

            # Load and verify exported data
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "translations" in data
            assert "export_date" in data
            assert "total_translations" in data

            # Check translation count matches
            assert len(data["translations"]) == data["total_translations"]

            # Check a specific translation
            kjv_data = next((t for t in data["translations"] if t["id"] == "kjv"), None)
            assert kjv_data is not None
            assert kjv_data["name"] == "King James Version"
            assert kjv_data["license"] == "public_domain"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_from_file(self):
        """Test loading translations from file."""
        # Create test data
        test_data = {
            "translations": [
                {
                    "id": "test1",
                    "name": "Test Translation 1",
                    "abbreviation": "TT1",
                    "language": "en",
                    "canon": "protestant",
                    "philosophy": "formal",
                    "year": 2020,
                    "license": "public_domain",
                    "digital": True,
                    "api": True,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Create new repository and load data
            repo = TranslationRepository()
            initial_count = len(repo.list_translations())

            repo.load_from_file(temp_path)

            # Check translation was loaded
            test1 = repo.get_translation("test1")
            assert test1 is not None
            assert test1.name == "Test Translation 1"
            assert test1.year_published == 2020

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_get_statistics(self, repository):
        """Test getting repository statistics."""
        stats = repository.get_statistics()

        assert "total_translations" in stats
        assert stats["total_translations"] > 0

        assert "languages" in stats
        assert stats["languages"] > 0

        assert "public_domain" in stats
        assert stats["public_domain"] > 0

        assert "digital_available" in stats
        assert "api_accessible" in stats

        assert "by_philosophy" in stats
        assert "formal" in stats["by_philosophy"]
        assert "dynamic" in stats["by_philosophy"]

        assert "by_license" in stats
        assert "public_domain" in stats["by_license"]
        assert "restricted" in stats["by_license"]

    def test_translation_properties(self, repository):
        """Test various translation properties."""
        # Test specific translations
        kjv = repository.get_translation("kjv")
        assert kjv.is_public_domain()
        assert kjv.allows_digital_use()
        assert kjv.get_attribution_text() == ""  # No attribution for public domain

        esv = repository.get_translation("esv")
        assert not esv.is_public_domain()
        assert esv.allows_digital_use()  # Has digital_distribution=True
        assert "English Standard Version" in esv.get_attribution_text()
        assert "Crossway" in esv.get_attribution_text()

        # Test original language texts
        sblgnt = repository.get_translation("sblgnt")
        assert sblgnt is not None
        assert sblgnt.language_code == "grc"  # Ancient Greek

        wlc = repository.get_translation("wlc")
        assert wlc is not None
        assert wlc.language_code == "hbo"  # Ancient Hebrew
        assert wlc.script_direction == "rtl"
