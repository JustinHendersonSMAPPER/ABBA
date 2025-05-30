"""
Translation repository for managing Bible translations.

Provides storage and retrieval of Bible translations with their metadata,
licensing information, and canon associations.
"""

import logging
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from datetime import datetime

from .models import Translation, TranslationPhilosophy, LicenseType, Canon, VersificationScheme


class TranslationRepository:
    """Repository for managing Bible translations."""

    def __init__(self, data_path: Optional[str] = None):
        """Initialize the translation repository.

        Args:
            data_path: Optional path to translation data directory
        """
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(data_path) if data_path else None

        # Translation storage
        self._translations: Dict[str, Translation] = {}

        # Indexes for efficient lookup
        self._by_language: Dict[str, List[str]] = {}
        self._by_canon: Dict[str, List[str]] = {}
        self._by_license: Dict[LicenseType, List[str]] = {}
        self._by_year: Dict[int, List[str]] = {}

        # Initialize default translations
        self._initialize_default_translations()

    def _initialize_default_translations(self):
        """Initialize repository with common translations."""
        # English translations
        self._add_english_translations()

        # Spanish translations
        self._add_spanish_translations()

        # Other major language translations
        self._add_other_translations()

    def _add_english_translations(self):
        """Add common English translations."""
        translations = [
            Translation(
                id="kjv",
                name="King James Version",
                abbreviation="KJV",
                language_code="en",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["Textus Receptus", "Masoretic Text"],
                year_published=1611,
                year_revised=1769,
                publisher="Cambridge University Press",
                license_type=LicenseType.PUBLIC_DOMAIN,
                digital_distribution=True,
                api_access=True,
                description="The Authorized Version, a landmark English translation",
            ),
            Translation(
                id="niv",
                name="New International Version",
                abbreviation="NIV",
                language_code="en",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.DYNAMIC_EQUIVALENCE,
                base_texts=["NA27/28", "BHS"],
                year_published=1978,
                year_revised=2011,
                publisher="Biblica",
                copyright_holder="Biblica, Inc.",
                license_type=LicenseType.RESTRICTED,
                license_details="Usage restrictions apply",
                digital_distribution=True,
                api_access=False,
                quotation_limit=500,
                attribution_required=True,
                commercial_use=False,
                description="Popular modern English translation",
            ),
            Translation(
                id="esv",
                name="English Standard Version",
                abbreviation="ESV",
                language_code="en",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["NA28", "BHS"],
                year_published=2001,
                year_revised=2016,
                publisher="Crossway",
                copyright_holder="Crossway",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                api_access=True,
                quotation_limit=1000,
                attribution_required=True,
                commercial_use=False,
                description="Essentially literal modern translation",
            ),
            Translation(
                id="nasb",
                name="New American Standard Bible",
                abbreviation="NASB",
                language_code="en",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["NA28", "BHS"],
                year_published=1971,
                year_revised=2020,
                publisher="The Lockman Foundation",
                copyright_holder="The Lockman Foundation",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                api_access=False,
                quotation_limit=500,
                attribution_required=True,
                description="Highly literal translation",
            ),
            Translation(
                id="nrsv",
                name="New Revised Standard Version",
                abbreviation="NRSV",
                language_code="en",
                canon_id="protestant",  # Also available in Catholic edition
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["NA28", "BHS"],
                year_published=1989,
                publisher="National Council of Churches",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                features=["gender-inclusive language", "scholarly"],
                description="Academic standard translation",
            ),
            Translation(
                id="nlt",
                name="New Living Translation",
                abbreviation="NLT",
                language_code="en",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.DYNAMIC_EQUIVALENCE,
                base_texts=["NA28", "BHS"],
                year_published=1996,
                year_revised=2015,
                publisher="Tyndale House Publishers",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                description="Thought-for-thought translation",
            ),
            Translation(
                id="msg",
                name="The Message",
                abbreviation="MSG",
                language_code="en",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.PARAPHRASE,
                base_texts=["Various"],
                year_published=2002,
                publisher="NavPress",
                translators=["Eugene H. Peterson"],
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                description="Contemporary paraphrase",
            ),
            Translation(
                id="web",
                name="World English Bible",
                abbreviation="WEB",
                language_code="en",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["Majority Text", "BHS"],
                year_published=2000,
                license_type=LicenseType.PUBLIC_DOMAIN,
                digital_distribution=True,
                api_access=True,
                attribution_required=False,
                commercial_use=True,
                description="Modern public domain translation",
            ),
            Translation(
                id="drb",
                name="Douay-Rheims Bible",
                abbreviation="DRB",
                language_code="en",
                canon_id="catholic",
                versification_scheme_id="vulgate",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["Latin Vulgate"],
                year_published=1609,
                year_revised=1752,  # Challoner revision
                license_type=LicenseType.PUBLIC_DOMAIN,
                digital_distribution=True,
                api_access=True,
                description="Traditional Catholic English translation",
            ),
            Translation(
                id="nabre",
                name="New American Bible Revised Edition",
                abbreviation="NABRE",
                language_code="en",
                canon_id="catholic",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.DYNAMIC_EQUIVALENCE,
                base_texts=["NA28", "BHS"],
                year_published=1970,
                year_revised=2011,
                publisher="United States Conference of Catholic Bishops",
                church_approval="USCCB",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                description="Official Catholic Bible in the United States",
            ),
            Translation(
                id="rsv-ce",
                name="Revised Standard Version Catholic Edition",
                abbreviation="RSV-CE",
                language_code="en",
                canon_id="catholic",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["Critical Text", "BHS"],
                year_published=1966,
                publisher="Ignatius Press",
                church_approval="Catholic Church",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                description="RSV adapted for Catholic use",
            ),
        ]

        for translation in translations:
            self.add_translation(translation)

    def _add_spanish_translations(self):
        """Add common Spanish translations."""
        translations = [
            Translation(
                id="rvr1960",
                name="Reina-Valera 1960",
                abbreviation="RVR1960",
                language_code="es",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["Textus Receptus", "Masoretic Text"],
                year_published=1569,
                year_revised=1960,
                publisher="Sociedades Bíblicas Unidas",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                description="Most popular Spanish Protestant Bible",
            ),
            Translation(
                id="nvi",
                name="Nueva Versión Internacional",
                abbreviation="NVI",
                language_code="es",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.DYNAMIC_EQUIVALENCE,
                base_texts=["NA27", "BHS"],
                year_published=1999,
                publisher="Biblica",
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                description="Spanish NIV",
            ),
            Translation(
                id="bhti",
                name="Biblia Hispanoamericana",
                abbreviation="BHTI",
                language_code="es",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.DYNAMIC_EQUIVALENCE,
                base_texts=["NA28", "BHS"],
                year_published=2017,
                license_type=LicenseType.OPEN_LICENSE,
                digital_distribution=True,
                api_access=True,
                description="Open license Spanish translation",
            ),
        ]

        for translation in translations:
            self.add_translation(translation)

    def _add_other_translations(self):
        """Add translations in other major languages."""
        translations = [
            # German
            Translation(
                id="lut2017",
                name="Lutherbibel 2017",
                abbreviation="LUT2017",
                language_code="de",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["NA28", "BHS"],
                year_published=1534,
                year_revised=2017,
                publisher="Deutsche Bibelgesellschaft",
                translators=["Martin Luther", "Revision Committee"],
                license_type=LicenseType.RESTRICTED,
                digital_distribution=True,
                description="Luther's German Bible, 2017 revision",
            ),
            # French
            Translation(
                id="lsg",
                name="Louis Segond",
                abbreviation="LSG",
                language_code="fr",
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["Textus Receptus", "Masoretic Text"],
                year_published=1910,
                license_type=LicenseType.PUBLIC_DOMAIN,
                digital_distribution=True,
                api_access=True,
                description="Classic French Protestant Bible",
            ),
            # Greek
            Translation(
                id="sblgnt",
                name="SBL Greek New Testament",
                abbreviation="SBLGNT",
                language_code="grc",  # Ancient Greek
                canon_id="protestant",
                versification_scheme_id="standard",
                philosophy=TranslationPhilosophy.LITERAL,
                base_texts=["Critical Text"],
                year_published=2010,
                publisher="Society of Biblical Literature",
                license_type=LicenseType.OPEN_LICENSE,
                license_details="CC BY 4.0",
                digital_distribution=True,
                api_access=True,
                commercial_use=True,
                description="Critical edition of Greek NT",
                features=["critical apparatus", "textual notes"],
            ),
            # Hebrew
            Translation(
                id="wlc",
                name="Westminster Leningrad Codex",
                abbreviation="WLC",
                language_code="hbo",  # Ancient Hebrew
                canon_id="protestant",
                versification_scheme_id="masoretic",
                philosophy=TranslationPhilosophy.LITERAL,
                base_texts=["Leningrad Codex"],
                year_published=2008,  # Digital edition
                license_type=LicenseType.OPEN_LICENSE,
                digital_distribution=True,
                api_access=True,
                commercial_use=True,
                script_direction="rtl",
                uses_diacritics=True,
                description="Digital Masoretic Text",
            ),
            # Latin
            Translation(
                id="vulgate",
                name="Biblia Sacra Vulgata",
                abbreviation="VULG",
                language_code="la",
                canon_id="catholic",
                versification_scheme_id="vulgate",
                philosophy=TranslationPhilosophy.FORMAL_EQUIVALENCE,
                base_texts=["Hebrew", "Greek", "Old Latin"],
                year_published=405,  # Jerome's completion
                year_revised=1979,  # Nova Vulgata
                translators=["Jerome"],
                license_type=LicenseType.PUBLIC_DOMAIN,
                digital_distribution=True,
                api_access=True,
                description="Jerome's Latin Bible",
            ),
        ]

        for translation in translations:
            self.add_translation(translation)

    def add_translation(self, translation: Translation) -> None:
        """Add a translation to the repository."""
        self._translations[translation.id] = translation

        # Update indexes
        if translation.language_code not in self._by_language:
            self._by_language[translation.language_code] = []
        self._by_language[translation.language_code].append(translation.id)

        if translation.canon_id not in self._by_canon:
            self._by_canon[translation.canon_id] = []
        self._by_canon[translation.canon_id].append(translation.id)

        if translation.license_type not in self._by_license:
            self._by_license[translation.license_type] = []
        self._by_license[translation.license_type].append(translation.id)

        if translation.year_published:
            if translation.year_published not in self._by_year:
                self._by_year[translation.year_published] = []
            self._by_year[translation.year_published].append(translation.id)

        self.logger.info(f"Added translation: {translation.name} ({translation.abbreviation})")

    def get_translation(self, translation_id: str) -> Optional[Translation]:
        """Get a translation by ID."""
        return self._translations.get(translation_id)

    def get_translation_by_abbreviation(self, abbreviation: str) -> Optional[Translation]:
        """Get a translation by abbreviation."""
        for translation in self._translations.values():
            if translation.abbreviation.upper() == abbreviation.upper():
                return translation
        return None

    def list_translations(self) -> List[str]:
        """List all translation IDs."""
        return list(self._translations.keys())

    def get_translations_by_language(self, language_code: str) -> List[Translation]:
        """Get all translations for a specific language."""
        translation_ids = self._by_language.get(language_code, [])
        return [self._translations[tid] for tid in translation_ids]

    def get_translations_by_canon(self, canon_id: str) -> List[Translation]:
        """Get all translations using a specific canon."""
        translation_ids = self._by_canon.get(canon_id, [])
        return [self._translations[tid] for tid in translation_ids]

    def get_translations_by_license(self, license_type: LicenseType) -> List[Translation]:
        """Get all translations with a specific license type."""
        translation_ids = self._by_license.get(license_type, [])
        return [self._translations[tid] for tid in translation_ids]

    def get_public_domain_translations(self) -> List[Translation]:
        """Get all public domain translations."""
        return self.get_translations_by_license(LicenseType.PUBLIC_DOMAIN)

    def get_digital_translations(self) -> List[Translation]:
        """Get all translations available for digital distribution."""
        return [t for t in self._translations.values() if t.allows_digital_use()]

    def get_api_accessible_translations(self) -> List[Translation]:
        """Get all translations accessible via API."""
        return [t for t in self._translations.values() if t.api_access or t.is_public_domain()]

    def search_translations(
        self,
        language: Optional[str] = None,
        canon: Optional[str] = None,
        philosophy: Optional[TranslationPhilosophy] = None,
        digital_only: bool = False,
        api_only: bool = False,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> List[Translation]:
        """Search translations with multiple criteria."""
        results = list(self._translations.values())

        if language:
            results = [t for t in results if t.language_code == language]

        if canon:
            results = [t for t in results if t.canon_id == canon]

        if philosophy:
            results = [t for t in results if t.philosophy == philosophy]

        if digital_only:
            results = [t for t in results if t.allows_digital_use()]

        if api_only:
            results = [t for t in results if t.api_access or t.is_public_domain()]

        if year_range:
            start_year, end_year = year_range
            results = [
                t
                for t in results
                if t.year_published and start_year <= t.year_published <= end_year
            ]

        return results

    def get_translation_families(self) -> Dict[str, List[Translation]]:
        """Group translations by their base text families."""
        families = {}

        for translation in self._translations.values():
            for base_text in translation.base_texts:
                if base_text not in families:
                    families[base_text] = []
                families[base_text].append(translation)

        return families

    def export_metadata(self, output_path: str, format: str = "json") -> None:
        """Export translation metadata to file."""
        metadata = {
            "translations": [
                {
                    "id": t.id,
                    "name": t.name,
                    "abbreviation": t.abbreviation,
                    "language": t.language_code,
                    "canon": t.canon_id,
                    "philosophy": t.philosophy.value,
                    "year": t.year_published,
                    "license": t.license_type.value,
                    "digital": t.digital_distribution,
                    "api": t.api_access,
                }
                for t in self._translations.values()
            ],
            "export_date": datetime.utcnow().isoformat(),
            "total_translations": len(self._translations),
        }

        output_file = Path(output_path)
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported {len(self._translations)} translations to {output_path}")

    def load_from_file(self, file_path: str) -> None:
        """Load translations from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for trans_data in data.get("translations", []):
            translation = Translation(
                id=trans_data["id"],
                name=trans_data["name"],
                abbreviation=trans_data["abbreviation"],
                language_code=trans_data["language"],
                canon_id=trans_data["canon"],
                versification_scheme_id=trans_data.get("versification", "standard"),
                philosophy=TranslationPhilosophy(trans_data.get("philosophy", "formal")),
                year_published=trans_data.get("year"),
                license_type=LicenseType(trans_data.get("license", "restricted")),
                digital_distribution=trans_data.get("digital", False),
                api_access=trans_data.get("api", False),
            )
            self.add_translation(translation)

        self.logger.info(
            f"Loaded {len(data.get('translations', []))} translations from {file_path}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return {
            "total_translations": len(self._translations),
            "languages": len(self._by_language),
            "canons": len(self._by_canon),
            "public_domain": len(self.get_public_domain_translations()),
            "digital_available": len(self.get_digital_translations()),
            "api_accessible": len(self.get_api_accessible_translations()),
            "by_philosophy": {
                philosophy.value: len(
                    [t for t in self._translations.values() if t.philosophy == philosophy]
                )
                for philosophy in TranslationPhilosophy
            },
            "by_license": {
                license_type.value: len(self._by_license.get(license_type, []))
                for license_type in LicenseType
            },
        }
