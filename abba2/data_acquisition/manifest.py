"""
Source Manifest Management
Loads and manages data source definitions
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Types of biblical data sources"""
    LEXICON = "lexicon"
    MORPHOLOGY = "morphology"
    TREEBANK = "treebank"
    TEXT = "text"
    SEMANTIC = "semantic"
    ALIGNMENT = "alignment"


class SourceLanguage(str, Enum):
    """Supported languages"""
    GREEK = "greek"
    HEBREW = "hebrew"
    ARAMAIC = "aramaic"
    ENGLISH = "english"
    LATIN = "latin"


class SourceFormat(str, Enum):
    """Data format types"""
    TEI_XML = "tei_xml"
    XML = "xml"
    OSIS_XML = "osis_xml"
    PROIEL_XML = "proiel_xml"
    LOWFAT_XML = "lowfat_xml"
    ETCBC = "etcbc"
    MORPHGNT = "morphgnt"
    OSHB = "oshb"
    WLC = "wlc"
    BEREAN = "berean"
    CUSTOM = "custom"
    JSON = "json"
    TSV = "tsv"


class DataSource(BaseModel):
    """Individual data source definition"""
    name: str = Field(..., description="Human-readable source name")
    type: SourceType = Field(..., description="Type of source")
    language: Optional[SourceLanguage] = Field(None, description="Primary language")
    languages: Optional[List[SourceLanguage]] = Field(None, description="Multiple languages")
    year: Optional[int] = Field(None, description="Publication year")
    license: str = Field(..., description="License type")
    url: str = Field(..., description="Download URL")
    checksum: Optional[str] = Field(None, description="SHA256 checksum")
    description: str = Field(..., description="Source description")
    format: SourceFormat = Field(..., description="Data format")
    note: Optional[str] = Field(None, description="Additional notes")
    
    @field_validator("languages", mode="after")
    @classmethod
    def validate_languages(cls, v: Optional[List[SourceLanguage]], info) -> Optional[List[SourceLanguage]]:
        """Ensure either language or languages is set"""
        if v is None and info.data.get("language") is None:
            raise ValueError("Either 'language' or 'languages' must be specified")
        return v
    
    def get_languages(self) -> List[SourceLanguage]:
        """Get all languages for this source"""
        if self.languages:
            return self.languages
        return [self.language] if self.language else []
    
    def get_filename(self) -> str:
        """Generate local filename for this source"""
        ext = self.url.split(".")[-1]
        if ext in ["zip", "xml", "json", "tsv", "txt"]:
            return f"{self.name.lower().replace(' ', '_')}.{ext}"
        return f"{self.name.lower().replace(' ', '_')}.data"
    
    def requires_manual_entry(self) -> bool:
        """Check if source requires manual data entry"""
        return self.url == "manual_entry"


class DownloadSettings(BaseModel):
    """Download configuration"""
    parallel: bool = Field(True, description="Enable parallel downloads")
    max_concurrent: int = Field(3, description="Max concurrent downloads")
    retry_count: int = Field(3, description="Number of retry attempts")
    timeout: int = Field(300, description="Download timeout in seconds")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    user_agent: str = Field(
        "ABBA/2.0",
        description="User agent string"
    )


class ValidationSettings(BaseModel):
    """Validation requirements"""
    required_coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Required vocabulary coverage"
    )
    minimum_sources: Dict[str, int] = Field(
        default_factory=dict,
        description="Minimum number of sources per language"
    )
    checksum_verification: bool = Field(True, description="Verify checksums")
    structure_validation: bool = Field(True, description="Validate data structure")


class SourceManifest:
    """Manages the collection of data sources"""
    
    def __init__(self, manifest_path: Optional[Path] = None):
        """
        Initialize manifest from YAML file
        
        Args:
            manifest_path: Path to sources.yaml file
        """
        self.manifest_path = manifest_path or Path("sources.yaml")
        self.sources: Dict[str, DataSource] = {}
        self.download_settings: Optional[DownloadSettings] = None
        self.validation_settings: Optional[ValidationSettings] = None
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load and parse the manifest file"""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Parse sources
        for key, source_data in data.get("sources", {}).items():
            try:
                self.sources[key] = DataSource(**source_data)
            except Exception as e:
                logger.error(f"Failed to parse source {key}: {e}")
                raise
        
        # Parse settings
        if "download" in data:
            self.download_settings = DownloadSettings(**data["download"])
        else:
            self.download_settings = DownloadSettings()
        
        if "validation" in data:
            self.validation_settings = ValidationSettings(**data["validation"])
        else:
            self.validation_settings = ValidationSettings()
        
        logger.info(f"Loaded {len(self.sources)} sources from manifest")
    
    def get_source(self, key: str) -> Optional[DataSource]:
        """Get a specific source by key"""
        return self.sources.get(key)
    
    def get_sources_by_type(self, source_type: SourceType) -> List[DataSource]:
        """Get all sources of a specific type"""
        return [s for s in self.sources.values() if s.type == source_type]
    
    def get_sources_by_language(self, language: SourceLanguage) -> List[DataSource]:
        """Get all sources for a specific language"""
        return [
            s for s in self.sources.values()
            if language in s.get_languages()
        ]
    
    def get_required_sources(self) -> List[DataSource]:
        """Get sources required for minimum validation"""
        required = []
        
        # Get minimum required Greek sources
        greek_sources = self.get_sources_by_language(SourceLanguage.GREEK)
        min_greek = self.validation_settings.minimum_sources.get("greek", 0)
        required.extend(greek_sources[:min_greek])
        
        # Get minimum required Hebrew sources
        hebrew_sources = self.get_sources_by_language(SourceLanguage.HEBREW)
        min_hebrew = self.validation_settings.minimum_sources.get("hebrew", 0)
        required.extend(hebrew_sources[:min_hebrew])
        
        return required
    
    def get_download_queue(self, skip_manual: bool = True) -> List[DataSource]:
        """Get list of sources to download"""
        sources = list(self.sources.values())
        if skip_manual:
            sources = [s for s in sources if not s.requires_manual_entry()]
        return sources
    
    def validate_manifest(self) -> bool:
        """Validate manifest completeness"""
        errors = []
        
        # Check minimum sources
        for lang, min_count in self.validation_settings.minimum_sources.items():
            count = len(self.get_sources_by_language(SourceLanguage(lang)))
            if count < min_count:
                errors.append(f"Insufficient {lang} sources: {count} < {min_count}")
        
        # Check for essential source types
        essential_types = [SourceType.LEXICON, SourceType.TEXT]
        for source_type in essential_types:
            if not self.get_sources_by_type(source_type):
                errors.append(f"No sources of type: {source_type}")
        
        if errors:
            for error in errors:
                logger.error(f"Manifest validation: {error}")
            return False
        
        logger.info("Manifest validation passed")
        return True
    
    def update_checksum(self, source_key: str, checksum: str) -> None:
        """Update checksum for a source after download"""
        if source_key in self.sources:
            self.sources[source_key].checksum = checksum
            self._save_manifest()
    
    def _save_manifest(self) -> None:
        """Save updated manifest back to file"""
        data = {
            "sources": {
                key: source.model_dump(exclude_none=True)
                for key, source in self.sources.items()
            },
            "download": self.download_settings.model_dump() if self.download_settings else {},
            "validation": self.validation_settings.model_dump() if self.validation_settings else {},
        }
        
        with open(self.manifest_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated manifest saved to {self.manifest_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the manifest"""
        stats = {
            "total_sources": len(self.sources),
            "by_type": {},
            "by_language": {},
            "by_license": {},
            "manual_entry_required": 0,
        }
        
        # Count by type
        for source_type in SourceType:
            count = len(self.get_sources_by_type(source_type))
            if count > 0:
                stats["by_type"][source_type.value] = count
        
        # Count by language
        for language in SourceLanguage:
            count = len(self.get_sources_by_language(language))
            if count > 0:
                stats["by_language"][language.value] = count
        
        # Count by license
        licenses = {}
        for source in self.sources.values():
            licenses[source.license] = licenses.get(source.license, 0) + 1
        stats["by_license"] = licenses
        
        # Count manual entries
        stats["manual_entry_required"] = sum(
            1 for s in self.sources.values() if s.requires_manual_entry()
        )
        
        return stats