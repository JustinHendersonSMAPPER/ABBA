"""
Data models for biblical canon support.

Defines structures for representing different biblical canons, versification
schemes, and translation metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime


class CanonTradition(Enum):
    """Biblical canon traditions."""

    PROTESTANT = "protestant"
    CATHOLIC = "catholic"
    EASTERN_ORTHODOX = "eastern_orthodox"
    ORIENTAL_ORTHODOX = "oriental_orthodox"
    ETHIOPIAN_ORTHODOX = "ethiopian_orthodox"
    SLAVONIC = "slavonic"
    SYRIAC = "syriac"
    ARMENIAN = "armenian"
    GEORGIAN = "georgian"
    COPTIC = "coptic"


class BookClassification(Enum):
    """Classification of biblical books by canonical status."""

    PROTOCANONICAL = "protocanonical"  # Accepted by all traditions
    DEUTEROCANONICAL = "deuterocanonical"  # Catholic/Orthodox additions
    APOCRYPHAL = "apocryphal"  # Various additional texts
    PSEUDEPIGRAPHAL = "pseudepigraphal"  # Attributed writings


class BookSection(Enum):
    """Sections where books appear in different canons."""

    OLD_TESTAMENT = "old_testament"
    NEW_TESTAMENT = "new_testament"
    APOCRYPHA = "apocrypha"
    APPENDIX = "appendix"


class TranslationPhilosophy(Enum):
    """Translation philosophy/methodology."""

    FORMAL_EQUIVALENCE = "formal"  # Word-for-word (e.g., NASB, ESV)
    DYNAMIC_EQUIVALENCE = "dynamic"  # Thought-for-thought (e.g., NIV)
    PARAPHRASE = "paraphrase"  # Free translation (e.g., Message)
    OPTIMAL_EQUIVALENCE = "optimal"  # Balance of formal/dynamic
    LITERAL = "literal"  # Extremely literal (e.g., YLT)


class LicenseType(Enum):
    """Types of licenses for translations."""

    PUBLIC_DOMAIN = "public_domain"
    OPEN_LICENSE = "open_license"
    RESTRICTED = "restricted"
    COMMERCIAL = "commercial"
    ACADEMIC = "academic"
    CUSTOM = "custom"


class MappingType(Enum):
    """Types of verse mappings between versification schemes."""

    ONE_TO_ONE = "one_to_one"  # Simple renumbering
    ONE_TO_MANY = "one_to_many"  # Verse split
    MANY_TO_ONE = "many_to_one"  # Verse merge
    NULL_MAPPING = "null"  # Verse doesn't exist in target


@dataclass
class Canon:
    """Represents a biblical canon tradition."""

    id: str
    name: str
    tradition: CanonTradition
    description: str
    book_count: int

    # Additional metadata
    established_date: Optional[str] = None  # Historical date when established
    authority: Optional[str] = None  # Church/tradition that recognizes it
    notes: Optional[str] = None

    # Book collections
    books: List["CanonBook"] = field(default_factory=list)
    versification_scheme_id: str = "standard"

    # Language/region associations
    primary_languages: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)

    def get_book_ids(self) -> List[str]:
        """Get list of book IDs in canonical order."""
        return [cb.book_id for cb in sorted(self.books, key=lambda x: x.order)]

    def has_book(self, book_id: str) -> bool:
        """Check if canon includes a specific book."""
        return any(cb.book_id == book_id for cb in self.books)

    def get_book_order(self, book_id: str) -> Optional[int]:
        """Get canonical order of a book."""
        for cb in self.books:
            if cb.book_id == book_id:
                return cb.order
        return None


@dataclass
class CanonBook:
    """Represents a book's presence and position in a specific canon."""

    canon_id: str
    book_id: str
    order: int  # Position in canon (1-based)
    section: BookSection
    classification: BookClassification

    # Book-specific names in this tradition
    canonical_name: Optional[str] = None
    abbreviated_name: Optional[str] = None
    alternate_names: List[str] = field(default_factory=list)

    # Versification info
    chapter_count: Optional[int] = None
    verse_count: Optional[int] = None

    # Additional metadata
    is_combined: bool = False  # e.g., Ezra-Nehemiah in some traditions
    combined_with: Optional[str] = None
    notes: Optional[str] = None

    def __lt__(self, other: "CanonBook") -> bool:
        """Enable sorting by canonical order."""
        return self.order < other.order


@dataclass
class VersificationScheme:
    """Represents a verse numbering system."""

    id: str
    name: str
    description: str
    base_text: str  # e.g., "Masoretic", "Septuagint", "Vulgate"

    # Scheme characteristics
    includes_apocrypha: bool = False
    includes_verse_variants: bool = False

    # Notable differences from standard
    differences: Dict[str, str] = field(default_factory=dict)

    # Associated canons/translations
    used_by_canons: List[str] = field(default_factory=list)
    used_by_translations: List[str] = field(default_factory=list)


@dataclass
class VerseMapping:
    """Maps verses between different versification schemes."""

    source_scheme_id: str
    target_scheme_id: str
    mapping_type: MappingType

    # Source verse(s)
    source_book: str
    source_chapter: int
    source_verses: List[int]  # Can be single or multiple

    # Target verse(s)
    target_book: str
    target_chapter: int
    target_verses: List[int]  # Can be single or multiple

    # Mapping metadata
    confidence: float = 1.0  # Confidence in mapping accuracy
    notes: Optional[str] = None
    authority: Optional[str] = None  # Source of mapping information

    def get_source_references(self) -> List[str]:
        """Get source verse references."""
        return [f"{self.source_book}.{self.source_chapter}.{v}" for v in self.source_verses]

    def get_target_references(self) -> List[str]:
        """Get target verse references."""
        if self.mapping_type == MappingType.NULL_MAPPING:
            return []
        return [f"{self.target_book}.{self.target_chapter}.{v}" for v in self.target_verses]


@dataclass
class Translation:
    """Represents a Bible translation with metadata."""

    id: str
    name: str
    abbreviation: str
    language_code: str  # ISO 639-1/639-3 code

    # Canon and versification
    canon_id: str
    versification_scheme_id: str

    # Translation details
    philosophy: TranslationPhilosophy
    base_texts: List[str] = field(default_factory=list)

    # Publication info
    year_published: Optional[int] = None
    year_revised: Optional[int] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None

    # Legal/licensing
    license_type: LicenseType = LicenseType.RESTRICTED
    license_details: Optional[str] = None
    copyright_holder: Optional[str] = None

    # Digital rights
    digital_distribution: bool = False
    api_access: bool = False
    quotation_limit: Optional[int] = None  # Max verses for fair use
    attribution_required: bool = True
    commercial_use: bool = False

    # Translation team/authority
    translators: List[str] = field(default_factory=list)
    translation_committee: Optional[str] = None
    church_approval: Optional[str] = None

    # Language features
    script_direction: str = "ltr"  # ltr or rtl
    uses_diacritics: bool = False
    requires_special_font: bool = False

    # Additional metadata
    description: Optional[str] = None
    features: List[str] = field(default_factory=list)  # e.g., "study notes", "cross-references"

    def is_public_domain(self) -> bool:
        """Check if translation is in public domain."""
        return self.license_type == LicenseType.PUBLIC_DOMAIN

    def allows_digital_use(self) -> bool:
        """Check if translation allows digital distribution."""
        return self.digital_distribution or self.is_public_domain()

    def get_attribution_text(self) -> str:
        """Get required attribution text."""
        if not self.attribution_required or self.is_public_domain():
            return ""

        base_text = f"{self.name} ({self.abbreviation})"
        if self.copyright_holder:
            base_text += f", Copyright © {self.copyright_holder}"
        if self.year_published:
            base_text += f", {self.year_published}"

        return base_text


@dataclass
class CanonDifference:
    """Represents a difference between two canons."""

    difference_type: str  # "book_presence", "book_order", "book_name", "versification"
    book_id: Optional[str] = None

    # For book presence differences
    in_first_canon: bool = False
    in_second_canon: bool = False

    # For ordering differences
    first_canon_position: Optional[int] = None
    second_canon_position: Optional[int] = None

    # For naming differences
    first_canon_name: Optional[str] = None
    second_canon_name: Optional[str] = None

    # For section differences
    first_canon_section: Optional[BookSection] = None
    second_canon_section: Optional[BookSection] = None

    # Human-readable description
    description: str = ""
