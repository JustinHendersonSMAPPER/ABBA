"""
Data models for manuscript variants and critical apparatus.

Provides structures for representing textual variants, manuscript witnesses,
and critical apparatus data for biblical texts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime

from ..verse_id import VerseID


class ManuscriptFamily(Enum):
    """Major manuscript family traditions."""

    ALEXANDRIAN = "alexandrian"  # Early, generally reliable
    BYZANTINE = "byzantine"  # Majority text, later
    WESTERN = "western"  # Early but free
    CAESAREAN = "caesarean"  # Mixed type
    PRE_CAESAREAN = "pre_caesarean"  # Earlier form
    UNKNOWN = "unknown"


class ManuscriptType(Enum):
    """Types of manuscript witnesses."""

    PAPYRUS = "papyrus"  # Earliest manuscripts (P46, P75, etc.)
    UNCIAL = "uncial"  # Majuscule codices (א, A, B, etc.)
    MINUSCULE = "minuscule"  # Cursive manuscripts (1, 33, etc.)
    LECTIONARY = "lectionary"  # Church reading manuscripts
    VERSION = "version"  # Ancient translations
    PATRISTIC = "patristic"  # Church father quotations


class VariantType(Enum):
    """Types of textual variants."""

    ADDITION = "addition"  # Text added
    OMISSION = "omission"  # Text omitted
    SUBSTITUTION = "substitution"  # Different word/phrase
    TRANSPOSITION = "transposition"  # Word order change
    SPELLING = "spelling"  # Orthographic variation
    GRAMMATICAL = "grammatical"  # Case, number, tense changes
    CONFLATION = "conflation"  # Combination of readings
    CORRECTION = "correction"  # Scribal correction


class WitnessType(Enum):
    """Types of textual witnesses."""

    MANUSCRIPT = "manuscript"  # Greek/Hebrew manuscripts
    VERSION = "version"  # Ancient translations
    FATHER = "father"  # Patristic quotations
    CONJECTURE = "conjecture"  # Scholarly emendation


@dataclass
class Manuscript:
    """Represents a biblical manuscript."""

    siglum: str  # Standard abbreviation (P46, א, B, etc.)
    name: str  # Full name
    type: ManuscriptType
    family: ManuscriptFamily

    # Dating
    date: str  # Century or specific date
    date_numeric: Optional[int] = None  # Year (approximate)

    # Content
    contents: List[str] = field(default_factory=list)  # Books contained
    script: str = "greek"  # greek, hebrew, latin, etc.

    # Quality indicators
    accuracy_rating: float = 0.5  # 0.0-1.0 general accuracy
    completeness: float = 0.0  # 0.0-1.0 how complete

    # Metadata
    location: Optional[str] = None  # Current location
    catalog_number: Optional[str] = None
    notes: Optional[str] = None

    def contains_book(self, book_id: str) -> bool:
        """Check if manuscript contains a book."""
        return book_id in self.contents

    def get_weight(self) -> float:
        """Calculate manuscript weight for variant evaluation."""
        # Earlier manuscripts get higher weight
        age_weight = 1.0
        if self.date_numeric:
            if self.date_numeric < 300:
                age_weight = 1.0
            elif self.date_numeric < 500:
                age_weight = 0.8
            elif self.date_numeric < 1000:
                age_weight = 0.6
            else:
                age_weight = 0.4

        # Manuscript type affects weight
        type_weight = {
            ManuscriptType.PAPYRUS: 1.0,
            ManuscriptType.UNCIAL: 0.9,
            ManuscriptType.MINUSCULE: 0.7,
            ManuscriptType.LECTIONARY: 0.5,
            ManuscriptType.VERSION: 0.6,
            ManuscriptType.PATRISTIC: 0.5,
        }.get(self.type, 0.5)

        # Family affects weight
        family_weight = {
            ManuscriptFamily.ALEXANDRIAN: 1.0,
            ManuscriptFamily.PRE_CAESAREAN: 0.9,
            ManuscriptFamily.CAESAREAN: 0.8,
            ManuscriptFamily.WESTERN: 0.7,
            ManuscriptFamily.BYZANTINE: 0.6,
            ManuscriptFamily.UNKNOWN: 0.5,
        }.get(self.family, 0.5)

        # Combine factors
        return (
            age_weight * 0.4 + type_weight * 0.3 + family_weight * 0.2 + self.accuracy_rating * 0.1
        )


@dataclass
class Witness:
    """A witness to a textual reading."""

    type: WitnessType
    siglum: str  # Manuscript siglum or identifier

    # Optional qualifiers
    certainty: float = 1.0  # 0.0-1.0 certainty of reading
    corrector: Optional[int] = None  # Corrector hand (1, 2, etc.)
    marginal: bool = False  # Reading in margin
    partial: bool = False  # Partially supports reading

    # For patristic witnesses
    father_name: Optional[str] = None
    work: Optional[str] = None

    def __str__(self) -> str:
        """String representation of witness."""
        base = self.siglum
        if self.corrector:
            base += f"^{self.corrector}"
        if self.marginal:
            base += "^mg"
        if self.partial:
            base = f"({base})"
        return base


@dataclass
class VariantReading:
    """A specific variant reading."""

    text: str  # The variant text
    text_translated: Optional[str] = None  # English translation

    # Location in verse
    word_start: int = 0  # Starting word position (0-based)
    word_end: int = 0  # Ending word position

    # Support
    witnesses: List[Witness] = field(default_factory=list)
    manuscript_count: int = 0  # Total manuscripts supporting

    # Analysis
    variant_type: Optional[VariantType] = None
    is_original: bool = False  # Likely original reading
    confidence: float = 0.5  # Confidence in originality

    def add_witness(self, witness: Witness) -> None:
        """Add a witness to this reading."""
        self.witnesses.append(witness)
        if witness.type == WitnessType.MANUSCRIPT:
            self.manuscript_count += 1


@dataclass
class VariantUnit:
    """A unit of variation (location where manuscripts differ)."""

    verse_id: VerseID
    unit_id: str  # Unique ID within verse

    # Location
    word_positions: List[int]  # Affected word positions
    context: str = ""  # Surrounding text for context

    # Readings
    readings: List[VariantReading] = field(default_factory=list)
    base_text: str = ""  # Text in base edition (NA28, etc.)

    # Significance
    significant: bool = True  # Affects meaning/translation
    affects_meaning: bool = False  # Changes theological meaning

    def get_preferred_reading(self) -> Optional[VariantReading]:
        """Get the most likely original reading."""
        original = [r for r in self.readings if r.is_original]
        if original:
            return original[0]

        # Fall back to highest confidence
        if self.readings:
            return max(self.readings, key=lambda r: r.confidence)

        return None

    def get_reading_by_text(self, text: str) -> Optional[VariantReading]:
        """Find a reading by its text."""
        for reading in self.readings:
            if reading.text == text:
                return reading
        return None


@dataclass
class TextualVariant:
    """High-level textual variant information."""

    id: str  # Unique variant ID
    verse_id: VerseID
    variant_units: List[VariantUnit] = field(default_factory=list)

    # Summary
    total_variants: int = 0
    significant_variants: int = 0

    # Metadata
    source: str = ""  # Source critical edition
    notes: Optional[str] = None

    def add_unit(self, unit: VariantUnit) -> None:
        """Add a variant unit."""
        self.variant_units.append(unit)
        self.total_variants = len(self.variant_units)
        self.significant_variants = sum(1 for u in self.variant_units if u.significant)


@dataclass
class Attestation:
    """Attestation information for a reading."""

    reading: VariantReading
    witnesses: List[Witness]

    # Geographic distribution
    regions: Set[str] = field(default_factory=set)

    # Temporal distribution
    earliest_witness: Optional[str] = None
    earliest_date: Optional[int] = None

    # Support strength
    greek_manuscripts: int = 0
    versional_support: int = 0
    patristic_support: int = 0

    def calculate_strength(self) -> float:
        """Calculate overall attestation strength."""
        # Factor in different types of support
        ms_score = min(self.greek_manuscripts / 10, 1.0) * 0.5
        version_score = min(self.versional_support / 5, 1.0) * 0.3
        father_score = min(self.patristic_support / 5, 1.0) * 0.2

        # Geographic distribution bonus
        geo_bonus = min(len(self.regions) / 4, 1.0) * 0.1

        # Age bonus
        age_bonus = 0.0
        if self.earliest_date:
            if self.earliest_date < 300:
                age_bonus = 0.2
            elif self.earliest_date < 500:
                age_bonus = 0.1

        return min(ms_score + version_score + father_score + geo_bonus + age_bonus, 1.0)


@dataclass
class ApparatusEntry:
    """Entry in a critical apparatus."""

    verse_id: VerseID
    location: str  # Location description
    lemma: str  # Base text reading

    # Variants
    variants: List[VariantReading] = field(default_factory=list)

    # Apparatus notation
    symbols: List[str] = field(default_factory=list)  # Special symbols used
    abbreviations: Dict[str, str] = field(default_factory=dict)

    # Editorial notes
    editor_note: Optional[str] = None
    certainty_rating: Optional[str] = None  # A, B, C, D rating


@dataclass
class CriticalApparatus:
    """Complete critical apparatus for a text unit."""

    text_unit: str  # Book, chapter, or verse range
    source_edition: str  # NA28, UBS5, BHS, etc.

    entries: List[ApparatusEntry] = field(default_factory=list)

    # Manuscript information
    manuscripts_cited: Set[str] = field(default_factory=set)
    manuscript_info: Dict[str, Manuscript] = field(default_factory=dict)

    # Statistics
    total_variation_units: int = 0
    significant_variants: int = 0

    def add_entry(self, entry: ApparatusEntry) -> None:
        """Add an apparatus entry."""
        self.entries.append(entry)

        # Update manuscript list
        for variant in entry.variants:
            for witness in variant.witnesses:
                if witness.type == WitnessType.MANUSCRIPT:
                    self.manuscripts_cited.add(witness.siglum)

    def get_entries_for_verse(self, verse_id: VerseID) -> List[ApparatusEntry]:
        """Get all entries for a specific verse."""
        return [e for e in self.entries if e.verse_id == verse_id]


@dataclass
class ReadingSupport:
    """Support analysis for a variant reading."""

    reading: VariantReading

    # Manuscript support by family
    family_support: Dict[ManuscriptFamily, int] = field(default_factory=dict)

    # Geographical support
    geographic_support: Dict[str, int] = field(default_factory=dict)

    # Temporal support
    century_support: Dict[int, int] = field(default_factory=dict)

    # Calculated scores
    external_score: float = 0.0  # Based on manuscripts
    internal_score: float = 0.0  # Based on scribal tendencies
    total_score: float = 0.0

    def calculate_scores(self) -> None:
        """Calculate support scores."""
        # External evidence
        if self.family_support:
            # Alexandrian readings preferred
            alex_weight = self.family_support.get(ManuscriptFamily.ALEXANDRIAN, 0) * 2
            byz_weight = self.family_support.get(ManuscriptFamily.BYZANTINE, 0) * 0.5
            other_weight = sum(
                count
                for fam, count in self.family_support.items()
                if fam not in [ManuscriptFamily.ALEXANDRIAN, ManuscriptFamily.BYZANTINE]
            )

            total_witnesses = sum(self.family_support.values())
            if total_witnesses > 0:
                self.external_score = (alex_weight + byz_weight + other_weight) / total_witnesses

        # Geographic distribution
        if len(self.geographic_support) > 3:
            self.external_score += 0.2

        # Age of attestation
        if self.century_support:
            earliest = min(self.century_support.keys())
            if earliest <= 3:
                self.external_score += 0.3
            elif earliest <= 5:
                self.external_score += 0.1

        # Normalize
        self.external_score = min(self.external_score, 1.0)

        # Total (internal evidence would be calculated separately)
        self.total_score = self.external_score * 0.7 + self.internal_score * 0.3
