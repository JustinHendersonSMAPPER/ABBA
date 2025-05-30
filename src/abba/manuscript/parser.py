"""
Parser for critical apparatus notation.

Handles parsing of textual variant notation from various critical editions
including NA28, UBS5, and BHS formats.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from .models import (
    Witness,
    WitnessType,
    VariantReading,
    VariantType,
    ApparatusEntry,
    VariantUnit,
    ManuscriptType,
)
from ..verse_id import VerseID


@dataclass
class VariantNotation:
    """Parsed variant notation."""

    lemma: str  # Base text reading
    variants: List[Tuple[str, List[str]]]  # (reading, witnesses)
    location: Optional[str] = None  # Word/phrase location
    symbols: List[str] = field(default_factory=list)


class CriticalApparatusParser:
    """Parser for critical apparatus notation."""

    # Common manuscript sigla patterns
    PAPYRUS_PATTERN = re.compile(r"𝔓(\d+)")
    UNCIAL_PATTERN = re.compile(r"[אאABCDEFGHKLPQWΔΘΨΩ]")
    MINUSCULE_PATTERN = re.compile(r"\d+")
    CORRECTOR_PATTERN = re.compile(r"(.+)\*|(.+)c(\d*)")
    VERSION_PATTERN = re.compile(r"(it|vg|syr|cop|eth|arm|geo|slav|arab)")

    # Symbol meanings
    SYMBOLS = {
        "txt": "text (reading of the base text)",
        "om": "omit",
        "add": "add",
        "tr": "transpose",
        "vid": "videtur (apparently)",
        "pc": "pauci (a few manuscripts)",
        "al": "alii (others)",
        "rell": "reliqui (the remaining witnesses)",
        "mg": "margin",
        "s": "supplement",
        "v.l.": "varia lectio (variant reading)",
    }

    def __init__(self):
        """Initialize the parser."""
        self.logger = logging.getLogger(__name__)

        # Witness abbreviations
        self.witness_abbrev = {
            # Papyri
            "𝔓46": "P46",
            "𝔓66": "P66",
            "𝔓75": "P75",
            # Major uncials
            "א": "01",
            "A": "02",
            "B": "03",
            "C": "04",
            "D": "05",
            "E": "06",
            "F": "07",
            "G": "08",
            "H": "09",
            "I": "10",
            # Versions
            "it": "Old Latin",
            "vg": "Vulgate",
            "syr": "Syriac",
            "cop": "Coptic",
            "eth": "Ethiopic",
            "arm": "Armenian",
            # Church fathers
            "Or": "Origen",
            "Ir": "Irenaeus",
            "Cl": "Clement",
            "Chr": "Chrysostom",
            "Aug": "Augustine",
        }

    def parse_apparatus_entry(self, notation: str, verse_id: VerseID) -> ApparatusEntry:
        """Parse a critical apparatus entry.

        Args:
            notation: Apparatus notation (e.g., "om B D it")
            verse_id: Verse reference

        Returns:
            Parsed apparatus entry
        """
        entry = ApparatusEntry(verse_id=verse_id, location="", lemma="")

        # Split notation into components
        parts = self._tokenize_notation(notation)

        # Extract lemma if present
        if parts and not self._is_witness_siglum(parts[0]):
            entry.lemma = parts[0]
            parts = parts[1:]

        # Parse variants
        current_reading = None
        current_witnesses = []

        for part in parts:
            if self._is_reading_separator(part):
                # Save current variant if any
                if current_reading is not None:
                    variant = VariantReading(
                        text=current_reading, witnesses=self._parse_witnesses(current_witnesses)
                    )
                    entry.variants.append(variant)

                current_reading = None
                current_witnesses = []

            elif self._is_variant_reading(part):
                # Start new variant
                if current_reading is not None:
                    variant = VariantReading(
                        text=current_reading, witnesses=self._parse_witnesses(current_witnesses)
                    )
                    entry.variants.append(variant)

                current_reading = part
                current_witnesses = []

            elif self._is_witness_siglum(part):
                current_witnesses.append(part)

            elif part in self.SYMBOLS:
                entry.symbols.append(part)

        # Don't forget last variant
        if current_reading is not None:
            variant = VariantReading(
                text=current_reading, witnesses=self._parse_witnesses(current_witnesses)
            )
            entry.variants.append(variant)

        return entry

    def parse_na28_apparatus(self, apparatus_text: str) -> List[ApparatusEntry]:
        """Parse NA28-style critical apparatus.

        NA28 format examples:
        - "1 ¦ txt א A B C : om D it"
        - "3-4 ¦ τον θεον B D : θεον א C"

        Args:
            apparatus_text: Full apparatus text

        Returns:
            List of parsed entries
        """
        entries = []

        # Split by entry markers
        lines = apparatus_text.strip().split("\n")

        for line in lines:
            if not line.strip():
                continue

            # Extract verse reference and notation
            match = re.match(r"(\d+(?:-\d+)?)\s*¦\s*(.+)", line)
            if match:
                verse_nums = match.group(1)
                notation = match.group(2)

                # Parse verse numbers
                if "-" in verse_nums:
                    start, end = map(int, verse_nums.split("-"))
                    verses = list(range(start, end + 1))
                else:
                    verses = [int(verse_nums)]

                # Parse each verse's apparatus
                for verse_num in verses:
                    # This is simplified - real implementation would need book/chapter
                    verse_id = VerseID("MAT", 1, verse_num)  # Example
                    entry = self._parse_na28_notation(notation, verse_id)
                    entries.append(entry)

        return entries

    def _parse_na28_notation(self, notation: str, verse_id: VerseID) -> ApparatusEntry:
        """Parse NA28 notation for a single entry."""
        entry = ApparatusEntry(verse_id=verse_id, location="", lemma="")

        # Split by colon (separates readings)
        readings = notation.split(":")

        for i, reading_text in enumerate(readings):
            reading_text = reading_text.strip()

            # First reading often has "txt" marker
            if i == 0 and reading_text.startswith("txt "):
                reading_text = reading_text[4:]
                variant = VariantReading(text="[base text]", is_original=True, confidence=0.9)
            else:
                # Extract reading and witnesses
                parts = reading_text.split()
                if parts:
                    # Determine where reading ends and witnesses begin
                    reading_end = 0
                    for j, part in enumerate(parts):
                        if self._is_witness_siglum(part):
                            reading_end = j
                            break

                    if reading_end > 0:
                        reading = " ".join(parts[:reading_end])
                        witnesses = parts[reading_end:]
                    else:
                        # All witnesses, reading might be "om"
                        reading = "om" if "om" in parts else ""
                        witnesses = [p for p in parts if p != "om"]

                    variant = VariantReading(text=reading)

            # Parse witnesses
            if "witnesses" in locals():
                variant.witnesses = self._parse_witnesses(witnesses)

            entry.variants.append(variant)

        return entry

    def parse_ubs5_apparatus(self, apparatus_text: str) -> List[ApparatusEntry]:
        """Parse UBS5-style critical apparatus.

        UBS5 uses letter ratings {A}, {B}, {C}, {D} for certainty.
        """
        entries = []

        # UBS5 format: verse reference, {rating}, variants
        pattern = re.compile(r"(\d+:\d+)\s*\{([ABCD])\}\s*(.+)")

        for match in pattern.finditer(apparatus_text):
            ref = match.group(1)
            rating = match.group(2)
            notation = match.group(3)

            # Parse verse reference
            chapter, verse = map(int, ref.split(":"))
            verse_id = VerseID("MAT", chapter, verse)  # Example book

            entry = self.parse_apparatus_entry(notation, verse_id)
            entry.certainty_rating = rating

            entries.append(entry)

        return entries

    def _tokenize_notation(self, notation: str) -> List[str]:
        """Tokenize apparatus notation into parts."""
        # Handle special characters and maintain structure
        notation = notation.replace("¦", " ¦ ")
        notation = notation.replace(":", " : ")
        notation = notation.replace(";", " ; ")

        # Split and clean
        tokens = notation.split()
        return [t.strip() for t in tokens if t.strip()]

    def _is_witness_siglum(self, token: str) -> bool:
        """Check if token is a manuscript siglum."""
        # Check various patterns
        if self.PAPYRUS_PATTERN.match(token):
            return True
        if self.UNCIAL_PATTERN.match(token):
            return True
        if self.MINUSCULE_PATTERN.match(token) and token.isdigit():
            return True
        if self.VERSION_PATTERN.match(token):
            return True
        if token in self.witness_abbrev:
            return True

        # Check for corrector notation
        if self.CORRECTOR_PATTERN.match(token):
            return True

        return False

    def _is_reading_separator(self, token: str) -> bool:
        """Check if token separates readings."""
        return token in [":", ";", "¦", "|"]

    def _is_variant_reading(self, token: str) -> bool:
        """Check if token starts a variant reading."""
        # Not a witness and not a separator
        return not self._is_witness_siglum(token) and not self._is_reading_separator(token)

    def _parse_witnesses(self, witness_tokens: List[str]) -> List[Witness]:
        """Parse witness tokens into Witness objects."""
        witnesses = []

        for token in witness_tokens:
            witness = self._parse_single_witness(token)
            if witness:
                witnesses.append(witness)

        return witnesses

    def _parse_single_witness(self, token: str) -> Optional[Witness]:
        """Parse a single witness token."""
        # Check for corrector notation
        corrector_match = self.CORRECTOR_PATTERN.match(token)
        if corrector_match:
            base = corrector_match.group(1) or corrector_match.group(2)
            corrector_num = corrector_match.group(3) or "1"

            witness = Witness(
                type=WitnessType.MANUSCRIPT,
                siglum=base,
                corrector=int(corrector_num) if corrector_num.isdigit() else 1,
            )
            return witness

        # Check for marginal reading
        if token.endswith("mg"):
            base = token[:-2]
            witness = Witness(type=WitnessType.MANUSCRIPT, siglum=base, marginal=True)
            return witness

        # Check for partial support
        if token.startswith("(") and token.endswith(")"):
            base = token[1:-1]
            witness = Witness(type=WitnessType.MANUSCRIPT, siglum=base, partial=True)
            return witness

        # Determine witness type
        witness_type = WitnessType.MANUSCRIPT

        if self.VERSION_PATTERN.match(token):
            witness_type = WitnessType.VERSION
        elif token in ["Or", "Ir", "Cl", "Chr", "Aug"]:
            witness_type = WitnessType.FATHER

        # Create witness
        witness = Witness(type=witness_type, siglum=token)

        return witness

    def identify_variant_type(self, base_text: str, variant_text: str) -> VariantType:
        """Identify the type of variant."""
        if variant_text == "om" or not variant_text:
            return VariantType.OMISSION

        if not base_text or base_text == "om":
            return VariantType.ADDITION

        # Check for transposition (simple check)
        base_words = set(base_text.split())
        variant_words = set(variant_text.split())

        if base_words == variant_words:
            return VariantType.TRANSPOSITION

        # Check for spelling
        if len(base_text) == len(variant_text):
            differences = sum(1 for a, b in zip(base_text, variant_text) if a != b)
            if differences <= 2:
                return VariantType.SPELLING

        # Default to substitution
        return VariantType.SUBSTITUTION
