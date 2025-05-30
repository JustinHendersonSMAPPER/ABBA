"""
Confidence scorer for textual variants.

Calculates confidence scores for variant readings based on multiple factors
including manuscript quality, age, geographic distribution, and text-critical
principles.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math

from .models import (
    VariantReading,
    VariantUnit,
    Witness,
    WitnessType,
    Manuscript,
    ManuscriptFamily,
    ManuscriptType,
)


@dataclass
class ManuscriptWeight:
    """Weight factors for a manuscript."""

    siglum: str
    age_weight: float = 0.5
    text_type_weight: float = 0.5
    accuracy_weight: float = 0.5
    completeness_weight: float = 0.5

    # Calculated weights
    family_weight: float = 0.5
    geographic_weight: float = 0.5
    total_weight: float = 0.5

    def calculate_total(self) -> float:
        """Calculate total weight."""
        self.total_weight = (
            self.age_weight * 0.3
            + self.text_type_weight * 0.25
            + self.accuracy_weight * 0.2
            + self.family_weight * 0.15
            + self.geographic_weight * 0.1
        )
        return self.total_weight


class ConfidenceScorer:
    """Calculate confidence scores for variant readings."""

    # Manuscript age weights (century -> weight)
    AGE_WEIGHTS = {
        2: 1.0,  # 2nd century
        3: 0.95,  # 3rd century
        4: 0.9,  # 4th century
        5: 0.8,  # 5th century
        6: 0.7,  # 6th century
        7: 0.6,  # 7th century
        8: 0.5,  # 8th century
        9: 0.4,  # 9th century
        10: 0.3,  # 10th century and later
    }

    # Text family weights
    FAMILY_WEIGHTS = {
        ManuscriptFamily.ALEXANDRIAN: 1.0,
        ManuscriptFamily.PRE_CAESAREAN: 0.9,
        ManuscriptFamily.CAESAREAN: 0.8,
        ManuscriptFamily.WESTERN: 0.7,
        ManuscriptFamily.BYZANTINE: 0.5,
        ManuscriptFamily.UNKNOWN: 0.3,
    }

    # Geographic distribution bonus
    GEOGRAPHIC_WEIGHTS = {
        1: 0.0,  # Single location
        2: 0.1,  # Two locations
        3: 0.2,  # Three locations
        4: 0.3,  # Four or more locations
    }

    def __init__(self, manuscript_db: Optional[Dict[str, Manuscript]] = None):
        """Initialize the scorer.

        Args:
            manuscript_db: Database of manuscript information
        """
        self.logger = logging.getLogger(__name__)
        self.manuscript_db = manuscript_db or {}

        # Cache for manuscript weights
        self._weight_cache: Dict[str, ManuscriptWeight] = {}

        # Known manuscript locations
        self.manuscript_locations = {
            "P46": "Egypt",
            "P66": "Egypt",
            "P75": "Egypt",
            "א": "Egypt/Palestine",
            "01": "Egypt/Palestine",
            "A": "Egypt",
            "02": "Egypt",
            "B": "Egypt",
            "03": "Egypt",
            "C": "Egypt",
            "04": "Egypt",
            "D": "Western",
            "05": "Western",
            "W": "Egypt",
            "032": "Egypt",
        }

    def score_variant_unit(self, unit: VariantUnit) -> Dict[str, float]:
        """Score all readings in a variant unit.

        Args:
            unit: Variant unit to score

        Returns:
            Map of reading text to confidence score
        """
        scores = {}

        # Score each reading
        for reading in unit.readings:
            score = self.score_reading(reading, unit)
            scores[reading.text] = score
            reading.confidence = score

        # Mark most likely original
        if scores:
            best_reading = max(unit.readings, key=lambda r: r.confidence)
            best_reading.is_original = True

        return scores

    def score_reading(self, reading: VariantReading, unit: Optional[VariantUnit] = None) -> float:
        """Calculate confidence score for a reading.

        Args:
            reading: Variant reading to score
            unit: Parent variant unit (for context)

        Returns:
            Confidence score (0.0-1.0)
        """
        # Component scores
        external_score = self._calculate_external_score(reading)
        internal_score = self._calculate_internal_score(reading, unit)
        distribution_score = self._calculate_distribution_score(reading)

        # Combine scores with weights
        total_score = (
            external_score * 0.5  # Manuscript evidence
            + internal_score * 0.3  # Text-critical principles
            + distribution_score * 0.2  # Geographic/temporal spread
        )

        return min(total_score, 1.0)

    def _calculate_external_score(self, reading: VariantReading) -> float:
        """Calculate score based on external (manuscript) evidence."""
        if not reading.witnesses:
            return 0.0

        # Calculate weighted sum of witness support
        total_weight = 0.0
        witness_weights = []

        for witness in reading.witnesses:
            weight = self._get_witness_weight(witness)
            witness_weights.append(weight)
            total_weight += weight

        # Calculate average quality
        avg_weight = total_weight / len(witness_weights) if witness_weights else 0.0

        # Bonus for number of witnesses (diminishing returns)
        quantity_bonus = min(math.log(len(witness_weights) + 1) / 10, 0.3)

        # Penalty for late witnesses only
        if self._has_only_late_witnesses(reading):
            avg_weight *= 0.7

        return min(avg_weight + quantity_bonus, 1.0)

    def _calculate_internal_score(
        self, reading: VariantReading, unit: Optional[VariantUnit]
    ) -> float:
        """Calculate score based on internal evidence."""
        score = 0.5  # Start neutral

        if not unit:
            return score

        # Prefer shorter reading (lectio brevior)
        reading_lengths = [len(r.text) for r in unit.readings if r.text != "om"]
        if reading_lengths and reading.text != "om":
            if len(reading.text) == min(reading_lengths):
                score += 0.2
            elif len(reading.text) == max(reading_lengths):
                score -= 0.1

        # Prefer more difficult reading (lectio difficilior)
        if self._is_difficult_reading(reading, unit):
            score += 0.2

        # Check for harmonization (negative)
        if self._is_harmonization(reading, unit):
            score -= 0.2

        # Check for theological modification (negative)
        if self._is_theological_modification(reading, unit):
            score -= 0.15

        return max(0.0, min(score, 1.0))

    def _calculate_distribution_score(self, reading: VariantReading) -> float:
        """Calculate score based on geographic and temporal distribution."""
        # Geographic distribution
        locations = set()
        earliest_century = 10

        for witness in reading.witnesses:
            # Get location
            location = self.manuscript_locations.get(witness.siglum, "Unknown")
            locations.add(location)

            # Get date (simplified)
            if witness.siglum.startswith("P") or witness.siglum.startswith("𝔓"):
                earliest_century = min(earliest_century, 2)
            elif witness.siglum in ["א", "01", "B", "03"]:
                earliest_century = min(earliest_century, 4)
            elif witness.siglum in ["A", "02", "C", "04", "D", "05"]:
                earliest_century = min(earliest_century, 5)

        # Geographic score
        geo_score = self.GEOGRAPHIC_WEIGHTS.get(min(len(locations), 4), 0.0)

        # Temporal score
        temporal_score = self.AGE_WEIGHTS.get(earliest_century, 0.3)

        return (geo_score + temporal_score) / 2

    def _get_witness_weight(self, witness: Witness) -> float:
        """Get weight for a single witness."""
        # Check cache
        if witness.siglum in self._weight_cache:
            return self._weight_cache[witness.siglum].total_weight

        # Look up manuscript
        ms = self.manuscript_db.get(witness.siglum)
        if ms:
            weight = self._calculate_manuscript_weight(ms)
        else:
            # Use heuristics based on siglum
            weight = self._estimate_witness_weight(witness)

        return weight

    def _calculate_manuscript_weight(self, ms: Manuscript) -> float:
        """Calculate weight for a known manuscript."""
        weight = ManuscriptWeight(siglum=ms.siglum)

        # Age weight
        century = (ms.date_numeric // 100) + 1 if ms.date_numeric else 10
        weight.age_weight = self.AGE_WEIGHTS.get(century, 0.3)

        # Text type weight
        weight.text_type_weight = {
            ManuscriptType.PAPYRUS: 1.0,
            ManuscriptType.UNCIAL: 0.9,
            ManuscriptType.MINUSCULE: 0.6,
            ManuscriptType.LECTIONARY: 0.4,
            ManuscriptType.VERSION: 0.5,
            ManuscriptType.PATRISTIC: 0.4,
        }.get(ms.type, 0.5)

        # Family weight
        weight.family_weight = self.FAMILY_WEIGHTS.get(ms.family, 0.5)

        # Quality weights
        weight.accuracy_weight = ms.accuracy_rating
        weight.completeness_weight = ms.completeness

        # Calculate total
        total = weight.calculate_total()

        # Cache it
        self._weight_cache[ms.siglum] = weight

        return total

    def _estimate_witness_weight(self, witness: Witness) -> float:
        """Estimate weight for unknown witness using heuristics."""
        weight = 0.5  # Default

        # Papyri are early and valuable
        if witness.siglum.startswith("P") or witness.siglum.startswith("𝔓"):
            weight = 0.95

        # Major uncials
        elif witness.siglum in ["א", "01"]:  # Sinaiticus
            weight = 0.9
        elif witness.siglum in ["B", "03"]:  # Vaticanus
            weight = 0.95
        elif witness.siglum in ["A", "02"]:  # Alexandrinus
            weight = 0.85
        elif witness.siglum in ["C", "04"]:  # Ephraemi
            weight = 0.8
        elif witness.siglum in ["D", "05"]:  # Bezae
            weight = 0.75

        # Versions
        elif witness.type == WitnessType.VERSION:
            version_weights = {
                "it": 0.6,  # Old Latin
                "vg": 0.5,  # Vulgate
                "syr": 0.6,  # Syriac
                "cop": 0.7,  # Coptic
            }
            weight = version_weights.get(witness.siglum[:3], 0.5)

        # Church fathers
        elif witness.type == WitnessType.FATHER:
            weight = 0.4

        # Correctors have slightly less weight
        if witness.corrector:
            weight *= 0.9

        # Marginal readings have less weight
        if witness.marginal:
            weight *= 0.8

        # Partial support has less weight
        if witness.partial:
            weight *= 0.7

        return weight

    def _has_only_late_witnesses(self, reading: VariantReading) -> bool:
        """Check if reading has only late witnesses."""
        for witness in reading.witnesses:
            # Check for early witnesses
            if witness.siglum.startswith("P") or witness.siglum.startswith("𝔓"):
                return False
            if witness.siglum in ["א", "01", "B", "03", "A", "02", "C", "04"]:
                return False

        return True

    def _is_difficult_reading(self, reading: VariantReading, unit: VariantUnit) -> bool:
        """Check if reading is the more difficult one."""
        # Simplified heuristics

        # Omissions can be difficult
        if reading.text == "om":
            return True

        # Shorter readings often more difficult
        if all(
            len(reading.text) < len(r.text)
            for r in unit.readings
            if r != reading and r.text != "om"
        ):
            return True

        # Would need lexical/grammatical analysis for full implementation

        return False

    def _is_harmonization(self, reading: VariantReading, unit: VariantUnit) -> bool:
        """Check if reading appears to be harmonization."""
        # This would need parallel passage database
        # For now, just check for expansions that look like harmonization

        # Common harmonizing additions
        harmonizing_phrases = [
            "ὁ Ἰησοῦς",  # "Jesus" added
            "ὁ Χριστός",  # "Christ" added
            "ὁ κύριος",  # "Lord" added
            "ἡμῶν",  # "our" added
        ]

        for phrase in harmonizing_phrases:
            if phrase in reading.text:
                # Check if it's an addition
                for other in unit.readings:
                    if other != reading and phrase not in other.text:
                        return True

        return False

    def _is_theological_modification(self, reading: VariantReading, unit: VariantUnit) -> bool:
        """Check if reading appears to be theological modification."""
        # Check for reverent additions
        if "κύριος" in reading.text or "Χριστός" in reading.text:
            # See if it's added compared to other readings
            for other in unit.readings:
                if other != reading and "κύριος" not in other.text and "Χριστός" not in other.text:
                    return True

        return False

    def calculate_apparatus_confidence(self, readings: List[VariantReading]) -> str:
        """Calculate overall confidence rating (A, B, C, D) for apparatus entry."""
        if not readings:
            return "D"

        # Get confidence scores
        scores = [r.confidence for r in readings if r.confidence > 0]
        if not scores:
            return "D"

        # Calculate spread (how much disagreement)
        max_score = max(scores)
        min_score = min(scores)
        spread = max_score - min_score

        # Rating based on clarity of decision
        if max_score > 0.8 and spread > 0.3:
            return "A"  # Clear winner
        elif max_score > 0.7 and spread > 0.2:
            return "B"  # Probable winner
        elif max_score > 0.6:
            return "C"  # Uncertain
        else:
            return "D"  # Very uncertain
