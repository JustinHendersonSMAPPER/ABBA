"""
Analyzer for textual variants.

Provides analysis of variant significance, relationships between readings,
and genealogical relationships between manuscripts.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from .models import (
    VariantReading,
    VariantUnit,
    VariantType,
    Witness,
    WitnessType,
    ManuscriptFamily,
    TextualVariant,
    ReadingSupport,
)


@dataclass
class VariantImpact:
    """Impact assessment of a textual variant."""

    affects_meaning: bool = False
    affects_translation: bool = False
    theological_significance: bool = False

    # Specific impacts
    changes_subject: bool = False
    changes_object: bool = False
    changes_verb: bool = False
    adds_content: bool = False
    removes_content: bool = False

    # Impact score (0-1)
    impact_score: float = 0.0

    # Description
    impact_description: str = ""


class VariantAnalyzer:
    """Analyzer for textual variants."""

    def __init__(self):
        """Initialize the analyzer."""
        self.logger = logging.getLogger(__name__)

        # Theologically significant terms (Greek)
        self.theological_terms = {
            "θεός",
            "θεοῦ",
            "θεόν",
            "θεῷ",  # God
            "Χριστός",
            "Χριστοῦ",
            "Χριστόν",
            "Χριστῷ",  # Christ
            "Ἰησοῦς",
            "Ἰησοῦ",
            "Ἰησοῦν",  # Jesus
            "κύριος",
            "κυρίου",
            "κύριον",
            "κυρίῳ",  # Lord
            "πνεῦμα",
            "πνεύματος",
            "πνεύματι",  # Spirit
            "σωτηρία",
            "σωτηρίας",
            "σωτηρίαν",  # salvation
            "πίστις",
            "πίστεως",
            "πίστιν",
            "πίστει",  # faith
            "ἁμαρτία",
            "ἁμαρτίας",
            "ἁμαρτίαν",  # sin
            "χάρις",
            "χάριτος",
            "χάριν",
            "χάριτι",  # grace
            "ἀγάπη",
            "ἀγάπης",
            "ἀγάπην",
            "ἀγάπῃ",  # love
        }

        # Common scribal tendencies
        self.scribal_tendencies = {
            "harmonization": 0.3,  # Tendency to harmonize parallel passages
            "expansion": 0.2,  # Tendency to add clarifying words
            "contraction": 0.1,  # Tendency to shorten
            "correction": 0.15,  # Tendency to "correct" grammar
            "reverence": 0.25,  # Tendency to add titles to Jesus
        }

    def analyze_variant_unit(self, unit: VariantUnit) -> Dict[str, any]:
        """Analyze a variant unit comprehensively.

        Args:
            unit: Variant unit to analyze

        Returns:
            Analysis results
        """
        analysis = {
            "unit_id": unit.unit_id,
            "verse_id": str(unit.verse_id),
            "num_readings": len(unit.readings),
            "impacts": [],
            "relationships": [],
            "preferred_reading": None,
            "confidence_scores": {},
        }

        # Analyze each reading
        for reading in unit.readings:
            impact = self.assess_impact(unit.base_text, reading)
            analysis["impacts"].append({"reading": reading.text, "impact": impact})

            # Analyze support
            support = self.analyze_reading_support(reading)
            analysis["confidence_scores"][reading.text] = support.total_score

        # Find relationships between readings
        analysis["relationships"] = self.find_reading_relationships(unit.readings)

        # Determine preferred reading
        analysis["preferred_reading"] = self.determine_preferred_reading(unit)

        return analysis

    def assess_impact(self, base_text: str, variant: VariantReading) -> VariantImpact:
        """Assess the impact of a variant reading.

        Args:
            base_text: Base text reading
            variant: Variant reading

        Returns:
            Impact assessment
        """
        impact = VariantImpact()

        # Check for omission/addition
        if variant.variant_type == VariantType.OMISSION:
            impact.removes_content = True
            impact.affects_translation = True

            # Check if theological term is omitted
            for term in self.theological_terms:
                if term in base_text:
                    impact.theological_significance = True
                    impact.impact_description = f"Omits theological term: {term}"
                    break

        elif variant.variant_type == VariantType.ADDITION:
            impact.adds_content = True
            impact.affects_translation = True

            # Check if theological term is added
            for term in self.theological_terms:
                if term in variant.text:
                    impact.theological_significance = True
                    impact.impact_description = f"Adds theological term: {term}"
                    break

        elif variant.variant_type == VariantType.SUBSTITUTION:
            # Analyze what changed
            impact.affects_translation = True
            
            # Check for theological terms
            for term in self.theological_terms:
                if term in base_text or term in variant.text:
                    impact.theological_significance = True
                    impact.impact_description = f"Substitution involves theological term: {term}"
                    break

            # Simple word comparison
            base_words = base_text.split()
            variant_words = variant.text.split()

            # Check for subject/object/verb changes
            if len(base_words) > 0 and len(variant_words) > 0:
                if base_words[0] != variant_words[0]:
                    impact.changes_subject = True

                if len(base_words) > 1 and len(variant_words) > 1:
                    if base_words[-1] != variant_words[-1]:
                        impact.changes_object = True

        # Calculate impact score
        score = 0.0
        if impact.theological_significance:
            score += 0.5
        if impact.affects_meaning:
            score += 0.3
        if impact.affects_translation:
            score += 0.2
        if impact.changes_subject or impact.changes_object:
            score += 0.2
        if impact.changes_verb:
            score += 0.1

        impact.impact_score = min(score, 1.0)

        # Set affects_meaning based on score
        if impact.impact_score > 0.3:
            impact.affects_meaning = True

        return impact

    def analyze_reading_support(self, reading: VariantReading) -> ReadingSupport:
        """Analyze manuscript support for a reading.

        Args:
            reading: Variant reading

        Returns:
            Support analysis
        """
        support = ReadingSupport(reading=reading)

        # Categorize witnesses
        for witness in reading.witnesses:
            if witness.type == WitnessType.MANUSCRIPT:
                # This would need manuscript database to work fully
                # For now, use simple heuristics based on siglum

                # Papyri are early and Alexandrian
                if witness.siglum.startswith("P") or witness.siglum.startswith("𝔓"):
                    support.family_support[ManuscriptFamily.ALEXANDRIAN] = (
                        support.family_support.get(ManuscriptFamily.ALEXANDRIAN, 0) + 1
                    )
                    support.century_support[2] = support.century_support.get(2, 0) + 1

                # Major uncials
                elif witness.siglum in ["א", "01", "B", "03"]:
                    support.family_support[ManuscriptFamily.ALEXANDRIAN] = (
                        support.family_support.get(ManuscriptFamily.ALEXANDRIAN, 0) + 1
                    )
                    support.century_support[4] = support.century_support.get(4, 0) + 1

                elif witness.siglum in ["A", "02", "C", "04"]:
                    support.family_support[ManuscriptFamily.BYZANTINE] = (
                        support.family_support.get(ManuscriptFamily.BYZANTINE, 0) + 1
                    )
                    support.century_support[5] = support.century_support.get(5, 0) + 1

                elif witness.siglum in ["D", "05"]:
                    support.family_support[ManuscriptFamily.WESTERN] = (
                        support.family_support.get(ManuscriptFamily.WESTERN, 0) + 1
                    )
                    support.century_support[5] = support.century_support.get(5, 0) + 1

        # Calculate scores
        support.calculate_scores()

        return support

    def find_reading_relationships(self, readings: List[VariantReading]) -> List[Dict[str, any]]:
        """Find genealogical relationships between readings.

        Args:
            readings: List of variant readings

        Returns:
            List of relationships
        """
        relationships = []

        for i, reading1 in enumerate(readings):
            for j, reading2 in enumerate(readings[i + 1 :], i + 1):
                rel = self._compare_readings(reading1, reading2)
                if rel:
                    relationships.append(
                        {
                            "reading1": reading1.text,
                            "reading2": reading2.text,
                            "relationship": rel["type"],
                            "confidence": rel["confidence"],
                        }
                    )

        return relationships

    def _compare_readings(
        self, reading1: VariantReading, reading2: VariantReading
    ) -> Optional[Dict[str, any]]:
        """Compare two readings for relationships."""
        # Simple heuristics for demonstration

        # Check for expansion/contraction
        if reading1.text in reading2.text:
            return {
                "type": "expansion",
                "confidence": 0.8,
                "direction": f"{reading1.text} → {reading2.text}",
            }
        elif reading2.text in reading1.text:
            return {
                "type": "contraction",
                "confidence": 0.8,
                "direction": f"{reading2.text} → {reading1.text}",
            }

        # Check for simple substitution
        words1 = set(reading1.text.split())
        words2 = set(reading2.text.split())

        if len(words1 & words2) > 0 and len(words1 ^ words2) == 2:
            return {"type": "substitution", "confidence": 0.7, "direction": "unclear"}

        return None

    def determine_preferred_reading(self, unit: VariantUnit) -> Optional[VariantReading]:
        """Determine the most likely original reading.

        Uses both external and internal evidence.

        Args:
            unit: Variant unit

        Returns:
            Preferred reading or None
        """
        if not unit.readings:
            return None

        best_reading = None
        best_score = 0.0

        for reading in unit.readings:
            score = 0.0

            # External evidence (manuscript support)
            support = self.analyze_reading_support(reading)
            score += support.total_score * 0.6

            # Internal evidence (scribal tendencies)
            internal_score = self._calculate_internal_probability(unit.base_text, reading)
            score += internal_score * 0.4

            # Update if better
            if score > best_score:
                best_score = score
                best_reading = reading

        # Set confidence and mark as original
        if best_reading:
            best_reading.confidence = best_score
            best_reading.is_original = True

        return best_reading

    def _calculate_internal_probability(self, base_text: str, reading: VariantReading) -> float:
        """Calculate internal probability of a reading being original."""
        score = 0.5  # Start neutral

        # Shorter reading preferred (lectio brevior)
        if len(reading.text) < len(base_text):
            score += 0.1

        # More difficult reading preferred (lectio difficilior)
        # This is simplified - real implementation would analyze grammar/theology
        if reading.variant_type == VariantType.SUBSTITUTION:
            # Assume substitutions that remove theological terms are harder
            for term in self.theological_terms:
                if term in base_text and term not in reading.text:
                    score += 0.2
                    break

        # Check for harmonization (lower score)
        # This would need parallel passage database
        # For now, just a placeholder

        # Check for grammatical improvement (lower score)
        # Scribes tend to "fix" grammar

        return min(score, 1.0)

    def find_contamination(self, variants: List[TextualVariant]) -> Dict[str, Set[str]]:
        """Find potential contamination between manuscript families.

        Args:
            variants: List of textual variants

        Returns:
            Map of manuscripts showing mixed readings
        """
        contamination = defaultdict(set)

        # Track which families each manuscript agrees with
        ms_family_agreements = defaultdict(lambda: defaultdict(int))

        for variant in variants:
            for unit in variant.variant_units:
                for reading in unit.readings:
                    support = self.analyze_reading_support(reading)

                    # Track agreements
                    for witness in reading.witnesses:
                        if witness.type == WitnessType.MANUSCRIPT:
                            for family, count in support.family_support.items():
                                ms_family_agreements[witness.siglum][family] += count

        # Find manuscripts with mixed loyalties
        for ms, agreements in ms_family_agreements.items():
            if len(agreements) > 1:
                # Calculate percentages
                total = sum(agreements.values())
                percentages = {family: count / total for family, count in agreements.items()}

                # If no family has >70% agreement, likely contaminated
                if all(pct < 0.7 for pct in percentages.values()):
                    contamination[ms] = set(agreements.keys())

        return dict(contamination)
