"""Tests for manuscript variant analyzer."""

import pytest

from abba.manuscript.analyzer import VariantAnalyzer
from abba.manuscript.models import VariantReading, VariantUnit, VariantType, Witness, WitnessType


class TestVariantAnalyzer:
    """Test variant analyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = VariantAnalyzer()

    def test_analyze_simple_variant(self):
        """Test analyzing a simple variant."""
        # Create variant unit with two readings
        reading1 = VariantReading(
            text="εν χριστω ιησου",
            witnesses=[
                Witness(siglum="P46", type=WitnessType.MANUSCRIPT),
                Witness(siglum="B", type=WitnessType.MANUSCRIPT),
            ],
        )
        reading2 = VariantReading(
            text="εν ιησου χριστω", witnesses=[Witness(siglum="D", type=WitnessType.MANUSCRIPT)]
        )

        unit = VariantUnit(
            start_word=1, end_word=3, readings=[reading1, reading2], type=VariantType.WORD_ORDER
        )

        analysis = self.analyzer.analyze_variant(unit)

        assert analysis is not None
        assert analysis.type == VariantType.WORD_ORDER
        assert analysis.num_readings == 2
        assert analysis.significance in ["minor", "moderate", "major"]
        assert "word order" in analysis.description.lower()

    def test_analyze_omission(self):
        """Test analyzing an omission variant."""
        reading1 = VariantReading(
            text="του θεου",
            witnesses=[
                Witness(siglum="P46", type=WitnessType.MANUSCRIPT),
                Witness(siglum="א", type=WitnessType.MANUSCRIPT),
            ],
        )
        reading2 = VariantReading(
            text="om", witnesses=[Witness(siglum="B", type=WitnessType.MANUSCRIPT)]
        )

        unit = VariantUnit(
            start_word=5, end_word=6, readings=[reading1, reading2], type=VariantType.OMISSION
        )

        analysis = self.analyzer.analyze_variant(unit)

        assert analysis.type == VariantType.OMISSION
        assert analysis.affects_meaning == True  # Omission of "of God" affects meaning
        assert analysis.significance in ["moderate", "major"]

    def test_analyze_theological_variant(self):
        """Test analyzing theological variant."""
        # Example: "Jesus" vs "Lord Jesus"
        reading1 = VariantReading(
            text="ιησου",
            witnesses=[
                Witness(siglum="P46", type=WitnessType.MANUSCRIPT),
                Witness(siglum="B", type=WitnessType.MANUSCRIPT),
            ],
        )
        reading2 = VariantReading(
            text="κυριου ιησου",
            witnesses=[
                Witness(siglum="D", type=WitnessType.MANUSCRIPT),
                Witness(siglum="F", type=WitnessType.MANUSCRIPT),
            ],
        )

        unit = VariantUnit(
            start_word=10, end_word=10, readings=[reading1, reading2], type=VariantType.ADDITION
        )

        analysis = self.analyzer.analyze_variant(unit)

        assert analysis.type == VariantType.ADDITION
        assert analysis.affects_meaning == True
        assert any("theological" in factor.lower() for factor in analysis.factors)

    def test_detect_christological_title(self):
        """Test detection of Christological titles."""
        # Test various titles
        assert self.analyzer._is_christological_title("κυριος")
        assert self.analyzer._is_christological_title("χριστος")
        assert self.analyzer._is_christological_title("σωτηρ")
        assert self.analyzer._is_christological_title("υιος θεου")

        # Test non-titles
        assert not self.analyzer._is_christological_title("και")
        assert not self.analyzer._is_christological_title("εν")

    def test_calculate_significance(self):
        """Test significance calculation."""
        # Minor: spelling variant
        unit1 = VariantUnit(
            readings=[VariantReading(text="ιωαννης"), VariantReading(text="ιωανης")],
            type=VariantType.SPELLING,
        )
        analysis1 = self.analyzer.analyze_variant(unit1)
        assert analysis1.significance == "minor"

        # Major: theological addition
        unit2 = VariantUnit(
            readings=[VariantReading(text="ιησου"), VariantReading(text="κυριου ιησου χριστου")],
            type=VariantType.ADDITION,
        )
        analysis2 = self.analyzer.analyze_variant(unit2)
        assert analysis2.significance == "major"

    def test_analyze_multiple_variants(self):
        """Test analyzing multiple variants in a verse."""
        units = [
            VariantUnit(
                start_word=1,
                end_word=1,
                readings=[VariantReading(text="εν"), VariantReading(text="om")],
                type=VariantType.OMISSION,
            ),
            VariantUnit(
                start_word=5,
                end_word=6,
                readings=[VariantReading(text="χριστω ιησου"), VariantReading(text="ιησου χριστω")],
                type=VariantType.WORD_ORDER,
            ),
        ]

        analyses = [self.analyzer.analyze_variant(unit) for unit in units]

        assert len(analyses) == 2
        assert all(a is not None for a in analyses)
        assert analyses[0].type == VariantType.OMISSION
        assert analyses[1].type == VariantType.WORD_ORDER

    def test_get_description(self):
        """Test variant description generation."""
        # Test word order
        unit1 = VariantUnit(
            readings=[VariantReading(text="A B"), VariantReading(text="B A")],
            type=VariantType.WORD_ORDER,
        )
        desc1 = self.analyzer._get_variant_description(unit1)
        assert "word order" in desc1.lower()

        # Test omission
        unit2 = VariantUnit(
            readings=[VariantReading(text="word"), VariantReading(text="om")],
            type=VariantType.OMISSION,
        )
        desc2 = self.analyzer._get_variant_description(unit2)
        assert "omission" in desc2.lower() or "omitted" in desc2.lower()

    def test_theological_factors(self):
        """Test detection of theological factors."""
        # Divine name addition
        unit = VariantUnit(
            readings=[VariantReading(text="ιησου"), VariantReading(text="ιησου του θεου")],
            type=VariantType.ADDITION,
        )

        analysis = self.analyzer.analyze_variant(unit)
        factors = analysis.factors

        assert any("divine" in f.lower() or "theological" in f.lower() for f in factors)
