"""Tests for manuscript variant models."""

import pytest
from datetime import datetime

from abba.manuscript.models import (
    ManuscriptType,
    ManuscriptFamily,
    Manuscript,
    WitnessType,
    Witness,
    VariantType,
    VariantReading,
    VariantUnit,
    TextualVariant,
)


class TestManuscript:
    """Test Manuscript model."""

    def test_manuscript_creation(self):
        """Test creating a manuscript."""
        ms = Manuscript(
            siglum="P46",
            name="Chester Beatty II",
            type=ManuscriptType.PAPYRUS,
            date="ca. 200",
            date_numeric=200,
            family=ManuscriptFamily.ALEXANDRIAN,
            location="Dublin, Chester Beatty Library",
            contents=["Romans", "1 Corinthians", "2 Corinthians"],
            lacunae=["Rom 1:1-5:17", "2 Cor 11:33-12:9"],
            accuracy_rating=0.95,
            completeness=0.85,
        )

        assert ms.siglum == "P46"
        assert ms.type == ManuscriptType.PAPYRUS
        assert ms.family == ManuscriptFamily.ALEXANDRIAN
        assert ms.accuracy_rating == 0.95
        assert "Romans" in ms.contents
        assert len(ms.lacunae) == 2

    def test_manuscript_str(self):
        """Test manuscript string representation."""
        ms = Manuscript(siglum="B", name="Codex Vaticanus", type=ManuscriptType.UNCIAL)

        assert str(ms) == "B (Codex Vaticanus)"

    def test_manuscript_defaults(self):
        """Test manuscript default values."""
        ms = Manuscript(siglum="Test", type=ManuscriptType.MINUSCULE)

        assert ms.name is None
        assert ms.date is None
        assert ms.family == ManuscriptFamily.UNKNOWN
        assert ms.accuracy_rating == 0.5
        assert ms.completeness == 0.5
        assert ms.contents == []
        assert ms.lacunae == []


class TestWitness:
    """Test Witness model."""

    def test_witness_creation(self):
        """Test creating a witness."""
        witness = Witness(
            siglum="א",
            type=WitnessType.MANUSCRIPT,
            text="εν χριστω ιησου",
            corrector="c",
            marginal=False,
            partial=False,
            videtur=False,
        )

        assert witness.siglum == "א"
        assert witness.type == WitnessType.MANUSCRIPT
        assert witness.text == "εν χριστω ιησου"
        assert witness.corrector == "c"
        assert not witness.marginal

    def test_witness_str(self):
        """Test witness string representation."""
        witness = Witness(siglum="P75", type=WitnessType.MANUSCRIPT)
        assert str(witness) == "P75"

        witness_corrector = Witness(siglum="א", type=WitnessType.MANUSCRIPT, corrector="2")
        assert str(witness_corrector) == "א2"

        witness_marginal = Witness(siglum="L", type=WitnessType.MANUSCRIPT, marginal=True)
        assert str(witness_marginal) == "Lmg"

    def test_witness_complex_str(self):
        """Test complex witness string representation."""
        witness = Witness(
            siglum="D", type=WitnessType.MANUSCRIPT, corrector="1", marginal=True, videtur=True
        )
        assert str(witness) == "D1mgvid"


class TestVariantReading:
    """Test VariantReading model."""

    def test_variant_reading_creation(self):
        """Test creating a variant reading."""
        witnesses = [
            Witness(siglum="P46", type=WitnessType.MANUSCRIPT),
            Witness(siglum="B", type=WitnessType.MANUSCRIPT),
            Witness(siglum="א", type=WitnessType.MANUSCRIPT),
        ]

        reading = VariantReading(
            text="εν χριστω ιησου", witnesses=witnesses, is_original=True, confidence=0.95
        )

        assert reading.text == "εν χριστω ιησου"
        assert len(reading.witnesses) == 3
        assert reading.is_original
        assert reading.confidence == 0.95

    def test_variant_reading_defaults(self):
        """Test variant reading defaults."""
        reading = VariantReading(text="om")

        assert reading.text == "om"
        assert reading.witnesses == []
        assert not reading.is_original
        assert reading.confidence == 0.0

    def test_variant_reading_support_str(self):
        """Test variant reading support string."""
        witnesses = [
            Witness(siglum="P46", type=WitnessType.MANUSCRIPT),
            Witness(siglum="B", type=WitnessType.MANUSCRIPT),
            Witness(siglum="א", type=WitnessType.MANUSCRIPT, corrector="c"),
        ]

        reading = VariantReading(text="test", witnesses=witnesses)
        assert reading.get_support_string() == "P46 B אc"


class TestVariantUnit:
    """Test VariantUnit model."""

    def test_variant_unit_creation(self):
        """Test creating a variant unit."""
        reading1 = VariantReading(
            text="εν χριστω ιησου", witnesses=[Witness(siglum="P46", type=WitnessType.MANUSCRIPT)]
        )
        reading2 = VariantReading(
            text="εν ιησου χριστω", witnesses=[Witness(siglum="D", type=WitnessType.MANUSCRIPT)]
        )

        unit = VariantUnit(
            start_word=5,
            end_word=7,
            readings=[reading1, reading2],
            type=VariantType.WORD_ORDER,
            significance="minor",
        )

        assert unit.start_word == 5
        assert unit.end_word == 7
        assert len(unit.readings) == 2
        assert unit.type == VariantType.WORD_ORDER
        assert unit.significance == "minor"


class TestTextualVariant:
    """Test TextualVariant model."""

    def test_textual_variant_creation(self):
        """Test creating a textual variant."""
        unit1 = VariantUnit(
            start_word=1,
            end_word=1,
            readings=[VariantReading(text="εν"), VariantReading(text="om")],
        )

        unit2 = VariantUnit(
            start_word=5,
            end_word=7,
            readings=[VariantReading(text="χριστω ιησου"), VariantReading(text="ιησου χριστω")],
        )

        variant = TextualVariant(
            reference="Rom.1.1",
            text="Παυλος δουλος χριστου ιησου",
            units=[unit1, unit2],
            base_text="NA28",
            confidence_rating="B",
        )

        assert variant.reference == "Rom.1.1"
        assert len(variant.units) == 2
        assert variant.base_text == "NA28"
        assert variant.confidence_rating == "B"

    def test_textual_variant_defaults(self):
        """Test textual variant defaults."""
        variant = TextualVariant(reference="test", text="test text")

        assert variant.units == []
        assert variant.base_text == "NA28"
        assert variant.confidence_rating == "C"
