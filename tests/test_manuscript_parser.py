"""Tests for critical apparatus parser."""

import pytest

from abba.manuscript.parser import CriticalApparatusParser
from abba.manuscript.models import WitnessType, VariantType


class TestCriticalApparatusParser:
    """Test critical apparatus parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CriticalApparatusParser()

    def test_parse_na28_simple(self):
        """Test parsing simple NA28 apparatus."""
        # Example: word omitted by some manuscripts
        notation = "om. D* it"
        variant = self.parser.parse_na28_notation(notation, "εν")

        assert variant is not None
        assert len(variant.readings) == 2

        # Check omission reading
        om_reading = next(r for r in variant.readings if r.text == "om")
        assert om_reading is not None
        assert len(om_reading.witnesses) == 2
        witness_sigla = [str(w) for w in om_reading.witnesses]
        assert "D*" in witness_sigla
        assert "it" in witness_sigla

        # Check inclusion reading
        inc_reading = next(r for r in variant.readings if r.text == "εν")
        assert inc_reading is not None

    def test_parse_na28_substitution(self):
        """Test parsing NA28 substitution."""
        # Example: different word order
        notation = "ιησου χριστου P46 B א | χριστου ιησου D it"
        variant = self.parser.parse_na28_notation(notation, "χριστου ιησου")

        assert variant is not None
        assert len(variant.readings) == 2

        # Check first reading
        reading1 = variant.readings[0]
        assert reading1.text == "ιησου χριστου"
        assert len(reading1.witnesses) == 3
        witness_sigla = [w.siglum for w in reading1.witnesses]
        assert "P46" in witness_sigla
        assert "B" in witness_sigla
        assert "א" in witness_sigla

        # Check second reading
        reading2 = variant.readings[1]
        assert reading2.text == "χριστου ιησου"
        assert len(reading2.witnesses) == 2

    def test_parse_na28_addition(self):
        """Test parsing NA28 addition."""
        notation = "post ιησου add. κυριου D"
        variant = self.parser.parse_na28_notation(notation, "ιησου")

        assert variant is not None
        assert variant.type == VariantType.ADDITION

        # Should have two readings: with and without addition
        assert len(variant.readings) == 2

        # Check addition reading
        add_reading = next(r for r in variant.readings if "κυριου" in r.text)
        assert add_reading is not None
        assert len(add_reading.witnesses) == 1
        assert add_reading.witnesses[0].siglum == "D"

    def test_parse_witness_simple(self):
        """Test parsing simple witness."""
        witness = self.parser.parse_witness("P46")

        assert witness is not None
        assert witness.siglum == "P46"
        assert witness.type == WitnessType.MANUSCRIPT
        assert witness.corrector is None
        assert not witness.marginal
        assert not witness.videtur

    def test_parse_witness_with_corrector(self):
        """Test parsing witness with corrector."""
        witness = self.parser.parse_witness("אc")

        assert witness is not None
        assert witness.siglum == "א"
        assert witness.corrector == "c"

        # Test numbered corrector
        witness2 = self.parser.parse_witness("D2")
        assert witness2.siglum == "D"
        assert witness2.corrector == "2"

    def test_parse_witness_original_hand(self):
        """Test parsing original hand notation."""
        witness = self.parser.parse_witness("B*")

        assert witness is not None
        assert witness.siglum == "B"
        assert witness.corrector is None  # Original hand

    def test_parse_witness_marginal(self):
        """Test parsing marginal reading."""
        witness = self.parser.parse_witness("Lmg")

        assert witness is not None
        assert witness.siglum == "L"
        assert witness.marginal

    def test_parse_witness_videtur(self):
        """Test parsing uncertain reading."""
        witness = self.parser.parse_witness("Cvid")

        assert witness is not None
        assert witness.siglum == "C"
        assert witness.videtur

    def test_parse_witness_complex(self):
        """Test parsing complex witness notation."""
        witness = self.parser.parse_witness("א2mgvid")

        assert witness is not None
        assert witness.siglum == "א"
        assert witness.corrector == "2"
        assert witness.marginal
        assert witness.videtur

    def test_parse_witness_version(self):
        """Test parsing version witness."""
        witness = self.parser.parse_witness("it")

        assert witness is not None
        assert witness.siglum == "it"
        assert witness.type == WitnessType.VERSION

        # Test specific version
        witness2 = self.parser.parse_witness("ita")
        assert witness2.siglum == "ita"
        assert witness2.type == WitnessType.VERSION

    def test_parse_witness_church_father(self):
        """Test parsing church father witness."""
        witness = self.parser.parse_witness("Orig")

        assert witness is not None
        assert witness.siglum == "Orig"
        assert witness.type == WitnessType.FATHER

    def test_parse_witness_list(self):
        """Test parsing witness list."""
        witnesses = self.parser.parse_witness_list("P46 B א* D2mg it")

        assert len(witnesses) == 5

        # Check each witness
        assert witnesses[0].siglum == "P46"
        assert witnesses[1].siglum == "B"
        assert witnesses[2].siglum == "א"
        assert witnesses[3].siglum == "D"
        assert witnesses[3].corrector == "2"
        assert witnesses[3].marginal
        assert witnesses[4].siglum == "it"
        assert witnesses[4].type == WitnessType.VERSION

    def test_parse_ubs5_notation(self):
        """Test parsing UBS5 notation."""
        # UBS5 uses {A} {B} {C} {D} ratings
        notation = "{B} εν χριστω ιησου P46 B א | χριστω ιησου D | om. it"
        variant = self.parser.parse_ubs5_notation(notation)

        assert variant is not None
        assert variant.confidence_rating == "B"
        assert len(variant.readings) == 3

        # Check readings
        assert variant.readings[0].text == "εν χριστω ιησου"
        assert variant.readings[1].text == "χριστω ιησου"
        assert variant.readings[2].text == "om"

    def test_parse_generic_notation(self):
        """Test parsing generic notation."""
        notation = "εν χριστω] εν τω χριστω D; om. it"
        variant = self.parser.parse_generic_notation(notation)

        assert variant is not None
        assert len(variant.readings) >= 2

    def test_detect_variant_type(self):
        """Test variant type detection."""
        # Test omission
        readings = [self.parser.parse_na28_notation("om. D", "word").readings[0]]
        assert self.parser._detect_variant_type(readings) == VariantType.OMISSION

        # Test word order (would need more complex example)
        # Test spelling (would need Greek text analysis)

    def test_parse_lacuna_notation(self):
        """Test parsing lacuna notation."""
        witness = self.parser.parse_witness("P46lac")

        # Lacuna should be indicated differently
        assert witness is None or witness.partial

    def test_parse_majority_text(self):
        """Test parsing majority text notation."""
        witnesses = self.parser.parse_witness_list("𝔐 K L")

        # Should recognize majority text siglum
        assert any(w.siglum == "𝔐" for w in witnesses)
