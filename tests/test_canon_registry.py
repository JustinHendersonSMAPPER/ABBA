"""Tests for canon registry."""

import pytest
from abba.canon.registry import CanonRegistry
from abba.canon.models import (
    Canon,
    CanonTradition,
    VersificationScheme,
    BookClassification,
    BookSection,
)


class TestCanonRegistry:
    """Test CanonRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a registry for testing."""
        return CanonRegistry()

    def test_registry_initialization(self, registry):
        """Test registry initializes with default canons."""
        # Check that default canons are loaded
        assert "protestant" in registry.list_canons()
        assert "catholic" in registry.list_canons()
        assert "eastern_orthodox" in registry.list_canons()
        assert "ethiopian_orthodox" in registry.list_canons()

        # Check versification schemes
        assert "standard" in registry.list_versification_schemes()
        assert "septuagint" in registry.list_versification_schemes()
        assert "vulgate" in registry.list_versification_schemes()
        assert "masoretic" in registry.list_versification_schemes()

    def test_get_canon(self, registry):
        """Test retrieving canons."""
        protestant = registry.get_canon("protestant")
        assert protestant is not None
        assert protestant.name == "Protestant Canon"
        assert protestant.tradition == CanonTradition.PROTESTANT
        assert protestant.book_count == 66

        catholic = registry.get_canon("catholic")
        assert catholic is not None
        assert catholic.book_count == 73

        # Test non-existent canon
        assert registry.get_canon("invalid") is None

    def test_protestant_canon_structure(self, registry):
        """Test Protestant canon has correct structure."""
        canon = registry.get_canon("protestant")
        assert canon is not None

        # Check book count
        assert len(canon.books) == 66

        # Check first and last books
        book_ids = canon.get_book_ids()
        assert book_ids[0] == "GEN"  # Genesis first
        assert book_ids[-1] == "REV"  # Revelation last

        # Check some key books
        assert canon.has_book("PSA")  # Psalms
        assert canon.has_book("ISA")  # Isaiah
        assert canon.has_book("MAT")  # Matthew
        assert canon.has_book("ROM")  # Romans

        # Should not have deuterocanonical books
        assert not canon.has_book("TOB")  # Tobit
        assert not canon.has_book("WIS")  # Wisdom
        assert not canon.has_book("SIR")  # Sirach

    def test_catholic_canon_structure(self, registry):
        """Test Catholic canon has correct structure."""
        canon = registry.get_canon("catholic")
        assert canon is not None

        # Check book count
        assert len(canon.books) == 73

        # Should have all Protestant books
        assert canon.has_book("GEN")
        assert canon.has_book("REV")

        # Should also have deuterocanonical books
        assert canon.has_book("TOB")  # Tobit
        assert canon.has_book("JDT")  # Judith
        assert canon.has_book("WIS")  # Wisdom
        assert canon.has_book("SIR")  # Sirach
        assert canon.has_book("BAR")  # Baruch
        assert canon.has_book("1MA")  # 1 Maccabees
        assert canon.has_book("2MA")  # 2 Maccabees

    def test_orthodox_canon_structure(self, registry):
        """Test Eastern Orthodox canon has correct structure."""
        canon = registry.get_canon("eastern_orthodox")
        assert canon is not None

        # Should have more books than Catholic
        assert canon.book_count == 76
        assert len(canon.books) == 76

        # Should have Catholic books plus additional
        assert canon.has_book("3MA")  # 3 Maccabees
        assert canon.has_book("1ES")  # 1 Esdras
        assert canon.has_book("MAN")  # Prayer of Manasseh

    def test_ethiopian_canon_structure(self, registry):
        """Test Ethiopian Orthodox canon has correct structure."""
        canon = registry.get_canon("ethiopian_orthodox")
        assert canon is not None

        # Should be the largest canon
        assert canon.book_count == 81

        # Should have unique Ethiopian books
        assert canon.has_book("ENO")  # 1 Enoch
        assert canon.has_book("JUB")  # Jubilees

        # Note: The registry creates placeholder books for Ethiopian-specific texts
        # In a real implementation, these would need proper book code mappings

    def test_get_canons_by_tradition(self, registry):
        """Test filtering canons by tradition."""
        protestant_canons = registry.get_canons_by_tradition(CanonTradition.PROTESTANT)
        assert len(protestant_canons) == 1
        assert protestant_canons[0].id == "protestant"

        catholic_canons = registry.get_canons_by_tradition(CanonTradition.CATHOLIC)
        assert len(catholic_canons) == 1
        assert catholic_canons[0].id == "catholic"

    def test_get_book_canons(self, registry):
        """Test finding which canons contain a book."""
        # Genesis should be in all canons
        gen_canons = registry.get_book_canons("GEN")
        assert len(gen_canons) == 4

        # Tobit should not be in Protestant
        tob_canons = registry.get_book_canons("TOB")
        canon_ids = [c.id for c in tob_canons]
        assert "protestant" not in canon_ids
        assert "catholic" in canon_ids
        assert "eastern_orthodox" in canon_ids

    def test_get_common_books(self, registry):
        """Test finding common books between canons."""
        # Common books between Protestant and Catholic
        common = registry.get_common_books(["protestant", "catholic"])
        assert len(common) == 66  # All Protestant books are in Catholic
        assert "GEN" in common
        assert "REV" in common
        assert "TOB" not in common  # Tobit is not in Protestant

        # Common books among all canons
        all_common = registry.get_common_books(registry.list_canons())
        assert len(all_common) == 66  # Protestant canon is the common subset

    def test_get_unique_books(self, registry):
        """Test finding books unique to a canon."""
        # Catholic books not in Protestant
        catholic_unique = registry.get_unique_books("catholic", ["protestant"])
        assert len(catholic_unique) == 7  # 73 - 66
        assert "TOB" in catholic_unique
        assert "WIS" in catholic_unique

        # Protestant has no unique books compared to Catholic
        protestant_unique = registry.get_unique_books("protestant", ["catholic"])
        assert len(protestant_unique) == 0

    def test_register_custom_canon(self, registry):
        """Test registering a custom canon."""
        custom_canon = Canon(
            id="custom",
            name="Custom Canon",
            tradition=CanonTradition.PROTESTANT,
            description="Test custom canon",
            book_count=5,
        )

        registry.register_canon(custom_canon)

        assert "custom" in registry.list_canons()
        retrieved = registry.get_canon("custom")
        assert retrieved is not None
        assert retrieved.name == "Custom Canon"

    def test_versification_schemes(self, registry):
        """Test versification scheme operations."""
        standard = registry.get_versification_scheme("standard")
        assert standard is not None
        assert standard.name == "Standard"
        assert not standard.includes_apocrypha

        lxx = registry.get_versification_scheme("septuagint")
        assert lxx is not None
        assert lxx.includes_apocrypha
        assert "psalms" in lxx.differences

    def test_book_classifications(self, registry):
        """Test book classifications in canons."""
        catholic = registry.get_canon("catholic")

        # Count books by classification
        proto_count = sum(
            1 for cb in catholic.books if cb.classification == BookClassification.PROTOCANONICAL
        )
        deutero_count = sum(
            1 for cb in catholic.books if cb.classification == BookClassification.DEUTEROCANONICAL
        )

        assert proto_count == 66  # Same as Protestant
        assert deutero_count == 7  # Catholic additions

    def test_book_sections(self, registry):
        """Test book sections in canons."""
        protestant = registry.get_canon("protestant")

        # Count by section
        ot_count = sum(1 for cb in protestant.books if cb.section == BookSection.OLD_TESTAMENT)
        nt_count = sum(1 for cb in protestant.books if cb.section == BookSection.NEW_TESTAMENT)

        assert ot_count == 39
        assert nt_count == 27
        assert ot_count + nt_count == 66
