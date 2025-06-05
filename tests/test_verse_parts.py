"""
Tests for verse parts functionality in verse_id module.
"""

import unittest

from abba.verse_id import VerseID, parse_verse_id, get_verse_parts


class TestVerseParts(unittest.TestCase):
    """Test verse parts functionality."""

    def test_get_verse_parts_simple(self):
        """Test getting parts for a simple verse."""
        # Test with a verse that has no parts
        parts = get_verse_parts("GEN.1.1")
        self.assertEqual(len(parts), 1)
        self.assertEqual(str(parts[0]), "GEN.1.1")

    def test_get_verse_parts_known_divisions(self):
        """Test getting parts for verses with known divisions."""
        # Test Romans 3:23 which has parts a and b
        parts = get_verse_parts("ROM.3.23")
        self.assertEqual(len(parts), 3)
        
        # Should return base verse plus parts a and b
        verse_strs = [str(p) for p in parts]
        self.assertIn("ROM.3.23", verse_strs)
        self.assertIn("ROM.3.23a", verse_strs)
        self.assertIn("ROM.3.23b", verse_strs)

    def test_get_verse_parts_with_verse_id(self):
        """Test getting parts with VerseID object input."""
        verse_id = VerseID("JHN", 5, 3)
        parts = get_verse_parts(verse_id)
        
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0].book, "JHN")
        self.assertEqual(parts[0].chapter, 5)
        self.assertEqual(parts[0].verse, 3)
        self.assertIsNone(parts[0].part)
        self.assertEqual(parts[1].part, "a")
        self.assertEqual(parts[2].part, "b")

    def test_get_verse_parts_already_has_part(self):
        """Test getting parts for a verse that already has a part."""
        verse_id = VerseID("ROM", 3, 23, part="a")
        parts = get_verse_parts(verse_id)
        
        # Should just return the input verse
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], verse_id)

    def test_get_verse_parts_triple_division(self):
        """Test verse with three parts (Daniel 3:23)."""
        parts = get_verse_parts("DAN.3.23")
        self.assertEqual(len(parts), 4)  # Base + a, b, c
        
        part_letters = [p.part for p in parts if p.part]
        self.assertEqual(part_letters, ["a", "b", "c"])

    def test_get_verse_parts_invalid_input(self):
        """Test with invalid verse reference."""
        parts = get_verse_parts("INVALID")
        self.assertEqual(len(parts), 0)

    def test_get_verse_parts_old_testament(self):
        """Test Old Testament verse with parts."""
        parts = get_verse_parts("PSA.145.13")
        self.assertEqual(len(parts), 3)
        
        # Verify it includes the base verse
        base_verse = next(p for p in parts if p.part is None)
        self.assertEqual(base_verse.book, "PSA")
        self.assertEqual(base_verse.chapter, 145)
        self.assertEqual(base_verse.verse, 13)

    def test_get_verse_parts_textual_variants(self):
        """Test verses that commonly have textual variants."""
        # Mark 7:16 is often bracketed or omitted
        parts = get_verse_parts("MRK.7.16")
        self.assertGreater(len(parts), 1)
        
        # Acts 8:37 is a famous textual variant
        parts = get_verse_parts("ACT.8.37")
        self.assertGreater(len(parts), 1)

    def test_verse_part_ordering(self):
        """Test that verse parts are returned in correct order."""
        parts = get_verse_parts("ROM.3.23")
        
        # First should be the base verse (no part)
        self.assertIsNone(parts[0].part)
        
        # Then parts in alphabetical order
        if len(parts) > 1:
            self.assertEqual(parts[1].part, "a")
        if len(parts) > 2:
            self.assertEqual(parts[2].part, "b")

    def test_get_verse_parts_string_formats(self):
        """Test various string input formats."""
        # Test canonical format
        parts1 = get_verse_parts("ROM.3.23")
        
        # Test common format
        parts2 = get_verse_parts("Romans 3:23")
        
        # Both should return the same verses
        self.assertEqual(len(parts1), len(parts2))
        self.assertEqual(parts1[0].book, parts2[0].book)
        self.assertEqual(parts1[0].chapter, parts2[0].chapter)
        self.assertEqual(parts1[0].verse, parts2[0].verse)


if __name__ == "__main__":
    unittest.main()