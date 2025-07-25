"""Tests for the linguistic analysis API."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from abba.api import AnalysisAPI, LexicalCluster, MorphologyPattern, WordFrequency
from abba.database import SQLiteManager


class TestAnalysisAPI(unittest.TestCase):
    """Test AnalysisAPI functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db_manager = SQLiteManager(self.db_path)
        self.db_manager.initialize_database()
        self.api = AnalysisAPI(self.db_manager)

        # Insert test data
        self._insert_test_data()

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def _insert_test_data(self):
        """Insert test data into database."""
        # Insert morphology codes
        morphology_data = [
            {"code": "HNcmsa", "description": "Hebrew Noun common masculine singular absolute", "language": "hebrew"},
            {"code": "HVqp3ms", "description": "Hebrew Verb qal perfect 3rd masculine singular", "language": "hebrew"},
            {"code": "GNnms", "description": "Greek Noun nominative masculine singular", "language": "greek"},
        ]
        for morph in morphology_data:
            self.db_manager.insert_morphology_entry(morph)

        # Insert lexicon entries
        lexicon_data = [
            {
                "strongs_number": "H1234",
                "original_word": "דָּבָר",
                "transliteration": "dabar",
                "gloss": "word",
                "definition": "speech, word, matter",
                "language": "hebrew",
            },
            {
                "strongs_number": "H1235",
                "original_word": "דֶּבֶר",
                "transliteration": "deber",
                "gloss": "pestilence",
                "definition": "plague, pestilence",
                "language": "hebrew",
            },
            {
                "strongs_number": "G3056",
                "original_word": "λόγος",
                "transliteration": "logos",
                "gloss": "word",
                "definition": "word, reason, account",
                "language": "greek",
            },
        ]
        for lex in lexicon_data:
            self.db_manager.insert_lexicon_entry(lex)

        # Insert words
        words_data = [
            {
                "book": "Gen",
                "chapter": 1,
                "verse": 1,
                "word_num": 1,
                "word_ref": "Gen.1.1.1",
                "hebrew_text": "בְּרֵאשִׁית",
                "transliteration": "bereshit",
                "translation": "in beginning",
                "strongs_primary": "H7225",
                "morphology_code": "HNcmsa",
                "language": "hebrew",
            },
            {
                "book": "Gen",
                "chapter": 1,
                "verse": 1,
                "word_num": 2,
                "word_ref": "Gen.1.1.2",
                "hebrew_text": "בָּרָא",
                "transliteration": "bara",
                "translation": "created",
                "strongs_primary": "H1254",
                "morphology_code": "HVqp3ms",
                "language": "hebrew",
            },
            {
                "book": "John",
                "chapter": 1,
                "verse": 1,
                "word_num": 1,
                "word_ref": "John.1.1.1",
                "greek_text": "λόγος",
                "transliteration": "logos",
                "translation": "word",
                "strongs_primary": "G3056",
                "morphology_code": "GNnms",
                "language": "greek",
            },
            # Hapax legomenon (appears only once)
            {
                "book": "Gen",
                "chapter": 2,
                "verse": 1,
                "word_num": 1,
                "word_ref": "Gen.2.1.1",
                "hebrew_text": "unique",
                "strongs_primary": "H9999",
                "morphology_code": "HNcmsa",
                "language": "hebrew",
            },
        ]
        for word in words_data:
            self.db_manager.insert_word(word)

    def test_analyze_morphology_patterns(self):
        """Test morphological pattern analysis."""
        # Test Hebrew patterns
        patterns = self.api.analyze_morphology_patterns(language="hebrew", limit=10)
        self.assertIsInstance(patterns, list)
        self.assertTrue(len(patterns) > 0)
        self.assertIsInstance(patterns[0], MorphologyPattern)
        self.assertEqual(patterns[0].pattern, "HNcmsa")
        self.assertEqual(patterns[0].count, 2)

        # Test with pattern filter
        verb_patterns = self.api.analyze_morphology_patterns(language="hebrew", pattern="HV%")
        self.assertTrue(len(verb_patterns) > 0)
        self.assertTrue(all(p.pattern.startswith("HV") for p in verb_patterns))

    def test_word_frequency_analysis(self):
        """Test word frequency analysis."""
        frequencies = self.api.word_frequency_analysis(min_frequency=1, limit=10)
        self.assertIsInstance(frequencies, list)
        self.assertTrue(len(frequencies) > 0)
        self.assertIsInstance(frequencies[0], WordFrequency)

        # Check sorting by frequency
        if len(frequencies) > 1:
            self.assertGreaterEqual(frequencies[0].frequency, frequencies[1].frequency)

    def test_find_hapax_legomena(self):
        """Test finding hapax legomena."""
        hapax = self.api.find_hapax_legomena(language="hebrew")
        self.assertIsInstance(hapax, list)
        self.assertTrue(len(hapax) > 0)
        # H9999 should be in the list
        strongs_numbers = [h["strongs_number"] for h in hapax]
        self.assertIn("H9999", strongs_numbers)

    def test_analyze_word_clusters(self):
        """Test lexical cluster analysis."""
        # Mock more related words
        with patch.object(self.db_manager, "execute_query") as mock_query:
            mock_query.return_value = [
                ("H1234", "דָּבָר", "word", "noun"),
                ("H1235", "דֶּבֶר", "pestilence", "noun"),
                ("H1236", "דִּבֵּר", "speak", "verb"),
            ]

            clusters = self.api.analyze_word_clusters("H123%")
            self.assertIsInstance(clusters, list)
            if clusters:
                self.assertIsInstance(clusters[0], LexicalCluster)
                self.assertEqual(clusters[0].root, "H1234")
                self.assertTrue(len(clusters[0].words) > 1)

    def test_compare_translations(self):
        """Test translation comparison."""
        # Would need translation data to test properly
        result = self.api.compare_translations("Gen", 1, 1, ["KJV", "ESV"])
        self.assertIsInstance(result, dict)
        self.assertIn("reference", result)
        self.assertIn("original_words", result)
        self.assertIn("translations", result)

    def test_semantic_domain_analysis(self):
        """Test semantic domain analysis."""
        # Test with "word" domain
        words = self.api.semantic_domain_analysis("word")
        self.assertIsInstance(words, list)
        # Should find both Hebrew and Greek "word"
        self.assertTrue(len(words) > 0)
        strongs = [w["strongs_number"] for w in words]
        self.assertIn("H1234", strongs)
        self.assertIn("G3056", strongs)

    def test_parallel_passage_detection(self):
        """Test parallel passage detection."""
        # Would need more verse data to test properly
        parallels = self.api.parallel_passage_detection("Gen", 1, 1, threshold=0.5)
        self.assertIsInstance(parallels, list)
        # With limited test data, might not find parallels
        for parallel in parallels:
            self.assertIn("reference", parallel)
            self.assertIn("similarity", parallel)
            self.assertGreaterEqual(parallel["similarity"], 0)
            self.assertLessEqual(parallel["similarity"], 1)

    def test_analyze_grammatical_constructions(self):
        """Test grammatical construction analysis."""
        # Test finding infinitives
        constructions = self.api.analyze_grammatical_constructions("perfect", language="hebrew")
        self.assertIsInstance(constructions, list)
        # Should find our perfect verb
        if constructions:
            refs = [c["reference"] for c in constructions]
            self.assertIn("Gen.1.1.2", refs)


if __name__ == "__main__":
    unittest.main()