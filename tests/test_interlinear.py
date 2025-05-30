"""Unit tests for interlinear text processing."""

import pytest

from abba.interlinear import (
    AlignedToken,
    AlignmentType,
    ExtractedToken,
    GreekTokenExtractor,
    HebrewTokenExtractor,
    InterlinearGenerator,
    InterlinearVerse,
    InterlinearWord,
    LexiconIntegrator,
    LexicalEntry,
    SemanticDomain,
    TokenAligner,
    TokenAlignment,
)
from abba.morphology import Language, UnifiedMorphologyParser
from abba.parsers.greek_parser import GreekVerse, GreekWord
from abba.parsers.hebrew_parser import HebrewVerse, HebrewWord
from abba.verse_id import parse_verse_id


class TestTokenExtractors:
    """Test token extraction from biblical texts."""

    def test_hebrew_token_extraction(self) -> None:
        """Test extracting tokens from Hebrew verse."""
        extractor = HebrewTokenExtractor()

        # Create test verse
        verse = HebrewVerse(
            verse_id=parse_verse_id("GEN.1.1"),
            osis_id="Gen.1.1",
            words=[
                HebrewWord(
                    text="בְּרֵאשִׁית",
                    lemma="7225",
                    strong_number="H7225",
                    morph="HNcfsa",
                    gloss="beginning",
                ),
                HebrewWord(
                    text="בָּרָא",
                    lemma="1254 a",
                    strong_number="H1254",
                    morph="Vqp3ms",
                    gloss="created",
                ),
            ],
        )

        tokens = extractor.extract_tokens(verse)

        assert len(tokens) == 2
        assert tokens[0].text == "בְּרֵאשִׁית"
        assert tokens[0].strong_number == "H7225"
        assert tokens[0].position == 0
        assert tokens[0].language == Language.HEBREW
        assert tokens[0].transliteration  # Should have transliteration

        assert tokens[1].text == "בָּרָא"
        assert tokens[1].strong_number == "H1254"
        assert tokens[1].morphology is not None

    def test_hebrew_text_extraction(self) -> None:
        """Test extracting tokens from Hebrew plain text."""
        extractor = HebrewTokenExtractor()

        text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        tokens = extractor.extract_tokens_from_text(text)

        assert len(tokens) == 3
        assert tokens[0].text == "בְּרֵאשִׁית"
        assert tokens[1].text == "בָּרָא"
        assert tokens[2].text == "אֱלֹהִים"
        assert all(t.language == Language.HEBREW for t in tokens)

    def test_greek_token_extraction(self) -> None:
        """Test extracting tokens from Greek verse."""
        extractor = GreekTokenExtractor()

        verse = GreekVerse(
            verse_id=parse_verse_id("JHN.1.1"),
            tei_id="B04K1V1",
            words=[
                GreekWord(text="Ἐν", lemma="G1722", morph="P", id="w1"),
                GreekWord(text="ἀρχῇ", lemma="G746", morph="N-DSF", id="w2"),
            ],
        )

        tokens = extractor.extract_tokens(verse)

        assert len(tokens) == 2
        assert tokens[0].text == "Ἐν"
        assert tokens[0].strong_number == "G1722"
        assert tokens[0].transliteration == "En"

        assert tokens[1].text == "ἀρχῇ"
        assert tokens[1].strong_number == "G746"
        assert tokens[1].morphology is not None

    def test_greek_transliteration(self) -> None:
        """Test Greek transliteration."""
        extractor = GreekTokenExtractor()

        # Test various Greek words
        assert extractor._transliterate("Ἰησοῦς") == "Iēsous"
        assert extractor._transliterate("Χριστός") == "Christos"
        assert extractor._transliterate("καί") == "kai"
        assert extractor._transliterate("οὐ") == "ou"


class TestTokenAlignment:
    """Test token alignment functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.aligner = TokenAligner()

        # Add Strong's mapping for testing
        self.aligner.strongs_mapping = {
            "H7225": ["beginning", "first"],
            "H1254": ["created", "create"],
            "H430": ["God", "gods"],
        }

    def test_simple_alignment(self) -> None:
        """Test simple one-to-one alignment."""
        source_tokens = [
            ExtractedToken(
                text="בְּרֵאשִׁית", strong_number="H7225", position=0, language=Language.HEBREW
            ),
            ExtractedToken(text="בָּרָא", strong_number="H1254", position=1, language=Language.HEBREW),
        ]

        target_text = "In beginning created"

        alignment = self.aligner.align_tokens(
            source_tokens, target_text, "GEN.1.1", "hebrew", "english"
        )

        assert len(alignment.alignments) >= 2
        assert alignment.alignment_score > 0

        # Check first alignment
        first_align = alignment.alignments[0]
        assert first_align.alignment_type == AlignmentType.ONE_TO_ONE
        assert first_align.strong_numbers == ["H7225"]
        assert "beginning" in first_align.target_words

    def test_one_to_many_alignment(self) -> None:
        """Test one-to-many token alignment."""
        source_tokens = [
            ExtractedToken(
                text="אֱלֹהִים", strong_number="H430", position=0, language=Language.HEBREW
            ),
        ]

        target_text = "the God of heaven"

        alignment = self.aligner.align_tokens(
            source_tokens, target_text, "GEN.1.1", "hebrew", "english"
        )

        # Should find "God" in target
        assert len(alignment.alignments) >= 1
        god_alignment = alignment.alignments[0]
        assert "God" in god_alignment.target_words

    def test_positional_alignment(self) -> None:
        """Test positional alignment fallback."""
        # Tokens without Strong's numbers
        source_tokens = [
            ExtractedToken(text="word1", position=0, language=Language.HEBREW),
            ExtractedToken(text="word2", position=1, language=Language.HEBREW),
        ]

        target_text = "target1 target2"

        alignment = self.aligner.align_tokens(source_tokens, target_text, "TEST.1.1")

        # Should align by position
        assert len(alignment.alignments) == 2
        assert alignment.alignments[0].alignment_type == AlignmentType.ONE_TO_ONE
        assert alignment.alignments[0].target_words == ["target1"]
        assert alignment.alignments[1].target_words == ["target2"]

    def test_alignment_coverage_stats(self) -> None:
        """Test alignment coverage statistics."""
        alignment = TokenAlignment(
            verse_id="TEST.1.1",
            source_language="hebrew",
            target_language="english",
            alignments=[
                AlignedToken(
                    source_tokens=[ExtractedToken("test", position=0)],
                    target_words=["test"],
                    alignment_type=AlignmentType.ONE_TO_ONE,
                )
            ],
            unaligned_source=[ExtractedToken("unaligned", position=1)],
            unaligned_target=["extra"],
        )

        stats = alignment.get_coverage_stats()

        assert stats["source_coverage"] == 0.5  # 1 of 2 aligned
        assert stats["target_coverage"] == 0.5  # 1 of 2 aligned
        assert stats["overall_coverage"] == 0.5


class TestInterlinearGeneration:
    """Test interlinear data generation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.generator = InterlinearGenerator()

    def test_hebrew_interlinear_generation(self) -> None:
        """Test generating Hebrew interlinear data."""
        verse = HebrewVerse(
            verse_id=parse_verse_id("GEN.1.1"),
            osis_id="Gen.1.1",
            words=[
                HebrewWord(
                    text="בְּרֵאשִׁית",
                    lemma="7225",
                    strong_number="H7225",
                    morph="HNcfsa",
                    gloss="beginning",
                ),
                HebrewWord(
                    text="בָּרָא", lemma="1254", strong_number="H1254", morph="Vqp3ms", gloss="created"
                ),
            ],
        )

        interlinear = self.generator.generate_hebrew_interlinear(verse, "In the beginning created")

        assert interlinear.verse_id == verse.verse_id
        assert interlinear.language == Language.HEBREW
        assert len(interlinear.words) == 2

        # Check first word
        word1 = interlinear.words[0]
        assert word1.original_text == "בְּרֵאשִׁית"
        assert word1.strong_number == "H7225"
        assert word1.gloss == "beginning"
        assert word1.morphology is not None
        assert word1.morphology_gloss is not None

    def test_greek_interlinear_generation(self) -> None:
        """Test generating Greek interlinear data."""
        verse = GreekVerse(
            verse_id=parse_verse_id("JHN.1.1"),
            tei_id="B04K1V1",
            words=[
                GreekWord(text="Ἐν", lemma="G1722", morph="P"),
                GreekWord(text="ἀρχῇ", lemma="G746", morph="N-DSF"),
            ],
        )

        interlinear = self.generator.generate_greek_interlinear(verse, "In the beginning")

        assert interlinear.language == Language.GREEK
        assert len(interlinear.words) == 2

        word1 = interlinear.words[0]
        assert word1.original_text == "Ἐν"
        assert word1.transliteration == "En"

        word2 = interlinear.words[1]
        assert word2.morphology is not None
        assert "noun" in word2.morphology_gloss.lower()

    def test_interlinear_display_aligned(self) -> None:
        """Test aligned interlinear display format."""
        word1 = InterlinearWord(
            original_text="בְּרֵאשִׁית",
            transliteration="bereshit",
            morphology_gloss="noun fem sing abs",
            gloss="beginning",
            translation_words=["In", "beginning"],
            position=0,
        )

        word2 = InterlinearWord(
            original_text="בָּרָא",
            transliteration="bara",
            morphology_gloss="verb qal perf 3ms",
            gloss="created",
            translation_words=["created"],
            position=1,
        )

        verse = InterlinearVerse(
            verse_id=parse_verse_id("GEN.1.1"), language=Language.HEBREW, words=[word1, word2]
        )

        display = verse.get_interlinear_display("aligned")

        # Should have 5 lines
        lines = display.split("\n")
        assert len(lines) == 5

        # Check that Hebrew is on first line
        assert "בְּרֵאשִׁית" in lines[0]
        assert "בָּרָא" in lines[0]

    def test_interlinear_to_dict(self) -> None:
        """Test converting interlinear data to dictionary."""
        word = InterlinearWord(
            original_text="test",
            transliteration="test",
            strong_number="H1234",
            gloss="test",
            position=0,
        )

        result = word.to_dict()

        assert result["original"] == "test"
        assert result["transliteration"] == "test"
        assert result["strong_number"] == "H1234"
        assert result["gloss"] == "test"
        assert result["position"] == 0


class TestLexiconIntegration:
    """Test lexicon integration functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.integrator = LexiconIntegrator()

        # Add test entries
        self.integrator.entries["H430"] = LexicalEntry(
            strong_number="H430",
            lemma="אֱלֹהִים",
            gloss="God",
            definition="God, gods, judges, angels",
            language=Language.HEBREW,
            semantic_domains=[SemanticDomain.GOD_DIVINE],
            frequency=2600,
        )

        self.integrator.entries["H1254"] = LexicalEntry(
            strong_number="H1254",
            lemma="בָּרָא",
            gloss="create",
            definition="to create, shape, form",
            language=Language.HEBREW,
            semantic_domains=[SemanticDomain.CREATION_NATURE],
            frequency=54,
        )

        # Update lemma index
        self.integrator._lemma_index["אֱלֹהִים"] = ["H430"]
        self.integrator._lemma_index["בָּרָא"] = ["H1254"]

    def test_get_entry(self) -> None:
        """Test getting lexicon entry."""
        entry = self.integrator.get_entry("H430")

        assert entry is not None
        assert entry.lemma == "אֱלֹהִים"
        assert entry.gloss == "God"
        assert SemanticDomain.GOD_DIVINE in entry.semantic_domains

    def test_search_by_lemma(self) -> None:
        """Test searching by lemma."""
        # Exact search
        results = self.integrator.search_by_lemma("אֱלֹהִים", exact=True)
        assert len(results) == 1
        assert results[0].strong_number == "H430"

        # Substring search
        results = self.integrator.search_by_lemma("אֱלֹ", exact=False)
        assert len(results) == 1

    def test_search_by_gloss(self) -> None:
        """Test searching by gloss."""
        results = self.integrator.search_by_gloss("God")

        assert len(results) == 1
        assert results[0].strong_number == "H430"

    def test_search_by_semantic_domain(self) -> None:
        """Test searching by semantic domain."""
        results = self.integrator.search_by_semantic_domain(SemanticDomain.GOD_DIVINE)

        assert len(results) == 1
        assert results[0].strong_number == "H430"

        results = self.integrator.search_by_semantic_domain(SemanticDomain.CREATION_NATURE)
        assert len(results) == 1
        assert results[0].strong_number == "H1254"

    def test_frequency_analysis(self) -> None:
        """Test frequency analysis."""
        analysis = self.integrator.get_frequency_analysis()

        assert analysis["total_entries"] == 2
        assert analysis["total_occurrences"] == 2654  # 2600 + 54
        assert len(analysis["top_words"]) == 2
        assert analysis["top_words"][0]["strong_number"] == "H430"  # Higher frequency

    def test_semantic_classification(self) -> None:
        """Test automatic semantic domain classification."""
        domains = self.integrator._classify_semantic_domains(
            "worship in the temple with sacrifice", "worship"
        )

        assert SemanticDomain.WORSHIP_RITUAL in domains

    def test_lexical_entry_to_dict(self) -> None:
        """Test converting lexical entry to dictionary."""
        entry = self.integrator.get_entry("H430")
        result = entry.to_dict()

        assert result["strong_number"] == "H430"
        assert result["lemma"] == "אֱלֹהִים"
        assert result["language"] == "hebrew"
        assert "god_divine" in result["semantic_domains"]


class TestInterlinearIntegration:
    """Test integration of interlinear components."""

    def test_full_interlinear_workflow(self) -> None:
        """Test complete interlinear generation workflow."""
        # Create components
        extractor = HebrewTokenExtractor()
        aligner = TokenAligner(strongs_mapping={"H430": ["God"]})
        generator = InterlinearGenerator()

        # Create test verse
        verse = HebrewVerse(
            verse_id=parse_verse_id("GEN.1.1"),
            osis_id="Gen.1.1",
            words=[HebrewWord(text="אֱלֹהִים", lemma="430", strong_number="H430", morph="Ncmpa")],
        )

        # Extract tokens
        tokens = extractor.extract_tokens(verse)

        # Align with translation
        translation = "God created"
        alignment = aligner.align_tokens(tokens, translation, "GEN.1.1")

        # Generate interlinear
        interlinear = generator.generate_hebrew_interlinear(verse, translation, alignment)

        assert len(interlinear.words) == 1
        assert interlinear.words[0].translation_words  # Should have aligned translation
        assert "God" in interlinear.words[0].translation_words
