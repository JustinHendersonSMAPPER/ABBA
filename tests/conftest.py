"""Shared test fixtures for ABBA integration tests."""

# pylint: disable=redefined-outer-name
import pytest

from abba.database import SQLiteManager


@pytest.fixture
def seeded_db(tmp_path):  # noqa: C901 - linear seed inserts; complexity is inherent, not a code smell
    """Create a fully seeded test database with realistic biblical data.

    This fixture provides a database populated with a small but representative
    slice of biblical data: translations, books, verses, original-language words,
    lexicon entries, and morphology codes.  It is designed for end-to-end tests
    that exercise the full API surface without requiring any external downloads.
    """
    db_path = tmp_path / "test_abba.db"
    db = SQLiteManager(db_path)
    db.initialize_database()

    # --- Translations (provide canon directly to skip bible.db dependency) ---
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("engbsb", "Berean Standard Bible", "Berean Standard Bible", "en", "protestant"),
    )
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("engkjv", "King James Version", "King James Version", "en", "protestant"),
    )
    # Real default-translation id used by the app (DEFAULT_TRANSLATION_ID = "BSB").
    db.execute_update(
        "INSERT OR REPLACE INTO translations (id, name, english_name, language, canon) VALUES (?, ?, ?, ?, ?)",
        ("BSB", "Berean Standard Bible", "Berean Standard Bible", "eng", "protestant"),
    )

    # --- Books ---
    books = [
        ("engbsb", 1, "Genesis", "Genesis", 1, 50, "old"),
        ("engbsb", 43, "John", "John", 43, 21, "new"),
        ("engkjv", 1, "Genesis", "Genesis", 1, 50, "old"),
        ("engkjv", 43, "John", "John", 43, 21, "new"),
        ("BSB", 1, "Genesis", "Genesis", 1, 50, "old"),
        ("BSB", 43, "John", "John", 43, 21, "new"),
    ]
    for tid, bid, name, common, order, chapters, testament in books:
        db.execute_update(
            "INSERT OR REPLACE INTO books (translation_id, book_id, name, common_name, book_order, "
            "number_of_chapters, testament) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, bid, name, common, order, chapters, testament),
        )

    # --- Verses (BSB) ---
    bsb_gen1 = [
        (1, 1, "In the beginning God created the heavens and the earth."),
        (1, 2, "Now the earth was formless and void, and darkness was over the surface of the deep."),
        (1, 3, "And God said, 'Let there be light,' and there was light."),
        (1, 4, "And God saw that the light was good, and He separated the light from the darkness."),
        (1, 5, "God called the light 'day,' and the darkness He called 'night.'"),
    ]
    for ch, vs, text in bsb_gen1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 1, ch, vs, text),
        )

    # Populate FTS for BSB Genesis
    for ch, vs, text in bsb_gen1:
        db.execute_update(
            "INSERT INTO verses_fts (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 1, ch, vs, text),
        )

    bsb_john1 = [
        (1, 1, "In the beginning was the Word, and the Word was with God, and the Word was God."),
        (1, 2, "He was with God in the beginning."),
        (1, 3, "Through Him all things were made; without Him nothing was made that has been made."),
    ]
    for ch, vs, text in bsb_john1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 43, ch, vs, text),
        )
        db.execute_update(
            "INSERT INTO verses_fts (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engbsb", 43, ch, vs, text),
        )

    # --- Verses (KJV) for comparison tests ---
    kjv_gen1 = [
        (1, 1, "In the beginning God created the heaven and the earth."),
        (1, 2, "And the earth was without form, and void; and darkness was upon the face of the deep."),
    ]
    for ch, vs, text in kjv_gen1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("engkjv", 1, ch, vs, text),
        )

    # --- Original Language Words (Genesis 1:1 — Hebrew) ---
    gen1_words = [
        (1, "Gen.1.1#01", "בְּרֵאשִׁ֖ית", None, "bereshit", "In beginning", "H9003/{H7225G}", "HR/Ncfsa", "H7225"),
        (2, "Gen.1.1#02", "בָּרָ֣א", None, "bara", "created", "{H1254A}", "HVqp3ms", "H1254"),
        (3, "Gen.1.1#03", "אֱלֹהִ֑ים", None, "elohim", "God", "{H0430}", "HNcmpa", "H0430"),
        (4, "Gen.1.1#04", "אֵ֥ת", None, "et", "[obj]", "{H0853}", "HTo", "H0853"),
        (5, "Gen.1.1#05", "הַשָּׁמַ֖יִם", None, "hashamayim", "the heavens", "H9009/{H8064}", "HTd/Ncmpa", "H8064"),
        (6, "Gen.1.1#06", "וְאֵ֥ת", None, "ve'et", "and", "H9002/{H0853}", "HC/To", "H0853"),
        (7, "Gen.1.1#07", "הָאָֽרֶץ", None, "ha'aretz", "the earth", "H9009/{H0776}", "HTd/Ncbsa", "H0776"),
    ]
    for wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary in gen1_words:
        db.execute_update(
            "INSERT OR REPLACE INTO words "
            "(book, chapter, verse, word_num, word_ref, hebrew_text, greek_text, "
            "transliteration, translation, strongs_raw, morphology_code, strongs_primary, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("Gen", 1, 1, wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary, "hebrew"),
        )

    # --- Original Language Words (John 1:1 — Greek) ---
    john1_words = [
        (1, "John.1.1#01", None, "Ἐν", "en", "In", "{G1722}", "GP", "G1722"),
        (2, "John.1.1#02", None, "ἀρχῇ", "arche", "beginning", "{G0746}", "GNdfs", "G0746"),
        (3, "John.1.1#03", None, "ἦν", "en", "was", "{G1510V}", "GVIia3s", "G1510"),
        (4, "John.1.1#04", None, "ὁ", "ho", "the", "{G3588}", "GEdnms", "G3588"),
        (5, "John.1.1#05", None, "Λόγος", "Logos", "Word", "{G3056}", "GNnms", "G3056"),
    ]
    for wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary in john1_words:
        db.execute_update(
            "INSERT OR REPLACE INTO words "
            "(book, chapter, verse, word_num, word_ref, hebrew_text, greek_text, "
            "transliteration, translation, strongs_raw, morphology_code, strongs_primary, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("John", 1, 1, wnum, wref, heb, grk, translit, trans, sraw, morph, sprimary, "greek"),
        )

    # --- Lexicon Entries ---
    lexicon_entries = [
        ("H7225", None, None, None, "רֵאשִׁית", "reshit", "noun", "beginning", "beginning, chief, first", "hebrew"),
        ("H1254", None, None, None, "בָּרָא", "bara", "verb", "create", "to create, shape, form", "hebrew"),
        ("H0430", None, None, None, "אֱלֹהִים", "elohim", "noun", "God", "God, gods, judges", "hebrew"),
        ("H0853", None, None, None, "אֵת", "et", "particle", "[obj]", "accusative marker", "hebrew"),
        ("H8064", None, None, None, "שָׁמַיִם", "shamayim", "noun", "heavens", "heaven, heavens, sky", "hebrew"),
        ("H0776", None, None, None, "אֶרֶץ", "erets", "noun", "earth", "earth, land, ground", "hebrew"),
        ("G3056", None, None, None, "λόγος", "logos", "noun", "word", "word, reason, statement", "greek"),
        ("G0746", None, None, None, "ἀρχή", "arche", "noun", "beginning", "beginning, origin, first cause", "greek"),
        ("G1722", None, None, None, "ἐν", "en", "preposition", "in", "in, on, among", "greek"),
        ("G1510", None, None, None, "εἰμί", "eimi", "verb", "am", "I am, I exist", "greek"),
        ("G3588", None, None, None, "ὁ", "ho", "article", "the", "the definite article", "greek"),
    ]
    for strongs, ext, disamb, uni, orig, translit, pos, gloss, defn, lang in lexicon_entries:
        db.execute_update(
            "INSERT OR REPLACE INTO lexicon "
            "(strongs_number, extended_strongs, disambiguated_strongs, unified_strongs, "
            "original_word, transliteration, part_of_speech, gloss, definition, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (strongs, ext, disamb, uni, orig, translit, pos, gloss, defn, lang),
        )

    # --- Morphology Codes ---
    morphology_entries = [
        ("HR/Ncfsa", "Hebrew Preposition + Noun common feminine singular absolute", None, "hebrew"),
        ("HVqp3ms", "Hebrew Verb qal perfect 3rd masculine singular", None, "hebrew"),
        ("HNcmpa", "Hebrew Noun common masculine plural absolute", None, "hebrew"),
        ("HTo", "Hebrew Particle accusative marker", None, "hebrew"),
        ("HTd/Ncmpa", "Hebrew Article + Noun common masculine plural absolute", None, "hebrew"),
        ("HC/To", "Hebrew Conjunction + Particle accusative marker", None, "hebrew"),
        ("HTd/Ncbsa", "Hebrew Article + Noun common both-gender singular absolute", None, "hebrew"),
        ("GP", "Greek Preposition", None, "greek"),
        ("GNdfs", "Greek Noun dative feminine singular", None, "greek"),
        ("GVIia3s", "Greek Verb Indicative imperfect active 3rd singular", None, "greek"),
        ("GEdnms", "Greek Article definite nominative masculine singular", None, "greek"),
        ("GNnms", "Greek Noun nominative masculine singular", None, "greek"),
    ]
    for code, desc, components, lang in morphology_entries:
        db.execute_update(
            "INSERT OR REPLACE INTO morphology (code, description, components, language) VALUES (?, ?, ?, ?)",
            (code, desc, components, lang),
        )

    # --- Verses mirrored under the real default translation id "BSB" ---
    # The app defaults to translation_id "BSB" (DEFAULT_TRANSLATION_ID); mirror the BSB text + FTS
    # so default-translation endpoints (mobile sync, text search, /books) work in tests.
    for ch, vs, text in bsb_gen1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("BSB", 1, ch, vs, text),
        )
        db.execute_update(
            "INSERT INTO verses_fts (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("BSB", 1, ch, vs, text),
        )
    for ch, vs, text in bsb_john1:
        db.execute_update(
            "INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("BSB", 43, ch, vs, text),
        )
        db.execute_update(
            "INSERT INTO verses_fts (translation_id, book_id, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
            ("BSB", 43, ch, vs, text),
        )

    # --- Original-language words in stepbible_verses (the current word source for the API) ---
    # STEP uses 3-letter book codes (Gen, Jhn). get_words_for_verse maps numeric book_id -> code.
    gen_step_words = [
        (1, "בְּרֵאשִׁ֖ית", "bereshit", "In beginning", "H9003/{H7225G}", "H9003", "HR/Ncfsa"),
        (2, "בָּרָ֣א", "bara", "created", "{H1254A}", "", "HVqp3ms"),
        (3, "אֱלֹהִ֑ים", "elohim", "God", "{H0430}", "", "HNcmpa"),
        (4, "אֵ֥ת", "et", "[obj]", "{H0853}", "", "HTo"),
        (5, "הַשָּׁמַ֖יִם", "hashamayim", "the heavens", "H9009/{H8064}", "H9009", "HTd/Ncmpa"),
        (6, "וְאֵ֥ת", "ve'et", "and", "H9002/{H0853}", "H9002", "HC/To"),
        (7, "הָאָֽרֶץ", "ha'aretz", "the earth", "H9009/{H0776}", "H9009", "HTd/Ncbsa"),
    ]
    for wnum, orig, translit, eng, sraw, sprimary, morph in gen_step_words:
        db.execute_update(
            "INSERT OR REPLACE INTO stepbible_verses "
            "(source_file, book, chapter, verse, word_number, original_word, transliteration, english, "
            "strongs_raw, strongs_primary, morphology, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test", "Gen", 1, 1, wnum, orig, translit, eng, sraw, sprimary, morph, "hebrew"),
        )
    john_step_words = [
        (1, "Ἐν", "en", "In", "{G1722}", "G1722", "GP"),
        (2, "ἀρχῇ", "arche", "beginning", "{G0746}", "G0746", "GNdfs"),
        (3, "ἦν", "en", "was", "{G1510V}", "G1510", "GVIia3s"),
        (4, "ὁ", "ho", "the", "{G3588}", "G3588", "GEdnms"),
        (5, "Λόγος", "Logos", "Word", "{G3056}", "G3056", "GNnms"),
    ]
    for wnum, orig, translit, eng, sraw, sprimary, morph in john_step_words:
        db.execute_update(
            "INSERT OR REPLACE INTO stepbible_verses "
            "(source_file, book, chapter, verse, word_number, original_word, transliteration, english, "
            "strongs_raw, strongs_primary, morphology, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test", "Jhn", 1, 1, wnum, orig, translit, eng, sraw, sprimary, morph, "greek"),
        )

    # verses_fts is an external-content FTS5 table; direct row inserts don't build a searchable
    # index. Rebuild it from the verses content so MATCH works for every seeded translation.
    from abba.database.search_index import rebuild_search_index  # noqa: PLC0415

    rebuild_search_index(db_path)

    yield db_path

    # Cleanup handled by tmp_path fixture


@pytest.fixture
def seeded_db_manager(seeded_db):
    """Return a SQLiteManager connected to the seeded test database."""
    return SQLiteManager(seeded_db)
