"""Tests for DEFAULT_TRANSLATION_ID constant."""


def test_default_translation_id_value():
    from abba.api.constants import DEFAULT_TRANSLATION_ID

    assert DEFAULT_TRANSLATION_ID == "BSB"


def test_default_translation_importable():
    from abba.api.constants import DEFAULT_TRANSLATION_ID

    assert isinstance(DEFAULT_TRANSLATION_ID, str)
