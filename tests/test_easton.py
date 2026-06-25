"""Tests for Easton's Bible Dictionary parser + importer.

Fully self-contained: a tiny synthetic SWORD ``zLD`` module is built in
``tmp_path`` (no network, no live DB, no Ollama). A guarded test exercises the
real ``bible_data/sources/Easton.zip`` only when it happens to be present.

Covers:
- the zLD block decoder + TEI article extraction (``abba.sources.easton``)
- headword normalization + idempotent import (``abba.database.easton_importer``)
- Tier-A provenance recording
"""

from __future__ import annotations

import sqlite3
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from abba.database.easton_importer import (
    SOURCE_LICENSE,
    SOURCE_NAME,
    import_easton_entries,
    normalize_headword,
)
from abba.sources.easton import (
    DictionaryEntry,
    iter_easton_entries,
    parse_entry,
)

# ---------------------------------------------------------------------------
# Synthetic zLD fixture builder
# ---------------------------------------------------------------------------

_MODULE_PREFIX = "modules/lexdict/zld/easton/easton"

# (headword, raw TEI article markup) — payloads encoded into the synthetic module.
_FIXTURE_ENTRIES: list[tuple[str, str]] = [
    (
        "Aaron",
        '<entryFree n="Aaron">\n<title>Aaron</title>\n<p>The eldest son of Amram and Jochebed '
        '(<ref osisRef="Bible:Exod.6.20">Ex. 6:20</ref>). He was the first high priest '
        '(<ref osisRef="Bible:Num.3.32">Num. 3:32</ref>).</p>',
    ),
    (
        "Bethlehem",
        '<entryFree n="Bethlehem">\n<title>Bethlehem</title>\n<p>House of bread, a city in Judah '
        '(<ref osisRef="Bible:Ruth.1.1">Ruth 1:1</ref>), the birthplace of David.</p>',
    ),
    (
        # Comma-style headword to exercise normalization; no osisRef.
        "Zuph, Land of",
        '<entryFree n="Zuph, Land of">\n<title>Zuph, Land of</title>\n<p>A district visited by Saul.</p>',
    ),
]


def _build_block(payloads: list[bytes]) -> bytes:
    """Build one decompressed zLD block: count + (off,size) header + payloads."""
    count = len(payloads)
    header_size = 4 + count * 8
    body = b""
    offsets: list[tuple[int, int]] = []
    for payload in payloads:
        offsets.append((header_size + len(body), len(payload)))
        body += payload
    header = struct.pack("<I", count)
    for off, size in offsets:
        header += struct.pack("<II", off, size)
    return header + body


def _write_synthetic_easton(zip_path: Path, entries: list[tuple[str, str]]) -> None:
    """Write a minimal but format-correct Easton zLD module zip.

    Splits the entries across two compressed blocks to exercise multi-block
    iteration. Produces ``.idx``/``.dat``/``.zdx``/``.zdt`` plus a conf file.
    """
    payloads = [markup.encode("utf-8") for _, markup in entries]

    # Two blocks: first holds entry 0, second holds the rest (covers multi-block).
    split = 1 if len(payloads) > 1 else len(payloads)
    block_payload_groups = [payloads[:split], payloads[split:]] if payloads[split:] else [payloads[:split]]

    zdt = b""
    zdx = b""
    for group in block_payload_groups:
        if not group:
            continue
        compressed = zlib.compress(_build_block(group))
        zdx += struct.pack("<II", len(zdt), len(compressed))
        zdt += compressed

    # .dat holds key records: KEY \r\n \0 <running uint32 index>; .idx points into it.
    dat = b""
    idx = b""
    for i, (headword, _markup) in enumerate(entries):
        rec = headword.encode("utf-8") + b"\r\n\x00" + struct.pack("<I", i)
        idx += struct.pack("<II", len(dat), len(rec))
        dat += rec

    conf = (
        "[Easton]\n"
        "Description=Easton's Bible Dictionary\n"
        f"DataPath=./{_MODULE_PREFIX}\n"
        "ModDrv=zLD\n"
        "SourceType=TEI\n"
        "DistributionLicense=Public Domain\n"
        "TextSource=CCEL\n"
    )

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{_MODULE_PREFIX}.idx", idx)
        zf.writestr(f"{_MODULE_PREFIX}.dat", dat)
        zf.writestr(f"{_MODULE_PREFIX}.zdx", zdx)
        zf.writestr(f"{_MODULE_PREFIX}.zdt", zdt)
        zf.writestr("mods.d/easton.conf", conf)


@pytest.fixture
def synthetic_easton_zip(tmp_path: Path) -> Path:
    """Path to a freshly built synthetic Easton zLD module in tmp_path."""
    zip_path = tmp_path / "Easton.zip"
    _write_synthetic_easton(zip_path, _FIXTURE_ENTRIES)
    return zip_path


# ---------------------------------------------------------------------------
# Parser unit tests (synthetic fixture)
# ---------------------------------------------------------------------------


class TestParser:
    def test_iter_returns_all_entries(self, synthetic_easton_zip: Path) -> None:
        entries = list(iter_easton_entries(synthetic_easton_zip))
        assert len(entries) == 3
        headwords = [e.headword for e in entries]
        assert headwords == ["Aaron", "Bethlehem", "Zuph, Land of"]

    def test_article_is_plain_text(self, synthetic_easton_zip: Path) -> None:
        entries = {e.headword: e for e in iter_easton_entries(synthetic_easton_zip)}
        aaron = entries["Aaron"]
        # Markup stripped; title + body present; no angle brackets remain.
        assert "<" not in aaron.article and ">" not in aaron.article
        assert "eldest son of Amram" in aaron.article
        assert "Ex. 6:20" in aaron.article  # ref display text preserved

    def test_ref_targets_extracted(self, synthetic_easton_zip: Path) -> None:
        entries = {e.headword: e for e in iter_easton_entries(synthetic_easton_zip)}
        assert entries["Aaron"].ref_targets == ("Exod.6.20", "Num.3.32")
        assert entries["Bethlehem"].ref_targets == ("Ruth.1.1",)
        assert entries["Zuph, Land of"].ref_targets == ()

    def test_parse_entry_missing_headword_returns_none(self) -> None:
        assert parse_entry(b"<p>No headword here.</p>") is None

    def test_parse_entry_empty_payload_returns_none(self) -> None:
        assert parse_entry(b"") is None
        assert parse_entry(b"   \n  ") is None

    def test_parse_entry_title_fallback(self) -> None:
        # No entryFree attr; headword falls back to <title>.
        entry = parse_entry(b"<title>Fallback</title>\n<p>Body text.</p>")
        assert entry is not None
        assert entry.headword == "Fallback"
        assert "Body text." in entry.article

    def test_cp1252_decoding_does_not_crash(self) -> None:
        # 0x93/0x94 are CP1252 curly quotes, invalid as UTF-8 — must not raise.
        payload = b'<entryFree n="Test">\n<p>He said \x93hello\x94 to all.</p>'
        entry = parse_entry(payload)
        assert entry is not None
        assert entry.headword == "Test"
        assert "hello" in entry.article

    def test_missing_zip_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list(iter_easton_entries(tmp_path / "nope.zip"))

    def test_missing_member_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(str(bad), "w") as zf:
            zf.writestr("mods.d/easton.conf", "[Easton]\n")
        with pytest.raises(FileNotFoundError):
            list(iter_easton_entries(bad))


# ---------------------------------------------------------------------------
# Headword normalization
# ---------------------------------------------------------------------------


class TestNormalizeHeadword:
    def test_basic_uppercase(self) -> None:
        assert normalize_headword("Aaron") == "AARON"

    def test_comma_and_spaces(self) -> None:
        assert normalize_headword("Zuph, Land of") == "ZUPH LAND OF"

    def test_parenthetical_removed(self) -> None:
        assert normalize_headword("Mary (mother of Jesus)") == "MARY"

    def test_apostrophe_and_hyphen(self) -> None:
        assert normalize_headword("Beth-el's") == "BETH EL S"

    def test_non_alpha_only_is_empty(self) -> None:
        assert normalize_headword("(...)") == ""


# ---------------------------------------------------------------------------
# Importer tests (temp DB only)
# ---------------------------------------------------------------------------


class TestImporter:
    def test_import_populates_and_is_idempotent(self, tmp_path: Path, synthetic_easton_zip: Path) -> None:
        db_path = tmp_path / "test.db"

        count1 = import_easton_entries(db_path, synthetic_easton_zip)
        assert count1 == 3

        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM dictionary_entries").fetchone()
            assert rows[0] == 3
            sources = {r[0] for r in conn.execute("SELECT DISTINCT source FROM dictionary_entries")}
            assert sources == {SOURCE_NAME}
            licenses = {r[0] for r in conn.execute("SELECT DISTINCT license FROM dictionary_entries")}
            assert licenses == {SOURCE_LICENSE}

        # Second run is idempotent.
        count2 = import_easton_entries(db_path, synthetic_easton_zip)
        assert count2 == 0
        with sqlite3.connect(str(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM dictionary_entries").fetchone()[0] == 3

    def test_normalized_headword_stored(self, tmp_path: Path, synthetic_easton_zip: Path) -> None:
        db_path = tmp_path / "test.db"
        import_easton_entries(db_path, synthetic_easton_zip)
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT headword FROM dictionary_entries WHERE headword_normalized = ?",
                ("ZUPH LAND OF",),
            ).fetchone()
            assert row is not None
            assert row[0] == "Zuph, Land of"

    def test_ref_targets_stored_as_json(self, tmp_path: Path, synthetic_easton_zip: Path) -> None:
        db_path = tmp_path / "test.db"
        import_easton_entries(db_path, synthetic_easton_zip)
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT ref_targets FROM dictionary_entries WHERE headword = ?", ("Aaron",)).fetchone()
            assert row is not None
            assert "Exod.6.20" in row[0]
            null_row = conn.execute(
                "SELECT ref_targets FROM dictionary_entries WHERE headword = ?", ("Zuph, Land of",)
            ).fetchone()
            assert null_row[0] is None

    def test_provenance_recorded_when_table_present(self, tmp_path: Path, synthetic_easton_zip: Path) -> None:
        db_path = tmp_path / "test.db"
        # Pre-create the provenance table the way the live schema does.
        from abba.database.migrations import add_provenance_table

        # add_provenance_table needs the db to exist; create an empty db first.
        sqlite3.connect(str(db_path)).close()
        add_provenance_table(db_path)

        import_easton_entries(db_path, synthetic_easton_zip)

        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT trust_tier, confidence, generated_by FROM provenance WHERE entity_type = 'dictionary_entry'"
            ).fetchall()
            assert len(rows) == 3
            for trust_tier, confidence, generated_by in rows:
                assert trust_tier == "A"
                assert confidence is None
                assert generated_by is None

    def test_import_without_provenance_table_succeeds(self, tmp_path: Path, synthetic_easton_zip: Path) -> None:
        # No provenance table created — importer must still populate entries.
        db_path = tmp_path / "test.db"
        count = import_easton_entries(db_path, synthetic_easton_zip)
        assert count == 3
        with sqlite3.connect(str(db_path)) as conn:
            tbls = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "provenance" not in tbls
            assert "dictionary_entries" in tbls


# ---------------------------------------------------------------------------
# Guarded test against the real module (file parsing only; no network/LLM)
# ---------------------------------------------------------------------------

_REAL_EASTON = Path("bible_data/sources/Easton.zip")


@pytest.mark.skipif(not _REAL_EASTON.exists(), reason="bible_data/sources/Easton.zip not present")
class TestRealModule:
    def test_yields_full_dictionary(self) -> None:
        entries = list(iter_easton_entries(_REAL_EASTON))
        # Easton ships ~3,963 entries; conservative lower bound.
        assert len(entries) > 3_500
        headwords = {e.headword for e in entries}
        assert "Aaron" in headwords
        assert any(h.startswith("Bethlehem") for h in headwords)

    def test_entries_are_well_formed(self) -> None:
        for entry in iter_easton_entries(_REAL_EASTON):
            assert isinstance(entry, DictionaryEntry)
            assert entry.headword
            assert entry.article
            assert "<entryFree" not in entry.article
