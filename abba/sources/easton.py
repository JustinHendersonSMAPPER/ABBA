"""Easton's Bible Dictionary (1897) dictionary-entry parser.

Reads the CrossWire SWORD "Easton" dictionary module (public domain, 1897)
and extracts every entry as a ``(headword, article)`` record for later
entity-linking to verses (decision D5).

Source: bible_data/sources/Easton.zip
    Re-fetch from:
        https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/Easton.zip
License: Public Domain
    The module's ``mods.d/easton.conf`` declares ``DistributionLicense=Public
    Domain`` and ``TextSource=CCEL``; the About field reads "Public Domain --
    Copy Freely ... M.G. Easton M.A., D.D., Illustrated Bible Dictionary, Third
    Edition, published by Thomas Nelson, 1897." Easton died in 1894, so the work
    is public domain worldwide.

Why a bespoke parser (not pysword like TSK):
    Easton ships as a SWORD ``zLD`` (compressed lexicon/dictionary) module.
    pysword only implements Bible/commentary drivers (zText/zCom), not the
    dictionary driver, so the zLD container is decoded here directly. The format
    is small and stable; this module reverse-implements the documented zLD
    layout and is validated against the real 3,963-entry module.

zLD on-disk layout (all integers little-endian uint32):
    ``<mod>.idx``  : N records of 8 bytes ``(dat_offset, dat_size)`` — one per
                     entry, in headword (sorted) order. Locates the *key* record.
    ``<mod>.dat``  : per-entry key records: ``KEY`` then ``\\r\\n`` then a NUL,
                     then a 4-byte running entry index.
    ``<mod>.zdx``  : M records of 8 bytes ``(zdt_offset, zdt_size)`` — one per
                     compressed block.
    ``<mod>.zdt``  : M zlib-compressed blocks. Each *decompressed* block begins
                     with a uint32 sub-entry count C, followed by C pairs of
                     ``(in_block_offset, in_block_size)``, then the article
                     payloads. Sub-entries are stored in global entry order, so
                     concatenating blocks in order yields all entries in order.

Article markup is TEI: ``<entryFree n="Headword"><title>..</title><p>..</p>``
with ``<ref osisRef="Bible:Book.C.V">`` cross-references inside.
"""

from __future__ import annotations

import logging
import re
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical SWORD paths inside the rawzip distribution.
_MODULE_DATA_PREFIX = "modules/lexdict/zld/easton/easton"
_CONF_PATH = "mods.d/easton.conf"

# TEI extraction patterns.
_ENTRY_HEADWORD = re.compile(r'<entryFree\b[^>]*\bn="([^"]*)"', re.IGNORECASE)
_TITLE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class DictionaryEntry:
    """A single dictionary entry: a headword and its plain-text article.

    Attributes:
        headword: The entry's lookup term (e.g. ``"Aaron"``, ``"Bethlehem"``).
        article: The article body as plain text, TEI markup stripped, with
            inner whitespace normalized. Cross-reference targets, if any, are
            preserved in ``ref_targets``.
        ref_targets: OSIS-style verse references cited by the article
            (e.g. ``["Exod.6.20", "Num.3.32"]``), in document order, with
            duplicates removed. Empty if the article cites no verses.
    """

    headword: str
    article: str
    ref_targets: tuple[str, ...] = ()


def _read_module_bytes(zip_path: str | Path) -> dict[str, bytes]:
    """Read the four zLD data members from the Easton SWORD zip.

    Args:
        zip_path: Path to Easton.zip (CrossWire SWORD rawzip distribution).

    Returns:
        Dict with keys ``idx``, ``dat``, ``zdx``, ``zdt`` mapping to raw bytes.

    Raises:
        FileNotFoundError: If the zip or any required member is missing.
    """
    import zipfile  # noqa: PLC0415 - stdlib, only needed at ingest time

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Easton SWORD module not found: {zip_path}")

    members = {
        "idx": f"{_MODULE_DATA_PREFIX}.idx",
        "dat": f"{_MODULE_DATA_PREFIX}.dat",
        "zdx": f"{_MODULE_DATA_PREFIX}.zdx",
        "zdt": f"{_MODULE_DATA_PREFIX}.zdt",
    }
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(str(zip_path)) as zf:
        names = set(zf.namelist())
        for key, member in members.items():
            if member not in names:
                raise FileNotFoundError(f"Easton module is missing required member: {member}")
            out[key] = zf.read(member)
    return out


def _iter_block_payloads(zdx: bytes, zdt: bytes) -> Iterator[bytes]:
    """Yield each entry's raw article bytes, in global entry order.

    Walks every compressed block (via ``zdx``), decompresses it, reads the
    block's sub-entry index header, and yields each sub-entry's payload slice.

    Args:
        zdx: Raw ``<mod>.zdx`` bytes (8-byte ``(offset, size)`` block records).
        zdt: Raw ``<mod>.zdt`` bytes (zlib-compressed blocks).

    Yields:
        Raw bytes of one entry's article payload.
    """
    block_count = len(zdx) // 8
    for block_i in range(block_count):
        z_off, z_size = struct.unpack_from("<II", zdx, block_i * 8)
        if z_size == 0:
            continue
        try:
            raw = zlib.decompress(zdt[z_off : z_off + z_size])
        except zlib.error as exc:  # pragma: no cover - corrupt module guard
            logger.warning("Skipping undecompressable Easton block %d: %s", block_i, exc)
            continue

        if len(raw) < 4:
            continue
        sub_count = struct.unpack_from("<I", raw, 0)[0]
        header_end = 4 + sub_count * 8
        if header_end > len(raw):
            logger.warning("Easton block %d header overruns payload; skipping", block_i)
            continue

        for sub_i in range(sub_count):
            in_off, in_size = struct.unpack_from("<II", raw, 4 + sub_i * 8)
            if in_size == 0:
                continue
            yield raw[in_off : in_off + in_size]


def _decode_payload(payload: bytes) -> str:
    """Decode a raw article payload to text.

    The module declares UTF-8 but the CCEL source contains stray CP1252 bytes
    (curly quotes around quoted nicknames). We try UTF-8 strict first, then fall
    back to CP1252, then to a lossy UTF-8 decode so no entry is ever dropped.

    Args:
        payload: Raw article bytes from a zLD block.

    Returns:
        The decoded article string (still TEI-marked-up).
    """
    for encoding in ("utf-8", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _extract_ref_targets(article_markup: str) -> tuple[str, ...]:
    """Extract OSIS verse references from TEI ``<ref osisRef="Bible:...">`` tags.

    Args:
        article_markup: The raw TEI article string.

    Returns:
        Tuple of OSIS reference strings (the part after ``Bible:``), in document
        order with duplicates removed. E.g. ``("Exod.6.20", "Num.3.32")``.
    """
    seen: dict[str, None] = {}
    for match in re.finditer(r'osisRef="(?:Bible:)?([^"]+)"', article_markup, flags=re.IGNORECASE):
        ref = match.group(1).strip()
        if ref:
            seen.setdefault(ref, None)
    return tuple(seen.keys())


def _strip_markup(article_markup: str) -> str:
    """Strip TEI markup to plain text and normalize whitespace.

    Paragraph boundaries (``</p>``) become blank lines; all other tags are
    removed. Runs of intra-line whitespace collapse to single spaces and runs of
    3+ newlines collapse to a paragraph break.

    Args:
        article_markup: The raw TEI article string.

    Returns:
        Clean plain-text article body.
    """
    text = re.sub(r"</p\s*>", "\n\n", article_markup, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _TAG.sub("", text)
    # HTML/XML entities used by the module are minimal; decode the common ones.
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    # Normalize whitespace line-by-line, preserving paragraph breaks.
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def parse_entry(payload: bytes) -> DictionaryEntry | None:
    """Parse one raw zLD payload into a :class:`DictionaryEntry`.

    Args:
        payload: Raw article bytes for a single entry.

    Returns:
        A populated :class:`DictionaryEntry`, or ``None`` if the payload has no
        usable headword or article text (such entries are skipped, not errors).
    """
    markup = _decode_payload(payload)
    if not markup.strip():
        return None

    headword_match = _ENTRY_HEADWORD.search(markup)
    if headword_match:
        headword = headword_match.group(1).strip()
    else:
        # Fallback: use the <title> if the entryFree attribute is absent.
        title_match = _TITLE.search(markup)
        headword = _strip_markup(title_match.group(1)).strip() if title_match else ""

    if not headword:
        return None

    ref_targets = _extract_ref_targets(markup)
    article = _strip_markup(markup)
    if not article:
        return None

    return DictionaryEntry(headword=headword, article=article, ref_targets=ref_targets)


def iter_easton_entries(zip_path: str | Path) -> Iterator[DictionaryEntry]:
    """Iterate every entry in the Easton SWORD dictionary module.

    Args:
        zip_path: Path to Easton.zip (CrossWire SWORD rawzip distribution).

    Yields:
        :class:`DictionaryEntry` records in headword (alphabetical) order.

    Raises:
        FileNotFoundError: If the zip or a required member is missing.
    """
    data = _read_module_bytes(zip_path)
    yielded = 0
    skipped = 0
    for payload in _iter_block_payloads(data["zdx"], data["zdt"]):
        entry = parse_entry(payload)
        if entry is None:
            skipped += 1
            continue
        yielded += 1
        yield entry

    logger.info("Easton iteration complete: %d entries yielded, %d skipped", yielded, skipped)
