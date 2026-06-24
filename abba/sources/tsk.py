"""Treasury of Scripture Knowledge (TSK) cross-reference importer.

Reads the CrossWire SWORD TSK module (public domain, ~1880) via pysword
and extracts verse-level cross-references with anchor phrases.

Source: bible_data/sources/TSK.zip
License: Public Domain (DistributionLicense=Public Domain in SWORD conf)
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SWORD abbreviation → canonical book_id (1-based, Protestant 66-book order)
# Fallback map covering all TSK abbreviations. Authoritative values are
# derived at runtime by scanning passage= attributes in the module itself.
# ---------------------------------------------------------------------------
_FALLBACK_ABBREV_MAP: dict[str, int] = {
    # Old Testament
    "Ge": 1,
    "Ex": 2,
    "Le": 3,
    "Nu": 4,
    "De": 5,
    "Jos": 6,
    "Jdg": 7,
    "Ru": 8,
    "1Sa": 9,
    "2Sa": 10,
    "1Ki": 11,
    "2Ki": 12,
    "1Ch": 13,
    "2Ch": 14,
    "Ezr": 15,
    "Ne": 16,
    "Es": 17,
    "Job": 18,
    "Ps": 19,
    "Pr": 20,
    "Ec": 21,
    "So": 22,
    "Isa": 23,
    "Jer": 24,
    "La": 25,
    "Eze": 26,
    "Da": 27,
    "Ho": 28,
    "Joe": 29,
    "Am": 30,
    "Ob": 31,
    "Jon": 32,
    "Mic": 33,
    "Na": 34,
    "Hab": 35,
    "Zep": 36,
    "Hag": 37,
    "Zec": 38,
    "Mal": 39,
    # New Testament
    "Mt": 40,
    "Mr": 41,
    "Lu": 42,
    "Joh": 43,
    "Ac": 44,
    "Ro": 45,
    "1Co": 46,
    "2Co": 47,
    "Ga": 48,
    "Eph": 49,
    "Php": 50,
    "Col": 51,
    "1Th": 52,
    "2Th": 53,
    "1Ti": 54,
    "2Ti": 55,
    "Tit": 56,
    "Phm": 57,
    "Heb": 58,
    "Jas": 59,
    "1Pe": 60,
    "2Pe": 61,
    "1Jo": 62,
    "2Jo": 63,
    "3Jo": 64,
    "Jude": 65,
    "Re": 66,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_tsk_bible(zip_path: str | Path) -> Any:
    """Open the TSK SWORD module as a ZText-compatible object.

    Args:
        zip_path: Path to TSK.zip (CrossWire SWORD zCom module).

    Returns:
        A ZTextModule instance ready for ``get()`` calls.
    """
    from pysword.bible import SwordBible, SwordModuleType  # type: ignore[import-untyped]
    from pysword.modules import SwordModules  # type: ignore[import-untyped]

    m = SwordModules(str(zip_path))
    conf = m.parse_modules()["TSK"]
    module_path = os.path.join(m._module_paths["TSK"], conf["datapath"])
    ztext_cls = SwordBible._MODULE_CLASSES[SwordModuleType.ZTEXT]  # type: ignore[attr-defined]
    return ztext_cls(
        module_path,
        SwordModuleType.ZTEXT,
        conf.get("versification", "kjv").lower(),
        (conf.get("encoding") or "utf-8").lower(),
        conf.get("sourcetype"),
        conf.get("blocktype", "BOOK"),
    )


def open_tsk_bible(zip_path: str | Path) -> Any:
    """Open the TSK SWORD module as a ZText-compatible object.

    Public alias for ``_open_tsk_bible``, retained for callers that need
    direct access to the module object.

    Args:
        zip_path: Path to TSK.zip (CrossWire SWORD zCom module).

    Returns:
        A ZTextModule instance ready for ``get()`` calls.
    """
    return _open_tsk_bible(zip_path)


def _get_kjv_books() -> list[tuple[str, list[int]]]:
    """Return [(pysword_book_name, [verse_counts_per_chapter, ...]), ...] for KJV 66-book canon."""
    from pysword.canons import canons  # type: ignore[import-untyped]

    kjv = canons["kjv"]
    result: list[tuple[str, list[int]]] = []
    for testament_key in ("ot", "nt"):
        for book_tuple in kjv[testament_key]:
            # book_tuple: (full_name, osis_short, osis_short, [verse_counts])
            result.append((book_tuple[0], book_tuple[3]))
    return result


# ---------------------------------------------------------------------------
# Abbreviation map builder
# ---------------------------------------------------------------------------

_PASSAGE_PATTERN = re.compile(r'passage="([A-Za-z0-9]+)\s+\d+:\d+"')


def build_abbrev_map(bible: Any) -> dict[str, int]:
    """Derive TSK abbreviation → book_id from the module's passage= attributes.

    Scans verse 1 of chapter 1 for every book (where chapter summaries appear),
    extracting the book abbreviation used in ``passage="ABBR C:V"`` attributes.
    Fills gaps from the hardcoded fallback table.

    Args:
        bible: ZTextModule returned by ``open_tsk_bible()``.

    Returns:
        Dict mapping TSK abbreviation strings (e.g. "Ge", "Mt") to 1-based
        canonical book_id (1 = Genesis … 66 = Revelation).
    """
    abbrev_map: dict[str, int] = {}
    kjv_books = _get_kjv_books()

    for book_id, (book_name, _verse_counts) in enumerate(kjv_books, start=1):
        try:
            raw = bible.get(books=[book_name], chapters=[1], verses=[1], clean=False)
        except Exception:
            logger.debug("Could not fetch %s for abbreviation scan", book_name)
            continue

        if not raw:
            continue

        for abbr in _PASSAGE_PATTERN.findall(raw):
            if abbr not in abbrev_map:
                abbrev_map[abbr] = book_id
                logger.debug("Discovered abbrev %s -> book_id %d (%s)", abbr, book_id, book_name)
                break  # first match is sufficient

    # Fill gaps from fallback
    gaps = 0
    for abbr, book_id in _FALLBACK_ABBREV_MAP.items():
        if abbr not in abbrev_map:
            abbrev_map[abbr] = book_id
            gaps += 1

    logger.debug("build_abbrev_map: %d discovered, %d filled from fallback", len(abbrev_map) - gaps, gaps)
    return abbrev_map


# ---------------------------------------------------------------------------
# ThML parser
# ---------------------------------------------------------------------------

# Matches a single <scripRef> tag (with or without passage=)
_SCRIPREF_PATTERN = re.compile(
    r"<scripRef(?P<attrs>[^>]*)>(?P<content>[^<]*)</scripRef>",
    re.IGNORECASE,
)
_PASSAGE_ATTR = re.compile(r'\bpassage\s*=\s*"[^"]*"', re.IGNORECASE)

# Separators used within a reference list
_SEMI = re.compile(r"\s*;\s*")
_COMMA = re.compile(r"\s*,\s*")

# Patterns for parsing individual references
# Matches:  [BookAbbrev ][ chapter :]verse[-endverse]
# Full ref with single/range verse, no comma: "Joh 1:1-3" or "Joh 1:1"
_FULL_REF = re.compile(r"^(?P<book>[A-Za-z0-9]+)\s+(?P<chapter>\d+):(?P<verses>\d+(?:-\d+)?)$")
# Full ref with book + chapter + comma-separated verse list: "Ps 33:6,9" or "Ps 33:6,9,10"
_FULL_REF_COMMA = re.compile(r"^(?P<book>[A-Za-z0-9]+)\s+(?P<chapter>\d+):(?P<verses>\d[\d,\s-]*)$")
_CHAP_VERSE = re.compile(r"^(?P<chapter>\d+):(?P<verses>\d+(?:-\d+)?)$")
_CHAP_VERSES_COMMA = re.compile(r"^(?P<chapter>\d+):(?P<verses>\d[\d,\s-]*)$")
_BARE_VERSE = re.compile(r"^(?P<verse>\d+)$")
_BARE_VERSE_RANGE = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)$")


def _expand_verse_range(start: int, end: int) -> list[int]:
    """Expand a verse range to a list of verse numbers.

    If end < start (malformed), treat as [start, end] sorted.
    Caps range to a maximum of 200 to guard against runaway data.
    """
    if end < start:
        return sorted({start, end})
    return list(range(start, min(end, start + 200) + 1))


def _expand_verse_part(verse_part: str, book_id: int, chapter: int) -> list[tuple[int, int, int]]:
    """Expand a single verse part (possibly a range) into (book, chapter, verse) tuples."""
    vp = verse_part.strip()
    if "-" in vp:
        rng = vp.split("-", 1)
        try:
            return [(book_id, chapter, v) for v in _expand_verse_range(int(rng[0]), int(rng[1]))]
        except ValueError:
            return []
    try:
        return [(book_id, chapter, int(vp))]
    except ValueError:
        return []


def _expand_verse_list(verses_part: str, book_id: int, chapter: int) -> list[tuple[int, int, int]]:
    """Expand a comma-separated verse list (e.g. '6,9' or '1-3,5') for a given book/chapter."""
    results: list[tuple[int, int, int]] = []
    for part in _COMMA.split(verses_part):
        results.extend(_expand_verse_part(part, book_id, chapter))
    return results


def _parse_verse_token(
    token: str, cur_book: int, cur_chapter: int, abbrev_map: dict[str, int]
) -> list[tuple[int, int, int]]:
    """Parse a single reference token (between semicolons) into (book, ch, verse) tuples.

    Returns:
        list of (book_id, chapter, verse) tuples
    """
    token = token.strip()
    if not token:
        return []

    # Try full ref with single/range verse: "Joh 1:1-3" or "Joh 1:1"
    m = _FULL_REF.match(token)
    if m:
        book_id = abbrev_map.get(m.group("book"))
        if book_id is None:
            logger.debug("Unknown book abbreviation: %r", m.group("book"))
            return []
        chapter = int(m.group("chapter"))
        verses_str = m.group("verses")
        if "-" in verses_str:
            parts = verses_str.split("-", 1)
            return [(book_id, chapter, v) for v in _expand_verse_range(int(parts[0]), int(parts[1]))]
        return [(book_id, chapter, int(verses_str))]

    # Try full ref with comma-separated verse list: "Ps 33:6,9" or "Ps 33:6,9,10"
    m = _FULL_REF_COMMA.match(token)
    if m:
        book_id = abbrev_map.get(m.group("book"))
        if book_id is None:
            logger.debug("Unknown book abbreviation: %r", m.group("book"))
            return []
        return _expand_verse_list(m.group("verses"), book_id, int(m.group("chapter")))

    # Try chapter:verses with possible comma list: "1:14,18" or "16:4"
    m = _CHAP_VERSES_COMMA.match(token)
    if m:
        return _expand_verse_list(m.group("verses"), cur_book, int(m.group("chapter")))

    # Try bare verse: "15"
    m = _BARE_VERSE.match(token)
    if m:
        return [(cur_book, cur_chapter, int(m.group("verse")))]

    # Try bare range: "22-24" (no chapter prefix — inherits current chapter)
    m = _BARE_VERSE_RANGE.match(token)
    if m:
        return [(cur_book, cur_chapter, v) for v in _expand_verse_range(int(m.group("start")), int(m.group("end")))]

    logger.debug("Could not parse reference token: %r", token)
    return []


def _get_context_from_token(token: str, cur_book: int, cur_chapter: int, abbrev_map: dict[str, int]) -> tuple[int, int]:
    """Return (new_book, new_chapter) context after parsing token."""
    token = token.strip()
    # Full ref with single/range verse: "Joh 1:1"
    m = _FULL_REF.match(token)
    if m:
        abbr = m.group("book")
        book_id = abbrev_map.get(abbr)
        if book_id is not None:
            return book_id, int(m.group("chapter"))
        return cur_book, cur_chapter

    # Full ref with comma verse list: "Ps 33:6,9"
    m = _FULL_REF_COMMA.match(token)
    if m:
        abbr = m.group("book")
        book_id = abbrev_map.get(abbr)
        if book_id is not None:
            return book_id, int(m.group("chapter"))
        return cur_book, cur_chapter

    m = _CHAP_VERSES_COMMA.match(token)
    if m:
        return cur_book, int(m.group("chapter"))

    return cur_book, cur_chapter


def parse_verse_refs(
    thml: str,
    source_book_id: int,
    source_chapter: int,
    abbrev_map: dict[str, int],
) -> list[tuple[str, list[tuple[int, int, int]]]]:
    """Parse a verse's ThML into (anchor_phrase, [(book, chapter, verse), ...]) groups.

    Chapter-summary scripRefs (those with a ``passage=`` attribute) are silently
    skipped — they are section headings, not cross-references.

    Args:
        thml: Raw ThML string from the SWORD module for a single verse.
        source_book_id: 1-based book_id of the source verse (for context init).
        source_chapter: Chapter number of the source verse (for context init).
        abbrev_map: Abbreviation map from ``build_abbrev_map()``.

    Returns:
        List of (anchor_phrase, targets) tuples where targets is a list of
        (book_id, chapter, verse) tuples. Returns empty list if no cross-
        references are found.
    """
    results: list[tuple[str, list[tuple[int, int, int]]]] = []

    # Split on <br /> to get segments; strip HTML tags other than scripRef
    # We'll walk segments looking for "anchor text" followed by <scripRef> tags.
    # Strategy: find all scripRef elements and their surrounding context.

    # Remove any passage= scripRef (chapter summaries) first, then re-scan.
    # We identify them by presence of passage= attribute.
    cleaned = re.sub(
        r'<scripRef[^>]*\bpassage\s*=\s*"[^"]*"[^>]*>[^<]*</scripRef>',
        "",
        thml,
        flags=re.IGNORECASE,
    )

    # Now split on <br /> to walk segments
    segments = re.split(r"<br\s*/?>", cleaned, flags=re.IGNORECASE)

    current_anchor: str | None = None

    for raw_seg in segments:
        seg = raw_seg.strip()
        if not seg:
            continue

        # Check if this segment contains a <scripRef> (cross-ref)
        scripref_match = _SCRIPREF_PATTERN.search(seg)
        if scripref_match:
            # Verify it has no passage= attribute (already cleaned, but double-check)
            attrs = scripref_match.group("attrs")
            if _PASSAGE_ATTR.search(attrs):
                current_anchor = None
                continue

            ref_text = scripref_match.group("content").strip()
            if not ref_text:
                continue

            anchor = current_anchor or ""
            targets: list[tuple[int, int, int]] = []

            cur_book = source_book_id
            cur_chapter = source_chapter

            for raw_part in _SEMI.split(ref_text):
                token = raw_part.strip()
                if not token:
                    continue
                refs = _parse_verse_token(token, cur_book, cur_chapter, abbrev_map)
                targets.extend(refs)
                # Update context
                cur_book, cur_chapter = _get_context_from_token(token, cur_book, cur_chapter, abbrev_map)

            if targets:
                results.append((anchor, targets))
            current_anchor = None

        else:
            # Plain text segment — potential anchor phrase
            # Strip any residual HTML tags
            plain = re.sub(r"<[^>]+>", "", seg).strip()
            # Strip trailing punctuation to get clean anchor phrase
            anchor_clean = plain.rstrip(".,;:!?").strip()
            if anchor_clean:
                current_anchor = anchor_clean

    return results


# ---------------------------------------------------------------------------
# Iterator over the entire canon
# ---------------------------------------------------------------------------


def iter_tsk_cross_references(zip_path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate all cross-references in the TSK module for the full KJV 66-book canon.

    Each yielded dict has keys:
        source_book_id (int), source_chapter (int), source_verse (int),
        target_book_id (int), target_chapter (int), target_verse (int),
        anchor_phrase (str)

    Skips any references that cannot be resolved (logs a debug count summary
    at end).

    Args:
        zip_path: Path to TSK.zip.

    Yields:
        Cross-reference record dicts.
    """
    bible = _open_tsk_bible(zip_path)
    abbrev_map = build_abbrev_map(bible)
    kjv_books = _get_kjv_books()

    unresolved_count = 0
    total_refs = 0

    for book_id, (book_name, verse_counts_per_chapter) in enumerate(kjv_books, start=1):
        for chapter_idx, verse_count in enumerate(verse_counts_per_chapter, start=1):
            for verse_num in range(1, verse_count + 1):
                try:
                    raw = bible.get(
                        books=[book_name],
                        chapters=[chapter_idx],
                        verses=[verse_num],
                        clean=False,
                    )
                except Exception as e:
                    logger.debug("Error fetching %s %d:%d: %s", book_name, chapter_idx, verse_num, e)
                    continue

                if not raw:
                    continue

                groups = parse_verse_refs(raw, book_id, chapter_idx, abbrev_map)
                for anchor_phrase, targets in groups:
                    for target_book_id, target_chapter, target_verse in targets:
                        if target_book_id < 1 or target_book_id > 66:
                            unresolved_count += 1
                            continue
                        total_refs += 1
                        yield {
                            "source_book_id": book_id,
                            "source_chapter": chapter_idx,
                            "source_verse": verse_num,
                            "target_book_id": target_book_id,
                            "target_chapter": target_chapter,
                            "target_verse": target_verse,
                            "anchor_phrase": anchor_phrase,
                        }

    logger.info(
        "TSK iteration complete: %d cross-references yielded, %d unresolved skipped",
        total_refs,
        unresolved_count,
    )
