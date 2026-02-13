"""Parsers for free/open-source biblical lexicon data.

Supports:
- OpenScriptures HebrewStrong.xml (Strong's Hebrew Dictionary, public domain, CC BY 4.0)
- Abbott-Smith Greek Lexicon TEI XML (public domain, 1922)
- Brown-Driver-Briggs Hebrew Lexicon XML (public domain 1906, CC BY 4.0 markup)
- Dodson Greek-English Lexicon CSV (public domain, CC0)

All sources are free to use without license restrictions.
"""

import csv
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from abba.logging_setup import get_logger

logger = get_logger(__name__)

# Namespaces
HEBREW_NS = {"lex": "http://openscriptures.github.com/morphhb/namespace"}
ABBOTT_NS = {"tei": "http://www.crosswire.org/2013/TEIOSIS/namespace"}

# Download URLs for free lexicon sources
LEXICON_URLS = {
    "hebrew_strongs.xml": "https://raw.githubusercontent.com/openscriptures/HebrewLexicon/master/HebrewStrong.xml",
    "abbott_smith.xml": (
        "https://raw.githubusercontent.com/translatable-exegetical-tools/Abbott-Smith/master/abbott-smith.tei.xml"
    ),
    "bdb.xml": "https://raw.githubusercontent.com/openscriptures/HebrewLexicon/master/BrownDriverBriggs.xml",
    "lexical_index.xml": "https://raw.githubusercontent.com/openscriptures/HebrewLexicon/master/LexicalIndex.xml",
    "dodson.csv": "https://raw.githubusercontent.com/biblicalhumanities/Dodson-Greek-Lexicon/master/dodson.csv",
}


def _get_element_text(element: Optional[ET.Element]) -> str:
    """Extract all text content from an XML element, including children."""
    if element is None:
        return ""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        child_text = _get_element_text(child)
        if child_text:
            parts.append(child_text)
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts).strip()


def _clean_text(text: str) -> str:
    """Clean up whitespace in extracted text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_hebrew_gloss_and_definition(entry_elem: ET.Element) -> tuple:
    """Extract gloss, meaning text, and usage text from a Hebrew entry element.

    Returns:
        Tuple of (gloss, definition) strings.
    """
    meaning_elem = entry_elem.find("lex:meaning", HEBREW_NS)
    meaning_text = _clean_text(_get_element_text(meaning_elem))

    gloss = ""
    if meaning_elem is not None:
        first_def = meaning_elem.find("lex:def", HEBREW_NS)
        if first_def is not None and first_def.text:
            gloss = first_def.text.strip()

    usage_elem = entry_elem.find("lex:usage", HEBREW_NS)
    usage_text = _clean_text(_get_element_text(usage_elem))

    definition_parts = []
    if meaning_text:
        definition_parts.append(meaning_text)
    if usage_text:
        definition_parts.append(f"Usage: {usage_text}")
    definition = "; ".join(definition_parts)

    if not gloss and usage_text:
        gloss = usage_text.split(",")[0].strip().rstrip(".")

    return gloss, definition


def _parse_hebrew_entry(entry_elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a single Hebrew lexicon entry element into a dict."""
    entry_id = entry_elem.get("id", "")
    if not entry_id:
        return None

    w_elem = entry_elem.find("lex:w", HEBREW_NS)
    if w_elem is None:
        return None

    original_word = w_elem.text or ""
    pos = w_elem.get("pos", "")
    pron = w_elem.get("pron", "")
    xlit = w_elem.get("xlit", "")

    gloss, definition = _extract_hebrew_gloss_and_definition(entry_elem)

    return {
        "strongs_number": entry_id,
        "original_word": original_word,
        "transliteration": xlit or pron,
        "part_of_speech": pos,
        "gloss": gloss,
        "definition": definition,
        "language": "hebrew",
    }


def parse_hebrew_strongs_xml(file_path: Path) -> List[Dict[str, Any]]:
    """Parse OpenScriptures HebrewStrong.xml into lexicon entries.

    Source: https://github.com/openscriptures/HebrewLexicon
    License: CC BY 4.0 (markup) / Public Domain (dictionary content)

    Args:
        file_path: Path to HebrewStrong.xml

    Returns:
        List of lexicon entry dicts ready for database insertion.
    """
    entries: List[Dict[str, Any]] = []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except (ET.ParseError, OSError) as e:
        logger.error(f"Failed to parse Hebrew lexicon XML: {e}")
        return entries

    for entry_elem in root.findall(".//lex:entry", HEBREW_NS):
        entry = _parse_hebrew_entry(entry_elem)
        if entry is not None:
            entries.append(entry)

    logger.info(f"Parsed {len(entries)} Hebrew lexicon entries from {file_path.name}")
    return entries


def _parse_abbott_smith_entry(entry_elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a single Abbott-Smith Greek lexicon entry element into a dict."""
    n_attr = entry_elem.get("n", "")
    if not n_attr:
        return None

    parts = n_attr.split("|")
    if len(parts) < 2:
        return None

    greek_word = parts[0]
    strongs_raw = parts[1]

    if not strongs_raw.startswith("G"):
        return None

    # Extract orth (canonical form) from <form><orth>
    form_elem = entry_elem.find("tei:form", ABBOTT_NS)
    if form_elem is not None:
        orth_elem = form_elem.find("tei:orth", ABBOTT_NS)
        if orth_elem is not None and orth_elem.text:
            greek_word = orth_elem.text.strip()

    # Extract gloss from first <sense><gloss>
    gloss = ""
    sense_elem = entry_elem.find(".//tei:sense", ABBOTT_NS)
    if sense_elem is not None:
        gloss_elem = sense_elem.find("tei:gloss", ABBOTT_NS)
        if gloss_elem is not None and gloss_elem.text:
            gloss = gloss_elem.text.strip()

    definition = _clean_text(_get_element_text(sense_elem)) if sense_elem is not None else ""
    if len(definition) > 2000:
        definition = definition[:2000] + "..."

    return {
        "strongs_number": strongs_raw,
        "original_word": greek_word,
        "transliteration": "",
        "part_of_speech": "",
        "gloss": gloss,
        "definition": definition,
        "language": "greek",
    }


def parse_abbott_smith_xml(file_path: Path) -> List[Dict[str, Any]]:
    """Parse Abbott-Smith Greek Lexicon TEI XML into lexicon entries.

    Source: https://github.com/translatable-exegetical-tools/Abbott-Smith
    License: Public Domain (1922, out of copyright)

    Args:
        file_path: Path to abbott-smith.tei.xml

    Returns:
        List of lexicon entry dicts ready for database insertion.
    """
    entries: List[Dict[str, Any]] = []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except (ET.ParseError, OSError) as e:
        logger.error(f"Failed to parse Abbott-Smith XML: {e}")
        return entries

    for entry_elem in root.findall(".//tei:entry", ABBOTT_NS):
        entry = _parse_abbott_smith_entry(entry_elem)
        if entry is not None:
            entries.append(entry)

    logger.info(f"Parsed {len(entries)} Greek lexicon entries from {file_path.name}")
    return entries


# ── BDB (Brown-Driver-Briggs) Hebrew Lexicon ──────────────────────────


def _build_strongs_to_bdb_map(index_path: Path) -> Dict[str, str]:
    """Build a mapping from Strong's numbers (H-prefixed) to BDB entry IDs.

    Uses LexicalIndex.xml which bridges Strong's numbers to BDB entries.

    Args:
        index_path: Path to LexicalIndex.xml

    Returns:
        Dict mapping Strong's number (e.g. "H0006") to BDB entry ID (e.g. "a.ac.aa")
    """
    mapping: Dict[str, str] = {}

    try:
        tree = ET.parse(index_path)
        root = tree.getroot()
    except (ET.ParseError, OSError) as e:
        logger.error(f"Failed to parse LexicalIndex.xml: {e}")
        return mapping

    for entry_elem in root.findall(".//lex:entry", HEBREW_NS):
        xref = entry_elem.find("lex:xref", HEBREW_NS)
        if xref is None:
            continue

        strong_num = xref.get("strong", "")
        bdb_id = xref.get("bdb", "")

        if strong_num and bdb_id:
            # Normalize Strong's number to H-prefixed format with zero-padding
            h_num = f"H{int(strong_num):04d}" if strong_num.isdigit() else strong_num
            mapping[h_num] = bdb_id

    logger.info(f"Built Strong's-to-BDB mapping with {len(mapping)} entries")
    return mapping


def _extract_bdb_entry_text(entry_elem: ET.Element) -> Dict[str, str]:
    """Extract definition components from a BDB entry element.

    Args:
        entry_elem: XML element for a BDB entry

    Returns:
        Dict with 'word', 'pos', 'gloss', and 'definition' keys.
    """
    # Extract Hebrew word from <w> element
    w_elem = entry_elem.find("lex:w", HEBREW_NS)
    word = w_elem.text.strip() if w_elem is not None and w_elem.text else ""

    # Extract part of speech from <pos>
    pos_elem = entry_elem.find("lex:pos", HEBREW_NS)
    pos = pos_elem.text.strip() if pos_elem is not None and pos_elem.text else ""

    # Extract primary gloss from first <def>
    def_elem = entry_elem.find(".//lex:def", HEBREW_NS)
    gloss = def_elem.text.strip() if def_elem is not None and def_elem.text else ""

    # Build full definition from all senses
    sense_parts: List[str] = []

    # Add top-level def if present
    if gloss:
        sense_parts.append(gloss)

    # Collect numbered senses
    for sense in entry_elem.findall(".//lex:sense", HEBREW_NS):
        sense_num = sense.get("n", "")
        sense_text = _clean_text(_get_element_text(sense))
        if sense_text:
            if sense_num:
                sense_parts.append(f"{sense_num}. {sense_text}")
            else:
                sense_parts.append(sense_text)

    definition = "; ".join(sense_parts) if sense_parts else ""
    if len(definition) > 2000:
        definition = definition[:2000] + "..."

    return {"word": word, "pos": pos, "gloss": gloss, "definition": definition}


def _invert_strongs_mapping(strongs_to_bdb: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert Strong's-to-BDB mapping to BDB-to-Strong's list."""
    bdb_to_strongs: Dict[str, List[str]] = {}
    for h_num, bdb_id in strongs_to_bdb.items():
        bdb_to_strongs.setdefault(bdb_id, []).append(h_num)
    return bdb_to_strongs


def _index_bdb_entries(root: ET.Element) -> Dict[str, ET.Element]:
    """Index all BDB XML entries by their ID attribute."""
    bdb_entries: Dict[str, ET.Element] = {}
    for entry_elem in root.findall(".//lex:entry", HEBREW_NS):
        entry_id = entry_elem.get("id", "")
        if entry_id:
            bdb_entries[entry_id] = entry_elem
    return bdb_entries


def parse_bdb_xml(bdb_path: Path, index_path: Path) -> List[Dict[str, Any]]:
    """Parse BDB (Brown-Driver-Briggs) Hebrew Lexicon XML into lexicon definitions.

    Uses LexicalIndex.xml to map BDB entries to Strong's numbers.

    Source: https://github.com/openscriptures/HebrewLexicon
    License: Public Domain (1906) / CC BY 4.0 (XML markup)

    Args:
        bdb_path: Path to BrownDriverBriggs.xml
        index_path: Path to LexicalIndex.xml

    Returns:
        List of lexicon definition dicts ready for database insertion.
    """
    entries: List[Dict[str, Any]] = []

    strongs_to_bdb = _build_strongs_to_bdb_map(index_path)
    if not strongs_to_bdb:
        logger.warning("No Strong's-to-BDB mapping found; BDB import will be empty")
        return entries

    bdb_to_strongs = _invert_strongs_mapping(strongs_to_bdb)

    try:
        tree = ET.parse(bdb_path)
        root = tree.getroot()
    except (ET.ParseError, OSError) as e:
        logger.error(f"Failed to parse BDB XML: {e}")
        return entries

    bdb_entries = _index_bdb_entries(root)

    for bdb_id, strongs_list in bdb_to_strongs.items():
        entry_elem = bdb_entries.get(bdb_id)
        if entry_elem is None:
            continue

        extracted = _extract_bdb_entry_text(entry_elem)
        if not extracted["definition"] and not extracted["gloss"]:
            continue

        for h_num in strongs_list:
            entries.append(
                {
                    "strongs_number": h_num,
                    "source_lexicon": "bdb",
                    "original_word": extracted["word"],
                    "transliteration": "",
                    "part_of_speech": extracted["pos"],
                    "gloss": extracted["gloss"],
                    "definition": extracted["definition"],
                    "language": "hebrew",
                }
            )

    logger.info(f"Parsed {len(entries)} BDB Hebrew lexicon definitions from {bdb_path.name}")
    return entries


# ── Dodson Greek-English Lexicon ──────────────────────────────────────


def parse_dodson_csv(file_path: Path) -> List[Dict[str, Any]]:
    """Parse Dodson Greek-English Lexicon CSV into lexicon definitions.

    The Dodson lexicon is a public-domain compilation from Abbott-Smith (1922),
    Berry (1897), Souter (1917), and Strong (1890).

    Source: https://github.com/biblicalhumanities/Dodson-Greek-Lexicon
    License: Public Domain (CC0)

    Args:
        file_path: Path to dodson.csv

    Returns:
        List of lexicon definition dicts ready for database insertion.
    """
    entries: List[Dict[str, Any]] = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.error(f"Failed to read Dodson CSV: {e}")
        return entries

    reader = csv.reader(io.StringIO(content))

    for row in reader:
        if len(row) < 4:
            continue

        # Expected columns: Strong's number, Greek word, transliteration,
        # brief gloss, full definition (some rows may have fewer/more columns)
        strongs_raw = row[0].strip()
        greek_word = row[1].strip() if len(row) > 1 else ""
        transliteration = row[2].strip() if len(row) > 2 else ""
        pos = row[3].strip() if len(row) > 3 else ""
        gloss = row[4].strip() if len(row) > 4 else ""
        definition = row[5].strip() if len(row) > 5 else ""

        # Only process G-prefixed entries (Greek)
        if not strongs_raw.startswith("G"):
            continue

        # Use gloss as definition fallback
        if not definition and gloss:
            definition = gloss

        entries.append(
            {
                "strongs_number": strongs_raw,
                "source_lexicon": "dodson",
                "original_word": greek_word,
                "transliteration": transliteration,
                "part_of_speech": pos,
                "gloss": gloss,
                "definition": definition,
                "language": "greek",
            }
        )

    logger.info(f"Parsed {len(entries)} Dodson Greek lexicon definitions from {file_path.name}")
    return entries
