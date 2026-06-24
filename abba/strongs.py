"""Strong's number normalization utilities."""

import re
from typing import Optional


def normalize_strongs(s: Optional[str]) -> str:
    """Normalize a Strong's number by removing leading zeros from the numeric part.

    Examples:
        H0430 -> H430
        G0746 -> G746
        G0901a -> G901a
        H9003 -> H9003 (no leading zeros anyway)
        "" or None -> ""
    """
    if not s:
        return ""
    s = s.strip()
    if not s:
        return ""
    match = re.match(r"^([HGhg])0*(\d+[a-z]?)$", s)
    if match:
        return match.group(1).upper() + match.group(2)
    return s.strip()


def extract_lexical_strongs(strongs_primary: Optional[str], strongs_raw: Optional[str]) -> str:
    """Extract the lexical Strong's number from STEP Bible data.

    Returns the Strong's in its SOURCE form (not zero-stripped) — e.g. ``G0746``, ``H0430`` — so the
    displayed/stored value matches the conventional padded form. Lexicon lookups normalize on their
    own (see :func:`normalize_strongs` used by ``get_lexicon_entry``), so ``/lexicon/H0430`` still
    resolves to the unpadded ``H430`` entry.

    Greek: strongs_primary is the lexical code (e.g. ``G0746``).
    Hebrew: strongs_primary is often a STEP prefix (``H9003``, NOT in the lexicon); the real code is
            in strongs_raw braces: ``H9003/{H7225G}`` -> ``H7225``; ``{H1254A}`` -> ``H1254``;
            ``{H0430G}`` -> ``H0430``.

    Logic:
    - If strongs_primary matches ^[HG]\\d and is NOT in the H9000-H9999 STEP-prefix range, return it.
    - Else take the FIRST {...} group in strongs_raw, strip braces and ONE trailing UPPERCASE STEP
      tag letter if present.
    - Else "".
    """
    # Check if strongs_primary is a valid lexical code (not H9000-H9999 STEP prefix range)
    if strongs_primary:
        primary_match = re.match(r"^([HGhg])(\d+)", strongs_primary)
        if primary_match:
            letter = primary_match.group(1).upper()
            num = int(primary_match.group(2))
            # H9000-H9999 are STEP internal codes, not lexical
            if not (letter == "H" and 9000 <= num <= 9999):
                return strongs_primary.strip()

    # Fall back to strongs_raw: extract first {...} group
    if strongs_raw:
        brace_match = re.search(r"\{([^}]+)\}", strongs_raw)
        if brace_match:
            code = brace_match.group(1)
            # Strip ONE trailing UPPERCASE letter if present (STEP tag): H7225G -> H7225, H0430G -> H0430
            code = re.sub(r"[A-Z]$", "", code)
            return code.strip()

    return ""
