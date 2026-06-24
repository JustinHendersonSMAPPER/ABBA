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

    Greek: strongs_primary is the lexical code, PADDED (G0746; lexicon has G746).
    Hebrew: strongs_primary is often a STEP prefix (H9003, NOT in lexicon);
            the real code is in strongs_raw braces: H9003/{H7225G}->H7225; {H1254A}->H1254;
            {H0430G}->H0430->unpad->H430 (Elohim).

    Logic:
    - If strongs_primary matches ^[HG]\\d and is NOT in H9000-H9999 range, return normalize_strongs(strongs_primary).
    - Else take the FIRST {...} group in strongs_raw, strip braces, strip ONE trailing UPPERCASE letter if present,
      then normalize_strongs.
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
                return normalize_strongs(strongs_primary)

    # Fall back to strongs_raw: extract first {...} group
    if strongs_raw:
        brace_match = re.search(r"\{([^}]+)\}", strongs_raw)
        if brace_match:
            code = brace_match.group(1)
            # Strip ONE trailing UPPERCASE letter if present (e.g., H7225G -> H7225, H0430G -> H0430)
            code = re.sub(r"[A-Z]$", "", code)
            return normalize_strongs(code)

    return ""
