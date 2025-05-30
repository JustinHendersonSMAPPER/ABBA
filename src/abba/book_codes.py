"""
Book ID standardization for the ABBA project.

This module defines the canonical 3-letter book codes and provides
utilities for normalizing various book naming conventions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set


class BookCode(str, Enum):
    """Canonical 3-letter book codes for all biblical books."""

    # Old Testament (39 books)
    GEN = "GEN"  # Genesis
    EXO = "EXO"  # Exodus
    LEV = "LEV"  # Leviticus
    NUM = "NUM"  # Numbers
    DEU = "DEU"  # Deuteronomy
    JOS = "JOS"  # Joshua
    JDG = "JDG"  # Judges
    RUT = "RUT"  # Ruth
    SA1 = "1SA"  # 1 Samuel
    SA2 = "2SA"  # 2 Samuel
    KI1 = "1KI"  # 1 Kings
    KI2 = "2KI"  # 2 Kings
    CH1 = "1CH"  # 1 Chronicles
    CH2 = "2CH"  # 2 Chronicles
    EZR = "EZR"  # Ezra
    NEH = "NEH"  # Nehemiah
    EST = "EST"  # Esther
    JOB = "JOB"  # Job
    PSA = "PSA"  # Psalms
    PRO = "PRO"  # Proverbs
    ECC = "ECC"  # Ecclesiastes
    SNG = "SNG"  # Song of Solomon (Song of Songs)
    ISA = "ISA"  # Isaiah
    JER = "JER"  # Jeremiah
    LAM = "LAM"  # Lamentations
    EZK = "EZK"  # Ezekiel
    DAN = "DAN"  # Daniel
    HOS = "HOS"  # Hosea
    JOL = "JOL"  # Joel
    AMO = "AMO"  # Amos
    OBA = "OBA"  # Obadiah
    JON = "JON"  # Jonah
    MIC = "MIC"  # Micah
    NAM = "NAM"  # Nahum
    HAB = "HAB"  # Habakkuk
    ZEP = "ZEP"  # Zephaniah
    HAG = "HAG"  # Haggai
    ZEC = "ZEC"  # Zechariah
    MAL = "MAL"  # Malachi

    # New Testament (27 books)
    MAT = "MAT"  # Matthew
    MRK = "MRK"  # Mark
    LUK = "LUK"  # Luke
    JHN = "JHN"  # John
    ACT = "ACT"  # Acts
    ROM = "ROM"  # Romans
    CO1 = "1CO"  # 1 Corinthians
    CO2 = "2CO"  # 2 Corinthians
    GAL = "GAL"  # Galatians
    EPH = "EPH"  # Ephesians
    PHP = "PHP"  # Philippians
    COL = "COL"  # Colossians
    TH1 = "1TH"  # 1 Thessalonians
    TH2 = "2TH"  # 2 Thessalonians
    TI1 = "1TI"  # 1 Timothy
    TI2 = "2TI"  # 2 Timothy
    TIT = "TIT"  # Titus
    PHM = "PHM"  # Philemon
    HEB = "HEB"  # Hebrews
    JAS = "JAS"  # James
    PE1 = "1PE"  # 1 Peter
    PE2 = "2PE"  # 2 Peter
    JN1 = "1JN"  # 1 John
    JN2 = "2JN"  # 2 John
    JN3 = "3JN"  # 3 John
    JUD = "JUD"  # Jude
    REV = "REV"  # Revelation


class Testament(Enum):
    """Testament classification."""

    OLD = "OT"
    NEW = "NT"


class Canon(Enum):
    """Biblical canon traditions."""

    PROTESTANT = "protestant"
    CATHOLIC = "catholic"
    ORTHODOX = "orthodox"
    ETHIOPIAN = "ethiopian"


# Book metadata
BOOK_INFO: Dict[str, Dict[str, Any]] = {
    # Old Testament
    "GEN": {"name": "Genesis", "abbr": "Gen", "chapters": 50, "testament": Testament.OLD},
    "EXO": {"name": "Exodus", "abbr": "Exod", "chapters": 40, "testament": Testament.OLD},
    "LEV": {"name": "Leviticus", "abbr": "Lev", "chapters": 27, "testament": Testament.OLD},
    "NUM": {"name": "Numbers", "abbr": "Num", "chapters": 36, "testament": Testament.OLD},
    "DEU": {"name": "Deuteronomy", "abbr": "Deut", "chapters": 34, "testament": Testament.OLD},
    "JOS": {"name": "Joshua", "abbr": "Josh", "chapters": 24, "testament": Testament.OLD},
    "JDG": {"name": "Judges", "abbr": "Judg", "chapters": 21, "testament": Testament.OLD},
    "RUT": {"name": "Ruth", "abbr": "Ruth", "chapters": 4, "testament": Testament.OLD},
    "1SA": {"name": "1 Samuel", "abbr": "1Sam", "chapters": 31, "testament": Testament.OLD},
    "2SA": {"name": "2 Samuel", "abbr": "2Sam", "chapters": 24, "testament": Testament.OLD},
    "1KI": {"name": "1 Kings", "abbr": "1Kgs", "chapters": 22, "testament": Testament.OLD},
    "2KI": {"name": "2 Kings", "abbr": "2Kgs", "chapters": 25, "testament": Testament.OLD},
    "1CH": {"name": "1 Chronicles", "abbr": "1Chr", "chapters": 29, "testament": Testament.OLD},
    "2CH": {"name": "2 Chronicles", "abbr": "2Chr", "chapters": 36, "testament": Testament.OLD},
    "EZR": {"name": "Ezra", "abbr": "Ezra", "chapters": 10, "testament": Testament.OLD},
    "NEH": {"name": "Nehemiah", "abbr": "Neh", "chapters": 13, "testament": Testament.OLD},
    "EST": {"name": "Esther", "abbr": "Esth", "chapters": 10, "testament": Testament.OLD},
    "JOB": {"name": "Job", "abbr": "Job", "chapters": 42, "testament": Testament.OLD},
    "PSA": {"name": "Psalms", "abbr": "Ps", "chapters": 150, "testament": Testament.OLD},
    "PRO": {"name": "Proverbs", "abbr": "Prov", "chapters": 31, "testament": Testament.OLD},
    "ECC": {"name": "Ecclesiastes", "abbr": "Eccl", "chapters": 12, "testament": Testament.OLD},
    "SNG": {"name": "Song of Solomon", "abbr": "Song", "chapters": 8, "testament": Testament.OLD},
    "ISA": {"name": "Isaiah", "abbr": "Isa", "chapters": 66, "testament": Testament.OLD},
    "JER": {"name": "Jeremiah", "abbr": "Jer", "chapters": 52, "testament": Testament.OLD},
    "LAM": {"name": "Lamentations", "abbr": "Lam", "chapters": 5, "testament": Testament.OLD},
    "EZK": {"name": "Ezekiel", "abbr": "Ezek", "chapters": 48, "testament": Testament.OLD},
    "DAN": {"name": "Daniel", "abbr": "Dan", "chapters": 12, "testament": Testament.OLD},
    "HOS": {"name": "Hosea", "abbr": "Hos", "chapters": 14, "testament": Testament.OLD},
    "JOL": {"name": "Joel", "abbr": "Joel", "chapters": 3, "testament": Testament.OLD},
    "AMO": {"name": "Amos", "abbr": "Amos", "chapters": 9, "testament": Testament.OLD},
    "OBA": {"name": "Obadiah", "abbr": "Obad", "chapters": 1, "testament": Testament.OLD},
    "JON": {"name": "Jonah", "abbr": "Jonah", "chapters": 4, "testament": Testament.OLD},
    "MIC": {"name": "Micah", "abbr": "Mic", "chapters": 7, "testament": Testament.OLD},
    "NAM": {"name": "Nahum", "abbr": "Nah", "chapters": 3, "testament": Testament.OLD},
    "HAB": {"name": "Habakkuk", "abbr": "Hab", "chapters": 3, "testament": Testament.OLD},
    "ZEP": {"name": "Zephaniah", "abbr": "Zeph", "chapters": 3, "testament": Testament.OLD},
    "HAG": {"name": "Haggai", "abbr": "Hag", "chapters": 2, "testament": Testament.OLD},
    "ZEC": {"name": "Zechariah", "abbr": "Zech", "chapters": 14, "testament": Testament.OLD},
    "MAL": {"name": "Malachi", "abbr": "Mal", "chapters": 4, "testament": Testament.OLD},
    # New Testament
    "MAT": {"name": "Matthew", "abbr": "Matt", "chapters": 28, "testament": Testament.NEW},
    "MRK": {"name": "Mark", "abbr": "Mark", "chapters": 16, "testament": Testament.NEW},
    "LUK": {"name": "Luke", "abbr": "Luke", "chapters": 24, "testament": Testament.NEW},
    "JHN": {"name": "John", "abbr": "John", "chapters": 21, "testament": Testament.NEW},
    "ACT": {"name": "Acts", "abbr": "Acts", "chapters": 28, "testament": Testament.NEW},
    "ROM": {"name": "Romans", "abbr": "Rom", "chapters": 16, "testament": Testament.NEW},
    "1CO": {"name": "1 Corinthians", "abbr": "1Cor", "chapters": 16, "testament": Testament.NEW},
    "2CO": {"name": "2 Corinthians", "abbr": "2Cor", "chapters": 13, "testament": Testament.NEW},
    "GAL": {"name": "Galatians", "abbr": "Gal", "chapters": 6, "testament": Testament.NEW},
    "EPH": {"name": "Ephesians", "abbr": "Eph", "chapters": 6, "testament": Testament.NEW},
    "PHP": {"name": "Philippians", "abbr": "Phil", "chapters": 4, "testament": Testament.NEW},
    "COL": {"name": "Colossians", "abbr": "Col", "chapters": 4, "testament": Testament.NEW},
    "1TH": {"name": "1 Thessalonians", "abbr": "1Thess", "chapters": 5, "testament": Testament.NEW},
    "2TH": {"name": "2 Thessalonians", "abbr": "2Thess", "chapters": 3, "testament": Testament.NEW},
    "1TI": {"name": "1 Timothy", "abbr": "1Tim", "chapters": 6, "testament": Testament.NEW},
    "2TI": {"name": "2 Timothy", "abbr": "2Tim", "chapters": 4, "testament": Testament.NEW},
    "TIT": {"name": "Titus", "abbr": "Titus", "chapters": 3, "testament": Testament.NEW},
    "PHM": {"name": "Philemon", "abbr": "Phlm", "chapters": 1, "testament": Testament.NEW},
    "HEB": {"name": "Hebrews", "abbr": "Heb", "chapters": 13, "testament": Testament.NEW},
    "JAS": {"name": "James", "abbr": "Jas", "chapters": 5, "testament": Testament.NEW},
    "1PE": {"name": "1 Peter", "abbr": "1Pet", "chapters": 5, "testament": Testament.NEW},
    "2PE": {"name": "2 Peter", "abbr": "2Pet", "chapters": 3, "testament": Testament.NEW},
    "1JN": {"name": "1 John", "abbr": "1John", "chapters": 5, "testament": Testament.NEW},
    "2JN": {"name": "2 John", "abbr": "2John", "chapters": 1, "testament": Testament.NEW},
    "3JN": {"name": "3 John", "abbr": "3John", "chapters": 1, "testament": Testament.NEW},
    "JUD": {"name": "Jude", "abbr": "Jude", "chapters": 1, "testament": Testament.NEW},
    "REV": {"name": "Revelation", "abbr": "Rev", "chapters": 22, "testament": Testament.NEW},
}


# Alternative book names and abbreviations mapping
BOOK_ALIASES: Dict[str, str] = {
    # Genesis variations
    "genesis": "GEN",
    "gen": "GEN",
    "ge": "GEN",
    "gn": "GEN",
    # Exodus variations
    "exodus": "EXO",
    "exod": "EXO",
    "ex": "EXO",
    "exo": "EXO",
    # Leviticus variations
    "leviticus": "LEV",
    "lev": "LEV",
    "le": "LEV",
    "lv": "LEV",
    # Numbers variations
    "numbers": "NUM",
    "num": "NUM",
    "nu": "NUM",
    "nm": "NUM",
    "nb": "NUM",
    # Deuteronomy variations
    "deuteronomy": "DEU",
    "deut": "DEU",
    "dt": "DEU",
    "de": "DEU",
    # Joshua variations
    "joshua": "JOS",
    "josh": "JOS",
    "jos": "JOS",
    "jsh": "JOS",
    # Judges variations
    "judges": "JDG",
    "judg": "JDG",
    "jdg": "JDG",
    "jg": "JDG",
    "jdgs": "JDG",
    # Ruth variations
    "ruth": "RUT",
    "rut": "RUT",
    "ru": "RUT",
    "rth": "RUT",
    # 1 Samuel variations
    "1 samuel": "1SA",
    "1samuel": "1SA",
    "1sam": "1SA",
    "1sa": "1SA",
    "1 sam": "1SA",
    "1 sa": "1SA",
    "i sam": "1SA",
    "i samuel": "1SA",
    "1s": "1SA",
    # 2 Samuel variations
    "2 samuel": "2SA",
    "2samuel": "2SA",
    "2sam": "2SA",
    "2sa": "2SA",
    "2 sam": "2SA",
    "ii sam": "2SA",
    "ii samuel": "2SA",
    "2s": "2SA",
    # 1 Kings variations
    "1 kings": "1KI",
    "1kings": "1KI",
    "1kgs": "1KI",
    "1ki": "1KI",
    "1 kgs": "1KI",
    "i kgs": "1KI",
    "i kings": "1KI",
    "1k": "1KI",
    # 2 Kings variations
    "2 kings": "2KI",
    "2kings": "2KI",
    "2kgs": "2KI",
    "2ki": "2KI",
    "2 kgs": "2KI",
    "ii kgs": "2KI",
    "ii kings": "2KI",
    "2k": "2KI",
    # 1 Chronicles variations
    "1 chronicles": "1CH",
    "1chronicles": "1CH",
    "1chr": "1CH",
    "1ch": "1CH",
    "1 chr": "1CH",
    "i chr": "1CH",
    "i chronicles": "1CH",
    "1 chron": "1CH",
    # 2 Chronicles variations
    "2 chronicles": "2CH",
    "2chronicles": "2CH",
    "2chr": "2CH",
    "2ch": "2CH",
    "2 chr": "2CH",
    "ii chr": "2CH",
    "ii chronicles": "2CH",
    "2 chron": "2CH",
    # Ezra variations
    "ezra": "EZR",
    "ezr": "EZR",
    # Nehemiah variations
    "nehemiah": "NEH",
    "neh": "NEH",
    "ne": "NEH",
    # Esther variations
    "esther": "EST",
    "esth": "EST",
    "est": "EST",
    "es": "EST",
    # Job variations
    "job": "JOB",
    "jb": "JOB",
    # Psalms variations
    "psalms": "PSA",
    "psalm": "PSA",
    "ps": "PSA",
    "psa": "PSA",
    "pss": "PSA",
    # Proverbs variations
    "proverbs": "PRO",
    "prov": "PRO",
    "pro": "PRO",
    "pr": "PRO",
    "prv": "PRO",
    # Ecclesiastes variations
    "ecclesiastes": "ECC",
    "eccl": "ECC",
    "ecc": "ECC",
    "ec": "ECC",
    "qoh": "ECC",
    # Song of Solomon variations
    "song of solomon": "SNG",
    "song of songs": "SNG",
    "song": "SNG",
    "canticles": "SNG",
    "sos": "SNG",
    "sng": "SNG",
    "ss": "SNG",
    # Isaiah variations
    "isaiah": "ISA",
    "isa": "ISA",
    "is": "ISA",
    # Jeremiah variations
    "jeremiah": "JER",
    "jer": "JER",
    "je": "JER",
    "jr": "JER",
    # Lamentations variations
    "lamentations": "LAM",
    "lam": "LAM",
    "la": "LAM",
    # Ezekiel variations
    "ezekiel": "EZK",
    "ezek": "EZK",
    "ezk": "EZK",
    "eze": "EZK",
    "ez": "EZK",
    # Daniel variations
    "daniel": "DAN",
    "dan": "DAN",
    "da": "DAN",
    "dn": "DAN",
    # Minor Prophets
    "hosea": "HOS",
    "hos": "HOS",
    "ho": "HOS",
    "joel": "JOL",
    "jol": "JOL",
    "joe": "JOL",
    "jl": "JOL",
    "amos": "AMO",
    "amo": "AMO",
    "am": "AMO",
    "obadiah": "OBA",
    "obad": "OBA",
    "oba": "OBA",
    "ob": "OBA",
    "jonah": "JON",
    "jon": "JON",
    "jnh": "JON",
    "micah": "MIC",
    "mic": "MIC",
    "mi": "MIC",
    "nahum": "NAM",
    "nah": "NAM",
    "nam": "NAM",
    "na": "NAM",
    "habakkuk": "HAB",
    "hab": "HAB",
    "hb": "HAB",
    "zephaniah": "ZEP",
    "zeph": "ZEP",
    "zep": "ZEP",
    "zp": "ZEP",
    "haggai": "HAG",
    "hag": "HAG",
    "hg": "HAG",
    "zechariah": "ZEC",
    "zech": "ZEC",
    "zec": "ZEC",
    "zc": "ZEC",
    "malachi": "MAL",
    "mal": "MAL",
    "ml": "MAL",
    # Matthew variations
    "matthew": "MAT",
    "matt": "MAT",
    "mat": "MAT",
    "mt": "MAT",
    # Mark variations
    "mark": "MRK",
    "mrk": "MRK",
    "mk": "MRK",
    "mr": "MRK",
    "mar": "MRK",
    # Luke variations
    "luke": "LUK",
    "luk": "LUK",
    "lk": "LUK",
    "lu": "LUK",
    # John variations
    "john": "JHN",
    "jhn": "JHN",
    "jn": "JHN",
    "joh": "JHN",
    # Acts variations
    "acts": "ACT",
    "act": "ACT",
    "ac": "ACT",
    "acts of the apostles": "ACT",
    # Romans variations
    "romans": "ROM",
    "rom": "ROM",
    "ro": "ROM",
    "rm": "ROM",
    # 1 Corinthians variations
    "1 corinthians": "1CO",
    "1corinthians": "1CO",
    "1cor": "1CO",
    "1co": "1CO",
    "1 cor": "1CO",
    "i cor": "1CO",
    "i corinthians": "1CO",
    "1c": "1CO",
    # 2 Corinthians variations
    "2 corinthians": "2CO",
    "2corinthians": "2CO",
    "2cor": "2CO",
    "2co": "2CO",
    "2 cor": "2CO",
    "ii cor": "2CO",
    "ii corinthians": "2CO",
    "2c": "2CO",
    # Galatians variations
    "galatians": "GAL",
    "gal": "GAL",
    "ga": "GAL",
    # Ephesians variations
    "ephesians": "EPH",
    "eph": "EPH",
    "ep": "EPH",
    # Philippians variations
    "philippians": "PHP",
    "phil": "PHP",
    "php": "PHP",
    "ph": "PHP",
    "pp": "PHP",
    # Colossians variations
    "colossians": "COL",
    "col": "COL",
    "co": "COL",
    # 1 Thessalonians variations
    "1 thessalonians": "1TH",
    "1thessalonians": "1TH",
    "1thess": "1TH",
    "1th": "1TH",
    "1 thess": "1TH",
    "i thess": "1TH",
    "i thessalonians": "1TH",
    # 2 Thessalonians variations
    "2 thessalonians": "2TH",
    "2thessalonians": "2TH",
    "2thess": "2TH",
    "2th": "2TH",
    "2 thess": "2TH",
    "ii thess": "2TH",
    "ii thessalonians": "2TH",
    # 1 Timothy variations
    "1 timothy": "1TI",
    "1timothy": "1TI",
    "1tim": "1TI",
    "1ti": "1TI",
    "1 tim": "1TI",
    "i tim": "1TI",
    "i timothy": "1TI",
    "1t": "1TI",
    # 2 Timothy variations
    "2 timothy": "2TI",
    "2timothy": "2TI",
    "2tim": "2TI",
    "2ti": "2TI",
    "2 tim": "2TI",
    "ii tim": "2TI",
    "ii timothy": "2TI",
    "2t": "2TI",
    # Titus variations
    "titus": "TIT",
    "tit": "TIT",
    "ti": "TIT",
    # Philemon variations
    "philemon": "PHM",
    "phlm": "PHM",
    "phm": "PHM",
    "phile": "PHM",
    # Hebrews variations
    "hebrews": "HEB",
    "heb": "HEB",
    "he": "HEB",
    # James variations
    "james": "JAS",
    "jas": "JAS",
    "jm": "JAS",
    "ja": "JAS",
    "jam": "JAS",
    # 1 Peter variations
    "1 peter": "1PE",
    "1peter": "1PE",
    "1pet": "1PE",
    "1pe": "1PE",
    "1 pet": "1PE",
    "i pet": "1PE",
    "i peter": "1PE",
    "1p": "1PE",
    # 2 Peter variations
    "2 peter": "2PE",
    "2peter": "2PE",
    "2pet": "2PE",
    "2pe": "2PE",
    "2 pet": "2PE",
    "ii pet": "2PE",
    "ii peter": "2PE",
    "2p": "2PE",
    # 1 John variations
    "1 john": "1JN",
    "1john": "1JN",
    "1jn": "1JN",
    "1jo": "1JN",
    "1 jn": "1JN",
    "i jn": "1JN",
    "i john": "1JN",
    "1j": "1JN",
    # 2 John variations
    "2 john": "2JN",
    "2john": "2JN",
    "2jn": "2JN",
    "2jo": "2JN",
    "2 jn": "2JN",
    "ii jn": "2JN",
    "ii john": "2JN",
    "2j": "2JN",
    # 3 John variations
    "3 john": "3JN",
    "3john": "3JN",
    "3jn": "3JN",
    "3jo": "3JN",
    "3 jn": "3JN",
    "iii jn": "3JN",
    "iii john": "3JN",
    "3j": "3JN",
    # Jude variations
    "jude": "JUD",
    "jud": "JUD",
    "jd": "JUD",
    # Revelation variations
    "revelation": "REV",
    "rev": "REV",
    "re": "REV",
    "revelations": "REV",
    "apocalypse": "REV",
    "apoc": "REV",
}


# Canon membership
CANON_BOOKS: Dict[Canon, Set[str]] = {
    Canon.PROTESTANT: set(BOOK_INFO.keys()),  # All 66 books
    Canon.CATHOLIC: set(BOOK_INFO.keys()),  # TODO: Add deuterocanonical books
    Canon.ORTHODOX: set(BOOK_INFO.keys()),  # TODO: Add additional Orthodox books
    Canon.ETHIOPIAN: set(BOOK_INFO.keys()),  # TODO: Add Ethiopian canon books
}


def normalize_book_name(book_name: str) -> Optional[str]:
    """
    Normalize a book name to its canonical 3-letter code.

    Args:
        book_name: Book name in any common format

    Returns:
        Canonical 3-letter book code, or None if not found
    """
    # First check if it's already a valid book code
    if book_name.upper() in BOOK_INFO:
        return book_name.upper()

    # Try lowercase lookup in aliases
    normalized = book_name.lower().strip()
    return BOOK_ALIASES.get(normalized)


def get_book_info(book_code: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a book by its code.

    Args:
        book_code: 3-letter book code

    Returns:
        Dictionary with book metadata, or None if not found
    """
    return BOOK_INFO.get(book_code.upper())


def get_book_name(book_code: str, form: str = "name") -> Optional[str]:
    """
    Get the name of a book in various forms.

    Args:
        book_code: 3-letter book code
        form: "name" for full name, "abbr" for abbreviation

    Returns:
        Book name in requested form, or None if not found
    """
    info = get_book_info(book_code)
    if info:
        return info.get(form)
    return None


def get_books_by_testament(testament: Testament) -> List[str]:
    """
    Get all book codes for a given testament.

    Args:
        testament: Testament.OLD or Testament.NEW

    Returns:
        List of 3-letter book codes
    """
    return [code for code, info in BOOK_INFO.items() if info["testament"] == testament]


def get_books_by_canon(canon: Canon) -> Set[str]:
    """
    Get all book codes included in a specific canon.

    Args:
        canon: Canon tradition

    Returns:
        Set of 3-letter book codes
    """
    return CANON_BOOKS.get(canon, set())


def get_book_order(book_code: str) -> Optional[int]:
    """
    Get the canonical order of a book (1-66 for Protestant canon).

    Args:
        book_code: 3-letter book code

    Returns:
        Book order number, or None if not found
    """
    all_books = list(BOOK_INFO.keys())
    try:
        return all_books.index(book_code.upper()) + 1
    except ValueError:
        return None


def is_valid_book_code(book_code: str) -> bool:
    """
    Check if a string is a valid canonical book code.

    Args:
        book_code: String to check

    Returns:
        True if valid book code, False otherwise
    """
    return book_code.upper() in BOOK_INFO


def get_chapter_count(book_code: str) -> Optional[int]:
    """
    Get the number of chapters in a book.

    Args:
        book_code: 3-letter book code

    Returns:
        Number of chapters, or None if book not found
    """
    info = get_book_info(book_code)
    if info:
        return info.get("chapters")
    return None
