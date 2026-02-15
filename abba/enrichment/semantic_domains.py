"""Louw-Nida semantic domain classification for biblical Greek and Hebrew terms."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Representative top-level Louw-Nida semantic domains covering key biblical concepts.
# Based on the Louw-Nida Greek-English Lexicon of the New Testament Based on Semantic Domains.
# Format: (domain_number, domain_label, description, subdomain_count)
SEMANTIC_DOMAINS: List[Dict[str, Any]] = [
    {
        "domain_number": 1,
        "domain_label": "Geographical Objects and Features",
        "description": "Terms for land, water, sky, and other physical features of the earth and cosmos",
        "subdomain_count": 7,
    },
    {
        "domain_number": 2,
        "domain_label": "Natural Substances",
        "description": "Terms for materials like stone, metal, wood, water, fire, and other natural elements",
        "subdomain_count": 7,
    },
    {
        "domain_number": 4,
        "domain_label": "Animals",
        "description": "Terms for living creatures including domestic animals, wild beasts, birds, and fish",
        "subdomain_count": 9,
    },
    {
        "domain_number": 6,
        "domain_label": "Artifacts",
        "description": "Terms for human-made objects including tools, vessels, clothing, and buildings",
        "subdomain_count": 22,
    },
    {
        "domain_number": 9,
        "domain_label": "People",
        "description": "Terms for persons by kinship, social role, ethnicity, and other characteristics",
        "subdomain_count": 13,
    },
    {
        "domain_number": 10,
        "domain_label": "Kinship Terms",
        "description": "Terms for family relationships including parents, children, spouses, and extended family",
        "subdomain_count": 7,
    },
    {
        "domain_number": 11,
        "domain_label": "Groups and Classes of Persons",
        "description": "Terms for groups such as nations, tribes, congregations, and social classes",
        "subdomain_count": 7,
    },
    {
        "domain_number": 12,
        "domain_label": "Supernatural Beings and Powers",
        "description": "Terms for God, gods, angels, demons, Satan, and spiritual forces",
        "subdomain_count": 5,
    },
    {
        "domain_number": 13,
        "domain_label": "Be, Become, Exist, Happen",
        "description": "Terms for states of being, existence, occurrence, and change",
        "subdomain_count": 14,
    },
    {
        "domain_number": 15,
        "domain_label": "Linear Movement",
        "description": "Terms for physical movement including going, coming, walking, and traveling",
        "subdomain_count": 17,
    },
    {
        "domain_number": 20,
        "domain_label": "Violence, Harm, Destroy",
        "description": "Terms for violent actions, injury, destruction, and warfare",
        "subdomain_count": 9,
    },
    {
        "domain_number": 21,
        "domain_label": "Danger, Risk, Safe, Save",
        "description": "Terms for danger, safety, rescue, salvation, and deliverance",
        "subdomain_count": 5,
    },
    {
        "domain_number": 23,
        "domain_label": "Physiological Processes and States",
        "description": "Terms for bodily functions including birth, death, health, sickness, eating, and sleeping",
        "subdomain_count": 16,
    },
    {
        "domain_number": 24,
        "domain_label": "Sensory Events and States",
        "description": "Terms for seeing, hearing, tasting, smelling, and touching",
        "subdomain_count": 7,
    },
    {
        "domain_number": 25,
        "domain_label": "Attitudes and Emotions",
        "description": "Terms for love, hate, joy, sorrow, fear, anger, desire, and other emotional states",
        "subdomain_count": 13,
    },
    {
        "domain_number": 26,
        "domain_label": "Psychological Faculties",
        "description": "Terms for the mind, heart, soul, will, and other inner faculties of thought and intention",
        "subdomain_count": 4,
    },
    {
        "domain_number": 27,
        "domain_label": "Learn",
        "description": "Terms for learning, studying, acquiring knowledge, and intellectual development",
        "subdomain_count": 5,
    },
    {
        "domain_number": 28,
        "domain_label": "Know",
        "description": "Terms for knowledge, understanding, recognition, and awareness",
        "subdomain_count": 7,
    },
    {
        "domain_number": 30,
        "domain_label": "Think",
        "description": "Terms for reasoning, deciding, planning, intending, and mental processes",
        "subdomain_count": 11,
    },
    {
        "domain_number": 31,
        "domain_label": "Hold a View, Believe, Trust",
        "description": "Terms for opinions, beliefs, faith, trust, doubt, and conviction",
        "subdomain_count": 10,
    },
    {
        "domain_number": 33,
        "domain_label": "Communication",
        "description": "Terms for speaking, writing, proclaiming, teaching, commanding, and all forms of communication",
        "subdomain_count": 48,
    },
    {
        "domain_number": 34,
        "domain_label": "Association",
        "description": "Terms for fellowship, friendship, community, joining, separating, and social interaction",
        "subdomain_count": 9,
    },
    {
        "domain_number": 35,
        "domain_label": "Help, Care For",
        "description": "Terms for helping, serving, caring, supporting, and giving aid to others",
        "subdomain_count": 6,
    },
    {
        "domain_number": 36,
        "domain_label": "Guide, Discipline, Follow",
        "description": "Terms for leading, guiding, disciplining, obeying, following, and discipleship",
        "subdomain_count": 6,
    },
    {
        "domain_number": 37,
        "domain_label": "Control, Rule",
        "description": "Terms for authority, power, ruling, governing, and exercising dominion",
        "subdomain_count": 14,
    },
    {
        "domain_number": 38,
        "domain_label": "Punish, Reward",
        "description": "Terms for punishment, judgment, reward, retribution, and discipline",
        "subdomain_count": 3,
    },
    {
        "domain_number": 39,
        "domain_label": "Hostility, Strife",
        "description": "Terms for opposition, conflict, persecution, rebellion, and antagonism",
        "subdomain_count": 7,
    },
    {
        "domain_number": 40,
        "domain_label": "Reconciliation, Forgiveness",
        "description": "Terms for making peace, forgiving, pardoning, atonement, and restoring relationships",
        "subdomain_count": 5,
    },
    {
        "domain_number": 41,
        "domain_label": "Behavior and Related States",
        "description": "Terms for conduct, customs, habits, lifestyle, and moral behavior",
        "subdomain_count": 5,
    },
    {
        "domain_number": 53,
        "domain_label": "Religious Activities",
        "description": "Terms for worship, prayer, sacrifice, fasting, blessing, and ritual observances",
        "subdomain_count": 16,
    },
    {
        "domain_number": 56,
        "domain_label": "Courts and Legal Procedures",
        "description": "Terms for justice, judgment, courts, witnesses, laws, and legal actions",
        "subdomain_count": 6,
    },
    {
        "domain_number": 57,
        "domain_label": "Possess, Transfer, Exchange",
        "description": "Terms for owning, giving, receiving, buying, selling, and stewardship of possessions",
        "subdomain_count": 22,
    },
    {
        "domain_number": 58,
        "domain_label": "Nature, Class, Example",
        "description": "Terms for types, categories, patterns, models, and classifications",
        "subdomain_count": 7,
    },
    {
        "domain_number": 59,
        "domain_label": "Quantity",
        "description": "Terms for numbers, amounts, measures, fullness, and completeness",
        "subdomain_count": 10,
    },
    {
        "domain_number": 65,
        "domain_label": "Value",
        "description": "Terms for worth, preciousness, honor, disgrace, and comparative worth",
        "subdomain_count": 4,
    },
    {
        "domain_number": 67,
        "domain_label": "Time",
        "description": "Terms for time periods, ages, seasons, beginnings, endings, and duration",
        "subdomain_count": 20,
    },
    {
        "domain_number": 72,
        "domain_label": "True, False",
        "description": "Terms for truth, falsehood, reality, deception, and genuineness",
        "subdomain_count": 4,
    },
    {
        "domain_number": 76,
        "domain_label": "Power, Force",
        "description": "Terms for strength, ability, might, weakness, and divine power",
        "subdomain_count": 4,
    },
    {
        "domain_number": 79,
        "domain_label": "Features of Objects",
        "description": "Terms for physical characteristics including clean, unclean, pure, and defiled",
        "subdomain_count": 17,
    },
    {
        "domain_number": 88,
        "domain_label": "Moral and Ethical Qualities",
        "description": "Terms for righteousness, sin, holiness, wickedness, virtue, and ethical behavior",
        "subdomain_count": 32,
    },
    {
        "domain_number": 89,
        "domain_label": "Relations",
        "description": "Terms for logical relations including cause, result, purpose, condition, and contrast",
        "subdomain_count": 14,
    },
    {
        "domain_number": 90,
        "domain_label": "Case Relations",
        "description": "Terms for agent, instrument, beneficiary, and other grammatical-semantic case roles",
        "subdomain_count": 9,
    },
]

# Mappings from Strong's numbers to Louw-Nida subdomain codes.
# Format: (strongs_number, strongs_lang, word_transliteration, subdomain_code, subdomain_label, gloss)
STRONGS_DOMAIN_MAPPINGS: List[Dict[str, Any]] = [
    # Greek NT terms
    {
        "strongs": "G0026",
        "language": "greek",
        "transliteration": "agape",
        "subdomain_code": "25.43",
        "subdomain_label": "Love",
        "gloss": "love, selfless devotion",
    },
    {
        "strongs": "G0266",
        "language": "greek",
        "transliteration": "hamartia",
        "subdomain_code": "88.289",
        "subdomain_label": "Sin",
        "gloss": "sin, wrongdoing, missing the mark",
    },
    {
        "strongs": "G0281",
        "language": "greek",
        "transliteration": "amen",
        "subdomain_code": "72.6",
        "subdomain_label": "Truly",
        "gloss": "truly, so be it, verily",
    },
    {
        "strongs": "G0932",
        "language": "greek",
        "transliteration": "basileia",
        "subdomain_code": "37.64",
        "subdomain_label": "Kingdom",
        "gloss": "kingdom, reign, royal rule",
    },
    {
        "strongs": "G1097",
        "language": "greek",
        "transliteration": "ginosko",
        "subdomain_code": "28.1",
        "subdomain_label": "Know",
        "gloss": "to know, understand, perceive",
    },
    {
        "strongs": "G1342",
        "language": "greek",
        "transliteration": "dikaios",
        "subdomain_code": "88.12",
        "subdomain_label": "Righteous",
        "gloss": "righteous, just, upright",
    },
    {
        "strongs": "G1343",
        "language": "greek",
        "transliteration": "dikaiosyne",
        "subdomain_code": "88.13",
        "subdomain_label": "Righteousness",
        "gloss": "righteousness, justice, justification",
    },
    {
        "strongs": "G1515",
        "language": "greek",
        "transliteration": "eirene",
        "subdomain_code": "22.42",
        "subdomain_label": "Peace",
        "gloss": "peace, harmony, well-being",
    },
    {
        "strongs": "G1680",
        "language": "greek",
        "transliteration": "elpis",
        "subdomain_code": "25.59",
        "subdomain_label": "Hope",
        "gloss": "hope, expectation, confident anticipation",
    },
    {
        "strongs": "G1849",
        "language": "greek",
        "transliteration": "exousia",
        "subdomain_code": "37.35",
        "subdomain_label": "Authority",
        "gloss": "authority, power, right, jurisdiction",
    },
    {
        "strongs": "G2098",
        "language": "greek",
        "transliteration": "euangelion",
        "subdomain_code": "33.217",
        "subdomain_label": "Gospel",
        "gloss": "gospel, good news, glad tidings",
    },
    {
        "strongs": "G2316",
        "language": "greek",
        "transliteration": "theos",
        "subdomain_code": "12.1",
        "subdomain_label": "God",
        "gloss": "God, deity, the divine being",
    },
    {
        "strongs": "G2362",
        "language": "greek",
        "transliteration": "thronos",
        "subdomain_code": "37.72",
        "subdomain_label": "Throne",
        "gloss": "throne, seat of authority",
    },
    {
        "strongs": "G2411",
        "language": "greek",
        "transliteration": "hieron",
        "subdomain_code": "7.16",
        "subdomain_label": "Temple",
        "gloss": "temple, temple complex, sacred precinct",
    },
    {
        "strongs": "G2588",
        "language": "greek",
        "transliteration": "kardia",
        "subdomain_code": "26.3",
        "subdomain_label": "Heart",
        "gloss": "heart, inner self, mind, will",
    },
    {
        "strongs": "G2889",
        "language": "greek",
        "transliteration": "kosmos",
        "subdomain_code": "1.1",
        "subdomain_label": "World",
        "gloss": "world, universe, created order",
    },
    {
        "strongs": "G3056",
        "language": "greek",
        "transliteration": "logos",
        "subdomain_code": "33.100",
        "subdomain_label": "Word",
        "gloss": "word, message, discourse, the Word",
    },
    {
        "strongs": "G3551",
        "language": "greek",
        "transliteration": "nomos",
        "subdomain_code": "33.333",
        "subdomain_label": "Law",
        "gloss": "law, regulation, principle",
    },
    {
        "strongs": "G3588",
        "language": "greek",
        "transliteration": "ho",
        "subdomain_code": "92.24",
        "subdomain_label": "Article",
        "gloss": "the (definite article)",
    },
    {
        "strongs": "G4102",
        "language": "greek",
        "transliteration": "pistis",
        "subdomain_code": "31.85",
        "subdomain_label": "Trust",
        "gloss": "faith, trust, belief, confidence",
    },
    {
        "strongs": "G4151",
        "language": "greek",
        "transliteration": "pneuma",
        "subdomain_code": "12.18",
        "subdomain_label": "Spirit",
        "gloss": "spirit, Spirit, wind, breath",
    },
    {
        "strongs": "G4396",
        "language": "greek",
        "transliteration": "prophetes",
        "subdomain_code": "53.79",
        "subdomain_label": "Prophet",
        "gloss": "prophet, spokesperson for God",
    },
    {
        "strongs": "G4561",
        "language": "greek",
        "transliteration": "sarx",
        "subdomain_code": "26.7",
        "subdomain_label": "Flesh",
        "gloss": "flesh, human nature, physical body",
    },
    {
        "strongs": "G4991",
        "language": "greek",
        "transliteration": "soteria",
        "subdomain_code": "21.25",
        "subdomain_label": "Salvation",
        "gloss": "salvation, deliverance, rescue",
    },
    {
        "strongs": "G5485",
        "language": "greek",
        "transliteration": "charis",
        "subdomain_code": "88.66",
        "subdomain_label": "Grace",
        "gloss": "grace, unmerited favor, kindness",
    },
    {
        "strongs": "G5547",
        "language": "greek",
        "transliteration": "christos",
        "subdomain_code": "53.82",
        "subdomain_label": "Christ",
        "gloss": "Christ, Messiah, the Anointed One",
    },
    {
        "strongs": "G5590",
        "language": "greek",
        "transliteration": "psyche",
        "subdomain_code": "26.4",
        "subdomain_label": "Soul",
        "gloss": "soul, life, self, inner being",
    },
    {
        "strongs": "G1242",
        "language": "greek",
        "transliteration": "diatheke",
        "subdomain_code": "34.44",
        "subdomain_label": "Covenant",
        "gloss": "covenant, pact, testament, agreement",
    },
    {
        "strongs": "G3341",
        "language": "greek",
        "transliteration": "metanoia",
        "subdomain_code": "41.52",
        "subdomain_label": "Repentance",
        "gloss": "repentance, change of mind and heart",
    },
    {
        "strongs": "G0225",
        "language": "greek",
        "transliteration": "aletheia",
        "subdomain_code": "72.1",
        "subdomain_label": "Truth",
        "gloss": "truth, reality, genuineness",
    },
    # Hebrew OT terms
    {
        "strongs": "H0430",
        "language": "hebrew",
        "transliteration": "elohim",
        "subdomain_code": "12.1",
        "subdomain_label": "Supernatural Beings",
        "gloss": "God, gods, divine beings",
    },
    {
        "strongs": "H3068",
        "language": "hebrew",
        "transliteration": "YHWH",
        "subdomain_code": "12.1",
        "subdomain_label": "God",
        "gloss": "LORD, Yahweh, the covenant name of God",
    },
    {
        "strongs": "H7307",
        "language": "hebrew",
        "transliteration": "ruach",
        "subdomain_code": "12.18",
        "subdomain_label": "Spirit",
        "gloss": "spirit, wind, breath, Spirit of God",
    },
    {
        "strongs": "H1285",
        "language": "hebrew",
        "transliteration": "berith",
        "subdomain_code": "34.44",
        "subdomain_label": "Covenant",
        "gloss": "covenant, agreement, treaty",
    },
    {
        "strongs": "H2617",
        "language": "hebrew",
        "transliteration": "chesed",
        "subdomain_code": "25.51",
        "subdomain_label": "Mercy",
        "gloss": "steadfast love, lovingkindness, mercy, covenant faithfulness",
    },
    {
        "strongs": "H6664",
        "language": "hebrew",
        "transliteration": "tsedeq",
        "subdomain_code": "88.12",
        "subdomain_label": "Righteousness",
        "gloss": "righteousness, justice, rightness",
    },
    {
        "strongs": "H8451",
        "language": "hebrew",
        "transliteration": "torah",
        "subdomain_code": "33.333",
        "subdomain_label": "Law",
        "gloss": "law, instruction, teaching, Torah",
    },
    {
        "strongs": "H7965",
        "language": "hebrew",
        "transliteration": "shalom",
        "subdomain_code": "22.42",
        "subdomain_label": "Peace",
        "gloss": "peace, wholeness, well-being, prosperity",
    },
    {
        "strongs": "H0539",
        "language": "hebrew",
        "transliteration": "aman",
        "subdomain_code": "31.82",
        "subdomain_label": "Believe",
        "gloss": "to believe, be faithful, trust, be firm",
    },
    {
        "strongs": "H3467",
        "language": "hebrew",
        "transliteration": "yasha",
        "subdomain_code": "21.25",
        "subdomain_label": "Save",
        "gloss": "to save, deliver, rescue, help",
    },
    {
        "strongs": "H2403",
        "language": "hebrew",
        "transliteration": "chatta'ah",
        "subdomain_code": "88.289",
        "subdomain_label": "Sin",
        "gloss": "sin, sin offering, transgression",
    },
    {
        "strongs": "H1697",
        "language": "hebrew",
        "transliteration": "dabar",
        "subdomain_code": "33.100",
        "subdomain_label": "Word",
        "gloss": "word, matter, thing, speech",
    },
    {
        "strongs": "H4428",
        "language": "hebrew",
        "transliteration": "melek",
        "subdomain_code": "37.64",
        "subdomain_label": "King",
        "gloss": "king, ruler, sovereign",
    },
    {
        "strongs": "H5315",
        "language": "hebrew",
        "transliteration": "nephesh",
        "subdomain_code": "26.4",
        "subdomain_label": "Soul",
        "gloss": "soul, life, self, person, breath",
    },
    {
        "strongs": "H6918",
        "language": "hebrew",
        "transliteration": "qadosh",
        "subdomain_code": "88.24",
        "subdomain_label": "Holy",
        "gloss": "holy, sacred, set apart, consecrated",
    },
    {
        "strongs": "H3519",
        "language": "hebrew",
        "transliteration": "kabod",
        "subdomain_code": "65.5",
        "subdomain_label": "Glory",
        "gloss": "glory, honor, splendor, weight",
    },
    {
        "strongs": "H4899",
        "language": "hebrew",
        "transliteration": "mashiach",
        "subdomain_code": "53.82",
        "subdomain_label": "Anointed One",
        "gloss": "anointed one, messiah",
    },
    {
        "strongs": "H2580",
        "language": "hebrew",
        "transliteration": "chen",
        "subdomain_code": "88.66",
        "subdomain_label": "Grace",
        "gloss": "grace, favor, charm",
    },
    {
        "strongs": "H0571",
        "language": "hebrew",
        "transliteration": "emeth",
        "subdomain_code": "72.1",
        "subdomain_label": "Truth",
        "gloss": "truth, faithfulness, reliability",
    },
    {
        "strongs": "H5030",
        "language": "hebrew",
        "transliteration": "nabi",
        "subdomain_code": "53.79",
        "subdomain_label": "Prophet",
        "gloss": "prophet, spokesperson, one who speaks for God",
    },
    {
        "strongs": "H3820",
        "language": "hebrew",
        "transliteration": "leb",
        "subdomain_code": "26.3",
        "subdomain_label": "Heart",
        "gloss": "heart, mind, inner person, will",
    },
    {
        "strongs": "H1254",
        "language": "hebrew",
        "transliteration": "bara",
        "subdomain_code": "42.35",
        "subdomain_label": "Create",
        "gloss": "to create, make (used exclusively of divine creation)",
    },
    {
        "strongs": "H1350",
        "language": "hebrew",
        "transliteration": "gaal",
        "subdomain_code": "37.128",
        "subdomain_label": "Redeem",
        "gloss": "to redeem, act as kinsman-redeemer, buy back",
    },
    {
        "strongs": "H5545",
        "language": "hebrew",
        "transliteration": "salach",
        "subdomain_code": "40.8",
        "subdomain_label": "Forgive",
        "gloss": "to forgive, pardon (used exclusively of divine forgiveness)",
    },
    {
        "strongs": "H7725",
        "language": "hebrew",
        "transliteration": "shuv",
        "subdomain_code": "41.52",
        "subdomain_label": "Repent",
        "gloss": "to turn, return, repent, restore",
    },
    {
        "strongs": "H3045",
        "language": "hebrew",
        "transliteration": "yada",
        "subdomain_code": "28.1",
        "subdomain_label": "Know",
        "gloss": "to know, perceive, understand, experience",
    },
    {
        "strongs": "H6213",
        "language": "hebrew",
        "transliteration": "asah",
        "subdomain_code": "42.29",
        "subdomain_label": "Make",
        "gloss": "to do, make, accomplish, perform",
    },
    {
        "strongs": "H8199",
        "language": "hebrew",
        "transliteration": "shaphat",
        "subdomain_code": "56.20",
        "subdomain_label": "Judge",
        "gloss": "to judge, govern, vindicate, decide",
    },
    {
        "strongs": "H5769",
        "language": "hebrew",
        "transliteration": "olam",
        "subdomain_code": "67.95",
        "subdomain_label": "Everlasting",
        "gloss": "everlasting, eternal, forever, ancient time",
    },
]


class SemanticDomainPopulator:
    """Populates the semantic_domains and strongs_domain_mappings tables."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _ensure_tables(self, cursor: sqlite3.Cursor) -> bool:
        """Create tables if they do not exist.

        Returns:
            True if tables are ready, False otherwise.
        """
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_domains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_code TEXT UNIQUE NOT NULL,
                domain_name TEXT NOT NULL,
                parent_domain TEXT,
                description TEXT,
                level INTEGER DEFAULT 1
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strongs_domain_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strongs_number TEXT NOT NULL,
                domain_code TEXT NOT NULL,
                confidence REAL DEFAULT 0.9,
                UNIQUE(strongs_number, domain_code),
                FOREIGN KEY (domain_code) REFERENCES semantic_domains(domain_code)
            )
            """
        )
        return True

    def populate(self, force: bool = False) -> int:
        """Insert semantic domains and Strong's mappings into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            self._ensure_tables(cursor)

            if force:
                cursor.execute("DELETE FROM strongs_domain_mappings")
                cursor.execute("DELETE FROM semantic_domains")

            # Insert domains
            for domain in SEMANTIC_DOMAINS:
                try:
                    verb = "INSERT OR REPLACE" if force else "INSERT OR IGNORE"
                    domain_code = str(domain["domain_number"])
                    cursor.execute(
                        f"{verb} INTO semantic_domains "  # noqa: S608
                        "(domain_code, domain_name, parent_domain, description, level) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            domain_code,
                            domain["domain_label"],
                            None,
                            domain["description"],
                            1,
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert domain %s: %s", domain["domain_number"], e)

            # Insert Strong's mappings
            for mapping in STRONGS_DOMAIN_MAPPINGS:
                try:
                    verb = "INSERT OR REPLACE" if force else "INSERT OR IGNORE"
                    cursor.execute(
                        f"{verb} INTO strongs_domain_mappings "  # noqa: S608
                        "(strongs_number, domain_code, confidence) "
                        "VALUES (?, ?, ?)",
                        (
                            mapping["strongs"],
                            mapping["subdomain_code"],
                            0.9,
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert Strong's mapping %s: %s", mapping["strongs"], e)

            conn.commit()

        logger.info("Populated semantic domains: %d rows inserted", inserted)
        return inserted
