"""Plain-English word explanations for common original-language words.

Provides accessible explanations of where meaning is richer in the original
language, framed positively as 'the original adds richness' rather than
'your Bible is wrong'.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (strongs_number, language, explanation)
# Top Hebrew and Greek words where English translations lose nuance
CURATED_EXPLANATIONS: List[Tuple[str, str, str]] = [
    # --- Top Hebrew Words ---
    (
        "H0430",
        "hebrew",
        "Elohim is plural in form but often singular in meaning when referring to God. "
        "This plurality-in-unity has fascinated scholars for millennia and adds depth "
        "to passages about the nature of God that a single English word 'God' cannot capture.",
    ),
    (
        "H7225",
        "hebrew",
        "Reshit means not just temporal 'beginning' but also 'firstfruits,' 'chief,' or 'best.' "
        "Genesis 1:1 hints at something qualitative — God's act of creation is His 'masterwork.'",
    ),
    (
        "H1254",
        "hebrew",
        "Bara is used exclusively with God as its subject. Unlike other Hebrew words for making "
        "or forming, bara implies creating something genuinely new — a divine prerogative.",
    ),
    (
        "H2617",
        "hebrew",
        "Chesed (often 'lovingkindness' or 'mercy') combines loyal love, covenant faithfulness, "
        "and gracious kindness. No single English word captures its full meaning — it describes "
        "God's committed, active, self-giving love.",
    ),
    (
        "H7307",
        "hebrew",
        "Ruach means 'spirit,' 'wind,' and 'breath' simultaneously. When Scripture speaks of "
        "the ruach of God, all three dimensions are present — the invisible power, the life-giving "
        "breath, and the sovereign Spirit.",
    ),
    (
        "H8064",
        "hebrew",
        "Shamayim (heavens) is inherently plural in Hebrew. It can refer to the atmosphere, "
        "outer space, or God's dwelling. The plurality invites awareness that the biblical 'sky' "
        "encompasses more than our modern category.",
    ),
    (
        "H0776",
        "hebrew",
        "Erets can mean 'earth' (the planet), 'land' (a region), or 'ground' (soil). Context "
        "determines meaning, but the semantic range shows how interconnected these ideas were "
        "in ancient thought.",
    ),
    (
        "H3068",
        "hebrew",
        "YHWH (the LORD) is God's personal, covenant name, derived from the verb 'to be.' "
        "English 'LORD' in small capitals marks this word, but the original carries the weight "
        "of God's self-revelation: 'I AM who I AM.'",
    ),
    (
        "H5315",
        "hebrew",
        "Nephesh (often 'soul') actually means the whole living being — body, desires, appetites, "
        "and life-force together. It resists the Greek soul/body split and speaks of the entire "
        "person.",
    ),
    (
        "H3820",
        "hebrew",
        "Lev (heart) in Hebrew refers not primarily to emotions but to the center of thought, "
        "will, and decision-making. When Scripture says 'with all your heart,' it means with "
        "your whole mind and will, not just feelings.",
    ),
    (
        "H6663",
        "hebrew",
        "Tsadaq (righteous/just) implies being in right relationship — with God, with community, "
        "with the created order. It is relational and active, not merely a moral status.",
    ),
    (
        "H7965",
        "hebrew",
        "Shalom means far more than 'peace' as absence of conflict. It encompasses wholeness, "
        "completeness, welfare, harmony, and flourishing in every dimension of life.",
    ),
    (
        "H0539",
        "hebrew",
        "Aman (believe/faith) carries the root meaning of firmness and reliability. To 'believe' "
        "in Hebrew is to lean on, to find solid ground — faith is about trustworthy support, "
        "not just mental assent.",
    ),
    (
        "H1285",
        "hebrew",
        "Berit (covenant) implies a binding agreement sealed by sacrifice. The word carries "
        "weight of blood commitment, mutual obligation, and unbreakable promise — deeper than "
        "a modern contract.",
    ),
    (
        "H3045",
        "hebrew",
        "Yada (know) implies intimate, experiential knowledge — not abstract head-knowledge. "
        "When Scripture says God 'knows' you or Adam 'knew' Eve, it speaks of deep personal "
        "encounter and relationship.",
    ),
    # --- Top Greek Words ---
    (
        "G0026",
        "greek",
        "Agape was a rare word that early Christians filled with new meaning: self-giving, "
        "sacrificial love that acts for the good of others regardless of their response. "
        "English 'love' covers too many concepts; agape is specifically divine-quality love.",
    ),
    (
        "G3056",
        "greek",
        "Logos means 'word,' but also 'reason,' 'logic,' 'discourse,' and 'ordering principle.' "
        "John 1:1 draws on both Hebrew (God's creative speech) and Greek philosophical "
        "traditions (the rational principle behind reality).",
    ),
    (
        "G4151",
        "greek",
        "Pneuma (spirit) echoes Hebrew ruach — it means breath, wind, and spirit. "
        "The wordplay is intentional: just as wind is invisible but powerfully real, "
        "so is God's Spirit.",
    ),
    (
        "G5485",
        "greek",
        "Charis (grace) originally meant beauty, charm, or favor. In the New Testament, it "
        "became the word for God's unmerited favor — a gift that transforms the recipient. "
        "It is active generosity, not passive permission.",
    ),
    (
        "G1343",
        "greek",
        "Dikaiosyne (righteousness/justice) carries both forensic (declared right) and "
        "transformative (made right) dimensions. Paul uses it to describe God's act of "
        "making people right with Himself — a status and a reality.",
    ),
    (
        "G0032",
        "greek",
        "Angelos means 'messenger' — human or heavenly. English 'angel' has become "
        "exclusively supernatural, but the Greek word reminds us that God communicates "
        "through many kinds of messengers.",
    ),
    (
        "G1577",
        "greek",
        "Ekklesia (church) literally means 'called-out assembly.' It was a political term "
        "for a citizen assembly, not a religious building. The early church was a gathered "
        "community of people, not a place.",
    ),
    (
        "G3340",
        "greek",
        "Metanoeo (repent) literally means to change one's mind or perception. It is not "
        "primarily about feeling sorry but about a fundamental shift in thinking and direction "
        "— seeing reality differently.",
    ),
    (
        "G4102",
        "greek",
        "Pistis (faith) encompasses trust, faithfulness, loyalty, and conviction. "
        "It is active reliance, not passive belief. The word connects 'I believe' with "
        "'I am faithful' — faith as lived commitment.",
    ),
    (
        "G0746",
        "greek",
        "Arche (beginning) also means 'origin,' 'first cause,' and 'ruling principle.' "
        "In John 1:1, 'In the beginning' echoes Genesis 1:1 while adding the philosophical "
        "dimension of ultimate origin and authority.",
    ),
    (
        "G2889",
        "greek",
        "Kosmos (world) carries multiple senses: the ordered universe, the human world, "
        "and the world system opposed to God. John uses it in all three senses, sometimes "
        "within a single discourse.",
    ),
    (
        "G1515",
        "greek",
        "Eirene (peace) echoes Hebrew shalom — wholeness, well-being, reconciliation. "
        "When Jesus says 'Peace I leave with you,' He offers more than calm feelings — "
        "He offers restored wholeness with God.",
    ),
    (
        "G3875",
        "greek",
        "Parakletos (Comforter/Advocate/Helper) combines legal advocacy with personal "
        "encouragement. The Holy Spirit is both defender and strengthener — a rich title "
        "that no single English word conveys.",
    ),
    (
        "G2316",
        "greek",
        "Theos (God) in Greek culture referred to any deity. The New Testament fills it "
        "with the content of YHWH — the one true God of Israel now revealed in Christ. "
        "Same word, transformed meaning.",
    ),
    (
        "G4982",
        "greek",
        "Sozo (save) means to rescue, heal, preserve, and make whole. 'Salvation' in the "
        "New Testament is not only spiritual deliverance but holistic restoration — body, "
        "soul, and community.",
    ),
]


class WordExplanationPopulator:
    """Populates word_explanations table with plain-English explanations."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert curated word explanations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if force:
                cursor.execute("DELETE FROM word_explanations")

            count = 0
            for strongs_number, language, explanation in CURATED_EXPLANATIONS:
                cursor.execute(
                    "SELECT COUNT(*) FROM word_explanations WHERE strongs_number = ?",
                    (strongs_number,),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO word_explanations (strongs_number, language, explanation) VALUES (?, ?, ?)",
                        (strongs_number, language, explanation),
                    )
                    count += 1
            conn.commit()
        logger.info("Populated %d word explanations", count)
        return count

    @staticmethod
    def get_explanation(db_path: Path, strongs_number: str) -> str:
        """Get plain-English explanation for a Strong's number."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT explanation FROM word_explanations WHERE strongs_number = ?",
                (strongs_number,),
            )
            row = cursor.fetchone()
            return row[0] if row else ""
