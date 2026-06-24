"""Manuscript variant surfacing for textual criticism awareness."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Well-known textual variants from the manuscript tradition.
# These represent significant differences between manuscript families that
# affect modern Bible translation decisions.
# Fields: book_id, chapter, verse, variant_type (addition/omission/substitution/transposition),
#         base_text, variant_text, manuscripts, explanation, significance (major/minor/orthographic),
#         confidence (probability that the critical text reading is original)
MANUSCRIPT_VARIANTS: List[Dict[str, Any]] = [
    # Mark 16:9-20 — The Longer Ending of Mark
    {
        "book_id": 41,
        "chapter": 16,
        "verse": 9,
        "variant_type": "addition",
        "base_text": "[Mark ends at 16:8]",
        "variant_text": "Longer ending (16:9-20) including resurrection appearances, Great Commission, "
        "signs following believers, and the ascension",
        "manuscripts": "Sinaiticus, Vaticanus omit; Alexandrinus, Ephraemi, D include",
        "explanation": "The two oldest and most reliable manuscripts (Sinaiticus and Vaticanus) end Mark at "
        "16:8. The longer ending appears to be a later summary drawing from the other Gospels and Acts. "
        "The abrupt ending at 16:8 ('for they were afraid') is likely original.",
        "significance": "major",
        "confidence": 0.90,
    },
    # John 7:53-8:11 — Pericope Adulterae (Woman caught in adultery)
    {
        "book_id": 43,
        "chapter": 7,
        "verse": 53,
        "variant_type": "addition",
        "base_text": "[Passage absent from earliest manuscripts]",
        "variant_text": "The story of the woman caught in adultery (7:53-8:11), 'Let him who is without "
        "sin cast the first stone'",
        "manuscripts": "P66, P75, Sinaiticus, Vaticanus, earliest versions omit; D, later Byzantine include",
        "explanation": "Absent from the earliest and best Greek manuscripts. When it does appear, it is placed "
        "in different locations (after John 7:36, 7:44, 21:25, or even Luke 21:38). The story may preserve "
        "an authentic oral tradition but was not part of the original Gospel of John.",
        "significance": "major",
        "confidence": 0.95,
    },
    # 1 John 5:7-8 — Comma Johanneum
    {
        "book_id": 62,
        "chapter": 5,
        "verse": 7,
        "variant_type": "addition",
        "base_text": "For there are three that testify: the Spirit and the water and the blood",
        "variant_text": "For there are three that bear record in heaven, the Father, the Word, and the Holy "
        "Ghost: and these three are one. And there are three that bear witness in earth, the Spirit, "
        "and the water, and the blood",
        "manuscripts": "All Greek manuscripts before the 14th century omit; found in late Vulgate manuscripts",
        "explanation": "The Trinitarian formula (Comma Johanneum) is absent from all Greek manuscripts before "
        "the 14th century and from all early versions. It likely originated as a marginal gloss that was "
        "incorporated into the Latin Vulgate text. Erasmus famously included it only under pressure.",
        "significance": "major",
        "confidence": 0.99,
    },
    # Romans 16:24 — Grace benediction
    {
        "book_id": 45,
        "chapter": 16,
        "verse": 24,
        "variant_type": "addition",
        "base_text": "[Verse absent from earliest manuscripts]",
        "variant_text": "The grace of our Lord Jesus Christ be with you all. Amen.",
        "manuscripts": "P61, Sinaiticus, Vaticanus, A, B, C omit; later Byzantine manuscripts include",
        "explanation": "This verse duplicates the benediction already found in 16:20b. The earliest and best "
        "manuscripts omit it entirely. It appears to be a scribal addition harmonizing with the typical "
        "Pauline letter closing formula.",
        "significance": "minor",
        "confidence": 0.90,
    },
    # Matthew 6:13b — Lord's Prayer Doxology
    {
        "book_id": 40,
        "chapter": 6,
        "verse": 13,
        "variant_type": "addition",
        "base_text": "And lead us not into temptation, but deliver us from evil.",
        "variant_text": "And lead us not into temptation, but deliver us from evil. For thine is the kingdom, "
        "and the power, and the glory, for ever. Amen.",
        "manuscripts": "Sinaiticus, Vaticanus, D, Latin, Sahidic omit; later Byzantine manuscripts include",
        "explanation": "The doxology ('For thine is the kingdom...') is absent from the earliest Greek "
        "manuscripts and the early Latin and Coptic versions. It was likely added for liturgical use in "
        "early Christian worship, drawing from 1 Chronicles 29:11-13.",
        "significance": "major",
        "confidence": 0.85,
    },
    # Luke 22:43-44 — Angel strengthening Jesus / Bloody sweat
    {
        "book_id": 42,
        "chapter": 22,
        "verse": 43,
        "variant_type": "addition",
        "base_text": "[Verses absent from some early manuscripts]",
        "variant_text": "An angel from heaven appeared to him and strengthened him. And being in anguish, "
        "he prayed more earnestly, and his sweat was like drops of blood falling to the ground.",
        "manuscripts": "P75, Sinaiticus (first hand), Vaticanus, A, W omit; Sinaiticus (corrector), D include",
        "explanation": "These verses describing the angel and bloody sweat are absent from several important "
        "early manuscripts. Some scholars suggest they were removed to avoid depicting Jesus in a state of "
        "weakness, while others argue they were added to enhance the Gethsemane scene. Textual evidence is "
        "closely divided.",
        "significance": "major",
        "confidence": 0.60,
    },
    # John 5:3b-4 — Angel stirring the water at Bethesda
    {
        "book_id": 43,
        "chapter": 5,
        "verse": 3,
        "variant_type": "addition",
        "base_text": "[Explanation about the angel absent from earliest manuscripts]",
        "variant_text": "...waiting for the moving of the waters. For an angel of the Lord went down at certain "
        "times into the pool and stirred up the water; then whoever stepped in first after the stirring of "
        "the water was made well of whatever disease he had.",
        "manuscripts": "P66, P75, Sinaiticus, Vaticanus, C, early versions omit; A, later Byzantine include",
        "explanation": "The explanation about an angel stirring the pool is absent from all the earliest "
        "manuscripts. It was likely added by scribes to explain why the sick man was waiting at the pool "
        "and why only the first person in was healed.",
        "significance": "major",
        "confidence": 0.95,
    },
    # Acts 8:37 — Ethiopian eunuch's confession
    {
        "book_id": 44,
        "chapter": 8,
        "verse": 37,
        "variant_type": "addition",
        "base_text": "[Verse absent from earliest manuscripts]",
        "variant_text": "And Philip said, 'If you believe with all your heart, you may.' And he answered "
        "and said, 'I believe that Jesus Christ is the Son of God.'",
        "manuscripts": "P45, P74, Sinaiticus, Vaticanus, A, C omit; E, later manuscripts include",
        "explanation": "This baptismal confession is absent from the earliest manuscripts. It was likely "
        "added to provide an explicit confession of faith before baptism, reflecting later liturgical "
        "practice of requiring a profession of faith before the rite.",
        "significance": "minor",
        "confidence": 0.95,
    },
    # Mark 1:1 — "Son of God"
    {
        "book_id": 41,
        "chapter": 1,
        "verse": 1,
        "variant_type": "omission",
        "base_text": "The beginning of the gospel of Jesus Christ, the Son of God.",
        "variant_text": "The beginning of the gospel of Jesus Christ.",
        "manuscripts": "Sinaiticus (first hand), Theta, some Old Latin omit 'Son of God'; Vaticanus, A, D, W include",
        "explanation": "The phrase 'Son of God' (huiou theou) is absent from Sinaiticus and a few other "
        "witnesses. While the omission could be accidental (homoeoteleuton), the shorter reading may be "
        "original. Most scholars favor inclusion based on broader attestation and Mark's theological emphasis.",
        "significance": "major",
        "confidence": 0.55,
    },
    # Luke 23:34a — "Father, forgive them"
    {
        "book_id": 42,
        "chapter": 23,
        "verse": 34,
        "variant_type": "omission",
        "base_text": "And Jesus said, 'Father, forgive them, for they do not know what they are doing.'",
        "variant_text": "[Words of Jesus omitted]",
        "manuscripts": "P75, Sinaiticus (first hand), Vaticanus, D omit; Sinaiticus (corrector), A, C include",
        "explanation": "The prayer for forgiveness is absent from several early and diverse manuscripts. "
        "Some scholars suggest it was removed by scribes who felt it was inappropriate after the destruction "
        "of Jerusalem (70 AD), while others argue it was added to conform to the pattern of Stephen's "
        "prayer in Acts 7:60.",
        "significance": "major",
        "confidence": 0.55,
    },
    # Matthew 17:21 — Fasting and prayer
    {
        "book_id": 40,
        "chapter": 17,
        "verse": 21,
        "variant_type": "addition",
        "base_text": "[Verse absent from earliest manuscripts]",
        "variant_text": "However, this kind does not go out except by prayer and fasting.",
        "manuscripts": "Sinaiticus, Vaticanus, Theta omit; C, D, K, later Byzantine include",
        "explanation": "This verse is absent from the earliest manuscripts of Matthew. It appears to be an "
        "assimilation from Mark 9:29, where a similar saying is better attested. Scribes likely added it "
        "to harmonize the Synoptic accounts.",
        "significance": "minor",
        "confidence": 0.90,
    },
    # Matthew 18:11 — Son of Man came to save the lost
    {
        "book_id": 40,
        "chapter": 18,
        "verse": 11,
        "variant_type": "addition",
        "base_text": "[Verse absent from earliest manuscripts]",
        "variant_text": "For the Son of Man came to save the lost.",
        "manuscripts": "Sinaiticus, Vaticanus, L, early versions omit; D, K, later Byzantine include",
        "explanation": "This verse is absent from the best early manuscripts. It is a scribal importation "
        "from Luke 19:10, inserted here because of the thematic connection with the parable of the lost "
        "sheep in the preceding verses.",
        "significance": "minor",
        "confidence": 0.90,
    },
    # Matthew 23:14 — Woe to scribes and Pharisees (devouring widows' houses)
    {
        "book_id": 40,
        "chapter": 23,
        "verse": 14,
        "variant_type": "addition",
        "base_text": "[Verse absent from earliest manuscripts]",
        "variant_text": "Woe to you, scribes and Pharisees, hypocrites! For you devour widows' houses, and "
        "for a pretense make long prayers. Therefore you will receive greater condemnation.",
        "manuscripts": "Sinaiticus, Vaticanus, D, L omit; later Byzantine manuscripts include",
        "explanation": "This woe oracle is absent from the earliest manuscripts. It was likely imported from "
        "Mark 12:40 or Luke 20:47 to create a fuller series of seven (or eight) woes. Different manuscripts "
        "place it before or after v.13.",
        "significance": "minor",
        "confidence": 0.90,
    },
    # Acts 15:34 — Silas remained
    {
        "book_id": 44,
        "chapter": 15,
        "verse": 34,
        "variant_type": "addition",
        "base_text": "[Verse absent from earliest manuscripts]",
        "variant_text": "But it seemed good to Silas to remain there.",
        "manuscripts": "P74, Sinaiticus, Vaticanus, A omit; C, D, later manuscripts include",
        "explanation": "This verse was likely added by scribes to explain how Silas was available in v.40 "
        "when Paul chose him as a companion. The earliest manuscripts do not include it, and Luke may not "
        "have felt the need to explain Silas's continued presence in Antioch.",
        "significance": "minor",
        "confidence": 0.90,
    },
    # Acts 24:6b-8a — Lysias intervening
    {
        "book_id": 44,
        "chapter": 24,
        "verse": 6,
        "variant_type": "addition",
        "base_text": "...and we seized him.",
        "variant_text": "...and we seized him, and we would have judged him according to our law. But the "
        "commander Lysias came and with great violence took him out of our hands, commanding his accusers "
        "to come before you.",
        "manuscripts": "P74, Sinaiticus, Vaticanus, A omit; E, later Byzantine manuscripts include",
        "explanation": "The longer text about Lysias's intervention is part of the Western text tradition. "
        "It provides additional narrative detail about the tribune's role but is absent from the earliest "
        "Alexandrian manuscripts. The shorter text is likely original.",
        "significance": "minor",
        "confidence": 0.85,
    },
]


class ManuscriptVariantPopulator:
    """Populates the manuscript_variants table with textual criticism data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _ensure_table(self, cursor: sqlite3.Cursor) -> bool:
        """Create the manuscript_variants table if it does not exist.

        Returns:
            True if table is ready, False otherwise.
        """
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS manuscript_variants (
                variant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                chapter INTEGER NOT NULL,
                verse INTEGER NOT NULL,
                variant_type TEXT NOT NULL,
                base_text TEXT,
                variant_text TEXT,
                manuscripts TEXT,
                explanation TEXT,
                significance TEXT NOT NULL DEFAULT 'minor',
                confidence REAL DEFAULT 0.5,
                UNIQUE(book_id, chapter, verse, variant_type, variant_text)
            )
            """
        )
        return True

    def populate(self, force: bool = False) -> int:
        """Insert manuscript variants into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            self._ensure_table(cursor)

            if force:
                cursor.execute("DELETE FROM manuscript_variants")

            for variant in MANUSCRIPT_VARIANTS:
                try:
                    verb = "INSERT OR REPLACE" if force else "INSERT OR IGNORE"
                    cursor.execute(
                        f"{verb} INTO manuscript_variants "  # noqa: S608
                        "(book_id, chapter, verse, variant_type, base_text, variant_text, "
                        "manuscripts, explanation, significance, confidence) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            variant["book_id"],
                            variant["chapter"],
                            variant["verse"],
                            variant["variant_type"],
                            variant["base_text"],
                            variant["variant_text"],
                            variant["manuscripts"],
                            variant["explanation"],
                            variant["significance"],
                            variant["confidence"],
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error(
                        "Failed to insert variant for book %d %d:%d: %s",
                        variant["book_id"],
                        variant["chapter"],
                        variant["verse"],
                        e,
                    )

            conn.commit()

        logger.info("Populated manuscript_variants: %d rows inserted", inserted)
        return inserted
