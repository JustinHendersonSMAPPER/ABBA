"""Cultural context population for book-level introductions and annotations."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Curated book-level cultural context introductions.
# Format: (book_id, context_type, title, summary, detailed_content, time_period, geographic_region, confidence)
BOOK_INTRODUCTIONS: List[Tuple[int, str, str, str, str, str, str, str]] = [
    (
        1,
        "historical_background",
        "Genesis: Origins and Patriarchs",
        "Genesis covers creation through the patriarchal period, foundational to Israelite identity.",
        "Written in the ancient Near Eastern context where creation accounts were common. "
        "The patriarchal narratives reflect semi-nomadic life in the Fertile Crescent circa 2000-1800 BCE. "
        "Covenant-making ceremonies mirror Hittite suzerainty treaties. "
        "Understanding ANE parallels (Enuma Elish, Atrahasis) illuminates the distinctive theology.",
        "2000-1400 BCE",
        "Mesopotamia, Canaan, Egypt",
        "high",
    ),
    (
        2,
        "historical_background",
        "Exodus: Deliverance and Law",
        "Exodus narrates Israel's liberation from Egypt and covenant at Sinai.",
        "Set during the Late Bronze Age when Egypt controlled Canaan. "
        "The ten plagues directly challenged specific Egyptian deities. "
        "The covenant structure parallels ancient suzerainty treaties. "
        "The tabernacle design reflects ANE sacred space concepts.",
        "1446-1406 BCE or 1290-1250 BCE",
        "Egypt, Sinai Peninsula",
        "high",
    ),
    (
        3,
        "historical_background",
        "Leviticus: Holiness and Worship",
        "Leviticus provides Israel's worship system — sacrifices, purity, and priestly duties.",
        "Holiness meant separation for God's purposes. The sacrificial system used blood as a symbol "
        "of life given to atone. Purity laws distinguished Israel from surrounding cultures. "
        "The Day of Atonement (ch. 16) was the annual ritual for national cleansing.",
        "1446-1406 BCE",
        "Sinai Wilderness",
        "high",
    ),
    (
        5,
        "historical_background",
        "Deuteronomy: Covenant Renewal",
        "Moses' farewell speeches renewing the covenant before entering the Promised Land.",
        "Structured as an ancient Near Eastern treaty: preamble, historical prologue, stipulations, "
        "blessings/curses, and witnesses. The Shema (6:4-9) became Israel's central confession. "
        "The book addresses a transition generation preparing for settled agricultural life.",
        "1406 BCE",
        "Plains of Moab",
        "high",
    ),
    (
        18,
        "historical_background",
        "Job: Suffering and Divine Justice",
        "Job explores why the righteous suffer, challenging retribution theology.",
        "Set in the patriarchal period in the land of Uz (likely Edom). "
        "The dialogue form resembles ancient Near Eastern wisdom debates. "
        "Job's friends represent orthodox retribution theology — suffering proves sin. "
        "God's speeches from the whirlwind reframe the question entirely.",
        "Patriarchal period (setting); writing date debated",
        "Land of Uz",
        "medium",
    ),
    (
        19,
        "historical_background",
        "Psalms: Israel's Songbook",
        "The Psalms are Israel's collected prayers, praises, and laments spanning centuries.",
        "Psalms were sung in temple worship with musical accompaniment. "
        "Major types: hymns of praise, individual/communal laments, thanksgiving, royal, wisdom, "
        "and pilgrimage psalms. Hebrew poetry uses parallelism rather than rhyme. "
        "Many psalms have historical superscriptions linking them to David's life events.",
        "1000-400 BCE",
        "Jerusalem, various",
        "high",
    ),
    (
        20,
        "historical_background",
        "Proverbs: Wisdom for Daily Life",
        "Proverbs collects practical wisdom for living well in God's ordered world.",
        "Part of the ancient Near Eastern wisdom tradition (cf. Egyptian Instruction of Amenemope). "
        "Proverbs are general principles, not absolute promises. "
        "Lady Wisdom (chs. 1-9) personifies God's ordering principle in creation. "
        "The fear of the Lord is the foundational principle (1:7).",
        "950-700 BCE",
        "Jerusalem",
        "high",
    ),
    (
        23,
        "historical_background",
        "Isaiah: Prophet to Judah",
        "Isaiah prophesied during Assyria's rise, calling Judah to trust God alone.",
        "Isaiah ministered across four kings' reigns (740-680 BCE). "
        "Chapters 1-39 address the Assyrian crisis; 40-66 look beyond exile to restoration. "
        "The Suffering Servant songs (42, 49, 50, 52-53) are central to Christian interpretation. "
        "Isaiah's vision of universal salvation extends beyond Israel to all nations.",
        "740-680 BCE",
        "Jerusalem, Judah",
        "high",
    ),
    (
        40,
        "historical_background",
        "Matthew: Jesus as Jewish Messiah",
        "Matthew presents Jesus as the promised Messiah fulfilling Hebrew Scripture.",
        "Written primarily for Jewish Christians. Uses formula quotations ('this was to fulfill...') "
        "to connect Jesus to OT prophecy. Structures Jesus' teaching in five major discourses "
        "(parallel to the five books of Moses). Emphasizes the 'kingdom of heaven.' "
        "The Sermon on the Mount (chs. 5-7) presents Jesus as the authoritative interpreter of Torah.",
        "50-70 CE",
        "Antioch (likely)",
        "high",
    ),
    (
        41,
        "historical_background",
        "Mark: The Suffering Servant",
        "Mark is the earliest Gospel, emphasizing Jesus' actions and the cost of discipleship.",
        "Written for Roman Christians during persecution. Fast-paced narrative uses 'immediately' frequently. "
        "The 'Messianic Secret' — Jesus repeatedly tells people not to reveal his identity. "
        "The second half focuses on Jesus' journey to the cross. "
        "Discipleship means taking up one's cross (8:34).",
        "65-70 CE",
        "Rome (likely)",
        "high",
    ),
    (
        42,
        "historical_background",
        "Luke: Universal Savior",
        "Luke presents Jesus as savior for all people — Gentiles, women, poor, and marginalized.",
        "Written by a physician and travel companion of Paul. Unique parables: Good Samaritan, "
        "Prodigal Son, Rich Man and Lazarus. Emphasizes the Holy Spirit, prayer, and joy. "
        "Jesus' inaugural sermon in Nazareth (4:16-30) programmatically announces his mission. "
        "A two-volume work with Acts.",
        "60-80 CE",
        "Unknown (for Theophilus)",
        "high",
    ),
    (
        43,
        "historical_background",
        "John: The Divine Word",
        "John presents Jesus as the pre-existent Word of God through seven signs and discourses.",
        "Distinct from the Synoptics in structure, vocabulary, and chronology. "
        "Seven 'I am' statements reveal Jesus' identity (bread of life, light of the world, etc.). "
        "Seven signs (miracles) demonstrate his divine authority. "
        "The farewell discourse (chs. 13-17) is unique to John. "
        "Themes: light/darkness, belief/unbelief, above/below, life/death.",
        "85-95 CE",
        "Ephesus (tradition)",
        "high",
    ),
    (
        44,
        "historical_background",
        "Acts: The Early Church",
        "Acts narrates the Spirit-empowered expansion of the church from Jerusalem to Rome.",
        "Continues Luke's Gospel. Pentecost inaugurates the church age. "
        "The narrative follows two main figures: Peter (chs. 1-12) and Paul (chs. 13-28). "
        "Key speeches articulate early Christian theology. "
        "The Jerusalem Council (ch. 15) is pivotal for Gentile inclusion.",
        "30-62 CE (events); 62-80 CE (composition)",
        "Jerusalem to Rome",
        "high",
    ),
    (
        45,
        "historical_background",
        "Romans: The Gospel Explained",
        "Paul's most systematic letter explaining justification by faith and life in the Spirit.",
        "Written to a church Paul had not yet visited. Addresses Jew-Gentile tensions. "
        "Chapters 1-8: theological argument (sin, justification, sanctification, glorification). "
        "Chapters 9-11: Israel's role in God's plan. Chapters 12-16: practical ethics. "
        "The 'Romans Road' traces the logic of salvation.",
        "57 CE",
        "Corinth (written to Rome)",
        "high",
    ),
    (
        58,
        "historical_background",
        "Hebrews: Christ Superior to All",
        "Hebrews argues for Christ's superiority over angels, Moses, and the old covenant priesthood.",
        "Written to Jewish Christians tempted to return to Judaism. "
        "Uses extensive OT quotation and typological interpretation. "
        "Christ is the ultimate high priest after the order of Melchizedek. "
        "The 'Hall of Faith' (ch. 11) surveys OT examples of faithfulness.",
        "60-70 CE",
        "Unknown (possibly Rome)",
        "medium",
    ),
    (
        66,
        "historical_background",
        "Revelation: Apocalyptic Hope",
        "Revelation unveils God's ultimate victory over evil through vivid symbolic imagery.",
        "Written during Roman persecution (likely Domitian, 81-96 CE). "
        "Apocalyptic literature uses symbolic numbers, colors, and creatures. "
        "The number 7 (completeness) structures the book: 7 churches, seals, trumpets, bowls. "
        "Not primarily a timeline but a theological vision of God's sovereignty. "
        "Ends with the new creation — God dwelling with humanity forever.",
        "90-96 CE",
        "Patmos (written to seven churches in Asia Minor)",
        "high",
    ),
    # --- Remaining OT Books ---
    (
        4,
        "historical_background",
        "Numbers: Wilderness Wanderings",
        "Numbers records Israel's forty-year journey from Sinai to the edge of the Promised Land.",
        "Two censuses bookend the narrative — the first generation that refused to enter Canaan and the "
        "second that would inherit it. The book illustrates the consequences of unbelief (chs. 13-14). "
        "Balaam's oracles (chs. 22-24) come from a pagan prophet compelled by God to bless Israel.",
        "1446-1406 BCE",
        "Sinai Wilderness, Transjordan",
        "high",
    ),
    (
        6,
        "historical_background",
        "Joshua: Conquest and Settlement",
        "Joshua narrates Israel's entry into and settlement of the Promised Land under Joshua's leadership.",
        "Set in the Late Bronze Age during a period of political instability in Canaan. "
        "The conquest narrative demonstrates God's faithfulness to the Abrahamic land promise. "
        "The covenant renewal at Shechem (ch. 24) mirrors ancient treaty ceremonies.",
        "1406-1375 BCE",
        "Canaan",
        "high",
    ),
    (
        7,
        "historical_background",
        "Judges: Cycles of Apostasy",
        "Judges describes a dark period of repeated apostasy, oppression, and deliverance.",
        "The recurring cycle — sin, servitude, supplication, salvation — reveals Israel's unfaithfulness. "
        "The refrain 'everyone did what was right in his own eyes' captures the era's moral chaos. "
        "Judge figures were military deliverers, not courtroom officials.",
        "1375-1050 BCE",
        "Canaan tribal territories",
        "high",
    ),
    (
        8,
        "historical_background",
        "Ruth: Loyalty and Redemption",
        "Ruth tells of a Moabite woman's devoted loyalty, set during the period of the Judges.",
        "The kinsman-redeemer (goel) institution is central — the nearest relative could buy back "
        "family property and marry a widow to preserve the family line. Ruth's inclusion in the "
        "Davidic lineage (and thus Jesus') underscores God's universal purposes.",
        "1100 BCE (setting)",
        "Moab and Bethlehem",
        "high",
    ),
    (
        9,
        "historical_background",
        "1 Samuel: Monarchy's Beginning",
        "1 Samuel covers the transition from judges to monarchy — Samuel, Saul, and David's rise.",
        "Israel's demand for a king 'like the nations' marks a theological watershed. "
        "Samuel serves as the last judge and first kingmaker. "
        "Saul's reign demonstrates the danger of human kingship apart from divine obedience.",
        "1050-1010 BCE",
        "Israel, primarily Benjamin and Judah",
        "high",
    ),
    (
        10,
        "historical_background",
        "2 Samuel: David's Reign",
        "2 Samuel recounts David's reign — his triumphs, failures, and lasting covenant promise.",
        "The Davidic Covenant (ch. 7) promises an eternal dynasty, foundational to messianic expectation. "
        "David's sin with Bathsheba (chs. 11-12) triggers family dysfunction that shapes the rest of the book. "
        "David is portrayed honestly as both heroic and deeply flawed.",
        "1010-970 BCE",
        "Israel, Jerusalem",
        "high",
    ),
    (
        11,
        "historical_background",
        "1 Kings: Kingdom United and Divided",
        "1 Kings covers Solomon's glorious reign and the kingdom's tragic division.",
        "Solomon's temple represents the apex of Israelite worship. "
        "The kingdom splits under Rehoboam due to heavy taxation and forced labour. "
        "Elijah's contest on Mount Carmel (ch. 18) confronts Baal worship head-on.",
        "970-850 BCE",
        "Israel and Judah",
        "high",
    ),
    (
        12,
        "historical_background",
        "2 Kings: Exile Approaches",
        "2 Kings traces both kingdoms' decline to exile — Israel (722 BCE) and Judah (586 BCE).",
        "Elisha continues Elijah's prophetic ministry with signs and wonders. "
        "Kings are evaluated by faithfulness to Yahweh and the Deuteronomic covenant. "
        "Josiah's reform (chs. 22-23) provides a brief revival before Babylon's conquest.",
        "850-586 BCE",
        "Israel, Judah, Assyria, Babylon",
        "high",
    ),
    (
        13,
        "historical_background",
        "1 Chronicles: David and Temple Worship",
        "1 Chronicles retells Israel's history emphasizing David's role in establishing temple worship.",
        "Written for the post-exilic community to reconnect with their heritage. "
        "Genealogies (chs. 1-9) establish continuity. David's preparations for the temple "
        "and Levitical worship occupy the bulk of the narrative.",
        "Post-exilic (450-400 BCE composition)",
        "Israel, Jerusalem",
        "medium",
    ),
    (
        14,
        "historical_background",
        "2 Chronicles: Temple and Reform",
        "2 Chronicles focuses on Judah's kings, the temple, and periods of spiritual reform.",
        "Ignores the northern kingdom almost entirely, focusing on the Davidic line. "
        "Highlights reformer-kings: Asa, Jehoshaphat, Hezekiah, Josiah. "
        "Ends with Cyrus's decree to rebuild the temple — hope beyond exile.",
        "Post-exilic (450-400 BCE composition)",
        "Judah, Jerusalem",
        "medium",
    ),
    (
        15,
        "historical_background",
        "Ezra: Return and Restoration",
        "Ezra records the Jewish return from Babylonian exile and rebuilding of the temple.",
        "Cyrus's edict (538 BCE) fulfils Jeremiah's prophecy. The rebuilt temple was modest "
        "compared to Solomon's, yet represented God's continued presence. "
        "Ezra's reform addressed intermarriage with surrounding peoples.",
        "538-458 BCE",
        "Babylon and Jerusalem",
        "high",
    ),
    (
        16,
        "historical_background",
        "Nehemiah: Rebuilding the Walls",
        "Nehemiah leads the rebuilding of Jerusalem's walls and covenant renewal.",
        "Nehemiah served as cupbearer to Persian King Artaxerxes before requesting "
        "permission to rebuild. The wall was completed in 52 days despite opposition. "
        "The covenant renewal ceremony (chs. 8-10) centered on public Torah reading.",
        "445-430 BCE",
        "Jerusalem, under Persian rule",
        "high",
    ),
    (
        17,
        "historical_background",
        "Esther: Providence in Exile",
        "Esther reveals God's hidden providence protecting Jews in the Persian Empire.",
        "Set in the Persian court at Susa. God is never mentioned by name, yet divine "
        "providence pervades the narrative. The Festival of Purim celebrates this deliverance. "
        "Reversal is a key literary motif — intended victims become victors.",
        "486-465 BCE",
        "Susa, Persian Empire",
        "high",
    ),
    (
        21,
        "historical_background",
        "Ecclesiastes: Life Under the Sun",
        "Ecclesiastes wrestles honestly with life's apparent meaninglessness apart from God.",
        "The Teacher (Qoheleth) systematically examines work, pleasure, wisdom, and wealth, "
        "finding all 'hebel' (vapour/fleeting). Not nihilism but honest realism. "
        "The conclusion (12:13-14) anchors meaning in fearing God and keeping his commands.",
        "950 BCE (Solomonic tradition)",
        "Jerusalem",
        "medium",
    ),
    (
        22,
        "historical_background",
        "Song of Solomon: Love Poetry",
        "The Song celebrates romantic love between a man and woman in vivid poetic imagery.",
        "Ancient Near Eastern love poetry tradition. Metaphors drawn from nature, agriculture, "
        "and spices. Read literally as celebrating marital love, allegorically as God-Israel or "
        "Christ-Church relationship. The repeated refrain 'do not arouse love until it so desires' "
        "structures the poem.",
        "950 BCE (tradition)",
        "Israel, Lebanon",
        "medium",
    ),
    (
        24,
        "historical_background",
        "Jeremiah: Prophet of Judgment and Hope",
        "Jeremiah prophesied during Judah's final decades, warning of Babylon's coming destruction.",
        "Called as a young man (627 BCE), Jeremiah preached for 40 years through national catastrophe. "
        "The 'weeping prophet' suffered rejection, imprisonment, and exile. "
        "The New Covenant prophecy (31:31-34) is foundational for Christian theology.",
        "627-586 BCE",
        "Jerusalem, Egypt",
        "high",
    ),
    (
        25,
        "historical_background",
        "Lamentations: Grief over Jerusalem",
        "Lamentations is a collection of funeral poems mourning Jerusalem's destruction in 586 BCE.",
        "Five acrostic poems following the Hebrew alphabet — structured grief. "
        "Raw, honest anguish over destruction coexists with affirmations of God's faithfulness (3:22-23). "
        "Used liturgically on the ninth of Av to commemorate the temple's destruction.",
        "586-580 BCE",
        "Jerusalem (destroyed)",
        "high",
    ),
    (
        26,
        "historical_background",
        "Ezekiel: Visions of Exile and Restoration",
        "Ezekiel prophesied among the Babylonian exiles, combining bizarre visions with pastoral care.",
        "Ezekiel was a priest called to prophesy in exile. His sign-acts dramatised judgment. "
        "The departure of God's glory from the temple (chs. 8-11) and its return (ch. 43) "
        "frame the book. The valley of dry bones (ch. 37) promises national restoration.",
        "593-571 BCE",
        "Babylon (Tel-Abib by the Chebar canal)",
        "high",
    ),
    (
        27,
        "historical_background",
        "Daniel: Faithfulness in Exile",
        "Daniel combines court narratives of faithfulness with apocalyptic visions of God's kingdom.",
        "Set in the Babylonian and Persian courts. The fiery furnace and lions' den demonstrate "
        "that God protects the faithful even in hostile empires. "
        "Apocalyptic visions (chs. 7-12) use symbolic beasts to represent successive empires.",
        "605-535 BCE",
        "Babylon, Persia",
        "high",
    ),
    (
        28,
        "historical_background",
        "Hosea: Unfaithful Love",
        "Hosea's marriage to an unfaithful wife embodies God's persistent love for unfaithful Israel.",
        "Hosea prophesied to the northern kingdom during its final, prosperous-but-corrupt decades. "
        "Religious syncretism — mixing Yahweh worship with Baal fertility religion — was the core problem. "
        "Despite devastating judgment, restoration is promised (ch. 14).",
        "750-715 BCE",
        "Northern Israel",
        "high",
    ),
    (
        29,
        "historical_background",
        "Joel: The Day of the Lord",
        "Joel uses a devastating locust plague as a lens for understanding the coming Day of the Lord.",
        "The locust plague served as both historical calamity and prophetic metaphor. "
        "The call to national repentance (2:12-13) emphasises internal transformation. "
        "The promise of the Spirit's outpouring (2:28-32) is quoted at Pentecost (Acts 2).",
        "835-400 BCE (debated)",
        "Judah, Jerusalem",
        "medium",
    ),
    (
        30,
        "historical_background",
        "Amos: Social Justice",
        "Amos is a shepherd from Judah sent to confront the northern kingdom's social injustice.",
        "Preached during a time of prosperity under Jeroboam II — but prosperity masked oppression. "
        "The rich exploited the poor through rigged scales and corrupt courts. "
        "God's demand: 'Let justice roll on like a river' (5:24).",
        "760-750 BCE",
        "Northern Israel (Bethel)",
        "high",
    ),
    (
        31,
        "historical_background",
        "Obadiah: Edom's Judgment",
        "Obadiah, the shortest OT book, pronounces judgment on Edom for betraying Judah.",
        "Edomites (descendants of Esau) gloated over Jerusalem's fall and helped plunder it. "
        "The ancient sibling rivalry (Jacob/Esau) finds its prophetic resolution.",
        "586-550 BCE",
        "Edom (southeast of the Dead Sea)",
        "medium",
    ),
    (
        32,
        "historical_background",
        "Jonah: Reluctant Prophet",
        "Jonah is a satirical narrative about a prophet who runs from God's compassion for enemies.",
        "Nineveh was the capital of Assyria — Israel's brutal oppressor. "
        "Jonah's reluctance exposes nationalistic theology that limits God's mercy. "
        "The book challenges readers to embrace God's compassion for all peoples.",
        "8th century BCE (setting)",
        "Israel, Mediterranean Sea, Nineveh (Assyria)",
        "high",
    ),
    (
        33,
        "historical_background",
        "Micah: Justice, Mercy, and Humility",
        "Micah addresses social injustice in Judah and envisions a future ruler from Bethlehem.",
        "A contemporary of Isaiah, Micah prophesied during the Assyrian crisis. "
        "Micah 6:8 summarises prophetic ethics: act justly, love mercy, walk humbly with God. "
        "The Bethlehem prophecy (5:2) is quoted in Matthew's birth narrative.",
        "735-700 BCE",
        "Judah (Moresheth)",
        "high",
    ),
    (
        34,
        "historical_background",
        "Nahum: Nineveh's Fall",
        "Nahum celebrates the imminent fall of Nineveh, the brutal Assyrian capital.",
        "Where Jonah proclaimed mercy to Nineveh, Nahum announces judgment on its unchanged brutality. "
        "Nineveh fell in 612 BCE to a Babylonian-Median coalition. "
        "The book affirms God's justice against oppressive empires.",
        "663-612 BCE",
        "Judah (regarding Nineveh)",
        "medium",
    ),
    (
        35,
        "historical_background",
        "Habakkuk: Questioning God",
        "Habakkuk dares to question God about injustice and receives a challenging answer.",
        "A dialogue between prophet and God. Why does God tolerate Judah's wickedness? "
        "God's answer — Babylon will punish Judah — raises a harder question: "
        "how can God use someone even more wicked? The answer: 'the righteous shall live by faith' (2:4).",
        "610-600 BCE",
        "Judah",
        "high",
    ),
    (
        36,
        "historical_background",
        "Zephaniah: Day of Judgment and Renewal",
        "Zephaniah warns of universal judgment while promising a humble, purified remnant.",
        "Prophesied during Josiah's reign, before his reform took effect. "
        "The Day of the Lord is described in cosmic terms — silence before God (1:7). "
        "The remnant promises (3:12-13) envision a meek and humble people trusting in the Lord.",
        "640-625 BCE",
        "Judah, Jerusalem",
        "medium",
    ),
    (
        37,
        "historical_background",
        "Haggai: Rebuild the Temple",
        "Haggai urges the returned exiles to stop neglecting the temple and rebuild it.",
        "After returning from exile, the people rebuilt their own houses while the temple lay in ruins. "
        "Haggai links their economic struggles to misplaced priorities. "
        "His four messages in 520 BCE spurred completion of the Second Temple.",
        "520 BCE",
        "Jerusalem (post-exilic)",
        "high",
    ),
    (
        38,
        "historical_background",
        "Zechariah: Visions of Restoration",
        "Zechariah combines apocalyptic visions with messianic prophecy pointing to God's future king.",
        "A contemporary of Haggai, encouraging temple rebuilding. "
        "Eight night visions (chs. 1-6) assure God's commitment to Jerusalem. "
        "Messianic prophecies include the humble king on a donkey (9:9) "
        "and the pierced one mourned by all (12:10).",
        "520-480 BCE",
        "Jerusalem (post-exilic)",
        "high",
    ),
    (
        39,
        "historical_background",
        "Malachi: Final OT Prophet",
        "Malachi confronts a spiritually apathetic post-exilic community through dialogue format.",
        "The temple was rebuilt but worship had become perfunctory — blemished offerings, "
        "unfaithful priests, and withheld tithes. God says 'I have loved you' (1:2) but the "
        "people ask 'How?' The book ends looking forward to Elijah's return before the Day of the Lord.",
        "460-430 BCE",
        "Jerusalem (post-exilic)",
        "high",
    ),
    # --- Remaining NT Epistles ---
    (
        46,
        "historical_background",
        "1 Corinthians: Church Problems",
        "Paul addresses divisions, immorality, and theological confusion in the Corinthian church.",
        "Corinth was a wealthy, cosmopolitan Roman city known for moral laxity. "
        "The church reflected its culture: factions, lawsuits, sexual immorality, and confusion "
        "about spiritual gifts. Chapter 13 (love) and chapter 15 (resurrection) are theological peaks.",
        "55 CE",
        "Ephesus (written to Corinth)",
        "high",
    ),
    (
        47,
        "historical_background",
        "2 Corinthians: Apostolic Defence",
        "Paul defends his ministry and reveals strength through weakness.",
        "The most personal of Paul's letters. Rival teachers questioned his authority. "
        "Paul's 'thorn in the flesh' (12:7-10) demonstrates God's power in human weakness. "
        "Contains the most extended NT teaching on generous giving (chs. 8-9).",
        "55-56 CE",
        "Macedonia (written to Corinth)",
        "high",
    ),
    (
        48,
        "historical_background",
        "Galatians: Freedom in Christ",
        "Paul passionately argues that Gentile believers need not follow Jewish law for salvation.",
        "Written to churches in central or southern Asia Minor. Rival teachers insisted on circumcision. "
        "Paul's argument: justification by faith alone, not law-keeping. "
        "The fruit of the Spirit (5:22-23) describes the character of Spirit-led life.",
        "48-55 CE (debated)",
        "Galatia (Asia Minor)",
        "high",
    ),
    (
        49,
        "historical_background",
        "Ephesians: The Church Universal",
        "Ephesians presents a cosmic vision of the church as Christ's body, uniting Jew and Gentile.",
        "Likely a circular letter to multiple churches. The first half is doctrinal (identity in Christ); "
        "the second half is practical (how to live it out). The armour of God passage (6:10-20) "
        "describes spiritual warfare. Emphasises grace, unity, and the mystery of the gospel.",
        "60-62 CE",
        "Rome (written to Ephesus and region)",
        "high",
    ),
    (
        50,
        "historical_background",
        "Philippians: Joy in All Circumstances",
        "Paul writes joyfully from prison, encouraging the Philippian church to rejoice always.",
        "Philippi was a Roman colony; the church was Paul's first in Europe (Acts 16). "
        "The Christ Hymn (2:5-11) is a key Christological text: incarnation, humiliation, exaltation. "
        "Paul models contentment in every circumstance (4:11-13).",
        "60-62 CE",
        "Rome (written to Philippi)",
        "high",
    ),
    (
        51,
        "historical_background",
        "Colossians: Christ Supreme",
        "Paul counters false teaching by exalting Christ's supremacy over all creation and powers.",
        "The Colossian heresy combined Jewish legalism, proto-gnostic philosophy, and angel worship. "
        "The Christ Hymn (1:15-20) affirms Christ as creator, sustainer, and reconciler of all things. "
        "In Christ 'all the fullness of the deity dwells in bodily form' (2:9).",
        "60-62 CE",
        "Rome (written to Colossae)",
        "high",
    ),
    (
        52,
        "historical_background",
        "1 Thessalonians: Christ's Return",
        "Paul's earliest letter encourages a young church enduring persecution and awaiting Christ's return.",
        "Written within months of leaving Thessalonica. Addresses concern about believers who had died "
        "before Christ's return. Paul assures them: the dead in Christ will rise first (4:13-18). "
        "The letter models pastoral encouragement.",
        "50-51 CE",
        "Corinth (written to Thessalonica)",
        "high",
    ),
    (
        53,
        "historical_background",
        "2 Thessalonians: Steadfastness",
        "Paul corrects misunderstandings about Christ's return and urges continued faithfulness.",
        "Some Thessalonians had stopped working, believing Christ's return was imminent. "
        "Paul describes events that must precede the Day of the Lord (2:1-12). "
        "The practical exhortation: 'if anyone will not work, neither shall he eat' (3:10).",
        "51 CE",
        "Corinth (written to Thessalonica)",
        "high",
    ),
    (
        54,
        "historical_background",
        "1 Timothy: Church Leadership",
        "Paul instructs his young delegate Timothy on church order and sound doctrine in Ephesus.",
        "A Pastoral Epistle addressing qualifications for overseers and deacons, "
        "combating false teaching, and caring for widows. Timothy was Paul's trusted representative "
        "in a complex, multi-ethnic urban church.",
        "62-65 CE",
        "Macedonia (written to Ephesus)",
        "medium",
    ),
    (
        55,
        "historical_background",
        "2 Timothy: Final Words",
        "Paul's final letter — a deeply personal charge to Timothy as Paul faces execution.",
        "Written from a Roman prison, likely during Nero's persecution. "
        "Paul reflects on his life: 'I have fought the good fight' (4:7). "
        "Urges Timothy to guard the gospel, endure hardship, and preach the Word.",
        "66-67 CE",
        "Rome",
        "high",
    ),
    (
        56,
        "historical_background",
        "Titus: Church Order in Crete",
        "Paul instructs Titus on organizing churches in Crete — a culture with a rough reputation.",
        "Cretans were stereotyped as liars and gluttons (1:12, quoting a Cretan poet). "
        "The letter emphasises 'good works' as the visible fruit of grace (2:11-14; 3:4-8). "
        "Practical church governance with attention to cultural context.",
        "62-65 CE",
        "Unknown (written to Crete)",
        "medium",
    ),
    (
        57,
        "historical_background",
        "Philemon: Slavery and Brotherhood",
        "Paul appeals to Philemon to receive back his runaway slave Onesimus as a brother in Christ.",
        "The shortest of Paul's letters and the most personal. Does not demand abolition but plants "
        "seeds that would ultimately undermine slavery: if Onesimus is 'no longer a slave but a beloved "
        "brother' (v. 16), the institution's foundations crack.",
        "60-62 CE",
        "Rome (written to Colossae)",
        "high",
    ),
    (
        59,
        "historical_background",
        "James: Faith in Action",
        "James provides practical wisdom for living out faith through actions, not mere words.",
        "Written by Jesus' brother to Jewish Christians scattered abroad. "
        "Strongly influenced by Jewish wisdom tradition and Jesus' Sermon on the Mount. "
        "'Faith without works is dead' (2:26) complements, not contradicts, Paul's justification teaching.",
        "45-49 CE (possibly earliest NT writing)",
        "Jerusalem (written to the diaspora)",
        "high",
    ),
    (
        60,
        "historical_background",
        "1 Peter: Suffering with Hope",
        "Peter encourages persecuted Christians in Asia Minor to stand firm, rooted in resurrection hope.",
        "Addresses 'elect exiles' — a blend of Jewish and Gentile believers facing social marginalisation. "
        "Suffering is framed as participation in Christ's sufferings. "
        "The 'living hope' of resurrection sustains believers through trials (1:3-9).",
        "62-64 CE",
        "Rome (written to Asia Minor)",
        "high",
    ),
    (
        61,
        "historical_background",
        "2 Peter: Guarding Truth",
        "2 Peter warns against false teachers and reaffirms the certainty of Christ's return.",
        "Addresses scoffers who mock the delay of Christ's coming: 'a day is like a thousand years' (3:8). "
        "Urges growth in knowledge and godliness as safeguards against deception.",
        "65-68 CE",
        "Unknown",
        "medium",
    ),
    (
        62,
        "historical_background",
        "1 John: Fellowship and Assurance",
        "1 John combats early proto-gnostic teaching while assuring genuine believers of their salvation.",
        "False teachers denied that Jesus came in physical flesh. "
        "John's tests of authentic faith: believing the incarnation, obeying God's commands, and loving others. "
        "'God is light' (1:5) and 'God is love' (4:8) are central theological affirmations.",
        "85-95 CE",
        "Ephesus (likely)",
        "high",
    ),
    (
        63,
        "historical_background",
        "2 John: Truth and Love",
        "A brief letter warning a church ('chosen lady') against hospitality to false teachers.",
        "Balances love and doctrinal fidelity. Welcoming false teachers who deny Christ's incarnation "
        "would make the host complicit in their error.",
        "85-95 CE",
        "Ephesus (likely)",
        "medium",
    ),
    (
        64,
        "historical_background",
        "3 John: Hospitality and Leadership",
        "3 John commends Gaius for hospitality and condemns Diotrephes for authoritarian control.",
        "A window into early church dynamics: itinerant teachers depended on local hospitality, "
        "and local leaders could either enable or block the gospel's spread.",
        "85-95 CE",
        "Ephesus (likely)",
        "medium",
    ),
    (
        65,
        "historical_background",
        "Jude: Contend for the Faith",
        "Jude urgently warns against false teachers who have infiltrated the church.",
        "Uses vivid OT examples (Sodom, Balaam, Korah) and draws on Jewish apocalyptic traditions "
        "(1 Enoch, Assumption of Moses). The doxology (vv. 24-25) is one of the most beloved in Scripture.",
        "65-80 CE",
        "Unknown",
        "medium",
    ),
]


class CulturalContextPopulator:
    """Populates the cultural_context table with book-level introductions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def populate(self, force: bool = False) -> int:
        """Insert cultural context data into the database.

        Args:
            force: If True, replace existing rows.

        Returns:
            Number of rows inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cultural_context'")
            if cursor.fetchone()[0] == 0:
                logger.warning("cultural_context table does not exist; run migrations first")
                return 0

            if force:
                cursor.execute("DELETE FROM cultural_context WHERE context_type = 'historical_background'")

            for entry in BOOK_INTRODUCTIONS:
                book_id, ctx_type, title, summary, detailed, time_period, geo_region, confidence = entry
                try:
                    # Check for existing entry (no unique constraint on table)
                    cursor.execute(
                        "SELECT COUNT(*) FROM cultural_context WHERE book_id = ? AND context_type = ? AND title = ?",
                        (book_id, ctx_type, title),
                    )
                    if cursor.fetchone()[0] > 0:
                        continue
                    cursor.execute(
                        "INSERT INTO cultural_context "
                        "(book_id, context_type, title, summary, detailed_content, "
                        "time_period, geographic_region, confidence, display_priority) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (book_id, ctx_type, title, summary, detailed, time_period, geo_region, confidence, 1),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.error("Failed to insert cultural context for book %d: %s", book_id, e)

            conn.commit()

        logger.info("Populated cultural_context: %d rows inserted", inserted)
        return inserted

    @staticmethod
    def get_context_for_book(db_path: Path, book_id: int) -> List[Dict[str, Any]]:
        """Get cultural context entries for a book.

        Args:
            db_path: Path to the database.
            book_id: Book ID.

        Returns:
            List of cultural context dictionaries.
        """
        contexts: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_id, context_type, title, summary, detailed_content, "
                "time_period, geographic_region, confidence "
                "FROM cultural_context WHERE book_id = ? ORDER BY display_priority",
                (book_id,),
            )
            for row in cursor.fetchall():
                contexts.append(
                    {
                        "context_id": row[0],
                        "context_type": row[1],
                        "title": row[2],
                        "summary": row[3],
                        "detailed_content": row[4],
                        "time_period": row[5],
                        "geographic_region": row[6],
                        "confidence": row[7],
                    }
                )
        return contexts

    @staticmethod
    def get_context_for_verse(db_path: Path, book_id: int, chapter: int, verse: int) -> List[Dict[str, Any]]:
        """Get cultural context entries applicable to a specific verse.

        Includes both book-level and verse-range entries.

        Args:
            db_path: Path to the database.
            book_id: Book ID.
            chapter: Chapter number.
            verse: Verse number.

        Returns:
            List of cultural context dictionaries.
        """
        contexts: List[Dict[str, Any]] = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Book-level (no chapter/verse specified)
            cursor.execute(
                "SELECT context_id, context_type, title, summary, detailed_content, "
                "time_period, geographic_region, confidence "
                "FROM cultural_context "
                "WHERE book_id = ? AND start_chapter IS NULL "
                "ORDER BY display_priority",
                (book_id,),
            )
            for row in cursor.fetchall():
                contexts.append(
                    {
                        "context_id": row[0],
                        "context_type": row[1],
                        "title": row[2],
                        "summary": row[3],
                        "detailed_content": row[4],
                        "time_period": row[5],
                        "geographic_region": row[6],
                        "confidence": row[7],
                    }
                )

            # Verse-range entries
            cursor.execute(
                "SELECT context_id, context_type, title, summary, detailed_content, "
                "time_period, geographic_region, confidence "
                "FROM cultural_context "
                "WHERE book_id = ? AND start_chapter IS NOT NULL "
                "AND (start_chapter < ? OR (start_chapter = ? AND start_verse <= ?)) "
                "AND (end_chapter > ? OR (end_chapter = ? AND end_verse >= ?)) "
                "ORDER BY display_priority",
                (book_id, chapter, chapter, verse, chapter, chapter, verse),
            )
            for row in cursor.fetchall():
                contexts.append(
                    {
                        "context_id": row[0],
                        "context_type": row[1],
                        "title": row[2],
                        "summary": row[3],
                        "detailed_content": row[4],
                        "time_period": row[5],
                        "geographic_region": row[6],
                        "confidence": row[7],
                    }
                )
        return contexts
