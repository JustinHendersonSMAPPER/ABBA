#!/usr/bin/env python3
"""
Create manual alignment files for high-frequency biblical words.
"""

import json
from pathlib import Path

def create_hebrew_alignments():
    """Create manual alignments for most common Hebrew words."""
    alignments = {
        "H430": {
            "strongs": "H430",
            "lemma": "אֱלֹהִים",
            "translit": "Elohim",
            "primary_translations": ["God", "gods"],
            "all_translations": ["God", "gods", "divine", "deity", "mighty"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Most common word for God in OT (2600+ occurrences)"
        },
        "H3068": {
            "strongs": "H3068",
            "lemma": "יְהוָה",
            "translit": "YHWH/Yahweh",
            "primary_translations": ["LORD", "GOD"],
            "all_translations": ["LORD", "GOD", "Yahweh", "Jehovah", "the Lord"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Divine name, usually rendered LORD in capitals (6800+ occurrences)"
        },
        "H1": {
            "strongs": "H1",
            "lemma": "אָב",
            "translit": "ab",
            "primary_translations": ["father"],
            "all_translations": ["father", "forefather", "ancestor", "patriarch"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Father, ancestor (1200+ occurrences)"
        },
        "H776": {
            "strongs": "H776",
            "lemma": "אֶרֶץ",
            "translit": "erets",
            "primary_translations": ["earth", "land"],
            "all_translations": ["earth", "land", "ground", "country", "world"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Earth or land (2500+ occurrences)"
        },
        "H8064": {
            "strongs": "H8064",
            "lemma": "שָׁמַיִם",
            "translit": "shamayim",
            "primary_translations": ["heaven", "heavens"],
            "all_translations": ["heaven", "heavens", "sky", "air"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Heaven, sky (always plural in Hebrew) (400+ occurrences)"
        },
        "H1254": {
            "strongs": "H1254",
            "lemma": "בָּרָא",
            "translit": "bara",
            "primary_translations": ["create"],
            "all_translations": ["create", "created", "creator", "make"],
            "confidence": 1.0,
            "frequency": "high",
            "notes": "To create, always with God as subject (50+ occurrences)"
        },
        "H559": {
            "strongs": "H559",
            "lemma": "אָמַר",
            "translit": "amar",
            "primary_translations": ["say", "said"],
            "all_translations": ["say", "said", "speak", "spoke", "tell", "told", "command"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To say, speak (5300+ occurrences)"
        },
        "H1697": {
            "strongs": "H1697",
            "lemma": "דָּבָר",
            "translit": "dabar",
            "primary_translations": ["word", "thing"],
            "all_translations": ["word", "thing", "matter", "affair", "message", "speech"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Word, matter, thing (1400+ occurrences)"
        },
        "H3117": {
            "strongs": "H3117",
            "lemma": "יוֹם",
            "translit": "yom",
            "primary_translations": ["day"],
            "all_translations": ["day", "days", "time", "year", "today"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Day, time period (2300+ occurrences)"
        },
        "H5971": {
            "strongs": "H5971",
            "lemma": "עַם",
            "translit": "am",
            "primary_translations": ["people"],
            "all_translations": ["people", "nation", "folk", "men"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "People, nation (1800+ occurrences)"
        },
        "H1121": {
            "strongs": "H1121",
            "lemma": "בֵּן",
            "translit": "ben",
            "primary_translations": ["son"],
            "all_translations": ["son", "child", "children", "descendant", "young"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Son, child (4900+ occurrences)"
        },
        "H376": {
            "strongs": "H376",
            "lemma": "אִישׁ",
            "translit": "ish",
            "primary_translations": ["man"],
            "all_translations": ["man", "husband", "person", "one", "each"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Man, husband, person (2100+ occurrences)"
        },
        "H4428": {
            "strongs": "H4428",
            "lemma": "מֶלֶךְ",
            "translit": "melek",
            "primary_translations": ["king"],
            "all_translations": ["king", "royal", "kingdom"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "King (2500+ occurrences)"
        },
        "H3478": {
            "strongs": "H3478",
            "lemma": "יִשְׂרָאֵל",
            "translit": "Yisrael",
            "primary_translations": ["Israel"],
            "all_translations": ["Israel", "Israelite"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Israel (2500+ occurrences)"
        },
        "H1004": {
            "strongs": "H1004",
            "lemma": "בַּיִת",
            "translit": "bayit",
            "primary_translations": ["house"],
            "all_translations": ["house", "home", "temple", "family", "household"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "House, household, temple (2000+ occurrences)"
        },
        "H3027": {
            "strongs": "H3027",
            "lemma": "יָד",
            "translit": "yad",
            "primary_translations": ["hand"],
            "all_translations": ["hand", "power", "side", "possession"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Hand, power (1600+ occurrences)"
        },
        "H5869": {
            "strongs": "H5869",
            "lemma": "עַיִן",
            "translit": "ayin",
            "primary_translations": ["eye"],
            "all_translations": ["eye", "eyes", "sight", "presence", "spring"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Eye, sight, spring (880+ occurrences)"
        },
        "H3820": {
            "strongs": "H3820",
            "lemma": "לֵב",
            "translit": "leb",
            "primary_translations": ["heart"],
            "all_translations": ["heart", "mind", "understanding", "will"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Heart, mind, will (850+ occurrences)"
        },
        "H6440": {
            "strongs": "H6440",
            "lemma": "פָּנִים",
            "translit": "panim",
            "primary_translations": ["face", "before"],
            "all_translations": ["face", "before", "presence", "countenance", "person"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Face, presence, before (always plural) (2100+ occurrences)"
        },
        "H8085": {
            "strongs": "H8085",
            "lemma": "שָׁמַע",
            "translit": "shama",
            "primary_translations": ["hear"],
            "all_translations": ["hear", "heard", "listen", "obey", "understand"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To hear, listen, obey (1150+ occurrences)"
        },
        "H7200": {
            "strongs": "H7200",
            "lemma": "רָאָה",
            "translit": "raah",
            "primary_translations": ["see"],
            "all_translations": ["see", "saw", "look", "appear", "perceive"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To see, look, perceive (1300+ occurrences)"
        },
        "H3045": {
            "strongs": "H3045",
            "lemma": "יָדַע",
            "translit": "yada",
            "primary_translations": ["know"],
            "all_translations": ["know", "knew", "known", "understand", "perceive"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To know, understand (940+ occurrences)"
        },
        "H5414": {
            "strongs": "H5414",
            "lemma": "נָתַן",
            "translit": "natan",
            "primary_translations": ["give"],
            "all_translations": ["give", "gave", "given", "put", "set", "make"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To give, put, set (2000+ occurrences)"
        },
        "H935": {
            "strongs": "H935",
            "lemma": "בּוֹא",
            "translit": "bo",
            "primary_translations": ["come"],
            "all_translations": ["come", "came", "go", "went", "enter", "bring"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To come, go, enter (2500+ occurrences)"
        },
        "H3318": {
            "strongs": "H3318",
            "lemma": "יָצָא",
            "translit": "yatsa",
            "primary_translations": ["go out"],
            "all_translations": ["go out", "went out", "come out", "depart", "proceed"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To go out, come forth (1000+ occurrences)"
        }
    }
    
    return alignments

def create_greek_alignments():
    """Create manual alignments for most common Greek words."""
    alignments = {
        "G2316": {
            "strongs": "G2316",
            "lemma": "θεός",
            "translit": "theos",
            "primary_translations": ["God"],
            "all_translations": ["God", "god", "divine", "deity"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "God (1300+ occurrences in NT)"
        },
        "G2424": {
            "strongs": "G2424",
            "lemma": "Ἰησοῦς",
            "translit": "Iesous",
            "primary_translations": ["Jesus"],
            "all_translations": ["Jesus"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Jesus (900+ occurrences)"
        },
        "G5547": {
            "strongs": "G5547",
            "lemma": "Χριστός",
            "translit": "Christos",
            "primary_translations": ["Christ"],
            "all_translations": ["Christ", "Messiah", "Anointed"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Christ, Messiah (530+ occurrences)"
        },
        "G2962": {
            "strongs": "G2962",
            "lemma": "κύριος",
            "translit": "kurios",
            "primary_translations": ["Lord"],
            "all_translations": ["Lord", "lord", "master", "sir"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Lord, master (700+ occurrences)"
        },
        "G4151": {
            "strongs": "G4151",
            "lemma": "πνεῦμα",
            "translit": "pneuma",
            "primary_translations": ["spirit", "Spirit"],
            "all_translations": ["spirit", "Spirit", "wind", "breath", "ghost"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Spirit, wind, breath (380+ occurrences)"
        },
        "G3004": {
            "strongs": "G3004",
            "lemma": "λέγω",
            "translit": "lego",
            "primary_translations": ["say"],
            "all_translations": ["say", "said", "speak", "tell", "call"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To say, speak (1300+ occurrences)"
        },
        "G1096": {
            "strongs": "G1096",
            "lemma": "γίνομαι",
            "translit": "ginomai",
            "primary_translations": ["become", "be"],
            "all_translations": ["become", "be", "happen", "come", "arise", "made"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To become, happen, be (670+ occurrences)"
        },
        "G2064": {
            "strongs": "G2064",
            "lemma": "ἔρχομαι",
            "translit": "erchomai",
            "primary_translations": ["come"],
            "all_translations": ["come", "came", "go", "went"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To come, go (630+ occurrences)"
        },
        "G444": {
            "strongs": "G444",
            "lemma": "ἄνθρωπος",
            "translit": "anthropos",
            "primary_translations": ["man"],
            "all_translations": ["man", "human", "person", "people"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Man, human being (550+ occurrences)"
        },
        "G3956": {
            "strongs": "G3956",
            "lemma": "πᾶς",
            "translit": "pas",
            "primary_translations": ["all", "every"],
            "all_translations": ["all", "every", "whole", "everyone", "everything"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "All, every, whole (1200+ occurrences)"
        },
        "G846": {
            "strongs": "G846",
            "lemma": "αὐτός",
            "translit": "autos",
            "primary_translations": ["he", "she", "it"],
            "all_translations": ["he", "she", "it", "self", "same", "they", "them"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "He/she/it, self, same (5500+ occurrences)"
        },
        "G3588": {
            "strongs": "G3588",
            "lemma": "ὁ",
            "translit": "ho",
            "primary_translations": ["the"],
            "all_translations": ["the", "this", "that", "who", "which"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "The definite article (19000+ occurrences)"
        },
        "G2532": {
            "strongs": "G2532",
            "lemma": "καί",
            "translit": "kai",
            "primary_translations": ["and"],
            "all_translations": ["and", "also", "even", "both", "then"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "And, also, even (8900+ occurrences)"
        },
        "G1722": {
            "strongs": "G1722",
            "lemma": "ἐν",
            "translit": "en",
            "primary_translations": ["in"],
            "all_translations": ["in", "on", "at", "by", "with", "among"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "In, on, among (2700+ occurrences)"
        },
        "G1519": {
            "strongs": "G1519",
            "lemma": "εἰς",
            "translit": "eis",
            "primary_translations": ["into", "to"],
            "all_translations": ["into", "to", "unto", "for", "toward"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Into, to, unto (1700+ occurrences)"
        },
        "G1537": {
            "strongs": "G1537",
            "lemma": "ἐκ",
            "translit": "ek",
            "primary_translations": ["out of", "from"],
            "all_translations": ["out of", "from", "by", "of"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Out of, from (900+ occurrences)"
        },
        "G3756": {
            "strongs": "G3756",
            "lemma": "οὐ",
            "translit": "ou",
            "primary_translations": ["not"],
            "all_translations": ["not", "no", "cannot"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Not, no (1600+ occurrences)"
        },
        "G1161": {
            "strongs": "G1161",
            "lemma": "δέ",
            "translit": "de",
            "primary_translations": ["but", "and"],
            "all_translations": ["but", "and", "now", "then"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "But, and, now (2700+ occurrences)"
        },
        "G3739": {
            "strongs": "G3739",
            "lemma": "ὅς",
            "translit": "hos",
            "primary_translations": ["who", "which"],
            "all_translations": ["who", "which", "what", "that", "whose"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "Who, which, that (1400+ occurrences)"
        },
        "G1063": {
            "strongs": "G1063",
            "lemma": "γάρ",
            "translit": "gar",
            "primary_translations": ["for"],
            "all_translations": ["for", "because", "indeed", "therefore"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "For, because (1000+ occurrences)"
        },
        "G2192": {
            "strongs": "G2192",
            "lemma": "ἔχω",
            "translit": "echo",
            "primary_translations": ["have"],
            "all_translations": ["have", "has", "had", "hold", "possess"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To have, hold (700+ occurrences)"
        },
        "G4160": {
            "strongs": "G4160",
            "lemma": "ποιέω",
            "translit": "poieo",
            "primary_translations": ["do", "make"],
            "all_translations": ["do", "make", "did", "made", "cause", "perform"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To do, make (560+ occurrences)"
        },
        "G1492": {
            "strongs": "G1492",
            "lemma": "εἴδω",
            "translit": "eido",
            "primary_translations": ["see", "know"],
            "all_translations": ["see", "saw", "know", "knew", "perceive", "behold"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To see, know, perceive (660+ occurrences)"
        },
        "G1325": {
            "strongs": "G1325",
            "lemma": "δίδωμι",
            "translit": "didomi",
            "primary_translations": ["give"],
            "all_translations": ["give", "gave", "given", "grant", "put"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To give (410+ occurrences)"
        },
        "G2983": {
            "strongs": "G2983",
            "lemma": "λαμβάνω",
            "translit": "lambano",
            "primary_translations": ["take", "receive"],
            "all_translations": ["take", "took", "receive", "received", "accept"],
            "confidence": 1.0,
            "frequency": "very_high",
            "notes": "To take, receive (260+ occurrences)"
        }
    }
    
    return alignments

def main():
    """Create and save manual alignment files."""
    # Create Hebrew alignments
    hebrew_alignments = create_hebrew_alignments()
    hebrew_path = Path('data/manual_alignments/high_frequency_hebrew.json')
    hebrew_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(hebrew_path, 'w', encoding='utf-8') as f:
        json.dump(hebrew_alignments, f, indent=2, ensure_ascii=False)
    
    print(f"Created Hebrew manual alignments: {hebrew_path}")
    print(f"  Total entries: {len(hebrew_alignments)}")
    
    # Create Greek alignments
    greek_alignments = create_greek_alignments()
    greek_path = Path('data/manual_alignments/high_frequency_greek.json')
    
    with open(greek_path, 'w', encoding='utf-8') as f:
        json.dump(greek_alignments, f, indent=2, ensure_ascii=False)
    
    print(f"Created Greek manual alignments: {greek_path}")
    print(f"  Total entries: {len(greek_alignments)}")
    
    # Create combined statistics
    stats = {
        "created": "2025-06-08",
        "hebrew_entries": len(hebrew_alignments),
        "greek_entries": len(greek_alignments),
        "total_entries": len(hebrew_alignments) + len(greek_alignments),
        "description": "Manual alignments for highest frequency biblical words",
        "coverage_estimate": "These words cover approximately 50-60% of biblical text"
    }
    
    stats_path = Path('data/manual_alignments/statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nCreated statistics file: {stats_path}")
    print(f"Total manual alignments: {stats['total_entries']}")

if __name__ == '__main__':
    main()