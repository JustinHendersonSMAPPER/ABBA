#!/usr/bin/env python3
"""Generate complete book mappings for Hebrew and Greek texts."""

import json
from pathlib import Path

def generate_book_mappings():
    """Generate complete book mappings based on available files."""
    
    data_dir = Path('data')
    
    # Hebrew mappings - based on files found
    hebrew_dir = data_dir / "sources" / "hebrew"
    hebrew_files = sorted(hebrew_dir.glob("*.xml"))
    
    print("Hebrew Book Mappings:")
    print("-" * 40)
    print("book_files = {")
    
    hebrew_mappings = []
    for f in hebrew_files:
        stem = f.stem
        if stem == 'VerseMap':
            continue
            
        # Generate standard book code mapping
        mapping = {
            'Gen': ('GEN', 'Genesis'),
            'Exod': ('EXO', 'Exodus'), 
            'Lev': ('LEV', 'Leviticus'),
            'Num': ('NUM', 'Numbers'),
            'Deut': ('DEU', 'Deuteronomy'),
            'Josh': ('JOS', 'Joshua'),
            'Judg': ('JDG', 'Judges'),
            'Ruth': ('RUT', 'Ruth'),
            '1Sam': ('1SA', '1 Samuel'),
            '2Sam': ('2SA', '2 Samuel'),
            '1Kgs': ('1KI', '1 Kings'),
            '2Kgs': ('2KI', '2 Kings'),
            '1Chr': ('1CH', '1 Chronicles'),
            '2Chr': ('2CH', '2 Chronicles'),
            'Ezra': ('EZR', 'Ezra'),
            'Neh': ('NEH', 'Nehemiah'),
            'Esth': ('EST', 'Esther'),
            'Job': ('JOB', 'Job'),
            'Ps': ('PSA', 'Psalms'),
            'Prov': ('PRO', 'Proverbs'),
            'Eccl': ('ECC', 'Ecclesiastes'),
            'Song': ('SNG', 'Song of Solomon'),
            'Isa': ('ISA', 'Isaiah'),
            'Jer': ('JER', 'Jeremiah'),
            'Lam': ('LAM', 'Lamentations'),
            'Ezek': ('EZK', 'Ezekiel'),
            'Dan': ('DAN', 'Daniel'),
            'Hos': ('HOS', 'Hosea'),
            'Joel': ('JOL', 'Joel'),
            'Amos': ('AMO', 'Amos'),
            'Obad': ('OBA', 'Obadiah'),
            'Jonah': ('JON', 'Jonah'),
            'Mic': ('MIC', 'Micah'),
            'Nah': ('NAH', 'Nahum'),
            'Hab': ('HAB', 'Habakkuk'),
            'Zeph': ('ZEP', 'Zephaniah'),
            'Hag': ('HAG', 'Haggai'),
            'Zech': ('ZEC', 'Zechariah'),
            'Mal': ('MAL', 'Malachi')
        }
        
        if stem in mapping:
            code, name = mapping[stem]
            hebrew_mappings.append(f"    '{code}': '{stem}.xml', '{stem}': '{stem}.xml',")
    
    for line in hebrew_mappings:
        print(line)
    print("}")
    
    # Greek mappings
    greek_dir = data_dir / "sources" / "greek"
    greek_files = sorted(greek_dir.glob("*.xml"))
    
    print("\n\nGreek Book Mappings:")
    print("-" * 40)
    print("book_files = {")
    
    greek_mappings = []
    for f in greek_files:
        stem = f.stem
        
        # Map file names to standard codes
        mapping = {
            'MAT': ('MAT', 'Matt', 'Matthew'),
            'MAR': ('MRK', 'Mark', 'Mark'),
            'LUK': ('LUK', 'Luke', 'Luke'),
            'JOH': ('JHN', 'John', 'John'),
            'ACT': ('ACT', 'Acts', 'Acts'),
            'ROM': ('ROM', 'Rom', 'Romans'),
            '1CO': ('1CO', '1Cor', '1 Corinthians'),
            '2CO': ('2CO', '2Cor', '2 Corinthians'),
            'GAL': ('GAL', 'Gal', 'Galatians'),
            'EPH': ('EPH', 'Eph', 'Ephesians'),
            'PHP': ('PHP', 'Phil', 'Philippians'),
            'COL': ('COL', 'Col', 'Colossians'),
            '1TH': ('1TH', '1Thess', '1 Thessalonians'),
            '2TH': ('2TH', '2Thess', '2 Thessalonians'),
            '1TI': ('1TI', '1Tim', '1 Timothy'),
            '2TI': ('2TI', '2Tim', '2 Timothy'),
            'TIT': ('TIT', 'Titus', 'Titus'),
            'PHM': ('PHM', 'Phlm', 'Philemon'),
            'HEB': ('HEB', 'Heb', 'Hebrews'),
            'JAM': ('JAM', 'Jas', 'James'),
            '1PE': ('1PE', '1Pet', '1 Peter'),
            '2PE': ('2PE', '2Pet', '2 Peter'),
            '1JO': ('1JO', '1John', '1 John'),
            '2JO': ('2JO', '2John', '2 John'),
            '3JO': ('3JO', '3John', '3 John'),
            'JUD': ('JUD', 'Jude', 'Jude'),
            'REV': ('REV', 'Rev', 'Revelation')
        }
        
        if stem in mapping:
            code, alt, name = mapping[stem]
            greek_mappings.append(f"    '{code}': '{stem}.xml', '{alt}': '{stem}.xml',")
    
    for line in greek_mappings:
        print(line)
    print("}")
    
    # Generate timeline data file
    print("\n\nSample Timeline Data (timeline_events.json):")
    print("-" * 40)
    
    timeline_data = {
        "format_version": "1.0",
        "description": "Biblical timeline events database",
        "events": [
            {
                "id": "creation",
                "name": "Creation",
                "description": "The creation of the world according to Genesis",
                "event_type": "point",
                "date_bce": 4004,
                "confidence": 0.3,
                "verse_refs": ["GEN.1.1"],
                "categories": ["theological", "cosmological"],
                "notes": "Traditional date based on Ussher chronology"
            },
            {
                "id": "flood",
                "name": "The Great Flood",
                "description": "The flood of Noah",
                "event_type": "point",
                "date_bce": 2349,
                "confidence": 0.3,
                "verse_refs": ["GEN.7.11", "GEN.8.4"],
                "categories": ["judgment", "covenant"]
            },
            {
                "id": "abraham_call",
                "name": "Call of Abraham",
                "description": "God calls Abraham to leave Ur",
                "event_type": "point",
                "date_bce": 2091,
                "confidence": 0.5,
                "verse_refs": ["GEN.12.1"],
                "categories": ["covenant", "patriarchal"]
            },
            {
                "id": "exodus",
                "name": "The Exodus",
                "description": "Israel leaves Egypt under Moses",
                "event_type": "point",
                "date_bce": 1446,
                "confidence": 0.7,
                "verse_refs": ["EXO.12.31", "EXO.12.41"],
                "categories": ["historical", "foundational", "redemption"]
            },
            {
                "id": "david_king",
                "name": "David Becomes King",
                "description": "David becomes king over all Israel",
                "event_type": "point",
                "date_bce": 1010,
                "confidence": 0.8,
                "verse_refs": ["2SA.5.3"],
                "categories": ["monarchy", "davidic_covenant"]
            },
            {
                "id": "temple_built",
                "name": "Solomon's Temple Built",
                "description": "Completion of the First Temple",
                "event_type": "point",
                "date_bce": 959,
                "confidence": 0.8,
                "verse_refs": ["1KI.6.38"],
                "categories": ["temple", "worship"]
            },
            {
                "id": "exile_babylon",
                "name": "Babylonian Exile",
                "description": "Judah taken into captivity by Babylon",
                "event_type": "point", 
                "date_bce": 586,
                "confidence": 0.9,
                "verse_refs": ["2KI.25.8", "JER.52.12"],
                "categories": ["judgment", "exile"]
            },
            {
                "id": "return_exile",
                "name": "Return from Exile",
                "description": "First return under Zerubbabel",
                "event_type": "point",
                "date_bce": 538,
                "confidence": 0.9,
                "verse_refs": ["EZR.1.1"],
                "categories": ["restoration", "prophecy_fulfillment"]
            },
            {
                "id": "jesus_birth",
                "name": "Birth of Jesus",
                "description": "The birth of Jesus Christ in Bethlehem",
                "event_type": "point",
                "date_bce": 4,
                "confidence": 0.7,
                "verse_refs": ["MAT.2.1", "LUK.2.7"],
                "categories": ["incarnation", "messianic"]
            },
            {
                "id": "crucifixion",
                "name": "Crucifixion of Jesus",
                "description": "The death of Jesus on the cross",
                "event_type": "point",
                "date_ce": 30,
                "confidence": 0.8,
                "verse_refs": ["MAT.27.35", "MRK.15.24", "LUK.23.33", "JHN.19.18"],
                "categories": ["redemption", "atonement", "prophecy_fulfillment"]
            },
            {
                "id": "resurrection",
                "name": "Resurrection of Jesus", 
                "description": "Jesus rises from the dead",
                "event_type": "point",
                "date_ce": 30,
                "confidence": 0.8,
                "verse_refs": ["MAT.28.6", "MRK.16.6", "LUK.24.6", "JHN.20.1"],
                "categories": ["resurrection", "victory"]
            },
            {
                "id": "pentecost",
                "name": "Day of Pentecost",
                "description": "The Holy Spirit comes upon the disciples",
                "event_type": "point",
                "date_ce": 30,
                "confidence": 0.8,
                "verse_refs": ["ACT.2.1"],
                "categories": ["church", "holy_spirit"]
            }
        ]
    }
    
    print(json.dumps(timeline_data, indent=2))

if __name__ == '__main__':
    generate_book_mappings()