#!/usr/bin/env python3
import json
import sys

if len(sys.argv) != 4:
    print("Usage: check_verse.py <json_file> <chapter> <verse>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    data = json.load(f)

chapter_num = int(sys.argv[2])
verse_num = int(sys.argv[3])

for chapter in data['chapters']:
    if chapter['chapter'] == chapter_num:
        for verse in chapter['verses']:
            if verse['verse'] == verse_num:
                print(json.dumps(verse, indent=2))
                sys.exit(0)

print(f"Verse {chapter_num}:{verse_num} not found")