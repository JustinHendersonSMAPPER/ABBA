#!/usr/bin/env python3
"""
Simple example of using ABBA to export biblical verses.
"""

from abba.export.minimal_sqlite import MinimalSQLiteExporter
from abba.export.minimal_json import MinimalJSONExporter


def main():
    """Demonstrate simple export functionality."""
    
    # Sample verse data (normally you would load this from a source)
    sample_verses = [
        ("GEN.1.1", "GEN", 1, 1, "In the beginning God created the heavens and the earth."),
        ("GEN.1.2", "GEN", 1, 2, "And the earth was without form, and void; and darkness was upon the face of the deep."),
        ("GEN.1.3", "GEN", 1, 3, "And God said, Let there be light: and there was light."),
        ("JHN.1.1", "JHN", 1, 1, "In the beginning was the Word, and the Word was with God, and the Word was God."),
        ("JHN.3.16", "JHN", 3, 16, "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life."),
    ]
    
    # Export to SQLite
    print("Exporting to SQLite...")
    with MinimalSQLiteExporter("example_bible.db") as exporter:
        for verse_id, book, chapter, verse, text in sample_verses:
            exporter.add_verse(verse_id, book, chapter, verse, text)
    print("✓ SQLite export complete: example_bible.db")
    
    # Export to JSON
    print("\nExporting to JSON...")
    with MinimalJSONExporter("example_bible.json") as exporter:
        for verse_id, book, chapter, verse, text in sample_verses:
            exporter.add_verse(verse_id, book, chapter, verse, text)
    print("✓ JSON export complete: example_bible.json")
    
    # Demonstrate reading from SQLite
    print("\nReading from SQLite database:")
    import sqlite3
    conn = sqlite3.connect("example_bible.db")
    cursor = conn.cursor()
    
    # Search for verses containing "beginning"
    cursor.execute("""
        SELECT verse_id, text 
        FROM verses 
        WHERE text LIKE '%beginning%'
    """)
    
    for verse_id, text in cursor.fetchall():
        print(f"  {verse_id}: {text[:50]}...")
    
    conn.close()
    
    # Demonstrate reading from JSON
    print("\nReading from JSON file:")
    import json
    with open("example_bible.json", "r") as f:
        data = json.load(f)
        print(f"  Total verses: {data['metadata']['verse_count']}")
        print(f"  First verse: {data['verses'][0]['verse_id']}")


if __name__ == "__main__":
    main()