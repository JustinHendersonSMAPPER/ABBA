#!/usr/bin/env python3
"""
Full Bible Export Example

This example demonstrates how to export complete Bible translations with all enrichments
using the ABBA framework.
"""

import json
import sqlite3
import sys
from pathlib import Path

# Add the parent directory to the path so we can import abba
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abba.cli import BibleProcessor, BibleExporter


def export_bible_example():
    """Export complete Bible with enrichments."""
    print("ABBA Full Bible Export Example")
    print("=" * 50)
    
    # Initialize processor with data directory
    data_dir = Path(__file__).parent.parent / "data" / "sources"
    processor = BibleProcessor(data_dir)
    
    # Process KJV translation
    print("\n1. Processing Bible data...")
    verses = processor.process_bible(["ENG_KJV"])
    print(f"   Processed {len(verses)} verses")
    
    # Export to JSON
    print("\n2. Exporting to JSON...")
    json_output = Path("example_bible_json")
    exporter = BibleExporter()
    exporter.export_to_json_by_translation(verses, json_output)
    print(f"   Exported to: {json_output}")
    
    # Export to SQLite
    print("\n3. Exporting to SQLite...")
    sqlite_output = Path("example_bible.db")
    exporter.export_to_sqlite(verses, sqlite_output)
    print(f"   Exported to: {sqlite_output}")
    
    # Demonstrate data access
    print("\n4. Sample data from exports:")
    
    # Read from JSON
    gen_path = json_output / "ENG_KJV" / "Gen.json"
    with open(gen_path) as f:
        genesis = json.load(f)
    
    first_verse = genesis["chapters"][0]["verses"][0]
    print(f"\n   First verse from JSON:")
    print(f"   {first_verse['verse_id']}: {first_verse['text']}")
    
    if first_verse["cross_references"]:
        print(f"   Cross-references: {len(first_verse['cross_references'])}")
        for xref in first_verse["cross_references"]:
            print(f"     - {xref['target']} ({xref['type']})")
    
    if first_verse["annotations"]:
        print(f"   Annotations:")
        for ann in first_verse["annotations"]:
            print(f"     - {ann['type']}: {ann['value']} (confidence: {ann['confidence']})")
    
    if first_verse["timeline_events"]:
        print(f"   Timeline events:")
        for event in first_verse["timeline_events"]:
            print(f"     - {event['name']}: {event['description']}")
    
    # Read from SQLite
    print(f"\n   Sample queries from SQLite:")
    conn = sqlite3.connect(str(sqlite_output))
    cursor = conn.cursor()
    
    # Count verses
    cursor.execute("SELECT COUNT(*) FROM verses")
    count = cursor.fetchone()[0]
    print(f"   Total verses: {count}")
    
    # Sample search
    cursor.execute("""
        SELECT verse_id, text 
        FROM verses 
        WHERE text LIKE '%love%' 
        LIMIT 3
    """)
    print(f"\n   Verses containing 'love':")
    for row in cursor.fetchall():
        verse_id, text = row
        print(f"   {verse_id}: {text[:60]}...")
    
    # Books with most verses
    cursor.execute("""
        SELECT book, COUNT(*) as count 
        FROM verses 
        GROUP BY book 
        ORDER BY count DESC 
        LIMIT 5
    """)
    print(f"\n   Books with most verses:")
    for row in cursor.fetchall():
        book, count = row
        print(f"   {book}: {count} verses")
    
    conn.close()
    
    print("\n" + "=" * 50)
    print("Export complete! You can now:")
    print("1. Browse the JSON files in 'example_bible_json/ENG_KJV/'")
    print("2. Query the SQLite database 'example_bible.db'")
    print("3. Build applications using this enriched Bible data")


def export_multiple_translations():
    """Export multiple translations for comparison."""
    print("\n\nMultiple Translation Export")
    print("=" * 50)
    
    data_dir = Path(__file__).parent.parent / "data" / "sources"
    processor = BibleProcessor(data_dir)
    
    # Process multiple translations
    translations = ["ENG_KJV", "ENG_ASV", "ENG_WEB"]
    print(f"\nProcessing {len(translations)} translations...")
    verses = processor.process_bible(translations)
    
    # Export to JSON
    output_dir = Path("multi_translation_export")
    exporter = BibleExporter()
    exporter.export_to_json_by_translation(verses, output_dir)
    
    print(f"\nExported to: {output_dir}")
    print("Directory structure:")
    for trans in translations:
        trans_dir = output_dir / trans
        if trans_dir.exists():
            book_count = len(list(trans_dir.glob("*.json"))) - 1  # Exclude manifest.json
            print(f"  {trans}/  ({book_count} books)")


if __name__ == "__main__":
    # Run the examples
    export_bible_example()
    
    # Uncomment to also export multiple translations
    # export_multiple_translations()