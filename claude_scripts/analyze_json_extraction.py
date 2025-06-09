#!/usr/bin/env python3
"""
Analyze JSON extraction from Strong's XML files to ensure completeness.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter


def analyze_hebrew_extraction():
    """Analyze Hebrew XML to JSON extraction."""
    print("\n=== Hebrew Extraction Analysis ===\n")
    
    xml_path = Path('data/sources/lexicons/strongs_hebrew.xml')
    json_path = Path('data/sources/lexicons/strongs_hebrew.json')
    
    if not xml_path.exists() or not json_path.exists():
        print("Hebrew files not found")
        return
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'osis': 'http://www.bibletechnologies.net/2003/OSIS/namespace'}
    
    # Count entries
    xml_entries = root.findall('.//osis:div[@type="entry"]', ns)
    print(f"XML entries: {len(xml_entries)}")
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"JSON entries: {len(json_data)}")
    
    # Analyze fields in first few entries
    print("\nSample Hebrew entry fields:")
    sample_entry = xml_entries[0]
    
    # Get w element
    w_elem = sample_entry.find('.//osis:w', ns)
    if w_elem is not None:
        print(f"  ID: {w_elem.get('ID')}")
        print(f"  Text: {w_elem.text}")
        print(f"  Attributes: {dict(w_elem.attrib)}")
    
    # Get notes
    notes = sample_entry.findall('.//osis:note', ns)
    print(f"\n  Note types: {[n.get('type') for n in notes]}")
    
    # Check JSON completeness
    first_id = list(json_data.keys())[0]
    print(f"\nJSON entry '{first_id}' has fields:")
    for key, value in json_data[first_id].items():
        if value:
            print(f"  {key}: {str(value)[:60]}...")
    
    # Count entries with each field populated
    field_counts = Counter()
    for entry in json_data.values():
        for field, value in entry.items():
            if value:
                field_counts[field] += 1
    
    print("\nField population counts:")
    for field, count in sorted(field_counts.items()):
        percentage = (count / len(json_data)) * 100
        print(f"  {field}: {count} ({percentage:.1f}%)")


def analyze_greek_extraction():
    """Analyze Greek XML to JSON extraction."""
    print("\n\n=== Greek Extraction Analysis ===\n")
    
    xml_path = Path('data/sources/lexicons/strongs_greek.xml')
    json_path = Path('data/sources/lexicons/strongs_greek.json')
    
    if not xml_path.exists() or not json_path.exists():
        print("Greek files not found")
        return
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Count entries
    xml_entries = root.findall('.//entry')
    print(f"XML entries: {len(xml_entries)}")
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"JSON entries: {len(json_data)}")
    
    # Analyze fields in first few entries
    print("\nSample Greek entry structure:")
    sample_entry = xml_entries[0]
    
    # List all child elements
    for child in sample_entry:
        if child.text and child.text.strip():
            print(f"  <{child.tag}>: {child.text.strip()[:60]}...")
        elif child.attrib:
            print(f"  <{child.tag}>: {dict(child.attrib)}")
    
    # Check JSON completeness
    first_id = list(json_data.keys())[0]
    print(f"\nJSON entry '{first_id}' has fields:")
    for key, value in json_data[first_id].items():
        if value:
            print(f"  {key}: {str(value)[:60]}...")
    
    # Count entries with each field populated
    field_counts = Counter()
    for entry in json_data.values():
        for field, value in entry.items():
            if value and (not isinstance(value, list) or len(value) > 0):
                field_counts[field] += 1
    
    print("\nField population counts:")
    for field, count in sorted(field_counts.items()):
        percentage = (count / len(json_data)) * 100
        print(f"  {field}: {count} ({percentage:.1f}%)")


def analyze_kjv_usage_patterns():
    """Analyze patterns in kjv_usage fields for both languages."""
    print("\n\n=== KJV Usage Pattern Analysis ===\n")
    
    # Hebrew patterns
    hebrew_json = Path('data/sources/lexicons/strongs_hebrew.json')
    if hebrew_json.exists():
        with open(hebrew_json, 'r', encoding='utf-8') as f:
            hebrew_data = json.load(f)
        
        print("Hebrew KJV usage patterns:")
        hebrew_patterns = Counter()
        sample_hebrew = []
        
        for entry_id, entry in hebrew_data.items():
            kjv = entry.get('kjv_usage', '')
            if kjv:
                # Categorize pattern
                if '(Compare' in kjv:
                    hebrew_patterns['has_compare'] += 1
                if '[idiom]' in kjv:
                    hebrew_patterns['has_idiom'] += 1
                if re.search(r'\(\d+x\)', kjv):
                    hebrew_patterns['has_counts'] += 1
                if ',' in kjv:
                    hebrew_patterns['comma_separated'] += 1
                else:
                    hebrew_patterns['single_word'] += 1
                
                if len(sample_hebrew) < 5:
                    sample_hebrew.append((entry_id, kjv[:80]))
        
        for pattern, count in sorted(hebrew_patterns.items()):
            print(f"  {pattern}: {count}")
        
        print("\nHebrew samples:")
        for sid, sample in sample_hebrew:
            print(f"  {sid}: {sample}")
    
    # Greek patterns
    greek_json = Path('data/sources/lexicons/strongs_greek.json')
    if greek_json.exists():
        with open(greek_json, 'r', encoding='utf-8') as f:
            greek_data = json.load(f)
        
        print("\n\nGreek KJV usage patterns:")
        greek_patterns = Counter()
        sample_greek = []
        
        for entry_id, entry in greek_data.items():
            kjv = entry.get('kjv_usage', '')
            if kjv:
                # Categorize pattern
                if kjv.startswith(':--'):
                    greek_patterns[':-- prefix'] += 1
                elif kjv.startswith('--'):
                    greek_patterns['-- prefix'] += 1
                if '\\n' in repr(kjv):
                    greek_patterns['has_newlines'] += 1
                if ',' in kjv:
                    greek_patterns['comma_separated'] += 1
                if kjv.strip().endswith('.'):
                    greek_patterns['ends_with_period'] += 1
                
                if len(sample_greek) < 5:
                    sample_greek.append((entry_id, kjv[:80]))
        
        for pattern, count in sorted(greek_patterns.items()):
            print(f"  {pattern}: {count}")
        
        print("\nGreek samples:")
        for sid, sample in sample_greek:
            print(f"  {sid}: {sample}")


if __name__ == "__main__":
    import re  # Import for pattern analysis
    
    analyze_hebrew_extraction()
    analyze_greek_extraction()
    analyze_kjv_usage_patterns()