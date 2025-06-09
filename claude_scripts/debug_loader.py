#!/usr/bin/env python3
"""Debug the cross-reference loader."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abba.cross_references.loader import CrossReferenceLoader
from abba.cross_references.models import ReferenceType, ReferenceRelationship
import logging

logging.basicConfig(level=logging.DEBUG)

def main():
    data_dir = Path(__file__).parent.parent / "data"
    loader = CrossReferenceLoader(data_dir)
    
    # Manually load and check
    xref_file = data_dir / "cross_references.json"
    print(f"Loading from: {xref_file}")
    print(f"File exists: {xref_file.exists()}")
    
    with open(xref_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nFound {len(data.get('references', []))} references in JSON")
    
    # Check enum values
    print("\nReferenceType values:")
    for rt in ReferenceType:
        print(f"  {rt.value}")
    
    print("\nReferenceRelationship values:")
    for rr in ReferenceRelationship:
        print(f"  {rr.value}")
    
    # Try loading with the loader
    refs = loader.load_from_json()
    print(f"\nLoader returned {len(refs)} references")

if __name__ == "__main__":
    main()