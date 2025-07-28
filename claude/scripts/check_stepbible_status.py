#!/usr/bin/env python3
"""Check STEPBible import status."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database.import_tracker import ImportTracker


def main():
    """Check STEPBible import status."""
    
    tracker = ImportTracker()
    
    print("STEPBible Import Status")
    print("="*60)
    
    # Check overall status
    is_complete = tracker.is_stepbible_file_imported("complete", "all_stepbible_data")
    print(f"Overall complete: {is_complete}")
    print()
    
    # Check individual files
    files_to_check = [
        ("lexicon", "tbesh.txt"),
        ("lexicon", "tbesg.txt"),
        ("morphology", "tehmc.txt"),
        ("morphology", "tegmc.txt"),
        ("tahot", "tahot_gen_deu.txt"),
        ("tahot", "tahot_jos_est.txt"),
        ("tahot", "tahot_job_sng.txt"),
        ("tahot", "tahot_isa_mal.txt"),
        ("tagnt", "tagnt_mat_jhn.txt"),
        ("tagnt", "tagnt_act_rev.txt"),
    ]
    
    print("Individual files:")
    for file_type, filename in files_to_check:
        is_imported = tracker.is_stepbible_file_imported(file_type, filename)
        import_time = tracker.get_stepbible_import_time(file_type, filename)
        print(f"  {filename:<25} {file_type:<12} Imported: {is_imported:<5} Time: {import_time or 'N/A'}")
    
    # Get summary
    print()
    print("Import Summary:")
    summary = tracker.get_import_summary()
    print(f"  Translations: {summary['translations_imported']}")
    print(f"  STEPBible files by type:")
    for file_type, count in summary['stepbible_files'].items():
        print(f"    {file_type}: {count}")
    
    # Check if we should mark as complete
    total_expected = 10  # 2 lexicon + 2 morphology + 6 text files
    total_imported = sum(summary['stepbible_files'].values())
    
    print(f"\nTotal imported: {total_imported}/{total_expected}")
    
    if total_imported == total_expected and not is_complete:
        print("\n⚠️  All files imported but 'complete' marker not set!")
        print("This is why import keeps trying to run.")


if __name__ == "__main__":
    main()