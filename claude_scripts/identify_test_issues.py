#!/usr/bin/env python3
"""Identify potential test issues by analyzing test files."""

import re
from pathlib import Path

def find_potential_issues():
    """Find potential issues in test files."""
    test_dir = Path("tests")
    issues = []
    
    patterns = {
        'missing_imports': r'from abba\.[^:\n]*? import .*?(?:Participant|HISTORICAL|HIGH)',
        'wrong_timepoint': r'TimePoint\(year=',
        'wrong_eventtype': r'EventType\.HISTORICAL',
        'wrong_certainty': r'CertaintyLevel\.HIGH',
    }
    
    for test_file in test_dir.glob("test_*.py"):
        if test_file.name.endswith('.disabled'):
            continue
            
        try:
            content = test_file.read_text()
            
            for issue_name, pattern in patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    issues.append({
                        'file': test_file.name,
                        'issue': issue_name,
                        'count': len(matches),
                        'matches': matches[:3]  # First 3 matches
                    })
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
    
    return issues

def analyze_imports():
    """Analyze imports to find mismatches."""
    test_dir = Path("tests")
    import_issues = []
    
    # Known problematic imports
    bad_imports = {
        'Participant': 'EntityRef',
        'EventType.HISTORICAL': 'EventType.POINT',
        'CertaintyLevel.HIGH': 'CertaintyLevel.CERTAIN',
    }
    
    for test_file in test_dir.glob("test_*.py"):
        if test_file.name.endswith('.disabled'):
            continue
            
        try:
            content = test_file.read_text()
            
            for bad, good in bad_imports.items():
                if bad in content:
                    import_issues.append({
                        'file': test_file.name,
                        'bad_import': bad,
                        'should_be': good,
                        'line': next((i+1 for i, line in enumerate(content.splitlines()) if bad in line), None)
                    })
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
    
    return import_issues

def main():
    print("Analyzing test files for potential issues...\n")
    
    # Find pattern-based issues
    issues = find_potential_issues()
    if issues:
        print("Pattern-based issues found:")
        for issue in issues:
            print(f"\n{issue['file']}:")
            print(f"  Issue: {issue['issue']} (found {issue['count']} times)")
            print(f"  Examples: {issue['matches']}")
    else:
        print("✓ No pattern-based issues found!")
    
    # Find import issues
    import_issues = analyze_imports()
    if import_issues:
        print("\n\nImport issues found:")
        for issue in import_issues:
            print(f"\n{issue['file']} (line {issue['line']}):")
            print(f"  Found: {issue['bad_import']}")
            print(f"  Should be: {issue['should_be']}")
    else:
        print("\n✓ No import issues found!")
    
    # Check for disabled tests
    test_dir = Path("tests")
    disabled = list(test_dir.glob("*.disabled"))
    if disabled:
        print(f"\n\nDisabled test files ({len(disabled)}):")
        for f in disabled:
            print(f"  - {f.name}")

if __name__ == "__main__":
    main()