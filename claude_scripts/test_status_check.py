#!/usr/bin/env python3
"""Check current test status and identify remaining failures."""

import subprocess
import re
from pathlib import Path

def run_pytest_summary():
    """Run pytest with summary output to check test status."""
    print("Running pytest to check current test status...\n")
    
    cmd = ["python", "-m", "pytest", "-v", "--tb=no", "-q"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Extract summary information
        output = result.stdout + result.stderr
        
        # Look for the summary line
        summary_match = re.search(r'(\d+) passed.*?(\d+) failed.*?(\d+) error', output)
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2))
            errors = int(summary_match.group(3))
            total = passed + failed + errors
            
            print(f"Test Summary:")
            print(f"✓ Passed: {passed}/{total} ({passed/total*100:.1f}%)")
            print(f"✗ Failed: {failed}")
            print(f"⚠ Errors: {errors}")
            print(f"\nTotal coverage: {passed/total*100:.1f}%")
            
            if passed/total >= 0.9:
                print("\n🎉 Achieved 90%+ test coverage!")
            else:
                print(f"\n📊 Need {int(total*0.9 - passed)} more passing tests for 90% coverage")
        
        # Extract failed test names
        failed_tests = re.findall(r'FAILED (test_\S+::\S+)', output)
        if failed_tests:
            print("\n\nFailed tests:")
            for test in failed_tests[:10]:  # Show first 10
                print(f"  - {test}")
            if len(failed_tests) > 10:
                print(f"  ... and {len(failed_tests) - 10} more")
                
        # Extract error test names  
        error_tests = re.findall(r'ERROR (test_\S+::\S+)', output)
        if error_tests:
            print("\n\nTests with errors:")
            for test in error_tests[:10]:  # Show first 10
                print(f"  - {test}")
            if len(error_tests) > 10:
                print(f"  ... and {len(error_tests) - 10} more")
                
    except Exception as e:
        print(f"Error running pytest: {e}")
        
def check_test_files():
    """Check which test files exist and their coverage."""
    print("\n\nChecking test file coverage...\n")
    
    src_dir = Path("src/abba")
    test_dir = Path("tests")
    
    # Find all Python modules in src
    src_modules = set()
    for py_file in src_dir.rglob("*.py"):
        if py_file.name != "__init__.py":
            module_name = py_file.stem
            src_modules.add(module_name)
    
    # Find all test files
    test_modules = set()
    for py_file in test_dir.rglob("test_*.py"):
        # Extract module name from test file
        module_name = py_file.stem.replace("test_", "")
        test_modules.add(module_name)
    
    # Find modules without tests
    missing_tests = src_modules - test_modules
    if missing_tests:
        print("Modules without test files:")
        for module in sorted(missing_tests):
            print(f"  - {module}")
    else:
        print("✓ All modules have test files!")
        
    print(f"\nTotal modules: {len(src_modules)}")
    print(f"Modules with tests: {len(test_modules)}")
    print(f"Test coverage: {len(test_modules)/len(src_modules)*100:.1f}%")

if __name__ == "__main__":
    run_pytest_summary()
    check_test_files()