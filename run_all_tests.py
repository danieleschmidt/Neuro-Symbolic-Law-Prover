#!/usr/bin/env python3
"""
Run all generation tests for quality gate validation
"""

import subprocess
import sys

def run_test(test_file):
    """Run a single test file."""
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    tests = [
        'test_generation1_simple.py',
        'test_generation2_robust.py', 
        'test_generation3_scale.py'
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        success, stdout, stderr = run_test(test)
        if success:
            print(f"✅ {test} PASSED")
            passed += 1
        else:
            print(f"❌ {test} FAILED")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
            failed += 1
    
    print(f"\nTest Results: {passed}/{len(tests)} passed")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())