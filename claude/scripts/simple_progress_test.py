#!/usr/bin/env python3
"""Simple test to verify tqdm works."""

from tqdm import tqdm
import time

print("Testing basic tqdm progress bar...")

# Test 1: Simple progress
print("\nTest 1: Basic progress")
for i in tqdm(range(10), desc="Processing"):
    time.sleep(0.1)

# Test 2: Manual update
print("\nTest 2: Manual update")
pbar = tqdm(total=100, desc="Manual", unit="items")
for i in range(10):
    time.sleep(0.1)
    pbar.update(10)
pbar.close()

# Test 3: Nested output
print("\nTest 3: With text output")
pbar = tqdm(total=5, desc="With output")
for i in range(5):
    pbar.write(f"Processing item {i}")
    time.sleep(0.2)
    pbar.update(1)
pbar.close()

print("\n✓ All tests complete!")