#!/usr/bin/env python3
"""Check the actual database schema."""

import sqlite3
from pathlib import Path

db_path = Path("bible_data/abba.db")
if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()

print("Tables in abba.db:")
for table in tables:
    print(f"  - {table[0]}")

# Check verses table schema
print("\nSchema for 'verses' table:")
cursor.execute("PRAGMA table_info(verses)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Check if there's any data
cursor.execute("SELECT COUNT(*) FROM verses")
count = cursor.fetchone()[0]
print(f"\nTotal verses in database: {count:,}")

# Show a sample verse
if count > 0:
    cursor.execute("SELECT * FROM verses LIMIT 1")
    sample = cursor.fetchone()
    cursor.execute("PRAGMA table_info(verses)")
    col_names = [col[1] for col in cursor.fetchall()]
    print("\nSample verse:")
    for i, col_name in enumerate(col_names):
        print(f"  {col_name}: {sample[i]}")

conn.close()