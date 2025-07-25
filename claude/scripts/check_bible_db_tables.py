#!/usr/bin/env python3
"""Check what tables exist in bible.db."""

import sqlite3
from pathlib import Path
from abba.config import config_manager

def main():
    config = config_manager.load_config([])
    bible_db_path = config.db_path
    
    if not bible_db_path.exists():
        print(f"bible.db not found at {bible_db_path}")
        return
    
    print(f"Examining bible.db at: {bible_db_path}")
    
    with sqlite3.connect(bible_db_path) as conn:
        cursor = conn.cursor()
        
        # List all tables
        print("\nAll tables in bible.db:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for table in tables:
            print(f"  {table[0]}")
            
        # Check each table structure
        for table_name, in tables:
            print(f"\n{table_name} table schema:")
            cursor.execute(f"PRAGMA table_info({table_name})")
            cols = cursor.fetchall()
            for col in cols:
                print(f"  {col}")
                
            # Sample data
            print(f"\nSample {table_name} data (first 2 rows):")
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                rows = cursor.fetchall()
                for row in rows:
                    print(f"  {row}")
                    if row:
                        print(f"  Types: {[type(v).__name__ for v in row]}")
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    main()