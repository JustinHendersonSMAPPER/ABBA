#!/usr/bin/env python3
"""Check for unmapped book names."""

import sys
from pathlib import Path
import sqlite3

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    db_path = Path("bible_data/abba.db")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get all book names
        cursor.execute("""
            SELECT DISTINCT book, COUNT(*) as count 
            FROM stepbible_verses 
            WHERE original_word IS NOT NULL AND original_word != ''
            GROUP BY book
            ORDER BY book
        """)
        
        all_books = cursor.fetchall()
        
        # Known mapped books
        mapped_books = {
            'Gen','Exo','Lev','Num','Deu','Jos','Jdg','Rut','1Sa','2Sa',
            '1Ki','2Ki','1Ch','2Ch','Ezr','Neh','Est','Job','Psa','Pro',
            'Ecc','Sng','Isa','Jer','Lam','Ezk','Dan','Hos','Jol','Amo',
            'Oba','Jon','Mic','Nam','Hab','Zep','Hag','Zec','Mal',
            'Mat','Mrk','Luk','Jhn','Act','Rom','1Co','2Co','Gal','Eph',
            'Php','Col','1Th','2Th','1Ti','2Ti','Tit','Phm','Heb','Jas',
            '1Pe','2Pe','1Jn','2Jn','3Jn','Jud','Rev'
        }
        
        print("All books in stepbible_verses:")
        print("=" * 40)
        
        unmapped = []
        for book, count in all_books:
            status = "✓" if book in mapped_books else "❌"
            print(f"{status} {book:10} {count:6,} verses")
            if book not in mapped_books:
                unmapped.append((book, count))
        
        if unmapped:
            print(f"\n❌ Found {len(unmapped)} unmapped books:")
            for book, count in unmapped:
                print(f"   {book}: {count} verses")
        else:
            print("\n✅ All books are mapped!")
        
        # Check for book_id = 0 in embeddings
        print("\n" + "="*40)
        print("Checking for book_id = 0 in query...")
        
        cursor.execute("""
            SELECT 
                CASE book
                    WHEN 'Gen' THEN 1 WHEN 'Exo' THEN 2 WHEN 'Lev' THEN 3 WHEN 'Num' THEN 4 WHEN 'Deu' THEN 5
                    WHEN 'Jos' THEN 6 WHEN 'Jdg' THEN 7 WHEN 'Rut' THEN 8 WHEN '1Sa' THEN 9 WHEN '2Sa' THEN 10
                    WHEN '1Ki' THEN 11 WHEN '2Ki' THEN 12 WHEN '1Ch' THEN 13 WHEN '2Ch' THEN 14 WHEN 'Ezr' THEN 15
                    WHEN 'Neh' THEN 16 WHEN 'Est' THEN 17 WHEN 'Job' THEN 18 WHEN 'Psa' THEN 19 WHEN 'Pro' THEN 20
                    WHEN 'Ecc' THEN 21 WHEN 'Sng' THEN 22 WHEN 'Isa' THEN 23 WHEN 'Jer' THEN 24 WHEN 'Lam' THEN 25
                    WHEN 'Ezk' THEN 26 WHEN 'Dan' THEN 27 WHEN 'Hos' THEN 28 WHEN 'Jol' THEN 29 WHEN 'Amo' THEN 30
                    WHEN 'Oba' THEN 31 WHEN 'Jon' THEN 32 WHEN 'Mic' THEN 33 WHEN 'Nam' THEN 34 WHEN 'Hab' THEN 35
                    WHEN 'Zep' THEN 36 WHEN 'Hag' THEN 37 WHEN 'Zec' THEN 38 WHEN 'Mal' THEN 39
                    WHEN 'Mat' THEN 40 WHEN 'Mrk' THEN 41 WHEN 'Luk' THEN 42 WHEN 'Jhn' THEN 43 WHEN 'Act' THEN 44
                    WHEN 'Rom' THEN 45 WHEN '1Co' THEN 46 WHEN '2Co' THEN 47 WHEN 'Gal' THEN 48 WHEN 'Eph' THEN 49
                    WHEN 'Php' THEN 50 WHEN 'Col' THEN 51 WHEN '1Th' THEN 52 WHEN '2Th' THEN 53 WHEN '1Ti' THEN 54
                    WHEN '2Ti' THEN 55 WHEN 'Tit' THEN 56 WHEN 'Phm' THEN 57 WHEN 'Heb' THEN 58 WHEN 'Jas' THEN 59
                    WHEN '1Pe' THEN 60 WHEN '2Pe' THEN 61 WHEN '1Jn' THEN 62 WHEN '2Jn' THEN 63 WHEN '3Jn' THEN 64
                    WHEN 'Jud' THEN 65 WHEN 'Rev' THEN 66
                    ELSE 0
                END as book_id,
                book,
                COUNT(*) as count
            FROM stepbible_verses
            WHERE original_word IS NOT NULL AND original_word != ''
            GROUP BY book_id, book
            HAVING book_id = 0
        """)
        
        zeros = cursor.fetchall()
        if zeros:
            print(f"Found {len(zeros)} books mapping to book_id = 0:")
            for book_id, book, count in zeros:
                print(f"   {book}: {count} verses")

if __name__ == "__main__":
    main()