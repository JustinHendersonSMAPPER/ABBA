#!/usr/bin/env python3
"""Fix original language embeddings with proper book name mapping and without validator conflicts."""

import sys
from pathlib import Path
import json

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager, ContextBuilder
from abba.embeddings.original_language_pipeline import OriginalLanguageEmbeddingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_book_mappings():
    """Check and display actual book names in the database."""
    db_path = Path("bible_data/abba.db")
    db_manager = SQLiteManager(db_path)
    
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT book, 
                   COUNT(DISTINCT chapter || ':' || verse) as verse_count
            FROM stepbible_verses 
            WHERE original_word IS NOT NULL AND original_word != ''
            GROUP BY book
            ORDER BY 
                CASE 
                    WHEN book IN ('Gen','Exo','Lev','Num','Deu','Jos','Jdg','Rut','1Sa','2Sa',
                                  '1Ki','2Ki','1Ch','2Ch','Ezr','Neh','Est','Job','Psa','Pro',
                                  'Ecc','Son','Isa','Jer','Lam','Eze','Dan','Hos','Joe','Amo',
                                  'Oba','Jon','Mic','Nah','Hab','Zep','Hag','Zec','Mal') THEN 1
                    ELSE 2
                END,
                book
        """)
        
        print("\n" + "=" * 60)
        print("Book Names in stepbible_verses:")
        print("=" * 60)
        
        total_verses = 0
        for book, count in cursor.fetchall():
            print(f"{book:10} {count:6,} verses")
            total_verses += count
        
        print(f"\nTotal canonical verses: {total_verses:,}")
        return total_verses

def main():
    # First check the actual book mappings
    expected_count = check_book_mappings()
    
    # Initialize components
    db_path = Path("bible_data/abba.db")
    vector_path = Path("bible_data/vectors")
    models_path = Path("bible_data/models")
    progress_path = Path("bible_data/.embedding_progress.json")
    
    # Clear progress for original verses to force re-embed
    if progress_path.exists():
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        
        if 'original_verses' in progress:
            del progress['original_verses']
            
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        print("\n✓ Cleared original verse progress")
    
    db_manager = SQLiteManager(db_path)
    chroma_manager = ChromaManager(persist_path=str(vector_path))
    
    # 1. Clean up bad original embeddings
    print("\n" + "="*60)
    print("Step 1: Cleaning up existing original embeddings")
    print("="*60)
    
    try:
        chroma_manager.delete_collection("original_verses")
        print("✓ Deleted original_verses collection")
    except:
        print("No original_verses collection to delete")
    
    # 2. Generate original language embeddings
    print("\n" + "="*60)
    print("Step 2: Generating original language embeddings")
    print("="*60)
    
    model_manager = EmbeddingModelManager(cache_dir=str(models_path))
    context_builder = ContextBuilder(db_manager)
    
    pipeline = OriginalLanguageEmbeddingPipeline(
        db_manager=db_manager,
        chroma_manager=chroma_manager,
        model_manager=model_manager,
        context_builder=context_builder
    )
    
    print(f"\nGenerating embeddings for ~{expected_count:,} canonical verses...")
    print("This will create ONE embedding per verse using Hebrew/Greek text")
    
    results = pipeline.embed_original_verses(
        batch_size=100,
        force_reembed=True  # Force since we cleaned up
    )
    
    if results.get('status') == 'already_embedded':
        print("\nOriginal verses already embedded")
    else:
        print(f"\n✓ Successfully embedded {results['verses_embedded']:,} canonical verses")
        if results.get('errors'):
            print(f"⚠️  Encountered {len(results['errors'])} errors:")
            for error in results['errors'][:5]:
                print(f"  - {error}")
    
    # 3. Verify the results (without creating new ChromaDB instance)
    print("\n" + "="*60)
    print("Step 3: Verification")
    print("="*60)
    
    stats = chroma_manager.get_database_stats()
    for collection_name, info in stats['collections'].items():
        print(f"\n{collection_name}:")
        print(f"  Count: {info.get('count', 0):,}")
        print(f"  Dimensions: {info.get('dimensions', 0)}")
        if 'metadata' in info:
            print(f"  Model: {info['metadata'].get('model', 'N/A')}")
            if collection_name == 'original_verses':
                print(f"  Type: {info['metadata'].get('type', 'N/A')}")
                print(f"  Languages: {info['metadata'].get('languages', 'N/A')}")
    
    # 4. Compare counts
    print("\n" + "="*60)
    print("Step 4: Count Comparison")
    print("="*60)
    
    original_verses_count = stats['collections'].get('original_verses', {}).get('count', 0)
    print(f"\nExpected verses: {expected_count:,}")
    print(f"Actual embeddings: {original_verses_count:,}")
    
    if original_verses_count < expected_count:
        print(f"⚠️  Missing {expected_count - original_verses_count:,} verses")
        print("\nInvestigating missing verses...")
        
        # Check which books might be missing
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get book IDs that were skipped (book_id = 0)
            cursor.execute("""
                SELECT book, COUNT(*) as count
                FROM stepbible_verses
                WHERE original_word IS NOT NULL AND original_word != ''
                AND book NOT IN ('Gen','Exo','Lev','Num','Deu','Jos','Jdg','Rut','1Sa','2Sa',
                                 '1Ki','2Ki','1Ch','2Ch','Ezr','Neh','Est','Job','Psa','Pro',
                                 'Ecc','Son','Isa','Jer','Lam','Eze','Dan','Hos','Joe','Amo',
                                 'Oba','Jon','Mic','Nah','Hab','Zep','Hag','Zec','Mal',
                                 'Mat','Mar','Luk','Joh','Act','Rom','1Co','2Co','Gal','Eph',
                                 'Phi','Col','1Th','2Th','1Ti','2Ti','Tit','Phm','Heb','Jam',
                                 '1Pe','2Pe','1Jn','2Jn','3Jn','Jud','Rev')
                GROUP BY book
            """)
            
            unmapped = cursor.fetchall()
            if unmapped:
                print("\nUnmapped book names:")
                for book, count in unmapped:
                    print(f"  {book}: {count} verses")
    else:
        print("✅ All verses successfully embedded!")

if __name__ == "__main__":
    main()