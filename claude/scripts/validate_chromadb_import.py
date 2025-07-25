#!/usr/bin/env python3
"""Validate that all embedding data is being imported correctly into ChromaDB."""

import sys
from pathlib import Path
from typing import Dict, Any, List
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database import SQLiteManager
from abba.embeddings import ChromaManager


class ChromaDBValidator:
    """Validate ChromaDB import integrity."""
    
    def __init__(self):
        """Initialize validator with database connections."""
        self.config = config_manager.load_config()
        self.db_manager = SQLiteManager(self.config.abba_db_path)
        self.chroma_manager = ChromaManager(persist_path=str(self.config.vectors_path))
        
    def validate_verse_embeddings(self) -> Dict[str, Any]:
        """Validate verse embeddings are properly imported."""
        print("\n1. Validating Verse Embeddings")
        print("-" * 50)
        
        results = {
            "source_verses": 0,
            "embedded_verses": 0,
            "missing_verses": [],
            "metadata_issues": [],
            "sample_data": []
        }
        
        # Get verse count from source database for BSB translation
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count total BSB verses
            cursor.execute("""
                SELECT COUNT(*) 
                FROM verses 
                WHERE translation_id = 'BSB'
                AND text IS NOT NULL
                AND text != ''
            """)
            results["source_verses"] = cursor.fetchone()[0]
            
            # Get sample verses to check
            cursor.execute("""
                SELECT book_id, chapter, verse, text
                FROM verses
                WHERE translation_id = 'BSB'
                AND text IS NOT NULL
                ORDER BY book_id, chapter, verse
                LIMIT 5
            """)
            sample_verses = cursor.fetchall()
        
        # Check ChromaDB verse collection
        verses_collection = self.chroma_manager.get_collection("verses")
        if verses_collection:
            results["embedded_verses"] = verses_collection.count()
            
            # Validate sample verses are in ChromaDB
            for book_id, chapter, verse, text in sample_verses:
                # Generate expected ID
                expected_id = self.chroma_manager.generate_verse_id("BSB", book_id, chapter, verse)
                
                # Try to get this specific verse
                try:
                    specific_result = verses_collection.get(
                        ids=[expected_id],
                        include=["metadatas"]
                    )
                    
                    if specific_result['ids']:
                        metadata = specific_result['metadatas'][0]
                        sample_info = {
                            "id": expected_id,
                            "found": True,
                            "book_name": metadata.get('book_name', 'missing'),
                            "text_match": text[:50] == metadata.get('text', '')[:50]
                        }
                    else:
                        sample_info = {
                            "id": expected_id,
                            "found": False,
                            "book_id": book_id,
                            "chapter": chapter,
                            "verse": verse
                        }
                        results["missing_verses"].append(expected_id)
                        
                except Exception as e:
                    sample_info = {
                        "id": expected_id,
                        "error": str(e)
                    }
                    
                results["sample_data"].append(sample_info)
        
        # Print results
        print(f"Source verses (BSB): {results['source_verses']:,}")
        print(f"Embedded verses: {results['embedded_verses']:,}")
        print(f"Coverage: {results['embedded_verses']/results['source_verses']*100:.1f}%")
        
        if results["sample_data"]:
            print("\nSample verse validation:")
            for sample in results["sample_data"]:
                if sample.get("found"):
                    status = "✓" if sample.get("text_match") else "⚠ (text mismatch)"
                    print(f"  {sample['id']}: {status} - {sample['book_name']}")
                else:
                    print(f"  {sample['id']}: ❌ NOT FOUND")
        
        return results
    
    def validate_word_embeddings(self) -> Dict[str, Any]:
        """Validate word embeddings are properly imported."""
        print("\n2. Validating Word Embeddings")
        print("-" * 50)
        
        results = {
            "source_words": 0,
            "embedded_words": 0,
            "metadata_completeness": {},
            "sample_data": []
        }
        
        # Get unique word count from source database
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count unique Strong's + morphology combinations
            cursor.execute("""
                SELECT COUNT(DISTINCT strongs_primary || ':' || COALESCE(morphology_code, ''))
                FROM words
                WHERE strongs_primary IS NOT NULL
                AND strongs_primary != ''
            """)
            results["source_words"] = cursor.fetchone()[0]
            
            # Get sample words
            cursor.execute("""
                SELECT DISTINCT 
                    strongs_primary,
                    morphology_code,
                    greek_text,
                    hebrew_text,
                    transliteration,
                    language
                FROM words
                WHERE strongs_primary IS NOT NULL
                AND strongs_primary != ''
                LIMIT 10
            """)
            sample_words = cursor.fetchall()
        
        # Check ChromaDB word collection
        words_collection = self.chroma_manager.get_collection("words")
        if words_collection:
            results["embedded_words"] = words_collection.count()
            
            # Analyze metadata completeness
            sample_metadata = words_collection.peek(100)
            if sample_metadata['metadatas']:
                required_fields = ['strongs', 'word', 'transliteration', 'language']
                field_counts = {field: 0 for field in required_fields}
                
                for metadata in sample_metadata['metadatas']:
                    for field in required_fields:
                        if metadata.get(field) and metadata[field].strip():
                            field_counts[field] += 1
                
                results["metadata_completeness"] = {
                    field: f"{count}/{len(sample_metadata['metadatas'])} ({count/len(sample_metadata['metadatas'])*100:.1f}%)"
                    for field, count in field_counts.items()
                }
            
            # Validate sample words
            for strongs, morph, greek, hebrew, trans, lang in sample_words:
                word_id = self.chroma_manager.generate_word_id(strongs, morph)
                
                try:
                    specific_result = words_collection.get(
                        ids=[word_id],
                        include=["metadatas"]
                    )
                    
                    if specific_result['ids']:
                        metadata = specific_result['metadatas'][0]
                        sample_info = {
                            "id": word_id,
                            "found": True,
                            "word": metadata.get('word', ''),
                            "metadata_strongs": metadata.get('strongs', ''),
                            "expected_strongs": strongs
                        }
                    else:
                        sample_info = {
                            "id": word_id,
                            "found": False,
                            "strongs": strongs
                        }
                        
                except Exception as e:
                    sample_info = {
                        "id": word_id,
                        "error": str(e)
                    }
                    
                results["sample_data"].append(sample_info)
        
        # Print results
        print(f"Source unique words: {results['source_words']:,}")
        print(f"Embedded words: {results['embedded_words']:,}")
        print(f"Coverage: {results['embedded_words']/results['source_words']*100:.1f}%")
        
        if results["metadata_completeness"]:
            print("\nMetadata field completeness:")
            for field, stats in results["metadata_completeness"].items():
                print(f"  {field}: {stats}")
        
        return results
    
    def validate_embedding_dimensions(self) -> Dict[str, Any]:
        """Validate embedding dimensions are correct."""
        print("\n3. Validating Embedding Dimensions")
        print("-" * 50)
        
        results = {}
        
        # Check verse embeddings (should be 1024D)
        verses_collection = self.chroma_manager.get_collection("verses")
        if verses_collection:
            sample = verses_collection.peek(1)
            if sample['embeddings']:
                verse_dims = len(sample['embeddings'][0])
                results["verses"] = {
                    "dimensions": verse_dims,
                    "expected": 1024,
                    "correct": verse_dims == 1024
                }
                print(f"Verse embeddings: {verse_dims}D {'✓' if verse_dims == 1024 else '❌ (expected 1024D)'}")
        
        # Check word embeddings (should be 768D)
        words_collection = self.chroma_manager.get_collection("words")
        if words_collection:
            sample = words_collection.peek(1)
            if sample['embeddings']:
                word_dims = len(sample['embeddings'][0])
                results["words"] = {
                    "dimensions": word_dims,
                    "expected": 768,
                    "correct": word_dims == 768
                }
                print(f"Word embeddings: {word_dims}D {'✓' if word_dims == 768 else '❌ (expected 768D)'}")
        
        return results
    
    def validate_book_coverage(self) -> Dict[str, Any]:
        """Validate which books have embeddings."""
        print("\n4. Validating Book Coverage")
        print("-" * 50)
        
        results = {
            "total_books": 0,
            "embedded_books": 0,
            "missing_books": [],
            "book_verse_counts": {}
        }
        
        # Get all BSB books from source database
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT book_id, COUNT(*) as verse_count
                FROM verses
                WHERE translation_id = 'BSB'
                AND text IS NOT NULL
                GROUP BY book_id
                ORDER BY book_id
            """)
            source_books = cursor.fetchall()
            results["total_books"] = len(source_books)
        
        # Check which books are in ChromaDB
        verses_collection = self.chroma_manager.get_collection("verses")
        if verses_collection:
            # Get a large sample to find all books
            all_verses = verses_collection.get(
                limit=verses_collection.count(),
                include=["metadatas"]
            )
            
            embedded_books = {}
            for metadata in all_verses['metadatas']:
                book_name = metadata.get('book_name', 'unknown')
                if book_name not in embedded_books:
                    embedded_books[book_name] = 0
                embedded_books[book_name] += 1
            
            results["embedded_books"] = len(embedded_books)
            results["book_verse_counts"] = embedded_books
            
            # Check for missing books
            for book_id, verse_count in source_books:
                book_name = f"Book{book_id}"
                if book_name not in embedded_books:
                    results["missing_books"].append({
                        "book_id": book_id,
                        "expected_verses": verse_count
                    })
        
        # Print results
        print(f"Total books in BSB: {results['total_books']}")
        print(f"Books with embeddings: {results['embedded_books']}")
        
        if results["missing_books"]:
            print(f"\n⚠️  Missing {len(results['missing_books'])} books:")
            for missing in results["missing_books"][:5]:
                print(f"  Book{missing['book_id']}: {missing['expected_verses']} verses")
            if len(results["missing_books"]) > 5:
                print(f"  ... and {len(results['missing_books']) - 5} more")
        
        # Show embedded book range
        if results["book_verse_counts"]:
            book_nums = []
            for book_name, count in results["book_verse_counts"].items():
                if book_name.startswith("Book"):
                    try:
                        book_nums.append(int(book_name.replace("Book", "")))
                    except ValueError:
                        pass
            
            if book_nums:
                print(f"\nEmbedded book range: Book{min(book_nums)} to Book{max(book_nums)}")
                print(f"Books 43-66 (NT) present: {'Yes' if max(book_nums) >= 43 else 'No'}")
        
        return results
    
    def validate_metadata_integrity(self) -> Dict[str, Any]:
        """Check for data integrity issues in metadata."""
        print("\n5. Validating Metadata Integrity")
        print("-" * 50)
        
        results = {
            "verses": {"issues": []},
            "words": {"issues": []}
        }
        
        # Check verse metadata
        verses_collection = self.chroma_manager.get_collection("verses")
        if verses_collection:
            sample = verses_collection.peek(100)
            
            for i, metadata in enumerate(sample['metadatas']):
                issues = []
                
                # Check required fields
                required = ['translation_id', 'book_name', 'chapter', 'verse', 'text']
                for field in required:
                    if not metadata.get(field):
                        issues.append(f"missing {field}")
                
                # Check data types
                if metadata.get('chapter') and not isinstance(metadata['chapter'], int):
                    issues.append(f"chapter not int: {type(metadata['chapter'])}")
                
                if metadata.get('verse') and not isinstance(metadata['verse'], int):
                    issues.append(f"verse not int: {type(metadata['verse'])}")
                
                if issues and len(results["verses"]["issues"]) < 5:
                    results["verses"]["issues"].append({
                        "id": sample['ids'][i],
                        "issues": issues
                    })
        
        # Check word metadata
        words_collection = self.chroma_manager.get_collection("words")
        if words_collection:
            sample = words_collection.peek(100)
            
            for i, metadata in enumerate(sample['metadatas']):
                issues = []
                
                # Check for the strongs/gloss issue
                strongs = metadata.get('strongs', '')
                gloss = metadata.get('gloss', '')
                
                if not gloss and strongs and not strongs.startswith(('G', 'H')):
                    issues.append(f"gloss in strongs field: '{strongs[:30]}'")
                
                if not metadata.get('word'):
                    issues.append("missing word text")
                
                if issues and len(results["words"]["issues"]) < 5:
                    results["words"]["issues"].append({
                        "id": sample['ids'][i],
                        "issues": issues
                    })
        
        # Print results
        print("Verse metadata issues:")
        if results["verses"]["issues"]:
            for issue in results["verses"]["issues"]:
                print(f"  {issue['id']}: {', '.join(issue['issues'])}")
        else:
            print("  ✓ No issues found in sample")
        
        print("\nWord metadata issues:")
        if results["words"]["issues"]:
            for issue in results["words"]["issues"]:
                print(f"  {issue['id']}: {', '.join(issue['issues'])}")
        else:
            print("  ✓ No issues found in sample")
        
        return results
    
    def generate_summary_report(self, all_results: Dict[str, Any]) -> str:
        """Generate a summary report of all validations."""
        report = [
            "\n" + "="*60,
            "CHROMADB IMPORT VALIDATION SUMMARY",
            "="*60,
            ""
        ]
        
        # Overall status
        verse_coverage = all_results["verses"]["embedded_verses"] / all_results["verses"]["source_verses"]
        word_coverage = all_results["words"]["embedded_words"] / all_results["words"]["source_words"]
        
        if verse_coverage > 0.99 and word_coverage > 0.99:
            report.append("✅ OVERALL STATUS: Excellent - Near complete coverage")
        elif verse_coverage > 0.90 and word_coverage > 0.90:
            report.append("⚠️  OVERALL STATUS: Good - High coverage with some gaps")
        else:
            report.append("❌ OVERALL STATUS: Issues detected - Significant gaps")
        
        # Key metrics
        report.extend([
            "",
            "KEY METRICS:",
            f"  Verse Coverage: {verse_coverage*100:.1f}% ({all_results['verses']['embedded_verses']:,}/{all_results['verses']['source_verses']:,})",
            f"  Word Coverage: {word_coverage*100:.1f}% ({all_results['words']['embedded_words']:,}/{all_results['words']['source_words']:,})",
            f"  Book Coverage: {all_results['books']['embedded_books']}/{all_results['books']['total_books']} books",
            ""
        ])
        
        # Dimension validation
        report.append("EMBEDDING DIMENSIONS:")
        if all_results["dimensions"].get("verses", {}).get("correct"):
            report.append("  ✓ Verses: 1024D (correct)")
        else:
            report.append("  ❌ Verses: dimension mismatch")
            
        if all_results["dimensions"].get("words", {}).get("correct"):
            report.append("  ✓ Words: 768D (correct)")
        else:
            report.append("  ❌ Words: dimension mismatch")
        
        # Issues
        report.extend([
            "",
            "IDENTIFIED ISSUES:"
        ])
        
        if all_results["books"]["missing_books"]:
            report.append(f"  • {len(all_results['books']['missing_books'])} books missing embeddings")
        
        if all_results["metadata"]["words"]["issues"]:
            report.append("  • Word metadata has gloss/strongs field confusion")
        
        if all_results["verses"]["missing_verses"]:
            report.append(f"  • Some verses missing from embeddings")
        
        # Recommendations
        report.extend([
            "",
            "RECOMMENDATIONS:",
            "  1. Re-run embedding generation with --force-reembed to ensure completeness",
            "  2. Fix word metadata extraction to properly map gloss fields",
            "  3. Verify all 66 books are included in embedding generation"
        ])
        
        return "\n".join(report)
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("Starting ChromaDB Import Validation...")
        print("="*60)
        
        all_results = {
            "verses": self.validate_verse_embeddings(),
            "words": self.validate_word_embeddings(),
            "dimensions": self.validate_embedding_dimensions(),
            "books": self.validate_book_coverage(),
            "metadata": self.validate_metadata_integrity()
        }
        
        # Generate and print summary
        summary = self.generate_summary_report(all_results)
        print(summary)
        
        # Save detailed results
        results_path = Path("claude/chromadb_validation_results.json")
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_path}")
        
        return all_results


def main():
    """Run ChromaDB validation."""
    validator = ChromaDBValidator()
    results = validator.run_full_validation()
    
    # Return exit code based on results
    verse_coverage = results["verses"]["embedded_verses"] / results["verses"]["source_verses"]
    word_coverage = results["words"]["embedded_words"] / results["words"]["source_words"]
    
    if verse_coverage > 0.99 and word_coverage > 0.99:
        return 0  # Success
    else:
        return 1  # Issues found


if __name__ == "__main__":
    sys.exit(main())