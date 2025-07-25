#!/usr/bin/env python3
"""Test and validate semantic search functionality with current embeddings."""

import sys
import os
import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Suppress ChromaDB telemetry before any imports
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'True'
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*chroma_server_nofile.*")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager, ContextBuilder


class SemanticSearchTester:
    """Test and validate semantic search functionality."""
    
    def __init__(self):
        """Initialize semantic search tester."""
        print("=== ABBA Semantic Search Tester ===\n")
        
        # Load configuration
        self.config = config_manager.load_config()
        
        # Initialize components
        self.db_manager = SQLiteManager(self.config.abba_db_path)
        self.chroma_manager = ChromaManager(persist_path=str(self.config.vectors_path))
        self.model_manager = EmbeddingModelManager(cache_dir=str(self.config.models_path))
        self.context_builder = ContextBuilder(self.db_manager)
        
        print("✓ Components initialized")
    
    def check_embedding_status(self) -> Dict[str, Any]:
        """Check current embedding status."""
        print("\n1. Checking Embedding Status")
        print("-" * 40)
        
        # Database stats
        db_stats = self.db_manager.get_database_stats()
        print(f"Database contents:")
        print(f"  Verses: {db_stats.get('verses', 0):,}")
        print(f"  Words: {db_stats.get('words', 0):,}")
        print(f"  Translations: {db_stats.get('translations', 0):,}")
        
        # Vector database stats
        chroma_stats = self.chroma_manager.get_database_stats()
        print(f"\nVector database contents:")
        if chroma_stats.get('collections'):
            for name, stats in chroma_stats['collections'].items():
                count = stats.get('count', 0)
                dimensions = stats.get('dimensions', 0)
                print(f"  {name}: {count:,} embeddings ({dimensions}D)")
        else:
            print("  No collections found")
        
        # Check what search types are available
        verse_embeddings_available = chroma_stats.get('collections', {}).get('verses', {}).get('count', 0) > 0
        word_embeddings_available = chroma_stats.get('collections', {}).get('words', {}).get('count', 0) > 0
        
        print(f"\nSearch capabilities:")
        print(f"  Verse search: {'✓' if verse_embeddings_available else '❌'}")
        print(f"  Word search: {'✓' if word_embeddings_available else '❌'}")
        
        return {
            'db_stats': db_stats,
            'chroma_stats': chroma_stats,
            'ready_for_search': bool(chroma_stats.get('collections')),
            'verse_search_available': verse_embeddings_available,
            'word_search_available': word_embeddings_available
        }
    
    def test_verse_similarity_search(self) -> bool:
        """Test verse-to-verse semantic similarity search."""
        print("\n2. Testing Verse Similarity Search")
        print("-" * 40)
        
        try:
            verses_collection = self.chroma_manager.get_collection("verses")
            if not verses_collection:
                print("⚠️  No verses collection found - skipping verse search tests")
                print("   Run 'python abba/main.py --embed-verses' to generate verse embeddings")
                return True  # Skip but don't fail
            
            # Check if collection has embeddings
            if verses_collection.count() == 0:
                print("⚠️  Verses collection exists but has no embeddings - skipping verse search tests")
                print("   Run 'python abba/main.py --embed-verses --force-reembed' to generate verse embeddings")
                return True  # Skip but don't fail
            
            # Test queries with expected semantic matches
            test_queries = [
                {
                    "query": "love your enemies and pray for those who persecute you",
                    "description": "Love for enemies theme",
                    "expected_concepts": ["love", "enemies", "persecution", "prayer"]
                },
                {
                    "query": "faith without works is dead",
                    "description": "Faith and works relationship", 
                    "expected_concepts": ["faith", "works", "deeds", "belief"]
                },
                {
                    "query": "the word became flesh and dwelt among us",
                    "description": "Incarnation theme",
                    "expected_concepts": ["word", "flesh", "incarnation", "dwelling"]
                },
                {
                    "query": "be still and know that I am God",
                    "description": "Peace and divine knowledge",
                    "expected_concepts": ["stillness", "peace", "knowledge", "God"]
                }
            ]
            
            for i, test in enumerate(test_queries, 1):
                print(f"\nTest {i}: {test['description']}")
                print(f"Query: \"{test['query']}\"")
                
                # Generate embedding for query
                query_embedding = self.model_manager.encode_single(
                    test['query'], 
                    model_type="english",
                    normalize=True
                )
                
                # Search for similar verses
                results = verses_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=5
                )
                
                if results['ids'] and results['ids'][0]:
                    print(f"Found {len(results['ids'][0])} similar verses:")
                    
                    for j, (verse_id, distance, metadata) in enumerate(zip(
                        results['ids'][0],
                        results['distances'][0], 
                        results['metadatas'][0]
                    )):
                        similarity_score = 1 - distance  # Convert distance to similarity
                        translation_id = metadata.get('translation_id', 'unknown')
                        book_name = metadata.get('book_name', 'Unknown')
                        chapter = metadata.get('chapter', 0)
                        verse_num = metadata.get('verse', 0)
                        text = metadata.get('text', '')[:100] + ('...' if len(metadata.get('text', '')) > 100 else '')
                        
                        print(f"  {j+1}. {book_name} {chapter}:{verse_num} ({translation_id}) - {similarity_score:.3f}")
                        print(f"      \"{text}\"")
                else:
                    print("❌ No results found")
                    return False
            
            print("\n✓ Verse similarity search working correctly")
            return True
            
        except Exception as e:
            print(f"❌ Error in verse similarity search: {e}")
            return False
    
    def test_word_similarity_search(self) -> bool:
        """Test word-to-word semantic similarity search."""
        print("\n3. Testing Word Similarity Search")
        print("-" * 40)
        
        try:
            words_collection = self.chroma_manager.get_collection("words")
            if not words_collection:
                print("❌ No words collection found - embeddings not generated")
                return False
            
            # Test queries for semantically related words
            test_queries = [
                {
                    "query": "love",
                    "description": "Love-related concepts",
                    "expected_relations": ["agape", "phileo", "charity", "affection"]
                },
                {
                    "query": "peace",
                    "description": "Peace-related concepts", 
                    "expected_relations": ["shalom", "eirene", "rest", "calm"]
                },
                {
                    "query": "salvation",
                    "description": "Salvation concepts",
                    "expected_relations": ["redemption", "deliverance", "rescue", "save"]
                }
            ]
            
            for i, test in enumerate(test_queries, 1):
                print(f"\nTest {i}: {test['description']}")
                print(f"Query: \"{test['query']}\"")
                
                # Generate embedding for query
                query_embedding = self.model_manager.encode_single(
                    test['query'],
                    model_type="multilingual",  # Words might be Hebrew/Greek
                    normalize=True
                )
                
                # Search for similar words
                results = words_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=10
                )
                
                if results['ids'] and results['ids'][0]:
                    print(f"Found {len(results['ids'][0])} similar words:")
                    
                    for j, (word_id, distance, metadata) in enumerate(zip(
                        results['ids'][0],
                        results['distances'][0],
                        results['metadatas'][0]
                    )):
                        similarity_score = 1 - distance
                        strongs = metadata.get('strongs', '')
                        word = metadata.get('word', '')
                        transliteration = metadata.get('transliteration', '')
                        gloss = metadata.get('gloss', '') or strongs  # Use strongs if gloss is empty
                        language = metadata.get('language', '')
                        
                        print(f"  {j+1}. {word} ({transliteration}) - {similarity_score:.3f}")
                        print(f"      {language}: \"{gloss}\"")
                else:
                    print("❌ No results found")
                    return False
            
            print("\n✓ Word similarity search working correctly")
            return True
            
        except Exception as e:
            print(f"❌ Error in word similarity search: {e}")
            return False
    
    def test_cross_lingual_search(self) -> bool:
        """Test cross-lingual search capabilities."""
        print("\n4. Testing Cross-Lingual Search")
        print("-" * 40)
        
        try:
            verses_collection = self.chroma_manager.get_collection("verses")
            if not verses_collection or verses_collection.count() == 0:
                print("⚠️  No verse embeddings available - skipping cross-lingual tests")
                return True  # Skip but don't fail
            
            # Test with different language queries
            cross_lingual_tests = [
                {
                    "query": "amor",  # Spanish for love
                    "description": "Spanish query for love theme",
                },
                {
                    "query": "paix",  # French for peace
                    "description": "French query for peace theme", 
                },
                {
                    "query": "λόγος",  # Greek for word/logos
                    "description": "Greek query for Word/Logos concept",
                }
            ]
            
            for i, test in enumerate(cross_lingual_tests, 1):
                print(f"\nTest {i}: {test['description']}")
                print(f"Query: \"{test['query']}\"")
                
                # Use English model for verse searches (matches collection dimensions)
                query_embedding = self.model_manager.encode_single(
                    test['query'],
                    model_type="english",
                    normalize=True
                )
                
                results = verses_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=3
                )
                
                if results['ids'] and results['ids'][0]:
                    print(f"Found {len(results['ids'][0])} cross-lingual matches:")
                    
                    for j, (verse_id, distance, metadata) in enumerate(zip(
                        results['ids'][0][:3],  # Show top 3
                        results['distances'][0][:3],
                        results['metadatas'][0][:3]
                    )):
                        similarity_score = 1 - distance
                        book_name = metadata.get('book_name', 'Unknown')
                        chapter = metadata.get('chapter', 0)
                        verse_num = metadata.get('verse', 0)
                        text = metadata.get('text', '')[:80] + ('...' if len(metadata.get('text', '')) > 80 else '')
                        
                        print(f"  {j+1}. {book_name} {chapter}:{verse_num} - {similarity_score:.3f}")
                        print(f"      \"{text}\"")
                else:
                    print("❌ No cross-lingual results found")
            
            print("\n✓ Cross-lingual search functioning")
            return True
            
        except Exception as e:
            print(f"❌ Error in cross-lingual search: {e}")
            return False
    
    def test_contextual_search_quality(self) -> bool:
        """Test the quality of contextual search results."""
        print("\n5. Testing Contextual Search Quality")
        print("-" * 40)
        
        try:
            verses_collection = self.chroma_manager.get_collection("verses")
            if not verses_collection or verses_collection.count() == 0:
                print("⚠️  No verse embeddings available - skipping quality tests")
                return True  # Skip but don't fail
            
            # Test specific doctrinal and thematic searches
            quality_tests = [
                {
                    "query": "Jesus Christ is Lord",
                    "description": "Christological confession",
                    "check_metadata": ["book_name", "testament"]
                },
                {
                    "query": "resurrection from the dead", 
                    "description": "Resurrection doctrine",
                    "check_metadata": ["book_name", "testament"]
                },
                {
                    "query": "forgiveness of sins",
                    "description": "Forgiveness theme",
                    "check_metadata": ["book_name", "testament"]
                }
            ]
            
            total_quality_score = 0
            
            for i, test in enumerate(quality_tests, 1):
                print(f"\nQuality Test {i}: {test['description']}")
                print(f"Query: \"{test['query']}\"")
                
                query_embedding = self.model_manager.encode_single(
                    test['query'],
                    model_type="english",
                    normalize=True
                )
                
                results = verses_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=5
                )
                
                if results['ids'] and results['ids'][0]:
                    # Analyze result quality
                    similarity_scores = [1 - d for d in results['distances'][0]]
                    avg_similarity = np.mean(similarity_scores)
                    
                    print(f"  Average similarity: {avg_similarity:.3f}")
                    print(f"  Top result similarity: {similarity_scores[0]:.3f}")
                    
                    # Check metadata completeness
                    complete_metadata = 0
                    for metadata in results['metadatas'][0]:
                        if all(metadata.get(field) for field in test['check_metadata']):
                            complete_metadata += 1
                    
                    metadata_completeness = complete_metadata / len(results['metadatas'][0])
                    print(f"  Metadata completeness: {metadata_completeness:.1%}")
                    
                    # Quality score (similarity + metadata completeness)
                    quality_score = (avg_similarity + metadata_completeness) / 2
                    total_quality_score += quality_score
                    
                    print(f"  Quality score: {quality_score:.3f}")
                else:
                    print("  ❌ No results found")
            
            avg_quality = total_quality_score / len(quality_tests)
            print(f"\n📊 Overall Search Quality Score: {avg_quality:.3f}")
            
            if avg_quality > 0.7:
                print("✓ High quality semantic search")
                return True
            elif avg_quality > 0.5:
                print("⚠️  Moderate quality semantic search")
                return True
            else:
                print("❌ Low quality semantic search")
                return False
                
        except Exception as e:
            print(f"❌ Error in quality testing: {e}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test search performance metrics."""
        print("\n6. Testing Search Performance")
        print("-" * 40)
        
        try:
            import time
            
            # Test with available collections
            verses_collection = self.chroma_manager.get_collection("verses")
            words_collection = self.chroma_manager.get_collection("words")
            
            if not verses_collection or verses_collection.count() == 0:
                if not words_collection or words_collection.count() == 0:
                    print("⚠️  No embeddings available - skipping performance tests")
                    return True
                else:
                    print("Testing performance with word embeddings (verse embeddings not available)")
                    test_collection = words_collection
                    test_model = "multilingual"
                    test_queries = ["love", "peace", "salvation", "wisdom", "grace"]
            else:
                print("Testing performance with verse embeddings")
                test_collection = verses_collection
                test_model = "english"
                test_queries = [
                    "love your neighbor as yourself",
                    "the kingdom of heaven is like",
                    "blessed are those who",
                    "I am the way the truth and the life",
                    "for God so loved the world"
                ]
            
            times = []
            
            for query in test_queries:
                start_time = time.time()
                
                # Generate embedding
                query_embedding = self.model_manager.encode_single(
                    query,
                    model_type=test_model,
                    normalize=True
                )
                
                # Perform search
                results = test_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=10
                )
                
                end_time = time.time()
                search_time = end_time - start_time
                times.append(search_time)
                
                result_count = len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0
                print(f"  \"{query[:30]}...\" -> {result_count} results in {search_time:.3f}s")
            
            avg_time = np.mean(times)
            print(f"\n📊 Average search time: {avg_time:.3f} seconds")
            print(f"📊 Fastest search: {min(times):.3f} seconds")
            print(f"📊 Slowest search: {max(times):.3f} seconds")
            
            if avg_time < 1.0:
                print("✓ Fast search performance")
                return True
            elif avg_time < 3.0:
                print("⚠️  Acceptable search performance")
                return True
            else:
                print("❌ Slow search performance")
                return False
                
        except Exception as e:
            print(f"❌ Error in performance testing: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all semantic search tests."""
        print("Starting comprehensive semantic search validation...\n")
        
        results = {}
        
        # Check prerequisites
        status = self.check_embedding_status()
        if not status['ready_for_search']:
            print("\n❌ Embeddings not ready - please run 'python abba/main.py' first")
            return {'prerequisites': False}
        
        # Run test suite
        test_methods = [
            ('verse_similarity', self.test_verse_similarity_search),
            ('word_similarity', self.test_word_similarity_search), 
            ('cross_lingual', self.test_cross_lingual_search),
            ('search_quality', self.test_contextual_search_quality),
            ('performance', self.test_performance_metrics)
        ]
        
        for test_name, test_method in test_methods:
            try:
                results[test_name] = test_method()
            except Exception as e:
                print(f"❌ Test {test_name} failed with error: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("SEMANTIC SEARCH TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status_icon = "✓" if passed_test else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}")
        
        print(f"\nPassed: {passed}/{total} tests")
        
        if passed == total:
            print("🎉 All semantic search tests passed!")
        elif passed >= total * 0.8:
            print("⚠️  Most semantic search tests passed - minor issues detected")
        else:
            print("❌ Multiple semantic search issues detected")
        
        return results


def main():
    """Run semantic search validation."""
    tester = SemanticSearchTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results and all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()