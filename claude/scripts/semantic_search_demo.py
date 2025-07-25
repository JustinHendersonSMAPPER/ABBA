#!/usr/bin/env python3
"""Interactive semantic search demo showing current capabilities."""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database import SQLiteManager
from abba.embeddings import ChromaManager, EmbeddingModelManager


class SemanticSearchDemo:
    """Interactive demo of semantic search capabilities."""
    
    def __init__(self):
        """Initialize semantic search demo."""
        # Load configuration
        self.config = config_manager.load_config()
        
        # Initialize components
        self.db_manager = SQLiteManager(self.config.abba_db_path)
        self.chroma_manager = ChromaManager(persist_path=str(self.config.vectors_path))
        self.model_manager = EmbeddingModelManager(cache_dir=str(self.config.models_path))
        
        # Check if embeddings exist
        chroma_stats = self.chroma_manager.get_database_stats()
        self.verses_available = bool(chroma_stats.get('collections', {}).get('verses', {}).get('count', 0))
        self.words_available = bool(chroma_stats.get('collections', {}).get('words', {}).get('count', 0))
        
        print("=== ABBA Semantic Search Demo ===\n")
        print("Available search types:")
        print(f"  📖 Verse search: {'✓' if self.verses_available else '❌ (no embeddings)'}")
        print(f"  📝 Word search: {'✓' if self.words_available else '❌ (no embeddings)'}")
        
        if not (self.verses_available or self.words_available):
            print("\n⚠️  No embeddings found. Run 'python abba/main.py' first.")
            sys.exit(1)
    
    def search_verses(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for semantically similar verses."""
        if not self.verses_available:
            return []
        
        try:
            verses_collection = self.chroma_manager.get_collection("verses")
            
            # Generate query embedding
            query_embedding = self.model_manager.encode_single(
                query,
                model_type="english",
                normalize=True
            )
            
            # Search for similar verses
            results = verses_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for verse_id, distance, metadata in zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                ):
                    similarity_score = 1 - distance
                    formatted_results.append({
                        'id': verse_id,
                        'similarity': similarity_score,
                        'translation_id': metadata.get('translation_id', ''),
                        'book_name': metadata.get('book_name', 'Unknown'),
                        'chapter': metadata.get('chapter', 0),
                        'verse': metadata.get('verse', 0),
                        'text': metadata.get('text', ''),
                        'testament': metadata.get('testament', 'unknown')
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching verses: {e}")
            return []
    
    def search_words(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for semantically similar words."""
        if not self.words_available:
            return []
        
        try:
            words_collection = self.chroma_manager.get_collection("words")
            
            # Generate query embedding (use multilingual for Hebrew/Greek compatibility)
            query_embedding = self.model_manager.encode_single(
                query,
                model_type="multilingual",
                normalize=True
            )
            
            # Search for similar words
            results = words_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for word_id, distance, metadata in zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                ):
                    similarity_score = 1 - distance
                    formatted_results.append({
                        'id': word_id,
                        'similarity': similarity_score,
                        'strongs': metadata.get('strongs', ''),
                        'word': metadata.get('word', ''),
                        'transliteration': metadata.get('transliteration', ''),
                        'gloss': metadata.get('gloss', ''),
                        'morphology': metadata.get('morphology', ''),
                        'language': metadata.get('language', ''),
                        'part_of_speech': metadata.get('part_of_speech', '')
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching words: {e}")
            return []
    
    def display_verse_results(self, results: List[Dict[str, Any]], query: str):
        """Display verse search results."""
        print(f"\n📖 Verse Search Results for: \"{query}\"")
        print("-" * 80)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            similarity = result['similarity']
            book_name = result['book_name']
            chapter = result['chapter']
            verse = result['verse']
            translation = result['translation_id']
            text = result['text']
            testament = result['testament']
            
            # Color coding by similarity
            if similarity > 0.8:
                sim_icon = "🔥"  # Very similar
            elif similarity > 0.6:
                sim_icon = "⭐"  # Similar
            else:
                sim_icon = "📝"  # Somewhat similar
            
            print(f"{sim_icon} {i}. {book_name} {chapter}:{verse} ({translation}) - {similarity:.3f}")
            print(f"      {testament.upper()} Testament")
            print(f"      \"{text}\"")
            print()
    
    def display_word_results(self, results: List[Dict[str, Any]], query: str):
        """Display word search results."""
        print(f"\n📝 Word Search Results for: \"{query}\"")
        print("-" * 80)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            similarity = result['similarity']
            word = result['word']
            transliteration = result['transliteration']
            strongs = result['strongs']
            gloss = result['gloss']
            language = result['language']
            morphology = result['morphology']
            pos = result['part_of_speech']
            
            # Language flag
            lang_flag = "🇬🇷" if language == "greek" else "🇮🇱" if language == "hebrew" else "🌐"
            
            print(f"{lang_flag} {i}. {word} ({transliteration}) - {strongs} - {similarity:.3f}")
            print(f"      {language.title()}: \"{gloss}\"")
            if morphology:
                print(f"      Morphology: {morphology}")
            if pos:
                print(f"      Part of Speech: {pos}")
            print()
    
    def run_predefined_demos(self):
        """Run predefined demo searches."""
        print("\n🎯 Running Predefined Demo Searches\n")
        
        # Verse search demos
        if self.verses_available:
            verse_demos = [
                ("love your enemies", "Love and forgiveness theme"),
                ("the kingdom of heaven", "Kingdom of heaven parables"),
                ("I am the resurrection", "Jesus' I AM statements"),
                ("peace be with you", "Peace and blessing"),
                ("faith hope and love", "Christian virtues")
            ]
            
            for query, description in verse_demos:
                print(f"\n🔍 Demo: {description}")
                results = self.search_verses(query, n_results=3)
                self.display_verse_results(results, query)
        
        # Word search demos
        if self.words_available:
            word_demos = [
                ("love", "Love-related words"),
                ("peace", "Peace-related words"),
                ("salvation", "Salvation concepts"),
                ("wisdom", "Wisdom and knowledge"),
                ("righteousness", "Righteousness concepts")
            ]
            
            for query, description in word_demos:
                print(f"\n🔍 Demo: {description}")
                results = self.search_words(query, n_results=3)
                self.display_word_results(results, query)
    
    def run_interactive_mode(self):
        """Run interactive search mode."""
        print("\n🎮 Interactive Search Mode")
        print("Commands: 'verses:<query>', 'words:<query>', 'demo', 'help', 'quit'")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nEnter search: ").strip()
                
                if not user_input or user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  verses:<your query>   - Search for similar verses")
                    print("  words:<your query>    - Search for similar words")
                    print("  demo                  - Run predefined demos")
                    print("  help                  - Show this help")
                    print("  quit                  - Exit")
                
                elif user_input.lower() == 'demo':
                    self.run_predefined_demos()
                
                elif user_input.startswith('verses:'):
                    query = user_input[7:].strip()
                    if query:
                        results = self.search_verses(query, n_results=5)
                        self.display_verse_results(results, query)
                    else:
                        print("Please provide a query after 'verses:'")
                
                elif user_input.startswith('words:'):
                    query = user_input[6:].strip()
                    if query:
                        results = self.search_words(query, n_results=5)
                        self.display_word_results(results, query)
                    else:
                        print("Please provide a query after 'words:'")
                
                else:
                    # Default to verse search
                    print(f"Assuming verse search for: \"{user_input}\"")
                    results = self.search_verses(user_input, n_results=5)
                    self.display_verse_results(results, user_input)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Run semantic search demo."""
    demo = SemanticSearchDemo()
    
    print("\nChoose mode:")
    print("1. Run predefined demos")
    print("2. Interactive search mode")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            demo.run_predefined_demos()
        elif choice == '2':
            demo.run_interactive_mode()
        elif choice == '3':
            demo.run_predefined_demos()
            demo.run_interactive_mode()
        else:
            print("Invalid choice. Running predefined demos...")
            demo.run_predefined_demos()
            
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()