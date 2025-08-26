"""ChromaDB vector database manager for ABBA."""

import os
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Completely disable ChromaDB telemetry and logging
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'True'
os.environ['CHROMA_SERVER_NOFILE'] = '1'
os.environ['CHROMA_CLIENT_AUTH_PROVIDER'] = ''

# Suppress all ChromaDB warnings and telemetry messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*chroma_server_nofile.*")

# Suppress logging from ChromaDB modules
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.config").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

# Monkey patch to block telemetry at the source
def _silence_telemetry():
    """Completely silence ChromaDB telemetry."""
    try:
        import chromadb.telemetry.product.posthog as posthog
        # Replace the capture method with a no-op
        if hasattr(posthog, 'Posthog'):
            original_capture = posthog.Posthog.capture
            def silent_capture(self, *args, **kwargs):
                pass  # Do nothing
            posthog.Posthog.capture = silent_capture
    except ImportError:
        pass  # ChromaDB telemetry module not available

# Apply telemetry silencing
_silence_telemetry()

logger = logging.getLogger(__name__)


class ChromaManager:
    """Manages ChromaDB instance and collections for biblical text embeddings."""
    
    def __init__(self, persist_path: str = "bible_data/vectors"):
        """Initialize ChromaDB manager with persistent storage.
        
        Args:
            persist_path: Directory path for ChromaDB storage
        """
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    persist_directory=str(self.persist_path)
                )
            )
        except Exception as e:
            # If telemetry error occurs, try again with minimal settings
            logger.warning(f"Initial ChromaDB setup failed: {e}, retrying with minimal settings")
            self.client = chromadb.PersistentClient(path=str(self.persist_path))
        
        # Cache for collections
        self._collections: Dict[str, chromadb.Collection] = {}
        
        logger.info(f"Initialized ChromaDB at {self.persist_path}")
    
    def get_or_create_collection(
        self, 
        name: str, 
        embedding_function: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """Get existing collection or create new one.
        
        Args:
            name: Collection name
            embedding_function: Optional embedding function
            metadata: Collection metadata including index settings
            
        Returns:
            ChromaDB collection instance
        """
        # Check cache first
        if name in self._collections:
            return self._collections[name]
        
        # Set default metadata
        if metadata is None:
            metadata = {"hnsw:space": "cosine"}
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=name,
                embedding_function=embedding_function
            )
            logger.info(f"Retrieved existing collection: {name}")
        except ValueError:
            # Create new collection if doesn't exist
            collection = self.client.create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata=metadata
            )
            logger.info(f"Created new collection: {name}")
        
        # Cache the collection
        self._collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """Get existing collection without creating if it doesn't exist.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection instance or None if doesn't exist
        """
        # Check cache first
        if name in self._collections:
            return self._collections[name]
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=name)
            # Cache the collection
            self._collections[name] = collection
            logger.info(f"Retrieved collection: {name}")
            return collection
        except ValueError:
            # Collection doesn't exist
            logger.warning(f"Collection {name} does not exist")
            return None
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Returns:
            True if deleted, False if didn't exist
        """
        try:
            self.client.delete_collection(name)
            # Remove from cache
            if name in self._collections:
                del self._collections[name]
            logger.info(f"Deleted collection: {name}")
            return True
        except ValueError:
            logger.warning(f"Collection {name} does not exist")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collection names.
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.get_or_create_collection(name)
            count = collection.count()
            
            # Get a sample to check dimensions
            sample = collection.peek(1)
            dimensions = len(sample['embeddings'][0]) if sample['embeddings'] else 0
            
            return {
                "name": name,
                "count": count,
                "dimensions": dimensions,
                "metadata": collection.metadata
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting stats for collection {name}: {error_msg}")
            
            # If database is corrupted, return empty stats
            if "database disk image is malformed" in error_msg:
                logger.warning(f"Collection {name} appears corrupted - returning empty stats")
                return {
                    "name": name,
                    "count": 0,
                    "dimensions": 0,
                    "error": "Database corrupted"
                }
            
            return {"name": name, "error": error_msg}
    
    def reset_all(self, confirm: bool = False) -> bool:
        """Reset all collections (dangerous operation).
        
        Args:
            confirm: Must be True to actually reset
            
        Returns:
            True if reset, False otherwise
        """
        if not confirm:
            logger.warning("Reset requested but not confirmed")
            return False
        
        try:
            # Clear cache
            self._collections.clear()
            
            # Reset ChromaDB
            self.client.reset()
            
            logger.info("Reset all ChromaDB collections")
            return True
        except Exception as e:
            logger.error(f"Error resetting ChromaDB: {e}")
            return False
    
    # ID generation methods for consistent identifiers
    
    def generate_verse_id(self, translation_id: str, book_id: int, 
                         chapter: int, verse: int) -> str:
        """Generate deterministic ID for a verse.
        
        Args:
            translation_id: Translation identifier (e.g., 'eng_kjv')
            book_id: Canonical book number (1-66)
            chapter: Chapter number
            verse: Verse number
            
        Returns:
            Unique verse identifier
        """
        return f"{translation_id}:{book_id:03d}:{chapter:03d}:{verse:03d}"
    
    def generate_word_id(self, strongs: str, morphology: Optional[str] = None) -> str:
        """Generate deterministic ID for a word.
        
        Args:
            strongs: Strong's number (e.g., 'G3056')
            morphology: Optional morphology code
            
        Returns:
            Unique word identifier
        """
        if morphology:
            return f"{strongs}:{morphology}"
        return strongs
    
    def generate_concept_id(self, concept_name: str, language: str = "en") -> str:
        """Generate deterministic ID for a concept.
        
        Args:
            concept_name: Concept name (e.g., 'love', 'forgiveness')
            language: Language code
            
        Returns:
            Unique concept identifier
        """
        # Normalize concept name
        normalized = concept_name.lower().replace(" ", "_")
        return f"concept:{language}:{normalized}"
    
    def parse_verse_id(self, verse_id: str) -> Dict[str, Any]:
        """Parse a verse ID back into components.
        
        Args:
            verse_id: Verse identifier
            
        Returns:
            Dictionary with translation_id, book_id, chapter, verse
        """
        parts = verse_id.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid verse ID format: {verse_id}")
        
        return {
            "translation_id": parts[0],
            "book_id": int(parts[1]),
            "chapter": int(parts[2]),
            "verse": int(parts[3])
        }
    
    def parse_word_id(self, word_id: str) -> Dict[str, Any]:
        """Parse a word ID back into components.
        
        Args:
            word_id: Word identifier
            
        Returns:
            Dictionary with strongs and optional morphology
        """
        parts = word_id.split(":")
        result = {"strongs": parts[0]}
        
        if len(parts) > 1:
            result["morphology"] = parts[1]
        
        return result
    
    def search_verses(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar verses.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Search results with verses and scores
        """
        collection = self.get_or_create_collection("verses")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        return self._format_search_results(results)
    
    def search_words(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar words.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Search results with words and scores
        """
        collection = self.get_or_create_collection("words")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        return self._format_search_results(results)
    
    def _format_search_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format ChromaDB search results for easier use.
        
        Args:
            raw_results: Raw results from ChromaDB
            
        Returns:
            Formatted results dictionary
        """
        if not raw_results['ids'] or not raw_results['ids'][0]:
            return {"results": [], "count": 0}
        
        # ChromaDB returns nested lists, extract first query results
        ids = raw_results['ids'][0]
        distances = raw_results['distances'][0]
        metadatas = raw_results['metadatas'][0]
        
        # Convert distances to similarity scores (1 - cosine distance)
        similarities = [1 - d for d in distances]
        
        # Format results
        results = []
        for i, (id_, similarity, metadata) in enumerate(zip(ids, similarities, metadatas)):
            result = {
                "id": id_,
                "similarity": similarity,
                "metadata": metadata,
                "rank": i + 1
            }
            results.append(result)
        
        return {
            "results": results,
            "count": len(results)
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics.
        
        Returns:
            Dictionary with stats for all collections
        """
        stats = {
            "collections": {},
            "total_embeddings": 0,
            "persist_path": str(self.persist_path)
        }
        
        for collection_name in self.list_collections():
            collection_stats = self.get_collection_stats(collection_name)
            stats["collections"][collection_name] = collection_stats
            if "count" in collection_stats:
                stats["total_embeddings"] += collection_stats["count"]
        
        return stats
    
    def close(self):
        """Close ChromaDB connection and persist any pending changes."""
        try:
            # Clear collection cache
            self._collections.clear()
            
            # ChromaDB PersistentClient automatically persists
            # but we can force a sync by accessing heartbeat
            if hasattr(self.client, '_server'):
                self.client.heartbeat()
            
            logger.info("ChromaDB connection closed successfully")
        except Exception as e:
            logger.warning(f"Error closing ChromaDB: {e}")