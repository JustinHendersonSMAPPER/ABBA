"""Pipeline for generating embeddings from original Hebrew/Greek texts."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .chroma_manager import ChromaManager
from .context_builder import ContextBuilder
from .model_manager import EmbeddingModelManager

logger = logging.getLogger(__name__)


class OriginalLanguageEmbeddingPipeline:
    """Manages embeddings for original Hebrew/Greek texts only."""

    def __init__(
        self,
        db_manager,
        chroma_manager: Optional[ChromaManager] = None,
        model_manager: Optional[EmbeddingModelManager] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """Initialize embedding pipeline for original languages.

        Args:
            db_manager: SQLiteManager for database access
            chroma_manager: ChromaManager instance (creates default if None)
            model_manager: EmbeddingModelManager instance (creates default if None)
            context_builder: ContextBuilder instance (creates default if None)
        """
        self.db = db_manager

        # Initialize components
        self.chroma = chroma_manager or ChromaManager()
        self.models = model_manager or EmbeddingModelManager()
        self.context = context_builder or ContextBuilder(db_manager)

        # Progress tracking
        self.progress_file = Path("bible_data/.embedding_progress.json")
        self.progress = self._load_progress()

        # Log GPU status at initialization
        self._log_gpu_status()

    def embed_original_verses(self, batch_size: int = 100, force_reembed: bool = False) -> Dict[str, Any]:  # noqa: C901
        """Generate embeddings for original language verses.

        This creates ONE embedding per canonical verse using the original
        Hebrew/Greek text, not separate embeddings for each translation.

        Args:
            batch_size: Number of verses to process at once
            force_reembed: Force re-embedding even if already done

        Returns:
            Summary of embedding results
        """
        # If rebuilding, clear existing embeddings
        if force_reembed:
            logger.info("Clearing existing verse embeddings for rebuild...")
            self.chroma.delete_collection("original_verses")

        # Get verses collection with appropriate metadata
        verses_collection = self.chroma.get_or_create_collection(
            "original_verses",
            metadata={
                "dimensions": 768,  # Multilingual model for Hebrew/Greek
                "model": "intfloat/multilingual-e5-base",
                "type": "original_biblical_verses",
                "languages": "hebrew,greek,aramaic",  # ChromaDB doesn't support lists
            },
        )

        # Check if already embedded
        if not force_reembed and self._are_original_verses_embedded():
            logger.info("Original verses already embedded")
            return {"status": "already_embedded"}

        # Get existing IDs to avoid duplicates
        existing_ids = set()
        if not force_reembed:
            try:
                # Get all existing IDs in batches
                offset = 0
                limit = 1000
                while True:
                    existing_results = verses_collection.get(
                        limit=limit,
                        offset=offset,
                        include=[],  # Don't need embeddings or metadata
                    )
                    if not existing_results["ids"]:
                        break
                    existing_ids.update(existing_results["ids"])
                    offset += limit
                logger.info("Found %d existing embeddings to skip", len(existing_ids))
            except Exception as e:
                logger.warning("Could not retrieve existing IDs: %s", e)

        logger.info("Loading unique canonical verses from original languages...")

        # Get unique canonical verses with original text
        canonical_verses = self._get_canonical_verses()

        if not canonical_verses:
            logger.warning("No canonical verses found")
            return {"status": "no_verses"}

        logger.info("Embedding %d unique canonical verses...", len(canonical_verses))

        results: Dict[str, Any] = {"verses_embedded": 0, "errors": []}

        # Check for resume point
        start_offset = 0
        if not force_reembed:
            progress_info = self.progress.get("original_verses", {})
            if "last_count" in progress_info and not progress_info.get("complete", False):
                start_offset = progress_info["last_count"]
                logger.info("Resuming from verse %d...", start_offset)

        # Process in batches
        for i in tqdm(
            range(start_offset, len(canonical_verses), batch_size),
            desc="Embedding original verses",
            initial=start_offset // batch_size,
            total=len(canonical_verses) // batch_size,
        ):
            try:
                batch = canonical_verses[i : i + batch_size]

                contexts = []
                ids = []
                metadatas = []

                for verse in batch:
                    # Generate canonical ID
                    verse_id = f"{verse['book_id']:03d}:{verse['chapter']:03d}:{verse['verse']:03d}"

                    # Skip if already exists
                    if verse_id in existing_ids:
                        continue

                    # Build rich context from original language
                    context = self._build_original_context(verse)

                    if not context:
                        continue

                    contexts.append(context)
                    ids.append(verse_id)

                    # Create metadata
                    metadata = {
                        "book_id": verse["book_id"],
                        "chapter": verse["chapter"],
                        "verse": verse["verse"],
                        "testament": "old" if verse["book_id"] <= 39 else "new",
                        "language": verse.get("primary_language", "unknown"),
                        "has_hebrew": bool(verse.get("hebrew_text")),
                        "has_greek": bool(verse.get("greek_text")),
                        "has_aramaic": bool(verse.get("aramaic_text")),
                        "word_count": verse.get("word_count", 0),
                    }
                    metadatas.append(metadata)

                if contexts:
                    # Generate embeddings using multilingual model
                    embeddings = self.models.encode_texts(
                        contexts, model_type="multilingual", batch_size=32, show_progress=False
                    )

                    # Add to ChromaDB with retry logic
                    max_retries = 3
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            # Ensure embeddings are properly formatted
                            embedding_list = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

                            # Add in smaller sub-batches to avoid overwhelming ChromaDB
                            sub_batch_size = 20  # Smaller batches for stability
                            for j in range(0, len(embedding_list), sub_batch_size):
                                end_idx = min(j + sub_batch_size, len(embedding_list))
                                verses_collection.add(
                                    embeddings=embedding_list[j:end_idx],
                                    ids=ids[j:end_idx],
                                    metadatas=metadatas[j:end_idx],  # type: ignore[arg-type]
                                )

                            results["verses_embedded"] += len(contexts)
                            break
                        except Exception as e:
                            retry_count += 1
                            error_str = str(e)

                            # Skip if it's a dict callable error - likely corruption
                            if "'dict' object is not callable" in error_str:
                                logger.error("ChromaDB corruption detected, skipping batch at index %d", i)
                                results["errors"].append(f"ChromaDB corruption at batch {i}")
                                break

                            if retry_count >= max_retries:
                                logger.error("Failed to add batch after %d attempts: %s", max_retries, error_str)
                                results["errors"].append(f"Failed batch at index {i}: {error_str}")
                                break

                            logger.warning(
                                "ChromaDB add failed (attempt %d/%d): %s", retry_count, max_retries, error_str
                            )
                            import time

                            time.sleep(2)  # Longer wait between retries

                # Update progress
                self._update_progress("original_verses", "canonical", i + len(batch))

            except Exception as e:
                error_msg = f"Error processing batch at index {i}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Mark as complete
        self._mark_original_verses_complete()

        logger.info("Embedded %d canonical verses", results["verses_embedded"])

        return results

    def _get_canonical_verses(self) -> List[Dict[str, Any]]:
        """Get unique canonical verses with original language text.

        Returns:
            List of verse dictionaries with original text
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get unique verses by grouping all original language data
            # Note: stepbible_verses uses book name, need to convert to book_id
            query = """
                SELECT
                    CASE sv.book
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
                    sv.chapter,
                    sv.verse,
                    GROUP_CONCAT(CASE WHEN sv.language = 'hebrew' THEN sv.original_word END, ' ') as hebrew_text,
                    GROUP_CONCAT(CASE WHEN sv.language = 'greek' THEN sv.original_word END, ' ') as greek_text,
                    GROUP_CONCAT(CASE WHEN sv.language = 'aramaic' THEN sv.original_word END, ' ') as aramaic_text,
                    GROUP_CONCAT(sv.strongs_primary || ' ', '') as strongs_sequence,
                    GROUP_CONCAT(sv.morphology || ' ', '') as morphology_sequence,
                    GROUP_CONCAT(sv.english || ' ', '') as english_gloss,
                    COUNT(*) as word_count,
                    sv.book as book_name,
                    CASE
                        WHEN sv.book IN ('Gen','Exo','Lev','Num','Deu','Jos','Jdg','Rut','1Sa','2Sa',
                                        '1Ki','2Ki','1Ch','2Ch','Ezr','Neh','Est','Job','Psa','Pro',
                                        'Ecc','Sng','Isa','Jer','Lam','Ezk','Dan','Hos','Jol','Amo',
                                        'Oba','Jon','Mic','Nam','Hab','Zep','Hag','Zec','Mal') THEN 'hebrew'
                        ELSE 'greek'
                    END as primary_language
                FROM stepbible_verses sv
                WHERE sv.original_word IS NOT NULL
                  AND sv.original_word != ''
                GROUP BY sv.book, sv.chapter, sv.verse
                ORDER BY book_id, sv.chapter, sv.verse
            """

            cursor.execute(query)

            verses = []
            for row in cursor.fetchall():
                verses.append(
                    {
                        "book_id": row[0],
                        "chapter": row[1],
                        "verse": row[2],
                        "hebrew_text": row[3],
                        "greek_text": row[4],
                        "aramaic_text": row[5],
                        "strongs_sequence": row[6],
                        "morphology_sequence": row[7],
                        "english_gloss": row[8],
                        "word_count": row[9],
                        "book_name": row[10],
                        "primary_language": row[11],
                    }
                )

            return verses

    def _build_original_context(self, verse: Dict[str, Any]) -> Optional[str]:
        """Build rich context from original language verse data.

        Args:
            verse: Dictionary with original language data

        Returns:
            Formatted context string for embedding
        """
        # Get the primary original text
        if verse.get("hebrew_text"):
            primary_text = verse["hebrew_text"]
            language = "Hebrew"
        elif verse.get("greek_text"):
            primary_text = verse["greek_text"]
            language = "Greek"
        elif verse.get("aramaic_text"):
            primary_text = verse["aramaic_text"]
            language = "Aramaic"
        else:
            return None

        # Build context with multiple layers of information
        context_parts = [
            f"{language}: {primary_text}",
            f"Gloss: {verse.get('english_gloss', '')}",
            f"Strong's: {verse.get('strongs_sequence', '')}",
            f"Morphology: {verse.get('morphology_sequence', '')}",
        ]

        # Add reference for context
        # Use book_name from query or fall back to book_id
        book_name = verse.get("book_name") or f"Book{verse['book_id']}"
        context_parts.insert(0, f"{book_name} {verse['chapter']}:{verse['verse']}")

        return " | ".join(filter(None, context_parts))

    def _are_original_verses_embedded(self) -> bool:
        """Check if original verses are already embedded."""
        return self.progress.get("original_verses", {}).get("complete", False)  # type: ignore[no-any-return]

    def _mark_original_verses_complete(self):
        """Mark original verses as complete."""
        if "original_verses" not in self.progress:
            self.progress["original_verses"] = {}

        self.progress["original_verses"]["canonical"] = {"complete": True, "timestamp": datetime.now().isoformat()}

        self._save_progress()

    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception as e:
                logger.error("Error loading progress: %s", e)

        return {"verses": {}, "words": {}, "concepts": {}, "original_verses": {}}

    def _save_progress(self):
        """Save progress to file."""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error("Error saving progress: %s", e)

    def _update_progress(self, category: str, key: str, count: int):
        """Update progress for a category."""
        if category not in self.progress:
            self.progress[category] = {}

        if key not in self.progress[category]:
            self.progress[category][key] = {}

        self.progress[category][key]["last_count"] = count
        self.progress[category][key]["last_update"] = datetime.now().isoformat()

        # Save more frequently for better crash recovery
        if count % 100 == 0:
            self._save_progress()

    def _log_gpu_status(self):
        """Log GPU availability and status information."""
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("GPU detected: %s (%.1fGB)", gpu_name, gpu_memory)
            logger.info("Embeddings will be generated using GPU acceleration")
        else:
            logger.info("No GPU detected - using CPU for embedding generation")
            logger.info("Note: CPU embedding generation is significantly slower than GPU")
