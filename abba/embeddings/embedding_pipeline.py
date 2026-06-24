"""Pipeline for generating and managing embeddings for biblical texts."""

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


class EmbeddingPipeline:
    """Manages the complete pipeline for generating embeddings."""

    def __init__(
        self,
        db_manager,
        chroma_manager: Optional[ChromaManager] = None,
        model_manager: Optional[EmbeddingModelManager] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """Initialize embedding pipeline.

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

    def embed_verses(  # noqa: C901
        self, translation_ids: Optional[List[str]] = None, batch_size: int = 100, force_reembed: bool = False
    ) -> Dict[str, Any]:
        """Generate embeddings for verses.

        Args:
            translation_ids: List of translations to embed (None for all)
            batch_size: Number of verses to process at once
            force_reembed: Force re-embedding even if already done

        Returns:
            Summary of embedding results
        """
        # If rebuilding, clear existing embeddings
        if force_reembed:
            logger.info("Clearing existing verse embeddings for rebuild...")
            self.chroma.delete_collection("verses")

        # Get verses collection
        verses_collection = self.chroma.get_or_create_collection(
            "verses", metadata={"dimensions": 1024, "model": "intfloat/e5-large-v2", "type": "biblical_verses"}
        )

        # Get translations to process
        if translation_ids is None:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT translation_id FROM verses")
                translation_ids = [row[0] for row in cursor.fetchall()]

        results: Dict[str, Any] = {"translations_processed": 0, "verses_embedded": 0, "errors": []}

        for translation_id in translation_ids:
            try:
                # Check if already embedded
                if not force_reembed and self._is_translation_embedded(translation_id):
                    logger.info("Skipping %s - already embedded", translation_id)
                    continue

                # Check if partially embedded (interrupted)
                start_offset = 0
                if not force_reembed:
                    progress_info = self.progress.get("verses", {}).get(translation_id, {})
                    if "last_count" in progress_info and not progress_info.get("complete", False):
                        start_offset = progress_info["last_count"]
                        logger.info("Resuming %s from verse %d...", translation_id, start_offset)

                logger.info("Embedding verses for %s...", translation_id)

                # Get all verses for translation
                verses = self._get_verses_for_translation(translation_id)

                if not verses:
                    logger.warning("No verses found for %s", translation_id)
                    continue

                # Process in batches
                for i in tqdm(
                    range(start_offset, len(verses), batch_size),
                    desc=f"Embedding {translation_id}",
                    initial=start_offset // batch_size,
                    total=len(verses) // batch_size,
                ):
                    batch = verses[i : i + batch_size]

                    # Build contexts
                    contexts = []
                    ids = []
                    metadatas = []

                    for verse in batch:
                        # Generate ID first to check if already exists
                        verse_id = self.chroma.generate_verse_id(
                            verse["translation_id"], verse["book_id"], verse["chapter"], verse["verse"]
                        )

                        # Skip if already embedded (for resume safety)
                        if start_offset > 0:
                            try:
                                existing = verses_collection.get(ids=[verse_id])
                                if existing and existing["ids"]:
                                    continue
                            except Exception:  # noqa: S110 - resume check is best-effort
                                pass  # Continue if error checking

                        # Build enhanced context
                        context = self.context.build_verse_context(
                            verse["translation_id"], verse["book_id"], verse["chapter"], verse["verse"]
                        )

                        if not context:
                            continue

                        contexts.append(context)
                        ids.append(verse_id)

                        # Create metadata (ensure no None values)
                        metadata = {
                            "translation_id": verse.get("translation_id", "") or "",
                            "book_id": int(verse.get("book_id", 0)) if verse.get("book_id") else 0,
                            "chapter": int(verse.get("chapter", 0)) if verse.get("chapter") else 0,
                            "verse": int(verse.get("verse", 0)) if verse.get("verse") else 0,
                            "text": (verse.get("text", "") or "")[:500],  # Truncate for storage
                            "testament": "old" if (verse.get("book_id", 0) or 0) <= 39 else "new",
                            "book_name": verse.get("book_name", "") or "",
                        }
                        metadatas.append(metadata)

                    if contexts:
                        try:
                            # Generate embeddings
                            embeddings = self.models.encode_texts(
                                contexts, model_type="english", batch_size=32, show_progress=False
                            )

                            # Add to ChromaDB with retry logic
                            max_retries = 3
                            retry_count = 0
                            while retry_count < max_retries:
                                try:
                                    verses_collection.add(
                                        embeddings=embeddings.tolist(),
                                        ids=ids,
                                        metadatas=metadatas,  # type: ignore[arg-type]
                                    )
                                    results["verses_embedded"] += len(contexts)
                                    break
                                except Exception as e:
                                    retry_count += 1
                                    if retry_count >= max_retries:
                                        raise
                                    logger.warning(
                                        "ChromaDB add failed (attempt %d/%d): %s",
                                        retry_count,
                                        max_retries,
                                        str(e),
                                    )
                                    # Wait briefly before retry
                                    import time

                                    time.sleep(1)
                        except Exception as e:
                            # Log error but continue with next batch
                            error_msg = f"Error processing batch at index {i} for {translation_id}: {str(e)}"
                            logger.error(error_msg)
                            results["errors"].append(error_msg)
                            # Update progress anyway to allow resume
                            self._update_progress("verses", translation_id, i)

                    # Update progress
                    self._update_progress("verses", translation_id, i + len(batch))

                # Mark translation as complete
                self._mark_translation_embedded(translation_id)
                results["translations_processed"] += 1

                logger.info("Embedded %d verses for %s", len(verses), translation_id)

            except Exception as e:
                error_msg = f"Error embedding {translation_id}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        return results

    def embed_words(self, batch_size: int = 500, force_reembed: bool = False) -> Dict[str, Any]:
        """Generate embeddings for unique words.

        Args:
            batch_size: Number of words to process at once
            force_reembed: Force re-embedding even if already done

        Returns:
            Summary of embedding results
        """
        # If rebuilding, clear existing embeddings
        if force_reembed:
            logger.info("Clearing existing word embeddings for rebuild...")
            self.chroma.delete_collection("words")

        # Check if already done
        if not force_reembed and self._are_words_embedded():
            logger.info("Words already embedded")
            return {"status": "already_embedded"}

        # Get words collection
        words_collection = self.chroma.get_or_create_collection(
            "words", metadata={"dimensions": 768, "model": "intfloat/multilingual-e5-base", "type": "biblical_words"}
        )

        logger.info("Loading unique words...")

        # Get unique words with their forms
        unique_words = self._get_unique_words()

        if not unique_words:
            logger.warning("No words found to embed")
            return {"status": "no_words"}

        logger.info("Embedding %d unique word forms...", len(unique_words))

        results: Dict[str, Any] = {"words_embedded": 0, "errors": []}

        # Process in batches
        for i in tqdm(range(0, len(unique_words), batch_size), desc="Embedding words"):
            try:
                batch = unique_words[i : i + batch_size]

                contexts = []
                ids = []
                metadatas = []

                for word in batch:
                    # Build context
                    context = self.context.build_word_context(word)

                    if not context:
                        continue

                    contexts.append(context)

                    # Generate ID
                    word_id = self.chroma.generate_word_id(word["strongs_primary"], word.get("morphology_code", ""))
                    ids.append(word_id)

                    # Create metadata (ensure no None values)
                    metadata = {
                        "strongs": word.get("strongs_primary", "") or "",
                        "morphology": word.get("morphology_code", "") or "",
                        "language": word.get("language", "") or "",
                        "word": word.get("greek_text", "") or word.get("hebrew_text", "") or "",
                        "transliteration": word.get("transliteration", "") or "",
                        "gloss": word.get("gloss", "") or "",
                        "part_of_speech": word.get("part_of_speech", "") or "",
                    }
                    metadatas.append(metadata)

                if contexts:
                    # Generate embeddings
                    embeddings = self.models.encode_texts(
                        contexts, model_type="multilingual", batch_size=32, show_progress=False
                    )

                    # Add to ChromaDB
                    words_collection.add(
                        embeddings=embeddings.tolist(),
                        ids=ids,
                        metadatas=metadatas,  # type: ignore[arg-type]
                    )

                    results["words_embedded"] += len(contexts)

                # Update progress
                self._update_progress("words", "all", i + len(batch))

            except Exception as e:
                error_msg = f"Error in batch {i}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Mark words as complete
        self._mark_words_embedded()

        logger.info("Embedded %d unique word forms", results["words_embedded"])

        return results

    def remove_legacy_translation_embeddings(self) -> Dict[str, Any]:
        """Remove legacy per-translation verse embeddings.

        The original architecture created one embedding per (translation, verse) pair,
        resulting in ~13M embeddings.  The current architecture uses ONE embedding per
        canonical verse from the original Hebrew/Greek (~31K), stored in the
        ``original_verses`` collection.

        This method deletes the old ``verses`` collection to reclaim space.

        Returns:
            Summary with count of removed embeddings.
        """
        result: Dict[str, Any] = {"removed": False, "count": 0}

        collection = self.chroma.get_collection("verses")
        if collection is not None:
            count = collection.count()
            if count > 0:
                logger.info("Removing %d legacy translation-specific embeddings", count)
                self.chroma.delete_collection("verses")
                result = {"removed": True, "count": count}
            else:
                logger.info("Legacy verses collection exists but is empty; removing")
                self.chroma.delete_collection("verses")
                result = {"removed": True, "count": 0}
        else:
            logger.info("No legacy verses collection found — nothing to remove")

        return result

    def verify_deduplication(self) -> Dict[str, Any]:
        """Verify that embeddings are properly deduplicated.

        Checks that the ``original_verses`` collection contains at most one
        embedding per canonical verse and that no legacy ``verses`` collection
        exists.

        Returns:
            Verification summary.
        """
        report: Dict[str, Any] = {"passed": True, "checks": []}

        # Check 1: no legacy collection
        legacy = self.chroma.get_collection("verses")
        if legacy is not None and legacy.count() > 0:
            report["passed"] = False
            report["checks"].append(
                {
                    "name": "legacy_removed",
                    "passed": False,
                    "detail": f"Legacy verses collection has {legacy.count()} entries",
                }
            )
        else:
            report["checks"].append({"name": "legacy_removed", "passed": True, "detail": "No legacy verses collection"})

        # Check 2: original_verses count is reasonable (≤ ~31,102 canonical verses)
        orig = self.chroma.get_collection("original_verses")
        if orig is not None:
            count = orig.count()
            reasonable = count <= 35000  # generous upper bound
            report["checks"].append(
                {
                    "name": "canonical_count",
                    "passed": reasonable,
                    "detail": f"original_verses has {count} embeddings",
                }
            )
            if not reasonable:
                report["passed"] = False
        else:
            report["checks"].append(
                {"name": "canonical_count", "passed": True, "detail": "original_verses collection not yet created"}
            )

        return report

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about current embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        stats = {"collections": {}, "progress": self.progress, "models": {}}

        # Get ChromaDB stats
        db_stats = self.chroma.get_database_stats()
        stats["collections"] = db_stats["collections"]

        # Get model info
        for model_type in ["english", "multilingual"]:
            stats["models"][model_type] = self.models.get_model_info(model_type)

        return stats

    def _get_verses_for_translation(self, translation_id: str) -> List[Dict[str, Any]]:
        """Get all verses for a translation.

        Args:
            translation_id: Translation identifier

        Returns:
            List of verse dictionaries
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    v.translation_id,
                    v.book_id,
                    v.chapter,
                    v.verse,
                    v.text,
                    b.name as book_name
                FROM verses v
                LEFT JOIN books b ON v.translation_id = b.translation_id
                                  AND v.book_id = b.book_id
                WHERE v.translation_id = ?
                ORDER BY v.book_id, v.chapter, v.verse
            """

            cursor.execute(query, (translation_id,))

            verses = []
            for row in cursor.fetchall():
                verses.append(
                    {
                        "translation_id": row[0],
                        "book_id": row[1],
                        "chapter": row[2],
                        "verse": row[3],
                        "text": row[4],
                        "book_name": row[5] or f"Book{row[1]}",
                    }
                )

            return verses

    def _get_unique_words(self) -> List[Dict[str, Any]]:
        """Get unique words with their linguistic data.

        Returns:
            List of word dictionaries
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get unique combinations of Strong's + morphology from stepbible_verses
            query = """
                SELECT DISTINCT
                    sv.strongs_primary,
                    sv.morphology,
                    sv.language,
                    MAX(sv.original_word) as original_word,
                    MAX(sv.transliteration) as transliteration,
                    MAX(sv.english) as english,
                    MAX(l.gloss) as gloss,
                    MAX(l.part_of_speech) as part_of_speech,
                    COUNT(*) as frequency
                FROM stepbible_verses sv
                LEFT JOIN lexicon l ON sv.strongs_primary = l.strongs_number
                WHERE sv.strongs_primary IS NOT NULL
                  AND sv.strongs_primary != ''
                GROUP BY sv.strongs_primary, sv.morphology, sv.language
                ORDER BY frequency DESC
            """

            cursor.execute(query)

            words = []
            for row in cursor.fetchall():
                words.append(
                    {
                        "strongs_primary": row[0],
                        "morphology_code": row[1],
                        "language": row[2],
                        "original_word": row[3],  # Can be either Greek or Hebrew
                        "greek_text": row[3] if row[2] == "greek" else None,
                        "hebrew_text": row[3] if row[2] == "hebrew" else None,
                        "transliteration": row[4],
                        "english": row[5],
                        "gloss": row[6] or row[5],  # Use English if no gloss
                        "part_of_speech": row[7],
                        "frequency": row[8],
                    }
                )

            return words

    # Progress tracking methods

    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception as e:
                logger.error("Error loading progress: %s", e)

        return {"verses": {}, "words": {}, "concepts": {}}

    def _save_progress(self):
        """Save progress to file."""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error("Error saving progress: %s", e)

    def _is_translation_embedded(self, translation_id: str) -> bool:
        """Check if translation is already embedded."""
        result: bool = self.progress.get("verses", {}).get(translation_id, {}).get("complete", False)
        return result

    def _mark_translation_embedded(self, translation_id: str):
        """Mark translation as complete."""
        if "verses" not in self.progress:
            self.progress["verses"] = {}

        self.progress["verses"][translation_id] = {
            "complete": True,
            "timestamp": datetime.now().isoformat(),
            "count": self._get_verse_count(translation_id),
        }

        self._save_progress()

    def _are_words_embedded(self) -> bool:
        """Check if words are already embedded."""
        return self.progress.get("words", {}).get("complete", False)  # type: ignore[no-any-return]

    def _mark_words_embedded(self):
        """Mark words as complete."""
        self.progress["words"] = {"complete": True, "timestamp": datetime.now().isoformat()}

        self._save_progress()

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

    def _get_verse_count(self, translation_id: str) -> int:
        """Get verse count for a translation."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM verses WHERE translation_id = ?", (translation_id,))
            return cursor.fetchone()[0]  # type: ignore[no-any-return]

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
