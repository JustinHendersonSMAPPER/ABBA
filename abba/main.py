#!/usr/bin/env python3
"""
Main entry point for ABBA - Annotated Bible and Background Analysis.

Initializes database, downloads STEPBible data, and imports biblical texts
for linguistic and semantic analysis.
"""

import multiprocessing
import sys
import time

from abba.bible_extractor import BibleExtractor
from abba.cli import cli_config
from abba.config import config_manager
from abba.database import SQLiteManager
from abba.database.import_tracker import ImportTracker
from abba.database.original_embedding_validator import OriginalEmbeddingValidator
from abba.database.post_import_validator import PostImportValidator
from abba.embeddings import ChromaManager, ContextBuilder, EmbeddingModelManager, EmbeddingPipeline
from abba.embeddings.original_language_pipeline import OriginalLanguageEmbeddingPipeline
from abba.logging_setup import configure_standard_logging, get_logger, setup_logging
from abba.operation_manager import OperationManager
from abba.stepbible_updater import STEPBibleUpdater


def main():  # noqa: C901  # pyright: ignore[reportGeneralTypeIssues]  # large CLI dispatcher; pyright bails on flow-analysis depth
    """Main application entry point."""
    try:
        # Load configuration from all sources
        config = config_manager.load_config()

        # Setup logging first
        setup_logging(log_level=config.log_level)
        configure_standard_logging()

        logger = get_logger(__name__)

        # Handle --serve: start API server only (skip pipeline)
        if cli_config.should_serve():
            import uvicorn

            from abba.api.app import create_app

            db_path = config.abba_db_path
            if not db_path.exists():
                logger.error(f"Database not found at {db_path}. Run the import pipeline first.")
                sys.exit(1)

            host = cli_config.get_host()
            port = cli_config.get_port()
            logger.info(f"Starting ABBA API server at http://{host}:{port}")
            logger.info(f"Using database: {db_path}")
            logger.info("API docs available at /docs")

            app = create_app(db_path=db_path)
            uvicorn.run(app, host=host, port=port)
            return None

        # Handle purge-all first before any other operations
        if cli_config.should_purge_all():
            if config.should_show_output():
                logger.warning("⚠️  WARNING: This will delete ALL data, embeddings, and tracking files!")
                logger.info("The following will be removed:")
                logger.info(f"  - {config.abba_db_path}")
                logger.info(f"  - {config.vectors_path}/")
                logger.info(f"  - {config.data_dir}/.import_status.json")
                logger.info(f"  - {config.data_dir}/.embedding_progress.json")
                logger.info(f"  - {config.data_dir}/.abba_state.json")

                # Ask for confirmation unless --yes flag is used
                if not cli_config.skip_confirmations():
                    response = input("\nAre you sure you want to continue? Type 'yes' to confirm: ")
                    if response.lower() != "yes":
                        logger.info("Purge cancelled.")
                        return None
                else:
                    logger.info("Skipping confirmation (--yes flag used)")

            # Perform the purge
            import shutil

            # Remove database
            if config.abba_db_path.exists():
                config.abba_db_path.unlink()
                if config.should_show_output():
                    logger.info(f"✓ Removed {config.abba_db_path}")

            # Remove vector database directory
            if config.vectors_path.exists():
                shutil.rmtree(config.vectors_path)
                if config.should_show_output():
                    logger.info(f"✓ Removed {config.vectors_path}/")

            # Remove tracking files
            tracking_files = [
                config.data_dir / ".import_status.json",  # Import tracker
                config.data_dir / ".import_progress.json",  # Legacy/alternate name
                config.data_dir / ".embedding_progress.json",  # Embedding tracker
                config.data_dir / ".abba_state.json",  # Operation state tracker
            ]

            for tracking_file in tracking_files:
                if tracking_file.exists():
                    tracking_file.unlink()
                    if config.should_show_output():
                        logger.info(f"✓ Removed {tracking_file}")

            if config.should_show_output():
                logger.info("✅ All data successfully purged. Starting fresh...")

        # (Logging already setup above)

        # Create directories
        config.create_directories()

        # Initialize database manager
        db_manager = SQLiteManager(config.abba_db_path)

        # Initialize or rebuild database if needed
        if cli_config.should_rebuild_db() or not config.abba_db_path.exists():
            if config.should_show_output():
                logger.info(f"Initializing ABBA database at {config.abba_db_path}...")
            db_manager.initialize_database()
        else:
            # Ensure database schema exists even if file exists
            try:
                stats = db_manager.get_database_stats()
                # If we can't get stats, the schema might be missing
            except Exception:
                if config.should_show_output():
                    logger.info("Database exists but schema missing, reinitializing...")
                db_manager.initialize_database()

        # Initialize operation manager for state tracking
        state_file = config.data_dir / ".abba_state.json"
        operation_manager = OperationManager(state_file, config.db_path)

        # Initialize extractor with config
        extractor = BibleExtractor(str(config.data_dir), config=config)
        extractor.operation_manager = operation_manager

        # Check for STEPBible updates if requested or rebuild requested
        force_stepbible_reimport = cli_config.should_rebuild_stepbible()
        stepbible_has_updates = False  # Track if reimport is due to updates

        if cli_config.should_check_for_updates():
            if config.should_show_output():
                logger.info("Checking for STEPBible data updates...")

            updater = STEPBibleUpdater(config.data_dir)
            has_updates, file_changes = updater.check_for_updates()

            if config.should_show_output():
                logger.info(updater.get_update_summary(file_changes))

            if has_updates:
                if config.should_show_output():
                    logger.info("\nSTEPBible data has been updated and will be re-imported.")
                force_stepbible_reimport = True
                stepbible_has_updates = True

        # Handle list command
        if cli_config.should_list():
            if not config.db_path.exists():
                logger.error("bible.db not found. Please download it first with --force-download.")
                return None

            translations = extractor.list_translations()
            if translations:
                logger.info(f"\nAvailable translations ({len(translations)}):")
                logger.info("-" * 60)
                for trans in translations:
                    logger.info(f"{trans['id']:10} {trans['language']:10} {trans['english_name']}")

                # Show database stats if database exists
                if config.abba_db_path.exists():
                    stats = db_manager.get_database_stats()
                    logger.info("\nABBA Database Stats:")
                    logger.info(f"Words: {stats.get('words', 0):,}")
                    logger.info(f"Verses: {stats.get('verses', 0):,}")
                    logger.info(f"Translations: {stats.get('translations', 0):,}")
                    logger.info(f"Lexicon entries: {stats.get('lexicon', 0):,}")
            else:
                logger.error("No translations found. Please download bible.db first.")
            return None

        # Handle concept data validation command
        if cli_config.should_validate_concept_data():
            from claude.scripts.concept_validator import ConceptValidator

            logger.info("Validating concept data against databases...")
            validator = ConceptValidator(config)
            results = validator.validate_all_concepts()
            validator.print_validation_report(results)
            validator.close()

            # Exit with appropriate code
            failed_count = sum(1 for r in results if not r.validation_passed)
            return failed_count == 0

        # Handle semantic search commands
        if cli_config.get_search_concept() or cli_config.get_export_concept_mappings():
            from abba.semantic.concept_mapper import ConceptMapper

            # Setup paths
            db_path = config.data_dir / "abba.db"
            chroma_path = config.vectors_path  # Use the correct vectors path

            # Ollama configuration
            ollama_config = {
                "host": config.ollama_host,
                "models": config.ollama_semantic_models,
                "consensus_threshold": config.ollama_consensus_threshold,
                "timeout": config.ollama_timeout,
            }

            mapper = ConceptMapper(db_path, chroma_path, ollama_config)

            # Handle concept search
            if cli_config.get_search_concept():
                concept_name = cli_config.get_search_concept()
                logger.info(f"Searching for concept: {concept_name}")

                matches = mapper.search_concept(concept_name)
                if matches:
                    logger.info(f"\nFound {len(matches)} matches for '{concept_name}':")

                    # Show top 10 matches
                    for i, match in enumerate(matches[:10], 1):
                        match_type = "Semantic" if match.is_semantic_only else "Lexical"
                        logger.info(f"\n{i}. {match.verse_id} ({match_type}, confidence: {match.confidence:.3f})")
                        logger.info(f"   {match.original_text[:60]}...")
                        logger.info(f"   Evidence: {match.evidence}")

                    if len(matches) > 10:
                        logger.info(f"\n... and {len(matches) - 10} more matches")
                else:
                    logger.warning(f"No matches found for '{concept_name}'")
                    logger.info("Try running --map-concepts first to process all concepts")

            # Handle export
            if cli_config.get_export_concept_mappings():
                output_path = cli_config.get_export_concept_mappings()
                fmt = "json" if output_path.endswith(".json") else "csv"

                mapper.export_mappings(output_path, output_format=fmt)
                logger.info(f"✅ Exported concept mappings to {output_path}")

            return None

        # Handle concept validation commands
        if (
            cli_config.should_validate_concepts()
            or cli_config.should_generate_concept_report()
            or cli_config.should_map_concepts()
        ):
            from abba.concept_validator import (  # type: ignore[import-untyped]  # noqa: E501, pylint: disable=no-name-in-module
                ConceptValidationPipeline,
            )

            concept_pipeline = ConceptValidationPipeline(config)

            # Test Ollama connection first
            if not concept_pipeline.test_ollama_connection():
                logger.error("Concept validation requires working Ollama connection")
                return None

            # Validate setup
            if not concept_pipeline.validate_setup():
                logger.error("Concept validation setup failed - see errors above")
                return None

            # Handle concept mapping with semantic concordance
            if cli_config.should_map_concepts():
                logger.info("Starting semantic concept mapping...")

                from abba.semantic.concept_mapper import ConceptMapper

                # Setup paths
                db_path = config.data_dir / "abba.db"
                chroma_path = config.vectors_path  # Use the correct vectors path

                # Ollama configuration
                ollama_config = {
                    "host": config.ollama_host,
                    "models": config.ollama_semantic_models,
                    "consensus_threshold": config.ollama_consensus_threshold,
                    "timeout": config.ollama_timeout,
                }

                mapper = ConceptMapper(db_path, chroma_path, ollama_config)

                # Process all concepts
                stats = mapper.process_all_concepts(
                    config.concepts_path,
                    max_semantic_per_concept=100,
                    validate_semantic=True,
                    force_reprocess=True,  # Always reprocess to avoid constraint issues
                )

                if stats:
                    logger.info(f"✅ Concept mapping completed for {len(stats)} concepts")

                    # Generate report if requested
                    if cli_config.should_generate_concept_report():
                        report = mapper.generate_report()
                        report_path = config.data_dir / f"concept_report_{time.strftime('%Y%m%d_%H%M%S')}.md"

                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(report)

                        logger.info(f"📄 Report saved to: {report_path}")
                else:
                    logger.warning("No concepts were successfully mapped")

            # Handle concept report only
            elif cli_config.should_generate_concept_report():
                logger.info("Generating concept validation report...")
                # Would need to load existing results from database
                logger.info("Report generation from existing data not yet implemented")

            # Handle concept validation only
            elif cli_config.should_validate_concepts():
                logger.info("Validating concept definitions...")
                concepts = concept_pipeline.list_concepts()
                logger.info(f"Found {len(concepts)} concepts: {', '.join(concepts)}")

                # Just validate the concept definitions, don't run LLM analysis
                validation_results = concept_pipeline.concept_manager.validate_concepts()
                if validation_results["errors"]:
                    logger.error("Concept validation errors:")
                    for error in validation_results["errors"]:
                        logger.error(f"  - {error}")
                else:
                    logger.info("✅ All concept definitions are valid")

            return None

        # Download bible.db if needed
        if config.should_download():
            if config.should_show_output():
                logger.info("Downloading bible.db...")

            if not extractor.download_bible_db():
                logger.error("Failed to download bible.db")
                sys.exit(1)

        # Download STEPBible data if needed (always download on first run, --download, or missing files)
        stepbible_dir = config.data_dir / "stepbible"
        stepbible_expected_files = [
            "tahot_gen_deu.txt",
            "tahot_jos_est.txt",
            "tahot_job_sng.txt",
            "tahot_isa_mal.txt",
            "tagnt_mat_jhn.txt",
            "tagnt_act_rev.txt",
            "hebrew_lexicon.txt",
            "greek_lexicon.txt",
            "hebrew_morphology.txt",
            "greek_morphology.txt",
        ]
        force_download_stepbible = (
            getattr(cli_config.args, "force_download_stepbible", False) if cli_config.args else False
        )
        if force_download_stepbible:
            # Delete existing files so download_stepbible_data() re-downloads them
            for f in stepbible_expected_files:
                fpath = stepbible_dir / f
                if fpath.exists():
                    fpath.unlink()
            stepbible_missing = stepbible_expected_files
        else:
            stepbible_missing = [f for f in stepbible_expected_files if not (stepbible_dir / f).exists()]
        if config.should_download() or stepbible_missing:
            if config.should_show_output():
                if stepbible_missing:
                    logger.info(
                        f"Downloading {len(stepbible_missing)} missing STEPBible file(s): {', '.join(stepbible_missing)}"
                    )
                else:
                    logger.info("Downloading STEPBible lexicon and morphology data...")

            if not extractor.download_stepbible_data():
                logger.warning("Failed to download STEPBible data (continuing without it)")
                # Don't exit - this is not critical for basic functionality

        # Check if bible.db exists
        if not config.db_path.exists():
            logger.error(f"Error: bible.db not found at {config.db_path}")
            logger.info("Please run with --force-download to download it.")
            sys.exit(1)

        # Initialize import tracker
        tracker = ImportTracker()

        # Check if we should force re-import
        force_reimport = cli_config.should_rebuild_db() or getattr(config, "rebuild_db", False)

        if force_reimport and config.should_show_output():
            logger.info("Force rebuild requested - will re-import all data")
            tracker.reset(confirm=True)

        # Import translations into database
        if config.should_show_output():
            logger.info("Importing biblical data into ABBA database...")

        # Import translations from bible.db
        translations = extractor.list_translations()
        if not translations:
            logger.error("No translations found in bible.db")
            sys.exit(1)

        logger.debug(f"Found {len(translations)} translations in bible.db")

        # Filter translations if specific ones requested
        if config.translations:
            # Filter to requested translations only
            translation_ids = set(config.translations)
            translations = [t for t in translations if t["id"] in translation_ids]

            # Check if all requested translations were found
            missing = translation_ids - {t["id"] for t in translations}
            if missing:
                logger.warning(f"Requested translations not found: {', '.join(missing)}")
                available_ids = [t["id"] for t in extractor.list_translations()]
                suffix = "..." if len(available_ids) > 10 else ""
                logger.info(f"Available translations include: {', '.join(available_ids[:10])}{suffix}")

        # Check which translations need importing
        translations_to_import = []
        for trans in translations:
            if not tracker.is_translation_imported(trans["id"]):
                translations_to_import.append(trans)
            elif config.verbose:
                import_time = tracker.get_translation_import_time(trans["id"])
                logger.debug(f"Skipping {trans['id']} - already imported at {import_time}")

        if not translations_to_import:
            if config.should_show_output():
                logger.info("All requested translations are already imported.")
        else:
            if config.verbose:
                logger.debug(f"Need to import {len(translations_to_import)} translations")

            # Import translation metadata for selected translations only
            for trans in translations_to_import:
                logger.debug(f"Importing metadata for {trans['english_name']} ({trans['id']})...")
                db_manager.insert_translation(trans)

            # Check for interrupted operations
            interrupted_warnings = operation_manager.handle_interrupted_operations()
            if interrupted_warnings and config.should_show_output():
                logger.warning("\nDetected interrupted operations:")
                for warning in interrupted_warnings:
                    logger.warning(f"  {warning}")
                logger.info("")

            # Import verses for selected translations using parallel processing
            translation_ids_to_import = [trans["id"] for trans in translations_to_import]

            # Use parallel import if enabled
            use_parallel = config.get_parallel_workers() > 1

            if config.should_show_output():
                if use_parallel:
                    logger.info(f"Using parallel import with {config.get_parallel_workers()} workers...")
                    logger.info("Parallelism: threads (I/O bound task)")
                else:
                    logger.info("Using sequential import...")
                logger.info(f"\nImporting {len(translation_ids_to_import)} translation(s)...")

            # Run import
            import_results = extractor.extract_translations_to_db_parallel(
                db_manager=db_manager, translation_ids=translation_ids_to_import, use_parallel=use_parallel
            )

            # Process results
            success_count = 0
            failed_translations = []

            for tid, result in import_results.items():
                if result.success:
                    success_count += 1
                    tracker.mark_translation_imported(tid)
                    if config.verbose:
                        logger.debug(f"✓ {tid}: {result.verse_count} verses in {result.duration:.1f}s")
                else:
                    failed_translations.append(tid)
                    logger.error(f"✗ {tid}: {result.error}")

            if config.should_show_output() and translations_to_import:
                total_time = sum(r.duration for r in import_results.values())
                total_verses = sum(r.verse_count for r in import_results.values() if r.success)
                logger.info(f"\nSuccessfully imported {success_count}/{len(translations_to_import)} translations")
                logger.info(f"Total verses: {total_verses:,}")
                logger.info(f"Total time: {total_time:.1f}s")
                if total_time > 0:
                    logger.info(f"Average rate: {total_verses / total_time:.0f} verses/second")
                if failed_translations:
                    logger.error(f"Failed translations: {', '.join(failed_translations)}")

            # Verify imports if requested
            if cli_config.args and hasattr(cli_config.args, "verify") and cli_config.args.verify:
                if config.should_show_output():
                    logger.info("\nVerifying imports with hash validation...")

                verify_results = extractor.verify_import_parallel(
                    db_manager=db_manager, translation_ids=translation_ids_to_import
                )

                invalid = [tid for tid, valid in verify_results.items() if not valid]
                if invalid:
                    logger.warning(f"\n⚠️  Validation failed for: {', '.join(invalid)}")
                else:
                    logger.info("\n✓ All imports validated successfully")

            # Run post-import validation
            if config.should_show_output():
                logger.info("\nRunning post-import validation...")

            validator = PostImportValidator(abba_db_path=config.abba_db_path, source_db_path=config.db_path)

            validation_summary = validator.validate_all_translations()

            if config.should_show_output():
                validator.print_summary(validation_summary)

            # Stop if validation failed
            if validation_summary.percentage < 100:
                logger.error("\n❌ Post-import validation failed. Stopping execution.")
                logger.error(f"   {validation_summary.failed_translations} translation(s) have issues.")
                logger.error("   Please review the failures above and fix any data issues.")
                sys.exit(1)

        # Print current database stats
        if config.should_show_output():
            stats = db_manager.get_database_stats()
            logger.info(f"Database now contains {stats.get('verses', 0):,} verses")

        # Import STEPBible data (Hebrew/Greek words, lexicon, morphology)
        stepbible_attribution = config.data_dir / "stepbible" / "ATTRIBUTION.txt"
        if stepbible_attribution.exists():
            # Check if STEPBible data needs importing or force re-import
            if force_stepbible_reimport or not tracker.is_stepbible_file_imported("complete", "all_stepbible_data"):
                if config.should_show_output():
                    if force_stepbible_reimport:
                        if stepbible_has_updates:
                            logger.info("Re-importing STEPBible data due to updates...")
                        elif cli_config.should_rebuild_stepbible():
                            logger.info("Rebuilding STEPBible data...")
                        else:
                            logger.info("Re-importing STEPBible data...")
                    else:
                        logger.info("Importing STEPBible lexicon and morphology data...")

                try:
                    if extractor.import_stepbible_data(db_manager, tracker, force_reimport=force_stepbible_reimport):
                        tracker.mark_stepbible_file_imported("complete", "all_stepbible_data")
                        if config.should_show_output():
                            updated_stats = db_manager.get_database_stats()
                            logger.info("STEPBible import complete:")
                            logger.info(f"  Words: {updated_stats.get('words', 0):,}")
                            logger.info(f"  Lexicon entries: {updated_stats.get('lexicon', 0):,}")
                            logger.info(f"  Morphology codes: {updated_stats.get('morphology', 0):,}")

                        # Validate STEPBible data
                        from abba.database.stepbible_validator import validate_stepbible_import

                        if not validate_stepbible_import(config.abba_db_path):
                            logger.error("STEPBible validation failed - data may be incomplete or corrupted")
                            if not cli_config.skip_confirmations():
                                response = input("\nContinue anyway? (y/N): ")
                                if response.lower() != "y":
                                    return 1
                    else:
                        logger.warning("STEPBible data import failed (continuing without it)")
                except Exception as e:
                    logger.error(f"Error importing STEPBible data: {e}")
            elif config.verbose:
                logger.debug("STEPBible data already imported - skipping")
        elif config.verbose:
            logger.debug("STEPBible data not available - skipping import")

        # Print import summary
        if config.should_show_output():
            summary = tracker.get_import_summary()
            logger.info("\nImport summary:")
            logger.info(f"  Translations: {summary['translations_imported']}")
            logger.info(f"  STEPBible files: {sum(summary['stepbible_files'].values())}")
            if summary["last_update"]:
                logger.info(f"  Last update: {summary['last_update']}")

        # Handle embedding generation
        # Check if embeddings should be generated (explicit CLI or automatic for missing data)
        explicit_embed_flags = (
            cli_config.should_embed_verses() or cli_config.should_embed_words() or cli_config.should_embed_all()
        )

        # Track what needs to be embedded
        auto_embed_verses = False
        auto_embed_words = False
        force_word_reembed = force_stepbible_reimport  # Re-embed words if STEPBible updated

        # Initialize components to check embedding status (needed for both checking and generating)
        chroma_manager = None

        # If no explicit embedding flags, check if embeddings are missing
        if not explicit_embed_flags:
            chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
            db_stats = db_manager.get_database_stats()
            chroma_stats = chroma_manager.get_database_stats()

            # Check if we have canonical verses but incomplete original verse embeddings
            # Count unique canonical verses from stepbible data
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(DISTINCT book || ':' || chapter || ':' || verse)
                    FROM stepbible_verses
                    WHERE original_word IS NOT NULL AND original_word != ''
                """
                )
                canonical_verse_count = cursor.fetchone()[0]

            original_verse_embeddings_count = (
                chroma_stats.get("collections", {}).get("original_verses", {}).get("count", 0)
            )

            # Check if we have words but incomplete word embeddings
            words_count = db_stats.get("words", 0)
            word_embeddings_count = chroma_stats.get("collections", {}).get("words", {}).get("count", 0)

            # Automatically generate missing embeddings
            # For verses, check if original language embeddings are incomplete
            if canonical_verse_count > 0 and original_verse_embeddings_count < canonical_verse_count * 0.9:
                auto_embed_verses = True
                if config.should_show_output():
                    coverage = (
                        (original_verse_embeddings_count / canonical_verse_count * 100)
                        if canonical_verse_count > 0
                        else 0
                    )
                    logger.info(
                        f"\n✓ Detected incomplete original verse embeddings "
                        f"({original_verse_embeddings_count:,}/{canonical_verse_count:,} "
                        f"= {coverage:.1f}%) - will generate automatically"
                    )

            # For words, check if embeddings exist at all
            if words_count > 0 and word_embeddings_count == 0:
                auto_embed_words = True
                if config.should_show_output():
                    logger.info("✓ Detected words without embeddings - will generate automatically")

        # Determine if any embedding generation should happen
        should_generate_embeddings = explicit_embed_flags or auto_embed_verses or auto_embed_words

        if should_generate_embeddings:
            if config.should_show_output():
                logger.info("\n" + "=" * 60)
                logger.info("Embedding Generation")
                logger.info("=" * 60)

            # Initialize embedding components (reuse chroma_manager if already created)
            if chroma_manager is None:
                chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
            model_manager = EmbeddingModelManager(cache_dir=str(config.models_path))
            context_builder = ContextBuilder(db_manager)

            # Use original language pipeline for verses
            original_pipeline = OriginalLanguageEmbeddingPipeline(
                db_manager=db_manager,
                chroma_manager=chroma_manager,
                model_manager=model_manager,
                context_builder=context_builder,
            )

            # Keep regular pipeline for words (already using original language)
            pipeline = EmbeddingPipeline(
                db_manager=db_manager,
                chroma_manager=chroma_manager,
                model_manager=model_manager,
                context_builder=context_builder,
            )

            rebuild_embeddings = cli_config.should_rebuild_embeddings()
            batch_size = cli_config.get_embedding_batch_size()

            # Generate verse embeddings (original language only)
            if cli_config.should_embed_verses() or cli_config.should_embed_all() or auto_embed_verses:
                if config.should_show_output():
                    logger.info("\nGenerating original language verse embeddings...")
                    logger.info("This creates ONE embedding per canonical verse using Hebrew/Greek text")

                results = original_pipeline.embed_original_verses(
                    batch_size=batch_size, force_reembed=rebuild_embeddings
                )

                if config.should_show_output():
                    if results.get("status") == "already_embedded":
                        logger.info("  Original verses already embedded (use --force-reembed to regenerate)")
                    else:
                        logger.info("\nOriginal verse embedding results:")
                        logger.info(f"  Canonical verses embedded: {results.get('verses_embedded', 0):,}")
                        if results.get("errors"):
                            logger.warning(f"  Errors: {len(results['errors'])}")
                            for error in results["errors"][:5]:  # Show first 5 errors
                                logger.warning(f"    - {error}")

            # Generate word embeddings
            if (
                cli_config.should_embed_words()
                or cli_config.should_embed_all()
                or auto_embed_words
                or force_word_reembed
            ):
                if config.should_show_output():
                    if force_word_reembed:
                        logger.info("\nRe-generating word embeddings due to STEPBible updates...")
                    else:
                        logger.info("\nGenerating word embeddings...")

                results = pipeline.embed_words(
                    batch_size=batch_size, force_reembed=rebuild_embeddings or force_word_reembed
                )

                if config.should_show_output():
                    logger.info("\nWord embedding results:")
                    if results.get("status") == "already_embedded":
                        logger.info("  Words already embedded (use --force-reembed to regenerate)")
                    else:
                        logger.info(f"  Words embedded: {results.get('words_embedded', 0):,}")
                        if results.get("errors"):
                            logger.warning(f"  Errors: {len(results['errors'])}")
                            for error in results["errors"][:5]:  # Show first 5 errors
                                logger.warning(f"    - {error}")

            # Show final statistics
            if config.should_show_output():
                logger.info("\n" + "=" * 60)
                logger.info("Embedding Statistics")
                logger.info("=" * 60)

                # Get stats from ChromaDB directly to include all collections
                chroma_stats = chroma_manager.get_database_stats()
                for collection_name, collection_stats in chroma_stats["collections"].items():
                    logger.info(f"\n{collection_name}:")
                    logger.info(f"  Count: {collection_stats.get('count', 0):,}")
                    logger.info(f"  Dimensions: {collection_stats.get('dimensions', 0)}")
                    if "metadata" in collection_stats:
                        logger.info(f"  Model: {collection_stats['metadata'].get('model', 'N/A')}")
                        if collection_name == "original_verses":
                            logger.info(f"  Type: {collection_stats['metadata'].get('type', 'N/A')}")
                            logger.info(f"  Languages: {collection_stats['metadata'].get('languages', 'N/A')}")

                logger.info(
                    f"\nTotal embeddings: {sum(c.get('count', 0) for c in chroma_stats['collections'].values()):,}"
                )

            # Validate embeddings
            if (explicit_embed_flags or auto_embed_verses or auto_embed_words) and config.should_show_output():
                logger.info("\n" + "=" * 60)
                logger.info("Validating Embeddings")
                logger.info("=" * 60)

                # Use original embedding validator for new structure
                # Pass existing chroma_manager to avoid conflicts
                embedding_validator = OriginalEmbeddingValidator(
                    db_path=config.abba_db_path, vector_path=config.vectors_path, chroma_manager=chroma_manager
                )

                results, success = embedding_validator.validate_all()
                embedding_validator.print_summary(results, success)

                if not success:
                    logger.error("Embedding validation failed - please check the errors above")
                    # Close ChromaDB before exiting
                    if "chroma_manager" in locals() and chroma_manager:
                        chroma_manager.close()
                    sys.exit(1)

        # Close ChromaDB connection properly
        if "chroma_manager" in locals() and chroma_manager:
            chroma_manager.close()

        # Auto-start FastAPI server after pipeline completes (unless --no-serve)
        if not cli_config.should_skip_serve():
            import uvicorn

            from abba.api.app import create_app

            host = cli_config.get_host()
            port = cli_config.get_port()
            logger.info(f"\nStarting ABBA API server at http://{host}:{port}")
            logger.info(f"Using database: {config.abba_db_path}")
            logger.info("API docs available at /docs")
            logger.info("Press Ctrl+C to stop the server")

            app = create_app(db_path=config.abba_db_path)
            uvicorn.run(app, host=host, port=port)

        return None

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        # Close ChromaDB before exiting
        if "chroma_manager" in locals() and chroma_manager:
            chroma_manager.close()
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        if config_manager.get_config().verbose:
            import traceback

            traceback.print_exc()
        # Close ChromaDB before exiting
        if "chroma_manager" in locals() and chroma_manager:
            chroma_manager.close()
        sys.exit(1)


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    if sys.platform == "win32":
        multiprocessing.freeze_support()

    main()
