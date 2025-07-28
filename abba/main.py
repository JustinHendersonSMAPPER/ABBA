#!/usr/bin/env python3
"""
Main entry point for ABBA - Annotated Bible and Background Analysis.

Initializes database, downloads STEPBible data, and imports biblical texts
for linguistic and semantic analysis.
"""

import logging
import sys
import multiprocessing

from abba.bible_extractor import BibleExtractor
from abba.cli import cli_config
from abba.config import config_manager
from abba.database import SQLiteManager
from abba.database.import_tracker import ImportTracker
from abba.embeddings import ChromaManager, EmbeddingModelManager, ContextBuilder, EmbeddingPipeline
from abba.operation_manager import OperationManager
from tqdm import tqdm


def main():
    """Main application entry point."""
    try:
        # Load configuration from all sources
        config = config_manager.load_config()

        # Setup logging
        if config.verbose:
            logging.basicConfig(level=logging.INFO)
        elif not config.quiet:
            logging.basicConfig(level=logging.WARNING)

        # Create directories
        config.create_directories()

        # Initialize database manager
        db_manager = SQLiteManager(config.abba_db_path)

        # Initialize or rebuild database if needed
        if cli_config.should_rebuild_db() or not config.abba_db_path.exists():
            if not config.quiet:
                print(f"Initializing ABBA database at {config.abba_db_path}...")
            db_manager.initialize_database()
        else:
            # Ensure database schema exists even if file exists
            try:
                stats = db_manager.get_database_stats()
                # If we can't get stats, the schema might be missing
            except Exception:
                if not config.quiet:
                    print(f"Database exists but schema missing, reinitializing...")
                db_manager.initialize_database()

        # Initialize operation manager for state tracking
        state_file = config.data_dir / ".abba_state.json"
        operation_manager = OperationManager(state_file, config.db_path)
        
        # Initialize extractor with config
        extractor = BibleExtractor(str(config.data_dir), config=config)
        extractor.operation_manager = operation_manager

        # Handle list command
        if cli_config.should_list():
            if not config.db_path.exists():
                print("bible.db not found. Please download it first with --force-download.")
                return

            translations = extractor.list_translations()
            if translations:
                print(f"\nAvailable translations ({len(translations)}):")
                print("-" * 60)
                for trans in translations:
                    print(f"{trans['id']:10} {trans['language']:10} {trans['english_name']}")

                # Show database stats if database exists
                if config.abba_db_path.exists():
                    stats = db_manager.get_database_stats()
                    print("\nABBA Database Stats:")
                    print(f"Words: {stats.get('words', 0):,}")
                    print(f"Verses: {stats.get('verses', 0):,}")
                    print(f"Translations: {stats.get('translations', 0):,}")
                    print(f"Lexicon entries: {stats.get('lexicon', 0):,}")
            else:
                print("No translations found. Please download bible.db first.")
            return

        # Download bible.db if needed
        if config.should_download():
            if not config.quiet:
                print("Downloading bible.db...")

            if not extractor.download_bible_db():
                print("Failed to download bible.db")
                sys.exit(1)

        # Download STEPBible data if needed (always download on first run or --download)
        stepbible_attribution = config.data_dir / "stepbible" / "ATTRIBUTION.txt"
        if config.should_download() or not stepbible_attribution.exists():
            if not config.quiet:
                print("Downloading STEPBible lexicon and morphology data...")

            if not extractor.download_stepbible_data():
                print("Warning: Failed to download STEPBible data (continuing without it)")
                # Don't exit - this is not critical for basic functionality

        # Check if bible.db exists
        if not config.db_path.exists():
            print(f"Error: bible.db not found at {config.db_path}")
            print("Please run with --force-download to download it.")
            sys.exit(1)

        # Initialize import tracker
        tracker = ImportTracker()
        
        # Check if we should force re-import
        force_reimport = cli_config.should_rebuild_db() or getattr(config, "rebuild_db", False)
        
        if force_reimport and not config.quiet:
            print("Force rebuild requested - will re-import all data")
            tracker.reset(confirm=True)

        # Import translations into database
        if not config.quiet:
            print("Importing biblical data into ABBA database...")

        # Import translations from bible.db
        translations = extractor.list_translations()
        if not translations:
            print("No translations found in bible.db")
            sys.exit(1)

        if config.verbose:
            print(f"Found {len(translations)} translations in bible.db")

        # Filter translations if specific ones requested
        if config.translations:
            # Filter to requested translations only
            translation_ids = set(config.translations)
            translations = [t for t in translations if t["id"] in translation_ids]
            
            # Check if all requested translations were found
            missing = translation_ids - {t["id"] for t in translations}
            if missing:
                print(f"Warning: Requested translations not found: {', '.join(missing)}")
                available_ids = [t["id"] for t in extractor.list_translations()]
                print(f"Available translations include: {', '.join(available_ids[:10])}{'...' if len(available_ids) > 10 else ''}")

        # Check which translations need importing
        translations_to_import = []
        for trans in translations:
            if not tracker.is_translation_imported(trans["id"]):
                translations_to_import.append(trans)
            elif config.verbose:
                import_time = tracker.get_translation_import_time(trans["id"])
                print(f"Skipping {trans['id']} - already imported at {import_time}")
        
        if not translations_to_import:
            if not config.quiet:
                print("All requested translations are already imported.")
        else:
            if config.verbose:
                print(f"Need to import {len(translations_to_import)} translations")

            # Import translation metadata for selected translations only
            for trans in translations_to_import:
                if config.verbose:
                    print(f"Importing metadata for {trans['english_name']} ({trans['id']})...")
                
                db_manager.insert_translation(trans)

            # Check for interrupted operations
            interrupted_warnings = operation_manager.handle_interrupted_operations()
            if interrupted_warnings and not config.quiet:
                print("\nDetected interrupted operations:")
                for warning in interrupted_warnings:
                    print(f"  {warning}")
                print()
            
            # Import verses for selected translations using parallel processing
            translation_ids_to_import = [trans["id"] for trans in translations_to_import]
            
            # Use parallel import if enabled
            use_parallel = config.get_parallel_workers() > 1
            
            if not config.quiet:
                if use_parallel:
                    print(f"Using parallel import with {config.get_parallel_workers()} workers...")
                    print(f"Parallelism: threads (I/O bound task)")
                else:
                    print("Using sequential import...")
                print(f"\nImporting {len(translation_ids_to_import)} translation(s)...")
            
            # Run import
            import_results = extractor.extract_translations_to_db_parallel(
                db_manager=db_manager,
                translation_ids=translation_ids_to_import,
                use_parallel=use_parallel
            )
            
            # Process results
            success_count = 0
            failed_translations = []
            
            for tid, result in import_results.items():
                if result.success:
                    success_count += 1
                    tracker.mark_translation_imported(tid)
                    if config.verbose:
                        print(f"✓ {tid}: {result.verse_count} verses in {result.duration:.1f}s")
                else:
                    failed_translations.append(tid)
                    print(f"✗ {tid}: {result.error}")
            
            if not config.quiet and translations_to_import:
                total_time = sum(r.duration for r in import_results.values())
                total_verses = sum(r.verse_count for r in import_results.values() if r.success)
                print(f"\nSuccessfully imported {success_count}/{len(translations_to_import)} translations")
                print(f"Total verses: {total_verses:,}")
                print(f"Total time: {total_time:.1f}s")
                if total_time > 0:
                    print(f"Average rate: {total_verses/total_time:.0f} verses/second")
                if failed_translations:
                    print(f"Failed translations: {', '.join(failed_translations)}")
            
            # Verify imports if requested
            if cli_config.args and hasattr(cli_config.args, 'verify') and cli_config.args.verify:
                if not config.quiet:
                    print("\nVerifying imports with hash validation...")
                
                verify_results = extractor.verify_import_parallel(
                    db_manager=db_manager,
                    translation_ids=translation_ids_to_import
                )
                
                invalid = [tid for tid, valid in verify_results.items() if not valid]
                if invalid:
                    print(f"\n⚠️  Validation failed for: {', '.join(invalid)}")
                else:
                    print("\n✓ All imports validated successfully")

        # Print current database stats
        if not config.quiet:
            stats = db_manager.get_database_stats()
            print(f"Database now contains {stats.get('verses', 0):,} verses")

        # Import STEPBible data (Hebrew/Greek words, lexicon, morphology)
        stepbible_attribution = config.data_dir / "stepbible" / "ATTRIBUTION.txt"
        if stepbible_attribution.exists():
            # Check if STEPBible data needs importing
            if not tracker.is_stepbible_file_imported("complete", "all_stepbible_data"):
                if not config.quiet:
                    print("Importing STEPBible lexicon and morphology data...")

                try:
                    if extractor.import_stepbible_data(db_manager, tracker):
                        tracker.mark_stepbible_file_imported("complete", "all_stepbible_data")
                        if not config.quiet:
                            updated_stats = db_manager.get_database_stats()
                            print("STEPBible import complete:")
                            print(f"  Words: {updated_stats.get('words', 0):,}")
                            print(f"  Lexicon entries: {updated_stats.get('lexicon', 0):,}")
                            print(f"  Morphology codes: {updated_stats.get('morphology', 0):,}")
                    else:
                        print("Warning: STEPBible data import failed (continuing without it)")
                except Exception as e:
                    print(f"Error importing STEPBible data: {e}")
            else:
                if config.verbose:
                    print("STEPBible data already imported - skipping")
        else:
            if config.verbose:
                print("STEPBible data not available - skipping import")
        
        # Print import summary
        if not config.quiet:
            summary = tracker.get_import_summary()
            print(f"\nImport summary:")
            print(f"  Translations: {summary['translations_imported']}")
            print(f"  STEPBible files: {sum(summary['stepbible_files'].values())}")
            if summary['last_update']:
                print(f"  Last update: {summary['last_update']}")
        
        # Handle embedding generation
        # Check if embeddings should be generated (explicit CLI or automatic for missing data)
        explicit_embed_flags = (
            cli_config.should_embed_verses() or 
            cli_config.should_embed_words() or 
            cli_config.should_embed_all()
        )
        
        # Track what needs to be embedded
        auto_embed_verses = False
        auto_embed_words = False
        
        # Initialize components to check embedding status (needed for both checking and generating)
        chroma_manager = None
        
        # If no explicit embedding flags, check if embeddings are missing
        if not explicit_embed_flags:
            chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
            db_stats = db_manager.get_database_stats()
            chroma_stats = chroma_manager.get_database_stats()
            
            # Check if we have verses but no verse embeddings
            verses_exist = db_stats.get('verses', 0) > 0
            verse_embeddings_exist = chroma_stats.get('collections', {}).get('verses', {}).get('count', 0) > 0
            
            # Check if we have words but no word embeddings
            words_exist = db_stats.get('words', 0) > 0
            word_embeddings_exist = chroma_stats.get('collections', {}).get('words', {}).get('count', 0) > 0
            
            # Automatically generate missing embeddings
            if verses_exist and not verse_embeddings_exist:
                auto_embed_verses = True
                if not config.quiet:
                    print("\n✓ Detected verses without embeddings - will generate automatically")
            
            if words_exist and not word_embeddings_exist:
                auto_embed_words = True
                if not config.quiet:
                    print("✓ Detected words without embeddings - will generate automatically")
        
        # Determine if any embedding generation should happen
        should_generate_embeddings = explicit_embed_flags or auto_embed_verses or auto_embed_words
        
        if should_generate_embeddings:
            if not config.quiet:
                print("\n" + "="*60)
                print("Embedding Generation")
                print("="*60)
            
            # Initialize embedding components (reuse chroma_manager if already created)
            if chroma_manager is None:
                chroma_manager = ChromaManager(persist_path=str(config.vectors_path))
            model_manager = EmbeddingModelManager(cache_dir=str(config.models_path))
            context_builder = ContextBuilder(db_manager)
            
            pipeline = EmbeddingPipeline(
                db_manager=db_manager,
                chroma_manager=chroma_manager,
                model_manager=model_manager,
                context_builder=context_builder
            )
            
            force_reembed = cli_config.should_force_reembed()
            batch_size = cli_config.get_embedding_batch_size()
            
            # Generate verse embeddings
            if cli_config.should_embed_verses() or cli_config.should_embed_all() or auto_embed_verses:
                if not config.quiet:
                    print("\nGenerating verse embeddings...")
                
                # Get translations to embed
                translations_to_embed = config.translations
                if not translations_to_embed and not config.quiet:
                    print("No specific translations requested - embedding all available translations")
                
                results = pipeline.embed_verses(
                    translation_ids=translations_to_embed,
                    batch_size=batch_size,
                    force_reembed=force_reembed
                )
                
                if not config.quiet:
                    print(f"\nVerse embedding results:")
                    print(f"  Translations processed: {results['translations_processed']}")
                    print(f"  Verses embedded: {results['verses_embedded']:,}")
                    if results['errors']:
                        print(f"  Errors: {len(results['errors'])}")
                        for error in results['errors'][:5]:  # Show first 5 errors
                            print(f"    - {error}")
            
            # Generate word embeddings
            if cli_config.should_embed_words() or cli_config.should_embed_all() or auto_embed_words:
                if not config.quiet:
                    print("\nGenerating word embeddings...")
                
                results = pipeline.embed_words(
                    batch_size=batch_size,
                    force_reembed=force_reembed
                )
                
                if not config.quiet:
                    print(f"\nWord embedding results:")
                    if results.get('status') == 'already_embedded':
                        print("  Words already embedded (use --force-reembed to regenerate)")
                    else:
                        print(f"  Words embedded: {results.get('words_embedded', 0):,}")
                        if results.get('errors'):
                            print(f"  Errors: {len(results['errors'])}")
                            for error in results['errors'][:5]:  # Show first 5 errors
                                print(f"    - {error}")
            
            # Show final statistics
            if not config.quiet:
                print("\n" + "="*60)
                print("Embedding Statistics")
                print("="*60)
                
                stats = pipeline.get_embedding_stats()
                for collection_name, collection_stats in stats['collections'].items():
                    print(f"\n{collection_name}:")
                    print(f"  Count: {collection_stats.get('count', 0):,}")
                    print(f"  Dimensions: {collection_stats.get('dimensions', 0)}")
                    if 'metadata' in collection_stats:
                        print(f"  Model: {collection_stats['metadata'].get('model', 'N/A')}")
                
                print(f"\nTotal embeddings: {sum(c.get('count', 0) for c in stats['collections'].values()):,}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        if config_manager.get_config().verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    
    main()
