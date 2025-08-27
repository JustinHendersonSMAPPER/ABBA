"""
ABBA (Annotated Bible and Background Analysis) 2.0 CLI
Command-line interface for biblical language analysis
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
import json
import yaml

from .config import settings
from .data_acquisition import SourceManifest, SourceDownloader, SourceValidator

# Set up rich console for beautiful output
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug mode"
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
def cli(debug: bool, config_file: Optional[str]):
    """
    ABBA 2.0 - Free Biblical Language Analysis System
    
    Annotated Bible and Background Analysis providing academically
    rigorous biblical language resources without paywalls.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        settings.debug = True
    
    if config_file:
        # Load additional config
        console.print(f"[cyan]Loading config from {config_file}[/cyan]")


@cli.group()
def sources():
    """Manage biblical data sources"""
    pass


@sources.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    default="sources.yaml",
    help="Path to sources manifest file"
)
def list(manifest: str):
    """List available data sources"""
    try:
        source_manifest = SourceManifest(Path(manifest))
        
        # Create table
        table = Table(title="ABBA 2.0 Data Sources")
        table.add_column("Key", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Language", style="yellow")
        table.add_column("License", style="blue")
        table.add_column("Status", style="white")
        
        for key, source in source_manifest.sources.items():
            # Check if downloaded
            file_path = settings.data_dir / "sources" / source.get_filename()
            status = "✓ Downloaded" if file_path.exists() else "⬇ Not downloaded"
            
            langs = ", ".join(source.get_languages()) if source.languages else source.language
            
            table.add_row(
                key,
                source.name,
                source.type.value,
                langs or "",
                source.license,
                status
            )
        
        console.print(table)
        
        # Show statistics
        stats = source_manifest.get_statistics()
        console.print("\n[bold]Statistics:[/bold]")
        console.print(f"  Total sources: {stats['total_sources']}")
        console.print(f"  By type: {stats['by_type']}")
        console.print(f"  By language: {stats['by_language']}")
        console.print(f"  Manual entry required: {stats['manual_entry_required']}")
        
    except Exception as e:
        console.print(f"[red]Error loading manifest: {e}[/red]")
        sys.exit(1)


@sources.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    default="sources.yaml",
    help="Path to sources manifest file"
)
@click.option(
    "--sources",
    multiple=True,
    help="Specific sources to download (can be used multiple times)"
)
@click.option(
    "--data-dir",
    type=click.Path(),
    help="Directory to store downloaded files"
)
@click.option(
    "--parallel/--sequential",
    default=True,
    help="Enable parallel downloads"
)
def download(
    manifest: str,
    sources: tuple,
    data_dir: Optional[str],
    parallel: bool
):
    """Download biblical data sources"""
    try:
        source_manifest = SourceManifest(Path(manifest))
        
        # Update settings if needed
        if not parallel:
            source_manifest.download_settings.parallel = False
        
        # Create downloader
        data_path = Path(data_dir) if data_dir else None
        downloader = SourceDownloader(source_manifest, data_path)
        
        # Prepare source list
        source_list = list(sources) if sources else None
        
        console.print("[cyan]Starting downloads...[/cyan]")
        
        # Run downloads
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(downloader.download_all(source_list))
        finally:
            loop.close()
        
        # Display results
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        console.print(f"\n[bold]Download Complete:[/bold]")
        console.print(f"  Success: {success_count}/{total_count}")
        
        if success_count < total_count:
            console.print("\n[yellow]Failed downloads:[/yellow]")
            for name, success in results.items():
                if not success:
                    console.print(f"  ❌ {name}")
        else:
            console.print("[green]All sources downloaded successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        sys.exit(1)


@sources.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    default="sources.yaml",
    help="Path to sources manifest file"
)
@click.option(
    "--data-dir",
    type=click.Path(),
    help="Directory containing downloaded files"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for validation report"
)
def validate(manifest: str, data_dir: Optional[str], output: Optional[str]):
    """Validate downloaded sources"""
    try:
        source_manifest = SourceManifest(Path(manifest))
        
        # Create validator
        data_path = Path(data_dir) if data_dir else None
        validator = SourceValidator(data_path)
        
        console.print("[cyan]Validating sources...[/cyan]")
        
        # Run validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Validating...", total=len(source_manifest.sources))
            
            results = validator.validate_all(source_manifest.sources)
            
            progress.update(task, completed=len(results))
        
        # Display summary
        summary = validator.get_summary()
        
        console.print(f"\n[bold]Validation Summary:[/bold]")
        console.print(f"  Total sources: {summary['total_sources']}")
        console.print(f"  Valid sources: {summary['valid_sources']}")
        console.print(f"  Invalid sources: {summary['invalid_sources']}")
        console.print(f"  Success rate: {summary['success_rate']:.1%}")
        console.print(f"  Total warnings: {summary['total_warnings']}")
        
        # Show any errors
        if summary['invalid_sources'] > 0:
            console.print("\n[red]Invalid sources:[/red]")
            for name, result in results.items():
                if not result.get("valid", False):
                    console.print(f"  ❌ {name}: {result.get('error', 'Unknown error')}")
        
        # Show warnings
        if summary['total_warnings'] > 0:
            console.print("\n[yellow]Warnings:[/yellow]")
            for name, result in results.items():
                for warning in result.get("warnings", []):
                    console.print(f"  ⚠ {name}: {warning}")
        
        # Save report if requested
        if output:
            report = validator.generate_report()
            Path(output).write_text(report)
            console.print(f"\n[green]Report saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        sys.exit(1)


@sources.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    default="sources.yaml",
    help="Path to sources manifest file"
)
@click.option(
    "--data-dir",
    type=click.Path(),
    help="Directory containing downloaded files"
)
def verify(manifest: str, data_dir: Optional[str]):
    """Verify checksums of downloaded sources"""
    try:
        source_manifest = SourceManifest(Path(manifest))
        
        # Create downloader for verification
        data_path = Path(data_dir) if data_dir else None
        downloader = SourceDownloader(source_manifest, data_path)
        
        console.print("[cyan]Verifying checksums...[/cyan]")
        
        # Run verification
        results = downloader.verify_all_downloads()
        
        # Display results
        valid_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        console.print(f"\n[bold]Verification Complete:[/bold]")
        console.print(f"  Valid: {valid_count}/{total_count}")
        
        if valid_count < total_count:
            console.print("\n[red]Invalid checksums:[/red]")
            for name, valid in results.items():
                if not valid:
                    console.print(f"  ❌ {name}")
        else:
            console.print("[green]All checksums valid![/green]")
        
    except Exception as e:
        console.print(f"[red]Verification failed: {e}[/red]")
        sys.exit(1)


@cli.group()
def config():
    """Configuration management"""
    pass


@config.command()
def show():
    """Show current configuration"""
    config_dict = settings.model_dump()
    
    # Convert Path objects to strings
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    console.print(yaml.dump(config_dict, default_flow_style=False))


@config.command()
@click.argument("key")
@click.argument("value")
def set(key: str, value: str):
    """Set a configuration value"""
    if hasattr(settings, key):
        # Parse value type
        current_value = getattr(settings, key)
        
        if isinstance(current_value, bool):
            parsed_value = value.lower() in ["true", "yes", "1"]
        elif isinstance(current_value, int):
            parsed_value = int(value)
        elif isinstance(current_value, float):
            parsed_value = float(value)
        elif isinstance(current_value, Path):
            parsed_value = Path(value)
        else:
            parsed_value = value
        
        setattr(settings, key, parsed_value)
        console.print(f"[green]Set {key} = {parsed_value}[/green]")
    else:
        console.print(f"[red]Unknown configuration key: {key}[/red]")
        sys.exit(1)


@cli.command()
def info():
    """Show system information"""
    console.print("[bold]ABBA 2.0 - Annotated Bible and Background Analysis[/bold]\n")
    console.print("Free biblical language analysis with academic rigor")
    console.print("No paywalls, no restrictions, just knowledge\n")
    
    console.print("[cyan]System Information:[/cyan]")
    console.print(f"  Data directory: {settings.data_dir}")
    console.print(f"  Cache directory: {settings.cache_dir}")
    console.print(f"  Database: {settings.database_path}")
    console.print(f"  Parallel workers: {settings.parallel_workers}")
    console.print(f"  Debug mode: {settings.debug}")
    
    console.print("\n[cyan]Quick Start:[/cyan]")
    console.print("  1. Download sources: abba2 sources download")
    console.print("  2. Validate data: abba2 sources validate")
    console.print("  3. Process lexicons: abba2 process lexicons")
    console.print("  4. Start API: abba2 api serve")
    
    console.print("\n[dim]For more help: abba2 --help[/dim]")


@cli.command()
def version():
    """Show version information"""
    console.print("ABBA 2.0.0")
    console.print("Free Biblical Language Analysis System")


def main():
    """Main entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if settings.debug:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()