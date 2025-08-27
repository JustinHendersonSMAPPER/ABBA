"""
Source Downloader
Downloads and verifies biblical data sources
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import aiohttp
import requests
from tqdm import tqdm
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor
import time

from .manifest import DataSource, SourceManifest
from ..config import settings

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Error during download"""
    pass


class ChecksumError(Exception):
    """Checksum verification failed"""
    pass


class SourceDownloader:
    """Downloads and manages biblical data sources"""
    
    def __init__(self, manifest: SourceManifest, data_dir: Optional[Path] = None):
        """
        Initialize downloader
        
        Args:
            manifest: Source manifest with download definitions
            data_dir: Directory to store downloaded files
        """
        self.manifest = manifest
        self.data_dir = data_dir or settings.data_dir / "sources"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download settings
        self.settings = manifest.download_settings
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Progress tracking
        self.download_progress: Dict[str, float] = {}
        self.download_status: Dict[str, str] = {}
    
    async def download_all(self, sources: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Download all or specified sources
        
        Args:
            sources: List of source keys to download (None for all)
            
        Returns:
            Dictionary of source key -> success status
        """
        # Get sources to download
        if sources:
            download_queue = [
                self.manifest.get_source(key) for key in sources
                if self.manifest.get_source(key)
            ]
        else:
            download_queue = self.manifest.get_download_queue()
        
        logger.info(f"Downloading {len(download_queue)} sources")
        
        # Create session
        timeout = aiohttp.ClientTimeout(total=self.settings.timeout)
        headers = {"User-Agent": self.settings.user_agent}
        
        connector = aiohttp.TCPConnector(
            limit=self.settings.max_concurrent,
            verify_ssl=self.settings.verify_ssl
        )
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=connector
        ) as self.session:
            # Download with concurrency limit
            if self.settings.parallel:
                semaphore = asyncio.Semaphore(self.settings.max_concurrent)
                tasks = [
                    self._download_with_semaphore(source, semaphore)
                    for source in download_queue
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []
                for source in download_queue:
                    result = await self._download_source(source)
                    results.append(result)
            
            # Process results
            status = {}
            for source, result in zip(download_queue, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to download {source.name}: {result}")
                    status[source.name] = False
                else:
                    status[source.name] = result
            
            return status
    
    async def _download_with_semaphore(
        self,
        source: DataSource,
        semaphore: asyncio.Semaphore
    ) -> bool:
        """Download with semaphore for concurrency control"""
        async with semaphore:
            return await self._download_source(source)
    
    async def _download_source(self, source: DataSource) -> bool:
        """
        Download a single source
        
        Args:
            source: Source to download
            
        Returns:
            True if successful
        """
        output_path = self.data_dir / source.get_filename()
        
        # Check if already downloaded and verified
        if output_path.exists() and await self._verify_checksum(output_path, source.checksum):
            logger.info(f"Source already downloaded and verified: {source.name}")
            self.download_status[source.name] = "cached"
            return True
        
        # Download with retries
        for attempt in range(self.settings.retry_count):
            try:
                logger.info(f"Downloading {source.name} (attempt {attempt + 1})")
                self.download_status[source.name] = "downloading"
                
                await self._download_file(source.url, output_path, source.name)
                
                # Extract if compressed
                if output_path.suffix == ".zip":
                    await self._extract_zip(output_path, source.name)
                elif output_path.suffix in [".tar", ".tar.gz", ".tgz"]:
                    await self._extract_tar(output_path, source.name)
                
                # Verify checksum if provided
                if source.checksum and source.checksum != "sha256:placeholder":
                    if not await self._verify_checksum(output_path, source.checksum):
                        raise ChecksumError(f"Checksum mismatch for {source.name}")
                else:
                    # Calculate and update checksum
                    checksum = await self._calculate_checksum(output_path)
                    self.manifest.update_checksum(source.name, f"sha256:{checksum}")
                    logger.info(f"Updated checksum for {source.name}")
                
                self.download_status[source.name] = "complete"
                logger.info(f"Successfully downloaded: {source.name}")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.settings.retry_count - 1:
                    self.download_status[source.name] = "failed"
                    raise DownloadError(f"Failed to download {source.name} after {self.settings.retry_count} attempts")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    async def _download_file(self, url: str, output_path: Path, name: str) -> None:
        """Download file with progress tracking"""
        async with self.session.get(url) as response:
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get("Content-Length", 0))
            
            # Download with progress
            chunk_size = 8192
            downloaded = 0
            
            with open(output_path, "wb") as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = downloaded / total_size
                        self.download_progress[name] = progress
    
    async def _extract_zip(self, zip_path: Path, name: str) -> None:
        """Extract ZIP file"""
        extract_dir = zip_path.parent / zip_path.stem
        
        def extract():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extract)
        
        logger.info(f"Extracted {name} to {extract_dir}")
    
    async def _extract_tar(self, tar_path: Path, name: str) -> None:
        """Extract TAR file"""
        extract_dir = tar_path.parent / tar_path.stem
        
        def extract():
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(extract_dir)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extract)
        
        logger.info(f"Extracted {name} to {extract_dir}")
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        def calculate():
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, calculate)
    
    async def _verify_checksum(self, file_path: Path, expected: Optional[str]) -> bool:
        """Verify file checksum"""
        if not expected or expected == "sha256:placeholder":
            return True
        
        if not file_path.exists():
            return False
        
        # Extract checksum from format "sha256:xxxx"
        if ":" in expected:
            _, expected_hash = expected.split(":", 1)
        else:
            expected_hash = expected
        
        actual_hash = await self._calculate_checksum(file_path)
        return actual_hash == expected_hash
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get current download status"""
        return {
            "progress": self.download_progress.copy(),
            "status": self.download_status.copy(),
        }
    
    def verify_all_downloads(self) -> Dict[str, bool]:
        """Verify all downloaded sources"""
        verification = {}
        
        for key, source in self.manifest.sources.items():
            if source.requires_manual_entry():
                continue
                
            file_path = self.data_dir / source.get_filename()
            
            if not file_path.exists():
                verification[key] = False
                logger.warning(f"Missing: {source.name}")
            elif source.checksum and source.checksum != "sha256:placeholder":
                # Run async verification in sync context
                loop = asyncio.new_event_loop()
                is_valid = loop.run_until_complete(
                    self._verify_checksum(file_path, source.checksum)
                )
                verification[key] = is_valid
                
                if not is_valid:
                    logger.warning(f"Checksum mismatch: {source.name}")
            else:
                verification[key] = True
        
        # Summary
        total = len(verification)
        valid = sum(1 for v in verification.values() if v)
        logger.info(f"Verification complete: {valid}/{total} sources valid")
        
        return verification


def download_sources(
    manifest_path: Optional[Path] = None,
    sources: Optional[List[str]] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, bool]:
    """
    Convenience function to download sources
    
    Args:
        manifest_path: Path to sources.yaml
        sources: List of specific sources to download
        data_dir: Directory for downloaded files
        
    Returns:
        Dictionary of source -> success status
    """
    # Load manifest
    manifest = SourceManifest(manifest_path)
    
    # Validate manifest
    if not manifest.validate_manifest():
        raise ValueError("Manifest validation failed")
    
    # Create downloader
    downloader = SourceDownloader(manifest, data_dir)
    
    # Run async download
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        results = loop.run_until_complete(downloader.download_all(sources))
        return results
    finally:
        loop.close()