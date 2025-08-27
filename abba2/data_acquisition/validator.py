"""
Source Validator
Validates structure and integrity of downloaded biblical sources
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import csv
from collections import defaultdict
import zipfile
import re

from .manifest import DataSource, SourceFormat, SourceType
from ..config import settings

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Validation error"""
    pass


class SourceValidator:
    """Validates biblical data sources"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize validator
        
        Args:
            data_dir: Directory containing downloaded sources
        """
        self.data_dir = data_dir or settings.data_dir / "sources"
        self.validation_results: Dict[str, Dict[str, Any]] = {}
    
    def validate_source(self, source: DataSource) -> Dict[str, Any]:
        """
        Validate a single source
        
        Args:
            source: Source to validate
            
        Returns:
            Validation results
        """
        file_path = self.data_dir / source.get_filename()
        
        if not file_path.exists():
            return {
                "valid": False,
                "error": "File not found",
                "path": str(file_path)
            }
        
        results = {
            "valid": True,
            "path": str(file_path),
            "format": source.format.value,
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Validate based on format
            if source.format == SourceFormat.TEI_XML:
                self._validate_tei_xml(file_path, results)
            elif source.format == SourceFormat.XML:
                self._validate_generic_xml(file_path, results)
            elif source.format == SourceFormat.OSIS_XML:
                self._validate_osis_xml(file_path, results)
            elif source.format == SourceFormat.PROIEL_XML:
                self._validate_proiel_xml(file_path, results)
            elif source.format == SourceFormat.MORPHGNT:
                self._validate_morphgnt(file_path, results)
            elif source.format == SourceFormat.OSHB:
                self._validate_oshb(file_path, results)
            elif source.format == SourceFormat.JSON:
                self._validate_json(file_path, results)
            elif source.format == SourceFormat.TSV:
                self._validate_tsv(file_path, results)
            else:
                results["warnings"].append(f"No specific validator for format: {source.format}")
            
            # Additional validation based on source type
            if source.type == SourceType.LEXICON:
                self._validate_lexicon_coverage(file_path, source, results)
            elif source.type == SourceType.TREEBANK:
                self._validate_treebank_structure(file_path, source, results)
            
        except Exception as e:
            results["valid"] = False
            results["error"] = str(e)
            logger.error(f"Validation failed for {source.name}: {e}")
        
        self.validation_results[source.name] = results
        return results
    
    def _validate_tei_xml(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate TEI XML format"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Check for TEI namespace
        if not root.tag.endswith("TEI") and not root.tag.endswith("teiCorpus"):
            results["warnings"].append("Root element is not TEI or teiCorpus")
        
        # Count entries
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        entries = root.findall(".//tei:entry", ns)
        results["statistics"]["entry_count"] = len(entries)
        
        # Check for required elements
        if len(entries) == 0:
            # Try without namespace
            entries = root.findall(".//entry")
            results["statistics"]["entry_count"] = len(entries)
            
        if results["statistics"]["entry_count"] == 0:
            results["warnings"].append("No entries found in TEI document")
        
        logger.info(f"TEI validation: {results['statistics']['entry_count']} entries found")
    
    def _validate_generic_xml(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate generic XML format"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Count elements
        all_elements = list(root.iter())
        results["statistics"]["element_count"] = len(all_elements)
        
        # Find potential entry elements
        entry_tags = ["entry", "word", "lexeme", "item", "strong"]
        for tag in entry_tags:
            entries = root.findall(f".//{tag}")
            if entries:
                results["statistics"]["entry_count"] = len(entries)
                results["statistics"]["entry_type"] = tag
                break
        
        if "entry_count" not in results["statistics"]:
            results["warnings"].append("No recognizable entry elements found")
        
        logger.info(f"XML validation: {results['statistics'].get('element_count', 0)} elements")
    
    def _validate_osis_xml(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate OSIS XML format"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Check for OSIS namespace
        if "osis" not in root.tag.lower():
            results["warnings"].append("Root element does not appear to be OSIS")
        
        # Count biblical books
        books = root.findall(".//div[@type='book']")
        if not books:
            books = root.findall(".//book")
        
        results["statistics"]["book_count"] = len(books)
        
        # Count verses
        verses = root.findall(".//verse")
        results["statistics"]["verse_count"] = len(verses)
        
        if results["statistics"]["verse_count"] == 0:
            results["warnings"].append("No verses found in OSIS document")
        
        logger.info(f"OSIS validation: {results['statistics']['book_count']} books, "
                   f"{results['statistics']['verse_count']} verses")
    
    def _validate_proiel_xml(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate PROIEL treebank format"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Count sentences and tokens
        sentences = root.findall(".//sentence")
        results["statistics"]["sentence_count"] = len(sentences)
        
        tokens = root.findall(".//token")
        results["statistics"]["token_count"] = len(tokens)
        
        # Check for dependency relations
        deps = [t for t in tokens if t.get("head-id")]
        results["statistics"]["dependency_count"] = len(deps)
        
        if results["statistics"]["token_count"] == 0:
            results["warnings"].append("No tokens found in PROIEL treebank")
        
        logger.info(f"PROIEL validation: {results['statistics']['sentence_count']} sentences, "
                   f"{results['statistics']['token_count']} tokens")
    
    def _validate_morphgnt(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate MorphGNT format"""
        # MorphGNT is typically a directory of files
        if file_path.suffix == ".zip":
            extract_dir = file_path.parent / file_path.stem
            if not extract_dir.exists():
                results["warnings"].append("MorphGNT zip not extracted")
                return
            file_path = extract_dir
        
        if file_path.is_dir():
            # Count book files
            book_files = list(file_path.glob("*.txt")) + list(file_path.glob("*/*.txt"))
            results["statistics"]["file_count"] = len(book_files)
            
            # Sample validation of first file
            if book_files:
                with open(book_files[0], "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    results["statistics"]["sample_line_count"] = len(lines)
                    
                    # Check format (6 columns expected)
                    if lines:
                        parts = lines[0].strip().split()
                        if len(parts) < 6:
                            results["warnings"].append(f"Unexpected format: {len(parts)} columns")
        else:
            results["warnings"].append("MorphGNT should be a directory or zip file")
        
        logger.info(f"MorphGNT validation: {results['statistics'].get('file_count', 0)} files")
    
    def _validate_oshb(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate Open Scriptures Hebrew Bible format"""
        # Similar to MorphGNT, typically a directory
        if file_path.suffix == ".zip":
            extract_dir = file_path.parent / file_path.stem
            if not extract_dir.exists():
                results["warnings"].append("OSHB zip not extracted")
                return
            file_path = extract_dir
        
        if file_path.is_dir():
            # Look for OSIS XML files
            xml_files = list(file_path.glob("*.xml")) + list(file_path.glob("*/*.xml"))
            results["statistics"]["file_count"] = len(xml_files)
            
            # Count verses across files
            total_verses = 0
            for xml_file in xml_files[:5]:  # Sample first 5 files
                try:
                    tree = ET.parse(xml_file)
                    verses = tree.findall(".//verse")
                    total_verses += len(verses)
                except:
                    pass
            
            results["statistics"]["sample_verse_count"] = total_verses
        else:
            results["warnings"].append("OSHB should be a directory or zip file")
        
        logger.info(f"OSHB validation: {results['statistics'].get('file_count', 0)} files")
    
    def _validate_json(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate JSON format"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results["statistics"]["type"] = type(data).__name__
        
        if isinstance(data, dict):
            results["statistics"]["key_count"] = len(data)
            results["statistics"]["sample_keys"] = list(data.keys())[:10]
        elif isinstance(data, list):
            results["statistics"]["item_count"] = len(data)
            if data:
                results["statistics"]["sample_item_type"] = type(data[0]).__name__
        
        logger.info(f"JSON validation: {results['statistics']['type']} with "
                   f"{results['statistics'].get('key_count', results['statistics'].get('item_count', 0))} items")
    
    def _validate_tsv(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Validate TSV format"""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)
        
        results["statistics"]["row_count"] = len(rows)
        
        if rows:
            results["statistics"]["column_count"] = len(rows[0])
            if len(rows) > 1:
                # Check consistency
                col_counts = [len(row) for row in rows[:100]]
                if len(set(col_counts)) > 1:
                    results["warnings"].append("Inconsistent column counts")
        
        logger.info(f"TSV validation: {results['statistics']['row_count']} rows, "
                   f"{results['statistics'].get('column_count', 0)} columns")
    
    def _validate_lexicon_coverage(
        self,
        file_path: Path,
        source: DataSource,
        results: Dict[str, Any]
    ) -> None:
        """Validate lexicon coverage for biblical vocabulary"""
        # This would check coverage against known vocabulary lists
        # For now, just ensure minimum entry count
        entry_count = results["statistics"].get("entry_count", 0)
        
        if source.language == "greek" and entry_count < 5000:
            results["warnings"].append(f"Low entry count for Greek lexicon: {entry_count}")
        elif source.language == "hebrew" and entry_count < 8000:
            results["warnings"].append(f"Low entry count for Hebrew lexicon: {entry_count}")
    
    def _validate_treebank_structure(
        self,
        file_path: Path,
        source: DataSource,
        results: Dict[str, Any]
    ) -> None:
        """Validate treebank has proper dependency structure"""
        token_count = results["statistics"].get("token_count", 0)
        dep_count = results["statistics"].get("dependency_count", 0)
        
        if token_count > 0 and dep_count == 0:
            results["warnings"].append("No dependency relations found in treebank")
        elif token_count > 0:
            dep_ratio = dep_count / token_count
            results["statistics"]["dependency_ratio"] = dep_ratio
            if dep_ratio < 0.8:
                results["warnings"].append(f"Low dependency coverage: {dep_ratio:.2%}")
    
    def validate_all(self, sources: Dict[str, DataSource]) -> Dict[str, Dict[str, Any]]:
        """
        Validate all sources
        
        Args:
            sources: Dictionary of source key -> DataSource
            
        Returns:
            Validation results for all sources
        """
        for key, source in sources.items():
            if source.requires_manual_entry():
                logger.info(f"Skipping manual entry source: {source.name}")
                continue
            
            logger.info(f"Validating: {source.name}")
            self.validate_source(source)
        
        return self.validation_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        total = len(self.validation_results)
        valid = sum(1 for r in self.validation_results.values() if r.get("valid", False))
        warnings = sum(
            len(r.get("warnings", []))
            for r in self.validation_results.values()
        )
        
        summary = {
            "total_sources": total,
            "valid_sources": valid,
            "invalid_sources": total - valid,
            "total_warnings": warnings,
            "success_rate": valid / total if total > 0 else 0,
        }
        
        # Aggregate statistics
        stats = defaultdict(int)
        for result in self.validation_results.values():
            for key, value in result.get("statistics", {}).items():
                if isinstance(value, (int, float)):
                    stats[f"total_{key}"] += value
        
        summary["aggregated_statistics"] = dict(stats)
        
        return summary
    
    def generate_report(self) -> str:
        """Generate validation report"""
        lines = ["ABBA 2.0 Source Validation Report", "=" * 40, ""]
        
        summary = self.get_summary()
        lines.append(f"Total Sources: {summary['total_sources']}")
        lines.append(f"Valid Sources: {summary['valid_sources']}")
        lines.append(f"Invalid Sources: {summary['invalid_sources']}")
        lines.append(f"Success Rate: {summary['success_rate']:.1%}")
        lines.append(f"Total Warnings: {summary['total_warnings']}")
        lines.append("")
        
        # Details for each source
        for name, result in self.validation_results.items():
            lines.append(f"Source: {name}")
            lines.append(f"  Valid: {result.get('valid', False)}")
            
            if "error" in result:
                lines.append(f"  Error: {result['error']}")
            
            if result.get("warnings"):
                lines.append(f"  Warnings:")
                for warning in result["warnings"]:
                    lines.append(f"    - {warning}")
            
            if result.get("statistics"):
                lines.append(f"  Statistics:")
                for key, value in result["statistics"].items():
                    if not isinstance(value, (list, dict)):
                        lines.append(f"    {key}: {value}")
            
            lines.append("")
        
        return "\n".join(lines)