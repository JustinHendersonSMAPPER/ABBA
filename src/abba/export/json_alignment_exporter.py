"""
JSON alignment exporter for ABBA word alignments.

This module exports word alignment results as structured JSON with
confidence scores, metadata, and validation.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class JSONAlignmentExporter:
    """Export word alignments as structured JSON."""
    
    def __init__(self, output_dir: str = "aligned_output", validate_output: bool = True):
        """
        Initialize JSON alignment exporter.
        
        Args:
            output_dir: Directory to save JSON files
            validate_output: Whether to validate JSON structure
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validate_output = validate_output
        
        logger.info(f"JSON alignment exporter initialized: {self.output_dir}")
    
    def export_translation_alignments(self, 
                                    translation_id: str,
                                    alignment_data: Dict,
                                    metadata: Optional[Dict] = None) -> Path:
        """
        Export alignment data for a single translation.
        
        Args:
            translation_id: Translation identifier (e.g., "eng_asv")
            alignment_data: Alignment results data
            metadata: Additional metadata about the alignment process
            
        Returns:
            Path to the exported JSON file
        """
        # Build output structure
        output_data = {
            "translation_id": translation_id,
            "export_timestamp": datetime.now().isoformat(),
            "format_version": "1.0",
            "alignment_summary": {
                "overall_coverage": alignment_data.get("overall_coverage", 0.0),
                "hebrew_coverage": alignment_data.get("hebrew_coverage"),
                "greek_coverage": alignment_data.get("greek_coverage"),
                "book_count": alignment_data.get("book_count", 0),
                "verse_count": alignment_data.get("verse_count", 0)
            },
            "confidence_statistics": {
                "hebrew_stats": alignment_data.get("hebrew_stats", {}).get("confidence_stats", {}),
                "greek_stats": alignment_data.get("greek_stats", {}).get("confidence_stats", {})
            },
            "metadata": metadata or {}
        }
        
        # Add detailed alignment data if available
        if "detailed_alignments" in alignment_data:
            output_data["detailed_alignments"] = alignment_data["detailed_alignments"]
        
        # Add detailed verses with alignments for Bible structure
        if "hebrew_stats" in alignment_data and "detailed_verses" in alignment_data["hebrew_stats"]:
            if "verses" not in output_data:
                output_data["verses"] = []
            output_data["verses"].extend(alignment_data["hebrew_stats"]["detailed_verses"])
        
        if "greek_stats" in alignment_data and "detailed_verses" in alignment_data["greek_stats"]:
            if "verses" not in output_data:
                output_data["verses"] = []
            output_data["verses"].extend(alignment_data["greek_stats"]["detailed_verses"])
        
        # Validate structure if enabled
        if self.validate_output:
            validation_errors = self._validate_structure(output_data)
            if validation_errors:
                logger.warning(f"Validation errors for {translation_id}: {validation_errors}")
        
        # Save to file with abba_ prefix
        output_file = self.output_dir / f"abba_{translation_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported alignments for {translation_id} to {output_file}")
        return output_file
    
    def export_batch_alignments(self, 
                               results: List[Dict],
                               metadata: Optional[Dict] = None) -> Tuple[Path, List[Path]]:
        """
        Export alignment data for multiple translations as separate files.
        Creates a report.json with overview and individual abba_<translation>.json files.
        
        Args:
            results: List of alignment result dictionaries
            metadata: Metadata about the batch process
            
        Returns:
            Tuple of (report_file_path, list_of_alignment_file_paths)
        """
        alignment_files = []
        
        # Extract and export morphological analysis to separate file
        morphological_file = self._export_morphological_analysis(results)
        
        # Create individual alignment files for each translation (without morphological analysis)
        for result in results:
            translation_id = result.get('translation_id', 'unknown')
            alignment_file = self._export_single_translation(result, translation_id, exclude_morphological=True)
            alignment_files.append(alignment_file)
        
        # Create overview report with summary statistics only
        report_data = {
            "export_timestamp": datetime.now().isoformat(),
            "format_version": "1.0",
            "export_type": "multi_file",
            "translations_count": len(results),
            "batch_summary": {
                "average_coverage": sum(r.get("overall_coverage", 0) for r in results) / len(results) if results else 0,
                "translations_with_hebrew": len([r for r in results if r.get("has_hebrew", False)]),
                "translations_with_greek": len([r for r in results if r.get("has_greek", False)]),
                "total_verses": sum(r.get("verse_count", 0) for r in results),
                "total_books": sum(r.get("book_count", 0) for r in results)
            },
            "metadata": metadata or {},
            "morphological_analysis_file": "abba_morphological_analysis.json",
            "translations": [],
            "alignment_files": []
        }
        
        # Add each translation's summary data (no verses)
        for i, result in enumerate(results):
            translation_id = result.get("translation_id", f"translation_{i}")
            
            translation_summary = {
                "translation_id": translation_id,
                "alignment_file": f"abba_{translation_id}.json",
                "overall_coverage": result.get("overall_coverage", 0.0),
                "hebrew_coverage": result.get("hebrew_coverage"),
                "greek_coverage": result.get("greek_coverage"),
                "book_count": result.get("book_count", 0),
                "verse_count": result.get("verse_count", 0),
                "has_hebrew": result.get("has_hebrew", False),
                "has_greek": result.get("has_greek", False),
                "confidence_statistics": {
                    "hebrew_stats": result.get("hebrew_stats", {}).get("confidence_stats", {}),
                    "greek_stats": result.get("greek_stats", {}).get("confidence_stats", {})
                }
            }
            
            report_data["translations"].append(translation_summary)
            report_data["alignment_files"].append(f"abba_{translation_id}.json")
        
        # Convert numpy types to JSON-serializable types  
        serializable_report = convert_numpy_types(report_data)
        
        # Save report file
        report_file = self.output_dir / "report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported batch alignments for {len(results)} translations to {len(alignment_files)} files")
        logger.info(f"Report saved to {report_file}")
        
        return report_file, alignment_files
    
    def _export_morphological_analysis(self, results: List[Dict]) -> Path:
        """Export morphological analysis to a separate file."""
        # Collect all morphological analysis data from all translations
        morphological_data = {
            "export_timestamp": datetime.now().isoformat(),
            "format_version": "1.0",
            "description": "Morphological analysis of original Hebrew and Greek texts",
            "verses": []
        }
        
        # Process all results to extract morphological analysis
        for result in results:
            # Hebrew verses
            if "hebrew_stats" in result and "detailed_verses" in result["hebrew_stats"]:
                for verse in result["hebrew_stats"]["detailed_verses"]:
                    if "morphological_analysis" in verse:
                        morphological_verse = {
                            "book": verse.get("book", ""),
                            "chapter": verse.get("chapter", 0),
                            "verse": verse.get("verse", 0),
                            "language": verse.get("language", ""),
                            "original_text": verse.get("original_text", ""),
                            "morphological_analysis": verse["morphological_analysis"]
                        }
                        morphological_data["verses"].append(morphological_verse)
            
            # Greek verses
            if "greek_stats" in result and "detailed_verses" in result["greek_stats"]:
                for verse in result["greek_stats"]["detailed_verses"]:
                    if "morphological_analysis" in verse:
                        morphological_verse = {
                            "book": verse.get("book", ""),
                            "chapter": verse.get("chapter", 0),
                            "verse": verse.get("verse", 0),
                            "language": verse.get("language", ""),
                            "original_text": verse.get("original_text", ""),
                            "morphological_analysis": verse["morphological_analysis"]
                        }
                        morphological_data["verses"].append(morphological_verse)
        
        # Convert numpy types to JSON-serializable types
        serializable_data = convert_numpy_types(morphological_data)
        
        # Save morphological analysis file
        morphological_file = self.output_dir / "abba_morphological_analysis.json"
        with open(morphological_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported morphological analysis to {morphological_file}")
        return morphological_file
    
    def _remove_morphological_analysis_from_verses(self, verses: List[Dict]) -> List[Dict]:
        """Remove morphological_analysis from verses, keeping only essential verse data."""
        cleaned_verses = []
        for verse in verses:
            cleaned_verse = verse.copy()
            # Remove morphological analysis but keep other data
            if "morphological_analysis" in cleaned_verse:
                del cleaned_verse["morphological_analysis"]
            cleaned_verses.append(cleaned_verse)
        return cleaned_verses
    
    def _export_single_translation(self, result: Dict, translation_id: str, exclude_morphological: bool = False) -> Path:
        """Export a single translation's alignment data to its own file."""
        # Collect all verses for this translation
        all_verses = []
        
        if "hebrew_stats" in result and "detailed_verses" in result["hebrew_stats"]:
            verses = result["hebrew_stats"]["detailed_verses"]
            if exclude_morphological:
                verses = self._remove_morphological_analysis_from_verses(verses)
            all_verses.extend(verses)
        
        if "greek_stats" in result and "detailed_verses" in result["greek_stats"]:
            verses = result["greek_stats"]["detailed_verses"]
            if exclude_morphological:
                verses = self._remove_morphological_analysis_from_verses(verses)
            all_verses.extend(verses)
        
        # Build single translation output structure
        output_data = {
            "export_timestamp": datetime.now().isoformat(),
            "format_version": "1.0",
            "translation_id": translation_id,
            "translation_info": {
                "overall_coverage": result.get("overall_coverage", 0.0),
                "hebrew_coverage": result.get("hebrew_coverage"),
                "greek_coverage": result.get("greek_coverage"),
                "book_count": result.get("book_count", 0),
                "verse_count": result.get("verse_count", 0),
                "has_hebrew": result.get("has_hebrew", False),
                "has_greek": result.get("has_greek", False)
            },
            "alignment_metadata": {
                "aligner_type": result.get("aligner_type", "ensemble"),
                "confidence_threshold": result.get("confidence_threshold", 0.0),
                "hebrew_confidence_stats": result.get("hebrew_stats", {}).get("confidence_stats", {}),
                "greek_confidence_stats": result.get("greek_stats", {}).get("confidence_stats", {}),
                "morphological_analysis_file": "abba_morphological_analysis.json" if exclude_morphological else None
            },
            "verses": all_verses
        }
        
        # Include translation metadata from source file if available
        translation_metadata = result.get("translation_metadata", {})
        if translation_metadata:
            # Add source metadata to the top level
            for field, value in translation_metadata.items():
                output_data[field] = value
        
        # Convert numpy types to JSON-serializable types
        serializable_data = convert_numpy_types(output_data)
        
        # Validate structure if enabled
        if self.validate_output:
            validation_errors = self._validate_structure(serializable_data)
            if validation_errors:
                logger.warning(f"Validation errors for {translation_id}: {validation_errors}")
        
        # Save to file with abba_ prefix
        output_file = self.output_dir / f"abba_{translation_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported alignments for {translation_id} to {output_file}")
        return output_file
    
    def _validate_structure(self, data: Dict) -> List[str]:
        """Validate single translation alignment structure."""
        errors = []
        
        # Check required fields for our new structure
        required_fields = [
            "translation_id", "export_timestamp", "format_version", 
            "translation_info", "alignment_metadata", "verses"
        ]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate translation info
        if "translation_info" in data:
            info = data["translation_info"]
            coverage_fields = ["overall_coverage"]
            
            for field in coverage_fields:
                if field in info:
                    value = info[field]
                    if not isinstance(value, (int, float)) or value < 0 or value > 100:
                        errors.append(f"Invalid coverage value for {field}: {value}")
        
        # Validate alignment metadata (our confidence stats are here now)
        if "alignment_metadata" in data:
            metadata = data["alignment_metadata"]
            for lang_stats in [metadata.get("hebrew_confidence_stats", {}), metadata.get("greek_confidence_stats", {})]:
                if "avg_confidence" in lang_stats:
                    conf = lang_stats["avg_confidence"]
                    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                        errors.append(f"Invalid confidence score: {conf}")
        
        # Validate verses structure
        if "verses" in data:
            if not isinstance(data["verses"], list):
                errors.append("verses field must be a list")
            else:
                for i, verse in enumerate(data["verses"][:3]):  # Check first 3 verses
                    if not isinstance(verse, dict):
                        errors.append(f"Verse {i} must be a dictionary")
                    elif "word_mappings" not in verse:
                        errors.append(f"Verse {i} missing word_mappings")
        
        return errors
    
    def _validate_batch_structure(self, data: Dict) -> List[str]:
        """Validate batch alignment structure."""
        errors = []
        
        # Check required fields
        required_fields = [
            "batch_export_timestamp", "format_version", "translations_count",
            "batch_summary", "metadata", "translations"
        ]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate translations list
        if "translations" in data:
            if not isinstance(data["translations"], list):
                errors.append("translations field must be a list")
            else:
                for i, translation in enumerate(data["translations"]):
                    if not isinstance(translation, dict):
                        errors.append(f"Translation {i} must be a dictionary")
                    elif "translation_id" not in translation:
                        errors.append(f"Translation {i} missing translation_id")
        
        return errors
    
    def validate_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a JSON alignment file.
        
        Args:
            file_path: Path to the JSON file to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a valid JSON
            validation_result = {
                "valid_json": True,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "structure_errors": []
            }
            
            # Determine if it's batch or single translation
            if "translations" in data:
                validation_result["type"] = "batch"
                validation_result["structure_errors"] = self._validate_batch_structure(data)
            else:
                validation_result["type"] = "single"
                validation_result["structure_errors"] = self._validate_structure(data)
            
            validation_result["valid_structure"] = len(validation_result["structure_errors"]) == 0
            
            return validation_result
            
        except json.JSONDecodeError as e:
            return {
                "valid_json": False,
                "error": str(e),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            return {
                "valid_json": False,
                "error": f"Validation failed: {e}",
                "file_size_mb": 0
            }