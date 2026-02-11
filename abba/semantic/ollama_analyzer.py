"""Ollama-based semantic analysis for biblical concept extraction and validation."""

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis on a verse or text."""

    verse_reference: Optional[str] = None
    text: Optional[str] = None
    concepts: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    model_responses: Dict[str, Any] = field(default_factory=dict)
    consensus_reached: bool = False
    processing_time: float = 0.0
    error: Optional[str] = None


class OllamaAnalyzer:
    """Semantic analyzer using Ollama for concept extraction and validation."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        models: List[str] = None,
        consensus_threshold: float = 0.7,
        timeout: int = 30,
        batch_size: int = 100,
    ):
        """Initialize Ollama analyzer.

        Args:
            host: Ollama API endpoint
            models: List of models to use for consensus
            consensus_threshold: Agreement threshold for consensus
            timeout: Request timeout in seconds
            batch_size: Batch size for processing multiple verses
        """
        self.host = host.rstrip("/")
        self.models = models or ["llama4:scout", "command-r-plus:latest"]
        self.consensus_threshold = consensus_threshold
        self.timeout = timeout
        self.batch_size = batch_size

        # Validate connection and models
        self._validate_setup()

    def _validate_setup(self):
        """Validate Ollama connection and models."""
        try:
            response = requests.get(urljoin(self.host, "/api/tags"), timeout=self.timeout)

            if response.status_code != 200:
                logger.warning(f"Ollama server not responding properly at {self.host}")
                return

            available_models = [model["name"] for model in response.json().get("models", [])]

            # Check if required models are available
            missing_models = []
            for model in self.models:
                # Check both exact match and base name match
                base_name = model.split(":")[0]
                if not any(model in available or base_name in available for available in available_models):
                    missing_models.append(model)

            if missing_models:
                logger.warning(f"Missing Ollama models: {', '.join(missing_models)}")
            else:
                logger.info(f"✅ Ollama setup validated with models: {', '.join(self.models)}")

        except Exception as e:
            logger.warning(f"Could not validate Ollama setup: {e}")

    def analyze_verse_for_concept(
        self, verse_text: str, concept_name: str, concept_description: str, verse_reference: str = None
    ) -> SemanticAnalysisResult:
        """Analyze a verse to determine if it relates to a specific concept.

        Args:
            verse_text: The biblical verse text
            concept_name: Name of the concept to check for
            concept_description: Detailed description of the concept
            verse_reference: Optional verse reference (e.g., "John 3:16")

        Returns:
            SemanticAnalysisResult with analysis details
        """
        start_time = time.time()

        result = SemanticAnalysisResult(verse_reference=verse_reference, text=verse_text)

        # Create analysis prompt
        prompt = self._create_concept_analysis_prompt(verse_text, concept_name, concept_description)

        try:
            # Get responses from all models
            model_responses = {}
            for model in self.models:
                response = self._query_model(model, prompt)
                if response:
                    model_responses[model] = response

            if not model_responses:
                result.error = "No model responses received"
                return result

            result.model_responses = model_responses

            # Parse and analyze responses
            parsed_responses = {}
            for model, response in model_responses.items():
                parsed = self._parse_concept_response(response)
                if parsed:
                    parsed_responses[model] = parsed

            if not parsed_responses:
                result.error = "Could not parse any model responses"
                return result

            # Calculate consensus
            result = self._calculate_consensus(result, parsed_responses)

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error analyzing verse for concept {concept_name}: {e}")

        finally:
            result.processing_time = time.time() - start_time

        return result

    def extract_concepts_from_verse(self, verse_text: str, verse_reference: str = None) -> SemanticAnalysisResult:
        """Extract semantic concepts from a verse.

        Args:
            verse_text: The biblical verse text
            verse_reference: Optional verse reference

        Returns:
            SemanticAnalysisResult with extracted concepts
        """
        start_time = time.time()

        result = SemanticAnalysisResult(verse_reference=verse_reference, text=verse_text)

        # Create concept extraction prompt
        prompt = self._create_concept_extraction_prompt(verse_text)

        try:
            # Get responses from all models
            model_responses = {}
            for model in self.models:
                response = self._query_model(model, prompt)
                if response:
                    model_responses[model] = response

            result.model_responses = model_responses

            if not model_responses:
                result.error = "No model responses received"
                return result

            # Parse and merge concept lists
            all_concepts = []
            for model, response in model_responses.items():
                concepts = self._parse_concept_list(response)
                all_concepts.extend(concepts)

            # Remove duplicates and filter by frequency
            concept_counts = {}
            for concept in all_concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

            # Only include concepts mentioned by multiple models or with high confidence
            min_mentions = max(1, len(self.models) // 2)
            result.concepts = [concept for concept, count in concept_counts.items() if count >= min_mentions]

            result.confidence = len(result.concepts) / max(1, len(all_concepts))

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error extracting concepts from verse: {e}")

        finally:
            result.processing_time = time.time() - start_time

        return result

    def batch_analyze_verses(
        self, verses: List[Tuple[str, str]], concept_name: str, concept_description: str  # (text, reference)
    ) -> List[SemanticAnalysisResult]:
        """Analyze multiple verses for a concept in batches.

        Args:
            verses: List of (text, reference) tuples
            concept_name: Name of the concept to check for
            concept_description: Detailed description of the concept

        Returns:
            List of SemanticAnalysisResult objects
        """
        results = []
        total_verses = len(verses)

        logger.info(f"Starting batch analysis of {total_verses} verses for concept: {concept_name}")

        # Process in batches
        for i in range(0, total_verses, self.batch_size):
            batch = verses[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_verses + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} verses)")

            # Analyze each verse in the batch
            batch_results = []
            for verse_text, verse_reference in batch:
                result = self.analyze_verse_for_concept(verse_text, concept_name, concept_description, verse_reference)
                batch_results.append(result)

            results.extend(batch_results)

            # Log batch summary
            relevant_count = sum(1 for r in batch_results if r.relevance_score > 0.5)
            logger.info(f"Batch {batch_num} complete: {relevant_count}/{len(batch)} verses relevant")

        # Overall summary
        total_relevant = sum(1 for r in results if r.relevance_score > 0.5)
        logger.info(f"Batch analysis complete: {total_relevant}/{total_verses} verses relevant to {concept_name}")

        return results

    def _create_concept_analysis_prompt(self, verse_text: str, concept_name: str, concept_description: str) -> str:
        """Create prompt for concept analysis."""
        return f"""Analyze the following biblical verse to determine if it relates to the concept "{concept_name}".

Concept: {concept_name}
Description: {concept_description}

Verse: {verse_text}

Please provide a JSON response with the following fields:
- "relevant": boolean (true if the verse relates to this concept)
- "relevance_score": float between 0.0 and 1.0 (how strongly it relates)
- "confidence": float between 0.0 and 1.0 (how confident you are in this assessment)
- "reasoning": string (brief explanation of your analysis)
- "key_concepts": list of strings (main concepts found in the verse)

Focus on the theological and semantic meaning, not just keyword matching. Consider metaphorical and implicit references.

JSON Response:"""

    def _create_concept_extraction_prompt(self, verse_text: str) -> str:
        """Create prompt for concept extraction."""
        return f"""Extract the main theological and semantic concepts from this biblical verse:

Verse: {verse_text}

Please provide a JSON response with:
- "concepts": list of main theological/semantic concepts (maximum 10)
- "primary_theme": the main theme or message
- "confidence": float between 0.0 and 1.0

Focus on theological concepts, moral teachings, spiritual themes, and biblical motifs. Avoid overly specific details.

JSON Response:"""

    def _query_model(self, model: str, prompt: str) -> Optional[str]:
        """Query a specific Ollama model.

        Args:
            model: Model name
            prompt: Prompt text

        Returns:
            Model response text or None if failed
        """
        try:
            response = requests.post(
                urljoin(self.host, "/api/generate"),
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent analysis
                        "num_predict": 500,  # Limit response length
                    },
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                logger.warning(f"Model {model} returned status {response.status_code}")

        except Exception as e:
            logger.warning(f"Error querying model {model}: {e}")

        return None

    def _parse_concept_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from concept analysis.

        Args:
            response: Raw model response

        Returns:
            Parsed response dict or None if failed
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)

        except json.JSONDecodeError:
            pass

        # Fallback: try to extract values with regex or simple parsing
        logger.warning("Could not parse JSON response, attempting fallback parsing")
        return None

    def _parse_concept_list(self, response: str) -> List[str]:
        """Parse concept list from response.

        Args:
            response: Raw model response

        Returns:
            List of extracted concepts
        """
        concepts = []

        try:
            parsed = self._parse_concept_response(response)
            if parsed and "concepts" in parsed:
                concepts = parsed["concepts"]
                if not isinstance(concepts, list):
                    concepts = [concepts]
        except:
            pass

        return [str(c).strip() for c in concepts if c]

    def _calculate_consensus(
        self, result: SemanticAnalysisResult, parsed_responses: Dict[str, Dict[str, Any]]
    ) -> SemanticAnalysisResult:
        """Calculate consensus from multiple model responses.

        Args:
            result: Result object to populate
            parsed_responses: Parsed responses from models

        Returns:
            Updated result with consensus data
        """
        if not parsed_responses:
            return result

        # Extract values from responses
        relevance_scores = []
        confidence_scores = []
        relevant_votes = []
        all_reasoning = []
        all_concepts = []

        for model, response in parsed_responses.items():
            if "relevance_score" in response:
                relevance_scores.append(float(response["relevance_score"]))

            if "confidence" in response:
                confidence_scores.append(float(response["confidence"]))

            if "relevant" in response:
                relevant_votes.append(bool(response["relevant"]))

            if "reasoning" in response:
                all_reasoning.append(f"{model}: {response['reasoning']}")

            if "key_concepts" in response:
                concepts = response["key_concepts"]
                if isinstance(concepts, list):
                    all_concepts.extend(concepts)

        # Calculate consensus metrics
        if relevance_scores:
            result.relevance_score = statistics.mean(relevance_scores)

        if confidence_scores:
            result.confidence = statistics.mean(confidence_scores)

        # Consensus on relevance
        if relevant_votes:
            relevant_ratio = sum(relevant_votes) / len(relevant_votes)
            result.consensus_reached = relevant_ratio >= self.consensus_threshold

        # Combine reasoning
        if all_reasoning:
            result.reasoning = " | ".join(all_reasoning)

        # Merge concepts (keep those mentioned by multiple models)
        if all_concepts:
            concept_counts = {}
            for concept in all_concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

            # Keep concepts mentioned by at least half the models
            min_mentions = max(1, len(parsed_responses) // 2)
            result.concepts = [concept for concept, count in concept_counts.items() if count >= min_mentions]

        return result
