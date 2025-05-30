"""
Quality control system for annotations.

Provides validation, consistency checking, and quality scoring
for both automatic and manual annotations.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np

from .models import (
    Annotation,
    AnnotationType,
    AnnotationLevel,
    AnnotationCollection,
    AnnotationConfidence,
)
from ..verse_id import VerseID


@dataclass
class QualityIssue:
    """Represents a quality issue found in an annotation."""

    severity: str  # "error", "warning", "info"
    category: str  # "consistency", "coverage", "accuracy", etc.
    message: str
    annotation_id: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive quality report for annotations."""

    overall_score: float  # 0-100
    issues: List[QualityIssue]
    statistics: Dict[str, Any]
    recommendations: List[str]

    def has_errors(self) -> bool:
        """Check if report contains any errors."""
        return any(issue.severity == "error" for issue in self.issues)

    def get_issues_by_severity(self, severity: str) -> List[QualityIssue]:
        """Get all issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]


class AnnotationQualityController:
    """
    Controls and ensures quality of biblical text annotations.

    Performs:
    - Consistency checking across annotations
    - Coverage analysis
    - Confidence validation
    - Overlap and conflict detection
    - Manual review prioritization
    """

    def __init__(self):
        """Initialize the quality controller."""
        self.validation_rules = self._build_validation_rules()
        self.quality_thresholds = {
            "min_confidence": 0.5,
            "max_overlap": 0.8,
            "min_coverage": 0.3,
            "consistency_threshold": 0.7,
        }

    def _build_validation_rules(self) -> Dict[str, callable]:
        """Build validation rules for annotations."""
        return {
            "confidence_range": self._validate_confidence_range,
            "topic_consistency": self._validate_topic_consistency,
            "level_appropriateness": self._validate_level_appropriateness,
            "no_conflicts": self._validate_no_conflicts,
            "proper_boundaries": self._validate_boundaries,
            "metadata_complete": self._validate_metadata,
        }

    def validate_annotation(self, annotation: Annotation) -> List[QualityIssue]:
        """
        Validate a single annotation.

        Args:
            annotation: Annotation to validate

        Returns:
            List of quality issues found
        """
        issues = []

        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            rule_issues = rule_func(annotation)
            issues.extend(rule_issues)

        return issues

    def _validate_confidence_range(self, annotation: Annotation) -> List[QualityIssue]:
        """Validate confidence scores are in valid range."""
        issues = []

        if not annotation.confidence:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="confidence",
                    message="Annotation lacks confidence scores",
                    annotation_id=annotation.id,
                    suggestion="Add confidence metrics for quality assessment",
                )
            )
        else:
            conf = annotation.confidence

            # Check overall score
            if not 0 <= conf.overall_score <= 1:
                issues.append(
                    QualityIssue(
                        severity="error",
                        category="confidence",
                        message=f"Invalid confidence score: {conf.overall_score}",
                        annotation_id=annotation.id,
                    )
                )

            # Check component scores
            for attr in ["model_confidence", "contextual_relevance", "semantic_similarity"]:
                value = getattr(conf, attr)
                if not 0 <= value <= 1:
                    issues.append(
                        QualityIssue(
                            severity="error",
                            category="confidence",
                            message=f"Invalid {attr}: {value}",
                            annotation_id=annotation.id,
                        )
                    )

        return issues

    def _validate_topic_consistency(self, annotation: Annotation) -> List[QualityIssue]:
        """Validate topic assignment consistency."""
        issues = []

        if annotation.topic_id and not annotation.topic_name:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="consistency",
                    message="Topic ID present but name missing",
                    annotation_id=annotation.id,
                )
            )

        if annotation.topic_name and not annotation.topic_id:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="consistency",
                    message="Topic name present but ID missing",
                    annotation_id=annotation.id,
                )
            )

        return issues

    def _validate_level_appropriateness(self, annotation: Annotation) -> List[QualityIssue]:
        """Validate annotation level is appropriate."""
        issues = []

        # Word-level annotations should have word positions
        if annotation.level == AnnotationLevel.WORD and not annotation.word_positions:
            issues.append(
                QualityIssue(
                    severity="error",
                    category="structure",
                    message="Word-level annotation missing word positions",
                    annotation_id=annotation.id,
                )
            )

        # Passage-level should have end verse
        if annotation.level == AnnotationLevel.PASSAGE and not annotation.end_verse:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="structure",
                    message="Passage annotation missing end verse",
                    annotation_id=annotation.id,
                    suggestion="Specify end verse for passage boundary",
                )
            )

        return issues

    def _validate_no_conflicts(self, annotation: Annotation) -> List[QualityIssue]:
        """Check for logical conflicts in annotation."""
        issues = []

        # This would check against other annotations
        # For now, just basic checks

        return issues

    def _validate_boundaries(self, annotation: Annotation) -> List[QualityIssue]:
        """Validate verse boundaries are logical."""
        issues = []

        if annotation.end_verse:
            # Check end comes after start
            start = annotation.start_verse
            end = annotation.end_verse

            # Simple check - would need full verse ordering logic
            if start.book != end.book:
                issues.append(
                    QualityIssue(
                        severity="warning",
                        category="boundaries",
                        message="Annotation spans multiple books",
                        annotation_id=annotation.id,
                        suggestion="Consider splitting into book-specific annotations",
                    )
                )

        return issues

    def _validate_metadata(self, annotation: Annotation) -> List[QualityIssue]:
        """Validate annotation metadata completeness."""
        issues = []

        if not annotation.source:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="metadata",
                    message="Missing source attribution",
                    annotation_id=annotation.id,
                )
            )

        if not annotation.created_date:
            issues.append(
                QualityIssue(
                    severity="info",
                    category="metadata",
                    message="Missing creation date",
                    annotation_id=annotation.id,
                )
            )

        return issues

    def analyze_collection(self, collection: AnnotationCollection) -> QualityReport:
        """
        Analyze quality of an entire annotation collection.

        Args:
            collection: Annotation collection to analyze

        Returns:
            Comprehensive quality report
        """
        all_issues = []
        statistics = {}

        # 1. Validate individual annotations
        for annotation in collection.annotations:
            issues = self.validate_annotation(annotation)
            all_issues.extend(issues)

        # 2. Calculate coverage statistics
        coverage_stats = self._calculate_coverage(collection)
        statistics.update(coverage_stats)

        # 3. Check for overlaps and conflicts
        overlap_issues = self._check_overlaps(collection)
        all_issues.extend(overlap_issues)

        # 4. Analyze consistency
        consistency_stats = self._analyze_consistency(collection)
        statistics.update(consistency_stats)

        # 5. Calculate overall quality score
        quality_score = self._calculate_quality_score(
            all_issues, statistics, len(collection.annotations)
        )

        # 6. Generate recommendations
        recommendations = self._generate_recommendations(all_issues, statistics)

        return QualityReport(
            overall_score=quality_score,
            issues=all_issues,
            statistics=statistics,
            recommendations=recommendations,
        )

    def _calculate_coverage(self, collection: AnnotationCollection) -> Dict[str, Any]:
        """Calculate annotation coverage statistics."""
        # Get unique verses annotated
        annotated_verses = set()

        for ann in collection.annotations:
            for verse in ann.get_verse_range():
                annotated_verses.add(str(verse))

        # Topic coverage
        topics_covered = set()
        annotation_types = Counter()

        for ann in collection.annotations:
            if ann.topic_id:
                topics_covered.add(ann.topic_id)
            annotation_types[ann.annotation_type] += 1

        # Confidence distribution
        confidence_scores = [
            ann.confidence.overall_score for ann in collection.annotations if ann.confidence
        ]

        return {
            "verses_annotated": len(annotated_verses),
            "topics_covered": len(topics_covered),
            "annotation_types": dict(annotation_types),
            "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "confidence_std": np.std(confidence_scores) if confidence_scores else 0,
            "low_confidence_count": sum(1 for s in confidence_scores if s < 0.5),
        }

    def _check_overlaps(self, collection: AnnotationCollection) -> List[QualityIssue]:
        """Check for overlapping annotations."""
        issues = []

        # Group by verse
        verse_annotations = defaultdict(list)

        for ann in collection.annotations:
            for verse in ann.get_verse_range():
                verse_annotations[str(verse)].append(ann)

        # Check each verse for conflicts
        for verse_id, annotations in verse_annotations.items():
            if len(annotations) > 1:
                # Check for topic conflicts
                topics = [ann.topic_id for ann in annotations if ann.topic_id]

                if len(set(topics)) > 3:  # Too many different topics
                    issues.append(
                        QualityIssue(
                            severity="warning",
                            category="overlap",
                            message=f"Verse {verse_id} has {len(set(topics))} different topics",
                            suggestion="Review for potential conflicts or consolidation",
                        )
                    )

                # Check for same topic with very different confidence
                topic_confidences = defaultdict(list)
                for ann in annotations:
                    if ann.topic_id and ann.confidence:
                        topic_confidences[ann.topic_id].append(ann.confidence.overall_score)

                for topic, scores in topic_confidences.items():
                    if len(scores) > 1 and max(scores) - min(scores) > 0.3:
                        issues.append(
                            QualityIssue(
                                severity="info",
                                category="consistency",
                                message=f"Topic '{topic}' has inconsistent confidence scores on {verse_id}",
                                suggestion="Review annotations for consensus",
                            )
                        )

        return issues

    def _analyze_consistency(self, collection: AnnotationCollection) -> Dict[str, Any]:
        """Analyze consistency across annotations."""
        # Source consistency
        sources = Counter(ann.source for ann in collection.annotations)

        # Method agreement
        verse_method_topics = defaultdict(lambda: defaultdict(set))

        for ann in collection.annotations:
            if ann.topic_id:
                verse_key = str(ann.start_verse)
                verse_method_topics[verse_key][ann.source].add(ann.topic_id)

        # Calculate agreement scores
        agreement_scores = []

        for verse, method_topics in verse_method_topics.items():
            if len(method_topics) > 1:
                # Get all topics from all methods
                all_topics = set()
                for topics in method_topics.values():
                    all_topics.update(topics)

                # Calculate Jaccard similarity between methods
                methods = list(method_topics.keys())
                for i in range(len(methods)):
                    for j in range(i + 1, len(methods)):
                        topics1 = method_topics[methods[i]]
                        topics2 = method_topics[methods[j]]

                        if topics1 and topics2:
                            jaccard = len(topics1 & topics2) / len(topics1 | topics2)
                            agreement_scores.append(jaccard)

        return {
            "annotation_sources": dict(sources),
            "avg_method_agreement": np.mean(agreement_scores) if agreement_scores else 1.0,
            "verses_with_disagreement": sum(1 for s in agreement_scores if s < 0.5),
        }

    def _calculate_quality_score(
        self, issues: List[QualityIssue], statistics: Dict[str, Any], total_annotations: int
    ) -> float:
        """Calculate overall quality score (0-100)."""
        # Start with perfect score
        score = 100.0

        # Deduct for issues
        issue_penalties = {"error": 5.0, "warning": 2.0, "info": 0.5}

        for issue in issues:
            score -= issue_penalties.get(issue.severity, 0)

        # Factor in statistics
        if "avg_confidence" in statistics:
            # Bonus for high average confidence
            confidence_bonus = (statistics["avg_confidence"] - 0.5) * 20
            score += max(0, confidence_bonus)

        if "avg_method_agreement" in statistics:
            # Bonus for method agreement
            agreement_bonus = statistics["avg_method_agreement"] * 10
            score += agreement_bonus

        # Ensure score is in valid range
        return max(0, min(100, score))

    def _generate_recommendations(
        self, issues: List[QualityIssue], statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check for common issues
        issue_categories = Counter(issue.category for issue in issues)

        if issue_categories["confidence"] > 5:
            recommendations.append(
                "Review and update confidence scores for better reliability assessment"
            )

        if issue_categories["overlap"] > 10:
            recommendations.append("Consolidate overlapping annotations to reduce redundancy")

        # Check statistics
        if statistics.get("avg_confidence", 1) < 0.6:
            recommendations.append("Consider manual review for low-confidence annotations")

        if statistics.get("avg_method_agreement", 1) < 0.7:
            recommendations.append("Investigate disagreements between annotation methods")

        if statistics.get("low_confidence_count", 0) > total_annotations * 0.2:
            recommendations.append(
                "High proportion of low-confidence annotations - consider retraining models"
            )

        return recommendations

    def prioritize_for_review(
        self, collection: AnnotationCollection, max_items: int = 100
    ) -> List[Annotation]:
        """
        Prioritize annotations for manual review.

        Args:
            collection: Annotation collection
            max_items: Maximum items to return

        Returns:
            Prioritized list of annotations needing review
        """
        # Score each annotation for review priority
        scored_annotations = []

        for ann in collection.annotations:
            priority_score = 0.0

            # Low confidence increases priority
            if ann.confidence:
                priority_score += (1 - ann.confidence.overall_score) * 50
            else:
                priority_score += 30  # No confidence = needs review

            # Unverified annotations
            if not ann.verified:
                priority_score += 20

            # Automatic annotations (vs manual)
            if ann.source != "manual":
                priority_score += 10

            # Controversial topics
            if ann.topic_id in ["predestination", "eschatology", "theodicy"]:
                priority_score += 15

            scored_annotations.append((ann, priority_score))

        # Sort by priority score
        scored_annotations.sort(key=lambda x: x[1], reverse=True)

        # Return top items
        return [ann for ann, _ in scored_annotations[:max_items]]

    def generate_quality_report_text(self, report: QualityReport) -> str:
        """Generate human-readable quality report."""
        lines = []

        lines.append("ANNOTATION QUALITY REPORT")
        lines.append("=" * 50)
        lines.append(f"\nOverall Quality Score: {report.overall_score:.1f}/100")

        # Issue summary
        error_count = len(report.get_issues_by_severity("error"))
        warning_count = len(report.get_issues_by_severity("warning"))
        info_count = len(report.get_issues_by_severity("info"))

        lines.append(f"\nIssues Found:")
        lines.append(f"  Errors: {error_count}")
        lines.append(f"  Warnings: {warning_count}")
        lines.append(f"  Info: {info_count}")

        # Statistics
        lines.append(f"\nKey Statistics:")
        for key, value in report.statistics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f}")
            elif isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")

        # Recommendations
        if report.recommendations:
            lines.append(f"\nRecommendations:")
            for rec in report.recommendations:
                lines.append(f"  • {rec}")

        # Critical issues
        errors = report.get_issues_by_severity("error")
        if errors:
            lines.append(f"\nCritical Issues Requiring Attention:")
            for error in errors[:5]:  # Show top 5
                lines.append(f"  • {error.message}")
                if error.suggestion:
                    lines.append(f"    Suggestion: {error.suggestion}")

        return "\n".join(lines)
