"""
Uncertainty handling for biblical chronology.

Provides tools for representing and calculating with uncertain dates,
including probability distributions and confidence aggregation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

from .models import TimePoint, TimeDelta, Event, CertaintyLevel


@dataclass
class DateDistribution:
    """Represents uncertainty in a date as a probability distribution."""

    distribution_type: str  # "uniform", "normal", "beta", "triangular"
    parameters: Dict[str, float]

    # Cached values
    _mean: Optional[float] = None
    _std: Optional[float] = None
    _confidence_intervals: Dict[float, Tuple[float, float]] = None

    def __post_init__(self):
        """Initialize cached values."""
        self._confidence_intervals = {}

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate samples from the distribution."""
        if self.distribution_type == "uniform":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            return np.random.uniform(low, high, n)

        elif self.distribution_type == "normal":
            mean = self.parameters.get("mean", 0)
            std = self.parameters.get("std", 1)
            return np.random.normal(mean, std, n)

        elif self.distribution_type == "beta":
            alpha = self.parameters.get("alpha", 2)
            beta = self.parameters.get("beta", 2)
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            samples = np.random.beta(alpha, beta, n)
            # Scale to desired range
            return low + samples * (high - low)

        elif self.distribution_type == "triangular":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            mode = self.parameters.get("mode", (low + high) / 2)
            return np.random.triangular(low, mode, high, n)

        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def pdf(self, x: float) -> float:
        """Probability density function at point x."""
        if self.distribution_type == "uniform":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            if low <= x <= high:
                return 1.0 / (high - low)
            return 0.0

        elif self.distribution_type == "normal":
            mean = self.parameters.get("mean", 0)
            std = self.parameters.get("std", 1)
            return stats.norm.pdf(x, mean, std)

        elif self.distribution_type == "beta":
            alpha = self.parameters.get("alpha", 2)
            beta_param = self.parameters.get("beta", 2)
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            # Transform x to [0, 1] range
            x_scaled = (x - low) / (high - low)
            if 0 <= x_scaled <= 1:
                return stats.beta.pdf(x_scaled, alpha, beta_param) / (high - low)
            return 0.0

        elif self.distribution_type == "triangular":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            mode = self.parameters.get("mode", (low + high) / 2)

            if x < low or x > high:
                return 0.0
            elif x < mode:
                return 2 * (x - low) / ((high - low) * (mode - low))
            elif x == mode:
                return 2 / (high - low)
            else:
                return 2 * (high - x) / ((high - low) * (high - mode))

        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def mean(self) -> float:
        """Get the mean of the distribution."""
        if self._mean is not None:
            return self._mean

        if self.distribution_type == "uniform":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            self._mean = (low + high) / 2

        elif self.distribution_type == "normal":
            self._mean = self.parameters.get("mean", 0)

        elif self.distribution_type == "beta":
            alpha = self.parameters.get("alpha", 2)
            beta = self.parameters.get("beta", 2)
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            beta_mean = alpha / (alpha + beta)
            self._mean = low + beta_mean * (high - low)

        elif self.distribution_type == "triangular":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            mode = self.parameters.get("mode", (low + high) / 2)
            self._mean = (low + mode + high) / 3

        return self._mean

    def std(self) -> float:
        """Get the standard deviation of the distribution."""
        if self._std is not None:
            return self._std

        if self.distribution_type == "uniform":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            self._std = (high - low) / np.sqrt(12)

        elif self.distribution_type == "normal":
            self._std = self.parameters.get("std", 1)

        elif self.distribution_type == "beta":
            alpha = self.parameters.get("alpha", 2)
            beta = self.parameters.get("beta", 2)
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            beta_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            self._std = np.sqrt(beta_var) * (high - low)

        elif self.distribution_type == "triangular":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            mode = self.parameters.get("mode", (low + high) / 2)
            var = (low**2 + high**2 + mode**2 - low * high - low * mode - high * mode) / 18
            self._std = np.sqrt(var)

        return self._std

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for the distribution."""
        if confidence in self._confidence_intervals:
            return self._confidence_intervals[confidence]

        if self.distribution_type == "uniform":
            low = self.parameters.get("low", 0)
            high = self.parameters.get("high", 1)
            interval = (low, high)

        elif self.distribution_type == "normal":
            mean = self.parameters.get("mean", 0)
            std = self.parameters.get("std", 1)
            z = stats.norm.ppf((1 + confidence) / 2)
            interval = (mean - z * std, mean + z * std)

        elif self.distribution_type in ["beta", "triangular"]:
            # Use sampling for complex distributions
            samples = self.sample(10000)
            lower = np.percentile(samples, (1 - confidence) / 2 * 100)
            upper = np.percentile(samples, (1 + confidence) / 2 * 100)
            interval = (lower, upper)

        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        self._confidence_intervals[confidence] = interval
        return interval


class UncertaintyCalculator:
    """Calculates and propagates uncertainty through timeline operations."""

    @staticmethod
    def from_time_point(time_point: TimePoint) -> DateDistribution:
        """Create a date distribution from a TimePoint."""
        if time_point.exact_date:
            # Certain date - very narrow distribution
            timestamp = time_point.exact_date.timestamp()
            return DateDistribution(
                distribution_type="normal",
                parameters={"mean": timestamp, "std": 86400},  # 1 day uncertainty
            )

        elif time_point.earliest_date and time_point.latest_date:
            # Date range
            early = time_point.earliest_date.timestamp()
            late = time_point.latest_date.timestamp()

            if time_point.distribution_type == "normal":
                # Normal distribution within range
                mean = (early + late) / 2
                # 95% of distribution within range
                std = (late - early) / 4
                return DateDistribution(
                    distribution_type="normal", parameters={"mean": mean, "std": std}
                )

            elif time_point.distribution_type == "beta":
                # Beta distribution (can be skewed)
                params = time_point.distribution_params
                return DateDistribution(
                    distribution_type="beta",
                    parameters={
                        "alpha": params.get("alpha", 2),
                        "beta": params.get("beta", 2),
                        "low": early,
                        "high": late,
                    },
                )

            else:
                # Default to uniform distribution
                return DateDistribution(
                    distribution_type="uniform", parameters={"low": early, "high": late}
                )

        else:
            # No date information - maximum uncertainty
            # Assume biblical time range: ~2000 BCE to 100 CE
            return DateDistribution(
                distribution_type="uniform",
                parameters={
                    "low": datetime(year=-2000, month=1, day=1).timestamp(),
                    "high": datetime(year=100, month=1, day=1).timestamp(),
                },
            )

    @staticmethod
    def calculate_time_between(event1: Event, event2: Event) -> DateDistribution:
        """Calculate time difference between two events with uncertainty."""
        dist1 = UncertaintyCalculator.from_time_point(event1.time_point)
        dist2 = UncertaintyCalculator.from_time_point(event2.time_point)

        # For simplicity, assume normal approximation
        mean_diff = dist2.mean() - dist1.mean()
        std_diff = np.sqrt(dist1.std() ** 2 + dist2.std() ** 2)

        return DateDistribution(
            distribution_type="normal", parameters={"mean": mean_diff, "std": std_diff}
        )

    @staticmethod
    def combine_distributions(
        distributions: List[DateDistribution], weights: Optional[List[float]] = None
    ) -> DateDistribution:
        """Combine multiple date distributions (e.g., from different sources)."""
        if not distributions:
            raise ValueError("No distributions to combine")

        if weights is None:
            weights = [1.0] * len(distributions)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calculate weighted mean and variance
        weighted_mean = sum(d.mean() * w for d, w in zip(distributions, weights))

        # For variance, use law of total variance
        mean_of_variances = sum(d.std() ** 2 * w for d, w in zip(distributions, weights))
        variance_of_means = sum(
            (d.mean() - weighted_mean) ** 2 * w for d, w in zip(distributions, weights)
        )
        total_variance = mean_of_variances + variance_of_means

        return DateDistribution(
            distribution_type="normal",
            parameters={"mean": weighted_mean, "std": np.sqrt(total_variance)},
        )

    @staticmethod
    def propagate_through_relationship(
        source_event: Event, relationship_type: str, time_distance: Optional[TimeDelta]
    ) -> DateDistribution:
        """Propagate uncertainty through an event relationship."""
        source_dist = UncertaintyCalculator.from_time_point(source_event.time_point)

        if time_distance:
            # Add time distance with its uncertainty
            days = time_distance.to_days()
            uncertainty_days = (
                time_distance.uncertainty_years * 365.25
                + time_distance.uncertainty_months * 30.44
                + time_distance.uncertainty_days
            )

            # Convert to seconds
            offset_mean = days * 86400
            offset_std = uncertainty_days * 86400

            # Combine uncertainties
            new_mean = source_dist.mean() + offset_mean
            new_std = np.sqrt(source_dist.std() ** 2 + offset_std**2)

            return DateDistribution(
                distribution_type="normal", parameters={"mean": new_mean, "std": new_std}
            )
        else:
            # No specific time distance - increase uncertainty
            return DateDistribution(
                distribution_type="normal",
                parameters={
                    "mean": source_dist.mean(),
                    "std": source_dist.std() * 1.5,  # Increase uncertainty
                },
            )


class ConfidenceAggregator:
    """Aggregates confidence scores from multiple sources."""

    @staticmethod
    def aggregate_event_confidence(event: Event, source_weights: Dict[str, float]) -> float:
        """Calculate overall confidence for an event based on sources."""
        if not event.scholarly_sources:
            return event.certainty_level.value == "certain" and 0.95 or 0.5

        # Base confidence from certainty level
        base_confidence = {
            CertaintyLevel.CERTAIN: 0.95,
            CertaintyLevel.PROBABLE: 0.80,
            CertaintyLevel.POSSIBLE: 0.60,
            CertaintyLevel.DISPUTED: 0.40,
            CertaintyLevel.LEGENDARY: 0.20,
            CertaintyLevel.SYMBOLIC: 0.10,
        }.get(event.certainty_level, 0.5)

        # Adjust based on source weights
        weighted_sum = 0.0
        total_weight = 0.0

        for source in event.scholarly_sources:
            weight = source_weights.get(source, 0.5)
            weighted_sum += weight
            total_weight += 1.0

        if total_weight > 0:
            source_factor = weighted_sum / total_weight
            # Blend base confidence with source factor
            return base_confidence * 0.7 + source_factor * 0.3

        return base_confidence

    @staticmethod
    def calculate_consensus(
        events: List[Event], threshold: float = 0.1
    ) -> Tuple[float, DateDistribution]:
        """Calculate consensus among multiple versions of the same event."""
        if not events:
            raise ValueError("No events to calculate consensus")

        if len(events) == 1:
            return 1.0, UncertaintyCalculator.from_time_point(events[0].time_point)

        # Get distributions for all events
        distributions = [UncertaintyCalculator.from_time_point(e.time_point) for e in events]

        # Calculate pairwise overlaps
        overlaps = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                # Approximate overlap using confidence intervals
                ci1 = distributions[i].confidence_interval(0.68)  # 1 sigma
                ci2 = distributions[j].confidence_interval(0.68)

                # Calculate overlap ratio
                overlap_start = max(ci1[0], ci2[0])
                overlap_end = min(ci1[1], ci2[1])

                if overlap_end > overlap_start:
                    overlap_size = overlap_end - overlap_start
                    total_size = max(ci1[1], ci2[1]) - min(ci1[0], ci2[0])
                    overlap_ratio = overlap_size / total_size if total_size > 0 else 0
                    overlaps.append(overlap_ratio)
                else:
                    overlaps.append(0.0)

        # Consensus score is average overlap
        consensus_score = np.mean(overlaps) if overlaps else 0.0

        # Combine distributions based on consensus
        if consensus_score > threshold:
            # High consensus - combine all
            combined = UncertaintyCalculator.combine_distributions(distributions)
        else:
            # Low consensus - use only the most confident events
            confident_events = sorted(events, key=lambda e: e.time_point.confidence, reverse=True)
            top_events = confident_events[: max(1, len(events) // 2)]
            top_distributions = [
                UncertaintyCalculator.from_time_point(e.time_point) for e in top_events
            ]
            combined = UncertaintyCalculator.combine_distributions(top_distributions)

        return consensus_score, combined

    @staticmethod
    def propagate_confidence_through_path(path: List[Tuple[Event, Optional[float]]]) -> float:
        """Propagate confidence through a path of related events."""
        if not path:
            return 0.0

        # Start with first event's confidence
        total_confidence = path[0][0].time_point.confidence if path[0][0].time_point else 0.5

        # Decay confidence through each step
        decay_factor = 0.9  # 10% confidence loss per step

        for i in range(1, len(path)):
            event, relationship_confidence = path[i]
            event_confidence = event.time_point.confidence if event.time_point else 0.5

            # Combine event and relationship confidence
            step_confidence = event_confidence
            if relationship_confidence is not None:
                step_confidence *= relationship_confidence

            # Apply decay
            total_confidence *= decay_factor * step_confidence

        return total_confidence
