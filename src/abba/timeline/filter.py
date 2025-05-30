"""
Filtering and preference system for timeline data.

Provides user-specific filtering capabilities to include/exclude scholars,
sources, traditions, and methodologies from timeline queries and outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
import json
from pathlib import Path

from .models import Event, TimePeriod, ChronologyTrack, CertaintyLevel, EventType
from .query import QueryFilter
from ..verse_id import VerseID


class FilterOperator(Enum):
    """Filter operators for complex filtering."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    BETWEEN = "between"


@dataclass
class FilterRule:
    """Individual filter rule."""

    field: str  # Field to filter on
    operator: FilterOperator
    value: Any  # Value to compare against
    enabled: bool = True

    def applies_to_event(self, event: Event) -> bool:
        """Check if this filter rule applies to an event."""
        if not self.enabled:
            return True  # Disabled filters always pass

        # Get field value from event
        field_value = self._get_field_value(event)

        if field_value is None:
            return self.operator in [
                FilterOperator.NOT_EQUALS,
                FilterOperator.NOT_CONTAINS,
                FilterOperator.NOT_IN,
            ]

        # Apply operator
        if self.operator == FilterOperator.EQUALS:
            return field_value == self.value
        elif self.operator == FilterOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == FilterOperator.CONTAINS:
            if isinstance(field_value, list):
                return any(self.value in item for item in field_value if isinstance(item, str))
            elif isinstance(field_value, str):
                return self.value in field_value
            return False
        elif self.operator == FilterOperator.NOT_CONTAINS:
            if isinstance(field_value, list):
                return not any(self.value in item for item in field_value if isinstance(item, str))
            elif isinstance(field_value, str):
                return self.value not in field_value
            return True
        elif self.operator == FilterOperator.IN:
            if isinstance(field_value, list):
                return any(item in self.value for item in field_value)
            else:
                return field_value in self.value
        elif self.operator == FilterOperator.NOT_IN:
            if isinstance(field_value, list):
                return not any(item in self.value for item in field_value)
            else:
                return field_value not in self.value
        elif self.operator == FilterOperator.GREATER_THAN:
            try:
                return float(field_value) > float(self.value)
            except (ValueError, TypeError):
                return False
        elif self.operator == FilterOperator.LESS_THAN:
            try:
                return float(field_value) < float(self.value)
            except (ValueError, TypeError):
                return False
        elif self.operator == FilterOperator.BETWEEN:
            try:
                val = float(field_value)
                return self.value[0] <= val <= self.value[1]
            except (ValueError, TypeError, IndexError):
                return False

        return True

    def _get_field_value(self, event: Event) -> Any:
        """Get field value from event."""
        if self.field == "scholars":
            return event.scholars
        elif self.field == "sources":
            return event.sources
        elif self.field == "methodologies":
            return event.methodologies
        elif self.field == "traditions":
            return event.traditions
        elif self.field == "categories":
            return event.categories
        elif self.field == "tags":
            return event.tags
        elif self.field == "certainty_level":
            return event.certainty_level.value
        elif self.field == "event_type":
            return event.event_type.value
        elif self.field == "confidence":
            return event.time_point.confidence if event.time_point else 0.0
        elif self.field == "participants":
            return [p.name for p in event.participants]
        elif self.field == "location":
            return event.location.name if event.location else None
        elif self.field == "verse_refs":
            return [str(v) for v in event.verse_refs]
        elif self.field == "year":
            if event.time_point and event.time_point.exact_date:
                return event.time_point.exact_date.year
            elif event.time_point and event.time_point.earliest_date:
                return event.time_point.earliest_date.year
            return None
        elif self.field == "name":
            return event.name
        elif self.field == "description":
            return event.description
        else:
            # Try to get attribute directly
            return getattr(event, self.field, None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterRule":
        """Create from dictionary format."""
        return cls(
            field=data["field"],
            operator=FilterOperator(data["operator"]),
            value=data["value"],
            enabled=data.get("enabled", True),
        )


@dataclass
class FilterGroup:
    """Group of filter rules with AND/OR logic."""

    name: str
    rules: List[FilterRule] = field(default_factory=list)
    logic: str = "AND"  # "AND" or "OR"
    enabled: bool = True

    def applies_to_event(self, event: Event) -> bool:
        """Check if this filter group applies to an event."""
        if not self.enabled or not self.rules:
            return True

        results = [rule.applies_to_event(event) for rule in self.rules if rule.enabled]

        if not results:
            return True

        if self.logic == "AND":
            return all(results)
        else:  # OR
            return any(results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "rules": [rule.to_dict() for rule in self.rules],
            "logic": self.logic,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterGroup":
        """Create from dictionary format."""
        return cls(
            name=data["name"],
            rules=[FilterRule.from_dict(rule_data) for rule_data in data.get("rules", [])],
            logic=data.get("logic", "AND"),
            enabled=data.get("enabled", True),
        )


@dataclass
class UserPreferences:
    """User-specific timeline preferences."""

    # Scholar preferences
    preferred_scholars: List[str] = field(default_factory=list)
    excluded_scholars: List[str] = field(default_factory=list)
    scholar_weights: Dict[str, float] = field(default_factory=dict)

    # Source preferences
    preferred_sources: List[str] = field(default_factory=list)
    excluded_sources: List[str] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)

    # Methodology preferences
    preferred_methodologies: List[str] = field(default_factory=list)
    excluded_methodologies: List[str] = field(default_factory=list)

    # Tradition preferences
    preferred_traditions: List[str] = field(default_factory=list)
    excluded_traditions: List[str] = field(default_factory=list)

    # Quality thresholds
    min_confidence: float = 0.0
    min_certainty_level: CertaintyLevel = CertaintyLevel.LEGENDARY
    require_biblical_support: bool = False
    require_archaeological_evidence: bool = False

    # Display preferences
    show_uncertain_dates: bool = True
    show_disputed_events: bool = True
    default_date_format: str = "BCE/CE"  # "BCE/CE", "BC/AD", "relative"

    # Custom filter groups
    filter_groups: List[FilterGroup] = field(default_factory=list)

    def to_query_filter(self) -> QueryFilter:
        """Convert preferences to a QueryFilter."""
        return QueryFilter(
            exclude_scholars=self.excluded_scholars,
            exclude_sources=self.excluded_sources,
            exclude_traditions=self.excluded_traditions,
            include_only_scholars=self.preferred_scholars if self.preferred_scholars else [],
            min_confidence=self.min_confidence,
            certainty_levels=[self.min_certainty_level] if self.min_certainty_level else [],
            require_biblical_support=self.require_biblical_support,
            require_archaeological_evidence=self.require_archaeological_evidence,
        )

    def applies_to_event(self, event: Event) -> bool:
        """Check if event passes all preference filters."""
        # Apply built-in filters first
        query_filter = self.to_query_filter()

        # Basic filtering logic (simplified from query.py)
        if self.excluded_scholars and any(s in event.scholars for s in self.excluded_scholars):
            return False

        if self.excluded_sources and any(s in event.sources for s in self.excluded_sources):
            return False

        if self.excluded_traditions and any(
            t in event.traditions for t in self.excluded_traditions
        ):
            return False

        if self.preferred_scholars and not any(
            s in event.scholars for s in self.preferred_scholars
        ):
            return False

        if event.time_point and event.time_point.confidence < self.min_confidence:
            return False

        if self.require_biblical_support and not event.verse_refs:
            return False

        if self.require_archaeological_evidence and not event.archaeological_evidence:
            return False

        # Apply custom filter groups
        for group in self.filter_groups:
            if not group.applies_to_event(event):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "scholars": {
                "preferred": self.preferred_scholars,
                "excluded": self.excluded_scholars,
                "weights": self.scholar_weights,
            },
            "sources": {
                "preferred": self.preferred_sources,
                "excluded": self.excluded_sources,
                "weights": self.source_weights,
            },
            "methodologies": {
                "preferred": self.preferred_methodologies,
                "excluded": self.excluded_methodologies,
            },
            "traditions": {
                "preferred": self.preferred_traditions,
                "excluded": self.excluded_traditions,
            },
            "quality": {
                "min_confidence": self.min_confidence,
                "min_certainty_level": self.min_certainty_level.value,
                "require_biblical_support": self.require_biblical_support,
                "require_archaeological_evidence": self.require_archaeological_evidence,
            },
            "display": {
                "show_uncertain_dates": self.show_uncertain_dates,
                "show_disputed_events": self.show_disputed_events,
                "default_date_format": self.default_date_format,
            },
            "filter_groups": [group.to_dict() for group in self.filter_groups],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create from dictionary format."""
        prefs = cls()

        # Scholar preferences
        if "scholars" in data:
            scholar_data = data["scholars"]
            prefs.preferred_scholars = scholar_data.get("preferred", [])
            prefs.excluded_scholars = scholar_data.get("excluded", [])
            prefs.scholar_weights = scholar_data.get("weights", {})

        # Source preferences
        if "sources" in data:
            source_data = data["sources"]
            prefs.preferred_sources = source_data.get("preferred", [])
            prefs.excluded_sources = source_data.get("excluded", [])
            prefs.source_weights = source_data.get("weights", {})

        # Methodology preferences
        if "methodologies" in data:
            method_data = data["methodologies"]
            prefs.preferred_methodologies = method_data.get("preferred", [])
            prefs.excluded_methodologies = method_data.get("excluded", [])

        # Tradition preferences
        if "traditions" in data:
            trad_data = data["traditions"]
            prefs.preferred_traditions = trad_data.get("preferred", [])
            prefs.excluded_traditions = trad_data.get("excluded", [])

        # Quality thresholds
        if "quality" in data:
            quality_data = data["quality"]
            prefs.min_confidence = quality_data.get("min_confidence", 0.0)
            prefs.min_certainty_level = CertaintyLevel(
                quality_data.get("min_certainty_level", "legendary")
            )
            prefs.require_biblical_support = quality_data.get("require_biblical_support", False)
            prefs.require_archaeological_evidence = quality_data.get(
                "require_archaeological_evidence", False
            )

        # Display preferences
        if "display" in data:
            display_data = data["display"]
            prefs.show_uncertain_dates = display_data.get("show_uncertain_dates", True)
            prefs.show_disputed_events = display_data.get("show_disputed_events", True)
            prefs.default_date_format = display_data.get("default_date_format", "BCE/CE")

        # Filter groups
        if "filter_groups" in data:
            prefs.filter_groups = [
                FilterGroup.from_dict(group_data) for group_data in data["filter_groups"]
            ]

        return prefs


class TimelineFilter:
    """Main filtering engine for timeline data."""

    def __init__(self, preferences: Optional[UserPreferences] = None):
        """Initialize the filter with user preferences."""
        self.preferences = preferences or UserPreferences()

    def filter_events(self, events: List[Event]) -> List[Event]:
        """Filter a list of events based on preferences."""
        return [event for event in events if self.preferences.applies_to_event(event)]

    def filter_periods(self, periods: List[TimePeriod]) -> List[TimePeriod]:
        """Filter time periods (could be based on contained events)."""
        # For now, return all periods - could implement period-specific filtering
        return periods

    def create_scholar_filter(self, excluded_scholars: List[str]) -> FilterGroup:
        """Create a filter group to exclude specific scholars."""
        return FilterGroup(
            name="Exclude Scholars",
            rules=[
                FilterRule(
                    field="scholars", operator=FilterOperator.NOT_IN, value=excluded_scholars
                )
            ],
        )

    def create_date_range_filter(self, start_year: int, end_year: int) -> FilterGroup:
        """Create a filter for events within a date range."""
        return FilterGroup(
            name="Date Range",
            rules=[
                FilterRule(
                    field="year", operator=FilterOperator.BETWEEN, value=[start_year, end_year]
                )
            ],
        )

    def create_quality_filter(
        self, min_confidence: float, min_certainty: CertaintyLevel
    ) -> FilterGroup:
        """Create a filter for event quality."""
        return FilterGroup(
            name="Quality Filter",
            rules=[
                FilterRule(
                    field="confidence", operator=FilterOperator.GREATER_THAN, value=min_confidence
                ),
                FilterRule(
                    field="certainty_level",
                    operator=FilterOperator.NOT_IN,
                    value=(
                        ["legendary", "symbolic"]
                        if min_certainty != CertaintyLevel.LEGENDARY
                        else []
                    ),
                ),
            ],
        )

    def apply_quick_filter(self, events: List[Event], filter_text: str) -> List[Event]:
        """Apply a quick text-based filter."""
        if not filter_text:
            return events

        filter_lower = filter_text.lower()

        # Parse filter expressions like "NOT scholars:Justin" or "sources:archaeological"
        if ":" in filter_text:
            field, value = filter_text.split(":", 1)
            field = field.strip().lower()
            value = value.strip()

            # Handle negation
            exclude = False
            if field.startswith("not "):
                exclude = True
                field = field[4:].strip()

            filtered_events = []
            for event in events:
                field_value = self._get_event_field_value(event, field)

                if field_value is None:
                    if exclude:
                        filtered_events.append(event)
                    continue

                # Check if value matches
                matches = False
                if isinstance(field_value, list):
                    matches = any(value.lower() in str(item).lower() for item in field_value)
                else:
                    matches = value.lower() in str(field_value).lower()

                # Include/exclude based on match and negation
                if (matches and not exclude) or (not matches and exclude):
                    filtered_events.append(event)

            return filtered_events

        else:
            # General text search
            return [
                event
                for event in events
                if (
                    filter_lower in event.name.lower()
                    or filter_lower in event.description.lower()
                    or any(filter_lower in cat.lower() for cat in event.categories)
                    or any(filter_lower in p.name.lower() for p in event.participants)
                )
            ]

    def _get_event_field_value(self, event: Event, field: str) -> Any:
        """Get field value for quick filtering."""
        if field == "scholars":
            return event.scholars
        elif field == "sources":
            return event.sources
        elif field == "methodologies":
            return event.methodologies
        elif field == "traditions":
            return event.traditions
        elif field == "categories":
            return event.categories
        elif field == "participants":
            return [p.name for p in event.participants]
        elif field == "location":
            return event.location.name if event.location else None
        else:
            return getattr(event, field, None)


class PreferenceManager:
    """Manages user preferences with persistence."""

    def __init__(self, preferences_file: Optional[str] = None):
        """Initialize the preference manager."""
        self.preferences_file = preferences_file
        self.preferences: Dict[str, UserPreferences] = {}

    def load_preferences(self, user_id: str = "default") -> UserPreferences:
        """Load preferences for a user."""
        if user_id in self.preferences:
            return self.preferences[user_id]

        if self.preferences_file and Path(self.preferences_file).exists():
            try:
                with open(self.preferences_file, "r") as f:
                    data = json.load(f)
                    if user_id in data:
                        prefs = UserPreferences.from_dict(data[user_id])
                        self.preferences[user_id] = prefs
                        return prefs
            except (json.JSONDecodeError, KeyError):
                pass

        # Return default preferences
        prefs = UserPreferences()
        self.preferences[user_id] = prefs
        return prefs

    def save_preferences(self, user_id: str, preferences: UserPreferences):
        """Save preferences for a user."""
        self.preferences[user_id] = preferences

        if self.preferences_file:
            # Load existing data
            data = {}
            if Path(self.preferences_file).exists():
                try:
                    with open(self.preferences_file, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    pass

            # Update with new preferences
            data[user_id] = preferences.to_dict()

            # Save back to file
            with open(self.preferences_file, "w") as f:
                json.dump(data, f, indent=2)

    def create_preset_preferences(self) -> Dict[str, UserPreferences]:
        """Create common preset preferences."""
        presets = {}

        # Conservative scholarly preference
        conservative = UserPreferences()
        conservative.preferred_methodologies = ["archaeological", "textual"]
        conservative.min_confidence = 0.6
        conservative.min_certainty_level = CertaintyLevel.POSSIBLE
        conservative.require_archaeological_evidence = True
        presets["conservative"] = conservative

        # Traditional preference
        traditional = UserPreferences()
        traditional.preferred_traditions = ["protestant", "jewish"]
        traditional.preferred_methodologies = ["traditional", "textual"]
        traditional.min_confidence = 0.3
        traditional.show_disputed_events = False
        presets["traditional"] = traditional

        # Academic preference
        academic = UserPreferences()
        academic.preferred_methodologies = ["archaeological", "astronomical", "comparative"]
        academic.min_confidence = 0.7
        academic.min_certainty_level = CertaintyLevel.PROBABLE
        academic.require_archaeological_evidence = True
        academic.show_uncertain_dates = False
        presets["academic"] = academic

        # Maximalist preference (show everything)
        maximalist = UserPreferences()
        maximalist.min_confidence = 0.0
        maximalist.min_certainty_level = CertaintyLevel.SYMBOLIC
        maximalist.show_uncertain_dates = True
        maximalist.show_disputed_events = True
        presets["maximalist"] = maximalist

        return presets

    def get_available_values(self, events: List[Event]) -> Dict[str, Set[str]]:
        """Get all available values for filtering from a set of events."""
        values = {
            "scholars": set(),
            "sources": set(),
            "methodologies": set(),
            "traditions": set(),
            "categories": set(),
            "participants": set(),
            "locations": set(),
            "certainty_levels": set(),
            "event_types": set(),
        }

        for event in events:
            values["scholars"].update(event.scholars)
            values["sources"].update(event.sources)
            values["methodologies"].update(event.methodologies)
            values["traditions"].update(event.traditions)
            values["categories"].update(event.categories)
            values["participants"].update(p.name for p in event.participants)
            if event.location:
                values["locations"].add(event.location.name)
            values["certainty_levels"].add(event.certainty_level.value)
            values["event_types"].add(event.event_type.value)

        return {k: v for k, v in values.items() if v}  # Remove empty sets
