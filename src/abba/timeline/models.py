"""
Core data models for the timeline system.

Provides flexible date representation, event modeling, and relationship tracking
with full support for uncertainty and multiple chronologies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import json

from ..verse_id import VerseID


class CalendarSystem(Enum):
    """Supported calendar systems for biblical chronology."""

    GREGORIAN = "gregorian"  # Modern calendar
    JULIAN = "julian"  # Roman calendar
    HEBREW = "hebrew"  # Jewish lunar calendar
    BABYLONIAN = "babylonian"  # Ancient Near East
    EGYPTIAN = "egyptian"  # Ancient Egyptian
    REGNAL = "regnal"  # Years of reign
    SABBATICAL = "sabbatical"  # 7-year cycles
    JUBILEE = "jubilee"  # 50-year cycles
    RELATIVE = "relative"  # Relative to other events


class CertaintyLevel(Enum):
    """Levels of certainty for historical events."""

    CERTAIN = "certain"  # Well-documented, undisputed
    PROBABLE = "probable"  # Strong evidence, minor disputes
    POSSIBLE = "possible"  # Some evidence, significant disputes
    DISPUTED = "disputed"  # Major scholarly disagreement
    LEGENDARY = "legendary"  # Traditional but unverified
    SYMBOLIC = "symbolic"  # Not intended as historical


class EventType(Enum):
    """Types of timeline events."""

    POINT = "point"  # Single moment in time
    PERIOD = "period"  # Span of time
    RECURRING = "recurring"  # Repeated events
    PROPHETIC = "prophetic"  # Future prophecy
    SYMBOLIC = "symbolic"  # Symbolic/theological time
    UNCERTAIN = "uncertain"  # Unknown exact timing


class RelationType(Enum):
    """Types of relationships between events."""

    # Temporal relationships
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    CONTEMPORARY = "contemporary"

    # Causal relationships
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    PREVENTS = "prevents"

    # Hierarchical relationships
    PART_OF = "part_of"
    CONTAINS = "contains"

    # Prophetic relationships
    PROPHESIES = "prophesies"
    FULFILLS = "fulfills"
    FORESHADOWS = "foreshadows"
    ECHOES = "echoes"

    # Other relationships
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"


@dataclass
class TimeDelta:
    """Represents a duration or time difference."""

    years: int = 0
    months: int = 0
    days: int = 0
    hours: int = 0

    # Uncertainty in the duration
    uncertainty_years: float = 0
    uncertainty_months: float = 0
    uncertainty_days: float = 0

    def to_days(self) -> float:
        """Convert to total days (approximate)."""
        return self.years * 365.25 + self.months * 30.44 + self.days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "years": self.years,
            "months": self.months,
            "days": self.days,
            "hours": self.hours,
            "uncertainty": {
                "years": self.uncertainty_years,
                "months": self.uncertainty_months,
                "days": self.uncertainty_days,
            },
        }


@dataclass
class TimePoint:
    """Flexible date representation handling uncertainty."""

    # Exact date (if known)
    exact_date: Optional[datetime] = None

    # Date range (if uncertain)
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None

    # Relative dating
    relative_to_event: Optional[str] = None  # Event ID
    relative_offset: Optional[TimeDelta] = None
    relative_type: str = "after"  # "before", "after", "during"

    # Calendar and confidence
    calendar_system: CalendarSystem = CalendarSystem.GREGORIAN
    confidence: float = 1.0  # 0-1 confidence score

    # Source information
    chronology_source: Optional[str] = None
    scholarly_support: List[str] = field(default_factory=list)

    # Probability distribution for uncertainty
    distribution_type: str = "uniform"  # "uniform", "normal", "beta"
    distribution_params: Dict[str, float] = field(default_factory=dict)

    def get_display_date(self) -> str:
        """Get human-readable date string."""
        if self.exact_date:
            year = self.exact_date.year
            if year < 0:
                return f"{abs(year)} BCE"
            else:
                return f"{year} CE"
        elif self.earliest_date and self.latest_date:
            early_year = self.earliest_date.year
            late_year = self.latest_date.year
            if early_year < 0:
                return f"{abs(late_year)}-{abs(early_year)} BCE"
            else:
                return f"{early_year}-{late_year} CE"
        elif self.relative_to_event:
            return (
                f"{self.relative_offset.years} years {self.relative_type} {self.relative_to_event}"
            )
        else:
            return "Unknown date"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "calendar_system": self.calendar_system.value,
            "confidence": self.confidence,
            "display_date": self.get_display_date(),
        }

        if self.exact_date:
            result["exact_date"] = self.exact_date.isoformat()
            result["year"] = self.exact_date.year

        if self.earliest_date:
            result["earliest_date"] = self.earliest_date.isoformat()
            result["earliest_year"] = self.earliest_date.year

        if self.latest_date:
            result["latest_date"] = self.latest_date.isoformat()
            result["latest_year"] = self.latest_date.year

        if self.relative_to_event:
            result["relative_to"] = self.relative_to_event
            result["relative_offset"] = self.relative_offset.to_dict()
            result["relative_type"] = self.relative_type

        if self.chronology_source:
            result["source"] = self.chronology_source

        if self.scholarly_support:
            result["scholars"] = self.scholarly_support

        return result


@dataclass
class TimeRange:
    """Represents a span of time."""

    start: TimePoint
    end: TimePoint

    # Duration (if known independently)
    duration: Optional[TimeDelta] = None

    # Whether boundaries are inclusive
    start_inclusive: bool = True
    end_inclusive: bool = True

    def contains_date(self, date: datetime) -> bool:
        """Check if a date falls within this range."""
        # Implementation would handle uncertainty
        pass

    def overlaps_with(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        # Implementation would handle uncertainty
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "duration": self.duration.to_dict() if self.duration else None,
            "inclusive": {"start": self.start_inclusive, "end": self.end_inclusive},
        }


@dataclass
class Location:
    """Geographic location for events."""

    name: str
    modern_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "modern_name": self.modern_name,
            "coordinates": (
                {"lat": self.latitude, "lng": self.longitude}
                if self.latitude and self.longitude
                else None
            ),
            "region": self.region,
        }


@dataclass
class EntityRef:
    """Reference to a person, nation, or other entity."""

    id: str
    name: str
    entity_type: str  # "person", "nation", "group", etc.
    role: Optional[str] = None  # Role in the event

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"id": self.id, "name": self.name, "type": self.entity_type, "role": self.role}


@dataclass
class Event:
    """Core timeline entity representing a historical event."""

    # Identification
    id: str
    name: str
    description: str

    # Temporal data
    event_type: EventType
    time_point: Optional[TimePoint] = None  # For point events
    time_range: Optional[TimeRange] = None  # For period events
    duration: Optional[TimeDelta] = None

    # Biblical linkage
    verse_refs: List[VerseID] = field(default_factory=list)
    primary_passage: Optional[str] = None  # Main biblical account

    # Context
    location: Optional[Location] = None
    participants: List[EntityRef] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # Uncertainty and sources
    certainty_level: CertaintyLevel = CertaintyLevel.POSSIBLE
    scholarly_sources: List[str] = field(default_factory=list)
    archaeological_evidence: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # Filtering support
    scholars: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    methodologies: List[str] = field(default_factory=list)
    traditions: List[str] = field(default_factory=list)

    def get_date_for_sorting(self) -> Optional[float]:
        """Get a single date value for sorting."""
        if self.time_point:
            if self.time_point.exact_date:
                return self.time_point.exact_date.timestamp()
            elif self.time_point.earliest_date and self.time_point.latest_date:
                # Use midpoint for sorting
                early = self.time_point.earliest_date.timestamp()
                late = self.time_point.latest_date.timestamp()
                return (early + late) / 2
        elif self.time_range and self.time_range.start.exact_date:
            return self.time_range.start.exact_date.timestamp()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage/export."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "event_type": self.event_type.value,
            "certainty_level": self.certainty_level.value,
            # Filterable arrays
            "scholars": self.scholars,
            "sources": self.sources,
            "methodologies": self.methodologies,
            "traditions": self.traditions,
            "categories": self.categories,
            "tags": self.tags,
            # Biblical references
            "verse_refs": [str(v) for v in self.verse_refs],
            "primary_passage": self.primary_passage,
            # Participants
            "participants": [p.to_dict() for p in self.participants],
            # Evidence
            "scholarly_sources": self.scholarly_sources,
            "archaeological_evidence": self.archaeological_evidence,
        }

        # Add temporal data
        if self.time_point:
            result["time_point"] = self.time_point.to_dict()
        if self.time_range:
            result["time_range"] = self.time_range.to_dict()
        if self.duration:
            result["duration"] = self.duration.to_dict()

        # Add location
        if self.location:
            result["location"] = self.location.to_dict()

        # Add notes
        if self.notes:
            result["notes"] = self.notes

        return result


@dataclass
class EventRelationship:
    """Represents a relationship between two events."""

    # Events
    source_event: str  # Event ID
    target_event: str  # Event ID
    relationship_type: RelationType

    # Relationship metadata
    confidence: float = 1.0
    time_distance: Optional[TimeDelta] = None
    causal_strength: Optional[float] = None  # For causal relationships

    # Evidence
    biblical_support: List[VerseID] = field(default_factory=list)
    scholarly_support: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "source": self.source_event,
            "target": self.target_event,
            "type": self.relationship_type.value,
            "confidence": self.confidence,
        }

        if self.time_distance:
            result["time_distance"] = self.time_distance.to_dict()

        if self.causal_strength is not None:
            result["causal_strength"] = self.causal_strength

        if self.biblical_support:
            result["biblical_support"] = [str(v) for v in self.biblical_support]

        if self.scholarly_support:
            result["scholarly_support"] = self.scholarly_support

        if self.notes:
            result["notes"] = self.notes

        return result


@dataclass
class TimePeriod:
    """Hierarchical time container for organizing events."""

    # Identification
    id: str
    name: str  # "United Kingdom", "Judges Period"
    description: Optional[str] = None

    # Hierarchy
    parent_period: Optional[str] = None
    child_periods: List[str] = field(default_factory=list)

    # Temporal bounds
    time_range: Optional[TimeRange] = None

    # Content
    events: List[str] = field(default_factory=list)  # Event IDs

    # Characteristics
    characteristics: Dict[str, Any] = field(default_factory=dict)
    political_entity: Optional[str] = None  # Kingdom, empire, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "id": self.id,
            "name": self.name,
            "events": self.events,
            "characteristics": self.characteristics,
        }

        if self.description:
            result["description"] = self.description

        if self.parent_period:
            result["parent"] = self.parent_period

        if self.child_periods:
            result["children"] = self.child_periods

        if self.time_range:
            result["time_range"] = self.time_range.to_dict()

        if self.political_entity:
            result["political_entity"] = self.political_entity

        return result


@dataclass
class ChronologyTrack:
    """Represents one scholarly chronology interpretation."""

    # Identification
    id: str
    name: str  # "Ussher Chronology", "Kitchen Timeline"
    scholar: Optional[str] = None
    source: Optional[str] = None

    # Methodology
    methodology: str = "traditional"  # "archaeological", "astronomical", etc.
    tradition: str = "academic"  # "jewish", "protestant", "catholic", etc.

    # Events in this chronology
    events: Dict[str, Event] = field(default_factory=dict)
    periods: Dict[str, TimePeriod] = field(default_factory=dict)

    # Confidence adjustments
    confidence_modifiers: Dict[str, float] = field(default_factory=dict)

    # Metadata
    description: Optional[str] = None
    publication_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "scholar": self.scholar,
            "source": self.source,
            "methodology": self.methodology,
            "tradition": self.tradition,
            "events": {k: v.to_dict() for k, v in self.events.items()},
            "periods": {k: v.to_dict() for k, v in self.periods.items()},
            "confidence_modifiers": self.confidence_modifiers,
            "metadata": {
                "description": self.description,
                "publication_date": (
                    self.publication_date.isoformat() if self.publication_date else None
                ),
            },
        }


@dataclass
class TimelineCollection:
    """Collection of timeline data with multiple chronology tracks."""

    # All unique events (canonical versions)
    events: Dict[str, Event] = field(default_factory=dict)

    # Relationships between events
    relationships: List[EventRelationship] = field(default_factory=list)

    # Time periods
    periods: Dict[str, TimePeriod] = field(default_factory=dict)

    # Different chronology interpretations
    chronology_tracks: Dict[str, ChronologyTrack] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: Event):
        """Add an event to the collection."""
        self.events[event.id] = event

    def add_relationship(self, relationship: EventRelationship):
        """Add a relationship between events."""
        self.relationships.append(relationship)

    def add_period(self, period: TimePeriod):
        """Add a time period."""
        self.periods[period.id] = period

        # Update parent/child relationships
        if period.parent_period and period.parent_period in self.periods:
            parent = self.periods[period.parent_period]
            if period.id not in parent.child_periods:
                parent.child_periods.append(period.id)

    def get_events_by_period(self, period_id: str) -> List[Event]:
        """Get all events in a time period."""
        if period_id not in self.periods:
            return []

        period = self.periods[period_id]
        return [self.events[eid] for eid in period.events if eid in self.events]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "events": {k: v.to_dict() for k, v in self.events.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "periods": {k: v.to_dict() for k, v in self.periods.items()},
            "chronology_tracks": {k: v.to_dict() for k, v in self.chronology_tracks.items()},
            "metadata": self.metadata,
        }
