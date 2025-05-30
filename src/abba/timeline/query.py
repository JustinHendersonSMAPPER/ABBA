"""
Query engine for timeline data.

Provides natural language and structured queries for biblical timeline data,
including date ranges, event patterns, and relationship traversal.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime, timedelta
import re
from collections import defaultdict

from .models import (
    Event,
    EventRelationship,
    TimePeriod,
    TimePoint,
    TimeRange,
    TimeDelta,
    EventType,
    RelationType,
    CertaintyLevel,
    CalendarSystem,
)
from .graph import TemporalGraph, GraphTraverser
from .uncertainty import UncertaintyCalculator, DateDistribution
from ..verse_id import parse_verse_id, VerseID


@dataclass
class QueryFilter:
    """Filter criteria for timeline queries."""

    # Date filters
    start_date: Optional[TimePoint] = None
    end_date: Optional[TimePoint] = None

    # Event filters
    event_types: List[EventType] = field(default_factory=list)
    certainty_levels: List[CertaintyLevel] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # Text filters
    name_contains: Optional[str] = None
    description_contains: Optional[str] = None

    # Biblical filters
    verse_refs: List[VerseID] = field(default_factory=list)
    books: List[str] = field(default_factory=list)

    # Source filters (for easy filtering)
    exclude_scholars: List[str] = field(default_factory=list)
    exclude_sources: List[str] = field(default_factory=list)
    exclude_traditions: List[str] = field(default_factory=list)
    include_only_scholars: List[str] = field(default_factory=list)
    include_only_sources: List[str] = field(default_factory=list)

    # Quality filters
    min_confidence: float = 0.0
    require_biblical_support: bool = False
    require_archaeological_evidence: bool = False

    # Participant filters
    participants: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)


@dataclass
class DateRangeQuery:
    """Query for events within a date range."""

    start: TimePoint
    end: TimePoint
    include_uncertain: bool = True
    overlap_threshold: float = 0.5  # Minimum overlap for uncertain dates

    def matches_event(self, event: Event) -> bool:
        """Check if an event matches this date range."""
        if not event.time_point and not event.time_range:
            return False

        # Get event time bounds
        if event.time_point:
            event_start = event.time_point
            event_end = event.time_point
        else:
            event_start = event.time_range.start
            event_end = event.time_range.end

        # Check overlap
        return self._check_overlap(event_start, event_end)

    def _check_overlap(self, event_start: TimePoint, event_end: TimePoint) -> bool:
        """Check if event time overlaps with query range."""
        # Simple implementation - could be enhanced with uncertainty
        query_start_ts = self._time_point_to_timestamp(self.start)
        query_end_ts = self._time_point_to_timestamp(self.end)
        event_start_ts = self._time_point_to_timestamp(event_start)
        event_end_ts = self._time_point_to_timestamp(event_end)

        # Check for any overlap
        return event_end_ts >= query_start_ts and event_start_ts <= query_end_ts

    def _time_point_to_timestamp(self, tp: TimePoint) -> float:
        """Convert TimePoint to timestamp for comparison."""
        if tp.exact_date:
            return tp.exact_date.timestamp()
        elif tp.earliest_date and tp.latest_date:
            # Use midpoint
            early = tp.earliest_date.timestamp()
            late = tp.latest_date.timestamp()
            return (early + late) / 2
        elif tp.earliest_date:
            return tp.earliest_date.timestamp()
        elif tp.latest_date:
            return tp.latest_date.timestamp()
        else:
            return 0.0


@dataclass
class EventPattern:
    """Pattern for finding sequences of related events."""

    event_sequence: List[Dict[str, Any]]  # Event criteria
    relationship_sequence: List[RelationType]  # Required relationships
    max_time_distance: Optional[TimeDelta] = None
    allow_gaps: bool = True  # Allow intervening events

    def __post_init__(self):
        """Validate pattern consistency."""
        if len(self.event_sequence) != len(self.relationship_sequence) + 1:
            raise ValueError("Relationship sequence must be one shorter than event sequence")


class PatternMatcher:
    """Matches event patterns in timeline data."""

    def __init__(self, graph: TemporalGraph):
        """Initialize the pattern matcher."""
        self.graph = graph
        self.traverser = GraphTraverser(graph)

    def find_pattern_matches(self, pattern: EventPattern) -> List[List[Event]]:
        """Find all instances of a pattern in the timeline."""
        matches = []

        # Start with events matching the first pattern element
        first_criteria = pattern.event_sequence[0]
        candidate_starts = self._find_events_matching_criteria(first_criteria)

        for start_event in candidate_starts:
            # Try to build the pattern from this starting point
            pattern_matches = self._build_pattern_from_event(start_event, pattern, 0)
            matches.extend(pattern_matches)

        return matches

    def _find_events_matching_criteria(self, criteria: Dict[str, Any]) -> List[Event]:
        """Find events matching the given criteria."""
        candidates = list(self.graph.nodes.values())

        # Filter by criteria
        if "event_type" in criteria:
            candidates = [
                c for c in candidates if c.event.event_type.value == criteria["event_type"]
            ]

        if "categories" in criteria:
            required_cats = set(criteria["categories"])
            candidates = [c for c in candidates if required_cats.issubset(set(c.event.categories))]

        if "participants" in criteria:
            required_parts = set(criteria["participants"])
            candidates = [
                c for c in candidates if any(p.name in required_parts for p in c.event.participants)
            ]

        return [c.event for c in candidates]

    def _build_pattern_from_event(
        self, current_event: Event, pattern: EventPattern, position: int
    ) -> List[List[Event]]:
        """Recursively build pattern matches from current event."""
        # Base case: reached end of pattern
        if position >= len(pattern.event_sequence) - 1:
            return [[current_event]]

        matches = []

        # Find next events connected by required relationship
        required_rel = pattern.relationship_sequence[position]
        next_criteria = pattern.event_sequence[position + 1]

        # Get related events
        current_node = self.graph.nodes.get(current_event.id)
        if not current_node:
            return []

        for edge in current_node.outgoing_edges:
            if edge.relationship.relationship_type == required_rel:
                next_event = edge.target.event

                # Check if next event matches criteria
                if self._event_matches_criteria(next_event, next_criteria):
                    # Check time distance if specified
                    if pattern.max_time_distance:
                        if not self._check_time_distance(
                            current_event, next_event, pattern.max_time_distance
                        ):
                            continue

                    # Recursively build rest of pattern
                    sub_matches = self._build_pattern_from_event(next_event, pattern, position + 1)

                    # Prepend current event to all sub-matches
                    for sub_match in sub_matches:
                        matches.append([current_event] + sub_match)

        return matches

    def _event_matches_criteria(self, event: Event, criteria: Dict[str, Any]) -> bool:
        """Check if event matches criteria."""
        if "event_type" in criteria:
            if event.event_type.value != criteria["event_type"]:
                return False

        if "categories" in criteria:
            required_cats = set(criteria["categories"])
            if not required_cats.issubset(set(event.categories)):
                return False

        if "participants" in criteria:
            required_parts = set(criteria["participants"])
            event_parts = {p.name for p in event.participants}
            if not required_parts.intersection(event_parts):
                return False

        return True

    def _check_time_distance(self, event1: Event, event2: Event, max_distance: TimeDelta) -> bool:
        """Check if events are within maximum time distance."""
        if not event1.time_point or not event2.time_point:
            return True  # Can't verify, assume valid

        # Calculate approximate time difference
        time_diff = UncertaintyCalculator.calculate_time_between(event1, event2)
        max_distance_seconds = max_distance.to_days() * 86400

        # Check if mean difference is within bounds
        return abs(time_diff.mean()) <= max_distance_seconds


class TimelineQuery:
    """Main query interface for timeline data."""

    def __init__(self, graph: TemporalGraph):
        """Initialize the query engine."""
        self.graph = graph
        self.traverser = GraphTraverser(graph)
        self.pattern_matcher = PatternMatcher(graph)

        # Natural language processing patterns
        self.nl_patterns = self._build_nl_patterns()

    def _build_nl_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Build natural language query patterns."""
        return [
            # Date range queries
            (re.compile(r"events?\s+(?:during|between)\s+(.+?)\s+and\s+(.+)", re.I), "date_range"),
            (
                re.compile(r"what\s+happened\s+(?:during|between)\s+(.+?)\s+and\s+(.+)", re.I),
                "date_range",
            ),
            (re.compile(r"events?\s+(?:in|during)\s+(.+)", re.I), "date_period"),
            # Participant queries
            (re.compile(r"events?\s+(?:involving|with)\s+(.+)", re.I), "participant"),
            (re.compile(r"what\s+did\s+(.+?)\s+do", re.I), "participant"),
            (re.compile(r"(.+?)\'s\s+(?:reign|life|ministry)", re.I), "participant_period"),
            # Causal queries
            (re.compile(r"what\s+caused\s+(.+)", re.I), "causes"),
            (re.compile(r"what\s+led\s+to\s+(.+)", re.I), "causes"),
            (re.compile(r"consequences\s+of\s+(.+)", re.I), "effects"),
            (re.compile(r"what\s+happened\s+after\s+(.+)", re.I), "effects"),
            # Contemporary queries
            (re.compile(r"contemporary\s+(?:events?|with)\s+(.+)", re.I), "contemporary"),
            (re.compile(r"what\s+else\s+happened\s+(?:during|when)\s+(.+)", re.I), "contemporary"),
            # Biblical queries
            (
                re.compile(r"events?\s+(?:in|from)\s+(.+?)\s+(?:chapter|book)", re.I),
                "biblical_book",
            ),
            (re.compile(r"(.+?)\s+(?:in\s+the\s+)?bible", re.I), "biblical_search"),
            # Pattern queries
            (re.compile(r"(?:pattern|cycle)\s+of\s+(.+)", re.I), "pattern"),
            (re.compile(r"all\s+(?:instances|examples)\s+of\s+(.+)", re.I), "pattern"),
        ]

    def query(self, query_text: str, filters: Optional[QueryFilter] = None) -> Dict[str, Any]:
        """Process a natural language query."""
        query_text = query_text.strip()

        # Try to match natural language patterns
        for pattern, pattern_type in self.nl_patterns:
            match = pattern.match(query_text)
            if match:
                return self._process_nl_query(match, pattern_type, filters)

        # Fall back to keyword search
        return self._keyword_search(query_text, filters)

    def _process_nl_query(
        self, match: re.Match, pattern_type: str, filters: Optional[QueryFilter]
    ) -> Dict[str, Any]:
        """Process a matched natural language pattern."""

        if pattern_type == "date_range":
            start_str = match.group(1)
            end_str = match.group(2)
            return self._query_date_range(start_str, end_str, filters)

        elif pattern_type == "date_period":
            period_str = match.group(1)
            return self._query_date_period(period_str, filters)

        elif pattern_type == "participant":
            participant = match.group(1)
            return self._query_participant(participant, filters)

        elif pattern_type == "causes":
            event_name = match.group(1)
            return self._query_causes(event_name, filters)

        elif pattern_type == "effects":
            event_name = match.group(1)
            return self._query_effects(event_name, filters)

        elif pattern_type == "contemporary":
            event_name = match.group(1)
            return self._query_contemporary(event_name, filters)

        elif pattern_type == "biblical_book":
            book_name = match.group(1)
            return self._query_biblical_book(book_name, filters)

        elif pattern_type == "pattern":
            pattern_name = match.group(1)
            return self._query_pattern(pattern_name, filters)

        return {"events": [], "query_type": "unknown"}

    def _query_date_range(
        self, start_str: str, end_str: str, filters: Optional[QueryFilter]
    ) -> Dict[str, Any]:
        """Query events in a date range."""
        from .parser import ChronologyParser

        parser = ChronologyParser()
        start_parsed = parser.parse_date(start_str)
        end_parsed = parser.parse_date(end_str)

        if not start_parsed.time_point or not end_parsed.time_point:
            return {
                "events": [],
                "error": f"Could not parse date range: {start_str} to {end_str}",
                "query_type": "date_range",
            }

        # Create date range query
        date_query = DateRangeQuery(start=start_parsed.time_point, end=end_parsed.time_point)

        # Find events in range
        events = []
        for event in self.graph.nodes.values():
            if date_query.matches_event(event.event):
                if not filters or self._apply_filter(event.event, filters):
                    events.append(event.event)

        # Sort by date
        events.sort(key=lambda e: e.get_date_for_sorting() or 0)

        return {
            "events": events,
            "query_type": "date_range",
            "date_range": {
                "start": start_str,
                "end": end_str,
                "parsed_start": start_parsed.time_point.to_dict(),
                "parsed_end": end_parsed.time_point.to_dict(),
            },
        }

    def _query_participant(
        self, participant: str, filters: Optional[QueryFilter]
    ) -> Dict[str, Any]:
        """Query events involving a participant."""
        events = []
        participant_lower = participant.lower()

        for event in self.graph.nodes.values():
            # Check participants
            for p in event.event.participants:
                if participant_lower in p.name.lower():
                    if not filters or self._apply_filter(event.event, filters):
                        events.append(event.event)
                    break

            # Also check event name and description
            if (
                participant_lower in event.event.name.lower()
                or participant_lower in event.event.description.lower()
            ):
                if not filters or self._apply_filter(event.event, filters):
                    if event.event not in events:
                        events.append(event.event)

        # Sort by date
        events.sort(key=lambda e: e.get_date_for_sorting() or 0)

        return {"events": events, "query_type": "participant", "participant": participant}

    def _query_causes(self, event_name: str, filters: Optional[QueryFilter]) -> Dict[str, Any]:
        """Query events that caused the specified event."""
        # Find the target event
        target_event = self._find_event_by_name(event_name)
        if not target_event:
            return {
                "events": [],
                "error": f"Could not find event: {event_name}",
                "query_type": "causes",
            }

        # Find causal predecessors
        causes = []
        target_node = self.graph.nodes.get(target_event.id)
        if target_node:
            for edge in target_node.incoming_edges:
                if edge.relationship.relationship_type in [
                    RelationType.CAUSES,
                    RelationType.ENABLES,
                ]:
                    cause_event = edge.source.event
                    if not filters or self._apply_filter(cause_event, filters):
                        causes.append(cause_event)

        return {"events": causes, "query_type": "causes", "target_event": target_event.name}

    def _query_contemporary(
        self, event_name: str, filters: Optional[QueryFilter]
    ) -> Dict[str, Any]:
        """Query events contemporary with the specified event."""
        # Find the reference event
        ref_event = self._find_event_by_name(event_name)
        if not ref_event:
            return {
                "events": [],
                "error": f"Could not find event: {event_name}",
                "query_type": "contemporary",
            }

        # Find contemporary events
        contemporary = self.graph.find_contemporary_events(
            ref_event, max_time_distance=TimeDelta(years=50)
        )

        # Apply filters
        filtered_events = []
        for event, score in contemporary:
            if not filters or self._apply_filter(event, filters):
                filtered_events.append(event)

        return {
            "events": filtered_events,
            "query_type": "contemporary",
            "reference_event": ref_event.name,
        }

    def _find_event_by_name(self, name: str) -> Optional[Event]:
        """Find event by name (fuzzy matching)."""
        name_lower = name.lower()

        # Try exact match first
        for event in self.graph.nodes.values():
            if event.event.name.lower() == name_lower:
                return event.event

        # Try partial match
        for event in self.graph.nodes.values():
            if name_lower in event.event.name.lower():
                return event.event

        # Try description match
        for event in self.graph.nodes.values():
            if name_lower in event.event.description.lower():
                return event.event

        return None

    def _apply_filter(self, event: Event, filters: QueryFilter) -> bool:
        """Apply filters to an event."""
        # Source filters (key for user preferences)
        if filters.exclude_scholars:
            if any(scholar in event.scholars for scholar in filters.exclude_scholars):
                return False

        if filters.exclude_sources:
            if any(source in event.sources for source in filters.exclude_sources):
                return False

        if filters.exclude_traditions:
            if any(tradition in event.traditions for tradition in filters.exclude_traditions):
                return False

        if filters.include_only_scholars:
            if not any(scholar in event.scholars for scholar in filters.include_only_scholars):
                return False

        if filters.include_only_sources:
            if not any(source in event.sources for source in filters.include_only_sources):
                return False

        # Event type filter
        if filters.event_types:
            if event.event_type not in filters.event_types:
                return False

        # Certainty filter
        if filters.certainty_levels:
            if event.certainty_level not in filters.certainty_levels:
                return False

        # Category filter
        if filters.categories:
            if not any(cat in event.categories for cat in filters.categories):
                return False

        # Confidence filter
        if event.time_point and event.time_point.confidence < filters.min_confidence:
            return False

        # Biblical support filter
        if filters.require_biblical_support and not event.verse_refs:
            return False

        # Archaeological evidence filter
        if filters.require_archaeological_evidence and not event.archaeological_evidence:
            return False

        # Text filters
        if filters.name_contains:
            if filters.name_contains.lower() not in event.name.lower():
                return False

        if filters.description_contains:
            if filters.description_contains.lower() not in event.description.lower():
                return False

        # Participant filter
        if filters.participants:
            event_participants = {p.name.lower() for p in event.participants}
            filter_participants = {p.lower() for p in filters.participants}
            if not filter_participants.intersection(event_participants):
                return False

        # Location filter
        if filters.locations:
            if not event.location:
                return False
            if not any(loc.lower() in event.location.name.lower() for loc in filters.locations):
                return False

        return True

    def _keyword_search(self, query_text: str, filters: Optional[QueryFilter]) -> Dict[str, Any]:
        """Perform keyword search across events."""
        keywords = query_text.lower().split()
        events = []

        for event in self.graph.nodes.values():
            # Score event based on keyword matches
            score = 0
            text_to_search = (
                event.event.name
                + " "
                + event.event.description
                + " "
                + " ".join(event.event.categories)
                + " "
                + " ".join(p.name for p in event.event.participants)
            ).lower()

            for keyword in keywords:
                if keyword in text_to_search:
                    score += 1

            # Include if at least half the keywords match
            if score >= len(keywords) / 2:
                if not filters or self._apply_filter(event.event, filters):
                    events.append((event.event, score))

        # Sort by score
        events.sort(key=lambda x: x[1], reverse=True)

        return {
            "events": [e[0] for e in events],
            "query_type": "keyword_search",
            "keywords": keywords,
        }

    def find_prophetic_fulfillments(self, prophecy_text: str) -> Dict[str, Any]:
        """Find events that fulfill a prophecy."""
        prophecy_event = self._find_event_by_name(prophecy_text)
        if not prophecy_event:
            return {"fulfillments": [], "error": f"Could not find prophecy: {prophecy_text}"}

        fulfillments = self.traverser.find_prophetic_fulfillments(prophecy_event)

        return {
            "fulfillments": [{"event": f[0], "confidence": f[1]} for f in fulfillments],
            "prophecy": prophecy_event.name,
        }

    def find_historical_patterns(self, pattern_name: str) -> Dict[str, Any]:
        """Find historical patterns."""
        patterns = self.traverser.find_historical_patterns(pattern_name)

        return {"patterns": patterns, "pattern_type": pattern_name, "count": len(patterns)}
