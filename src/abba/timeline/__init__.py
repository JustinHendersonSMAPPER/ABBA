"""
Timeline and historical context system for biblical texts.

This package provides comprehensive timeline functionality including:
- Flexible date representation with uncertainty
- Event relationship graphs
- Multi-chronology support
- Filtering and preference management
- Integration with biblical verses
- Query and visualization capabilities
"""

from .models import (
    CalendarSystem,
    CertaintyLevel,
    EventType,
    RelationType,
    TimePoint,
    TimeRange,
    TimeDelta,
    Event,
    EventRelationship,
    TimePeriod,
    ChronologyTrack,
    TimelineCollection,
)
from .uncertainty import DateDistribution, UncertaintyCalculator, ConfidenceAggregator
from .parser import ChronologyParser, BiblicalDateParser, ScholarlyNotationParser
from .graph import TemporalGraph, EventNode, RelationshipEdge, GraphTraverser, PathFinder
from .query import TimelineQuery, QueryFilter, DateRangeQuery, EventPattern, PatternMatcher
from .filter import (
    TimelineFilter,
    UserPreferences,
    PreferenceManager,
    FilterRule,
    FilterGroup,
    FilterOperator,
)
from .visualization import (
    TimelineVisualizer,
    VisualizationConfig,
    TimelineScale,
    EventVisualization,
    RelationshipVisualization,
)

__all__ = [
    # Core models
    "CalendarSystem",
    "CertaintyLevel",
    "EventType",
    "RelationType",
    "TimePoint",
    "TimeRange",
    "TimeDelta",
    "Event",
    "EventRelationship",
    "TimePeriod",
    "ChronologyTrack",
    "TimelineCollection",
    # Uncertainty handling
    "DateDistribution",
    "UncertaintyCalculator",
    "ConfidenceAggregator",
    # Parsing
    "ChronologyParser",
    "BiblicalDateParser",
    "ScholarlyNotationParser",
    # Graph system
    "TemporalGraph",
    "EventNode",
    "RelationshipEdge",
    "GraphTraverser",
    "PathFinder",
    # Query engine
    "TimelineQuery",
    "QueryFilter",
    "DateRangeQuery",
    "EventPattern",
    "PatternMatcher",
    # Filtering
    "TimelineFilter",
    "UserPreferences",
    "PreferenceManager",
    "FilterRule",
    "FilterGroup",
    "FilterOperator",
    # Visualization
    "TimelineVisualizer",
    "VisualizationConfig",
    "TimelineScale",
    "EventVisualization",
    "RelationshipVisualization",
]
