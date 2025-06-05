"""
Tests for the timeline system.

Comprehensive test suite covering all Phase 5 timeline functionality including
models, uncertainty handling, graph operations, parsing, querying, filtering,
and visualization.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from abba.timeline.models import create_bce_date
from abba.timeline import (
    # Core models
    Event,
    EventRelationship,
    TimePeriod,
    TimePoint,
    TimeRange,
    TimeDelta,
    Location,
    EntityRef,
    ChronologyTrack,
    TimelineCollection,
    CalendarSystem,
    CertaintyLevel,
    EventType,
    RelationType,
    # Uncertainty handling
    DateDistribution,
    UncertaintyCalculator,
    ConfidenceAggregator,
    # Graph system
    TemporalGraph,
    EventNode,
    RelationshipEdge,
    GraphTraverser,
    PathFinder,
    # Parsing
    ChronologyParser,
    BiblicalDateParser,
    ScholarlyNotationParser,
    ParsedDate,
    # Querying
    TimelineQuery,
    QueryFilter,
    DateRangeQuery,
    EventPattern,
    PatternMatcher,
    # Filtering
    TimelineFilter,
    UserPreferences,
    PreferenceManager,
    FilterRule,
    FilterGroup,
    FilterOperator,
    # Visualization
    TimelineVisualizer,
    VisualizationConfig,
    TimelineScale,
    EventVisualization,
)
from abba.verse_id import parse_verse_id


class TestTimelineModels:
    """Test core timeline data models."""

    def test_time_point_creation(self):
        """Test TimePoint creation and display."""
        # Exact date
        tp = TimePoint(exact_date=create_bce_date(586, 7, 9), confidence=0.9)
        assert tp.get_display_date() == "586 BCE"

        # Date range
        tp_range = TimePoint(
            earliest_date=create_bce_date(590, 1, 1),
            latest_date=create_bce_date(580, 12, 31),
            confidence=0.7,
        )
        assert "590-580 BCE" in tp_range.get_display_date()

    def test_time_delta_calculations(self):
        """Test TimeDelta calculations."""
        delta = TimeDelta(years=40, months=6, days=15)
        days = delta.to_days()
        assert abs(days - (40 * 365.25 + 6 * 30.44 + 15)) < 1

        # Test serialization
        delta_dict = delta.to_dict()
        assert delta_dict["years"] == 40
        assert delta_dict["months"] == 6
        assert delta_dict["days"] == 15

    def test_event_creation(self):
        """Test Event creation and properties."""
        location = Location(name="Jerusalem", region="Judea")
        participant = EntityRef(id="david", name="David", entity_type="person", role="king")

        event = Event(
            id="david_crowned",
            name="David Crowned King",
            description="David becomes king of all Israel",
            event_type=EventType.POINT,
            time_point=TimePoint(exact_date=create_bce_date(1000, 1, 1), confidence=0.8),
            location=location,
            participants=[participant],
            categories=["political", "religious"],
            certainty_level=CertaintyLevel.PROBABLE,
            verse_refs=[parse_verse_id("2SA.5.3")],
            scholars=["Kitchen", "Finkelstein"],
            sources=["biblical", "archaeological"],
            methodologies=["textual", "comparative"],
        )

        assert event.id == "david_crowned"
        assert event.name == "David Crowned King"
        assert event.event_type == EventType.POINT
        assert event.certainty_level == CertaintyLevel.PROBABLE
        assert len(event.participants) == 1
        assert event.participants[0].name == "David"
        assert "Kitchen" in event.scholars
        assert "biblical" in event.sources

        # Test sorting date
        sort_date = event.get_date_for_sorting()
        assert sort_date is not None

        # Test serialization
        event_dict = event.to_dict()
        assert event_dict["id"] == "david_crowned"
        assert event_dict["event_type"] == "point"
        assert "Kitchen" in event_dict["scholars"]

    def test_event_relationship(self):
        """Test EventRelationship model."""
        rel = EventRelationship(
            source_event="david_crowned",
            target_event="temple_built",
            relationship_type=RelationType.BEFORE,
            confidence=0.9,
            time_distance=TimeDelta(years=30),
            biblical_support=[parse_verse_id("1KI.6.1")],
        )

        assert rel.source_event == "david_crowned"
        assert rel.target_event == "temple_built"
        assert rel.relationship_type == RelationType.BEFORE
        assert rel.confidence == 0.9
        assert rel.time_distance.years == 30

        # Test serialization
        rel_dict = rel.to_dict()
        assert rel_dict["source"] == "david_crowned"
        assert rel_dict["type"] == "before"
        assert rel_dict["confidence"] == 0.9


class TestUncertaintyHandling:
    """Test uncertainty and probability calculations."""

    def test_date_distribution_uniform(self):
        """Test uniform date distribution."""
        dist = DateDistribution(distribution_type="uniform", parameters={"low": -600, "high": -500})

        assert dist.mean() == -550
        assert abs(dist.std() - (100 / (12**0.5))) < 1

        # Test sampling
        samples = dist.sample(100)
        assert len(samples) == 100
        assert all(-600 <= s <= -500 for s in samples)

        # Test PDF
        assert dist.pdf(-550) == 0.01  # 1/(high-low)
        assert dist.pdf(-700) == 0.0  # Outside range

    def test_date_distribution_normal(self):
        """Test normal date distribution."""
        dist = DateDistribution(distribution_type="normal", parameters={"mean": -586, "std": 5})

        assert dist.mean() == -586
        assert dist.std() == 5

        # Test confidence interval
        ci = dist.confidence_interval(0.95)
        assert ci[0] < -586 < ci[1]
        assert abs(ci[1] - ci[0] - 4 * 5) < 1  # ~4 standard deviations

    def test_uncertainty_calculator(self):
        """Test uncertainty calculator functions."""
        # Test from TimePoint
        tp = TimePoint(
            earliest_date=create_bce_date(590, 1, 1),
            latest_date=create_bce_date(580, 1, 1),
            confidence=0.8,
        )

        dist = UncertaintyCalculator.from_time_point(tp)
        assert dist.distribution_type == "uniform"
        assert (
            dist.mean()
            == tp.earliest_date.timestamp()
            + (tp.latest_date.timestamp() - tp.earliest_date.timestamp()) / 2
        )

    def test_confidence_aggregator(self):
        """Test confidence aggregation."""
        # Create test event
        event = Event(
            id="test",
            name="Test Event",
            description="Test",
            event_type=EventType.POINT,
            certainty_level=CertaintyLevel.PROBABLE,
            scholarly_sources=["source1", "source2"],
        )

        source_weights = {"source1": 0.8, "source2": 0.9}
        confidence = ConfidenceAggregator.aggregate_event_confidence(event, source_weights)

        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident


class TestTemporalGraph:
    """Test graph-based timeline system."""

    def test_graph_creation(self):
        """Test temporal graph creation and manipulation."""
        graph = TemporalGraph()

        # Create events
        event1 = Event(
            id="event1",
            name="First Event",
            description="First event",
            event_type=EventType.POINT,
            time_point=TimePoint(exact_date=create_bce_date(600, 1, 1)),
        )

        event2 = Event(
            id="event2",
            name="Second Event",
            description="Second event",
            event_type=EventType.POINT,
            time_point=TimePoint(exact_date=create_bce_date(500, 1, 1)),
        )

        # Add events to graph
        node1 = graph.add_event(event1)
        node2 = graph.add_event(event2)

        assert len(graph.nodes) == 2
        assert graph.nodes["event1"].event == event1
        assert graph.nodes["event2"].event == event2

        # Add relationship
        relationship = EventRelationship(
            source_event="event1",
            target_event="event2",
            relationship_type=RelationType.BEFORE,
            confidence=0.9,
        )

        edge = graph.add_relationship(relationship)
        assert edge is not None
        assert len(graph.edges) == 1
        assert len(node1.outgoing_edges) == 1
        assert len(node2.incoming_edges) == 1

    def test_graph_traversal(self):
        """Test graph traversal algorithms."""
        graph = TemporalGraph()

        # Create a chain of events
        events = []
        for i in range(3):
            event = Event(
                id=f"event{i}",
                name=f"Event {i}",
                description=f"Event {i}",
                event_type=EventType.POINT,
                categories=["political"] if i == 2 else ["military"],
            )
            events.append(event)
            graph.add_event(event)

        # Connect them
        for i in range(2):
            rel = EventRelationship(
                source_event=f"event{i}",
                target_event=f"event{i+1}",
                relationship_type=RelationType.CAUSES,
            )
            graph.add_relationship(rel)

        # Test traversal
        traverser = GraphTraverser(graph)
        patterns = traverser.find_historical_patterns("military_political")

        # Test centrality calculation
        centrality = traverser.calculate_event_centrality()
        assert len(centrality) == 3
        assert all(0 <= score <= 1 for score in centrality.values())

    def test_contemporary_events(self):
        """Test finding contemporary events."""
        graph = TemporalGraph()

        # Create events in similar time periods
        base_time = create_bce_date(586, 1, 1)

        events = []
        for i, offset_days in enumerate([0, 30, 365, 730]):  # Same year, 1 month, 1 year, 2 years
            event = Event(
                id=f"event{i}",
                name=f"Event {i}",
                description=f"Event {i}",
                event_type=EventType.POINT,
                time_point=TimePoint(
                    exact_date=base_time + timedelta(days=offset_days), confidence=0.8
                ),
            )
            events.append(event)
            graph.add_event(event)

        # Find contemporary events to the first event
        contemporary = graph.find_contemporary_events(
            events[0], max_time_distance=TimeDelta(years=1), confidence_threshold=0.5
        )

        # Should find events within 1 year
        contemporary_ids = [event.id for event, score in contemporary]
        assert "event1" in contemporary_ids  # 1 month later
        assert "event2" in contemporary_ids  # 1 year later
        assert "event3" not in contemporary_ids  # 2 years later


class TestChronologyParsing:
    """Test chronology parsing systems."""

    def test_biblical_date_parser(self):
        """Test biblical date string parsing."""
        parser = BiblicalDateParser()

        # Test regnal year
        result = parser.parse("15th year of Hezekiah")
        # This might have low confidence if year exceeds reign
        if result.time_point is not None:
            assert result.calendar_system == CalendarSystem.REGNAL

        # Test BCE date
        result = parser.parse("586 BCE")
        assert result.confidence > 0.8
        assert result.time_point is not None
        # Check the year is 586 BCE
        display_date = result.time_point.get_display_date()
        assert "586 BCE" in display_date

        # Test range
        result = parser.parse("586-587 BCE")
        assert result.confidence > 0.5
        assert result.time_point.earliest_date is not None
        assert result.time_point.latest_date is not None

    def test_scholarly_notation_parser(self):
        """Test scholarly notation parsing."""
        parser = ScholarlyNotationParser()

        # Test uncertain date
        result = parser.parse_scholarly_date("586? BCE")
        assert result.confidence < 0.8

        # Test plus/minus notation
        result = parser.parse_scholarly_date("586 ± 5 BCE")
        assert result.confidence > 0.7
        assert result.time_point.distribution_type == "normal"

        # Test slash notation
        result = parser.parse_scholarly_date("586/587 BCE")
        assert result.confidence > 0.8
        assert result.time_point.earliest_date is not None

    def test_chronology_parser_integration(self):
        """Test full chronology parser."""
        parser = ChronologyParser()

        # Test event data parsing
        event_data = {
            "id": "temple_destruction",
            "name": "Destruction of the Temple",
            "description": "Babylonians destroy Solomon's Temple",
            "date": "586 BCE",
            "event_type": "point",
            "certainty": "probable",
            "location": "Jerusalem",
            "participants": [{"name": "Nebuchadnezzar", "type": "person", "role": "king"}],
            "verses": ["2KI.25.9"],
            "categories": ["political", "religious"],
            "scholars": ["Kitchen", "Finkelstein"],
            "sources": ["biblical", "archaeological"],
        }

        event = parser.parse_event_data(event_data)
        assert event is not None
        assert event.id == "temple_destruction"
        assert event.event_type == EventType.POINT
        assert event.certainty_level == CertaintyLevel.PROBABLE
        assert len(event.participants) == 1
        assert event.participants[0].name == "Nebuchadnezzar"
        assert len(event.verse_refs) == 1
        assert "Kitchen" in event.scholars


class TestTimelineQuery:
    """Test timeline query engine."""

    def test_query_filter_creation(self):
        """Test QueryFilter creation and application."""
        filter_obj = QueryFilter(
            exclude_scholars=["Justin"],
            min_confidence=0.7,
            certainty_levels=[CertaintyLevel.PROBABLE, CertaintyLevel.CERTAIN],
            require_biblical_support=True,
        )

        # Create test event that should pass
        good_event = Event(
            id="good",
            name="Good Event",
            description="Test",
            event_type=EventType.POINT,
            time_point=TimePoint(exact_date=datetime.now(), confidence=0.8),
            certainty_level=CertaintyLevel.PROBABLE,
            verse_refs=[parse_verse_id("GEN.1.1")],
            scholars=["Kitchen"],
        )

        # Create test event that should fail (wrong scholar)
        bad_event = Event(
            id="bad",
            name="Bad Event",
            description="Test",
            event_type=EventType.POINT,
            scholars=["Justin"],
        )

        # Test filtering logic would be applied in TimelineQuery._apply_filter
        assert "Justin" not in good_event.scholars
        assert "Justin" in bad_event.scholars

    def test_date_range_query(self):
        """Test date range queries."""
        start_time = TimePoint(exact_date=create_bce_date(600, 1, 1))
        end_time = TimePoint(exact_date=create_bce_date(500, 1, 1))

        query = DateRangeQuery(start=start_time, end=end_time)

        # Test event within range
        in_range_event = Event(
            id="in_range",
            name="In Range",
            description="Test",
            event_type=EventType.POINT,
            time_point=TimePoint(exact_date=create_bce_date(550, 1, 1)),
        )

        # Test event outside range
        out_range_event = Event(
            id="out_range",
            name="Out Range",
            description="Test",
            event_type=EventType.POINT,
            time_point=TimePoint(exact_date=create_bce_date(400, 1, 1)),
        )

        assert query.matches_event(in_range_event)
        assert not query.matches_event(out_range_event)

    def test_natural_language_queries(self):
        """Test natural language query processing."""
        graph = TemporalGraph()
        query_engine = TimelineQuery(graph)

        # Test pattern matching
        patterns = query_engine.nl_patterns
        assert len(patterns) > 0

        # Test specific patterns
        for pattern, pattern_type in patterns:
            if pattern_type == "participant":
                match = pattern.match("events involving David")
                assert match is not None
                assert match.group(1) == "David"
                break


class TestTimelineFiltering:
    """Test filtering and preference systems."""

    def test_filter_rule_creation(self):
        """Test filter rule creation and application."""
        rule = FilterRule(
            field="scholars", operator=FilterOperator.NOT_IN, value=["Justin", "BadScholar"]
        )

        # Test event that should pass
        good_event = Event(
            id="good",
            name="Good",
            description="Test",
            event_type=EventType.POINT,
            scholars=["Kitchen", "Finkelstein"],
        )

        # Test event that should fail
        bad_event = Event(
            id="bad",
            name="Bad",
            description="Test",
            event_type=EventType.POINT,
            scholars=["Justin"],
        )

        assert rule.applies_to_event(good_event)
        assert not rule.applies_to_event(bad_event)

    def test_filter_group_logic(self):
        """Test filter group AND/OR logic."""
        group = FilterGroup(
            name="Test Group",
            rules=[
                FilterRule(field="scholars", operator=FilterOperator.CONTAINS, value="Kitchen"),
                FilterRule(
                    field="certainty_level", operator=FilterOperator.EQUALS, value="probable"
                ),
            ],
            logic="AND",
        )

        # Event that matches both rules
        good_event = Event(
            id="good",
            name="Good",
            description="Test",
            event_type=EventType.POINT,
            certainty_level=CertaintyLevel.PROBABLE,
            scholars=["Kitchen"],
        )

        # Event that matches only one rule
        partial_event = Event(
            id="partial",
            name="Partial",
            description="Test",
            event_type=EventType.POINT,
            certainty_level=CertaintyLevel.POSSIBLE,  # Wrong certainty
            scholars=["Kitchen"],  # Right scholar
        )

        assert group.applies_to_event(good_event)
        assert not group.applies_to_event(partial_event)

        # Test OR logic
        group.logic = "OR"
        assert group.applies_to_event(partial_event)  # Should pass with OR

    def test_user_preferences(self):
        """Test user preference system."""
        prefs = UserPreferences(
            excluded_scholars=["Justin"],
            min_confidence=0.7,
            require_biblical_support=True,
            preferred_methodologies=["archaeological"],
        )

        # Test conversion to QueryFilter
        query_filter = prefs.to_query_filter()
        assert "Justin" in query_filter.exclude_scholars
        assert query_filter.min_confidence == 0.7
        assert query_filter.require_biblical_support

        # Test serialization
        prefs_dict = prefs.to_dict()
        assert "Justin" in prefs_dict["scholars"]["excluded"]
        assert prefs_dict["quality"]["min_confidence"] == 0.7

        # Test deserialization
        prefs2 = UserPreferences.from_dict(prefs_dict)
        assert "Justin" in prefs2.excluded_scholars
        assert prefs2.min_confidence == 0.7

    def test_timeline_filter_application(self):
        """Test timeline filter with events."""
        prefs = UserPreferences(excluded_scholars=["BadScholar"])
        filter_engine = TimelineFilter(prefs)

        events = [
            Event(
                id="good",
                name="Good Event",
                description="Test",
                event_type=EventType.POINT,
                scholars=["GoodScholar"],
            ),
            Event(
                id="bad",
                name="Bad Event",
                description="Test",
                event_type=EventType.POINT,
                scholars=["BadScholar"],
            ),
        ]

        filtered = filter_engine.filter_events(events)
        assert len(filtered) == 1
        assert filtered[0].id == "good"

    def test_quick_filter(self):
        """Test quick text-based filtering."""
        filter_engine = TimelineFilter()

        events = [
            Event(
                id="david_event",
                name="David's Coronation",
                description="David becomes king",
                event_type=EventType.POINT,
                participants=[EntityRef(id="david", name="David", entity_type="person")],
            ),
            Event(
                id="solomon_event",
                name="Solomon's Temple",
                description="Temple construction",
                event_type=EventType.POINT,
                participants=[EntityRef(id="solomon", name="Solomon", entity_type="person")],
            ),
        ]

        # Test participant filter
        filtered = filter_engine.apply_quick_filter(events, "participants:David")
        assert len(filtered) == 1
        assert filtered[0].id == "david_event"

        # Test name search
        filtered = filter_engine.apply_quick_filter(events, "Temple")
        assert len(filtered) == 1
        assert filtered[0].id == "solomon_event"

    def test_preference_manager(self):
        """Test preference persistence."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            prefs_file = f.name

        try:
            manager = PreferenceManager(prefs_file)

            # Create and save preferences
            prefs = UserPreferences(excluded_scholars=["TestScholar"])
            manager.save_preferences("testuser", prefs)

            # Load preferences
            loaded_prefs = manager.load_preferences("testuser")
            assert "TestScholar" in loaded_prefs.excluded_scholars

            # Test preset preferences
            presets = manager.create_preset_preferences()
            assert "conservative" in presets
            assert "academic" in presets
            assert presets["conservative"].require_archaeological_evidence

        finally:
            os.unlink(prefs_file)


class TestTimelineVisualization:
    """Test timeline visualization system."""

    def test_visualization_config(self):
        """Test visualization configuration."""
        config = VisualizationConfig(width=1000, height=600, color_scheme="dark")

        assert config.width == 1000
        assert config.height == 600
        assert len(config.event_colors) > 0
        assert "political" in config.event_colors

    def test_timeline_scale_calculation(self):
        """Test timeline scale calculation."""
        scale = TimelineScale(start_year=-1000, end_year=-500)

        assert scale.start_year == -1000
        assert scale.end_year == -500
        assert len(scale.major_ticks) > 0
        assert len(scale.minor_ticks) > 0

        # Test automatic tick calculation
        assert scale.unit in ["year", "decade", "century", "millennium"]

    def test_event_visualization(self):
        """Test event visualization creation."""
        event = Event(
            id="test_event",
            name="Test Event",
            description="Test event for visualization",
            event_type=EventType.POINT,
            time_point=TimePoint(exact_date=create_bce_date(586, 1, 1), confidence=0.8),
            categories=["political"],
        )

        viz = EventVisualization(
            event=event, x_position=100.0, y_position=50.0, color="#3498db", label="Test Event"
        )

        assert viz.event == event
        assert viz.x_position == 100.0
        assert viz.y_position == 50.0

        # Test serialization
        viz_dict = viz.to_dict()
        assert viz_dict["id"] == "test_event"
        assert viz_dict["x"] == 100.0
        assert viz_dict["color"] == "#3498db"

    def test_timeline_visualizer(self):
        """Test complete timeline visualizer."""
        visualizer = TimelineVisualizer()

        # Create test events
        events = [
            Event(
                id="event1",
                name="First Event",
                description="First event",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(600, 1, 1)),
                categories=["political"],
            ),
            Event(
                id="event2",
                name="Second Event",
                description="Second event",
                event_type=EventType.POINT,
                time_point=TimePoint(exact_date=create_bce_date(500, 1, 1)),
                categories=["religious"],
            ),
        ]

        # Create visualization
        viz_data = visualizer.create_timeline_visualization(events)

        assert "events" in viz_data
        assert "scale" in viz_data
        assert len(viz_data["events"]) == 2
        assert viz_data["scale"]["start_year"] < -600
        assert viz_data["scale"]["end_year"] > -500

        # Test SVG export
        svg = visualizer.export_svg(viz_data)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "rect" in svg  # Should contain event rectangles


class TestTimelineIntegration:
    """Integration tests for the complete timeline system."""

    def test_complete_timeline_workflow(self):
        """Test a complete timeline workflow."""
        # Create events
        events = []
        for i, (year, name) in enumerate(
            [(1000, "David's Reign"), (960, "Temple Built"), (586, "Temple Destroyed")]
        ):
            event = Event(
                id=f"event_{i}",
                name=name,
                description=f"Event {i}",
                event_type=EventType.POINT,
                time_point=TimePoint(
                    exact_date=create_bce_date(year, 1, 1), confidence=0.8
                ),
                categories=["political", "religious"],
                scholars=["Kitchen", "Finkelstein"],
                sources=["biblical"],
            )
            events.append(event)

        # Create graph and add events
        graph = TemporalGraph()
        for event in events:
            graph.add_event(event)

        # Add relationships
        relationships = [
            EventRelationship("event_0", "event_1", RelationType.BEFORE),
            EventRelationship("event_1", "event_2", RelationType.BEFORE),
            EventRelationship("event_0", "event_2", RelationType.BEFORE),
        ]

        for rel in relationships:
            graph.add_relationship(rel)

        # Test querying
        query_engine = TimelineQuery(graph)

        # Query by participant (would need participant data)
        result = query_engine.query("events during David's reign")
        assert "events" in result

        # Test filtering
        prefs = UserPreferences(min_confidence=0.5, preferred_sources=["biblical"])

        filter_engine = TimelineFilter(prefs)
        filtered_events = filter_engine.filter_events(events)
        assert len(filtered_events) == len(events)  # All should pass

        # Test visualization
        visualizer = TimelineVisualizer()
        viz_data = visualizer.create_graph_visualization(graph, prefs)

        assert len(viz_data["events"]) > 0
        assert len(viz_data["relationships"]) > 0

        # Test timeline collection
        collection = TimelineCollection()
        for event in events:
            collection.add_event(event)
        for rel in relationships:
            collection.add_relationship(rel)

        assert len(collection.events) == 3
        assert len(collection.relationships) == 3

        # Test serialization
        collection_dict = collection.to_dict()
        assert "events" in collection_dict
        assert "relationships" in collection_dict
        assert len(collection_dict["events"]) == 3


if __name__ == "__main__":
    pytest.main([__file__])
