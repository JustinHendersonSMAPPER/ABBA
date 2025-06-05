"""
Tests for timeline visualization components.

Test coverage for generating timeline visualizations, charts, and
interactive displays suitable for web interfaces and static exports.
"""

import pytest
from datetime import datetime
import json

from abba.timeline.visualization import (
    TimelineScale,
    VisualizationConfig,
    EventVisualization,
    RelationshipVisualization,
    TimelineVisualizer
)
from abba.timeline.models import (
    Event,
    TimePeriod,
    TimelineCollection,
    TimePoint,
    TimeRange,
    EventType,
    CertaintyLevel,
    RelationType,
    create_bce_date,
    datetime_to_bce_year
)
from abba.timeline.graph import TemporalGraph
from abba.timeline.filter import UserPreferences


class TestTimelineScale:
    """Test timeline scale calculations."""
    
    def test_scale_initialization(self):
        """Test basic scale initialization."""
        scale = TimelineScale(start_year=-1000, end_year=100)
        
        assert scale.start_year == -1000
        assert scale.end_year == 100
        assert scale.unit in ["year", "decade", "century", "millennium"]
        assert len(scale.major_ticks) > 0
        assert len(scale.minor_ticks) > 0
        
    def test_yearly_scale(self):
        """Test scale for small time spans (years)."""
        scale = TimelineScale(start_year=0, end_year=50)
        
        assert scale.unit == "year"
        # Should have major ticks every 10 years
        assert 0 in scale.major_ticks
        assert 10 in scale.major_ticks
        assert 20 in scale.major_ticks
        
    def test_decade_scale(self):
        """Test scale for medium time spans (decades)."""
        scale = TimelineScale(start_year=-500, end_year=0)
        
        assert scale.unit == "decade"
        # Should have major ticks every 100 years
        assert -500 in scale.major_ticks
        assert -400 in scale.major_ticks
        assert 0 in scale.major_ticks
        
    def test_century_scale(self):
        """Test scale for large time spans (centuries)."""
        scale = TimelineScale(start_year=-2000, end_year=0)
        
        assert scale.unit == "century"
        # Should have major ticks every 500 years
        assert -2000 in scale.major_ticks
        assert -1500 in scale.major_ticks
        assert -1000 in scale.major_ticks
        
    def test_millennium_scale(self):
        """Test scale for very large time spans (millennia)."""
        scale = TimelineScale(start_year=-10000, end_year=2000)
        
        assert scale.unit == "millennium"
        # Should have major ticks every 1000 years
        assert -10000 in scale.major_ticks
        assert -5000 in scale.major_ticks
        assert 0 in scale.major_ticks
        assert 2000 in scale.major_ticks
        
    def test_custom_ticks(self):
        """Test providing custom tick marks."""
        major_ticks = [-1000, -500, 0, 500, 1000]
        minor_ticks = [-750, -250, 250, 750]
        
        scale = TimelineScale(
            start_year=-1000,
            end_year=1000,
            major_ticks=major_ticks,
            minor_ticks=minor_ticks
        )
        
        assert scale.major_ticks == major_ticks
        assert scale.minor_ticks == minor_ticks
        
    def test_bce_ce_transition(self):
        """Test scale spanning BCE/CE transition."""
        scale = TimelineScale(start_year=-100, end_year=100)
        
        # Should have tick at year 0
        assert 0 in scale.major_ticks or 0 in scale.minor_ticks


class TestVisualizationConfig:
    """Test visualization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizationConfig()
        
        assert config.width == 1200
        assert config.height == 800
        assert config.margin_top == 50
        assert config.margin_bottom == 100
        assert config.margin_left == 100
        assert config.margin_right == 50
        assert config.show_event_labels is True
        assert config.show_uncertainty is True
        assert config.show_relationships is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = VisualizationConfig(
            width=1600,
            height=900,
            show_event_labels=False,
            color_scheme="dark"
        )
        
        assert config.width == 1600
        assert config.height == 900
        assert config.show_event_labels is False
        assert config.color_scheme == "dark"
        
    def test_effective_dimensions(self):
        """Test calculation of effective drawing area."""
        config = VisualizationConfig()
        
        effective_width = config.width - config.margin_left - config.margin_right
        effective_height = config.height - config.margin_top - config.margin_bottom
        
        assert effective_width == 1050  # 1200 - 100 - 50
        assert effective_height == 650   # 800 - 50 - 100


class TestEventVisualization:
    """Test event visualization data structure."""
    
    def test_event_visualization_creation(self):
        """Test creating event visualization."""
        event = Event(
            id="exodus",
            name="The Exodus",
            description="The Exodus from Egypt",
            time_point=TimePoint(
                exact_date=create_bce_date(1446),
                confidence=0.7
            ),
            event_type=EventType.POINT
        )
        
        viz = EventVisualization(
            event=event,
            x_position=100.5,
            y_position=200.0,
            width=15,
            height=25,
            color="#3498db",
            label="Exodus",
            tooltip="The Exodus from Egypt"
        )
        
        assert viz.event.id == "exodus"
        assert viz.x_position == 100.5
        assert viz.y_position == 200.0
        assert viz.width == 15
        assert viz.height == 25
        assert viz.color == "#3498db"
        assert viz.label == "Exodus"
        assert viz.tooltip == "The Exodus from Egypt"
        
    def test_event_visualization_to_dict(self):
        """Test converting event visualization to dictionary."""
        event = Event(
            id="test",
            name="Test Event",
            description="Test event description",
            time_point=TimePoint(
                exact_date=create_bce_date(500),
                confidence=0.9
            ),
            event_type=EventType.POINT,
            certainty_level=CertaintyLevel.PROBABLE
        )
        
        viz = EventVisualization(
            event=event,
            x_position=150,
            y_position=250,
            color="#e74c3c"
        )
        
        data_dict = viz.to_dict()
        
        assert data_dict["id"] == "test"
        assert data_dict["name"] == "Test Event"
        assert data_dict["x"] == 150
        assert data_dict["y"] == 250
        assert data_dict["color"] == "#e74c3c"
        assert data_dict["event_type"] == "point"
        assert data_dict["certainty"] == "probable"
        
    def test_event_visualization_defaults(self):
        """Test event visualization with default values."""
        event = Event(
            id="test",
            name="Test",
            description="Test event",
            event_type=EventType.POINT
        )
        
        viz = EventVisualization(
            event=event,
            x_position=0,
            y_position=0
        )
        
        assert viz.width == 10  # Default width
        assert viz.height == 20  # Default height
        assert viz.color == "#3498db"  # Default color
        assert viz.label == ""  # Default empty label
        assert viz.tooltip == ""  # Default empty tooltip


class TestRelationshipVisualization:
    """Test relationship visualization data structure."""
    
    def test_relationship_visualization_creation(self):
        """Test creating relationship visualization."""
        rel_viz = RelationshipVisualization(
            source_event_id="event1",
            target_event_id="event2",
            relationship_type=RelationType.CAUSES,
            source_x=100,
            source_y=200,
            target_x=300,
            target_y=250,
            color="#e74c3c",
            style="dashed",
            width=3
        )
        
        assert rel_viz.source_event_id == "event1"
        assert rel_viz.target_event_id == "event2"
        assert rel_viz.relationship_type == RelationType.CAUSES
        assert rel_viz.source_x == 100
        assert rel_viz.source_y == 200
        assert rel_viz.target_x == 300
        assert rel_viz.target_y == 250
        assert rel_viz.color == "#e74c3c"
        assert rel_viz.style == "dashed"
        assert rel_viz.width == 3
        
    def test_relationship_visualization_to_dict(self):
        """Test converting relationship visualization to dictionary."""
        rel_viz = RelationshipVisualization(
            source_event_id="abraham",
            target_event_id="isaac",
            relationship_type=RelationType.BEFORE,
            source_x=150,
            source_y=100,
            target_x=250,
            target_y=100
        )
        
        data_dict = rel_viz.to_dict()
        
        assert data_dict["source"] == "abraham"
        assert data_dict["target"] == "isaac"
        assert data_dict["type"] == "before"
        assert data_dict["source_x"] == 150
        assert data_dict["source_y"] == 100
        assert data_dict["target_x"] == 250
        assert data_dict["target_y"] == 100
        assert data_dict["style"] == "solid"  # Default
        assert data_dict["width"] == 2  # Default


class TestTimelineVisualizer:
    """Test timeline visualization functionality."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing."""
        return [
            Event(
                id="creation",
                name="Creation",
                description="The creation of the world",
                time_point=TimePoint(relative_to_event="beginning"),
                event_type=EventType.POINT,
                categories=["religious"]
            ),
            Event(
                id="flood",
                name="The Flood",
                description="The great flood in Noah's time",
                time_point=TimePoint(
                    exact_date=create_bce_date(2348),
                    confidence=0.6
                ),
                event_type=EventType.POINT,
                categories=["natural", "religious"]
            ),
            Event(
                id="abraham",
                name="Abraham's Call",
                description="God calls Abraham to leave Ur",
                time_point=TimePoint(
                    exact_date=create_bce_date(2091),
                    confidence=0.7
                ),
                event_type=EventType.POINT,
                categories=["religious", "political"]
            )
        ]
        
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = TimelineVisualizer()
        
        assert visualizer is not None
        assert visualizer.config is not None
        assert hasattr(visualizer, 'create_timeline_visualization')
        
    def test_visualizer_with_custom_config(self):
        """Test visualizer with custom configuration."""
        config = VisualizationConfig(
            width=1600,
            height=900,
            color_scheme="dark"
        )
        visualizer = TimelineVisualizer(config)
        
        assert visualizer.config.width == 1600
        assert visualizer.config.height == 900
        assert visualizer.config.color_scheme == "dark"
        
    def test_create_timeline_visualization(self, sample_events):
        """Test creating timeline visualization."""
        visualizer = TimelineVisualizer()
        
        viz_data = visualizer.create_timeline_visualization(sample_events)
        
        assert isinstance(viz_data, dict)
        assert "events" in viz_data
        assert "scale" in viz_data
        assert "config" in viz_data
        
    def test_create_visualization_with_filters(self, sample_events):
        """Test creating visualization with event filters."""
        visualizer = TimelineVisualizer()
        preferences = UserPreferences(
            min_confidence=0.7,
            min_certainty_level=CertaintyLevel.PROBABLE
        )
        
        viz_data = visualizer.create_timeline_visualization(
            sample_events,
            preferences=preferences
        )
        
        # Should filter out the flood event (confidence 0.6 < 0.7)
        event_names = [e["name"] for e in viz_data["events"]]
        assert "Abraham's Call" in event_names
        assert "The Flood" not in event_names  # Below certainty threshold
        
    def test_export_svg(self, sample_events):
        """Test exporting to SVG format."""
        visualizer = TimelineVisualizer()
        
        # First create visualization data
        viz_data = visualizer.create_timeline_visualization(sample_events)
        
        # Then export to SVG
        svg = visualizer.export_svg(viz_data)
        
        assert isinstance(svg, str)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        
    def test_visualization_data_structure(self, sample_events):
        """Test structure of visualization data."""
        visualizer = TimelineVisualizer()
        
        # Create visualization data
        viz_data = visualizer.create_timeline_visualization(sample_events)
        
        # Check structure
        assert isinstance(viz_data, dict)
        assert "events" in viz_data
        assert "scale" in viz_data
        assert "config" in viz_data
        
        # Check events are converted to visualization format
        assert all("x" in event for event in viz_data["events"])
        assert all("y" in event for event in viz_data["events"])
        assert all("color" in event for event in viz_data["events"])


class TestVisualizationHelpers:
    """Test helper methods in TimelineVisualizer."""
    
    def test_group_events_by_time(self):
        """Test grouping events by time for overlap handling."""
        visualizer = TimelineVisualizer()
        events = [
            Event(
                id="e1",
                name="Event 1",
                description="First event",
                time_point=TimePoint(exact_date=create_bce_date(1000)),
                event_type=EventType.POINT
            ),
            Event(
                id="e2",
                name="Event 2",
                description="Second event",
                time_point=TimePoint(exact_date=create_bce_date(1000)),
                event_type=EventType.POINT
            ),
            Event(
                id="e3",
                name="Event 3",
                description="Third event",
                time_point=TimePoint(exact_date=create_bce_date(800)),
                event_type=EventType.POINT
            )
        ]
        
        scale = TimelineScale(start_year=-1500, end_year=0)
        groups = visualizer._group_events_by_time(events, scale)
        
        assert len(groups) >= 1
        # Events at year -1000 should be grouped together
        assert any(len(group) >= 2 for group in groups.values())
        
    def test_get_event_year(self):
        """Test extracting year from events."""
        visualizer = TimelineVisualizer()
        
        # Test with exact date
        event1 = Event(
            id="e1",
            name="Event 1",
            description="Event with exact date",
            time_point=TimePoint(exact_date=create_bce_date(500)),
            event_type=EventType.POINT
        )
        year1 = visualizer._get_event_year(event1)
        assert year1 == -500
        
        # Test with date range
        event2 = Event(
            id="e2",
            name="Event 2",
            description="Event with date range",
            time_point=TimePoint(
                earliest_date=create_bce_date(600),
                latest_date=create_bce_date(400)
            ),
            event_type=EventType.PERIOD
        )
        year2 = visualizer._get_event_year(event2)
        assert year2 == -500  # Midpoint
        
        # Test with no time point
        event3 = Event(
            id="e3", 
            name="Event 3",
            description="Event without time",
            event_type=EventType.POINT
        )
        year3 = visualizer._get_event_year(event3)
        assert year3 is None
        
    def test_get_event_color(self):
        """Test color assignment for events."""
        visualizer = TimelineVisualizer()
        
        # Test category-based color
        event1 = Event(
            id="e1",
            name="Political Event",
            description="A political event",
            categories=["political"],
            event_type=EventType.POINT
        )
        color1 = visualizer._get_event_color(event1)
        assert color1 == visualizer.config.event_colors["political"]
        
        # Test default color
        event2 = Event(
            id="e2",
            name="Unknown Event",
            description="An event with unknown category",
            categories=["unknown_category"],
            event_type=EventType.POINT
        )
        color2 = visualizer._get_event_color(event2)
        assert color2 is not None  # Should have some default


def test_full_visualization_workflow():
    """Test complete visualization workflow."""
    # Create timeline data
    events = [
        Event(
            id="exodus",
            name="The Exodus",
            description="The Israelites leave Egypt",
            time_point=TimePoint(
                exact_date=create_bce_date(1446),
                confidence=0.7
            ),
            event_type=EventType.POINT,
            categories=["religious", "political"]
        ),
        Event(
            id="temple",
            name="Solomon's Temple",
            description="Construction of the First Temple",
            time_point=TimePoint(
                exact_date=create_bce_date(966),
                confidence=0.8
            ),
            event_type=EventType.POINT,
            categories=["religious", "cultural"]
        )
    ]
    
    # Configure visualization
    config = VisualizationConfig(
        width=1400,
        height=600,
        show_uncertainty=True,
        color_scheme="default"
    )
    
    # Create visualizer
    visualizer = TimelineVisualizer(config)
    
    # Create visualization data
    viz_data = visualizer.create_timeline_visualization(events)
    
    assert viz_data is not None
    assert "events" in viz_data
    assert "scale" in viz_data
    assert len(viz_data["events"]) == 2
    
    # Test different color schemes
    dark_config = VisualizationConfig(color_scheme="dark")
    dark_visualizer = TimelineVisualizer(dark_config)
    dark_colors = dark_visualizer.config.event_colors
    default_colors = visualizer.config.event_colors
    assert dark_colors["religious"] != default_colors["religious"]
    
    # Test with user preferences (filtering by confidence)
    preferences = UserPreferences(
        min_confidence=0.75,
        min_certainty_level=CertaintyLevel.POSSIBLE
    )
    
    filtered_viz = visualizer.create_timeline_visualization(events, preferences)
    # Temple event should remain (0.8 > 0.75), Exodus should be filtered (0.7 < 0.75)
    assert len(filtered_viz["events"]) == 1
    assert filtered_viz["events"][0]["name"] == "Solomon's Temple"