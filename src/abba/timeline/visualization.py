"""
Timeline visualization helpers.

Provides utilities for generating timeline visualizations, charts, and
interactive displays suitable for web interfaces and static exports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
from math import log10, floor

from .models import (
    Event,
    TimePeriod,
    TimelineCollection,
    TimePoint,
    TimeRange,
    EventType,
    CertaintyLevel,
    RelationType,
    datetime_to_bce_year,
)
from .graph import TemporalGraph
from .filter import UserPreferences


@dataclass
class TimelineScale:
    """Represents the time scale for visualization."""

    start_year: int
    end_year: int
    unit: str = "year"  # "year", "decade", "century", "millennium"
    major_ticks: List[int] = field(default_factory=list)
    minor_ticks: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Calculate tick marks automatically."""
        if not self.major_ticks or not self.minor_ticks:
            self._calculate_ticks()

    def _calculate_ticks(self):
        """Calculate appropriate tick marks for the scale."""
        span = self.end_year - self.start_year

        if span <= 100:
            # Yearly scale
            self.unit = "year"
            major_interval = 10 if span <= 50 else 20
            minor_interval = 1 if span <= 20 else 5
        elif span <= 1000:
            # Decade scale
            self.unit = "decade"
            major_interval = 100
            minor_interval = 10
        elif span <= 5000:
            # Century scale
            self.unit = "century"
            major_interval = 500
            minor_interval = 100
        else:
            # Millennium scale
            self.unit = "millennium"
            major_interval = 1000
            minor_interval = 500

        # Generate ticks
        start_major = (self.start_year // major_interval) * major_interval
        end_major = ((self.end_year // major_interval) + 1) * major_interval

        self.major_ticks = list(range(start_major, end_major + 1, major_interval))

        start_minor = (self.start_year // minor_interval) * minor_interval
        end_minor = ((self.end_year // minor_interval) + 1) * minor_interval

        self.minor_ticks = [
            tick
            for tick in range(start_minor, end_minor + 1, minor_interval)
            if tick not in self.major_ticks
        ]


@dataclass
class VisualizationConfig:
    """Configuration for timeline visualizations."""

    # Layout settings
    width: int = 1200
    height: int = 800
    margin_top: int = 50
    margin_bottom: int = 100
    margin_left: int = 100
    margin_right: int = 50

    # Event display
    show_event_labels: bool = True
    show_uncertainty: bool = True
    show_relationships: bool = True
    max_events_per_layer: int = 20

    # Colors and styling
    color_scheme: str = "default"  # "default", "dark", "colorblind"
    event_colors: Dict[str, str] = field(default_factory=dict)
    uncertainty_alpha: float = 0.3

    # Interactive features
    enable_zoom: bool = True
    enable_tooltips: bool = True
    enable_filtering: bool = True

    # Export settings
    include_legend: bool = True
    include_scale: bool = True
    format: str = "svg"  # "svg", "png", "html", "json"

    def __post_init__(self):
        """Set default colors."""
        if not self.event_colors:
            self.event_colors = self._get_default_colors()

    def _get_default_colors(self) -> Dict[str, str]:
        """Get default color scheme."""
        if self.color_scheme == "dark":
            return {
                "political": "#4A90E2",
                "religious": "#7ED321",
                "military": "#D0021B",
                "cultural": "#F5A623",
                "natural": "#50E3C2",
                "prophetic": "#9013FE",
                "default": "#50E3C2",
            }
        elif self.color_scheme == "colorblind":
            return {
                "political": "#1f77b4",
                "religious": "#ff7f0e",
                "military": "#2ca02c",
                "cultural": "#d62728",
                "natural": "#9467bd",
                "prophetic": "#8c564b",
                "default": "#7f7f7f",
            }
        else:  # default
            return {
                "political": "#3498db",
                "religious": "#e74c3c",
                "military": "#e67e22",
                "cultural": "#f39c12",
                "natural": "#27ae60",
                "prophetic": "#9b59b6",
                "default": "#95a5a6",
            }


@dataclass
class EventVisualization:
    """Represents an event for visualization."""

    event: Event
    x_position: float
    y_position: float
    width: float = 10
    height: float = 20
    color: str = "#3498db"
    label: str = ""
    tooltip: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "id": self.event.id,
            "name": self.event.name,
            "description": self.event.description,
            "x": self.x_position,
            "y": self.y_position,
            "width": self.width,
            "height": self.height,
            "color": self.color,
            "label": self.label,
            "tooltip": self.tooltip,
            "event_type": self.event.event_type.value,
            "certainty": self.event.certainty_level.value,
            "date": self.event.time_point.get_display_date() if self.event.time_point else None,
        }


@dataclass
class RelationshipVisualization:
    """Represents a relationship for visualization."""

    source_event_id: str
    target_event_id: str
    relationship_type: RelationType
    source_x: float
    source_y: float
    target_x: float
    target_y: float
    color: str = "#7f8c8d"
    style: str = "solid"  # "solid", "dashed", "dotted"
    width: float = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "source": self.source_event_id,
            "target": self.target_event_id,
            "type": self.relationship_type.value,
            "source_x": self.source_x,
            "source_y": self.source_y,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "color": self.color,
            "style": self.style,
            "width": self.width,
        }


class TimelineVisualizer:
    """Main timeline visualization engine."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer."""
        self.config = config or VisualizationConfig()

        # Relationship styling
        self.relationship_styles = {
            RelationType.BEFORE: {"color": "#34495e", "style": "solid"},
            RelationType.AFTER: {"color": "#34495e", "style": "solid"},
            RelationType.CAUSES: {"color": "#e74c3c", "style": "solid", "width": 3},
            RelationType.FULFILLS: {"color": "#9b59b6", "style": "dashed"},
            RelationType.CONTEMPORARY: {"color": "#95a5a6", "style": "dotted"},
            RelationType.RELATED_TO: {"color": "#bdc3c7", "style": "dotted"},
        }

    def create_timeline_visualization(
        self, events: List[Event], preferences: Optional[UserPreferences] = None
    ) -> Dict[str, Any]:
        """Create a complete timeline visualization."""

        # Filter events if preferences provided
        if preferences:
            from .filter import TimelineFilter

            filter_engine = TimelineFilter(preferences)
            events = filter_engine.filter_events(events)

        if not events:
            return {
                "events": [],
                "relationships": [],
                "scale": None,
                "error": "No events to display",
            }

        # Calculate time scale
        scale = self._calculate_scale(events)

        # Position events
        event_viz = self._position_events(events, scale)

        # Create visualization data
        result = {
            "events": [ev.to_dict() for ev in event_viz],
            "relationships": [],
            "scale": {
                "start_year": scale.start_year,
                "end_year": scale.end_year,
                "unit": scale.unit,
                "major_ticks": scale.major_ticks,
                "minor_ticks": scale.minor_ticks,
            },
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "margins": {
                    "top": self.config.margin_top,
                    "bottom": self.config.margin_bottom,
                    "left": self.config.margin_left,
                    "right": self.config.margin_right,
                },
            },
            "legend": self._create_legend(events) if self.config.include_legend else None,
        }

        return result

    def create_graph_visualization(
        self, graph: TemporalGraph, preferences: Optional[UserPreferences] = None
    ) -> Dict[str, Any]:
        """Create a graph-based visualization with relationships."""

        events = [node.event for node in graph.nodes.values()]

        # Filter events
        if preferences:
            from .filter import TimelineFilter

            filter_engine = TimelineFilter(preferences)
            events = filter_engine.filter_events(events)

        if not events:
            return {"events": [], "relationships": [], "error": "No events to display"}

        # Calculate positions
        scale = self._calculate_scale(events)
        event_viz = self._position_events(events, scale)

        # Create event position lookup
        event_positions = {ev.event.id: (ev.x_position, ev.y_position) for ev in event_viz}

        # Add relationships
        relationships = []
        for edge in graph.edges:
            if edge.source.event.id in event_positions and edge.target.event.id in event_positions:

                source_x, source_y = event_positions[edge.source.event.id]
                target_x, target_y = event_positions[edge.target.event.id]

                style = self.relationship_styles.get(
                    edge.relationship.relationship_type,
                    {"color": "#bdc3c7", "style": "solid", "width": 2},
                )

                rel_viz = RelationshipVisualization(
                    source_event_id=edge.source.event.id,
                    target_event_id=edge.target.event.id,
                    relationship_type=edge.relationship.relationship_type,
                    source_x=source_x,
                    source_y=source_y,
                    target_x=target_x,
                    target_y=target_y,
                    color=style["color"],
                    style=style["style"],
                    width=style.get("width", 2),
                )
                relationships.append(rel_viz)

        return {
            "events": [ev.to_dict() for ev in event_viz],
            "relationships": [rel.to_dict() for rel in relationships],
            "scale": {
                "start_year": scale.start_year,
                "end_year": scale.end_year,
                "unit": scale.unit,
                "major_ticks": scale.major_ticks,
                "minor_ticks": scale.minor_ticks,
            },
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "show_relationships": self.config.show_relationships,
                "margins": {
                    "top": self.config.margin_top,
                    "bottom": self.config.margin_bottom,
                    "left": self.config.margin_left,
                    "right": self.config.margin_right,
                },
            },
        }

    def _calculate_scale(self, events: List[Event]) -> TimelineScale:
        """Calculate appropriate time scale for events."""
        years = []

        for event in events:
            if event.time_point:
                if event.time_point.exact_date:
                    years.append(datetime_to_bce_year(event.time_point.exact_date))
                elif event.time_point.earliest_date and event.time_point.latest_date:
                    years.extend(
                        [datetime_to_bce_year(event.time_point.earliest_date), 
                         datetime_to_bce_year(event.time_point.latest_date)]
                    )

        if not years:
            # Default biblical time range
            return TimelineScale(-2000, 100)

        min_year = min(years)
        max_year = max(years)

        # Add padding
        span = max_year - min_year
        padding = max(50, span * 0.1)

        return TimelineScale(start_year=int(min_year - padding), end_year=int(max_year + padding))

    def _position_events(
        self, events: List[Event], scale: TimelineScale
    ) -> List[EventVisualization]:
        """Position events on the timeline."""
        event_viz = []

        # Calculate available width and height
        available_width = self.config.width - self.config.margin_left - self.config.margin_right
        available_height = self.config.height - self.config.margin_top - self.config.margin_bottom

        # Group events by approximate time to handle overlaps
        time_groups = self._group_events_by_time(events, scale)

        for group_year, group_events in time_groups.items():
            # Calculate x position based on year
            year_ratio = (group_year - scale.start_year) / (scale.end_year - scale.start_year)
            base_x = self.config.margin_left + (year_ratio * available_width)

            # Stack overlapping events vertically
            for i, event in enumerate(group_events):
                y_offset = i * 30  # 30px spacing between stacked events
                y_position = self.config.margin_top + (available_height * 0.1) + y_offset

                # Choose color based on categories
                color = self._get_event_color(event)

                # Create tooltip
                tooltip = self._create_tooltip(event)

                # Handle uncertainty visualization
                width = 10
                if (
                    event.time_point
                    and event.time_point.earliest_date
                    and event.time_point.latest_date
                ):
                    # Show uncertainty as width
                    early_year = datetime_to_bce_year(event.time_point.earliest_date)
                    late_year = datetime_to_bce_year(event.time_point.latest_date)
                    uncertainty_span = abs(late_year - early_year)

                    if uncertainty_span > 0:
                        uncertainty_ratio = uncertainty_span / (scale.end_year - scale.start_year)
                        width = max(10, uncertainty_ratio * available_width)

                viz = EventVisualization(
                    event=event,
                    x_position=base_x,
                    y_position=y_position,
                    width=width,
                    height=20,
                    color=color,
                    label=event.name if self.config.show_event_labels else "",
                    tooltip=tooltip,
                )
                event_viz.append(viz)

        return event_viz

    def _group_events_by_time(
        self, events: List[Event], scale: TimelineScale
    ) -> Dict[int, List[Event]]:
        """Group events by approximate time to handle overlaps."""
        groups = {}

        # Determine grouping granularity based on scale
        if scale.unit == "year":
            granularity = 1
        elif scale.unit == "decade":
            granularity = 10
        elif scale.unit == "century":
            granularity = 100
        else:
            granularity = 1000

        for event in events:
            year = self._get_event_year(event)
            if year is not None:
                # Round to granularity
                group_year = (year // granularity) * granularity

                if group_year not in groups:
                    groups[group_year] = []
                groups[group_year].append(event)

        return groups

    def _get_event_year(self, event: Event) -> Optional[int]:
        """Get the primary year for an event."""
        if event.time_point:
            if event.time_point.exact_date:
                return datetime_to_bce_year(event.time_point.exact_date)
            elif event.time_point.earliest_date and event.time_point.latest_date:
                # Use midpoint
                early = datetime_to_bce_year(event.time_point.earliest_date)
                late = datetime_to_bce_year(event.time_point.latest_date)
                return (early + late) // 2
            elif event.time_point.earliest_date:
                return datetime_to_bce_year(event.time_point.earliest_date)
        return None

    def _get_event_color(self, event: Event) -> str:
        """Get color for an event based on its categories."""
        # Check categories for color mapping
        for category in event.categories:
            if category.lower() in self.config.event_colors:
                return self.config.event_colors[category.lower()]

        # Check event type
        event_type_colors = {
            EventType.POINT: "#3498db",
            EventType.PERIOD: "#e74c3c",
            EventType.PROPHETIC: "#9b59b6",
            EventType.SYMBOLIC: "#95a5a6",
        }

        return event_type_colors.get(event.event_type, self.config.event_colors["default"])

    def _create_tooltip(self, event: Event) -> str:
        """Create tooltip text for an event."""
        parts = [f"<strong>{event.name}</strong>"]

        if event.time_point:
            parts.append(f"Date: {event.time_point.get_display_date()}")

        if event.description:
            parts.append(
                f"Description: {event.description[:100]}{'...' if len(event.description) > 100 else ''}"
            )

        if event.participants:
            participants = ", ".join(p.name for p in event.participants[:3])
            if len(event.participants) > 3:
                participants += f" (+{len(event.participants) - 3} more)"
            parts.append(f"Participants: {participants}")

        if event.location:
            parts.append(f"Location: {event.location.name}")

        parts.append(f"Certainty: {event.certainty_level.value.title()}")

        return "<br>".join(parts)

    def _create_legend(self, events: List[Event]) -> Dict[str, Any]:
        """Create legend for the visualization."""
        # Collect unique categories and their colors
        categories = set()
        for event in events:
            categories.update(event.categories)

        legend_items = []
        for category in sorted(categories):
            color = self.config.event_colors.get(
                category.lower(), self.config.event_colors["default"]
            )
            legend_items.append({"label": category.title(), "color": color, "type": "category"})

        # Add certainty levels
        certainty_items = []
        for certainty in CertaintyLevel:
            alpha = {
                CertaintyLevel.CERTAIN: 1.0,
                CertaintyLevel.PROBABLE: 0.8,
                CertaintyLevel.POSSIBLE: 0.6,
                CertaintyLevel.DISPUTED: 0.4,
                CertaintyLevel.LEGENDARY: 0.2,
                CertaintyLevel.SYMBOLIC: 0.1,
            }.get(certainty, 0.5)

            certainty_items.append(
                {"label": certainty.value.title(), "alpha": alpha, "type": "certainty"}
            )

        return {
            "categories": legend_items,
            "certainty_levels": certainty_items,
            "relationship_types": [
                {"label": "Causes", "color": "#e74c3c", "style": "solid"},
                {"label": "Fulfills", "color": "#9b59b6", "style": "dashed"},
                {"label": "Contemporary", "color": "#95a5a6", "style": "dotted"},
            ],
        }

    def export_svg(self, visualization_data: Dict[str, Any]) -> str:
        """Export visualization as SVG."""
        svg_parts = [
            f'<svg width="{self.config.width}" height="{self.config.height}" xmlns="http://www.w3.org/2000/svg">'
        ]

        # Draw timeline axis
        svg_parts.append(self._create_svg_axis(visualization_data["scale"]))

        # Draw relationships first (behind events)
        if self.config.show_relationships:
            for rel in visualization_data.get("relationships", []):
                svg_parts.append(self._create_svg_relationship(rel))

        # Draw events
        for event in visualization_data["events"]:
            svg_parts.append(self._create_svg_event(event))

        # Add legend if configured
        if self.config.include_legend and visualization_data.get("legend"):
            svg_parts.append(self._create_svg_legend(visualization_data["legend"]))

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)

    def _create_svg_axis(self, scale_data: Dict[str, Any]) -> str:
        """Create SVG for timeline axis."""
        axis_y = self.config.height - self.config.margin_bottom + 20
        axis_start_x = self.config.margin_left
        axis_end_x = self.config.width - self.config.margin_right

        svg = f'<line x1="{axis_start_x}" y1="{axis_y}" x2="{axis_end_x}" y2="{axis_y}" stroke="#34495e" stroke-width="2"/>'

        # Add tick marks
        available_width = axis_end_x - axis_start_x
        year_span = scale_data["end_year"] - scale_data["start_year"]

        for year in scale_data["major_ticks"]:
            if scale_data["start_year"] <= year <= scale_data["end_year"]:
                x = axis_start_x + ((year - scale_data["start_year"]) / year_span) * available_width
                svg += f'<line x1="{x}" y1="{axis_y}" x2="{x}" y2="{axis_y + 10}" stroke="#34495e" stroke-width="2"/>'
                svg += f'<text x="{x}" y="{axis_y + 25}" text-anchor="middle" font-size="12" fill="#2c3e50">{abs(year)} {"BCE" if year < 0 else "CE"}</text>'

        return svg

    def _create_svg_event(self, event_data: Dict[str, Any]) -> str:
        """Create SVG for an event."""
        x = event_data["x"]
        y = event_data["y"]
        width = event_data["width"]
        height = event_data["height"]
        color = event_data["color"]

        # Event rectangle
        svg = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}" stroke="#2c3e50" stroke-width="1"'

        # Add tooltip if configured
        if self.config.enable_tooltips and event_data.get("tooltip"):
            svg += f' title="{event_data["tooltip"]}"'

        svg += "/>"

        # Event label
        if self.config.show_event_labels and event_data.get("label"):
            label_x = x + width + 5
            label_y = y + height / 2 + 4
            svg += f'<text x="{label_x}" y="{label_y}" font-size="10" fill="#2c3e50">{event_data["label"]}</text>'

        return svg

    def _create_svg_relationship(self, rel_data: Dict[str, Any]) -> str:
        """Create SVG for a relationship line."""
        style_attr = {
            "solid": "",
            "dashed": "stroke-dasharray='5,5'",
            "dotted": "stroke-dasharray='2,2'",
        }.get(rel_data["style"], "")

        return (
            f'<line x1="{rel_data["source_x"]}" y1="{rel_data["source_y"]}" '
            f'x2="{rel_data["target_x"]}" y2="{rel_data["target_y"]}" '
            f'stroke="{rel_data["color"]}" stroke-width="{rel_data["width"]}" {style_attr}/>'
        )

    def _create_svg_legend(self, legend_data: Dict[str, Any]) -> str:
        """Create SVG for legend."""
        legend_x = self.config.width - 200
        legend_y = 50

        svg = f'<rect x="{legend_x - 10}" y="{legend_y - 10}" width="190" height="200" fill="white" stroke="#bdc3c7" stroke-width="1"/>'
        svg += f'<text x="{legend_x}" y="{legend_y + 10}" font-size="14" font-weight="bold" fill="#2c3e50">Legend</text>'

        y_offset = 30
        for item in legend_data.get("categories", [])[:5]:  # Limit to 5 items
            color = item["color"]
            label = item["label"]
            svg += f'<rect x="{legend_x}" y="{legend_y + y_offset}" width="15" height="15" fill="{color}"/>'
            svg += f'<text x="{legend_x + 20}" y="{legend_y + y_offset + 12}" font-size="10" fill="#2c3e50">{label}</text>'
            y_offset += 20

        return svg
