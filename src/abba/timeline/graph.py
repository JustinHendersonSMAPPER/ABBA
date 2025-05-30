"""
Graph-based temporal event system.

Implements a graph structure for events and their relationships,
enabling complex queries and path-finding through biblical history.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
import heapq
import networkx as nx

from .models import Event, EventRelationship, RelationType, TimePoint, TimeDelta
from .uncertainty import UncertaintyCalculator, DateDistribution


@dataclass
class EventNode:
    """Node in the temporal graph representing an event."""

    event: Event
    incoming_edges: List["RelationshipEdge"] = field(default_factory=list)
    outgoing_edges: List["RelationshipEdge"] = field(default_factory=list)

    # Cached calculations
    _date_distribution: Optional[DateDistribution] = None
    _centrality_score: Optional[float] = None

    def get_date_distribution(self) -> DateDistribution:
        """Get the probability distribution for this event's date."""
        if self._date_distribution is None:
            self._date_distribution = UncertaintyCalculator.from_time_point(self.event.time_point)
        return self._date_distribution

    def get_related_events(
        self, relationship_types: Optional[List[RelationType]] = None
    ) -> List[Event]:
        """Get events related by specific relationship types."""
        related = []

        for edge in self.outgoing_edges:
            if (
                relationship_types is None
                or edge.relationship.relationship_type in relationship_types
            ):
                related.append(edge.target.event)

        for edge in self.incoming_edges:
            if (
                relationship_types is None
                or edge.relationship.relationship_type in relationship_types
            ):
                related.append(edge.source.event)

        return related


@dataclass
class RelationshipEdge:
    """Edge in the temporal graph representing a relationship."""

    source: EventNode
    target: EventNode
    relationship: EventRelationship

    # Cached calculations
    _time_distance_distribution: Optional[DateDistribution] = None

    def get_time_distance(self) -> Optional[DateDistribution]:
        """Get the time distance between events as a distribution."""
        if self._time_distance_distribution is None and self.relationship.time_distance:
            # Create distribution from time distance
            days = self.relationship.time_distance.to_days()
            uncertainty_days = (
                self.relationship.time_distance.uncertainty_years * 365.25
                + self.relationship.time_distance.uncertainty_months * 30.44
                + self.relationship.time_distance.uncertainty_days
            )

            self._time_distance_distribution = DateDistribution(
                distribution_type="normal",
                parameters={
                    "mean": days * 86400,  # Convert to seconds
                    "std": uncertainty_days * 86400,
                },
            )

        return self._time_distance_distribution

    def get_weight(self) -> float:
        """Get edge weight for graph algorithms."""
        # Weight based on confidence and relationship type
        base_weight = 1.0 - self.relationship.confidence

        # Adjust by relationship type importance
        type_weights = {
            RelationType.CAUSES: 0.8,
            RelationType.FULFILLS: 0.9,
            RelationType.BEFORE: 1.0,
            RelationType.AFTER: 1.0,
            RelationType.CONTEMPORARY: 0.5,
            RelationType.RELATED_TO: 1.5,
        }

        type_factor = type_weights.get(self.relationship.relationship_type, 1.0)

        return base_weight * type_factor


class TemporalGraph:
    """Main graph structure for timeline events and relationships."""

    def __init__(self):
        """Initialize the temporal graph."""
        self.nodes: Dict[str, EventNode] = {}
        self.edges: List[RelationshipEdge] = []

        # Indices for efficient lookup
        self._date_index: Dict[int, Set[str]] = defaultdict(set)  # Year -> Event IDs
        self._type_index: Dict[str, Set[str]] = defaultdict(set)  # Event type -> Event IDs
        self._verse_index: Dict[str, Set[str]] = defaultdict(set)  # Verse ref -> Event IDs

        # NetworkX graph for advanced algorithms
        self._nx_graph: Optional[nx.DiGraph] = None
        self._nx_dirty: bool = True

    def add_event(self, event: Event) -> EventNode:
        """Add an event to the graph."""
        if event.id in self.nodes:
            return self.nodes[event.id]

        node = EventNode(event=event)
        self.nodes[event.id] = node

        # Update indices
        self._update_event_indices(event)

        # Mark NetworkX graph as dirty
        self._nx_dirty = True

        return node

    def add_relationship(self, relationship: EventRelationship) -> Optional[RelationshipEdge]:
        """Add a relationship between events."""
        # Ensure both events exist
        if relationship.source_event not in self.nodes:
            return None
        if relationship.target_event not in self.nodes:
            return None

        source_node = self.nodes[relationship.source_event]
        target_node = self.nodes[relationship.target_event]

        # Create edge
        edge = RelationshipEdge(source=source_node, target=target_node, relationship=relationship)

        # Add to graph
        self.edges.append(edge)
        source_node.outgoing_edges.append(edge)
        target_node.incoming_edges.append(edge)

        # Mark NetworkX graph as dirty
        self._nx_dirty = True

        return edge

    def _update_event_indices(self, event: Event):
        """Update indices when adding an event."""
        # Date index
        if event.time_point:
            if event.time_point.exact_date:
                year = event.time_point.exact_date.year
                self._date_index[year].add(event.id)
            elif event.time_point.earliest_date and event.time_point.latest_date:
                # Index all years in range
                start_year = event.time_point.earliest_date.year
                end_year = event.time_point.latest_date.year
                for year in range(start_year, end_year + 1):
                    self._date_index[year].add(event.id)

        # Type index
        self._type_index[event.event_type.value].add(event.id)

        # Verse index
        for verse_ref in event.verse_refs:
            self._verse_index[str(verse_ref)].add(event.id)

    def _get_nx_graph(self) -> nx.DiGraph:
        """Get NetworkX representation of the graph."""
        if self._nx_dirty or self._nx_graph is None:
            self._nx_graph = nx.DiGraph()

            # Add nodes
            for event_id, node in self.nodes.items():
                self._nx_graph.add_node(
                    event_id, event=node.event, date_dist=node.get_date_distribution()
                )

            # Add edges
            for edge in self.edges:
                self._nx_graph.add_edge(
                    edge.source.event.id,
                    edge.target.event.id,
                    relationship=edge.relationship,
                    weight=edge.get_weight(),
                )

            self._nx_dirty = False

        return self._nx_graph

    def find_events_in_range(
        self, start: TimePoint, end: TimePoint, confidence_threshold: float = 0.5
    ) -> List[Event]:
        """Find all events within a time range."""
        # Convert to years for index lookup
        start_year = -2000  # Default ancient start
        end_year = 2100  # Default future end

        if start.exact_date:
            start_year = start.exact_date.year
        elif start.earliest_date:
            start_year = start.earliest_date.year

        if end.exact_date:
            end_year = end.exact_date.year
        elif end.latest_date:
            end_year = end.latest_date.year

        # Collect candidate events from index
        candidate_ids = set()
        for year in range(start_year, end_year + 1):
            candidate_ids.update(self._date_index.get(year, set()))

        # Filter by confidence and precise date checking
        events = []
        start_dist = UncertaintyCalculator.from_time_point(start)
        end_dist = UncertaintyCalculator.from_time_point(end)

        for event_id in candidate_ids:
            event = self.nodes[event_id].event

            # Check confidence
            if event.time_point and event.time_point.confidence >= confidence_threshold:
                # Check if event overlaps with range
                event_dist = self.nodes[event_id].get_date_distribution()

                # Simple overlap check using confidence intervals
                event_ci = event_dist.confidence_interval(0.95)
                range_start = start_dist.mean() - 2 * start_dist.std()
                range_end = end_dist.mean() + 2 * end_dist.std()

                if event_ci[1] >= range_start and event_ci[0] <= range_end:
                    events.append(event)

        # Sort by date
        events.sort(key=lambda e: e.get_date_for_sorting() or 0)

        return events

    def find_contemporary_events(
        self,
        event: Event,
        max_time_distance: Optional[TimeDelta] = None,
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[Event, float]]:
        """Find events happening around the same time."""
        if event.id not in self.nodes:
            return []

        if max_time_distance is None:
            # Default to 50 years
            max_time_distance = TimeDelta(years=50)

        event_node = self.nodes[event.id]
        event_dist = event_node.get_date_distribution()
        max_distance_seconds = max_time_distance.to_days() * 86400

        contemporary = []

        for other_id, other_node in self.nodes.items():
            if other_id == event.id:
                continue

            other_event = other_node.event

            # Check confidence
            if other_event.time_point and other_event.time_point.confidence >= confidence_threshold:

                # Calculate time distance
                other_dist = other_node.get_date_distribution()

                # Calculate probability of being within max_time_distance
                mean_diff = abs(other_dist.mean() - event_dist.mean())

                if mean_diff <= max_distance_seconds:
                    # Calculate overlap score
                    combined_std = (event_dist.std() + other_dist.std()) / 2
                    overlap_score = 1.0 - (mean_diff / max_distance_seconds)

                    # Adjust by confidence
                    overlap_score *= other_event.time_point.confidence

                    contemporary.append((other_event, overlap_score))

        # Sort by overlap score
        contemporary.sort(key=lambda x: x[1], reverse=True)

        return contemporary

    def trace_causal_chain(
        self, start_event: Event, end_event: Event, max_depth: int = 10
    ) -> List[List[Event]]:
        """Find causal paths between events."""
        if start_event.id not in self.nodes or end_event.id not in self.nodes:
            return []

        # Use NetworkX for path finding
        G = self._get_nx_graph()

        # Filter graph to only causal relationships
        causal_edges = [
            (u, v)
            for u, v, data in G.edges(data=True)
            if data["relationship"].relationship_type
            in [
                RelationType.CAUSES,
                RelationType.ENABLES,
                RelationType.PROPHESIES,
                RelationType.FULFILLS,
            ]
        ]

        causal_graph = G.edge_subgraph(causal_edges)

        try:
            # Find all simple paths
            paths = list(
                nx.all_simple_paths(causal_graph, start_event.id, end_event.id, cutoff=max_depth)
            )

            # Convert to event lists
            event_paths = []
            for path in paths:
                event_path = [self.nodes[node_id].event for node_id in path]
                event_paths.append(event_path)

            return event_paths

        except nx.NetworkXNoPath:
            return []


class GraphTraverser:
    """Utilities for traversing the temporal graph."""

    def __init__(self, graph: TemporalGraph):
        """Initialize the traverser."""
        self.graph = graph

    def find_prophetic_fulfillments(self, prophecy: Event) -> List[Tuple[Event, float]]:
        """Find events that potentially fulfill a prophecy."""
        if prophecy.id not in self.graph.nodes:
            return []

        fulfillments = []

        # Direct fulfillment relationships
        prophecy_node = self.graph.nodes[prophecy.id]
        for edge in prophecy_node.outgoing_edges:
            if edge.relationship.relationship_type == RelationType.FULFILLS:
                fulfillment = edge.target.event
                confidence = edge.relationship.confidence
                fulfillments.append((fulfillment, confidence))

        # Look for indirect fulfillments through pattern matching
        # This would involve more sophisticated analysis

        return sorted(fulfillments, key=lambda x: x[1], reverse=True)

    def find_historical_patterns(
        self, pattern_type: str, min_confidence: float = 0.7
    ) -> List[List[Event]]:
        """Find recurring patterns in history."""
        patterns = []

        if pattern_type == "exile_return":
            # Find exile->return patterns
            for node_id, node in self.graph.nodes.items():
                if "exile" in node.event.categories:
                    # Look for related return event
                    for edge in node.outgoing_edges:
                        if (
                            edge.relationship.relationship_type == RelationType.AFTER
                            and "return" in edge.target.event.categories
                        ):
                            pattern = [node.event, edge.target.event]
                            patterns.append(pattern)

        elif pattern_type == "judgment_restoration":
            # Find judgment->restoration patterns
            for node_id, node in self.graph.nodes.items():
                if "judgment" in node.event.categories:
                    # Look for restoration
                    paths = self._find_paths_by_category(node.event, "restoration", max_depth=5)
                    patterns.extend(paths)

        # Add more pattern types as needed

        return patterns

    def _find_paths_by_category(
        self, start_event: Event, end_category: str, max_depth: int = 5
    ) -> List[List[Event]]:
        """Find paths from an event to events with a specific category."""
        paths = []

        # BFS to find all events with the category
        visited = set()
        queue = deque([(start_event.id, [start_event])])

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current_id in visited:
                continue

            visited.add(current_id)
            current_node = self.graph.nodes[current_id]

            # Check if we found a match
            if end_category in current_node.event.categories and len(path) > 1:
                paths.append(path.copy())

            # Explore neighbors
            for edge in current_node.outgoing_edges:
                if edge.target.event.id not in visited:
                    new_path = path + [edge.target.event]
                    queue.append((edge.target.event.id, new_path))

        return paths

    def calculate_event_centrality(self) -> Dict[str, float]:
        """Calculate centrality scores for all events."""
        G = self.graph._get_nx_graph()

        # Use PageRank as centrality measure
        centrality = nx.pagerank(G, weight="weight")

        # Store in nodes
        for event_id, score in centrality.items():
            self.graph.nodes[event_id]._centrality_score = score

        return centrality

    def find_critical_path(self, start_event: Event, end_event: Event) -> Optional[List[Event]]:
        """Find the most likely/important path between events."""
        if start_event.id not in self.graph.nodes or end_event.id not in self.graph.nodes:
            return None

        G = self.graph._get_nx_graph()

        try:
            # Use Dijkstra's algorithm with custom weight
            path = nx.shortest_path(G, start_event.id, end_event.id, weight="weight")

            # Convert to events
            return [self.graph.nodes[node_id].event for node_id in path]

        except nx.NetworkXNoPath:
            return None


class PathFinder:
    """Advanced path-finding algorithms for the temporal graph."""

    def __init__(self, graph: TemporalGraph):
        """Initialize the path finder."""
        self.graph = graph

    def find_all_paths(
        self,
        start: Event,
        end: Event,
        max_length: int = 10,
        required_nodes: Optional[List[str]] = None,
    ) -> List[List[Event]]:
        """Find all paths between events with constraints."""
        G = self.graph._get_nx_graph()

        all_paths = []

        try:
            # Find simple paths
            for path in nx.all_simple_paths(G, start.id, end.id, cutoff=max_length):
                # Check if required nodes are in path
                if required_nodes:
                    if not all(node in path for node in required_nodes):
                        continue

                # Convert to events
                event_path = [self.graph.nodes[node_id].event for node_id in path]
                all_paths.append(event_path)

        except nx.NetworkXNoPath:
            pass

        return all_paths

    def find_shortest_temporal_path(
        self, start: Event, end: Event
    ) -> Optional[Tuple[List[Event], float]]:
        """Find path with minimum time distance."""
        # Custom A* implementation considering temporal distance

        if start.id not in self.graph.nodes or end.id not in self.graph.nodes:
            return None

        # Priority queue: (estimated_cost, current_node, path, actual_cost)
        pq = [(0, start.id, [start], 0)]
        visited = set()

        start_dist = self.graph.nodes[start.id].get_date_distribution()
        end_dist = self.graph.nodes[end.id].get_date_distribution()

        while pq:
            _, current_id, path, cost = heapq.heappop(pq)

            if current_id == end.id:
                return path, cost

            if current_id in visited:
                continue

            visited.add(current_id)
            current_node = self.graph.nodes[current_id]

            # Explore neighbors
            for edge in current_node.outgoing_edges:
                neighbor_id = edge.target.event.id

                if neighbor_id not in visited:
                    # Calculate temporal distance
                    neighbor_dist = edge.target.get_date_distribution()
                    time_cost = abs(
                        neighbor_dist.mean()
                        - self.graph.nodes[current_id].get_date_distribution().mean()
                    )

                    new_cost = cost + time_cost
                    new_path = path + [edge.target.event]

                    # Heuristic: straight-line temporal distance to goal
                    heuristic = abs(neighbor_dist.mean() - end_dist.mean())

                    heapq.heappush(pq, (new_cost + heuristic, neighbor_id, new_path, new_cost))

        return None
