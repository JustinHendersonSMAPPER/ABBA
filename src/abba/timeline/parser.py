"""
Parsing tools for biblical chronology data.

Handles various date formats, scholarly notations, and relative dating systems
used in biblical chronology.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .models import (
    TimePoint,
    TimeRange,
    TimeDelta,
    Event,
    EventRelationship,
    CalendarSystem,
    CertaintyLevel,
    EventType,
    RelationType,
    Location,
    EntityRef,
)
from ..verse_id import parse_verse_id, VerseID


@dataclass
class ParsedDate:
    """Result of parsing a date string."""

    time_point: Optional[TimePoint] = None
    time_range: Optional[TimeRange] = None
    confidence: float = 1.0
    notes: Optional[str] = None
    calendar_system: CalendarSystem = CalendarSystem.GREGORIAN


class BiblicalDateParser:
    """Parser for biblical date expressions."""

    def __init__(self):
        """Initialize the parser with patterns."""
        self.regnal_years = self._load_regnal_data()
        self.patterns = self._build_patterns()

    def _load_regnal_data(self) -> Dict[str, Tuple[int, int]]:
        """Load regnal year data for biblical rulers."""
        # Simplified data - would be loaded from comprehensive source
        return {
            # Kings of Judah (BCE)
            "david": (1010, 970),
            "solomon": (970, 931),
            "rehoboam": (931, 913),
            "asa": (911, 870),
            "jehoshaphat": (873, 848),
            "hezekiah": (715, 686),
            "josiah": (640, 609),
            # Kings of Israel (BCE)
            "jeroboam": (931, 910),
            "ahab": (874, 853),
            "jehu": (841, 814),
            # Foreign rulers
            "nebuchadnezzar": (605, 562),
            "cyrus": (559, 530),
            "darius": (522, 486),
            "artaxerxes": (465, 424),
            # Roman emperors (CE)
            "augustus": (-27, 14),
            "tiberius": (14, 37),
            "claudius": (41, 54),
            "nero": (54, 68),
        }

    def _build_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Build regex patterns for date parsing."""
        return [
            # Regnal year patterns
            (re.compile(r"(\d+)(?:st|nd|rd|th)?\s+year\s+of\s+(\w+)", re.I), "regnal"),
            (re.compile(r"year\s+(\d+)\s+of\s+(\w+)", re.I), "regnal"),
            # Relative patterns
            (re.compile(r"(\d+)\s+years?\s+(before|after)\s+(.+)", re.I), "relative"),
            (re.compile(r"during\s+the\s+reign\s+of\s+(\w+)", re.I), "reign_period"),
            # Absolute dates
            (re.compile(r"(\d+)\s*BCE?", re.I), "bce"),
            (re.compile(r"(\d+)\s*CE|AD", re.I), "ce"),
            # Ranges
            (re.compile(r"(\d+)\s*-\s*(\d+)\s*BCE?", re.I), "bce_range"),
            (re.compile(r"between\s+(\d+)\s+and\s+(\d+)\s*BCE?", re.I), "bce_range"),
            # Approximate dates
            (re.compile(r"c(?:a|irca)?\.?\s*(\d+)\s*BCE?", re.I), "circa_bce"),
            (re.compile(r"around\s+(\d+)\s*BCE?", re.I), "circa_bce"),
            # Biblical time references
            (re.compile(r"(\d+)(?:st|nd|rd|th)?\s+day\s+of\s+(\w+)\s+month", re.I), "hebrew_date"),
            (re.compile(r"passover\s+of\s+year\s+(\d+)", re.I), "festival_date"),
            # Special periods
            (re.compile(r"(early|middle|late)\s+bronze\s+age", re.I), "archaeological_period"),
            (re.compile(r"(first|second)\s+temple\s+period", re.I), "temple_period"),
        ]

    def parse(self, date_string: str) -> ParsedDate:
        """Parse a biblical date string."""
        date_string = date_string.strip()

        for pattern, pattern_type in self.patterns:
            match = pattern.match(date_string)
            if match:
                return self._parse_by_type(match, pattern_type, date_string)

        # No pattern matched
        return ParsedDate(confidence=0.0, notes=f"Could not parse date: {date_string}")

    def _parse_by_type(self, match: re.Match, pattern_type: str, original: str) -> ParsedDate:
        """Parse based on pattern type."""

        if pattern_type == "regnal":
            return self._parse_regnal_year(match)

        elif pattern_type == "relative":
            return self._parse_relative_date(match)

        elif pattern_type == "reign_period":
            return self._parse_reign_period(match)

        elif pattern_type == "bce":
            year = int(match.group(1))
            return ParsedDate(
                time_point=TimePoint(
                    exact_date=datetime(year=-year, month=1, day=1),
                    calendar_system=CalendarSystem.GREGORIAN,
                    confidence=0.9,
                ),
                confidence=0.9,
            )

        elif pattern_type == "ce":
            year = int(match.group(1))
            return ParsedDate(
                time_point=TimePoint(
                    exact_date=datetime(year=year, month=1, day=1),
                    calendar_system=CalendarSystem.GREGORIAN,
                    confidence=0.9,
                ),
                confidence=0.9,
            )

        elif pattern_type == "bce_range":
            year1 = int(match.group(1))
            year2 = int(match.group(2))
            # Ensure proper order (larger BCE number is earlier)
            if year1 > year2:
                year1, year2 = year2, year1

            return ParsedDate(
                time_point=TimePoint(
                    earliest_date=datetime(year=-year1, month=1, day=1),
                    latest_date=datetime(year=-year2, month=12, day=31),
                    calendar_system=CalendarSystem.GREGORIAN,
                    confidence=0.8,
                ),
                confidence=0.8,
            )

        elif pattern_type == "circa_bce":
            year = int(match.group(1))
            # Add uncertainty of +/- 25 years for "circa"
            return ParsedDate(
                time_point=TimePoint(
                    earliest_date=datetime(year=-(year + 25), month=1, day=1),
                    latest_date=datetime(year=-(year - 25), month=12, day=31),
                    calendar_system=CalendarSystem.GREGORIAN,
                    confidence=0.7,
                    distribution_type="normal",
                    distribution_params={"mean": -year, "std": 10},
                ),
                confidence=0.7,
                notes="Approximate date",
            )

        elif pattern_type == "temple_period":
            period = match.group(1).lower()
            if period == "first":
                # First Temple: ~960-586 BCE
                return ParsedDate(
                    time_range=TimeRange(
                        start=TimePoint(
                            exact_date=datetime(year=-960, month=1, day=1), confidence=0.8
                        ),
                        end=TimePoint(
                            exact_date=datetime(year=-586, month=1, day=1), confidence=0.9
                        ),
                    ),
                    confidence=0.85,
                )
            elif period == "second":
                # Second Temple: 516 BCE - 70 CE
                return ParsedDate(
                    time_range=TimeRange(
                        start=TimePoint(
                            exact_date=datetime(year=-516, month=1, day=1), confidence=0.9
                        ),
                        end=TimePoint(
                            exact_date=datetime(year=70, month=1, day=1), confidence=0.95
                        ),
                    ),
                    confidence=0.9,
                )

        return ParsedDate(confidence=0.5, notes=f"Partial parse of: {original}")

    def _parse_regnal_year(self, match: re.Match) -> ParsedDate:
        """Parse a regnal year reference."""
        year_num = int(match.group(1))
        ruler = match.group(2).lower()

        if ruler in self.regnal_years:
            start_year, end_year = self.regnal_years[ruler]

            # Calculate the actual year
            actual_year = start_year + year_num - 1

            # Check if within reign
            if start_year <= actual_year <= end_year:
                return ParsedDate(
                    time_point=TimePoint(
                        exact_date=datetime(
                            year=-actual_year if actual_year > 0 else actual_year, month=1, day=1
                        ),
                        calendar_system=CalendarSystem.REGNAL,
                        confidence=0.85,
                        chronology_source=f"Regnal year {year_num} of {ruler.title()}",
                    ),
                    confidence=0.85,
                    calendar_system=CalendarSystem.REGNAL,
                )
            else:
                return ParsedDate(
                    confidence=0.2, notes=f"Year {year_num} exceeds reign of {ruler.title()}"
                )

        return ParsedDate(confidence=0.3, notes=f"Unknown ruler: {ruler}")

    def _parse_relative_date(self, match: re.Match) -> ParsedDate:
        """Parse a relative date reference."""
        years = int(match.group(1))
        direction = match.group(2).lower()
        reference = match.group(3)

        # This would need the reference event to calculate
        return ParsedDate(
            time_point=TimePoint(
                relative_to_event=reference,
                relative_offset=TimeDelta(years=years),
                relative_type=direction,
                calendar_system=CalendarSystem.RELATIVE,
                confidence=0.7,
            ),
            confidence=0.7,
            notes=f"Relative to: {reference}",
        )

    def _parse_reign_period(self, match: re.Match) -> ParsedDate:
        """Parse a reign period reference."""
        ruler = match.group(1).lower()

        if ruler in self.regnal_years:
            start_year, end_year = self.regnal_years[ruler]

            return ParsedDate(
                time_range=TimeRange(
                    start=TimePoint(
                        exact_date=datetime(
                            year=-start_year if start_year > 0 else start_year, month=1, day=1
                        ),
                        confidence=0.8,
                    ),
                    end=TimePoint(
                        exact_date=datetime(
                            year=-end_year if end_year > 0 else end_year, month=12, day=31
                        ),
                        confidence=0.8,
                    ),
                ),
                confidence=0.8,
                calendar_system=CalendarSystem.REGNAL,
            )

        return ParsedDate(confidence=0.3, notes=f"Unknown ruler: {ruler}")


class ScholarlyNotationParser:
    """Parser for academic chronology notations."""

    def __init__(self):
        """Initialize the parser."""
        self.uncertainty_markers = {
            "?": 0.5,  # Uncertain
            "??": 0.3,  # Very uncertain
            "ca.": 0.7,  # Circa
            "c.": 0.7,  # Circa
            "~": 0.7,  # Approximately
            "±": 0.8,  # Plus/minus
            ">": 0.6,  # After
            "<": 0.6,  # Before
            "floruit": 0.5,  # Active during
        }

    def parse_scholarly_date(self, notation: str) -> ParsedDate:
        """Parse academic date notation."""
        # Remove extra whitespace
        notation = " ".join(notation.split())

        # Check for uncertainty markers
        confidence = 1.0
        for marker, conf in self.uncertainty_markers.items():
            if marker in notation:
                confidence = min(confidence, conf)
                notation = notation.replace(marker, "").strip()

        # Parse ranges with slash
        if "/" in notation:
            parts = notation.split("/")
            if len(parts) == 2:
                # E.g., "586/587 BCE"
                try:
                    year1 = int(parts[0])
                    year2 = int(parts[1].split()[0])  # Remove BCE/CE

                    return ParsedDate(
                        time_point=TimePoint(
                            earliest_date=datetime(year=-max(year1, year2), month=1, day=1),
                            latest_date=datetime(year=-min(year1, year2), month=12, day=31),
                            confidence=confidence * 0.9,
                        ),
                        confidence=confidence * 0.9,
                    )
                except:
                    pass

        # Parse plus/minus notation
        pm_match = re.search(r"(\d+)\s*±\s*(\d+)", notation)
        if pm_match:
            center = int(pm_match.group(1))
            margin = int(pm_match.group(2))

            is_bce = "BCE" in notation or "BC" in notation

            if is_bce:
                return ParsedDate(
                    time_point=TimePoint(
                        earliest_date=datetime(year=-(center + margin), month=1, day=1),
                        latest_date=datetime(year=-(center - margin), month=12, day=31),
                        distribution_type="normal",
                        distribution_params={"mean": -center, "std": margin / 2},
                        confidence=confidence * 0.85,
                    ),
                    confidence=confidence * 0.85,
                )

        # Try standard biblical parser
        biblical_parser = BiblicalDateParser()
        result = biblical_parser.parse(notation)

        # Adjust confidence based on uncertainty markers
        if result.confidence > 0:
            result.confidence *= confidence

        return result


class ChronologyParser:
    """Main parser for chronological data from various sources."""

    def __init__(self):
        """Initialize the parser."""
        self.biblical_parser = BiblicalDateParser()
        self.scholarly_parser = ScholarlyNotationParser()

    def parse_event_data(self, data: Dict[str, Any]) -> Optional[Event]:
        """Parse event data from dictionary format."""
        # Required fields
        if "id" not in data or "name" not in data:
            return None

        event = Event(id=data["id"], name=data["name"], description=data.get("description", ""))

        # Parse temporal data
        if "date" in data:
            parsed = self.parse_date(data["date"])
            if parsed.time_point:
                event.time_point = parsed.time_point
            elif parsed.time_range:
                event.time_range = parsed.time_range

        elif "date_range" in data:
            start_parsed = self.parse_date(data["date_range"].get("start", ""))
            end_parsed = self.parse_date(data["date_range"].get("end", ""))

            if start_parsed.time_point and end_parsed.time_point:
                event.time_range = TimeRange(
                    start=start_parsed.time_point, end=end_parsed.time_point
                )

        # Event type
        if "event_type" in data:
            try:
                event.event_type = EventType(data["event_type"])
            except ValueError:
                event.event_type = EventType.UNCERTAIN

        # Certainty level
        if "certainty" in data:
            try:
                event.certainty_level = CertaintyLevel(data["certainty"])
            except ValueError:
                pass

        # Biblical references
        if "verses" in data:
            for verse_str in data["verses"]:
                verse_id = parse_verse_id(verse_str)
                if verse_id:
                    event.verse_refs.append(verse_id)

        # Location
        if "location" in data:
            loc_data = data["location"]
            if isinstance(loc_data, str):
                event.location = Location(name=loc_data)
            elif isinstance(loc_data, dict):
                event.location = Location(
                    name=loc_data.get("name", ""),
                    modern_name=loc_data.get("modern_name"),
                    latitude=loc_data.get("lat"),
                    longitude=loc_data.get("lng"),
                    region=loc_data.get("region"),
                )

        # Participants
        if "participants" in data:
            for part_data in data["participants"]:
                if isinstance(part_data, str):
                    event.participants.append(
                        EntityRef(id=part_data, name=part_data, entity_type="person")
                    )
                elif isinstance(part_data, dict):
                    event.participants.append(
                        EntityRef(
                            id=part_data.get("id", ""),
                            name=part_data.get("name", ""),
                            entity_type=part_data.get("type", "person"),
                            role=part_data.get("role"),
                        )
                    )

        # Metadata for filtering
        event.categories = data.get("categories", [])
        event.tags = data.get("tags", [])
        event.scholars = data.get("scholars", [])
        event.sources = data.get("sources", [])
        event.methodologies = data.get("methodologies", [])
        event.traditions = data.get("traditions", [])
        event.scholarly_sources = data.get("scholarly_sources", [])
        event.archaeological_evidence = data.get("archaeological_evidence", [])
        event.notes = data.get("notes")

        return event

    def parse_date(self, date_string: str) -> ParsedDate:
        """Parse a date string using appropriate parser."""
        # Try scholarly notation first (handles more complex formats)
        result = self.scholarly_parser.parse_scholarly_date(date_string)

        # If low confidence, try biblical parser
        if result.confidence < 0.5:
            biblical_result = self.biblical_parser.parse(date_string)
            if biblical_result.confidence > result.confidence:
                result = biblical_result

        return result

    def parse_relationship_data(self, data: Dict[str, Any]) -> Optional[EventRelationship]:
        """Parse relationship data from dictionary format."""
        # Required fields
        if "source" not in data or "target" not in data:
            return None

        # Relationship type
        try:
            rel_type = RelationType(data.get("type", "related_to"))
        except ValueError:
            rel_type = RelationType.RELATED_TO

        rel = EventRelationship(
            source_event=data["source"], target_event=data["target"], relationship_type=rel_type
        )

        # Optional fields
        rel.confidence = data.get("confidence", 1.0)

        # Time distance
        if "time_distance" in data:
            td_data = data["time_distance"]
            if isinstance(td_data, dict):
                rel.time_distance = TimeDelta(
                    years=td_data.get("years", 0),
                    months=td_data.get("months", 0),
                    days=td_data.get("days", 0),
                    uncertainty_years=td_data.get("uncertainty", {}).get("years", 0),
                    uncertainty_months=td_data.get("uncertainty", {}).get("months", 0),
                    uncertainty_days=td_data.get("uncertainty", {}).get("days", 0),
                )

        # Causal strength
        rel.causal_strength = data.get("causal_strength")

        # Biblical support
        if "biblical_support" in data:
            for verse_str in data["biblical_support"]:
                verse_id = parse_verse_id(verse_str)
                if verse_id:
                    rel.biblical_support.append(verse_id)

        # Scholarly support
        rel.scholarly_support = data.get("scholarly_support", [])
        rel.notes = data.get("notes")

        return rel

    def import_timeline_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Import complete timeline data from dictionary format."""
        results = {"events": {}, "relationships": [], "errors": []}

        # Parse events
        if "events" in data:
            for event_data in data["events"]:
                try:
                    event = self.parse_event_data(event_data)
                    if event:
                        results["events"][event.id] = event
                    else:
                        results["errors"].append(
                            f"Failed to parse event: {event_data.get('id', 'unknown')}"
                        )
                except Exception as e:
                    results["errors"].append(f"Error parsing event: {str(e)}")

        # Parse relationships
        if "relationships" in data:
            for rel_data in data["relationships"]:
                try:
                    rel = self.parse_relationship_data(rel_data)
                    if rel:
                        # Verify events exist
                        if (
                            rel.source_event in results["events"]
                            and rel.target_event in results["events"]
                        ):
                            results["relationships"].append(rel)
                        else:
                            results["errors"].append(
                                f"Relationship references unknown events: {rel.source_event} -> {rel.target_event}"
                            )
                    else:
                        results["errors"].append("Failed to parse relationship")
                except Exception as e:
                    results["errors"].append(f"Error parsing relationship: {str(e)}")

        return results
