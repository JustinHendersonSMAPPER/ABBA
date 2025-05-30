"""
Lexicon integration for biblical language analysis.

This module integrates Strong's lexicon data and provides semantic domain
classification, lemma search, and frequency analysis tools.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..morphology import Language
from ..parsers.lexicon_parser import LexiconEntry, StrongsLexicon


class SemanticDomain(Enum):
    """Major semantic domains for biblical vocabulary."""

    # Religious/Theological
    GOD_DIVINE = "god_divine"
    WORSHIP_RITUAL = "worship_ritual"
    SIN_EVIL = "sin_evil"
    SALVATION_REDEMPTION = "salvation_redemption"
    COVENANT_LAW = "covenant_law"
    PRAYER_BLESSING = "prayer_blessing"

    # Human/Social
    FAMILY_KINSHIP = "family_kinship"
    AUTHORITY_LEADERSHIP = "authority_leadership"
    COMMUNITY_SOCIETY = "community_society"
    EMOTION_ATTITUDE = "emotion_attitude"
    BODY_HEALTH = "body_health"

    # Natural World
    CREATION_NATURE = "creation_nature"
    ANIMALS_CREATURES = "animals_creatures"
    PLANTS_AGRICULTURE = "plants_agriculture"
    TIME_SEASON = "time_season"
    GEOGRAPHY_PLACE = "geography_place"

    # Actions/States
    MOVEMENT_TRAVEL = "movement_travel"
    COMMUNICATION_SPEECH = "communication_speech"
    PERCEPTION_KNOWLEDGE = "perception_knowledge"
    CONFLICT_WAR = "conflict_war"
    WORK_OCCUPATION = "work_occupation"

    # Objects/Materials
    BUILDING_DWELLING = "building_dwelling"
    CLOTHING_ADORNMENT = "clothing_adornment"
    FOOD_DRINK = "food_drink"
    TOOLS_IMPLEMENTS = "tools_implements"
    MONEY_COMMERCE = "money_commerce"


@dataclass
class LexicalEntry:
    """Enhanced lexical entry with semantic and frequency data."""

    strong_number: str
    lemma: str
    gloss: str
    definition: str
    language: Language

    # Semantic classification
    semantic_domains: List[SemanticDomain] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)

    # Usage data
    frequency: int = 0
    book_distribution: Dict[str, int] = field(default_factory=dict)

    # Related entries
    synonyms: List[str] = field(default_factory=list)  # Strong's numbers
    antonyms: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)

    # Etymology
    root: Optional[str] = None
    derivatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format."""
        return {
            "strong_number": self.strong_number,
            "lemma": self.lemma,
            "gloss": self.gloss,
            "definition": self.definition,
            "language": self.language.value,
            "semantic_domains": [d.value for d in self.semantic_domains],
            "semantic_tags": self.semantic_tags,
            "frequency": self.frequency,
            "book_distribution": self.book_distribution,
            "synonyms": self.synonyms,
            "antonyms": self.antonyms,
            "related": self.related,
            "root": self.root,
            "derivatives": self.derivatives,
        }


class LexiconIntegrator:
    """Integrate and enhance lexicon data with semantic analysis."""

    def __init__(self, strongs_lexicon: Optional[StrongsLexicon] = None) -> None:
        """
        Initialize the lexicon integrator.

        Args:
            strongs_lexicon: Pre-loaded Strong's lexicon data
        """
        self.lexicon = strongs_lexicon
        self.entries: Dict[str, LexicalEntry] = {}
        self._semantic_keywords = self._build_semantic_keywords()
        self._lemma_index: Dict[str, List[str]] = {}  # lemma -> [strong_numbers]
        self._frequency_data: Dict[str, int] = {}

        # Initialize entries if lexicon provided
        if self.lexicon:
            self._initialize_entries()

    def _initialize_entries(self) -> None:
        """Initialize lexical entries from Strong's lexicon."""
        if not self.lexicon:
            return

        # Process Hebrew entries
        for strong_num, entry in self.lexicon.hebrew_entries.items():
            lexical_entry = self._create_lexical_entry(entry, Language.HEBREW)
            self.entries[strong_num] = lexical_entry

            # Update lemma index
            if lexical_entry.lemma:
                if lexical_entry.lemma not in self._lemma_index:
                    self._lemma_index[lexical_entry.lemma] = []
                self._lemma_index[lexical_entry.lemma].append(strong_num)

        # Process Greek entries
        for strong_num, entry in self.lexicon.greek_entries.items():
            lexical_entry = self._create_lexical_entry(entry, Language.GREEK)
            self.entries[strong_num] = lexical_entry

            # Update lemma index
            if lexical_entry.lemma:
                if lexical_entry.lemma not in self._lemma_index:
                    self._lemma_index[lexical_entry.lemma] = []
                self._lemma_index[lexical_entry.lemma].append(strong_num)

    def _create_lexical_entry(
        self, lexicon_entry: LexiconEntry, language: Language
    ) -> LexicalEntry:
        """Create enhanced lexical entry from basic lexicon entry."""
        entry = LexicalEntry(
            strong_number=lexicon_entry.strong_number,
            lemma=lexicon_entry.word,
            gloss=lexicon_entry.gloss or "",
            definition=lexicon_entry.definition or "",
            language=language,
        )

        # Classify semantic domains
        entry.semantic_domains = self._classify_semantic_domains(
            lexicon_entry.definition or "", lexicon_entry.gloss or ""
        )

        # Extract root information
        if lexicon_entry.etymology:
            entry.root = self._extract_root(lexicon_entry.etymology)

        return entry

    def _build_semantic_keywords(self) -> Dict[SemanticDomain, List[str]]:
        """Build keyword mappings for semantic domain classification."""
        return {
            SemanticDomain.GOD_DIVINE: [
                "god",
                "lord",
                "divine",
                "holy",
                "sacred",
                "deity",
                "yahweh",
                "elohim",
            ],
            SemanticDomain.WORSHIP_RITUAL: [
                "worship",
                "praise",
                "sacrifice",
                "offering",
                "temple",
                "altar",
                "priest",
            ],
            SemanticDomain.SIN_EVIL: [
                "sin",
                "evil",
                "wicked",
                "transgress",
                "iniquity",
                "guilt",
                "rebel",
            ],
            SemanticDomain.SALVATION_REDEMPTION: [
                "save",
                "redeem",
                "deliver",
                "rescue",
                "forgive",
                "mercy",
                "grace",
            ],
            SemanticDomain.COVENANT_LAW: [
                "covenant",
                "law",
                "commandment",
                "statute",
                "ordinance",
                "decree",
            ],
            SemanticDomain.FAMILY_KINSHIP: [
                "father",
                "mother",
                "son",
                "daughter",
                "brother",
                "sister",
                "family",
            ],
            SemanticDomain.AUTHORITY_LEADERSHIP: [
                "king",
                "ruler",
                "judge",
                "govern",
                "authority",
                "power",
                "throne",
            ],
            SemanticDomain.EMOTION_ATTITUDE: [
                "love",
                "hate",
                "fear",
                "joy",
                "anger",
                "compassion",
                "jealous",
            ],
            SemanticDomain.CREATION_NATURE: [
                "create",
                "earth",
                "heaven",
                "sky",
                "sea",
                "mountain",
                "nature",
            ],
            SemanticDomain.MOVEMENT_TRAVEL: [
                "go",
                "come",
                "walk",
                "run",
                "journey",
                "travel",
                "depart",
            ],
            SemanticDomain.COMMUNICATION_SPEECH: [
                "speak",
                "say",
                "word",
                "voice",
                "call",
                "answer",
                "command",
            ],
            SemanticDomain.CONFLICT_WAR: [
                "war",
                "battle",
                "fight",
                "enemy",
                "sword",
                "weapon",
                "conquer",
            ],
        }

    def _classify_semantic_domains(self, definition: str, gloss: str) -> List[SemanticDomain]:
        """Classify text into semantic domains based on keywords."""
        domains = []
        text = (definition + " " + gloss).lower()

        for domain, keywords in self._semantic_keywords.items():
            if any(keyword in text for keyword in keywords):
                domains.append(domain)

        return domains[:3]  # Limit to top 3 domains

    def _extract_root(self, etymology: str) -> Optional[str]:
        """Extract root form from etymology string."""
        if not etymology:
            return None

        # Simple extraction - look for patterns like "from H1234"
        import re

        match = re.search(r"from ([HG]\d+)", etymology)
        if match:
            return match.group(1)

        # Look for root letters in parentheses
        match = re.search(r"\(([^)]+)\)", etymology)
        if match:
            return match.group(1)

        return None

    def get_entry(self, strong_number: str) -> Optional[LexicalEntry]:
        """Get lexical entry by Strong's number."""
        return self.entries.get(strong_number)

    def search_by_lemma(self, lemma: str, exact: bool = True) -> List[LexicalEntry]:
        """
        Search for entries by lemma.

        Args:
            lemma: Lemma to search for
            exact: If True, exact match; if False, substring match

        Returns:
            List of matching entries
        """
        results = []

        if exact:
            strong_numbers = self._lemma_index.get(lemma, [])
            for strong_num in strong_numbers:
                if strong_num in self.entries:
                    results.append(self.entries[strong_num])
        else:
            # Substring search
            for lem, strong_numbers in self._lemma_index.items():
                if lemma.lower() in lem.lower():
                    for strong_num in strong_numbers:
                        if strong_num in self.entries:
                            results.append(self.entries[strong_num])

        return results

    def search_by_gloss(self, keyword: str) -> List[LexicalEntry]:
        """Search for entries by gloss keyword."""
        results = []
        keyword_lower = keyword.lower()

        for entry in self.entries.values():
            if keyword_lower in entry.gloss.lower():
                results.append(entry)

        return sorted(results, key=lambda e: e.frequency, reverse=True)

    def search_by_semantic_domain(self, domain: SemanticDomain) -> List[LexicalEntry]:
        """Get all entries in a semantic domain."""
        results = []

        for entry in self.entries.values():
            if domain in entry.semantic_domains:
                results.append(entry)

        return sorted(results, key=lambda e: e.frequency, reverse=True)

    def get_synonyms(self, strong_number: str) -> List[LexicalEntry]:
        """Get synonym entries for a Strong's number."""
        entry = self.get_entry(strong_number)
        if not entry:
            return []

        synonyms = []
        for syn_num in entry.synonyms:
            syn_entry = self.get_entry(syn_num)
            if syn_entry:
                synonyms.append(syn_entry)

        return synonyms

    def get_related_words(self, strong_number: str) -> Dict[str, List[LexicalEntry]]:
        """Get all related words (synonyms, antonyms, derivatives, etc.)."""
        entry = self.get_entry(strong_number)
        if not entry:
            return {}

        related = {"synonyms": [], "antonyms": [], "derivatives": [], "root": None, "same_root": []}

        # Get synonyms
        for num in entry.synonyms:
            rel_entry = self.get_entry(num)
            if rel_entry:
                related["synonyms"].append(rel_entry)

        # Get antonyms
        for num in entry.antonyms:
            rel_entry = self.get_entry(num)
            if rel_entry:
                related["antonyms"].append(rel_entry)

        # Get derivatives
        for num in entry.derivatives:
            rel_entry = self.get_entry(num)
            if rel_entry:
                related["derivatives"].append(rel_entry)

        # Get root
        if entry.root:
            root_entry = self.get_entry(entry.root)
            if root_entry:
                related["root"] = root_entry

        # Find words with same root
        if entry.root:
            for other_entry in self.entries.values():
                if other_entry.root == entry.root and other_entry.strong_number != strong_number:
                    related["same_root"].append(other_entry)

        return related

    def update_frequency_data(self, word_occurrences: List[Tuple[str, str, int]]) -> None:
        """
        Update frequency data from word occurrences.

        Args:
            word_occurrences: List of (strong_number, book_code, count) tuples
        """
        for strong_num, book_code, count in word_occurrences:
            if strong_num not in self.entries:
                continue

            entry = self.entries[strong_num]
            entry.frequency += count

            if book_code not in entry.book_distribution:
                entry.book_distribution[book_code] = 0
            entry.book_distribution[book_code] += count

            # Update global frequency
            if strong_num not in self._frequency_data:
                self._frequency_data[strong_num] = 0
            self._frequency_data[strong_num] += count

    def get_frequency_analysis(
        self, language: Optional[Language] = None, min_frequency: int = 1
    ) -> Dict[str, any]:
        """
        Get frequency analysis statistics.

        Args:
            language: Filter by language (None for all)
            min_frequency: Minimum frequency threshold

        Returns:
            Dictionary with frequency statistics
        """
        # Filter entries
        entries = list(self.entries.values())
        if language:
            entries = [e for e in entries if e.language == language]

        entries = [e for e in entries if e.frequency >= min_frequency]

        # Calculate statistics
        total_entries = len(entries)
        total_occurrences = sum(e.frequency for e in entries)

        # Top words
        top_words = sorted(entries, key=lambda e: e.frequency, reverse=True)[:20]

        # Distribution by semantic domain
        domain_stats = {}
        for domain in SemanticDomain:
            domain_entries = [e for e in entries if domain in e.semantic_domains]
            domain_stats[domain.value] = {
                "count": len(domain_entries),
                "total_frequency": sum(e.frequency for e in domain_entries),
            }

        # Hapax legomena (words occurring only once)
        hapax = [e for e in entries if e.frequency == 1]

        return {
            "total_entries": total_entries,
            "total_occurrences": total_occurrences,
            "average_frequency": total_occurrences / total_entries if total_entries > 0 else 0,
            "top_words": [
                {
                    "strong_number": e.strong_number,
                    "lemma": e.lemma,
                    "gloss": e.gloss,
                    "frequency": e.frequency,
                }
                for e in top_words
            ],
            "semantic_domain_distribution": domain_stats,
            "hapax_legomena_count": len(hapax),
            "hapax_percentage": len(hapax) / total_entries if total_entries > 0 else 0,
        }

    def build_semantic_field(self, strong_number: str, depth: int = 2) -> Dict[str, any]:
        """
        Build semantic field network around a word.

        Args:
            strong_number: Center word's Strong's number
            depth: How many levels of relationships to include

        Returns:
            Dictionary representing the semantic field
        """
        center_entry = self.get_entry(strong_number)
        if not center_entry:
            return {}

        visited = {strong_number}
        field = {"center": center_entry.to_dict(), "relationships": {}}

        # Build network
        queue = [(strong_number, 0)]

        while queue and depth > 0:
            current_num, current_depth = queue.pop(0)

            if current_depth >= depth:
                continue

            current_entry = self.get_entry(current_num)
            if not current_entry:
                continue

            # Get all related words
            related = self.get_related_words(current_num)

            for rel_type, rel_entries in related.items():
                if rel_type == "root" and rel_entries:
                    # Special handling for root
                    rel_entries = [rel_entries]

                for rel_entry in rel_entries:
                    if rel_entry.strong_number not in visited:
                        visited.add(rel_entry.strong_number)

                        # Add to field
                        rel_key = f"{current_num}_{rel_type}_{rel_entry.strong_number}"
                        field["relationships"][rel_key] = {
                            "from": current_num,
                            "to": rel_entry.strong_number,
                            "type": rel_type,
                            "entry": rel_entry.to_dict(),
                        }

                        # Add to queue for next level
                        if current_depth + 1 < depth:
                            queue.append((rel_entry.strong_number, current_depth + 1))

        return field
