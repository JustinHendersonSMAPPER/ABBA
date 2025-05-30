"""
Canon registry for managing biblical canon definitions.

Provides a central registry of all supported biblical canons with their
book lists, ordering, and metadata.
"""

import logging
from typing import Dict, List, Optional, Set
from .models import (
    Canon,
    CanonBook,
    CanonTradition,
    BookClassification,
    BookSection,
    VersificationScheme,
)
from ..book_codes import BOOK_INFO


class CanonRegistry:
    """Registry of biblical canon definitions."""

    def __init__(self):
        """Initialize the canon registry."""
        self.logger = logging.getLogger(__name__)
        self._canons: Dict[str, Canon] = {}
        self._versification_schemes: Dict[str, VersificationScheme] = {}

        # Initialize with default canons
        self._initialize_default_canons()
        self._initialize_versification_schemes()

    def _initialize_versification_schemes(self):
        """Initialize default versification schemes."""
        schemes = [
            VersificationScheme(
                id="standard",
                name="Standard",
                description="Modern standard versification (primarily Protestant)",
                base_text="Masoretic/Critical Text",
                includes_apocrypha=False,
            ),
            VersificationScheme(
                id="septuagint",
                name="Septuagint (LXX)",
                description="Greek Septuagint versification",
                base_text="Septuagint",
                includes_apocrypha=True,
                differences={
                    "psalms": "Different numbering after Psalm 9",
                    "3_john": "Different verse divisions",
                },
            ),
            VersificationScheme(
                id="vulgate",
                name="Latin Vulgate",
                description="Jerome's Latin Vulgate versification",
                base_text="Vulgate",
                includes_apocrypha=True,
                differences={
                    "psalms": "Follows LXX numbering",
                    "malachi": "Different chapter divisions",
                },
            ),
            VersificationScheme(
                id="masoretic",
                name="Masoretic Text",
                description="Hebrew Masoretic Text versification",
                base_text="Masoretic Text",
                includes_apocrypha=False,
            ),
        ]

        for scheme in schemes:
            self.register_versification_scheme(scheme)

    def _initialize_default_canons(self):
        """Initialize default canon definitions."""
        # Protestant Canon (66 books)
        protestant_canon = self._create_protestant_canon()
        self.register_canon(protestant_canon)

        # Catholic Canon (73 books)
        catholic_canon = self._create_catholic_canon()
        self.register_canon(catholic_canon)

        # Eastern Orthodox Canon (76+ books)
        orthodox_canon = self._create_eastern_orthodox_canon()
        self.register_canon(orthodox_canon)

        # Ethiopian Orthodox Canon (81 books)
        ethiopian_canon = self._create_ethiopian_orthodox_canon()
        self.register_canon(ethiopian_canon)

    def _create_protestant_canon(self) -> Canon:
        """Create Protestant canon definition (66 books)."""
        canon = Canon(
            id="protestant",
            name="Protestant Canon",
            tradition=CanonTradition.PROTESTANT,
            description="Standard Protestant Bible with 39 OT and 27 NT books",
            book_count=66,
            established_date="16th century",
            authority="Protestant Reformation",
            versification_scheme_id="standard",
            primary_languages=["en", "de", "nl"],
            regions=["North America", "Northern Europe"],
        )

        # Old Testament books (39)
        ot_books = [
            # Pentateuch
            ("GEN", "Genesis", 1),
            ("EXO", "Exodus", 2),
            ("LEV", "Leviticus", 3),
            ("NUM", "Numbers", 4),
            ("DEU", "Deuteronomy", 5),
            # Historical
            ("JOS", "Joshua", 6),
            ("JDG", "Judges", 7),
            ("RUT", "Ruth", 8),
            ("1SA", "1 Samuel", 9),
            ("2SA", "2 Samuel", 10),
            ("1KI", "1 Kings", 11),
            ("2KI", "2 Kings", 12),
            ("1CH", "1 Chronicles", 13),
            ("2CH", "2 Chronicles", 14),
            ("EZR", "Ezra", 15),
            ("NEH", "Nehemiah", 16),
            ("EST", "Esther", 17),
            # Wisdom/Poetry
            ("JOB", "Job", 18),
            ("PSA", "Psalms", 19),
            ("PRO", "Proverbs", 20),
            ("ECC", "Ecclesiastes", 21),
            ("SNG", "Song of Songs", 22),
            # Major Prophets
            ("ISA", "Isaiah", 23),
            ("JER", "Jeremiah", 24),
            ("LAM", "Lamentations", 25),
            ("EZK", "Ezekiel", 26),
            ("DAN", "Daniel", 27),
            # Minor Prophets
            ("HOS", "Hosea", 28),
            ("JOL", "Joel", 29),
            ("AMO", "Amos", 30),
            ("OBA", "Obadiah", 31),
            ("JON", "Jonah", 32),
            ("MIC", "Micah", 33),
            ("NAM", "Nahum", 34),
            ("HAB", "Habakkuk", 35),
            ("ZEP", "Zephaniah", 36),
            ("HAG", "Haggai", 37),
            ("ZEC", "Zechariah", 38),
            ("MAL", "Malachi", 39),
        ]

        # New Testament books (27)
        nt_books = [
            # Gospels
            ("MAT", "Matthew", 40),
            ("MRK", "Mark", 41),
            ("LUK", "Luke", 42),
            ("JHN", "John", 43),
            # History
            ("ACT", "Acts", 44),
            # Pauline Epistles
            ("ROM", "Romans", 45),
            ("1CO", "1 Corinthians", 46),
            ("2CO", "2 Corinthians", 47),
            ("GAL", "Galatians", 48),
            ("EPH", "Ephesians", 49),
            ("PHP", "Philippians", 50),
            ("COL", "Colossians", 51),
            ("1TH", "1 Thessalonians", 52),
            ("2TH", "2 Thessalonians", 53),
            ("1TI", "1 Timothy", 54),
            ("2TI", "2 Timothy", 55),
            ("TIT", "Titus", 56),
            ("PHM", "Philemon", 57),
            # General Epistles
            ("HEB", "Hebrews", 58),
            ("JAS", "James", 59),
            ("1PE", "1 Peter", 60),
            ("2PE", "2 Peter", 61),
            ("1JN", "1 John", 62),
            ("2JN", "2 John", 63),
            ("3JN", "3 John", 64),
            ("JUD", "Jude", 65),
            # Prophecy
            ("REV", "Revelation", 66),
        ]

        # Add OT books
        for book_id, name, order in ot_books:
            canon.books.append(
                CanonBook(
                    canon_id=canon.id,
                    book_id=book_id,
                    order=order,
                    section=BookSection.OLD_TESTAMENT,
                    classification=BookClassification.PROTOCANONICAL,
                    canonical_name=name,
                )
            )

        # Add NT books
        for book_id, name, order in nt_books:
            canon.books.append(
                CanonBook(
                    canon_id=canon.id,
                    book_id=book_id,
                    order=order,
                    section=BookSection.NEW_TESTAMENT,
                    classification=BookClassification.PROTOCANONICAL,
                    canonical_name=name,
                )
            )

        return canon

    def _create_catholic_canon(self) -> Canon:
        """Create Catholic canon definition (73 books)."""
        canon = Canon(
            id="catholic",
            name="Catholic Canon",
            tradition=CanonTradition.CATHOLIC,
            description="Roman Catholic Bible with 46 OT and 27 NT books",
            book_count=73,
            established_date="Council of Trent (1546)",
            authority="Roman Catholic Church",
            versification_scheme_id="vulgate",
            primary_languages=["la", "es", "it", "fr", "pt"],
            regions=["Southern Europe", "Latin America"],
        )

        # Start with Protestant canon
        protestant = self._create_protestant_canon()

        # Copy Protestant books but adjust ordering
        order = 1
        for pb in sorted(protestant.books, key=lambda x: x.order):
            if pb.book_id == "EST":  # Esther comes after deuterocanonical additions
                continue

            canon.books.append(
                CanonBook(
                    canon_id=canon.id,
                    book_id=pb.book_id,
                    order=order,
                    section=pb.section,
                    classification=pb.classification,
                    canonical_name=pb.canonical_name,
                )
            )
            order += 1

            # Insert deuterocanonical books in their positions
            if pb.book_id == "NEH":  # After Nehemiah
                # Add Tobit and Judith
                canon.books.extend(
                    [
                        CanonBook(
                            canon_id=canon.id,
                            book_id="TOB",
                            order=order,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="Tobit",
                        ),
                        CanonBook(
                            canon_id=canon.id,
                            book_id="JDT",
                            order=order + 1,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="Judith",
                        ),
                    ]
                )
                order += 2

                # Add Esther (with Greek additions)
                canon.books.append(
                    CanonBook(
                        canon_id=canon.id,
                        book_id="EST",
                        order=order,
                        section=BookSection.OLD_TESTAMENT,
                        classification=BookClassification.PROTOCANONICAL,
                        canonical_name="Esther",
                        notes="Includes Greek additions",
                    )
                )
                order += 1

                # Add 1-2 Maccabees
                canon.books.extend(
                    [
                        CanonBook(
                            canon_id=canon.id,
                            book_id="1MA",
                            order=order,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="1 Maccabees",
                        ),
                        CanonBook(
                            canon_id=canon.id,
                            book_id="2MA",
                            order=order + 1,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="2 Maccabees",
                        ),
                    ]
                )
                order += 2

            elif pb.book_id == "SNG":  # After Song of Songs
                # Add Wisdom and Sirach
                canon.books.extend(
                    [
                        CanonBook(
                            canon_id=canon.id,
                            book_id="WIS",
                            order=order,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="Wisdom",
                        ),
                        CanonBook(
                            canon_id=canon.id,
                            book_id="SIR",
                            order=order + 1,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="Sirach",
                            alternate_names=["Ecclesiasticus"],
                        ),
                    ]
                )
                order += 2

            elif pb.book_id == "JER":  # After Jeremiah
                # Add Baruch
                order += 1  # Skip for Baruch
                canon.books.append(
                    CanonBook(
                        canon_id=canon.id,
                        book_id="BAR",
                        order=order,
                        section=BookSection.OLD_TESTAMENT,
                        classification=BookClassification.DEUTEROCANONICAL,
                        canonical_name="Baruch",
                        notes="Includes Letter of Jeremiah as chapter 6",
                    )
                )
                order += 1

        # Adjust NT book ordering to continue from OT
        for book in canon.books:
            if book.section == BookSection.NEW_TESTAMENT:
                book.order = order
                order += 1

        # Note: Daniel includes Greek additions (Susanna, Bel and the Dragon)
        for book in canon.books:
            if book.book_id == "DAN":
                book.notes = "Includes Greek additions (chapters 13-14)"
                break

        return canon

    def _create_eastern_orthodox_canon(self) -> Canon:
        """Create Eastern Orthodox canon definition (76+ books)."""
        canon = Canon(
            id="eastern_orthodox",
            name="Eastern Orthodox Canon",
            tradition=CanonTradition.EASTERN_ORTHODOX,
            description="Eastern Orthodox Bible with expanded Old Testament",
            book_count=76,
            established_date="Synod of Jerusalem (1672)",
            authority="Eastern Orthodox Churches",
            versification_scheme_id="septuagint",
            primary_languages=["el", "ru", "ro", "bg", "sr"],
            regions=["Eastern Europe", "Greece", "Middle East"],
        )

        # Start with Catholic canon
        catholic = self._create_catholic_canon()

        # Copy all Catholic books
        order = 1
        for cb in sorted(catholic.books, key=lambda x: x.order):
            canon.books.append(
                CanonBook(
                    canon_id=canon.id,
                    book_id=cb.book_id,
                    order=order,
                    section=cb.section,
                    classification=cb.classification,
                    canonical_name=cb.canonical_name,
                )
            )
            order += 1

            # Add additional Orthodox books
            if cb.book_id == "2MA":  # After 2 Maccabees
                # Add 3 Maccabees and 1 Esdras
                canon.books.extend(
                    [
                        CanonBook(
                            canon_id=canon.id,
                            book_id="3MA",
                            order=order,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="3 Maccabees",
                        ),
                        CanonBook(
                            canon_id=canon.id,
                            book_id="1ES",
                            order=order + 1,
                            section=BookSection.OLD_TESTAMENT,
                            classification=BookClassification.DEUTEROCANONICAL,
                            canonical_name="1 Esdras",
                            alternate_names=["3 Ezra"],
                        ),
                    ]
                )
                order += 2

            elif cb.book_id == "PSA":  # After Psalms
                # Add Prayer of Manasseh
                canon.books.append(
                    CanonBook(
                        canon_id=canon.id,
                        book_id="MAN",
                        order=order,
                        section=BookSection.OLD_TESTAMENT,
                        classification=BookClassification.DEUTEROCANONICAL,
                        canonical_name="Prayer of Manasseh",
                    )
                )
                order += 1

        # Add Psalm 151 (as appendix to Psalms)
        for book in canon.books:
            if book.book_id == "PSA":
                book.notes = "Includes Psalm 151"
                break

        # Note: Some Orthodox churches also include 4 Maccabees, 2 Esdras
        # These vary by jurisdiction

        return canon

    def _create_ethiopian_orthodox_canon(self) -> Canon:
        """Create Ethiopian Orthodox canon definition (81 books)."""
        canon = Canon(
            id="ethiopian_orthodox",
            name="Ethiopian Orthodox Canon",
            tradition=CanonTradition.ETHIOPIAN_ORTHODOX,
            description="Ethiopian Orthodox Tewahedo Bible - the largest biblical canon",
            book_count=81,
            established_date="4th century",
            authority="Ethiopian Orthodox Tewahedo Church",
            versification_scheme_id="standard",  # Uses unique versification
            primary_languages=["am", "gez", "ti"],
            regions=["Ethiopia", "Eritrea"],
        )

        # Start with Eastern Orthodox canon
        orthodox = self._create_eastern_orthodox_canon()

        # Copy Orthodox books
        order = 1
        for ob in sorted(orthodox.books, key=lambda x: x.order):
            canon.books.append(
                CanonBook(
                    canon_id=canon.id,
                    book_id=ob.book_id,
                    order=order,
                    section=ob.section,
                    classification=ob.classification,
                    canonical_name=ob.canonical_name,
                )
            )
            order += 1

            # Add Ethiopian-specific books
            if ob.book_id == "MAN":  # After Prayer of Manasseh
                ethiopian_books = [
                    ("ENO", "1 Enoch", "Book of Enoch"),
                    ("JUB", "Jubilees", "Book of Jubilees"),
                    ("4BA", "4 Baruch", "Paralipomena of Jeremiah"),
                    ("1MQ", "1 Meqabyan", None),
                    ("2MQ", "2 Meqabyan", None),
                    ("3MQ", "3 Meqabyan", None),
                ]

                for book_id, name, alt_name in ethiopian_books:
                    book = CanonBook(
                        canon_id=canon.id,
                        book_id=book_id,
                        order=order,
                        section=BookSection.OLD_TESTAMENT,
                        classification=BookClassification.DEUTEROCANONICAL,
                        canonical_name=name,
                    )
                    if alt_name:
                        book.alternate_names = [alt_name]
                    canon.books.append(book)
                    order += 1

        # Ethiopian NT includes additional books
        nt_additions = [
            ("SOS", "Sinodos", "Book of the Covenant"),
            ("CLE", "Clement", "Qalementos"),
            ("DID", "Didascalia", "Apostolic Church Order"),
        ]

        for book_id, name, alt_name in nt_additions:
            canon.books.append(
                CanonBook(
                    canon_id=canon.id,
                    book_id=book_id,
                    order=order,
                    section=BookSection.NEW_TESTAMENT,
                    classification=BookClassification.DEUTEROCANONICAL,
                    canonical_name=name,
                    alternate_names=[alt_name] if alt_name else [],
                )
            )
            order += 1

        return canon

    def register_canon(self, canon: Canon) -> None:
        """Register a canon in the registry."""
        self._canons[canon.id] = canon
        self.logger.info(f"Registered canon: {canon.name} ({canon.book_count} books)")

    def register_versification_scheme(self, scheme: VersificationScheme) -> None:
        """Register a versification scheme."""
        self._versification_schemes[scheme.id] = scheme
        self.logger.info(f"Registered versification scheme: {scheme.name}")

    def get_canon(self, canon_id: str) -> Optional[Canon]:
        """Get a canon by ID."""
        return self._canons.get(canon_id)

    def get_versification_scheme(self, scheme_id: str) -> Optional[VersificationScheme]:
        """Get a versification scheme by ID."""
        return self._versification_schemes.get(scheme_id)

    def list_canons(self) -> List[str]:
        """List all registered canon IDs."""
        return list(self._canons.keys())

    def list_versification_schemes(self) -> List[str]:
        """List all registered versification scheme IDs."""
        return list(self._versification_schemes.keys())

    def get_canons_by_tradition(self, tradition: CanonTradition) -> List[Canon]:
        """Get all canons of a specific tradition."""
        return [canon for canon in self._canons.values() if canon.tradition == tradition]

    def get_book_canons(self, book_id: str) -> List[Canon]:
        """Get all canons that include a specific book."""
        return [canon for canon in self._canons.values() if canon.has_book(book_id)]

    def get_common_books(self, canon_ids: List[str]) -> Set[str]:
        """Get books common to all specified canons."""
        if not canon_ids:
            return set()

        canons = [self.get_canon(cid) for cid in canon_ids if self.get_canon(cid)]
        if not canons:
            return set()

        common_books = set(canons[0].get_book_ids())
        for canon in canons[1:]:
            common_books &= set(canon.get_book_ids())

        return common_books

    def get_unique_books(self, canon_id: str, compared_to: List[str]) -> Set[str]:
        """Get books unique to a canon compared to others."""
        canon = self.get_canon(canon_id)
        if not canon:
            return set()

        canon_books = set(canon.get_book_ids())

        for other_id in compared_to:
            other_canon = self.get_canon(other_id)
            if other_canon:
                canon_books -= set(other_canon.get_book_ids())

        return canon_books


# Global registry instance
canon_registry = CanonRegistry()
