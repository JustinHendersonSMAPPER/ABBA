"""One-shot: build verse<-dictionary links + report."""

import sqlite3

from abba.database.dictionary_linker import link_dictionary_entries

stats = link_dictionary_entries("bible_data/abba.db")
print("link stats:", stats)
c = sqlite3.connect("bible_data/abba.db")
print("total links:", c.execute("SELECT COUNT(*) FROM verse_dictionary_links").fetchone()[0])
print("distinct verses with context:", c.execute("SELECT COUNT(DISTINCT book_id||'.'||chapter||'.'||verse) FROM verse_dictionary_links").fetchone()[0])
# sample: entries linked to Exodus 6:20 (book 2)
rows = c.execute(
    "SELECT d.headword FROM verse_dictionary_links l JOIN dictionary_entries d ON d.entry_id=l.entry_id "
    "WHERE l.book_id=2 AND l.chapter=6 AND l.verse=20"
).fetchall()
print("Exodus 6:20 entries:", [r[0] for r in rows])
rows2 = c.execute(
    "SELECT d.headword FROM verse_dictionary_links l JOIN dictionary_entries d ON d.entry_id=l.entry_id "
    "WHERE l.book_id=1 AND l.chapter=22 AND l.verse=2"
).fetchall()
print("Genesis 22:2 entries:", [r[0] for r in rows2])
