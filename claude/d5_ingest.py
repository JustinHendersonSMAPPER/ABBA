"""One-shot: ingest Easton's + report counts."""

import sqlite3

from abba.database.easton_importer import import_easton_entries

n = import_easton_entries("bible_data/abba.db", "bible_data/sources/Easton.zip")
print("inserted:", n)
c = sqlite3.connect("bible_data/abba.db")
print("dictionary_entries total:", c.execute("SELECT COUNT(*) FROM dictionary_entries").fetchone()[0])
print("with ref_targets:", c.execute("SELECT COUNT(*) FROM dictionary_entries WHERE ref_targets IS NOT NULL").fetchone()[0])
print("tierA provenance:", c.execute("SELECT COUNT(*) FROM provenance WHERE entity_type='dictionary_entry'").fetchone()[0])
for hw, rt in c.execute("SELECT headword, ref_targets FROM dictionary_entries WHERE ref_targets IS NOT NULL LIMIT 3").fetchall():
    print(f"  {hw!r} -> {rt}")
