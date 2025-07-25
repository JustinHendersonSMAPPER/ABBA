# Import Tracking and Transaction Support

The ABBA project now includes:
1. **Import tracking** - Prevents re-processing of already imported data
2. **Transaction support** - Ensures atomic imports (all-or-nothing)

## How It Works

### Import Tracking

When you run `python abba/main.py`, the system:
- Checks `bible_data/.import_status.json` to see what's already imported
- Skips translations and STEPBible files that are already processed
- Only imports new or missing data

**Example tracking file:**
```json
{
  "schema_version": "1.0",
  "created_at": "2024-01-15T10:30:00",
  "imports": {
    "translations": {
      "KJV": "2024-01-15T10:31:00",
      "ESV": "2024-01-15T10:32:00",
      "NIV": "2024-01-15T10:33:00"
    },
    "stepbible": {
      "tahot": {
        "tahot_gen_deu.txt": "2024-01-15T10:34:00",
        "tahot_jos_est.txt": "2024-01-15T10:35:00"
      },
      "lexicon": {
        "tbesh.txt": "2024-01-15T10:36:00",
        "tbesg.txt": "2024-01-15T10:37:00"
      }
    }
  }
}
```

### Transaction Support

All imports now use database transactions:
- If any part of a translation import fails, the entire import is rolled back
- No partial data is left in the database
- Ensures data integrity

## Usage Examples

### First Run
```bash
$ python abba/main.py
Importing biblical data into ABBA database...
Found 50 translations in bible.db
Need to import 50 translations
Importing verses for KJV...
Importing verses for ESV...
...
Successfully imported 50/50 translations
Database now contains 31,102 verses
Importing STEPBible lexicon and morphology data...
STEPBible import complete:
  Words: 450,843
  Lexicon entries: 8,674
  Morphology codes: 1,234

Import summary:
  Translations: 50
  STEPBible files: 10
  Last update: 2024-01-15T10:45:00
```

### Subsequent Runs (No Re-processing)
```bash
$ python abba/main.py
Importing biblical data into ABBA database...
Found 50 translations in bible.db
All requested translations are already imported.
Database now contains 31,102 verses
STEPBible data already imported - skipping

Import summary:
  Translations: 50
  STEPBible files: 10
  Last update: 2024-01-15T10:45:00
```

### Force Re-import
```bash
$ python abba/main.py --rebuild-db
Force rebuild requested - will re-import all data
Importing biblical data into ABBA database...
Need to import 50 translations
...
```

### Import Specific Translations Only
```bash
$ python abba/main.py --translations KJV ESV
Importing biblical data into ABBA database...
Found 2 translations in bible.db
Skipping KJV - already imported at 2024-01-15T10:31:00
Skipping ESV - already imported at 2024-01-15T10:32:00
All requested translations are already imported.
```

### Handling Failed Imports

If an import fails partway through:
```bash
$ python abba/main.py
Importing verses for XYZ...
Error importing XYZ: Foreign key constraint failed
Successfully imported 49/50 translations
Failed translations: XYZ
```

The failed translation (XYZ) is:
- NOT marked as imported in the tracking file
- NOT partially present in the database (rolled back)
- Can be retried on the next run

## Benefits

1. **Efficiency**: No wasted time re-importing existing data
2. **Reliability**: Failed imports don't corrupt the database
3. **Transparency**: Clear feedback on what's imported/skipped
4. **Flexibility**: Can force re-import when needed
5. **Resumability**: Can stop and resume imports safely

## Technical Details

- Import status stored in: `bible_data/.import_status.json`
- Hidden file (starts with `.`) to avoid clutter
- JSON format for easy inspection/debugging
- Transaction support uses SQLite's ACID guarantees
- Each translation import is one atomic transaction