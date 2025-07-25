# STEPBible Data Format Documentation

This document describes the format of the STEPBible data files downloaded by ABBA for Hebrew, Aramaic, and Koine Greek biblical texts.

## Overview

STEPBible provides comprehensive lexical, morphological, and textual data for biblical languages under the CC BY 4.0 license. The data is maintained by Tyndale House, Cambridge and is available at https://github.com/STEPBible/STEPBible-Data.

## Data Files

The system downloads the following essential files to the `bible_data/stepbible/` directory (10 files total):

### 1-4. Hebrew OT Text (TAHOT - Split into 4 files)
**Files**:
- `tahot_gen_deu.txt`: Genesis to Deuteronomy
- `tahot_jos_est.txt`: Joshua to Esther
- `tahot_job_sng.txt`: Job to Song of Songs
- `tahot_isa_mal.txt`: Isaiah to Malachi

**Source**: TAHOT - Translators Amalgamated Hebrew OT

**Format**: Tab-separated values with the following columns:
- `Book`: Book name abbreviation
- `Chapter`: Chapter number
- `Verse`: Verse number
- `Word`: Hebrew word in transliteration
- `Morphology`: Morphological parsing code
- `StrongNumber`: Extended Strong's number
- `Translation`: English gloss/translation

**Note**: These files contain the entire Hebrew Old Testament including Aramaic portions (e.g., parts of Daniel and Ezra).

### 5-6. Greek NT Text (TAGNT - Split into 2 files)
**Files**:
- `tagnt_mat_jhn.txt`: Matthew to John
- `tagnt_act_rev.txt`: Acts to Revelation

**Source**: TAGNT - Translators Amalgamated Greek NT

**Format**: Tab-separated values with the following columns:
- `Book`: Book name abbreviation
- `Chapter`: Chapter number
- `Verse`: Verse number
- `Word`: Greek word in transliteration
- `Morphology`: Morphological parsing code
- `StrongNumber`: Extended Strong's number
- `Translation`: English gloss/translation

### 7. Hebrew Lexicon (`hebrew_lexicon.txt`)
**Source**: TBESH - Translators Brief lexicon of Extended Strongs for Hebrew

**Format**: Tab-separated values with the following columns:
- `StrongNumber`: Extended Strong's number (e.g., H1, H2, H1234a)
- `HebrewWord`: Hebrew word in Hebrew script
- `Transliteration`: Romanized transliteration
- `PartOfSpeech`: Grammatical category (noun, verb, etc.)
- `Definition`: Brief English definition
- `ExtendedDefinition`: More detailed definition when available

### 8. Greek Lexicon (`greek_lexicon.txt`)
**Source**: TBESG - Translators Brief lexicon of Extended Strongs for Greek

**Format**: Tab-separated values with the following columns:
- `StrongNumber`: Extended Strong's number (e.g., G1, G2, G1234a)
- `GreekWord`: Greek word in Greek script
- `Transliteration`: Romanized transliteration
- `PartOfSpeech`: Grammatical category
- `Definition`: Brief English definition
- `ExtendedDefinition`: More detailed definition when available

### 9. Hebrew Morphology (`hebrew_morphology.txt`)
**Source**: TEHMC - Translators Expansion of Hebrew Morphology Codes

**Format**: Tab-separated values describing morphological analysis codes:
- `Code`: Morphological code (e.g., Vqp3ms, Ncmsa)
- `Description`: Human-readable description
- `Components`: Breakdown of code components

**Code Structure**:
- **V**: Verb
- **N**: Noun
- **A**: Adjective
- **P**: Pronoun
- **R**: Adverb
- **C**: Conjunction
- **T**: Particle
- **S**: Preposition

**Verb Codes** (V + stem + conjugation + person + number + gender):
- **Stems**: q=Qal, n=Niphal, p=Piel, P=Pual, h=Hiphil, H=Hophal, t=Hithpael
- **Conjugations**: p=Perfect, i=Imperfect, w=Waw-consecutive, v=Imperative, r=Infinitive construct, s=Infinitive absolute, a=Participle active, t=Participle passive
- **Person**: 1,2,3
- **Number**: s=Singular, d=Dual, p=Plural
- **Gender**: m=Masculine, f=Feminine, c=Common

### 10. Greek Morphology (`greek_morphology.txt`)
**Source**: TEGMC - Translators Expansion of Greek Morphology Codes

**Format**: Tab-separated values describing morphological analysis codes:
- `Code`: Morphological code (e.g., V-PAI-1S, N-NSM)
- `Description`: Human-readable description
- `Components`: Breakdown of code components

**Code Structure**:
- **V**: Verb
- **N**: Noun
- **A**: Adjective
- **P**: Pronoun
- **R**: Adverb
- **C**: Conjunction
- **T**: Particle
- **I**: Interjection
- **X**: Other

**Verb Codes** (V-tense-voice-mood-person-number):
- **Tense**: P=Present, I=Imperfect, A=Aorist, F=Future, R=Perfect, L=Pluperfect, Y=Future Perfect
- **Voice**: A=Active, M=Middle, P=Passive, D=Middle/Passive
- **Mood**: I=Indicative, S=Subjunctive, O=Optative, M=Imperative, N=Infinitive, P=Participle
- **Person**: 1,2,3
- **Number**: S=Singular, P=Plural

## Usage Notes

1. **Character Encoding**: All files are UTF-8 encoded to properly handle Hebrew and Greek characters.

2. **Extended Strong's Numbers**: STEPBible uses extended Strong's numbers that include suffixes (a, b, c, etc.) to distinguish between different meanings of the same root.

3. **Morphological Analysis**: The morphological codes provide detailed grammatical information about each word form, including parsing for verbs and declension information for nouns.

4. **Data Quality**: The data has been cross-checked against multiple sources and manually verified by scholars at Tyndale House.

5. **Updates**: The data is periodically updated. Check the STEPBible GitHub repository for the latest versions.

## Attribution

This data is provided by STEPBible.org and Tyndale House, Cambridge under the CC BY 4.0 license. For more information, corrections, or updates, contact: STEPBible@gmail.com

## References

- STEPBible Repository: https://github.com/STEPBible/STEPBible-Data
- CC BY 4.0 License: https://creativecommons.org/licenses/by/4.0/
- STEPBible Website: https://www.stepbible.org/
- Tyndale House: https://www.tyndalehouse.com/