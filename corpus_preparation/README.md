# Corpus Preparation

This directory contains scripts for aggregating the Heritage Made Digital (HMD) and Living with Machines (LwM) newspaper collections into decade-level sub-corpora, and for computing token and target-term frequencies across decades.

## Files

| File | Description |
|---|---|
| `aggregate_and_partition.py` | Main aggregation script: scans both HMD and LwM plaintext directories, extracts publication years from file paths, and writes one text file per decade to `data/en_decade_corpus/` |
| `count_tokens_and_terms.py` | Counts total tokens and frequencies of the six target commodity terms (*coffee*, *tea*, *sugar*, *opium*, *cocoa*, *tobacco*) per decade, outputs raw and normalised (per-million) frequencies |
| `decade_term_frequencies.csv` | Output of `count_tokens_and_terms.py`: token counts and term frequencies per decade (both raw and per-million), used to produce Table 1 in the paper |
| `aggregation_report.txt` | Output of `aggregate_and_partition.py`: summary statistics of the aggregation run, including article counts, token counts, and byte sizes per decade, broken down by source (HMD vs LwM) |

## Pipeline

### Step 1: Aggregate and partition

Requires the raw HMD and LwM plaintext directories (see `data/README.md` for download instructions).

```bash
python aggregate_and_partition.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ../data/en_decade_corpus \
    --workers 4
```

**What it does:**
- Scans all `.txt` files under both plaintext directories (~9 million files total)
- Extracts the publication year from each file path using the British Library alto2txt naming convention (e.g., `0003038_18990929_art0087.txt` → year 1899)
- Writes all text from each decade into a single file: `en_1800s.txt`, `en_1810s.txt`, ..., `en_1910s.txt`
- Generates `aggregation_report.txt` (human-readable summary) and `aggregation_stats.json` (machine-readable statistics)

**Runtime:** Several hours on a multi-core machine, depending on disk I/O. The `--workers` flag controls parallelism.

**Key statistics from our run** (from `aggregation_report.txt`):
- Files processed: 7,982,524
- Files skipped: 1,399,130 (empty, too short, or year not extractable)
- Total articles aggregated: 7,927,079
- Total tokens: ~5.62 billion
- Period covered: 1801–1919

### Step 2: Count tokens and term frequencies

Requires the decade sub-corpora generated in Step 1.

```bash
python count_tokens_and_terms.py
```

**Note:** Update the `corpus_dir` variable at the top of the script to point to your local `en_decade_corpus/` directory before running.

**What it does:**
- Reads each decade file and tokenises using whitespace and basic punctuation boundaries
- Counts total tokens per decade
- Counts occurrences of the six target terms: *coffee*, *tea*, *sugar*, *opium*, *cocoa*, *tobacco*
- Computes normalised frequencies (per million tokens)
- Saves all results to `decade_term_frequencies.csv`

**Output columns in `decade_term_frequencies.csv`:**

| Column | Description |
|---|---|
| `decade` | Decade label (e.g., `1800`, `1810`, ..., `1910`) |
| `total_tokens` | Total token count for that decade |
| `coffee`, `tea`, ..., `tobacco` | Raw frequency of each target term |
| `coffee_per_million`, ..., `tobacco_per_million` | Frequency normalised per million tokens |

## Relationship to the Paper

- **Table 1** (Number of tokens per decade) is derived from the `total_tokens` column in `decade_term_frequencies.csv`. Note that minor differences between the token counts here and in Table 1 may arise from different tokenisation approaches (this script uses a simple regex tokeniser; the Word2Vec training used Gensim's tokenisation).
- **`aggregation_report.txt`** documents the provenance of the corpus: how many files were processed, how many were skipped, and the contribution of each source (HMD vs LwM) per decade. The report shows that HMD dominates the earlier decades (1800s–1860s) while LwM dominates the later decades (1870s–1910s), reflecting the different digitisation coverages of the two collections.
- The decade sub-corpora produced by Step 1 are the input to both the **static embedding analysis** (via the pretrained Word2Vec vectors from Pedrazzini & McGillivray 2022, which were trained on the same data) and the **domain adaptation** of MacBERTh (see `domain_adaptation/`).
