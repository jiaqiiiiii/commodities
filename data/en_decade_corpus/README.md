## Generating the Decade Sub-Corpora

### Step 1: Download and extract the source data

Download both collections from the links above and extract them. After extraction you will have two directories:

```
hmd-alto2txt/
├── plaintext/        ← this is what we need
└── metadata/

lwm-alto2txt/
├── plaintext/        ← this is what we need
└── metadata/
```

Only the `plaintext/` subdirectories are needed. Each contains newspaper articles as individual `.txt` files organised by newspaper ID, year, and date (e.g., `0003038/1899/0929/0003038_18990929_art0087.txt`).

### Step 2: Aggregate and partition by decade

Run the aggregation script from the `corpus_preparation/` directory:

```bash
python corpus_preparation/aggregate_and_partition.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ./data/en_decade_corpus \
    --workers 4
```

This script:
1. Scans all `.txt` files in both plaintext directories
2. Extracts the publication year from each file's path (pattern: `{newspaper_id}_{YYYYMMDD}_art{number}.txt`)
3. Aggregates all text into decade-level files (`en_1800s.txt`, `en_1810s.txt`, ..., `en_1910s.txt`)
4. Generates an aggregation report (`aggregation_report.txt`) and statistics (`aggregation_stats.json`)

Processing ~9 million files takes several hours depending on hardware. The `--workers` flag controls the number of parallel processes.

### Expected output

After running the script, `data/en_decade_corpus/` should contain one `.txt` file per decade with approximate token counts matching Table 1 in the paper:

| Decade | Approximate tokens |
|---|---|
| 1800s | 142,200,426 |
| 1810s | 191,189,208 |
| 1820s | 239,720,024 |
| 1830s | 183,986,090 |
| 1840s | 601,310,976 |
| 1850s | 667,080,887 |
| 1860s | 651,028,965 |
| 1870s | 539,171,959 |
| 1880s | 868,343,506 |
| 1890s | 747,699,750 |
| 1900s | 580,114,837 |
| 1910s | 318,302,058 |

