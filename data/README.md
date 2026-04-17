# Data

This directory introduces the decade-level sub-corpora used in the paper. The corpus files and pretrained vectors are too large to host on GitHub and must be generated or downloaded locally following the instructions below.

## Directory Structure

```
data/
├── README.md
├── en_decade_corpus/          # Generated — see instructions below
│   ├── en_1800s.txt
│   ├── en_1810s.txt
│   ├── en_1820s.txt
│   ├── en_1830s.txt
│   ├── en_1840s.txt
│   ├── en_1850s.txt
│   ├── en_1860s.txt
│   ├── en_1870s.txt
│   ├── en_1880s.txt
│   ├── en_1890s.txt
│   ├── en_1900s.txt
│   ├── en_1910s.txt
│   ├── aggregation_report.txt
│   └── aggregation_stats.json
└── word2vec_vectors/          # Downloaded — see instructions below
```

## Source Corpora

The underlying data comes from two British Library collections:

| Collection | Full Name | Download |
|---|---|---|
| **HMD** | Heritage Made Digital | [https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304](https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304) |
| **LwM** | Living with Machines | [https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0](https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0) |

The combined corpus spans 1800–1919 and comprises approximately 5.73 billion tokens. Our semantic change analysis uses the period 1840–1919 (~4.65 billion tokens); the earlier decades (1800s–1830s) are retained for domain-adaptation pretraining but excluded from the analysis.

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

## Pretrained Word2Vec Vectors

The static embedding analysis uses pretrained and aligned Word2Vec vectors from:

Nilo Pedrazzini, & Barbara McGillivray. (2022). Diachronic word embeddings from 19th-century newspapers digitised by the British Library (1800-1919) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7181682

These vectors were trained on the same HMD and LwM data with the following hyperparameters: skip-gram architecture, 5 epochs, 200 dimensions, context window of 3, minimum word count of 1. Decade-specific vector spaces were aligned using Orthogonal Procrustes (Schönemann, 1966).

We do not redistribute these vectors. Please download these vectors from Zenodo: https://zenodo.org/records/7181682. 

## Licensing

The HMD and LwM datasets are released by the British Library. Please consult the respective dataset pages linked above for licensing terms and conditions of use.
