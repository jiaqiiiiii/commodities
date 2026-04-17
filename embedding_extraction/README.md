# Embedding Extraction and Analysis

This directory contains the pipeline for extracting contextualised embeddings from MacBERTh for the six target commodity terms, clustering them to identify usage types, extracting representative snippets, and computing Jensen–Shannon Divergence (JSD) between consecutive decades. 

This is the core analytical pipeline of the paper, producing the data behind Tables 3, 4, 5–10, and Figures 8 and 9.

The directory follows scripts in the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change) and also [Andrey Kutuzov and Mario Giulianelli's participation in SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://github.com/akutuzov/semeval2020/tree/master) .

## Directory Structure

```
embedding_extraction/
├── README.md
├── gen_macberth.py                         # MacBERTh wrapper for embedding extraction
├── embeddings_extraction.py                # Extract target-word embeddings per decade
└── analysis/
    ├── commodity_analysis_finetuned.py     # Clustering + visualisation (domain-adapted model)
    ├── commodity_analysis_pretrained.py    # Clustering + visualisation (pretrained model)
    ├── extract_contexts_visible_only.py    # Centroid-closest snippets with target-word filtering
    ├── jsd.py                              # JSD computation via Affinity Propagation
    └── output/
        ├── csv/                            # Clustering summaries, JSD values
        ├── PCA/                            # Cross-decade PCA plots per commodity
        ├── clusters/                       # Per-decade cluster visualisation plots
        └── cluster_contexts/               # Representative snippets per cluster (CSV)
```

## Files

### Embedding extraction

| File | Description |
|---|---|
| `gen_macberth.py` | MacBERTh wrapper class. Handles tokenisation, subword-to-word alignment via transform matrices, and batched embedding extraction from the last hidden layer. Adapted from the `gen_berts.py` LatinBERT wrapper in the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change). Supports both the pretrained and domain-adapted MacBERTh |
| `embeddings_extraction.py` | Main extraction script. For each decade, finds all sentences containing target words (*coffee*, *tea*, *sugar*, *opium*, *cocoa*, *tobacco*), extracts the last-hidden-layer embedding at the target word's position, and saves embeddings with snippets to HDF5 files. Supports checkpointing (skips words already extracted) |

### Analysis

| File | Description |
|---|---|
| `commodity_analysis_finetuned.py` | Full analysis pipeline for the **domain-adapted** model: K-means clustering with silhouette-selected k ∈ [2, 10], per-decade cluster visualisations (PCA-projected), cross-decade PCA plots, and example snippets per cluster. Produces Table 4 (clustering statistics) and Figure 8 (coffee cluster plots) |
| `commodity_analysis_pretrained.py` | Identical pipeline for the **pretrained** model. Used for the domain-adaptation evaluation (Appendix): comparing whether the pretrained model can separate recognisable senses |
| `extract_contexts_visible_only.py` | Extracts **all** cluster contexts ranked by distance to centroid, filtered to keep only snippets where the target word is visible within the display window. Re-centres snippets around the target word (±150 characters). Produces the data behind Tables 5–10 in the paper |
| `jsd.py` | Computes JSD between consecutive decades and JSD vs. 1910 reference using Affinity Propagation on the combined usage matrix. Based on [Kutuzov & Giulianelli (2020)](https://github.com/akutuzov/semeval2020). Produces Table 3. Includes checkpointing so interrupted runs can be resumed |

## Pipeline

### Step 1: Extract embeddings

Requires the domain-adapted MacBERTh model (from `domain_adaptation/`) and the decade sub-corpora (from `data/en_decade_corpus/`).

```bash
# Extract embeddings using the domain-adapted model
python embeddings_extraction.py --model finetuned

# Extract embeddings using the pretrained model (for domain-adaptation evaluation)
python embeddings_extraction.py --model pretrained
```

**What it does:**
1. Loads MacBERTh (pretrained or domain-adapted) via the `gen_macberth.py` wrapper
2. For each decade file, scans all lines for sentences containing target words
3. Extracts the last-hidden-layer embedding (768-d) at the target word's token position, using a transform matrix to handle subword tokenisation
4. Saves to HDF5: one file per decade (`commodity_embeddings_{decade}.h5`), with embeddings, snippets, and positions organised by word

**Output format** (HDF5):
```
commodity_embeddings_1840.h5
├── coffee/
│   ├── usage_0/
│   │   └── embedding    # (768,) float32, gzip-compressed
│   │   └── attrs: snippet, position
│   ├── usage_1/
│   └── ...
├── tea/
└── ...
```

**Note:** Update `dir_in` at the top of the script to point to your local data directory. Extraction processes ~50 sentences per batch and clears GPU memory between batches to avoid OOM on large decades (e.g., tea in the 1890s has ~42,000 usages).

### Step 2: Clustering analysis

```bash
# Run analysis on domain-adapted embeddings
python analysis/commodity_analysis_finetuned.py

# Run analysis on pretrained embeddings (for comparison)
python analysis/commodity_analysis_pretrained.py
```

**What it does:**
1. Loads all decade HDF5 files
2. For each word–decade pair, finds optimal k via silhouette score over k ∈ [2, 10]
3. Runs K-means with the selected k (random_state=42, n_init=10)
4. Saves cluster labels, centroid vectors, and PCA-projected cluster plots
5. Prints example snippets per cluster

**Outputs:**
- `output/csv/cluster_summary.csv` — Table 4 data (word, decade, n_clusters, silhouette, cluster sizes)
- `output/clusters/clusters_{word}_{decade}.png` — Per-decade cluster visualisations (Figure 8)
- `output/PCA/pca_{word}_all_decades.png` — Cross-decade PCA projections (Figure 9)

### Step 3: Extract centroid-closest snippets

```bash
python analysis/extract_contexts_visible_only.py
```

**What it does:**
1. For each word–decade pair, runs the same K-means clustering as Step 2
2. Ranks all snippets within each cluster by Euclidean distance to the cluster centroid
3. Re-centres each snippet around the target word (±150 characters) to ensure visibility
4. Filters out snippets where the target word falls outside the display window
5. Saves all contexts (ranked by centroid proximity) to CSV files

The **five closest visible snippets per cluster** from these CSVs are what appear in the paper's appendix tables (Tables 5–10). The thematic category labels in those tables are the authors' interpretive readings, not algorithmically derived.

**Outputs:**
- `output/cluster_contexts/contexts_{word}_{decade}.csv` — Per word-decade files with all ranked contexts
- `output/cluster_contexts/all_cluster_contexts.csv` — Combined file
- `output/csv/cluster_summary_detailed.csv` — Summary with visible-snippet counts

### Step 4: Compute JSD

```bash
python analysis/jsd.py
```

**What it does:**
1. For each pair of consecutive decades, pools embeddings from both decades into a combined matrix [U^t1; U^t2]
2. Computes cosine similarity and clusters via Affinity Propagation (damping=0.9, precomputed affinity)
3. Derives per-decade probability distributions from normalised cluster counts
4. Computes Jensen–Shannon Divergence using `scipy.spatial.distance.jensenshannon`
5. Also computes JSD vs. the 1910s reference decade for all words

The AP-based JSD implementation follows [Kutuzov & Giulianelli (2020)](https://aclanthology.org/2020.semeval-1.14/), who in turn draw on [Giulianelli et al. (2020)](https://aclanthology.org/2020.acl-main.365/) and [Martinc et al. (2020)](https://aclanthology.org/2020.lrec-1.598/). See the paper's Section 4.2 for the methodological justification of using AP (rather than K-means) for the JSD computation.

**Note on checkpointing:** The script supports resumption — it checks which word-decade pairs already exist in the output CSV and skips them. This is useful because AP on large embedding matrices (e.g., tea with 40k+ usages per decade) can take substantial time.

**Outputs:**
- `output/csv/jsd_consecutive_decades.csv` — Table 3 data (word, decade_start, decade_end, JSD)
- `output/csv/jsd_vs_1910.csv` — JSD of each decade against the 1910s reference

## Relationship to the paper

| Paper element | Produced by | File(s) |
|---|---|---|
| Table 3 (JSD values) | `jsd.py` | `output/csv/jsd_consecutive_decades.csv` |
| Table 4 (clustering statistics) | `commodity_analysis_finetuned.py` | `output/csv/cluster_summary.csv` |
| Tables 5–10 (cluster contexts) | `extract_contexts_visible_only.py` | `output/cluster_contexts/` |
| Figure 8 (coffee clusters) | `commodity_analysis_finetuned.py` | `output/clusters/clusters_coffee_*.png` |
| Figure 9 (sugar/cocoa PCA) | `commodity_analysis_finetuned.py` | `output/PCA/pca_sugar_all_decades.png`, `pca_cocoa_all_decades.png` |
| Appendix (domain-adaptation comparison) | `commodity_analysis_pretrained.py` + `commodity_analysis_finetuned.py` | Compared qualitatively in the paper |

## Dependencies

Key packages beyond the standard scientific Python stack:

- `transformers` (HuggingFace) — for loading MacBERTh
- `h5py` — for HDF5 embedding storage
- `scikit-learn` — for K-means, Affinity Propagation, PCA, silhouette score
- `scipy` — for `jensenshannon` distance
- `torch` — for GPU-accelerated model inference

See the root `requirements.txt` for version pinning.
