# Static Embeddings Analysis

This directory contains the static Word2Vec analysis pipeline, addressing **RQ1: did the dominant usages of commodity terms shift across the decades?** It uses pretrained, decade-aligned Word2Vec vectors from [Pedrazzini & McGillivray (2022)](https://aclanthology.org/2022.nlp4dh-1.11/) to compute cosine similarity trajectories, extract nearest neighbours, and produce t-SNE visualisations of semantic change.

This corresponds to **Section 4.1** in the paper and produces **Figure 4** (cosine similarity), **Table 2** (nearest neighbours), and **Figures 5–7** (t-SNE trajectories).

## Files

| File | Description | Paper element |
|---|---|---|
| `extract_neighbors.py` | Extract nearest neighbours and vocabulary statistics for target words across all decades. Computes top-N neighbours by cosine similarity, tracks vocabulary rank, and analyses stable vs. new vs. lost neighbours over time | Table 2 |
| `visualize_semantic_trajectory.py` | t-SNE visualisation of semantic change trajectories. Projects target word vectors and their neighbours into a shared 2D space, with trajectory arrows showing the direction of semantic movement | Figures 5, 6, 7 |

**Note:** Cosine similarity vs. the 1910s reference decade (Figure 4) can be computed from the same vectors using the neighbour extraction script's output or directly from the loaded models. See usage examples below.

## Pretrained Word2Vec Vectors

We use pretrained, decade-aligned diachronic embeddings from:

> Nilo Pedrazzini and Barbara McGillivray (2022). *Diachronic word embeddings from 19th-century British newspapers* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7181682

These vectors were trained on the HMD and LwM newspaper corpus (~4.2 billion tokens) with skip-gram architecture, 5 epochs, 200 dimensions, context window of 3, minimum word count of 1. Decade-specific vector spaces were aligned using Orthogonal Procrustes (Schönemann, 1966).

**Download and extract:**

```bash
wget https://zenodo.org/records/7181682/files/lwm_vectors.zip
unzip lwm_vectors.zip -d vectors/
```

**Expected structure:**

```
vectors/
├── 1840s-vectors.txt
├── 1850s-vectors.txt
├── 1860s-vectors.txt
├── 1870s-vectors.txt
├── 1880s-vectors.txt
├── 1890s-vectors.txt
├── 1900s-vectors.txt
└── 1910s-vectors.txt
```

Supported formats: Word2Vec text (`*.txt`), binary (`*.bin`), or FastText (`*.vec`).

## Usage

### Extract nearest neighbours (Table 2)

```bash
# All six commodity terms
python extract_neighbors.py --all-commodities --vectors_dir ./vectors --output_dir ./output

# Single word with custom parameters
python extract_neighbors.py --word coffee --vectors_dir ./vectors --topn 50

# With OCR spelling variants
python extract_neighbors.py --word coffee --add-variant coffee coffe --add-variant coffee coffée
```

**Output files:**

| File | Description |
|---|---|
| `neighbors_{word}.csv` | All neighbours per decade for a specific word |
| `neighbors_{word}_wide.csv` | Wide format with decades as columns (directly corresponds to Table 2) |
| `all_neighbors.csv` | Combined neighbours for all target words |
| `neighbor_changes_summary.csv` | Stable/new/lost neighbour analysis across decades |
| `word_frequencies.csv` | Vocabulary rank per decade (proxy for frequency) |

### Visualise semantic trajectories (Figures 5–7)

```bash
# All commodities
python visualize_semantic_trajectory.py --all-commodities --vectors_dir ./vectors --output_dir ./output

# Single word
python visualize_semantic_trajectory.py --word coffee --vectors_dir ./vectors

# Specific decades
python visualize_semantic_trajectory.py --word tea --decades 1840s 1870s 1900s
```

**Output files:**

| File | Description |
|---|---|
| `trajectory_{word}.png` | t-SNE visualisation (Figures 5–7 in the paper) |
| `neighbors_{word}.csv` | Neighbour data used in the plot |

### Interpreting the t-SNE visualisations

- **Large coloured dots**: position of the target word at each decade
- **Arrows**: direction of semantic movement between consecutive decades
- **Small dots**: nearest neighbours, coloured by the decade in which they appear
- **Dark grey dots**: "stable" neighbours appearing across all decades
- Long arrows indicate major semantic shift; tight clustering indicates stable meaning

## Handling OCR Errors

Historical newspaper corpora contain OCR noise. The scripts support spelling variant specification:

```bash
python extract_neighbors.py \
    --word coffee \
    --add-variant coffee coffe \
    --add-variant coffee coffce
```

Or modify the `TARGET_VARIANTS` dictionary in the script:

```python
TARGET_VARIANTS = {
    'coffee': ['coffee', 'coffe', 'coffce'],
    'tea': ['tea', 'tee'],
    # add your own
}
```

The visualisation script also supports OCR filtering of neighbours (`--no-filter-ocr` to disable).

## Methodology

**Nearest Neighbour Extraction:**
1. Load aligned Word2Vec models for each decade
2. For each target word, find top-N most similar words by cosine similarity
3. Track vocabulary rank as proxy for frequency
4. Compare neighbour sets across decades to identify semantic stability and change

**Trajectory Visualisation:**
1. Collect vectors for the target word across all decades
2. Collect vectors for all unique neighbours (using the 1910s model as reference)
3. Filter likely OCR errors, retaining only recognised English words
4. Apply t-SNE dimensionality reduction (2 components, Euclidean distance)
5. Plot neighbours coloured by decade, with trajectory arrows for the target word

**Limitations:**
- t-SNE is stochastic; results may vary between runs (use `--random_state` for reproducibility)
- Embedding alignment assumes comparable training corpora across periods
- Neighbour quality depends on embedding quality and corpus size per decade

## Relationship to the paper

| Paper element | Produced by | Key finding |
|---|---|---|
| Figure 4 (cosine similarity) | Computed from loaded Word2Vec models | Coffee, tea, sugar show clear drift toward 1910s sense; opium relatively stable |
| Table 2 (nearest neighbours) | `extract_neighbors.py` | Coffee shifts from *sugar, tobacco, cocoa* (1850s) to *sandwiches, porridge, muffins* (1910s) |
| Figures 5–7 (t-SNE) | `visualize_semantic_trajectory.py` | Spatial separation between early trade neighbourhood and late culinary neighbourhood |

## Acknowledgements

- Pretrained vectors by [Pedrazzini & McGillivray (2022)](https://zenodo.org/records/7181682)
- Code adapted from [Living with Machines: DiachronicEmb-BigHistData](https://github.com/Living-with-machines/DiachronicEmb-BigHistData)
