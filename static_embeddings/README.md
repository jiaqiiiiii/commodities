# Diachronic Word Embedding Analysis Tools

Tools for analyzing semantic change using Word2Vec embeddings trained on historical text corpora. Originally developed for studying colonial commodity terms in 19th-century British newspapers.

## Overview

This repository provides two complementary scripts:

1. **`extract_neighbors.py`** — Extract nearest neighbors and vocabulary statistics across time periods
2. **`visualize_semantic_trajectory.py`** — Visualize semantic change trajectories using t-SNE

Together, these tools enable quantitative and visual analysis of how word meanings shift over time.

## Requirements

```bash
# Core dependencies
pip install numpy pandas matplotlib scikit-learn gensim tqdm

# Optional but recommended
pip install adjustText      # Prevents text label overlap in visualizations
pip install pyspellchecker  # Filters OCR errors from historical texts
```

## Input Data

Both scripts expect Word2Vec models in text or binary format, organized by time period:

```
vectors/
├── 1840s-vectors.txt    # or 1840s.txt, 1840s.vec, 1840s.bin
├── 1850s-vectors.txt
├── 1860s-vectors.txt
├── 1870s-vectors.txt
├── 1880s-vectors.txt
├── 1890s-vectors.txt
├── 1900s-vectors.txt
└── 1910s-vectors.txt
```

### Supported formats
- Word2Vec text format (`*.txt`, `*-vectors.txt`)
- Word2Vec binary format (`*.bin`)
- FastText vectors (`*.vec`)

### Training your own embeddings

---

## Script 1: Extract Neighbors

`extract_neighbors.py` extracts nearest neighbors for target words across all time periods.

### Features

- Extracts top-N nearest neighbors per decade
- Computes vocabulary rank (proxy for word frequency)
- Handles OCR/spelling variations
- Analyzes stable vs. new vs. lost neighbors over time

### Usage

```bash
# Single word
python extract_neighbors.py --word coffee --vectors_dir ./vectors --output_dir ./output

# Multiple words
python extract_neighbors.py --words coffee tea sugar --vectors_dir ./vectors

# All example commodities
python extract_neighbors.py --all-commodities --vectors_dir ./vectors

# Custom parameters
python extract_neighbors.py --word opium --topn 50 --decades 1840s 1860s 1880s 1900s

# Add spelling variant for noisy OCR text
python extract_neighbors.py --word coffee --add-variant coffee coffe --add-variant coffee coffée
```

### Output files

```
output/
├── word_frequencies.csv          # Vocabulary ranks per decade
├── word_frequencies_wide.csv     # Ranks in wide format (decades as columns)
├── all_neighbors.csv             # All neighbors for all words
├── neighbors_{word}.csv          # Neighbors for specific word
├── neighbors_{word}_wide.csv     # Wide format (decades as columns)
└── neighbor_changes_summary.csv  # Stable/new/lost neighbor analysis
```

### Example output: `neighbors_coffee_wide.csv`

| rank | 1840s | 1850s | 1860s | 1870s | 1880s |
|------|-------|-------|-------|-------|-------|
| 1 | tea | tea | tea | tea | tea |
| 2 | sugar | sugar | cocoa | cocoa | cocoa |
| 3 | spices | cocoa | sugar | chocolate | temperance |
| ... | ... | ... | ... | ... | ... |

---

## Script 2: Visualize Semantic Trajectory

`visualize_semantic_trajectory.py` creates t-SNE visualizations showing how words move through semantic space over time.

### Features

- Applies t-SNE to project embeddings to 2D
- Shows trajectory arrows between time periods
- Colors neighbors by the decade they appear in
- Optional OCR error filtering

### Usage

```bash
# Single word
python visualize_semantic_trajectory.py --word coffee --vectors_dir ./vectors --output_dir ./output

# Multiple words
python visualize_semantic_trajectory.py --words coffee tea sugar opium

# All commodities
python visualize_semantic_trajectory.py --all-commodities --vectors_dir ./vectors

# Fewer time points for cleaner visualization
python visualize_semantic_trajectory.py --word tea --decades 1840s 1870s 1900s

# More neighbors, no OCR filtering
python visualize_semantic_trajectory.py --word sugar --topn 30 --no-filter-ocr
```

### Output files

```
output/
├── trajectory_coffee.png    # Visualization
├── neighbors_coffee.csv     # Neighbor data used in plot
├── trajectory_tea.png
├── neighbors_tea.csv
└── ...
```

### Interpreting the visualization

- **Large colored dots**: Position of target word at each decade
- **Arrows**: Direction of semantic movement over time
- **Small dots**: Nearest neighbors, colored by decade of appearance
- **Dark gray dots**: "Stable" neighbors appearing across all decades
- **Trajectory pattern**: Long arrows = major semantic shift; clustering = stable meaning

---

## Complete Workflow Example

```bash
# 1. Extract neighbors for analysis
python extract_neighbors.py \
    --all-commodities \
    --vectors_dir ./lwm_vectors \
    --output_dir ./neighbor_analysis \
    --topn 100

# 2. Create trajectory visualizations
python visualize_semantic_trajectory.py \
    --all-commodities \
    --vectors_dir ./lwm_vectors \
    --output_dir ./trajectory_plots \
    --decades 1840s 1850s 1860s 1870s 1880s 1890s 1900s 1910s

# 3. Examine results
ls ./neighbor_analysis/
ls ./trajectory_plots/
```

---

## Customization

### Adding new target words

Both scripts accept any word via `--word` or `--words`:

```bash
python extract_neighbors.py --words railway steam telegraph --vectors_dir ./vectors
```

### Handling OCR errors / spelling variations

For historical corpora with OCR noise, you can specify spelling variants:

```bash
python extract_neighbors.py \
    --word coffee \
    --add-variant coffee coffe \
    --add-variant coffee coffce \
    --add-variant coffee eoffee
```

Or modify the `TARGET_VARIANTS` dictionary in the script:

```python
TARGET_VARIANTS = {
    'coffee': ['coffee', 'coffe', 'coffce', 'eoffee'],
    'theatre': ['theatre', 'theater', 'tbeatre'],
    # ... add your own
}
```

### Different time periods

Both scripts support custom decade lists:

```bash
# 20th century analysis
python extract_neighbors.py --word computer --decades 1950s 1960s 1970s 1980s 1990s 2000s

# Finer granularity (if you have per-year models)
python visualize_semantic_trajectory.py --word war --decades 1914 1915 1916 1917 1918
```

---

## Methodology

### Nearest Neighbor Extraction

1. Load Word2Vec models for each time period
2. For each target word, find top-N most similar words by cosine similarity
3. Track vocabulary rank as proxy for frequency
4. Compare neighbor sets across time to identify semantic stability/change

### Trajectory Visualization

1. Collect vectors for target word at each time point
2. Collect vectors for all unique neighbors (using final decade as reference)
3. Apply t-SNE dimensionality reduction
4. Plot neighbors colored by decade, with trajectory arrows for target word

### Limitations

- t-SNE is stochastic; results may vary between runs (use `--random_state` for reproducibility)
- Embedding alignment assumes comparable training corpora across periods
- Neighbor quality depends on embedding quality and corpus size

---

## Acknowledgments

- Developed as part of the TRIFECTA project at KNAW Humanities Cluster
- Trajectory visualization adapted from the Living with Machines project
- Word2Vec embeddings from the [LwM Semantic Change](https://github.com/Living-with-machines) pipeline
