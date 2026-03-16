# Results: Diachronic Semantic Analysis of Colonial Commodities

This folder contains precomputed results from analyzing semantic change in six colonial commodity terms (coffee, tea, sugar, opium, cocoa, tobacco) across 19th-century British newspaper corpora.

## Data Source

- **Corpus**: British newspaper articles (1800s–1910s)
- **Embeddings**: Word2Vec models trained per decade using the [Living with Machines](https://livingwithmachines.ac.uk/) pipeline
- **Time span**: 1840s–1910s (8 decades)

---

## File Descriptions

### Vocabulary Statistics

| File | Description |
|------|-------------|
| `word_frequencies.csv` | Vocabulary rank for each word in each decade (lower rank = more frequent) |
| `word_frequencies_wide.csv` | Same data in wide format (decades as columns) |

**Columns in `word_frequencies.csv`:**
- `word`: Target word (canonical form)
- `variant_used`: Spelling variant found in vocabulary (for OCR handling)
- `decade`: Time period
- `vocab_rank`: Position in vocabulary (1 = most frequent)
- `total_vocab`: Total vocabulary size for that decade
- `rank_percentile`: Rank as percentage of vocabulary

---

### Nearest Neighbors

| File | Description |
|------|-------------|
| `all_neighbors.csv` | All nearest neighbors for all words (combined) |
| `neighbors_{word}.csv` | Neighbors for a specific word (long format) |
| `neighbors_{word}_wide.csv` | Neighbors for a specific word (decades as columns) |

**Columns in neighbor files:**
- `word`: Target word
- `decade`: Time period
- `rank`: Neighbor rank (1 = most similar)
- `neighbor`: The neighboring word
- `similarity`: Cosine similarity score (0–1)

**Available words:** `coffee`, `tea`, `sugar`, `opium`, `cocoa`, `tobacco`

---

### Semantic Change Analysis

| File | Description |
|------|-------------|
| `neighbor_changes_summary.csv` | Summary of stable/new/lost neighbors between first and last decade |

**Columns:**
- `word`: Target word
- `first_decade`, `last_decade`: Comparison endpoints
- `n_stable`: Neighbors present in both decades
- `n_new`: Neighbors only in last decade
- `n_lost`: Neighbors only in first decade
- `stable_neighbors`, `new_neighbors`, `lost_neighbors`: Example words (truncated)

---

### Trajectory Visualizations

| File | Description |
|------|-------------|
| `trajectory_{word}.png` | t-SNE visualization of semantic trajectory |

**How to read the plots:**
- **Large colored circles**: Position of target word at each decade
- **Arrows**: Direction of semantic movement through time
- **Small dots**: Nearest neighbors, colored by decade of first appearance
- **Dark gray dots**: "Stable" neighbors appearing across all decades
- **Legend**: Shows color mapping for each decade

---

## Quick Summary of Findings

| Word | Pattern | Key observation |
|------|---------|-----------------|
| **Coffee** | Fragmentation → reconsolidation | Temperance movement creates new clusters in 1870s |
| **Tea** | Stable polysemy | Beverage and meal senses coexist throughout |
| **Sugar** | Domain shift | Political commodity → culinary ingredient |
| **Opium** | Persistent binary | Trade vs. moral discourse maintained |
| **Cocoa** | Brand-mediated domestication | Advertising (Epps) drives semantic change |
| **Tobacco** | Gradual shift | Trade → consumption contexts |

---

## Reproducing These Results

```bash
# Extract neighbors
python extract_neighbors.py \
    --all-commodities \
    --vectors_dir ./vectors \
    --output_dir ./results \
    --topn 100

# Generate trajectory visualizations
python visualize_semantic_trajectory.py \
    --all-commodities \
    --vectors_dir ./vectors \
    --output_dir ./results \
    --decades 1840s 1850s 1860s 1870s 1880s 1890s 1900s 1910s
```

---

## Citation

If you use this data, please cite:

```bibtex
@software{commodity_semantic_change,
  author = {Zhang, Jiaqi},
  title = {Semantic Change in Colonial Commodity Terms},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```

## License

Data: CC-BY 4.0  
Code: MIT License
