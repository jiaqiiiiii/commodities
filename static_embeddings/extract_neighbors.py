"""
Extract Nearest Neighbors from Diachronic Word2Vec Embeddings
==============================================================

For each target word, extract nearest neighbors across historical time periods.
Also extracts vocabulary rank statistics (proxy for word frequency).

Features:
- Handles OCR/spelling variations of target words
- Excludes spelling variants of target from neighbor lists
- Tracks which variant was used in each decade
- Computes stable/new/lost neighbor analysis

Requirements:
    pip install pandas gensim tqdm

Usage:
    python extract_neighbors.py --words coffee tea sugar --vectors_dir ./vectors --output_dir ./output
    python extract_neighbors.py --all-commodities --topn 50
    python extract_neighbors.py --word opium --decades 1840s 1850s 1860s 1870s

Author: Jiaqi (TRIFECTA project, KNAW Humanities Cluster)
"""

import os
import argparse
from glob import glob
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Example target words (colonial commodities)
EXAMPLE_WORDS = ['coffee', 'tea', 'sugar', 'opium', 'cocoa', 'tobacco']

# Default decades
DEFAULT_DECADES = ['1800s', '1810s', '1820s', '1830s', '1840s', '1850s', 
                   '1860s', '1870s', '1880s', '1890s', '1900s', '1910s']

# Spelling variations / OCR errors
# Maps canonical form -> list of possible spellings to search for
# Add your own variations here if working with noisy historical text
TARGET_VARIANTS = {
    'coffee': ['coffee'],
    'tea': ['tea'],
    'sugar': ['sugar'],
    'opium': ['opium'],
    'cocoa': ['cocoa'],
    'tobacco': ['tobacco'],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_word_in_vocab(model, canonical_word, variants_dict):
    """
    Find the best matching variant of a word in the model vocabulary.
    
    Args:
        model: KeyedVectors model
        canonical_word: The canonical form of the word
        variants_dict: Dictionary mapping words to their variants
    
    Returns:
        tuple: (found_variant, canonical_word) or (None, canonical_word)
    """
    variants = variants_dict.get(canonical_word, [canonical_word])
    
    for variant in variants:
        if variant in model:
            return variant, canonical_word
    
    return None, canonical_word


def is_variant_of_word(neighbor, canonical_word, variants_dict):
    """Check if a neighbor is a spelling variant of the target word."""
    variants = variants_dict.get(canonical_word, [canonical_word])
    return neighbor.lower() in [v.lower() for v in variants]


# =============================================================================
# LOAD VECTORS
# =============================================================================

def load_all_models(vectors_dir, decades):
    """
    Load Word2Vec models from a directory.
    
    Args:
        vectors_dir: Path to directory containing vector files
        decades: List of decade names to load
    
    Returns:
        Dict mapping decade name to KeyedVectors model
    """
    print("=" * 70)
    print("LOADING VECTORS")
    print("=" * 70)
    
    models = {}
    
    for decade in decades:
        # Try different file naming conventions
        possible_names = [
            f"{decade}-vectors.txt",
            f"{decade}.txt",
            f"{decade}.vec",
            f"{decade}-vectors.bin",
            f"{decade}.bin",
        ]
        
        filepath = None
        for name in possible_names:
            candidate = os.path.join(vectors_dir, name)
            if os.path.exists(candidate):
                filepath = candidate
                break
        
        if filepath is None:
            print(f"  {decade}: NOT FOUND")
            continue
        
        try:
            is_binary = filepath.endswith('.bin')
            model = KeyedVectors.load_word2vec_format(filepath, binary=is_binary)
            models[decade] = model
            print(f"  {decade}: {len(model):,} words, {model.vector_size} dimensions")
        except Exception as e:
            print(f"  {decade}: ERROR - {e}")
    
    return models


# =============================================================================
# FREQUENCY / VOCABULARY RANK EXTRACTION
# =============================================================================

def extract_frequencies(vectors_dir, target_words, decades, variants_dict):
    """
    Extract word frequencies by reading raw vector files.
    
    In Word2Vec text format, words are typically ordered by frequency
    (most frequent first). The rank serves as a proxy for frequency.
    
    Returns:
        DataFrame with vocabulary rank information
    """
    print("\n" + "=" * 70)
    print("EXTRACTING VOCABULARY RANKS (proxy for frequency)")
    print("=" * 70)
    print("Note: Lower rank = more frequent word.\n")
    
    freq_data = []
    
    for decade in decades:
        # Find the file
        filepath = None
        for pattern in [f"{decade}-vectors.txt", f"{decade}.txt"]:
            candidate = os.path.join(vectors_dir, pattern)
            if os.path.exists(candidate):
                filepath = candidate
                break
        
        if not filepath:
            continue
        
        # Read file to get word order (= frequency rank)
        word_ranks = {}
        total_words = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().split()
                total_words = int(first_line[0])
                
                for rank, line in enumerate(f, 1):
                    parts = line.split()
                    if parts:
                        word = parts[0]
                        word_ranks[word] = rank
                    
                    if rank > 100000:  # Safety limit
                        break
        except Exception as e:
            print(f"  {decade}: ERROR reading file - {e}")
            continue
        
        for canonical_word in target_words:
            variants = variants_dict.get(canonical_word, [canonical_word])
            found_variant = None
            found_rank = None
            
            for variant in variants:
                if variant in word_ranks:
                    found_variant = variant
                    found_rank = word_ranks[variant]
                    break
            
            if found_variant:
                freq_data.append({
                    'word': canonical_word,
                    'variant_used': found_variant,
                    'decade': decade,
                    'vocab_rank': found_rank,
                    'total_vocab': total_words,
                    'rank_percentile': round(100 * found_rank / total_words, 2)
                })
                variant_info = f" (as '{found_variant}')" if found_variant != canonical_word else ""
                print(f"  {decade} - {canonical_word}{variant_info}: "
                      f"rank {found_rank:,} / {total_words:,} ({100*found_rank/total_words:.2f}%)")
            else:
                freq_data.append({
                    'word': canonical_word,
                    'variant_used': None,
                    'decade': decade,
                    'vocab_rank': None,
                    'total_vocab': total_words,
                    'rank_percentile': None
                })
                print(f"  {decade} - {canonical_word}: NOT IN VOCABULARY")
    
    return pd.DataFrame(freq_data)


# =============================================================================
# NEIGHBOR EXTRACTION
# =============================================================================

def extract_neighbors(models, target_words, decades, topn, variants_dict):
    """
    Extract nearest neighbors for each word in each decade.
    
    Args:
        models: Dict of KeyedVectors models
        target_words: List of words to analyze
        decades: List of decades
        topn: Number of neighbors to extract
        variants_dict: Spelling variants dictionary
    
    Returns:
        DataFrame with columns: word, variant_used, decade, rank, neighbor, similarity
    """
    print("\n" + "=" * 70)
    print("EXTRACTING NEAREST NEIGHBORS")
    print("=" * 70)
    
    all_rows = []
    
    for canonical_word in target_words:
        print(f"\n{canonical_word.upper()}:")
        variants_to_exclude = [v.lower() for v in variants_dict.get(canonical_word, [canonical_word])]
        
        for decade in decades:
            if decade not in models:
                continue
            
            model = models[decade]
            found_variant, _ = find_word_in_vocab(model, canonical_word, variants_dict)
            
            if found_variant is None:
                variants = variants_dict.get(canonical_word, [canonical_word])
                print(f"  {decade}: not in vocabulary (tried: {variants})")
                continue
            
            try:
                # Request extra neighbors to account for filtered variants
                neighbors_raw = model.most_similar(positive=found_variant, topn=topn + 10)
                
                # Filter out variants of the target word
                neighbors_filtered = [
                    (neighbor, similarity) 
                    for neighbor, similarity in neighbors_raw 
                    if neighbor.lower() not in variants_to_exclude
                ][:topn]
                
                variant_info = f" (as '{found_variant}')" if found_variant != canonical_word else ""
                
                for rank, (neighbor, similarity) in enumerate(neighbors_filtered, 1):
                    all_rows.append({
                        'word': canonical_word,
                        'variant_used': found_variant,
                        'decade': decade,
                        'rank': rank,
                        'neighbor': neighbor,
                        'similarity': round(similarity, 4)
                    })
                
                top5 = [n for n, s in neighbors_filtered[:5]]
                print(f"  {decade}{variant_info}: {', '.join(top5)}")
                
            except Exception as e:
                print(f"  {decade}: ERROR - {e}")
    
    return pd.DataFrame(all_rows)


# =============================================================================
# NEIGHBOR CHANGE ANALYSIS
# =============================================================================

def analyze_neighbor_changes(df, decades):
    """
    Analyze which neighbors are stable, new, or lost between first and last decade.
    
    Returns:
        List of change summary dictionaries
    """
    print("\n" + "=" * 70)
    print("ANALYZING NEIGHBOR CHANGES")
    print("=" * 70)
    
    all_changes = []
    
    for word in df['word'].unique():
        word_df = df[df['word'] == word]
        available_decades = sorted(word_df['decade'].unique())
        
        if len(available_decades) < 2:
            print(f"  {word}: insufficient decades for comparison")
            continue
        
        first_decade = available_decades[0]
        last_decade = available_decades[-1]
        
        first_neighbors = set(word_df[word_df['decade'] == first_decade]['neighbor'])
        last_neighbors = set(word_df[word_df['decade'] == last_decade]['neighbor'])
        
        stable = first_neighbors & last_neighbors
        new = last_neighbors - first_neighbors
        lost = first_neighbors - last_neighbors
        
        all_changes.append({
            'word': word,
            'first_decade': first_decade,
            'last_decade': last_decade,
            'n_stable': len(stable),
            'n_new': len(new),
            'n_lost': len(lost),
            'stable_neighbors': ', '.join(sorted(stable)[:20]),
            'new_neighbors': ', '.join(sorted(new)[:20]),
            'lost_neighbors': ', '.join(sorted(lost)[:20]),
        })
        
        print(f"  {word}: {len(stable)} stable, {len(new)} new, {len(lost)} lost "
              f"({first_decade} → {last_decade})")
    
    return all_changes


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(neighbors_df, freq_df, changes, output_dir, target_words, decades):
    """Save all results to CSV files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save frequency data
    if not freq_df.empty:
        freq_path = os.path.join(output_dir, "word_frequencies.csv")
        freq_df.to_csv(freq_path, index=False)
        print(f"  {freq_path}")
        
        # Wide format
        pivot = freq_df.pivot(index='word', columns='decade', values='vocab_rank')
        ordered_cols = [d for d in decades if d in pivot.columns]
        if ordered_cols:
            pivot = pivot[ordered_cols]
            wide_path = os.path.join(output_dir, "word_frequencies_wide.csv")
            pivot.to_csv(wide_path)
            print(f"  {wide_path}")
    
    # Save all neighbors
    if not neighbors_df.empty:
        all_path = os.path.join(output_dir, "all_neighbors.csv")
        neighbors_df.to_csv(all_path, index=False)
        print(f"  {all_path}")
        
        # Per-word files
        for word in neighbors_df['word'].unique():
            word_df = neighbors_df[neighbors_df['word'] == word]
            
            # Long format
            word_path = os.path.join(output_dir, f"neighbors_{word}.csv")
            word_df.to_csv(word_path, index=False)
            print(f"  {word_path}")
            
            # Wide format (decades as columns)
            pivot = word_df.pivot(index='rank', columns='decade', values='neighbor')
            ordered_cols = [d for d in decades if d in pivot.columns]
            if ordered_cols:
                pivot = pivot[ordered_cols]
                wide_path = os.path.join(output_dir, f"neighbors_{word}_wide.csv")
                pivot.to_csv(wide_path)
                print(f"  {wide_path}")
    
    # Save change analysis
    if changes:
        changes_df = pd.DataFrame(changes)
        changes_path = os.path.join(output_dir, "neighbor_changes_summary.csv")
        changes_df.to_csv(changes_path, index=False)
        print(f"  {changes_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract nearest neighbors from diachronic Word2Vec embeddings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --word coffee --vectors_dir ./vectors --output_dir ./output
  %(prog)s --words coffee tea sugar --topn 50
  %(prog)s --all-commodities --vectors_dir ./lwm_vectors
  %(prog)s --word opium --decades 1840s 1850s 1860s 1870s 1880s

Available example words: coffee, tea, sugar, opium, cocoa, tobacco
        """
    )
    
    # Input/output
    parser.add_argument('--vectors_dir', type=str, default='./vectors',
                        help='Directory containing Word2Vec vector files (default: ./vectors)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results (default: ./output)')
    
    # Word selection
    parser.add_argument('--word', type=str,
                        help='Single target word to analyze')
    parser.add_argument('--words', type=str, nargs='+',
                        help='Multiple target words to analyze')
    parser.add_argument('--all-commodities', action='store_true',
                        help='Analyze all example commodity words')
    
    # Parameters
    parser.add_argument('--decades', type=str, nargs='+', default=DEFAULT_DECADES,
                        help=f'Decades to analyze (default: all from 1800s-1910s)')
    parser.add_argument('--topn', type=int, default=100,
                        help='Number of nearest neighbors per decade (default: 100)')
    
    # Variant handling
    parser.add_argument('--add-variant', type=str, nargs=2, action='append',
                        metavar=('WORD', 'VARIANT'),
                        help='Add spelling variant: --add-variant coffee coffe')
    
    args = parser.parse_args()
    
    # Determine target words
    if args.all_commodities:
        target_words = EXAMPLE_WORDS
    elif args.words:
        target_words = args.words
    elif args.word:
        target_words = [args.word]
    else:
        parser.error("Please specify --word, --words, or --all-commodities")
    
    # Build variants dictionary
    variants_dict = TARGET_VARIANTS.copy()
    
    # Add any custom variants
    if args.add_variant:
        for word, variant in args.add_variant:
            if word not in variants_dict:
                variants_dict[word] = [word]
            if variant not in variants_dict[word]:
                variants_dict[word].append(variant)
    
    # Ensure all target words have entries
    for word in target_words:
        if word not in variants_dict:
            variants_dict[word] = [word]
    
    print("=" * 70)
    print("EXTRACT NEAREST NEIGHBORS FROM WORD2VEC")
    print("=" * 70)
    print(f"\nVectors directory: {args.vectors_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Target words:      {', '.join(target_words)}")
    print(f"Decades:           {', '.join(args.decades)}")
    print(f"Top N neighbors:   {args.topn}")
    
    # Show variants if any non-trivial
    non_trivial = {k: v for k, v in variants_dict.items() 
                   if k in target_words and (len(v) > 1 or v[0] != k)}
    if non_trivial:
        print(f"\nSpelling variants:")
        for word, variants in non_trivial.items():
            print(f"  {word}: {variants}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract frequencies
    freq_df = extract_frequencies(args.vectors_dir, target_words, args.decades, variants_dict)
    
    # Load models
    models = load_all_models(args.vectors_dir, args.decades)
    
    if not models:
        print("\nERROR: No models loaded!")
        return 1
    
    # Extract neighbors
    neighbors_df = extract_neighbors(models, target_words, args.decades, 
                                      args.topn, variants_dict)
    
    if neighbors_df.empty:
        print("\nERROR: No neighbors extracted!")
        return 1
    
    # Analyze changes
    changes = analyze_neighbor_changes(neighbors_df, args.decades)
    
    # Save results
    save_results(neighbors_df, freq_df, changes, args.output_dir, 
                 target_words, args.decades)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for word in target_words:
        word_df = neighbors_df[neighbors_df['word'] == word]
        n_decades = word_df['decade'].nunique()
        print(f"\n{word.upper()}:")
        print(f"  Decades with data: {n_decades}")
        
        if not word_df.empty and n_decades >= 2:
            available = sorted(word_df['decade'].unique())
            first_neighbors = set(word_df[word_df['decade'] == available[0]]['neighbor'])
            last_neighbors = set(word_df[word_df['decade'] == available[-1]]['neighbor'])
            stable = len(first_neighbors & last_neighbors)
            print(f"  Stable neighbors ({available[0]}→{available[-1]}): {stable}/{args.topn}")
    
    print(f"\n\nResults saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
