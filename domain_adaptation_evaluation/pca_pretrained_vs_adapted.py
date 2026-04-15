"""
PCA Visualization: Pretrained vs Domain-Adapted MacBERTh

Adapted from Qiu & Xu (2022) HistBERT PCA visualization.
For each target commodity word and each decade, this script:
  1. Loads embeddings from both pretrained and domain-adapted MacBERTh
  2. Samples N usages (to keep plots readable)
  3. Fits a joint PCA on the combined embeddings
  4. Plots pretrained points vs adapted points with distinct markers

Usage:
    python pca_pretrained_vs_adapted.py \
        --pretrained_dir embeddings_macberth_pretrained \
        --adapted_dir embeddings_macberth_finetuned \
        --output_dir pca_plots \
        --decades 1840 1850 1860 1870 1880 1890 1900 1910 \
        --words coffee tea sugar opium cocoa tobacco \
        --sample_size 200
"""

import argparse
import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


def load_embeddings_from_h5(h5_path, word, max_n=None):
    """
    Load embeddings for a given word from an H5 file.
    Structure: f[word][usage_N]['embedding'] -> (768,)
    
    Returns numpy array of shape (n_usages, 768).
    """
    embeddings = []
    with h5py.File(h5_path, 'r') as f:
        if word not in f:
            print(f"  Warning: '{word}' not found in {h5_path}")
            return np.array([])
        
        word_group = f[word]
        # Sort usage keys numerically
        usage_keys = sorted(
            word_group.keys(),
            key=lambda x: int(re.search(r'\d+', x).group())
        )
        
        if max_n is not None and max_n > 0 and len(usage_keys) > max_n:
            # Random sample without replacement
            rng = np.random.RandomState(42)
            indices = rng.choice(len(usage_keys), size=max_n, replace=False)
            indices.sort()
            usage_keys = [usage_keys[i] for i in indices]
        
        for uk in usage_keys:
            emb = word_group[uk]['embedding'][:]
            embeddings.append(emb)
    
    if len(embeddings) == 0:
        return np.array([])
    return np.stack(embeddings)


def plot_pca_comparison(pretrained_embs, adapted_embs, word, decade, output_path):
    """
    Following Qiu & Xu (2022): fit joint PCA on combined embeddings,
    plot pretrained points and adapted points with different markers/colors.
    """
    n_pre = len(pretrained_embs)
    n_ada = len(adapted_embs)
    
    if n_pre == 0 or n_ada == 0:
        print(f"  Skipping {word} {decade}s: insufficient data "
              f"(pretrained={n_pre}, adapted={n_ada})")
        return
    
    # Combine and fit PCA jointly (as in HistBERT)
    combined = np.vstack([pretrained_embs, adapted_embs])
    pca = PCA(n_components=2, random_state=42)
    two_dim = pca.fit_transform(combined)
    
    pre_2d = two_dim[:n_pre]
    ada_2d = two_dim[n_pre:]
    
    explained = pca.explained_variance_ratio_
    
    # Auto-scale marker size and alpha for readability
    total = n_pre + n_ada
    if total > 10000:
        ms, alpha = 3, 0.15
    elif total > 5000:
        ms, alpha = 5, 0.2
    elif total > 2000:
        ms, alpha = 8, 0.3
    elif total > 500:
        ms, alpha = 12, 0.4
    else:
        ms, alpha = 20, 0.5
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(
        pre_2d[:, 0], pre_2d[:, 1],
        c='#7a9abb',       # blue-grey
        marker='o',
        s=ms,
        alpha=alpha,
        edgecolors='none',
        label=f'Pretrained MacBERTh (n={n_pre:,})'
    )
    
    ax.scatter(
        ada_2d[:, 0], ada_2d[:, 1],
        c='#c0392b',       # red
        marker='^',
        s=ms,
        alpha=alpha,
        edgecolors='none',
        label=f'Domain-adapted MacBERTh (n={n_ada:,})'
    )
    
    ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'{word} — {decade}s', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pca_all_decades(pretrained_dir, adapted_dir, word, decades, 
                         output_path, sample_size=200):
    """
    Single combined plot showing all decades for one word.
    Pretrained = circles, Adapted = triangles.
    Each decade gets a different color.
    """
    # Colormap for decades
    cmap = plt.cm.viridis
    decade_colors = {d: cmap(i / max(len(decades) - 1, 1)) 
                     for i, d in enumerate(decades)}
    
    all_pretrained = {}
    all_adapted = {}
    
    for decade in decades:
        pre_path = os.path.join(pretrained_dir, 
                                f'commodity_embeddings_{decade}.h5')
        ada_path = os.path.join(adapted_dir, 
                                f'commodity_embeddings_{decade}.h5')
        
        if not os.path.exists(pre_path) or not os.path.exists(ada_path):
            continue
        
        pre_embs = load_embeddings_from_h5(pre_path, word, max_n=sample_size)
        ada_embs = load_embeddings_from_h5(ada_path, word, max_n=sample_size)
        
        if len(pre_embs) > 0 and len(ada_embs) > 0:
            all_pretrained[decade] = pre_embs
            all_adapted[decade] = ada_embs
    
    if not all_pretrained:
        print(f"  No data for {word}, skipping combined plot.")
        return
    
    # Combine everything for joint PCA
    all_embs = []
    for d in decades:
        if d in all_pretrained:
            all_embs.append(all_pretrained[d])
            all_embs.append(all_adapted[d])
    
    combined = np.vstack(all_embs)
    pca = PCA(n_components=2, random_state=42)
    two_dim = pca.fit_transform(combined)
    explained = pca.explained_variance_ratio_
    
    # Split back
    total_points = len(combined)
    if total_points > 50000:
        ms, alpha = 2, 0.1
    elif total_points > 20000:
        ms, alpha = 3, 0.15
    elif total_points > 5000:
        ms, alpha = 5, 0.2
    elif total_points > 2000:
        ms, alpha = 8, 0.3
    else:
        ms, alpha = 15, 0.35
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    offset = 0
    for d in decades:
        if d not in all_pretrained:
            continue
        
        n_pre = len(all_pretrained[d])
        n_ada = len(all_adapted[d])
        color = decade_colors[d]
        
        pre_2d = two_dim[offset:offset + n_pre]
        offset += n_pre
        ada_2d = two_dim[offset:offset + n_ada]
        offset += n_ada
        
        ax.scatter(
            pre_2d[:, 0], pre_2d[:, 1],
            c=[color], marker='o', s=ms, alpha=alpha,
            edgecolors='none'
        )
        ax.scatter(
            ada_2d[:, 0], ada_2d[:, 1],
            c=[color], marker='^', s=ms, alpha=alpha,
            edgecolors='none'
        )
    
    # Custom legend
    decade_handles = [
        Line2D([0], [0], marker='s', color='w', 
               markerfacecolor=decade_colors[d], markersize=8, label=f'{d}s')
        for d in decades if d in all_pretrained
    ]
    model_handles = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='grey', markersize=8, label='Pretrained'),
        Line2D([0], [0], marker='^', color='w', 
               markerfacecolor='grey', markersize=8, label='Domain-adapted'),
    ]
    
    legend1 = ax.legend(handles=decade_handles, title='Decade', 
                        loc='upper left', fontsize=9, title_fontsize=10,
                        framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=model_handles, title='Model', 
              loc='upper right', fontsize=9, title_fontsize=10,
              framealpha=0.9)
    
    ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'{word} — all decades (pretrained ○ vs adapted △)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='PCA visualization comparing pretrained vs '
                    'domain-adapted MacBERTh embeddings'
    )
    parser.add_argument('--pretrained_dir', type=str, required=True,
                        help='Directory with pretrained embedding H5 files')
    parser.add_argument('--adapted_dir', type=str, required=True,
                        help='Directory with domain-adapted embedding H5 files')
    parser.add_argument('--output_dir', type=str, default='pca_plots',
                        help='Output directory for plots')
    parser.add_argument('--decades', type=int, nargs='+',
                        default=[1840, 1850, 1860, 1870, 
                                 1880, 1890, 1900, 1910],
                        help='Decades to process')
    parser.add_argument('--words', type=str, nargs='+',
                        default=['coffee', 'tea', 'sugar', 
                                 'opium', 'cocoa', 'tobacco'],
                        help='Target words')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Max usages to sample per word-decade. '
                             '0 = use all usages (default).')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Per-word, per-decade plots
    for word in args.words:
        print(f"\nProcessing: {word}")
        word_dir = os.path.join(args.output_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        
        for decade in args.decades:
            pre_path = os.path.join(
                args.pretrained_dir, 
                f'commodity_embeddings_{decade}.h5'
            )
            ada_path = os.path.join(
                args.adapted_dir, 
                f'commodity_embeddings_{decade}.h5'
            )
            
            if not os.path.exists(pre_path):
                print(f"  Missing pretrained file for {decade}s, skipping.")
                continue
            if not os.path.exists(ada_path):
                print(f"  Missing adapted file for {decade}s, skipping.")
                continue
            
            pre_embs = load_embeddings_from_h5(
                pre_path, word, max_n=args.sample_size
            )
            ada_embs = load_embeddings_from_h5(
                ada_path, word, max_n=args.sample_size
            )
            
            out_file = os.path.join(word_dir, f'{word}_{decade}s.png')
            plot_pca_comparison(pre_embs, ada_embs, word, decade, out_file)
        
        # Combined all-decades plot
        combined_out = os.path.join(word_dir, f'{word}_all_decades.png')
        plot_pca_all_decades(
            args.pretrained_dir, args.adapted_dir, word,
            args.decades, combined_out, sample_size=args.sample_size
        )
    
    print(f"\nDone. All plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
