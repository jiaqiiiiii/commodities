"""
Find the shared vocabulary across all decade vector files.
"""

import os
import sys

VECTOR_DIR = ".../lwm_vectors" # change it to your filepath
DECADES = ["1840s", "1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s"] # change it as you need

def load_vocab(filepath):
    """Read vocabulary from a word2vec text-format file."""
    vocab = set()
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split()
        # Check if first line is a header (two integers: vocab_size dim)
        if len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit():
            pass  # skip header
        else:
            vocab.add(first_line[0])
        for line in f:
            word = line.split(" ", 1)[0]
            if word:
                vocab.add(word)
    return vocab

# Load all vocabularies
vocabs = {}
for decade in DECADES:
    fpath = os.path.join(VECTOR_DIR, f"{decade}-vectors.txt")
    if not os.path.exists(fpath):
        print(f"WARNING: {fpath} not found, skipping.")
        continue
    v = load_vocab(fpath)
    vocabs[decade] = v
    print(f"{decade}: {len(v):,} words")

# Compute intersection
if len(vocabs) == 0:
    print("No files found.")
    sys.exit(1)

shared = None
for decade, v in vocabs.items():
    if shared is None:
        shared = v.copy()
    else:
        shared &= v

print(f"\nShared vocabulary across {len(vocabs)} decades: {len(shared):,} words")

# Sort and print
shared_sorted = sorted(shared)
print(f"\n--- Full shared vocabulary ({len(shared):,} words) ---")
for w in shared_sorted:
    print(w)
