import os
import torch
import random
from torch.utils.data import Dataset


class NewspaperDataset(Dataset):
    """
    Memory-safe dataset for large corpora.

    Accepts either:
      - A list of file paths (full corpus mode): indexes byte offsets,
        reads one line at a time from disk during training.
      - A list of strings (subcorpus mode): stores texts in memory,
        safe when corpus is small (~79k sentences).
    """

    def __init__(self, data, tokenizer, max_length=256, subsample=1.0, seed=42):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.mode       = None

        # Detect mode: file paths or in-memory strings
        if data and os.path.isfile(str(data[0])):
            # ---- File path mode: index byte offsets -------------------------
            self.mode  = "files"
            self.index = []  # (file_path, byte_offset)
            rng = random.Random(seed)
            print("Indexing corpus files (byte offsets only — no text loaded)...")
            for file_path in data:
                with open(file_path, 'rb') as f:
                    offset = 0
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            if subsample >= 1.0 or rng.random() < subsample:
                                self.index.append((file_path, offset))
                        offset += len(line)
            print(f"Indexed {len(self.index):,} lines")
        else:
            # ---- In-memory mode: small subcorpus ----------------------------
            self.mode  = "memory"
            self.texts = data
            print(f"In-memory dataset: {len(self.texts):,} sentences")

    def __len__(self):
        if self.mode == "files":
            return len(self.index)
        return len(self.texts)

    def __getitem__(self, idx):
        if self.mode == "files":
            file_path, offset = self.index[idx]
            with open(file_path, 'rb') as f:
                f.seek(offset)
                line = f.readline().decode('utf-8', errors='replace').strip()
        else:
            line = self.texts[idx]

        tokens     = self.tokenizer.tokenize(line)
        token_ids  = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        padding_length = self.max_length - len(token_ids)
        input_ids      = token_ids + [0] * padding_length
        attention_mask = [1] * len(token_ids) + [0] * padding_length

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
