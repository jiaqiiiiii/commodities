# Packages
import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from newspaper_dataset import NewspaperDataset

# Define paths
dir_in = "/data/groups/trifecta/jiaqiz/british_newspapers"
dir_corpus = os.path.join(dir_in, "en_decade_corpus")
dir_out = os.path.join(dir_in, "output")

# Create output directory if it doesn't exist
os.makedirs(dir_out, exist_ok=True)

# Find corpus files (decade txt files like 1800.txt, 1810.txt, etc.)
files = os.listdir(dir_corpus)
files = [f for f in files if f.endswith('.txt')]
files = sorted(files)  # Sort by decade

print(f"Found {len(files)} decade files: {files}")

# Prepare corpus
punctuation = ['.', ',', '...', ';', ':', '?', '(', ')', '-', '!', '[', ']', '"', "'", '""', '\n', '']
corpus = list()

# Read files and create corpus
for file_name in files:
    decade = file_name.replace('.txt', '')
    print(f"Reading {file_name}...")
    file_path = os.path.join(dir_corpus, file_name)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line != "":
                # Keep as sentence string (not tokenized list)
                corpus.append(line)

print(f"Total sentences in corpus: {len(corpus)}")

### Fine tune MacBERTh ###

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths to pre-trained model (MacBERTh from HuggingFace)
bert_path = "emanjavacas/MacBERTh"

# Load huggingface-compatible tokenizer
print("Loading MacBERTh tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(bert_path)

# Split texts into train and validation sets (90/10)
train_texts, val_texts = train_test_split(corpus, test_size=0.1, random_state=42)
print(f"Train set: {len(train_texts)} sentences")
print(f"Validation set: {len(val_texts)} sentences")

# Tokenize
train_dataset = NewspaperDataset(train_texts, tokenizer, max_length=256)
val_dataset = NewspaperDataset(val_texts, tokenizer, max_length=256)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=data_collator, num_workers=4)

# Adjust configuration for dropout
print("Loading MacBERTh model...")
config = BertConfig.from_pretrained(bert_path)
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

# Load pre-trained MacBERTh and prepare for fine-tuning
model = BertForMaskedLM.from_pretrained(bert_path, config=config)
model.to(device)
model.train() 

# Define optimizer & learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
epochs = 2
num_training_steps = len(train_dataloader) * epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

print(f"Starting fine-tuning for {epochs} epochs...")
print(f"Total training steps: {num_training_steps}")

# Fine-Tune MacBERTh
for epoch in range(epochs):
    loop = tqdm(train_dataloader, leave=True)
    total_loss = 0

    for batch in loop: 
        # Move batch to GPU if available
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],  
            labels=batch["labels"]
        )
        loss = outputs.loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Update progress bar
        loop.set_description(f"Epoch {epoch + 1}/{epochs}")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            val_loss += outputs.loss.item()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
    model.train()

# Save Fine-Tuned Model
output_model_path = os.path.join(dir_out, "fine_tuned_macberth")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print(f"Fine-tuning complete! Model saved to {output_model_path}")
