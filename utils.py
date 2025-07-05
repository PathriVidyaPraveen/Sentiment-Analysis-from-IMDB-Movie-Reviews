import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS




def load_glove_embeddings(glove_path, embedding_dim=100):
    word_to_idx = {}
    embeddings = []

    with open(glove_path, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_to_idx[word] = len(word_to_idx)
            embeddings.append(vector)

    # Add <PAD> and <UNK>
    pad_vector = np.zeros(embedding_dim)
    unk_vector = np.mean(np.stack(embeddings), axis=0)

    word_to_idx['<PAD>'] = len(word_to_idx)
    embeddings.append(pad_vector)

    word_to_idx['<UNK>'] = len(word_to_idx)
    embeddings.append(unk_vector)

    embedding_matrix = torch.tensor(np.stack(embeddings), dtype=torch.float)
    return embedding_matrix, word_to_idx



def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", '', text)
    tokens = text.strip().split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokens


def encode_text(texts, vocab):
    return [[vocab.get(token, vocab['<UNK>']) for token in preprocess_text(text)] for text in texts]




class CustomDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.vocab = vocab
        self.encoded = encode_text(self.texts, vocab)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.vocab['<PAD>'])
        return padded_texts, lengths, torch.tensor(labels, dtype=torch.long)




def load_imdb_dataset():
    from datasets import load_dataset, concatenate_datasets
    from sklearn.model_selection import train_test_split

    dataset = load_dataset("stanfordnlp/imdb")
    combined = concatenate_datasets([dataset['train'], dataset['test']])
    df = combined.to_pandas()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)



import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, dataloader, device, vocab, index_to_word=None, num_samples=3):
    model.eval()
    word_lookup = {v: k for k, v in vocab.items()}
    printed = 0

    with torch.no_grad():
        for inputs, lengths, labels in dataloader:
            inputs, lengths = inputs.to(device), lengths.to(device)
            outputs = model(inputs, lengths)

            if hasattr(model, 'attention_weights'):
                attn_weights = model.attention_weights 

                for i in range(min(num_samples, inputs.size(0))):
                    tokens = [word_lookup.get(tok.item(), '<UNK>') for tok in inputs[i][:lengths[i]]]
                    weights = attn_weights[i][:lengths[i]].cpu().numpy()

                    # Print attention values
                    print(f"\nSample {printed + 1}:")
                    print("Review:", ' '.join(tokens))
                    print("Attention Weights:", weights)

                    # Plot heatmap
                    plt.figure(figsize=(12, 1))
                    sns.heatmap([weights], xticklabels=tokens, cmap="YlGnBu", cbar=True, annot=True)
                    plt.title(f"Attention Heatmap for Sample {printed + 1}")
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks([])
                    plt.tight_layout()
                    plt.show()

                    printed += 1
            if printed >= num_samples:
                break
