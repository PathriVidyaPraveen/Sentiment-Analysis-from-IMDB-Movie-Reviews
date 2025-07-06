import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attention.bahdanau import BahdanauAttention
from attention.luong_dot import LuongDotAttention
from attention.luong_concat import LuongConcatAttention
from attention.luong_general import LuongGeneralAttention

from models.base_models import VanillaRNN, VanillaLSTM, BidirectionalRNN, BidirectionalLSTM, AttentionClassifier
from utils import load_glove_embeddings, load_imdb_dataset, CustomDataset, visualize_attention

# List of attention-based models only (excluding 'None')
models = ['VanillaRNN', 'VanillaLSTM', 'BidirectionalRNN', 'BidirectionalLSTM']
attentions = ['Bahdanau', 'LuongDot', 'LuongGeneral', 'LuongConcat']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GloVe embeddings and IMDB dataset
print("Loading GloVe...")
embedding_matrix, vocab = load_glove_embeddings("glove.6B.100d.txt")

print("Loading IMDB dataset...")
_, test_df = load_imdb_dataset()
test_dataset = CustomDataset(test_df, vocab)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn)


# Helper to build the model and load weights
def build_model(model_name, attention_name, embedding_matrix, hidden_dim, output_dim, weight_path):
    base_model_cls = {
        'VanillaRNN': VanillaRNN,
        'VanillaLSTM': VanillaLSTM,
        'BidirectionalRNN': BidirectionalRNN,
        'BidirectionalLSTM': BidirectionalLSTM,
    }[model_name]

    base_model = base_model_cls(embedding_matrix, hidden_dim, output_dim)

    attn_input_dim = hidden_dim if 'Vanilla' in model_name else hidden_dim * 2

    attention_cls = {
        'Bahdanau': BahdanauAttention(attn_input_dim, attn_input_dim, attention_dim=64),
        'LuongDot': LuongDotAttention(),
        'LuongGeneral': LuongGeneralAttention(attn_input_dim),
        'LuongConcat': LuongConcatAttention(attn_input_dim, hidden_dim),
    }[attention_name]

    model = AttentionClassifier(base_model, attention_cls, attn_input_dim, output_dim)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device)


# Iterate through all model-attention combinations
for model_name in models:
    for attention_name in attentions:
        print(f"\nVisualizing {model_name} + {attention_name}...")

        weight_path = f"saved_models/{model_name}_{attention_name}.pth"
        if not os.path.exists(weight_path):
            print(f"Weights not found at {weight_path}. Skipping.")
            continue

        model = build_model(model_name, attention_name, embedding_matrix, hidden_dim=128, output_dim=2,weight_path=weight_path)
        
        # Save attention heatmaps
        visualize_attention(model, test_loader, device, vocab, model_name=model_name, attention_name=attention_name, num_samples=3)

print("\nAll attention visualizations saved in 'attention_outputs/'")
