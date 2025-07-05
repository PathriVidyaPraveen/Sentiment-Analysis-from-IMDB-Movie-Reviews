import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from attention.bahdanau import BahdanauAttention
from attention.luong_dot import LuongDotAttention
from attention.luong_concat import LuongConcatAttention
from attention.luong_general import LuongGeneralAttention

from models.base_models import VanillaRNN, VanillaLSTM, BidirectionalRNN, BidirectionalLSTM,AttentionClassifier

from utils import load_glove_embeddings, load_imdb_dataset, CustomDataset, visualize_attention


def build_model(model_name, attention_name, embedding_matrix, hidden_dim, output_dim):
    base_model_cls = {
        'VanillaRNN': VanillaRNN,
        'VanillaLSTM': VanillaLSTM,
        'BidirectionalRNN': BidirectionalRNN,
        'BidirectionalLSTM': BidirectionalLSTM,
    }[model_name]

    base_model = base_model_cls(embedding_matrix, hidden_dim, output_dim)

    if attention_name == 'None':
        return base_model

    attention_cls = {
        'Bahdanau': BahdanauAttention,
        'LuongDot': LuongDotAttention,
        'LuongGeneral': LuongGeneralAttention,
        'LuongConcat': LuongConcatAttention,
    }[attention_name]

    attn_input_dim = hidden_dim if 'Vanilla' in model_name else hidden_dim * 2
    attention = attention_cls(attn_input_dim, hidden_dim)
    return AttentionClassifier(base_model, attention, attn_input_dim, output_dim)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, lengths, labels in dataloader:
        inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, lengths, labels in dataloader:
            inputs, lengths = inputs.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prec, rec, f1, cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['VanillaRNN', 'VanillaLSTM', 'BidirectionalRNN', 'BidirectionalLSTM'])
    parser.add_argument('--attention', type=str, default='None',
                        choices=['None', 'Bahdanau', 'LuongDot', 'LuongGeneral', 'LuongConcat'])
    parser.add_argument('--glove_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading GloVe from {args.glove_path}...")
    embedding_matrix, vocab = load_glove_embeddings(args.glove_path)

    print("Loading IMDB dataset...")
    train_df, test_df = load_imdb_dataset()
    train_dataset = CustomDataset(train_df, vocab)
    test_dataset = CustomDataset(test_df, vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    model = build_model(args.model, args.attention, embedding_matrix, hidden_dim=128, output_dim=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
    # Save model after final epoch
    model_path = f"saved_models/{args.model}_{args.attention}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    acc, prec, rec, f1, cm = evaluate(model, test_loader, device)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    if args.attention != 'None':
        visualize_attention(model, test_loader, device, vocab, model_name=args.model, attention_name=args.attention)

