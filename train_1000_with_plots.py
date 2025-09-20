import random, math
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# Set seeds
random.seed(1)
torch.manual_seed(1)

class KmerTokenizer:
    def __init__(self, k=4):
        self.k = k
        bases = ['A','C','G','T','N']
        self.vocab = {'<pad>':0, '<unk>':1}
        idx = 2
        for kmer in map(''.join, product(bases, repeat=self.k)):
            self.vocab[kmer] = idx
            idx += 1
        self.vocab_size = len(self.vocab)
    
    def encode(self, seq):
        seq = seq.upper().replace('U','T')
        toks = []
        for i in range(0, max(1, len(seq)-self.k+1)):
            kmer = seq[i:i+self.k]
            toks.append(self.vocab.get(kmer, self.vocab['<unk>']))
        return toks[:512]

def load_16s_data(csv_path='16S_sequences_1000.csv'):
    df = pd.read_csv(csv_path)
    data = []
    taxa = set()
    for _, row in df.iterrows():
        tax_parts = row['taxonomy'].split()
        genus = tax_parts[0] if tax_parts else 'Unknown'
        taxa.add(genus)
    
    taxa = sorted(list(taxa))
    taxa_to_idx = {t:i for i,t in enumerate(taxa)}
    
    for _, row in df.iterrows():
        tax_parts = row['taxonomy'].split()
        genus = tax_parts[0] if tax_parts else 'Unknown'
        gc = (row['sequence'].count('G') + row['sequence'].count('C')) / len(row['sequence'])
        novel = 1 if gc > 0.7 or gc < 0.3 else 0
        data.append({
            'seq': row['sequence'],
            'tax_idx': taxa_to_idx[genus],
            'role_idx': 0,
            'novel': novel
        })
    return data, taxa

class ASVDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        toks = self.tokenizer.encode(rec['seq'])
        return {
            'tokens': torch.tensor(toks, dtype=torch.long),
            'tax_idx': torch.tensor(rec['tax_idx'], dtype=torch.long),
            'role_idx': torch.tensor(rec['role_idx'], dtype=torch.long),
            'novel': torch.tensor(rec['novel'], dtype=torch.float)
        }

def collate_fn(batch):
    tokens = [b['tokens'] for b in batch]
    lengths = [t.size(0) for t in tokens]
    maxlen = max(lengths)
    padded = torch.zeros(len(tokens), maxlen, dtype=torch.long)
    mask = torch.zeros(len(tokens), maxlen, dtype=torch.bool)
    for i,t in enumerate(tokens):
        padded[i,:t.size(0)] = t
        mask[i,:t.size(0)] = 1
    tax_idx = torch.stack([b['tax_idx'] for b in batch])
    role_idx = torch.stack([b['role_idx'] for b in batch])
    novel = torch.stack([b['novel'] for b in batch])
    src_key_padding_mask = ~mask
    return padded, src_key_padding_mask, tax_idx, role_idx, novel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiTaskTaxonomyModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1, tax_classes=3, role_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.tax_head = nn.Linear(d_model, tax_classes)
        self.role_head = nn.Linear(d_model, role_classes)
        self.novel_head = nn.Linear(d_model, 1)
    
    def forward(self, x, src_key_padding_mask=None):
        emb = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        emb = self.pos(emb)
        emb_t = emb.transpose(0,1)
        enc = self.encoder(emb_t, src_key_padding_mask=src_key_padding_mask)
        enc = enc.transpose(0,1)
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask
            mask = mask.unsqueeze(-1).float()
            enc = enc * mask
            pooled = enc.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = enc.mean(dim=1)
        return {
            'tax_logits': self.tax_head(pooled),
            'role_logits': self.role_head(pooled),
            'novel_logits': self.novel_head(pooled).squeeze(-1)
        }

def train_epoch(model, loader, optimizer, device, ce, bce):
    model.train()
    total_loss = 0
    for batch in loader:
        tokens, mask, tax_idx, role_idx, novel = [x.to(device) for x in batch]
        optimizer.zero_grad()
        out = model(tokens, src_key_padding_mask=mask)
        tax_loss = ce(out['tax_logits'], tax_idx)
        role_loss = ce(out['role_logits'], role_idx)
        novel_loss = bce(out['novel_logits'], novel)
        loss = tax_loss + role_loss + novel_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device, ce, bce):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            tokens, mask, tax_idx, role_idx, novel = [x.to(device) for x in batch]
            out = model(tokens, src_key_padding_mask=mask)
            tax_loss = ce(out['tax_logits'], tax_idx)
            role_loss = ce(out['role_logits'], role_idx)
            novel_loss = bce(out['novel_logits'], novel)
            loss = tax_loss + role_loss + novel_loss
            total_loss += loss.item()
    return total_loss / len(loader)

# Load data
tokenizer = KmerTokenizer(k=4)
records, taxa = load_16s_data('16S_sequences_1000.csv')
print(f'Loaded {len(records)} sequences with {len(taxa)} taxa')

# Split data
split_idx = int(0.8 * len(records))
train_records = records[:split_idx]
val_records = records[split_idx:]

train_ds = ASVDataset(train_records, tokenizer)
val_ds = ASVDataset(val_records, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = MultiTaskTaxonomyModel(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    nhead=2,
    num_layers=2,
    tax_classes=len(taxa)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
bce = nn.BCEWithLogitsLoss()

# Training loop
epochs = 15
train_losses = []
val_losses = []

print("Starting training...")
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device, ce, bce)
    val_loss = validate(model, val_loader, device, ce, bce)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(range(1, epochs+1), val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time (1000 Samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_loss_1000.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'taxa': taxa,
    'tokenizer_vocab': tokenizer.vocab,
    'train_losses': train_losses,
    'val_losses': val_losses
}, 'model_16s_1000.pth')

print("Training completed!")
print("Model saved as 'model_16s_1000.pth'")
print("Loss plot saved as 'training_loss_1000.png'")