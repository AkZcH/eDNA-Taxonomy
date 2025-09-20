# Quick training with small dataset
import random, math, time
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from itertools import product

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

def load_16s_data_small(csv_path='16S_sequences.csv', max_samples=100):
    df = pd.read_csv(csv_path).head(max_samples)  # Only first 100 samples
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

# Quick setup
tokenizer = KmerTokenizer(k=4)
records, taxa = load_16s_data_small('16S_sequences.csv', max_samples=100)
print(f'Loaded {len(records)} sequences with {len(taxa)} taxa')

split_idx = int(0.8 * len(records))
train = records[:split_idx]
val = records[split_idx:]

train_ds = ASVDataset(train, tokenizer)
val_ds = ASVDataset(val, tokenizer)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = MultiTaskTaxonomyModel(vocab_size=tokenizer.vocab_size, d_model=64, nhead=2, num_layers=1, tax_classes=len(taxa)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
bce = nn.BCEWithLogitsLoss()

# Quick training
print("Starting quick training...")
for epoch in range(1, 6):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for padded, src_key_padding_mask, tax_idx, role_idx, novel in train_loader:
        padded = padded.to(device); src_key_padding_mask = src_key_padding_mask.to(device)
        tax_idx = tax_idx.to(device); role_idx = role_idx.to(device); novel = novel.to(device)
        
        out = model(padded, src_key_padding_mask=src_key_padding_mask)
        loss = ce(out['tax_logits'], tax_idx) + ce(out['role_logits'], role_idx) + bce(out['novel_logits'], novel)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for padded, src_key_padding_mask, tax_idx, role_idx, novel in val_loader:
            padded = padded.to(device); src_key_padding_mask = src_key_padding_mask.to(device)
            tax_idx = tax_idx.to(device); role_idx = role_idx.to(device); novel = novel.to(device)
            out = model(padded, src_key_padding_mask=src_key_padding_mask)
            loss = ce(out['tax_logits'], tax_idx) + ce(out['role_logits'], role_idx) + bce(out['novel_logits'], novel)
            val_loss += loss.item()
            correct += (out['tax_logits'].argmax(dim=1) == tax_idx).sum().item()
            total += tax_idx.size(0)
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch}: Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Acc: {correct/total:.3f}, Time: {epoch_time:.1f}s")

print("Quick training completed!")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'taxa': taxa,
    'tokenizer_vocab': tokenizer.vocab
}, 'model_16s_quick.pth')
print("Model saved as 'model_16s_quick.pth'")