import random, math
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Create sample dataset with 100 random sequences
def create_sample_dataset():
    df = pd.read_csv('16S_sequences.csv')
    print(f'Original dataset has {len(df)} sequences')
    
    # Sample 100 random sequences
    sample_df = df.sample(n=100, random_state=42)
    sample_df.to_csv('16S_sequences_sample100.csv', index=False)
    print(f'Created sample dataset with {len(sample_df)} sequences')
    return '16S_sequences_sample100.csv'

class KmerTokenizer:
    def __init__(self, k=4):
        self.k = k
        bases = ['A','C','G','T','N']
        from itertools import product
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
        return toks[:512]  # Truncate to 512 tokens

def load_16s_data(csv_path):
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

def evaluate_model(model, loader, ce, bce, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_tax = 0
    
    with torch.no_grad():
        for padded, src_key_padding_mask, tax_idx, role_idx, novel in loader:
            padded = padded.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            tax_idx = tax_idx.to(device)
            role_idx = role_idx.to(device)
            novel = novel.to(device)
            
            out = model(padded, src_key_padding_mask=src_key_padding_mask)
            loss = ce(out['tax_logits'], tax_idx) + ce(out['role_logits'], role_idx) + 1.5*bce(out['novel_logits'], novel)
            
            total_loss += loss.item() * tax_idx.size(0)
            total_samples += tax_idx.size(0)
            correct_tax += (out['tax_logits'].argmax(dim=1) == tax_idx).sum().item()
    
    avg_loss = total_loss / total_samples
    tax_acc = correct_tax / total_samples
    return avg_loss, tax_acc

# Main execution
if __name__ == "__main__":
    # Create sample dataset
    sample_csv = create_sample_dataset()
    
    # Load data and create model
    tokenizer = KmerTokenizer(k=4)
    records, taxa = load_16s_data(sample_csv)
    print(f'Loaded {len(records)} sequences with {len(taxa)} unique taxa: {taxa}')
    
    # Split data
    split_idx = int(0.8 * len(records))
    train = records[:split_idx]
    val = records[split_idx:]
    print(f'Train: {len(train)} samples, Test: {len(val)} samples')
    
    # Create datasets and loaders
    train_ds = ASVDataset(train, tokenizer)
    val_ds = ASVDataset(val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = MultiTaskTaxonomyModel(
        vocab_size=tokenizer.vocab_size, 
        d_model=64, 
        nhead=2, 
        num_layers=1, 
        tax_classes=len(taxa)
    ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    
    print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')
    print('\nStarting training...')
    print('Epoch | Train Loss | Test Loss | Test Acc')
    print('-' * 40)
    
    # Training loop with loss tracking
    for epoch in range(1, 11):  # 10 epochs
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for padded, src_key_padding_mask, tax_idx, role_idx, novel in train_loader:
            padded = padded.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            tax_idx = tax_idx.to(device)
            role_idx = role_idx.to(device)
            novel = novel.to(device)
            
            out = model(padded, src_key_padding_mask=src_key_padding_mask)
            loss = ce(out['tax_logits'], tax_idx) + ce(out['role_logits'], role_idx) + 1.5*bce(out['novel_logits'], novel)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * tax_idx.size(0)
            train_samples += tax_idx.size(0)
        
        avg_train_loss = train_loss / train_samples
        test_loss, test_acc = evaluate_model(model, val_loader, ce, bce, device)
        
        print(f'{epoch:5d} | {avg_train_loss:10.4f} | {test_loss:9.4f} | {test_acc:8.3f}')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(), 
        'tokenizer_vocab': tokenizer.vocab,
        'taxa': taxa
    }, 'model_16s_sample100.pth')
    
    print(f'\nTraining completed! Model saved as model_16s_sample100.pth')
    print(f'Sample dataset saved as {sample_csv}')