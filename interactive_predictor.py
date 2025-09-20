import torch
import torch.nn as nn
import math
from itertools import product

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

# Load model
checkpoint = torch.load('model_16s_sample100.pth', map_location='cpu')
tokenizer = KmerTokenizer(k=4)
taxa = checkpoint['taxa']

model = MultiTaskTaxonomyModel(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    nhead=2,
    num_layers=1,
    tax_classes=len(taxa)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("16S rRNA Taxonomy Predictor")
print("=" * 40)
print(f"Model trained on {len(taxa)} bacterial genera")
print("Enter 'quit' to exit\n")

while True:
    sequence = input("Enter 16S rRNA sequence: ").strip()
    
    if sequence.lower() == 'quit':
        break
    
    if not sequence:
        continue
    
    tokens = tokenizer.encode(sequence)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        tax_probs = torch.softmax(output['tax_logits'], dim=1)
        novel_score = torch.sigmoid(output['novel_logits']).item()
        
        top_probs, top_indices = torch.topk(tax_probs, k=3)
        
        print(f"\nSequence length: {len(sequence)} bp")
        print(f"Novel score: {novel_score:.3f}")
        print("Top 3 Predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            genus = taxa[idx.item()]
            confidence = prob.item() * 100
            print(f"{i+1}. {genus} ({confidence:.1f}%)")
        print()