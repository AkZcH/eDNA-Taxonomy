import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Any
import json

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
        return toks

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
        emb_t = emb.transpose(0,1)  # [L,B,D]
        enc = self.encoder(emb_t, src_key_padding_mask=src_key_padding_mask)
        enc = enc.transpose(0,1)   # [B,L,D]
        
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask  # True at valid positions
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

class ModelService:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = KmerTokenizer(k=4)
        self.model = self._load_model(model_path)
        self.taxa = ['Phylum_A','Phylum_B','Phylum_C']
        self.roles = ['primary_producer','microbial_grazer','decomposer']
        
    def _load_model(self, model_path: str) -> MultiTaskTaxonomyModel:
        checkpoint = torch.load(model_path, map_location=self.device)
        model = MultiTaskTaxonomyModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=64,
            nhead=2,
            num_layers=1
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def process_sequence(self, sequence: str) -> Dict[str, Any]:
        """Process a single DNA sequence and return predictions."""
        with torch.no_grad():
            # Tokenize sequence
            tokens = self.tokenizer.encode(sequence)
            tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            # Create mask
            mask = torch.ones(1, tokens_tensor.size(1), dtype=torch.bool).to(self.device)
            
            # Get predictions
            outputs = self.model(tokens_tensor, src_key_padding_mask=~mask)
            
            # Process outputs
            tax_pred = self.taxa[outputs['tax_logits'].argmax(dim=1).item()]
            role_pred = self.roles[outputs['role_logits'].argmax(dim=1).item()]
            novel_score = torch.sigmoid(outputs['novel_logits']).item()
            
            # Get confidence scores
            tax_probs = torch.softmax(outputs['tax_logits'], dim=1)[0]
            role_probs = torch.softmax(outputs['role_logits'], dim=1)[0]
            
            return {
                'taxonomy': {
                    'prediction': tax_pred,
                    'confidence': float(tax_probs.max().item()),
                    'probabilities': {
                        taxa: float(prob)
                        for taxa, prob in zip(self.taxa, tax_probs.tolist())
                    }
                },
                'ecological_role': {
                    'prediction': role_pred,
                    'confidence': float(role_probs.max().item()),
                    'probabilities': {
                        role: float(prob)
                        for role, prob in zip(self.roles, role_probs.tolist())
                    }
                },
                'novelty_score': novel_score
            }
    
    def batch_process(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Process multiple sequences in batch."""
        return [self.process_sequence(seq) for seq in sequences]