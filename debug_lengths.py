import pandas as pd
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
        print(f"Original length: {len(toks)}, Truncated length: {len(toks[:512])}")
        return toks[:512]

tokenizer = KmerTokenizer(k=4)
df = pd.read_csv('16S_sequences.csv')
for i, row in df.iterrows():
    print(f"Sequence {i+1}:")
    tokens = tokenizer.encode(row['sequence'])
    print(f"Final token count: {len(tokens)}")
    print()