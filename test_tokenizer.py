from itertools import product
import pandas as pd

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

tokenizer = KmerTokenizer(k=4)
df = pd.read_csv('16S_sequences.csv')
lengths = [len(tokenizer.encode(row['sequence'])) for _, row in df.iterrows()]
print(f'Tokenized lengths: {lengths}')
print(f'Max length: {max(lengths)}')