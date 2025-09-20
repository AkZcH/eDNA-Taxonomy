import json
import torch
import torch.nn as nn
import math
from itertools import product
from Bio import Entrez
import re

# Configure Entrez
Entrez.email = "your.email@example.com"  # Replace with your email

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

def get_taxon_id_from_accession(acc):
    """Given a nucleotide accession, fetch its taxid"""
    try:
        handle = Entrez.esearch(db="nucleotide", term=acc)
        record = Entrez.read(handle)
        handle.close()
        idlist = record["IdList"]
        if len(idlist) == 0:
            return None
        nucl_id = idlist[0]
        summary = Entrez.esummary(db="nucleotide", id=nucl_id)
        srec = Entrez.read(summary)
        summary.close()
        taxid = srec[0].get("TaxId")
        return taxid
    except Exception as e:
        print(f"Error fetching taxid for {acc}: {e}")
        return None

def get_lineage(taxid):
    """Given a taxid, fetch full lineage"""
    try:
        handle = Entrez.efetch(db="taxonomy", id=str(taxid), retmode="xml")
        recs = Entrez.read(handle)
        handle.close()
        if len(recs) == 0:
            return None
        rec = recs[0]
        lineage = rec.get("LineageEx", [])
        result = {}
        for item in lineage:
            rank = item.get("Rank")
            name = item.get("ScientificName")
            if rank in ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                result[rank] = name
        # Include the species itself
        current_rank = rec.get("Rank")
        current_name = rec.get("ScientificName")
        if current_rank in ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            result[current_rank] = current_name
        return result
    except Exception as e:
        print(f"Error fetching lineage for taxid {taxid}: {e}")
        return None

def get_lineage_from_accession(acc):
    """Get full lineage from accession number"""
    taxid = get_taxon_id_from_accession(acc)
    if taxid is None:
        return {}
    lineage = get_lineage(taxid)
    return lineage or {}

def extract_genus_from_taxonomy(taxonomy_string):
    """Extract genus from taxonomy description"""
    # Try to extract genus from strings like "Abyssibius alkaniclasticus strain..."
    words = taxonomy_string.split()
    if len(words) >= 2:
        return words[0]  # First word is usually genus
    return None

def load_model_and_predict():
    # Load the trained model
    checkpoint = torch.load('model_with_taxa.pth', map_location='cpu')
    tokenizer = KmerTokenizer(k=4)
    
    # Get taxa from checkpoint or infer from model size
    if 'taxa' in checkpoint:
        taxa = checkpoint['taxa']
    else:
        # Infer number of taxa from model weights
        tax_head_weight = checkpoint['model_state_dict']['tax_head.weight']
        num_taxa = tax_head_weight.shape[0]
        taxa = [f'Taxa_{i}' for i in range(num_taxa)]
        print(f"Warning: No taxa list found, created {num_taxa} placeholder taxa")
    
    # Comprehensive taxonomy map for common bacterial genera
    taxonomy_map = {
        "Agrobacterium": {
            "superkingdom": "Bacteria",
            "phylum": "Proteobacteria",
            "class": "Alphaproteobacteria",
            "order": "Rhizobiales",
            "family": "Rhizobiaceae",
            "genus": "Agrobacterium",
            "species": "Agrobacterium tumefaciens"
        },
        "Rhizobium": {
            "superkingdom": "Bacteria",
            "phylum": "Proteobacteria",
            "class": "Alphaproteobacteria",
            "order": "Rhizobiales",
            "family": "Rhizobiaceae",
            "genus": "Rhizobium",
            "species": "Rhizobium leguminosarum"
        },
        "Azorhizobium": {
            "superkingdom": "Bacteria",
            "phylum": "Proteobacteria",
            "class": "Alphaproteobacteria",
            "order": "Rhizobiales",
            "family": "Xanthobacteraceae",
            "genus": "Azorhizobium",
            "species": "Azorhizobium caulinodans"
        },
        "Escherichia": {
            "superkingdom": "Bacteria",
            "phylum": "Proteobacteria",
            "class": "Gammaproteobacteria",
            "order": "Enterobacterales",
            "family": "Enterobacteriaceae",
            "genus": "Escherichia",
            "species": "Escherichia coli"
        },
        "Bacillus": {
            "superkingdom": "Bacteria",
            "phylum": "Firmicutes",
            "class": "Bacilli",
            "order": "Bacillales",
            "family": "Bacillaceae",
            "genus": "Bacillus",
            "species": "Bacillus subtilis"
        },
        "Pseudomonas": {
            "superkingdom": "Bacteria",
            "phylum": "Proteobacteria",
            "class": "Gammaproteobacteria",
            "order": "Pseudomonadales",
            "family": "Pseudomonadaceae",
            "genus": "Pseudomonas",
            "species": "Pseudomonas aeruginosa"
        },
        "Staphylococcus": {
            "superkingdom": "Bacteria",
            "phylum": "Firmicutes",
            "class": "Bacilli",
            "order": "Bacillales",
            "family": "Staphylococcaceae",
            "genus": "Staphylococcus",
            "species": "Staphylococcus aureus"
        },
        "Streptococcus": {
            "superkingdom": "Bacteria",
            "phylum": "Firmicutes",
            "class": "Bacilli",
            "order": "Lactobacillales",
            "family": "Streptococcaceae",
            "genus": "Streptococcus",
            "species": "Streptococcus pyogenes"
        }
    }
    
    # Create model with correct number of classes
    model_1 = MultiTaskTaxonomyModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=2,
        num_layers=1,
        tax_classes=len(taxa)
    )
    model_1.load_state_dict(checkpoint['model_state_dict'])
    model_1.eval()
    
    print("16S rRNA Sequence Taxonomy Predictor (JSON Output)")
    print("=" * 40)
    print(f"Model trained on {len(taxa)} bacterial genera")
    print("Enter sequence, accession (acc:NR_189226.1), or 'quit' to exit\n")
    
    while True:
        user_input = input("Enter 16S rRNA sequence or accession: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a valid input.\n")
            continue
        
        try:
            # Check if input is accession
            if user_input.lower().startswith("acc:"):
                accession = user_input.split("acc:")[1].strip()
                sequence_id = accession
                
                # Get lineage from NCBI
                print(f"Fetching lineage for {accession}...")
                ncbi_lineage = get_lineage_from_accession(accession)
                
                if ncbi_lineage:
                    # Use NCBI lineage directly
                    taxonomy = ncbi_lineage.copy()
                    taxonomy["confidence_score"] = 1.0  # Full confidence for NCBI data
                    novel_score = 0.0  # Known sequence
                else:
                    taxonomy = {"genus": "Unknown", "confidence_score": 0.0}
                    novel_score = 1.0
                
                result = {
                    "taxonomic_classification": [
                        {
                            "sequence_id": sequence_id,
                            "predicted_taxonomy": taxonomy,
                            "novel_score": novel_score
                        }
                    ]
                }
                
            else:
                # Process as sequence
                sequence = user_input
                tokens = tokenizer.encode(sequence)
                if not tokens:
                    print("Invalid sequence. Please enter a DNA sequence.\n")
                    continue
                
                input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                
                with torch.no_grad():
                    output = model_1(input_tensor)
                    tax_probs = torch.softmax(output['tax_logits'], dim=1)
                    novel_score = torch.sigmoid(output['novel_logits']).item()
                    
                    top_prob, top_idx = torch.max(tax_probs, dim=1)
                    predicted_genus = taxa[top_idx.item()]
                    confidence = top_prob.item()
                    
                    # Try to get full lineage for predicted genus
                    print(f"Predicted genus: {predicted_genus}, fetching full lineage...")
                    try:
                        # Search for genus in NCBI
                        handle = Entrez.esearch(db="taxonomy", term=f"{predicted_genus}[Scientific Name]")
                        search_record = Entrez.read(handle)
                        handle.close()
                        
                        if search_record["IdList"]:
                            taxid = search_record["IdList"][0]
                            ncbi_lineage = get_lineage(taxid)
                            if ncbi_lineage:
                                taxonomy = ncbi_lineage.copy()
                            else:
                                taxonomy = taxonomy_map.get(predicted_genus, {
                                    "superkingdom": "Bacteria",
                                    "phylum": "Unknown",
                                    "class": "Unknown",
                                    "order": "Unknown",
                                    "family": "Unknown",
                                    "genus": predicted_genus,
                                    "species": "Unknown"
                                })
                        else:
                            taxonomy = taxonomy_map.get(predicted_genus, {
                                "superkingdom": "Bacteria",
                                "phylum": "Unknown",
                                "class": "Unknown",
                                "order": "Unknown",
                                "family": "Unknown",
                                "genus": predicted_genus,
                                "species": "Unknown"
                            })
                    except:
                        taxonomy = taxonomy_map.get(predicted_genus, {
                            "superkingdom": "Bacteria",
                            "phylum": "Unknown",
                            "class": "Unknown",
                            "order": "Unknown",
                            "family": "Unknown",
                            "genus": predicted_genus,
                            "species": "Unknown"
                        })
                    
                    # Ensure confidence_score is in taxonomy
                    taxonomy["confidence_score"] = confidence
                    
                    result = {
                        "taxonomic_classification": [
                            {
                                "sequence_id": "ASV_001",
                                "predicted_taxonomy": taxonomy,
                                "novel_score": novel_score
                            }
                        ]
                    }
            
            # Print JSON nicely
            print(json.dumps(result, indent=2))
            print()
        
        except Exception as e:
            print(f"Error processing input: {e}\n")

if __name__ == "__main__":
    try:
        load_model_and_predict()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required model file and dependencies installed.")
        print("Required: pip install torch biopython")
