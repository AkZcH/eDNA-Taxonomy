import pandas as pd
import random
from Bio import SeqIO

def create_1000_sample_dataset():
    # Load original dataset
    df = pd.read_csv('16S_sequences.csv')
    
    # If we have fewer than 1000, we'll duplicate and modify sequences
    original_count = len(df)
    print(f"Original dataset has {original_count} sequences")
    
    new_data = []
    
    # Add all original sequences
    for _, row in df.iterrows():
        new_data.append({
            'sequence_id': row['sequence_id'],
            'taxonomy': row['taxonomy'],
            'sequence': row['sequence']
        })
    
    # Generate additional sequences by modifying existing ones
    target_count = 1000
    while len(new_data) < target_count:
        # Pick a random original sequence
        base_row = df.iloc[random.randint(0, len(df)-1)]
        
        # Create variations
        sequence = base_row['sequence']
        
        # Method 1: Random mutations (substitute random bases)
        if len(new_data) < target_count:
            mutated_seq = list(sequence)
            num_mutations = random.randint(1, min(10, len(sequence)//50))
            for _ in range(num_mutations):
                pos = random.randint(0, len(mutated_seq)-1)
                mutated_seq[pos] = random.choice(['A', 'T', 'G', 'C'])
            
            new_data.append({
                'sequence_id': f"{base_row['sequence_id']}_mut_{len(new_data)}",
                'taxonomy': base_row['taxonomy'],
                'sequence': ''.join(mutated_seq)
            })
        
        # Method 2: Subsequences (take random portions)
        if len(new_data) < target_count and len(sequence) > 200:
            start = random.randint(0, len(sequence)//4)
            end = random.randint(len(sequence)*3//4, len(sequence))
            subseq = sequence[start:end]
            
            new_data.append({
                'sequence_id': f"{base_row['sequence_id']}_sub_{len(new_data)}",
                'taxonomy': base_row['taxonomy'],
                'sequence': subseq
            })
        
        # Method 3: Add random insertions/deletions
        if len(new_data) < target_count:
            modified_seq = list(sequence)
            # Random deletion
            if len(modified_seq) > 100 and random.random() < 0.5:
                del_pos = random.randint(0, len(modified_seq)-1)
                del modified_seq[del_pos]
            
            # Random insertion
            if random.random() < 0.5:
                ins_pos = random.randint(0, len(modified_seq))
                modified_seq.insert(ins_pos, random.choice(['A', 'T', 'G', 'C']))
            
            new_data.append({
                'sequence_id': f"{base_row['sequence_id']}_indel_{len(new_data)}",
                'taxonomy': base_row['taxonomy'],
                'sequence': ''.join(modified_seq)
            })
    
    # Trim to exactly 1000
    new_data = new_data[:1000]
    
    # Create DataFrame and save
    new_df = pd.DataFrame(new_data)
    new_df.to_csv('16S_sequences_1000.csv', index=False)
    
    print(f"Created dataset with {len(new_df)} sequences")
    print(f"Unique taxa: {len(new_df['taxonomy'].apply(lambda x: x.split()[0]).unique())}")
    
    return new_df

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    create_1000_sample_dataset()