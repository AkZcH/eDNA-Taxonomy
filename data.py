import os
import tarfile
import subprocess
import requests
from Bio import SeqIO
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
db_url = "https://ftp.ncbi.nlm.nih.gov/blast/db/16S_ribosomal_RNA.tar.gz"
db_tar = "16S_ribosomal_RNA.tar.gz"
db_name = "16S_ribosomal_RNA"
output_fasta = "16S_sequences.fasta"
output_csv = "16S_sequences.csv"

# ----------------------------
# Step 1: Download database
# ----------------------------
if not os.path.exists(db_tar):
    print("Downloading BLAST DB...")
    r = requests.get(db_url, stream=True)
    with open(db_tar, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete:", db_tar)
else:
    print("File already exists:", db_tar)

# ----------------------------
# Step 2: Extract .tar.gz
# ----------------------------
if not os.path.exists(db_name + ".nhr"):  # check if already extracted
    print("Extracting database...")
    with tarfile.open(db_tar, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.")
else:
    print("Database already extracted.")

# ----------------------------
# Step 3: Run blastdbcmd to get FASTA
# ----------------------------
if not os.path.exists(output_fasta):
    print("Extracting sequences with blastdbcmd...")
    subprocess.run([
        "blastdbcmd",
        "-db", db_name,
        "-entry", "all",
        "-out", output_fasta
    ])
    print("FASTA extraction complete:", output_fasta)
else:
    print("FASTA already exists:", output_fasta)

# ----------------------------
# Step 4: Parse FASTA -> CSV
# ----------------------------
print("Parsing FASTA into CSV...")
data = []
for record in SeqIO.parse(output_fasta, "fasta"):
    # Example header: gi|12345|ref|NR_123456.1| Homo sapiens 16S ribosomal RNA
    header_parts = record.description.split(" ", 1)
    seq_id = header_parts[0]
    taxonomy = header_parts[1] if len(header_parts) > 1 else "Unknown"
    sequence = str(record.seq)
    data.append([seq_id, taxonomy, sequence])

df = pd.DataFrame(data, columns=["sequence_id", "taxonomy", "sequence"])
df.to_csv(output_csv, index=False)

print("CSV file created:", output_csv)
