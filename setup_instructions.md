# Setup Instructions

## Required Files (Download Separately)

Due to GitHub file size limits, download these files separately:

### 1. Training Data
- **File**: `16S_sequences.csv` 
- **Size**: ~50MB
- **Required for**: Training the model
- **Download**: [Add your download link here]

### 2. Pre-trained Model
- **File**: `model_with_taxa.pth`
- **Size**: ~10MB  
- **Required for**: Running predictions
- **Download**: [Add your download link here]

## Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AkZcH/eDNA-Taxonomy.git
   cd eDNA-Taxonomy
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required files**:
   - Place `16S_sequences.csv` in the root directory
   - Place `model_with_taxa.pth` in the root directory

4. **Run the model**:
   ```bash
   python predict_taxonomy.py
   ```

## Alternative: Train Your Own Model

If you can't download the pre-trained model:

1. Get your own 16S rRNA data in CSV format with columns: `sequence_id`, `taxonomy`, `sequence`
2. Run the training notebook: `model_fixed.ipynb`
3. This will create `model_with_taxa.pth`

## File Structure
```
eDNA-Taxonomy/
├── predict_taxonomy.py
├── model_fixed.ipynb
├── 16S_sequences.csv          # Download separately
├── model_with_taxa.pth        # Download separately  
├── requirements.txt
└── README.md
```