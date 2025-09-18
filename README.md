# DeepBioID: eDNA Taxonomy Analysis Platform

DeepBioID is a full-stack web application for analyzing environmental DNA (eDNA) samples and identifying taxonomic classifications using deep learning. The platform provides an intuitive interface for uploading samples, running analyses, and visualizing biodiversity insights.

## Features

- **Sample Management**
  - Upload FASTA/FASTQ sequence files
  - Add metadata (location, environment type, collection date)
  - Track sample processing status

- **Deep Learning Analysis**
  - Multi-task taxonomy classification
  - Confidence scores for predictions
  - Hierarchical taxonomy tree visualization

- **Biodiversity Insights**
  - Shannon diversity index
  - Species richness and evenness metrics
  - Comparative analysis between samples
  - Interactive visualizations

- **Modern Web Interface**
  - Responsive Material-UI design
  - Real-time processing updates
  - Interactive data exploration
  - Comprehensive dashboard

## Technology Stack

- **Backend**
  - Flask (Python web framework)
  - PyTorch (Deep learning)
  - MongoDB (Database)
  - pytest (Testing)

- **Frontend**
  - React
  - Material-UI
  - Recharts (Visualization)
  - Axios (API client)

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- MongoDB 4.4+
- PyTorch with CUDA (optional, for GPU support)

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Create upload directory:
```bash
mkdir uploads
```

5. Run the Flask server:
```bash
python app.py
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

### Running Tests

```bash
# Run backend tests
pytest

# Run frontend tests
cd frontend
npm test
```

## API Documentation

### Sample Management

- `POST /api/samples/upload` - Upload new sample
- `GET /api/samples` - List samples
- `GET /api/samples/<id>` - Get sample details
- `DELETE /api/samples/<id>` - Delete sample

### Analysis

- `POST /api/samples/<id>/analyze` - Run analysis
- `GET /api/samples/<id>/results` - Get analysis results
- `GET /api/taxonomy/<id>/children` - Get taxonomy children
- `POST /api/samples/compare` - Compare multiple samples

### Dashboard

- `GET /api/dashboard/stats` - Get dashboard statistics

## Project Structure

```
eDNA-Taxonomy/
├── app.py                 # Flask application entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── database/             # Database models and utilities
├── services/             # Business logic and ML services
├── routes/               # API route handlers
├── uploads/              # Sample file storage
├── tests/                # Backend tests
└── frontend/            # React frontend application
    ├── src/
    ├── package.json
    └── public/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.