import os
import pytest
import tempfile

@pytest.fixture
def test_config():
    # Create a temporary directory for test uploads
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'TESTING': True,
            'UPLOAD_FOLDER': temp_dir,
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
            'MONGODB_URI': 'mongodb+srv://ritamvaskar0:Ritam2005@cluster0.lklwwgr.mongodb.net//test_edna',
            'MODEL_PATH': os.path.join(os.path.dirname(__file__), 'data', 'test_model.pth')
        }
        yield config

@pytest.fixture
def test_sample_data():
    return """>
ATCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG
ATCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG
"""

@pytest.fixture
def test_sample_file(tmp_path):
    # Create a test FASTA file
    file_path = tmp_path / "test_sample.fasta"
    with open(file_path, "w") as f:
        f.write(test_sample_data())
    return file_path

@pytest.fixture
def test_metadata():
    return {
        'location': 'Test River',
        'environment_type': 'Freshwater',
        'collection_date': '2024-03-20',
        'tags': ['test', 'sample'],
        'notes': 'Test sample for unit testing'
    }

@pytest.fixture
def test_taxonomy_tree():
    return [
        {
            'id': '507f1f77bcf86cd799439011',
            'name': 'Bacteria',
            'level': 0,
            'confidence': 0.95,
            'parent_id': None,
            'has_children': True
        },
        {
            'id': '507f1f77bcf86cd799439012',
            'name': 'Proteobacteria',
            'level': 1,
            'confidence': 0.85,
            'parent_id': '507f1f77bcf86cd799439011',
            'has_children': True
        },
        {
            'id': '507f1f77bcf86cd799439013',
            'name': 'Gammaproteobacteria',
            'level': 2,
            'confidence': 0.75,
            'parent_id': '507f1f77bcf86cd799439012',
            'has_children': False
        }
    ]

@pytest.fixture
def test_biodiversity_metrics():
    return {
        'shannon_index': 2.5,
        'species_richness': 15,
        'species_evenness': 0.85,
        'dominant_taxa': [
            {'name': 'Bacteria', 'abundance': 0.4},
            {'name': 'Proteobacteria', 'abundance': 0.3},
            {'name': 'Gammaproteobacteria', 'abundance': 0.2}
        ]
    }

@pytest.fixture
def test_comparative_analysis():
    return {
        'similarity_matrix': [
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ],
        'common_taxa': [
            'Bacteria',
            'Proteobacteria'
        ],
        'unique_taxa': {
            'sample1': ['Gammaproteobacteria'],
            'sample2': ['Betaproteobacteria'],
            'sample3': ['Alphaproteobacteria']
        },
        'diversity_comparison': {
            'shannon_index': [2.5, 2.3, 2.1],
            'species_richness': [15, 12, 10],
            'species_evenness': [0.85, 0.80, 0.75]
        }
    }