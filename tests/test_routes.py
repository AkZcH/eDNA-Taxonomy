import pytest
import json
from datetime import datetime
from bson import ObjectId
from unittest.mock import Mock, patch
from app import create_app
from database.db_utils import DatabaseManager
from services.model_service import ModelService

@pytest.fixture
def app():
    app = create_app(testing=True)
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def mock_db():
    with patch('database.db_utils.DatabaseManager') as mock:
        yield mock

@pytest.fixture
def mock_model_service():
    with patch('services.model_service.ModelService') as mock:
        yield mock

def test_upload_sample(client):
    data = {
        'file': (open('tests/data/test_sample.fasta', 'rb'), 'test_sample.fasta'),
        'location': 'Test Location',
        'environmentType': 'Marine',
        'collectionDate': '2024-03-20',
        'tags': 'test,sample'
    }
    
    response = client.post('/api/samples/upload', data=data)
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert result['status'] == 'success'
    assert 'sample_id' in result

def test_upload_sample_no_file(client):
    data = {
        'location': 'Test Location',
        'environmentType': 'Marine',
        'collectionDate': '2024-03-20'
    }
    
    response = client.post('/api/samples/upload', data=data)
    assert response.status_code == 400
    
    result = json.loads(response.data)
    assert 'error' in result

def test_list_samples(client, mock_db):
    mock_samples = [
        {
            '_id': ObjectId(),
            'filename': 'test1.fasta',
            'status': 'pending',
            'metadata': {
                'location': 'Location 1',
                'environment_type': 'Marine',
                'collection_date': datetime.utcnow(),
                'tags': ['test']
            },
            'upload_date': datetime.utcnow()
        }
    ]
    
    mock_db.list_samples.return_value = mock_samples
    mock_db.count_samples.return_value = len(mock_samples)
    
    response = client.get('/api/samples')
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert 'samples' in result
    assert 'pagination' in result
    assert len(result['samples']) == len(mock_samples)

def test_analyze_sample(client, mock_db, mock_model_service):
    sample_id = str(ObjectId())
    mock_sample = {
        '_id': ObjectId(sample_id),
        'sequence': 'ATCG',
        'status': 'pending'
    }
    
    mock_predictions = [
        ('Bacteria', 0.95),
        ('Proteobacteria', 0.85)
    ]
    
    mock_db.get_sample.return_value = mock_sample
    mock_model_service.process_sequence.return_value = mock_predictions
    
    response = client.post(f'/api/samples/{sample_id}/analyze')
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert result['status'] == 'success'
    assert 'result_id' in result

def test_get_sample_results(client, mock_db):
    sample_id = str(ObjectId())
    mock_results = {
        'taxonomy_tree': [
            {
                'id': str(ObjectId()),
                'name': 'Bacteria',
                'level': 0,
                'confidence': 0.95
            }
        ],
        'biodiversity_metrics': {
            'shannon_index': 1.5,
            'species_richness': 10
        }
    }
    
    mock_sample = {
        'collection_date': datetime.utcnow(),
        'location': 'Test Location',
        'environment_type': 'Marine',
        'tags': ['test']
    }
    
    mock_db.get_analysis_results.return_value = mock_results
    mock_db.get_sample.return_value = mock_sample
    
    response = client.get(f'/api/samples/{sample_id}/results')
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert 'metadata' in result
    assert 'taxonomy_tree' in result
    assert 'biodiversity' in result

def test_compare_samples(client, mock_db, mock_model_service):
    sample_ids = [str(ObjectId()), str(ObjectId())]
    mock_results = [
        {
            'taxonomy_tree': [],
            'biodiversity_metrics': {}
        }
    ]
    
    mock_comparison = {
        'similarity_matrix': [[1.0, 0.8], [0.8, 1.0]],
        'common_taxa': ['Bacteria']
    }
    
    mock_db.get_analysis_results.return_value = mock_results[0]
    mock_model_service.compare_samples.return_value = mock_comparison
    
    response = client.post('/api/samples/compare', 
                          json={'sample_ids': sample_ids})
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert result['status'] == 'success'
    assert 'comparison_id' in result
    assert 'results' in result

def test_get_dashboard_stats(client, mock_db):
    mock_stats = {
        'total_samples': 100,
        'processed_samples': 80,
        'taxonomy_distribution': [
            {'name': 'Bacteria', 'value': 50}
        ],
        'role_distribution': [
            {'name': 'Producer', 'value': 30}
        ],
        'recent_activity': [
            {'date': '2024-03', 'samples': 20}
        ]
    }
    
    mock_db.count_samples.return_value = mock_stats['total_samples']
    mock_db.count_processed_samples.return_value = mock_stats['processed_samples']
    mock_db.get_taxonomy_distribution.return_value = mock_stats['taxonomy_distribution']
    mock_db.get_role_distribution.return_value = mock_stats['role_distribution']
    mock_db.get_recent_activity.return_value = mock_stats['recent_activity']
    
    response = client.get('/api/dashboard/stats')
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert 'totalSamples' in result
    assert 'processedSamples' in result
    assert 'taxonomyDistribution' in result
    assert 'roleDistribution' in result
    assert 'recentActivity' in result