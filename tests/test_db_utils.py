import pytest
from datetime import datetime
from bson import ObjectId
from unittest.mock import Mock, patch
from database.db_utils import DatabaseManager
from database.schemas import Sample, SampleMetadata, AnalysisResult

@pytest.fixture
def db_manager():
    return DatabaseManager()

@pytest.fixture
def mock_collection():
    collection = Mock()
    collection.insert_one = Mock(return_value=Mock(inserted_id=ObjectId()))
    collection.find_one = Mock(return_value={
        '_id': ObjectId(),
        'filename': 'test.fasta',
        'status': 'pending',
        'metadata': {
            'location': 'Test Location',
            'environment_type': 'Marine',
            'collection_date': datetime.utcnow(),
            'tags': ['test']
        }
    })
    return collection

@pytest.fixture
def sample_data():
    return Sample(
        id=ObjectId(),
        filename='test.fasta',
        file_path='/path/to/test.fasta',
        sequence='ATCG',
        metadata=SampleMetadata(
            location='Test Location',
            environment_type='Marine',
            collection_date=datetime.utcnow(),
            tags=['test'],
            notes='Test notes'
        ),
        status='pending',
        upload_date=datetime.utcnow()
    )

def test_insert_sample(db_manager, mock_collection, sample_data):
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        result = db_manager.insert_sample(sample_data.dict())
        
        assert isinstance(result, ObjectId)
        mock_collection.insert_one.assert_called_once()

def test_get_sample(db_manager, mock_collection):
    sample_id = ObjectId()
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        sample = db_manager.get_sample(sample_id)
        
        assert sample is not None
        assert '_id' in sample
        assert 'filename' in sample
        assert 'metadata' in sample
        mock_collection.find_one.assert_called_with({'_id': sample_id})

def test_list_samples(db_manager, mock_collection):
    mock_collection.find = Mock(return_value=[
        {
            '_id': ObjectId(),
            'filename': 'test1.fasta',
            'status': 'pending'
        },
        {
            '_id': ObjectId(),
            'filename': 'test2.fasta',
            'status': 'completed'
        }
    ])
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        samples = db_manager.list_samples()
        
        assert isinstance(samples, list)
        assert len(samples) == 2
        assert all('_id' in s for s in samples)

def test_update_sample_status(db_manager, mock_collection):
    sample_id = ObjectId()
    mock_collection.update_one = Mock(return_value=Mock(modified_count=1))
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        result = db_manager.update_sample_status(sample_id, 'completed')
        
        assert result is True
        mock_collection.update_one.assert_called_with(
            {'_id': sample_id},
            {'$set': {'status': 'completed'}}
        )

def test_delete_sample(db_manager, mock_collection):
    sample_id = ObjectId()
    mock_collection.delete_one = Mock(return_value=Mock(deleted_count=1))
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        result = db_manager.delete_sample(sample_id)
        
        assert result is True
        mock_collection.delete_one.assert_called_with({'_id': sample_id})

def test_insert_analysis_result(db_manager, mock_collection):
    result = AnalysisResult(
        id=ObjectId(),
        sample_id=ObjectId(),
        taxonomy_tree=[],
        biodiversity_metrics={},
        analysis_date=datetime.utcnow()
    )
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        result_id = db_manager.insert_analysis_result(result.dict())
        
        assert isinstance(result_id, ObjectId)
        mock_collection.insert_one.assert_called_once()

def test_get_analysis_results(db_manager, mock_collection):
    sample_id = ObjectId()
    mock_collection.find_one = Mock(return_value={
        '_id': ObjectId(),
        'sample_id': sample_id,
        'taxonomy_tree': [],
        'biodiversity_metrics': {}
    })
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        results = db_manager.get_analysis_results(sample_id)
        
        assert results is not None
        assert 'taxonomy_tree' in results
        assert 'biodiversity_metrics' in results

def test_get_taxonomy_children(db_manager, mock_collection):
    node_id = ObjectId()
    mock_collection.find = Mock(return_value=[
        {
            '_id': ObjectId(),
            'name': 'Child1',
            'parent_id': node_id
        },
        {
            '_id': ObjectId(),
            'name': 'Child2',
            'parent_id': node_id
        }
    ])
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        children = db_manager.get_taxonomy_children(node_id)
        
        assert isinstance(children, list)
        assert len(children) == 2
        assert all('parent_id' in c for c in children)

def test_get_dashboard_stats(db_manager, mock_collection):
    mock_collection.count_documents = Mock(return_value=100)
    mock_collection.aggregate = Mock(return_value=[
        {'_id': 'Bacteria', 'count': 50},
        {'_id': 'Archaea', 'count': 30}
    ])
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        stats = db_manager.get_dashboard_stats()
        
        assert isinstance(stats, dict)
        assert 'total_samples' in stats
        assert 'taxonomy_distribution' in stats
        assert 'recent_activity' in stats

def test_sample_filtering(db_manager, mock_collection):
    filters = {
        'status': 'pending',
        'metadata.location': 'Test Location'
    }
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        samples = db_manager.list_samples(filters=filters)
        
        mock_collection.find.assert_called_with(filters)

def test_pagination(db_manager, mock_collection):
    page = 2
    per_page = 10
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        samples = db_manager.list_samples(page=page, per_page=per_page)
        
        mock_collection.find.assert_called_with(
            {},
            skip=(page - 1) * per_page,
            limit=per_page
        )

def test_error_handling(db_manager, mock_collection):
    mock_collection.find_one.side_effect = Exception("Database error")
    
    with patch.object(db_manager, '_get_collection', return_value=mock_collection):
        with pytest.raises(Exception):
            db_manager.get_sample(ObjectId())