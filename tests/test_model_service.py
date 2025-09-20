import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from services.model_service import ModelService, KmerTokenizer

@pytest.fixture
def model_service():
    return ModelService()

@pytest.fixture
def tokenizer():
    return KmerTokenizer(k=6)

@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock(return_value=None)
    model.forward = Mock(return_value=(
        torch.tensor([[0.8, 0.2], [0.6, 0.4]]),  # Kingdom predictions
        torch.tensor([[0.7, 0.3], [0.5, 0.5]]),  # Phylum predictions
        torch.tensor([[0.9, 0.1], [0.4, 0.6]])   # Class predictions
    ))
    return model

def test_kmer_tokenizer_encode(tokenizer):
    sequence = "ATCGATCG"
    tokens = tokenizer.encode(sequence)
    
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) == len(sequence) - tokenizer.k + 1

def test_kmer_tokenizer_decode(tokenizer):
    sequence = "ATCGATCG"
    tokens = tokenizer.encode(sequence)
    decoded = tokenizer.decode(tokens)
    
    assert isinstance(decoded, str)
    assert len(decoded) == tokenizer.k * len(tokens)

def test_process_sequence(model_service, mock_model):
    with patch.object(model_service, '_load_model', return_value=mock_model):
        sequence = "ATCGATCGATCG"
        predictions = model_service.process_sequence(sequence)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 3  # Kingdom, Phylum, Class
        assert all(isinstance(p, tuple) for p in predictions)
        assert all(isinstance(conf, float) for _, conf in predictions)

def test_batch_processing(model_service, mock_model):
    with patch.object(model_service, '_load_model', return_value=mock_model):
        sequences = [
            "ATCGATCGATCG",
            "GCTAGCTAGCTA"
        ]
        results = model_service.process_batch(sequences)
        
        assert isinstance(results, list)
        assert len(results) == len(sequences)
        assert all(isinstance(r, list) for r in results)

def test_calculate_biodiversity_metrics(model_service):
    predictions = [
        ('Bacteria', 0.9),
        ('Proteobacteria', 0.8),
        ('Gammaproteobacteria', 0.7)
    ]
    
    metrics = model_service.calculate_biodiversity_metrics(predictions)
    
    assert isinstance(metrics, dict)
    assert 'shannon_index' in metrics
    assert 'species_richness' in metrics
    assert 'species_evenness' in metrics
    assert isinstance(metrics['shannon_index'], float)
    assert isinstance(metrics['species_richness'], int)
    assert isinstance(metrics['species_evenness'], float)

def test_compare_samples(model_service):
    sample_results = [
        {
            'taxonomy_tree': [
                {'name': 'Bacteria', 'confidence': 0.9},
                {'name': 'Proteobacteria', 'confidence': 0.8}
            ],
            'biodiversity_metrics': {
                'shannon_index': 1.5,
                'species_richness': 10
            }
        },
        {
            'taxonomy_tree': [
                {'name': 'Bacteria', 'confidence': 0.85},
                {'name': 'Firmicutes', 'confidence': 0.75}
            ],
            'biodiversity_metrics': {
                'shannon_index': 1.3,
                'species_richness': 8
            }
        }
    ]
    
    comparison = model_service.compare_samples(sample_results)
    
    assert isinstance(comparison, dict)
    assert 'similarity_matrix' in comparison
    assert 'common_taxa' in comparison
    assert 'unique_taxa' in comparison
    
    similarity_matrix = np.array(comparison['similarity_matrix'])
    assert similarity_matrix.shape == (2, 2)
    assert np.allclose(similarity_matrix, similarity_matrix.T)
    assert np.allclose(np.diag(similarity_matrix), 1.0)

def test_invalid_sequence(model_service, mock_model):
    with patch.object(model_service, '_load_model', return_value=mock_model):
        with pytest.raises(ValueError):
            model_service.process_sequence("")
        
        with pytest.raises(ValueError):
            model_service.process_sequence("INVALID")

def test_model_loading_error(model_service):
    with patch('torch.load', side_effect=Exception("Model loading error")):
        with pytest.raises(Exception):
            model_service._load_model()

def test_sequence_preprocessing(model_service):
    sequence = "ATCG\nGCTA\n"
    processed = model_service._preprocess_sequence(sequence)
    
    assert isinstance(processed, str)
    assert "\n" not in processed
    assert processed == "ATCGGCTA"

def test_confidence_threshold(model_service, mock_model):
    with patch.object(model_service, '_load_model', return_value=mock_model):
        sequence = "ATCGATCGATCG"
        predictions = model_service.process_sequence(
            sequence,
            confidence_threshold=0.7
        )
        
        assert all(conf >= 0.7 for _, conf in predictions)

def test_taxonomy_tree_construction(model_service):
    predictions = [
        ('Bacteria', 0.9),
        ('Proteobacteria', 0.8),
        ('Gammaproteobacteria', 0.7)
    ]
    
    tree = model_service._construct_taxonomy_tree(predictions)
    
    assert isinstance(tree, list)
    assert len(tree) == len(predictions)
    assert all('id' in node for node in tree)
    assert all('parent_id' in node for node in tree)
    assert tree[0]['parent_id'] is None  # Root node