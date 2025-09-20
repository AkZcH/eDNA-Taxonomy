from flask import Blueprint, request, jsonify
from bson import ObjectId
from datetime import datetime
from services.model_service import ModelService
from database.db_utils import DatabaseManager
from database.schemas import Sample, AnalysisResult, TaxonomyNode

analysis_bp = Blueprint('analysis', __name__)
db = DatabaseManager()
model_service = ModelService()

@analysis_bp.route('/samples/<sample_id>/analyze', methods=['POST'])
async def analyze_sample(sample_id):
    try:
        # Retrieve sample from database
        sample = await db.get_sample(ObjectId(sample_id))
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404

        # Process sample with model
        sequence = sample['sequence']
        predictions = await model_service.process_sequence(sequence)

        # Create taxonomy tree
        taxonomy_tree = []
        for level, (taxon, confidence) in enumerate(predictions):
            node = TaxonomyNode(
                id=str(ObjectId()),
                name=taxon,
                level=level,
                confidence=confidence,
                parent_id=taxonomy_tree[-1]['id'] if taxonomy_tree else None,
                has_children=level < len(predictions) - 1
            )
            taxonomy_tree.append(node.dict())

        # Calculate biodiversity metrics
        biodiversity_metrics = await model_service.calculate_biodiversity_metrics(predictions)

        # Create analysis result
        result = AnalysisResult(
            sample_id=ObjectId(sample_id),
            taxonomy_tree=taxonomy_tree,
            biodiversity_metrics=biodiversity_metrics,
            analysis_date=datetime.utcnow()
        )

        # Save result to database
        await db.insert_analysis_result(result.dict())

        return jsonify({
            'status': 'success',
            'message': 'Analysis completed successfully',
            'result_id': str(result.id)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/samples/<sample_id>/results', methods=['GET'])
async def get_sample_results(sample_id):
    try:
        # Get analysis results
        results = await db.get_analysis_results(ObjectId(sample_id))
        if not results:
            return jsonify({'error': 'No results found'}), 404

        # Get sample metadata
        sample = await db.get_sample(ObjectId(sample_id))
        
        # Prepare response
        response = {
            'metadata': {
                'collectionDate': sample['collection_date'],
                'location': sample['location'],
                'environmentType': sample['environment_type'],
                'tags': sample['tags']
            },
            'taxonomy_tree': results['taxonomy_tree'],
            'biodiversity': results['biodiversity_metrics'],
            'summary': {
                'taxonomicLevels': [
                    {'level': node['level'], 'count': 1}
                    for node in results['taxonomy_tree']
                ]
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/taxonomy/<node_id>/children', methods=['GET'])
async def get_taxonomy_children(node_id):
    try:
        # Get children nodes
        children = await db.get_taxonomy_children(ObjectId(node_id))
        return jsonify([child.dict() for child in children])

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/samples/compare', methods=['POST'])
async def compare_samples():
    try:
        data = request.get_json()
        sample_ids = [ObjectId(id) for id in data['sample_ids']]

        # Get analysis results for all samples
        results = []
        for sample_id in sample_ids:
            result = await db.get_analysis_results(sample_id)
            if result:
                results.append(result)

        if not results:
            return jsonify({'error': 'No results found for comparison'}), 404

        # Perform comparative analysis
        comparison = await model_service.compare_samples(results)

        # Save comparison results
        comparison_id = await db.insert_comparative_analysis({
            'sample_ids': sample_ids,
            'comparison_results': comparison,
            'created_at': datetime.utcnow()
        })

        return jsonify({
            'status': 'success',
            'comparison_id': str(comparison_id),
            'results': comparison
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/dashboard/stats', methods=['GET'])
async def get_dashboard_stats():
    try:
        # Get overall statistics
        total_samples = await db.count_samples()
        processed_samples = await db.count_processed_samples()
        
        # Get taxonomy distribution
        taxonomy_dist = await db.get_taxonomy_distribution()
        
        # Get role distribution
        role_dist = await db.get_role_distribution()
        
        # Get recent activity
        recent_activity = await db.get_recent_activity()

        return jsonify({
            'totalSamples': total_samples,
            'processedSamples': processed_samples,
            'taxonomyDistribution': taxonomy_dist,
            'roleDistribution': role_dist,
            'recentActivity': recent_activity
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500