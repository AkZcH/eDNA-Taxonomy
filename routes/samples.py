from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from bson import ObjectId
from datetime import datetime
import os
import uuid

from database.db_utils import DatabaseManager
from database.schemas import Sample, SampleMetadata

samples_bp = Blueprint('samples', __name__)
db = DatabaseManager()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'fasta', 'fastq'}

@samples_bp.route('/samples/upload', methods=['POST'])
async def upload_sample():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Get metadata from form data
        metadata = request.form.to_dict()
        required_fields = ['location', 'environmentType', 'collectionDate']
        
        for field in required_fields:
            if field not in metadata:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{str(uuid.uuid4())}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)

        # Save file
        file.save(file_path)

        # Read sequence data
        with open(file_path, 'r') as f:
            sequence_data = f.read().strip()

        # Create sample document
        sample = Sample(
            id=ObjectId(),
            filename=unique_filename,
            file_path=file_path,
            sequence=sequence_data,
            metadata=SampleMetadata(
                location=metadata['location'],
                environment_type=metadata['environmentType'],
                collection_date=datetime.fromisoformat(metadata['collectionDate']),
                tags=metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                notes=metadata.get('notes', '')
            ),
            status='pending',
            upload_date=datetime.utcnow()
        )

        # Save to database
        await db.insert_sample(sample.dict())

        return jsonify({
            'status': 'success',
            'message': 'Sample uploaded successfully',
            'sample_id': str(sample.id)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@samples_bp.route('/samples', methods=['GET'])
async def list_samples():
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        status = request.args.get('status')
        location = request.args.get('location')
        environment_type = request.args.get('environment_type')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        # Build query filters
        filters = {}
        if status:
            filters['status'] = status
        if location:
            filters['metadata.location'] = location
        if environment_type:
            filters['metadata.environment_type'] = environment_type
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter['$gte'] = datetime.fromisoformat(date_from)
            if date_to:
                date_filter['$lte'] = datetime.fromisoformat(date_to)
            filters['metadata.collection_date'] = date_filter

        # Get samples from database
        samples = await db.list_samples(
            filters=filters,
            page=page,
            per_page=per_page
        )

        # Get total count for pagination
        total_count = await db.count_samples(filters)

        return jsonify({
            'samples': [{
                'id': str(sample['_id']),
                'filename': sample['filename'],
                'status': sample['status'],
                'metadata': {
                    'location': sample['metadata']['location'],
                    'environmentType': sample['metadata']['environment_type'],
                    'collectionDate': sample['metadata']['collection_date'].isoformat(),
                    'tags': sample['metadata']['tags']
                },
                'uploadDate': sample['upload_date'].isoformat()
            } for sample in samples],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_count': total_count,
                'total_pages': (total_count + per_page - 1) // per_page
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@samples_bp.route('/samples/<sample_id>', methods=['GET'])
async def get_sample(sample_id):
    try:
        sample = await db.get_sample(ObjectId(sample_id))
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404

        return jsonify({
            'id': str(sample['_id']),
            'filename': sample['filename'],
            'status': sample['status'],
            'metadata': {
                'location': sample['metadata']['location'],
                'environmentType': sample['metadata']['environment_type'],
                'collectionDate': sample['metadata']['collection_date'].isoformat(),
                'tags': sample['metadata']['tags'],
                'notes': sample['metadata']['notes']
            },
            'uploadDate': sample['upload_date'].isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@samples_bp.route('/samples/<sample_id>', methods=['DELETE'])
async def delete_sample(sample_id):
    try:
        # Get sample to retrieve file path
        sample = await db.get_sample(ObjectId(sample_id))
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404

        # Delete file if it exists
        if os.path.exists(sample['file_path']):
            os.remove(sample['file_path'])

        # Delete from database
        await db.delete_sample(ObjectId(sample_id))

        return jsonify({
            'status': 'success',
            'message': 'Sample deleted successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500