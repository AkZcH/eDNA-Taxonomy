from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import torch
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://ritamvaskar0:Ritam2005@cluster0.lklwwgr.mongodb.net/')
client = MongoClient(MONGO_URI)
db = client.deepbioid_db

# Load the trained model
def load_model():
    checkpoint = torch.load('multitask_model_demo_small.pth')
    model_state = checkpoint['model_state_dict']
    tokenizer_vocab = checkpoint['tokenizer_vocab']
    return model_state, tokenizer_vocab

# API Routes
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/v1/samples/upload', methods=['POST'])
def upload_sample():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        metadata = request.form.get('metadata', '{}')
        
        # Generate unique sample ID
        sample_id = str(uuid.uuid4())
        
        # Store sample information in MongoDB
        sample_doc = {
            "sample_id": sample_id,
            "filename": file.filename,
            "metadata": metadata,
            "status": "uploaded",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        db.samples.insert_one(sample_doc)
        
        # Save file to disk (you might want to use cloud storage in production)
        file.save(f"uploads/{file.filename}")
        
        return jsonify({
            "message": "Sample uploaded successfully",
            "sample_id": sample_id
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/samples/<sample_id>/status', methods=['GET'])
def get_sample_status(sample_id):
    try:
        sample = db.samples.find_one({"sample_id": sample_id})
        if not sample:
            return jsonify({"error": "Sample not found"}), 404
            
        return jsonify({
            "sample_id": sample_id,
            "status": sample["status"],
            "created_at": sample["created_at"],
            "updated_at": sample["updated_at"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/samples/<sample_id>/results', methods=['GET'])
def get_sample_results(sample_id):
    try:
        result = db.results.find_one({"sample_id": sample_id})
        if not result:
            return jsonify({"error": "Results not found"}), 404

        # Convert ObjectId to string
        result["_id"] = str(result["_id"])
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/analysis/comparative', methods=['POST'])
def comparative_analysis():
    try:
        sample_ids = request.json.get('sample_ids', [])
        if not sample_ids:
            return jsonify({"error": "No sample IDs provided"}), 400

        # Exclude _id so ObjectId never reaches jsonify
        results = list(db.results.find(
            {"sample_id": {"$in": sample_ids}},
            {"_id": 0}         # <--- add this projection
        ))

        return jsonify({
            "analysis_id": str(uuid.uuid4()),
            "samples": sample_ids,
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/taxonomy/tree', methods=['GET'])
def get_taxonomy_tree():
    try:
        # Fetch taxonomy tree from database
        taxonomy = list(db.taxonomy.find({}, {"_id": 0}))
        
        return jsonify({
            "taxonomy": taxonomy
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/v1/samples/get_all', methods=['GET'])
def get_all_samples():
    try:
        samples = list(db.samples.find({}, {"_id": 0}))
        return jsonify({"samples": samples})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)