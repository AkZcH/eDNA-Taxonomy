from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import torch
from datetime import datetime
import uuid
# from notebook_importer import import_notebook
import nbformat
from nbconvert import PythonExporter
import importlib.util
import os
import json



# Define taxa as a list of taxonomy labels
taxa = ["Taxon1", "Taxon2", "Taxon3", "Taxon4"]  # Replace with actual taxonomy labels
roles = ["Role1", "Role2", "Role3", "Role4"]  # Replace with actual taxonomy labels
def load_notebook_as_module(notebook_path, module_name):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = f.read()
    notebook = nbformat.reads(notebook_content, as_version=4)
    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)

    module_path = f"{module_name}.py"
    with open(module_path, 'w', encoding='utf-8') as f:
        f.write(python_code)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    os.remove(module_path)  # Clean up the temporary Python file
    return module

model_notebook = load_notebook_as_module('model.ipynb', 'model')
MultiTaskTaxonomyModel = model_notebook.MultiTaskTaxonomyModel
KmerTokenizer = model_notebook.KmerTokenizer

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

@app.route('/api/v1/analyze-result', methods=['POST'])
def analyze_result():
    try:
        # Get sample_id from request
        sample_id = request.json.get('sample_id')
        if not sample_id:
            return jsonify({"error": "No sample ID provided"}), 400

        # Fetch sample from database
        sample = db.samples.find_one({"sample_id": sample_id})
        if not sample:
            return jsonify({"error": "Sample not found"}), 404

        # Load the model and tokenizer
        model_state, tokenizer_vocab = load_model()
        model = MultiTaskTaxonomyModel(vocab_size=len(tokenizer_vocab))
        model.load_state_dict(model_state)
        model.eval()

        tokenizer = KmerTokenizer(k=4)
        # Fetch metadata and ensure it's a dictionary
        metadata = sample.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid metadata format"}), 400
        
        seq = metadata.get('sequence', '')
        tokens = torch.tensor(tokenizer.encode(seq), dtype=torch.long).unsqueeze(0)
        mask = torch.zeros(1, tokens.size(1), dtype=torch.bool)

        # Perform inference
        with torch.no_grad():
            output = model(tokens, src_key_padding_mask=~mask)

        # Prepare results
        results = {
            "tax_prediction": taxa[output['tax_logits'].argmax(dim=1).item()],
            "role_prediction": roles[output['role_logits'].argmax(dim=1).item()],
            "novel_score": torch.sigmoid(output['novel_logits']).item()
        }

        # Store results in database
        db.results.insert_one({"sample_id": sample_id, "results": results})

        return jsonify({"message": "Analysis completed", "results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

# Define taxa as a list of taxonomy labels
taxa = ["Taxon1", "Taxon2", "Taxon3", "Taxon4"]  # Replace with actual taxonomy labels