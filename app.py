from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import torch
from datetime import datetime
import uuid
import json

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://<username>:<password>@cluster0.lklwwgr.mongodb.net/')
client = MongoClient(MONGO_URI)
db = client.deepbioid_db

# Define taxa and roles (replace with your real labels)
taxa = ["Taxon1", "Taxon2", "Taxon3", "Taxon4"]
roles = ["Role1", "Role2", "Role3", "Role4"]

# Load the model and tokenizer at startup
from model import MultiTaskTaxonomyModel, KmerTokenizer  # pre-convert notebook to Python

MODEL_PATH = "multitask_model_demo_small.pth"

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model_state = checkpoint['model_state_dict']
    tokenizer_vocab = checkpoint['tokenizer_vocab']
    
    model = MultiTaskTaxonomyModel(vocab_size=len(tokenizer_vocab))
    model.load_state_dict(model_state)
    model.eval()
    
    tokenizer = KmerTokenizer(k=4)
    return model, tokenizer, tokenizer_vocab

model, tokenizer, tokenizer_vocab = load_model()


# Ensure uploads directory exists
os.makedirs('uploads', exist_ok=True)


# ------------------ API ROUTES ------------------ #

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
        
        # Store sample info in MongoDB
        sample_doc = {
            "sample_id": sample_id,
            "filename": file.filename,
            "metadata": metadata,
            "status": "uploaded",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        db.samples.insert_one(sample_doc)
        
        # Save file to disk
        file.save(f"uploads/{file.filename}")
        
        return jsonify({"message": "Sample uploaded successfully", "sample_id": sample_id}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/samples/<sample_id>/status', methods=['GET'])
def get_sample_status(sample_id):
    try:
        sample = db.samples.find_one({"sample_id": sample_id})
        if not sample:
            return jsonify({"error": "Sample not found"}), 404

        # Update status if results exist
        result = db.results.find_one({"sample_id": sample_id})
        if result:
            db.samples.update_one(
                {"sample_id": sample_id},
                {"$set": {"status": "analyzed", "updated_at": datetime.now()}}
            )
            sample["status"] = "analyzed"

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

        results = list(db.results.find(
            {"sample_id": {"$in": sample_ids}},
            {"_id": 0}  # exclude ObjectId
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
        taxonomy = list(db.taxonomy.find({}, {"_id": 0}))
        return jsonify({"taxonomy": taxonomy})
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
        sample_id = request.json.get('sample_id')
        if not sample_id:
            return jsonify({"error": "No sample ID provided"}), 400

        sample = db.samples.find_one({"sample_id": sample_id})
        if not sample:
            return jsonify({"error": "Sample not found"}), 404

        # Fetch metadata
        metadata = sample.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        seq = metadata.get('sequence', '')
        tokens = torch.tensor(tokenizer.encode(seq), dtype=torch.long).unsqueeze(0)
        mask = torch.zeros(1, tokens.size(1), dtype=torch.bool)

        with torch.no_grad():
            output = model(tokens, src_key_padding_mask=~mask)

        results = {
            "tax_prediction": taxa[output['tax_logits'].argmax(dim=1).item()],
            "role_prediction": roles[output['role_logits'].argmax(dim=1).item()],
            "novel_score": torch.sigmoid(output['novel_logits']).item()
        }

        db.results.insert_one({"sample_id": sample_id, "results": results})

        return jsonify({"message": "Analysis completed", "results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ MAIN ------------------ #
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
