from pymongo import MongoClient
from typing import Dict, List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGO_URI', 'mongodb+srv://ritamvaskar0:Ritam2005@cluster0.lklwwgr.mongodb.net/'))
        self.db = self.client.deepbioid_db
        self._setup_collections()
        
    def _setup_collections(self):
        """Setup database collections with indexes"""
        # Samples collection
        if 'samples' not in self.db.list_collection_names():
            self.db.create_collection('samples')
        self.db.samples.create_index('sample_id', unique=True)
        self.db.samples.create_index('created_at')
        
        # Results collection
        if 'results' not in self.db.list_collection_names():
            self.db.create_collection('results')
        self.db.results.create_index('sample_id')
        self.db.results.create_index('created_at')
        
        # Taxonomy collection
        if 'taxonomy' not in self.db.list_collection_names():
            self.db.create_collection('taxonomy')
        self.db.taxonomy.create_index('node_id', unique=True)
        
        # Comparative analysis collection
        if 'comparative_analysis' not in self.db.list_collection_names():
            self.db.create_collection('comparative_analysis')
        self.db.comparative_analysis.create_index('analysis_id', unique=True)
        
    def insert_sample(self, sample_data: Dict) -> str:
        """Insert a new sample record"""
        sample_data['created_at'] = datetime.now()
        sample_data['updated_at'] = datetime.now()
        result = self.db.samples.insert_one(sample_data)
        return str(result.inserted_id)
    
    def update_sample_status(self, sample_id: str, status: str) -> bool:
        """Update sample processing status"""
        result = self.db.samples.update_one(
            {'sample_id': sample_id},
            {'$set': {'status': status, 'updated_at': datetime.now()}}
        )
        return result.modified_count > 0
    
    def get_sample(self, sample_id: str) -> Optional[Dict]:
        """Retrieve sample information"""
        return self.db.samples.find_one({'sample_id': sample_id}, {'_id': 0})
    
    def insert_result(self, result_data: Dict) -> str:
        """Insert analysis results"""
        result_data['created_at'] = datetime.now()
        result = self.db.results.insert_one(result_data)
        return str(result.inserted_id)
    
    def get_result(self, sample_id: str) -> Optional[Dict]:
        """Retrieve analysis results for a sample"""
        return self.db.results.find_one({'sample_id': sample_id}, {'_id': 0})
    
    def insert_taxonomy_node(self, node_data: Dict) -> str:
        """Insert a taxonomy tree node"""
        result = self.db.taxonomy.insert_one(node_data)
        return str(result.inserted_id)
    
    def get_taxonomy_tree(self) -> List[Dict]:
        """Retrieve entire taxonomy tree"""
        return list(self.db.taxonomy.find({}, {'_id': 0}))
    
    def insert_comparative_analysis(self, analysis_data: Dict) -> str:
        """Insert comparative analysis results"""
        analysis_data['created_at'] = datetime.now()
        result = self.db.comparative_analysis.insert_one(analysis_data)
        return str(result.inserted_id)
    
    def get_comparative_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Retrieve comparative analysis results"""
        return self.db.comparative_analysis.find_one({'analysis_id': analysis_id}, {'_id': 0})
    
    def get_samples_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Retrieve samples within a date range"""
        return list(self.db.samples.find({
            'created_at': {'$gte': start_date, '$lte': end_date}
        }, {'_id': 0}))
    
    def get_samples_by_location(self, latitude: float, longitude: float, radius_km: float) -> List[Dict]:
        """Retrieve samples within a geographical radius"""
        return list(self.db.samples.find({
            'metadata.location': {
                '$geoWithin': {
                    '$centerSphere': [[longitude, latitude], radius_km/6371]
                }
            }
        }, {'_id': 0}))
    
    def close(self):
        """Close database connection"""
        self.client.close()