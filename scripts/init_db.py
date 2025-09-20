from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

def init_database():
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db_name = os.getenv('MONGODB_URI').split('/')[-1]
    db = client[db_name]
    
    try:
        # Create collections
        collections = [
            'samples',
            'analysis_results',
            'taxonomy_nodes',
            'comparative_analyses'
        ]
        
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                print(f'Created collection: {collection}')
        
        # Create indexes for samples collection
        samples = db.samples
        samples.create_index([('status', ASCENDING)])
        samples.create_index([('upload_date', DESCENDING)])
        samples.create_index([('metadata.location', TEXT)])
        samples.create_index([('metadata.environment_type', ASCENDING)])
        samples.create_index([('metadata.collection_date', ASCENDING)])
        samples.create_index([('metadata.tags', ASCENDING)])
        print('Created indexes for samples collection')
        
        # Create indexes for analysis_results collection
        results = db.analysis_results
        results.create_index([('sample_id', ASCENDING)])
        results.create_index([('analysis_date', DESCENDING)])
        print('Created indexes for analysis_results collection')
        
        # Create indexes for taxonomy_nodes collection
        taxonomy = db.taxonomy_nodes
        taxonomy.create_index([('parent_id', ASCENDING)])
        taxonomy.create_index([('level', ASCENDING)])
        taxonomy.create_index([('name', TEXT)])
        print('Created indexes for taxonomy_nodes collection')
        
        # Create indexes for comparative_analyses collection
        comparisons = db.comparative_analyses
        comparisons.create_index([('sample_ids', ASCENDING)])
        comparisons.create_index([('created_at', DESCENDING)])
        print('Created indexes for comparative_analyses collection')
        
        print('\nDatabase initialization completed successfully!')
        
    except Exception as e:
        print(f'Error initializing database: {str(e)}')
        sys.exit(1)
    finally:
        client.close()

def main():
    print('Initializing MongoDB database...')
    init_database()

if __name__ == '__main__':
    main()