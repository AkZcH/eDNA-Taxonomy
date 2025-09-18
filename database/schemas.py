from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class SampleMetadata(BaseModel):
    collection_date: datetime
    location: Dict[str, float] = Field(..., example={"latitude": 45.0, "longitude": -75.0, "depth": 1000})
    environmental_data: Optional[Dict[str, float]] = Field(None, example={"temperature": 4.5, "salinity": 35.0})
    collection_method: str
    project_id: str

class Sample(BaseModel):
    sample_id: str
    filename: str
    metadata: SampleMetadata
    status: str = Field(..., example="uploaded")
    created_at: datetime
    updated_at: datetime

class TaxonomyNode(BaseModel):
    node_id: str
    name: str
    rank: str
    parent_id: Optional[str]
    children: List[str] = []
    attributes: Optional[Dict[str, any]] = None

class AnalysisResult(BaseModel):
    result_id: str
    sample_id: str
    taxonomy: Dict[str, any]
    ecological_role: Dict[str, any]
    novelty_score: float
    confidence_scores: Dict[str, float]
    processing_time: float
    created_at: datetime

class ComparativeAnalysis(BaseModel):
    analysis_id: str
    sample_ids: List[str]
    analysis_type: str
    results: Dict[str, any]
    created_at: datetime