from pydantic import BaseModel
from typing import List

class FeedbackRequest(BaseModel):
    feedback: str

class FeedbackAnalysis(BaseModel):
    feedback: str
    sentiment: str
    attrition_risk: str
    recommendations: List[str]
