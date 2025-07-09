from pydantic import BaseModel
from typing import List, Dict, Any

class Recommendation(BaseModel):
    product_id: int
    rating: float # This is the ALS score for user recommendations
    recommendation_score: float
    # Add other relevant fields you want to return
    purchase_per_view: float = None
    relative_price: float = None
    distCol: float = None # For product recommendations


class UserRecommendationsResponse(BaseModel):
    user_id: int
    recommendations: List[Recommendation]

class ProductRecommendationsResponse(BaseModel):
    product_id: int
    recommendations: List[Recommendation]