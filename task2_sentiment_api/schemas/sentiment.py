"""Schemas for sentiment prediction endpoints."""
from pydantic import BaseModel, Field
from typing import List, Literal


class SentimentRequest(BaseModel):
    """Request schema for sentiment prediction."""
    text: str = Field(..., min_length=1, description="Text to analyze")


class SentimentResponse(BaseModel):
    """Response schema for sentiment prediction."""
    model_output: Literal["positive", "neutral", "negative"]
    confidence_score: float = Field(..., ge=0, le=100)


class BatchSentimentRequest(BaseModel):
    """Request schema for batch sentiment prediction."""
    texts: List[str] = Field(..., min_length=1, description="List of texts to analyze")


class BatchSentimentItem(BaseModel):
    """Single item in batch response."""
    text: str
    model_output: Literal["positive", "neutral", "negative"]
    confidence_score: float


class BatchSentimentResponse(BaseModel):
    """Response schema for batch sentiment prediction."""
    results: List[BatchSentimentItem]
