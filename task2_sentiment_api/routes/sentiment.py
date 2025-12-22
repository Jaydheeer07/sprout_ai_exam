"""Sentiment prediction endpoints."""
from fastapi import APIRouter, HTTPException
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from task1_ml_model.sentiment_classifier import get_classifier
from task2_sentiment_api.schemas.sentiment import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentItem,
    BatchSentimentResponse,
)

router = APIRouter(tags=["Sentiment"])


@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a single text.
    
    Args:
        request: SentimentRequest with text to analyze
        
    Returns:
        SentimentResponse with model_output and confidence_score
    """
    try:
        classifier = get_classifier()
        sentiment, confidence = classifier.predict(request.text)
        
        return SentimentResponse(
            model_output=sentiment,
            confidence_score=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_sentiment_batch(request: BatchSentimentRequest):
    """
    Predict sentiment for multiple texts.
    
    Args:
        request: BatchSentimentRequest with list of texts
        
    Returns:
        BatchSentimentResponse with results for each text
    """
    try:
        classifier = get_classifier()
        results = []
        
        for text in request.texts:
            sentiment, confidence = classifier.predict(text)
            results.append(BatchSentimentItem(
                text=text,
                model_output=sentiment,
                confidence_score=confidence
            ))
        
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
