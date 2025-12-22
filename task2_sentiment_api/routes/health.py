"""Health check endpoint."""
from fastapi import APIRouter
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from task1_ml_model.sentiment_classifier import get_classifier

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """
    Check the health status of the API.
    
    Returns:
        Health status including model loading state
    """
    classifier = get_classifier()
    
    return {
        "status": "healthy",
        "model_loaded": classifier.is_loaded,
        "version": "1.0.0"
    }
