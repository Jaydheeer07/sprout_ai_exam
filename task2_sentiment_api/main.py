"""
Sentiment API Service (Task 2)

FastAPI application exposing only sentiment prediction endpoints.
No agent logic - pure sentiment analysis service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task2_sentiment_api.routes import health, sentiment
from task1_ml_model.sentiment_classifier import get_classifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Sentiment API Service...")
    
    # Pre-load the sentiment model
    try:
        classifier = get_classifier()
        classifier.load_model()
        logger.info("Sentiment model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-load model: {e}. Will load on first request.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment API Service...")


# Create FastAPI application
app = FastAPI(
    title="Sprout AI Sentiment API",
    description="""
    Sentiment Analysis Microservice.
    
    ## Features
    
    - **Sentiment Analysis**: Classify text as positive, neutral, or negative
    - **Batch Processing**: Analyze multiple texts in a single request
    
    ## Endpoints
    
    - `/predict` - Single text sentiment prediction
    - `/predict/batch` - Batch sentiment prediction
    - `/health` - Service health check
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(sentiment.router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Sprout AI Sentiment API",
        "version": "1.0.0",
        "service": "sentiment-api",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(
        "task2_sentiment_api.main:app",
        host=host,
        port=port,
        reload=True
    )
