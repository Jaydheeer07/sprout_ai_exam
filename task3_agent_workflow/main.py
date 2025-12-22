"""
Agent API Service (Task 3)

FastAPI application exposing the ReACT agent endpoint.
Calls Sentiment API as a tool for sentiment analysis.
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

from task3_agent_workflow.routes import agent

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
    logger.info("Starting Agent API Service...")
    logger.info(f"Using OpenAI model: {os.getenv('OPENAI_MODEL', 'gpt-5-mini')}")
    logger.info(f"Sentiment API URL: {os.getenv('SENTIMENT_API_URL', 'http://localhost:8000')}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent API Service...")


# Create FastAPI application
app = FastAPI(
    title="Sprout AI Agent API",
    description="""
    ReACT Agent Microservice for Customer Support.
    
    ## Features
    
    - **AI-Powered Support**: Uses LangGraph ReACT agent for intelligent responses
    - **Sentiment-Aware**: Analyzes customer sentiment via Sentiment API
    - **Action-Based**: Takes appropriate actions based on sentiment analysis
    
    ## Endpoints
    
    - `/agent` - Process message through support agent
    - `/health` - Service health check
    
    ## Agent Actions
    
    - `escalate_to_support` - For highly negative sentiment
    - `ask_clarifying_question` - For unclear or low-confidence sentiment
    - `send_thank_you_response` - For positive feedback
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

# Include agent router
app.include_router(agent.router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Sprout AI Agent API",
        "version": "1.0.0",
        "service": "agent-api",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Agent API."""
    return {
        "status": "healthy",
        "service": "agent-api",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8001))
    
    uvicorn.run(
        "task3_agent_workflow.main:app",
        host=host,
        port=port,
        reload=True
    )
