"""Schemas for agent endpoint."""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class AgentRequest(BaseModel):
    """Request schema for agent endpoint."""
    user_message: str = Field(..., min_length=1, description="The user's message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = Field(..., ge=0, le=100)


class AgentResponse(BaseModel):
    """Response schema for agent endpoint."""
    sentiment_analysis: SentimentAnalysis
    action_taken: Literal["escalate_to_support", "ask_clarifying_question", "send_thank_you_response"]
    agent_response: str
    escalated: bool
    conversation_id: str
