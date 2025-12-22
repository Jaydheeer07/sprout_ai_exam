"""
Task 3: Agent Tools

Tools available to the support agent for handling customer interactions.
"""
from langchain.tools import tool
from typing import Optional
import httpx
import os
import logging

logger = logging.getLogger(__name__)

# API base URL - configurable via environment
# SENTIMENT_API_URL is used for Docker networking (service name resolution)
# Defaults to localhost for non-Docker local development
API_BASE_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")


@tool
def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a customer message.
    
    Use this tool to understand the emotional tone of the customer's message.
    Returns sentiment (positive/neutral/negative) and confidence score.
    
    Args:
        text: The customer's message to analyze
        
    Returns:
        Dictionary with 'sentiment' and 'confidence' keys
    """
    try:
        response = httpx.post(
            f"{API_BASE_URL}/predict",
            json={"text": text},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "sentiment": data["model_output"],
            "confidence": data["confidence_score"]
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        # Return neutral with low confidence on error
        return {
            "sentiment": "neutral",
            "confidence": 50.0,
            "error": str(e)
        }


@tool
def escalate_to_support(
    reason: str,
    priority: str = "high",
    customer_sentiment: Optional[str] = None
) -> dict:
    """
    Escalate the conversation to human support team.
    
    Use this when the customer is clearly upset (negative sentiment with high confidence)
    or when the issue requires human intervention.
    
    Args:
        reason: Brief description of why escalation is needed
        priority: Priority level - 'high', 'medium', or 'low'
        customer_sentiment: The detected sentiment of the customer
        
    Returns:
        Confirmation of escalation with ticket details
    """
    import uuid
    ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    
    logger.info(f"Escalation created: {ticket_id} - {reason}")
    
    return {
        "status": "escalated",
        "ticket_id": ticket_id,
        "priority": priority,
        "reason": reason,
        "message": f"Conversation escalated to support team. Ticket ID: {ticket_id}. A support representative will reach out within 2 hours."
    }


@tool
def ask_clarifying_question(
    question_type: str,
    context: Optional[str] = None
) -> dict:
    """
    Generate a clarifying question to better understand the customer's needs.
    
    Use this when sentiment confidence is low or when more information is needed
    to properly assist the customer.
    
    Args:
        question_type: Type of clarification needed - 'issue_details', 'order_info', 'feedback', 'general'
        context: Additional context about what information is needed
        
    Returns:
        Suggested clarifying question and guidance
    """
    question_templates = {
        "issue_details": "Could you please provide more details about the issue you're experiencing?",
        "order_info": "Could you share your order number or the email address associated with your account?",
        "feedback": "I'd love to hear more about your experience. What specifically stood out to you?",
        "general": "I want to make sure I understand correctly. Could you tell me more about what you need help with?"
    }
    
    base_question = question_templates.get(question_type, question_templates["general"])
    
    return {
        "status": "clarification_needed",
        "question_type": question_type,
        "suggested_question": base_question,
        "context": context,
        "message": f"Asking for clarification: {base_question}"
    }


@tool
def send_thank_you_response(
    feedback_type: str = "general",
    offer_additional_help: bool = True
) -> dict:
    """
    Send a thank you response for positive customer feedback.
    
    Use this when the customer expresses satisfaction or positive sentiment
    with high confidence.
    
    Args:
        feedback_type: Type of positive feedback - 'product', 'service', 'general'
        offer_additional_help: Whether to offer additional assistance
        
    Returns:
        Thank you message template and guidance
    """
    thank_you_templates = {
        "product": "Thank you so much for your kind words about our product!",
        "service": "We're thrilled to hear you had a great experience with our service!",
        "general": "Thank you for your wonderful feedback!"
    }
    
    base_message = thank_you_templates.get(feedback_type, thank_you_templates["general"])
    
    additional = " Is there anything else I can help you with today?" if offer_additional_help else ""
    
    return {
        "status": "positive_response",
        "feedback_type": feedback_type,
        "message": base_message + additional
    }


# Export all tools as a list for easy import
SUPPORT_TOOLS = [
    analyze_sentiment,
    escalate_to_support,
    ask_clarifying_question,
    send_thank_you_response
]
