"""
Task 3: Chainlit Chat UI for Support Agent

Pure frontend that calls the Agent API.
No agent logic - just HTTP client to Agent API service.
"""
import chainlit as cl
import httpx
import os
import logging
import uuid
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Agent API URL - configurable via environment
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session.
    
    Sets up conversation ID and session state.
    """
    # Generate a unique conversation ID for this session
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)
    cl.user_session.set("message_history", [])
    
    logger.info(f"New chat session started: {conversation_id}")
    logger.info(f"Agent API URL: {AGENT_API_URL}")
    
    # Welcome message
    await cl.Message(
        content="👋 Hello! I'm your AI support assistant. How can I help you today?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages.
    
    Sends the message to Agent API and displays the response.
    """
    conversation_id = cl.user_session.get("conversation_id")
    message_history = cl.user_session.get("message_history", [])
    
    # Add user message to history
    message_history.append({"role": "user", "content": message.content})
    
    # Show loading state
    loading_msg = cl.Message(content="")
    await loading_msg.send()
    
    try:
        logger.info(f"Sending message to Agent API: {message.content[:50]}...")
        
        # Call Agent API
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{AGENT_API_URL}/agent",
                json={
                    "user_message": message.content,
                    "conversation_id": conversation_id
                }
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract response data
        agent_response = data.get("agent_response", "I couldn't generate a response.")
        sentiment = data.get("sentiment_analysis", {})
        action_taken = data.get("action_taken", "unknown")
        escalated = data.get("escalated", False)
        
        # Log the response details
        logger.info(f"[Agent Response] Sentiment: {sentiment.get('sentiment')} ({sentiment.get('confidence')}%)")
        logger.info(f"[Agent Response] Action: {action_taken} | Escalated: {escalated}")
        
        # Update the loading message with the actual response
        loading_msg.content = agent_response
        await loading_msg.update()
        
        # Add assistant response to history
        message_history.append({"role": "assistant", "content": agent_response})
        cl.user_session.set("message_history", message_history)
        
    except httpx.ConnectError as e:
        error_message = "⚠️ Unable to connect to the Agent API. Please ensure the service is running."
        loading_msg.content = error_message
        await loading_msg.update()
        logger.error(f"Connection error: {e}")
        
    except httpx.TimeoutException as e:
        error_message = "⚠️ Request timed out. Please try again."
        loading_msg.content = error_message
        await loading_msg.update()
        logger.error(f"Timeout error: {e}")
        
    except httpx.HTTPStatusError as e:
        error_message = f"⚠️ Agent API error: {e.response.status_code}. Please try again."
        loading_msg.content = error_message
        await loading_msg.update()
        logger.error(f"HTTP error: {e}")
        
    except Exception as e:
        error_message = f"⚠️ An unexpected error occurred. Please try again."
        loading_msg.content = error_message
        await loading_msg.update()
        logger.error(f"Unexpected error: {e}")


@cl.on_stop
async def on_stop():
    """Handle conversation stop/end."""
    await cl.Message(
        content="Thank you for chatting with us! Have a great day! 👋"
    ).send()
