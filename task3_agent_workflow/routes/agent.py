"""
Agent endpoint for invoking the LangGraph support agent.

Provides a REST API interface to the customer support agent workflow.
"""
from fastapi import APIRouter, HTTPException
import os
import sys
import uuid
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from task3_agent_workflow.tools import SUPPORT_TOOLS
from task3_agent_workflow.prompts import SUPPORT_AGENT_SYSTEM_PROMPT
from task3_agent_workflow.schemas.agent import (
    AgentRequest,
    AgentResponse,
    SentimentAnalysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agent"])

# In-memory conversation store (for demo purposes)
_conversations: dict = {}


def _create_agent():
    """Create a new LangGraph agent instance."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = create_react_agent(
        model=llm,
        tools=SUPPORT_TOOLS,
        prompt=SUPPORT_AGENT_SYSTEM_PROMPT
    )
    
    return agent, model_name


def _extract_action_from_messages(messages) -> tuple[str, dict]:
    """
    Extract the action taken and tool result from agent messages.
    
    Returns:
        Tuple of (action_name, tool_result_dict)
    """
    action_taken = "ask_clarifying_question"  # Default
    tool_result = {}
    
    for msg in messages:
        # Look for tool calls in AI messages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get('name', '')
                if tool_name in ['escalate_to_support', 'ask_clarifying_question', 'send_thank_you_response']:
                    action_taken = tool_name
        
        # Look for tool message results
        if hasattr(msg, 'type') and msg.type == 'tool':
            if hasattr(msg, 'content'):
                try:
                    import json
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                except:
                    pass
    
    return action_taken, tool_result


def _extract_sentiment_from_messages(messages) -> dict:
    """
    Extract sentiment analysis result from agent messages.
    
    Returns:
        Dict with sentiment and confidence
    """
    sentiment = {"sentiment": "neutral", "confidence": 50.0}
    
    for msg in messages:
        # Look for tool message with sentiment result
        if hasattr(msg, 'type') and msg.type == 'tool':
            if hasattr(msg, 'name') and msg.name == 'analyze_sentiment':
                try:
                    import json
                    content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(content, dict) and 'sentiment' in content:
                        sentiment = {
                            "sentiment": content.get("sentiment", "neutral"),
                            "confidence": content.get("confidence", 50.0)
                        }
                except:
                    pass
    
    return sentiment


def _extract_final_response(messages) -> str:
    """Extract the final AI response from agent messages."""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            if hasattr(msg, 'type') and msg.type == 'ai':
                # Skip if it only has tool calls without content
                if hasattr(msg, 'tool_calls') and msg.tool_calls and not msg.content.strip():
                    continue
                return msg.content
    return "I apologize, but I couldn't generate a response. Please try again."


@router.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest):
    """
    Process a message through the support agent.
    
    The agent will:
    1. Analyze the sentiment of the message
    2. Take appropriate action based on sentiment and confidence
    3. Return a structured response
    
    Args:
        request: AgentRequest with user_message and optional conversation_id
        
    Returns:
        AgentResponse with sentiment analysis, action taken, and agent response
    """
    try:
        # Generate or use existing conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Create agent
        agent, model_name = _create_agent()
        
        # Prepare messages
        messages = [{"role": "user", "content": request.user_message}]
        
        # Invoke the agent
        logger.info(f"Processing agent message for conversation {conversation_id}")
        result = await agent.ainvoke({"messages": messages})
        
        # Extract results from agent messages
        agent_messages = result.get("messages", [])
        
        # Get sentiment analysis
        sentiment_data = _extract_sentiment_from_messages(agent_messages)
        
        # Get action taken
        action_taken, tool_result = _extract_action_from_messages(agent_messages)
        
        # Get final response
        agent_response = _extract_final_response(agent_messages)
        
        # Determine if escalated
        escalated = action_taken == "escalate_to_support"
        
        # Log token usage from AI messages
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        for msg in agent_messages:
            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                usage = msg.usage_metadata
                if hasattr(usage, 'input_tokens'):
                    prompt_tokens += usage.input_tokens
                if hasattr(usage, 'output_tokens'):
                    completion_tokens += usage.output_tokens
                if hasattr(usage, 'total_tokens'):
                    total_tokens += usage.total_tokens
        
        if total_tokens > 0:
            logger.info(f"[Token Usage] Model: {model_name} | Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}")
        else:
            # Fallback: try response_metadata
            for msg in agent_messages:
                if hasattr(msg, 'response_metadata') and msg.response_metadata:
                    token_usage = msg.response_metadata.get('token_usage', {})
                    if token_usage:
                        prompt_tokens += token_usage.get('prompt_tokens', 0)
                        completion_tokens += token_usage.get('completion_tokens', 0)
                        total_tokens += token_usage.get('total_tokens', 0)
            if total_tokens > 0:
                logger.info(f"[Token Usage] Model: {model_name} | Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}")
            else:
                logger.info(f"[Token Usage] Model: {model_name} | Token usage not available in response")
        
        # Store conversation (for potential future use)
        _conversations[conversation_id] = {
            "messages": agent_messages,
            "last_sentiment": sentiment_data
        }
        
        return AgentResponse(
            sentiment_analysis=SentimentAnalysis(
                sentiment=sentiment_data["sentiment"],
                confidence=sentiment_data["confidence"]
            ),
            action_taken=action_taken,
            agent_response=agent_response,
            escalated=escalated,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Agent endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
