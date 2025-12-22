# Task 3: Agent API & Chainlit Frontend

This folder contains the **Agent API** (Port 8001) and **Chainlit Frontend** (Port 8080).

## 🏗️ Architecture

```
task3_agent_workflow/
├── main.py              # Agent API entry point (Port 8001)
├── app.py               # Chainlit UI - Pure HTTP client (Port 8080)
├── routes/
│   └── agent.py         # /agent endpoint (ReACT agent)
├── schemas/
│   └── agent.py         # Agent request/response models
├── tools.py             # Agent tools (analyze_sentiment, escalate, etc.)
├── prompts.py           # Agent system prompt
└── Dockerfile           # Container configuration
```

## Agent API (Port 8001)

ReACT agent service that calls Sentiment API as a tool.

### Endpoints

| Method | Endpoint    | Description                           |
| ------ | ----------- | ------------------------------------- |
| GET    | `/`       | API info                              |
| GET    | `/health` | Health check                          |
| POST   | `/agent`  | Process message through support agent |

### Run Locally

```bash
python -m uvicorn task3_agent_workflow.main:app --reload --port 8001
```

### Environment Variables

| Variable              | Description       | Default               |
| --------------------- | ----------------- | --------------------- |
| `OPENAI_API_KEY`    | OpenAI API key    | Required              |
| `OPENAI_MODEL`      | Model to use      | gpt-5-mini            |
| `SENTIMENT_API_URL` | Sentiment API URL | http://localhost:8000 |
| `API_PORT`          | Port to run on    | 8001                  |

## Chainlit Frontend (Port 8080)

**Pure UI that calls Agent API** - No agent logic embedded.

The Chainlit app is a simple HTTP client that:

1. Accepts user messages
2. Sends them to Agent API (`/agent` endpoint)
3. Displays the response

### Run Locally

```bash
chainlit run task3_agent_workflow/app.py --host 0.0.0.0 --port 8080
```

### Environment Variables

| Variable          | Description    | Default               |
| ----------------- | -------------- | --------------------- |
| `AGENT_API_URL` | Agent API URL  | http://localhost:8001 |
| `CHAINLIT_HOST` | Host to bind   | 0.0.0.0               |
| `CHAINLIT_PORT` | Port to run on | 8080                  |

## Agent Tools

1. **analyze_sentiment** - Calls Sentiment API to get sentiment + confidence
2. **escalate_to_support** - Creates support ticket for upset customers
3. **ask_clarifying_question** - Gathers more information
4. **send_thank_you_response** - Acknowledges positive feedback

## Agent Workflow

```
User Message → analyze_sentiment → Decision → Action → Response
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
               NEGATIVE           NEUTRAL           POSITIVE
                    │                 │                 │
              ┌─────┴─────┐           │           ┌─────┴─────┐
              ▼           ▼           ▼           ▼           ▼
            ≥80%        <80%        Any         ≥80%        <80%
              │           │           │           │           │
              ▼           ▼           ▼           ▼           ▼
          ESCALATE    CLARIFY     CLARIFY      THANK      CLARIFY
```

## API Example

```bash
curl -X POST http://localhost:8001/agent \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I have been waiting 2 weeks for my order!"}'
```

Response:

```json
{
  "sentiment_analysis": {
    "sentiment": "negative",
    "confidence": 94.5
  },
  "action_taken": "escalate_to_support",
  "agent_response": "I sincerely apologize for the delay...",
  "escalated": true,
  "conversation_id": "conv_123"
}
```

## Interactive Docs

- Agent API Swagger: http://localhost:8001/docs
- Agent API ReDoc: http://localhost:8001/redoc
