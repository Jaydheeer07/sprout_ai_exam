# Sprout AI Sentiment Analysis & Support Agent

AI-powered sentiment analysis and customer support agent for Sprout AI Labs technical assessment.

## 🏗️ Architecture Overview

This project implements a **production-ready microservices architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MICROSERVICES ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│   │  Chainlit FE     │───▶│   Agent API      │───▶│  Sentiment API   │      │
│   │  (Port 8080)     │    │   (Port 8001)    │    │  (Port 8000)     │      │
│   │                  │    │                  │    │                  │      │
│   │  Pure UI         │    │  ReACT Agent     │    │  ML Model        │      │
│   │  HTTP Client     │    │  LangGraph       │    │  Predictions     │      │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Services

| Service | Port | Description | Folder |
|---------|------|-------------|--------|
| **Sentiment API** | 8000 | ML sentiment prediction service | `task2_sentiment_api/` |
| **Agent API** | 8001 | ReACT agent with LangGraph | `task3_agent_workflow/` |
| **Chainlit Frontend** | 8080 | Pure UI, calls Agent API | `task3_agent_workflow/` |

## 🔄 High-Level Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER REQUEST FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

  User: "I've been waiting 2 weeks for my order!"
                    │
                    ▼
    ┌───────────────────────────────┐
    │      CHAINLIT FRONTEND        │  ← User types message in chat UI
    │         (Port 8080)           │
    │   HTTP POST to /agent         │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │         AGENT API             │  ← LangGraph ReACT Agent processes
    │        (Port 8001)            │
    │                               │
    │  1. Receives user message     │
    │  2. Invokes analyze_sentiment │
    │  3. Makes decision            │
    │  4. Takes action              │
    │  5. Generates response        │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │       SENTIMENT API           │  ← ML model predicts sentiment
    │        (Port 8000)            │
    │                               │
    │  HuggingFace RoBERTa Model    │
    │  Returns: negative (94.5%)    │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │      AGENT DECISION           │  ← Based on sentiment + confidence
    │                               │
    │  Negative + ≥80% confidence   │
    │  → escalate_to_support        │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │       FINAL RESPONSE          │
    │                               │
    │  "I sincerely apologize for   │
    │   the delay. I've escalated   │
    │   this to our support team."  │
    └───────────────────────────────┘
```

## 📋 Task Overview

This project addresses **3 interconnected tasks**:

| Task | Description | Implementation |
|------|-------------|----------------|
| **Task 1** | ML Foundation - Sentiment Classification Model | `task1_ml_model/` |
| **Task 2** | Sentiment API Service | `task2_sentiment_api/` |
| **Task 3** | Agent API + Chainlit Frontend | `task3_agent_workflow/` |

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your-api-key-here
```

### Run with Docker

```bash
# Build and run all 3 services
docker-compose up --build

# Services will be available at:
# - Sentiment API: http://localhost:8000
# - Agent API: http://localhost:8001
# - Chat UI: http://localhost:8080
```

### Run Services Individually (Development)

```bash
# Terminal 1: Sentiment API (Task 2)
python -m uvicorn task2_sentiment_api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Agent API (Task 3)
python -m uvicorn task3_agent_workflow.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 3: Chainlit Frontend (Task 3)
chainlit run task3_agent_workflow/app.py --host 0.0.0.0 --port 8080
```

## 📡 API Endpoints

### Service 1: Sentiment API (Port 8000)

#### Health Check
```bash
GET http://localhost:8000/health
```

#### Sentiment Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "I love this product!"
}

# Response
{
  "model_output": "positive",
  "confidence_score": 95.67
}
```

#### Batch Prediction
```bash
POST http://localhost:8000/predict/batch
Content-Type: application/json

{
  "texts": ["Great service!", "Terrible experience", "Package arrived"]
}
```

### Service 2: Agent API (Port 8001)

#### Agent Endpoint
```bash
POST http://localhost:8001/agent
Content-Type: application/json

{
  "user_message": "I've been waiting for my order for 2 weeks!",
  "conversation_id": "conv_123"
}

# Response
{
  "sentiment_analysis": {
    "sentiment": "negative",
    "confidence": 94.5
  },
  "action_taken": "escalate_to_support",
  "agent_response": "I sincerely apologize...",
  "escalated": true,
  "conversation_id": "conv_123"
}
```

### Service 3: Chainlit Frontend (Port 8080)

Open http://localhost:8080 in your browser for the chat interface.

## 🧪 Running Evaluation

```bash
# Run sentiment model evaluation on test data
python task1_ml_model/evaluate.py --input data/sentiment_test_cases_2025.csv --output data/output_sentiment_test.csv
```

## 📊 Model Performance

**Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

### Overall Metrics

| Metric              | Score  |
| ------------------- | ------ |
| Accuracy            | 86.75% |
| Precision (macro)   | 86.87% |
| Recall (macro)      | 86.68% |
| F1-Score (macro)    | 86.60% |
| Precision (weighted)| 87.16% |
| Recall (weighted)   | 86.75% |
| F1-Score (weighted) | 86.77% |

### Per-Class Metrics

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Negative | 93.59%    | 82.49% | 87.69%   | 177     |
| Neutral  | 82.76%    | 86.33% | 84.51%   | 139     |
| Positive | 84.26%    | 91.21% | 87.60%   | 182     |

### Confusion Matrix

|          | Pred: Neg | Pred: Neu | Pred: Pos |
| -------- | --------- | --------- | --------- |
| **Neg**  | 146       | 14        | 17        |
| **Neu**  | 5         | 120       | 14        |
| **Pos**  | 5         | 11        | 166       |

*Evaluated on 498 test cases from `data/sentiment_test_cases_2025.csv`. All metrics exceed the 80% threshold.*

## 🤖 Agent Workflow

The support agent follows this decision matrix:

| Sentiment | Confidence | Action                  |
| --------- | ---------- | ----------------------- |
| Negative  | ≥80%      | Escalate to support     |
| Negative  | <80%       | Ask clarifying question |
| Neutral   | Any        | Ask clarifying question |
| Positive  | ≥80%      | Send thank you response |
| Positive  | <80%       | Ask clarifying question |

## 📝 API Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔮 Future Improvements

- Fine-tune model on domain-specific data
- Add conversation memory/context persistence
- Implement multi-language support
- Add monitoring and observability (Prometheus/Grafana)
- A/B testing framework for agent responses
- MCP server implementation for broader tool integration

## 📄 License

This project is created for the Sprout AI Labs technical assessment.

---

**Author**: Dherick Jay Paalam
**Date**: December 2025
