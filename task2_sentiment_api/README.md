# Task 2: Sentiment API Service

This folder contains the **Sentiment API** - a pure ML sentiment prediction service.

## 🏗️ Architecture

```
task2_sentiment_api/
├── main.py              # Sentiment API entry point (Port 8000)
├── routes/
│   ├── sentiment.py     # /predict, /predict/batch endpoints
│   └── health.py        # /health endpoint
└── schemas/
    └── sentiment.py     # Sentiment request/response models
```

## Sentiment API (Port 8000)

Pure ML sentiment prediction service using HuggingFace RoBERTa model.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Single sentiment prediction |
| POST | `/predict/batch` | Batch predictions |

### Run Locally

```bash
python -m uvicorn task2_sentiment_api.main:app --reload --port 8000
```

## API Examples

### Sentiment Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Response:
```json
{
  "model_output": "positive",
  "confidence_score": 95.67
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!", "Okay"]}'
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SENTIMENT_MODEL_NAME` | HuggingFace model | cardiffnlp/twitter-roberta-base-sentiment-latest |
| `API_PORT` | Port to run on | 8000 |

## Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
