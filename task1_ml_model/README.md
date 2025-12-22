# Task 1: ML Foundation - Sentiment Classification Model

## Objective

Create a Python script that accepts text input and returns:
- `model_output`: positive | neutral | negative
- `confidence_score`: float (rounded to 2 decimals)

## Model

**cardiffnlp/twitter-roberta-base-sentiment-latest**
- Pre-trained on 124M tweets (2018-2021)
- 3-class sentiment classification
- Optimized for social media text
- **Validated as best choice** among 5 comparable models (see Model Comparison below)

## Files

| File | Description |
|------|-------------|
| `sentiment_classifier.py` | Main classifier module with `SentimentClassifier` class |
| `evaluate.py` | Batch evaluation script for test CSV |
| `compare_models.py` | Model comparison script (5 models, all 3-class) |

## Usage

### Single Prediction

```bash
python sentiment_classifier.py --text "I love this product!"
```

Output:
```json
{
  "model_output": "positive",
  "confidence_score": 95.67
}
```

### Batch Evaluation

```bash
python evaluate.py --input ../data/sentiment_test_cases_2025.csv --output ../data/output_sentiment_test.csv
```

### Model Comparison

```bash
python compare_models.py
```

Compares 5 models with native 3-class output for fair evaluation.

## Output Format

The evaluation script generates `output_sentiment_test.csv` with columns:
- `text` - Original input text
- `expected_sentiment` - Ground truth label
- `model_output` - Model prediction
- `confidence_score` - Prediction confidence (0-100)

## Model Comparison Results

| Model | Accuracy | F1-Macro | Speed |
|-------|----------|----------|-------|
| **Twitter-RoBERTa-Latest** ✅ | 86.75% | 86.60% | 36.1ms |
| BERTweet-Sentiment | 86.75% | 86.36% | 41.3ms |
| Twitter-RoBERTa-Original | 83.53% | 83.38% | 37.0ms |
| RoBERTa-Large-3Class | 81.73% | 81.69% | 145.2ms |
| DistilBERT-Multilingual | 58.63% | 47.23% | 21.9ms |

See `presentation/MODEL_COMPARISON.md` for detailed analysis.

## Accuracy Target

> **Required: >80% accuracy on test cases**
> **Achieved: 86.75% accuracy, 86.60% F1-score**
