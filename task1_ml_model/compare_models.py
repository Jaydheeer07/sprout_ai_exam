"""
Compare multiple sentiment analysis models on the test dataset.

Models selected based on:
- 3-class output (positive, neutral, negative) for fair comparison
- Trained/fine-tuned on social media or similar text
- Comparable architecture to Twitter-RoBERTa

Usage:
    python compare_models.py
"""
import pandas as pd
import sys
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Updated models - ALL are 3-class (positive/neutral/negative) for fair comparison
MODELS_TO_TEST = [
    {
        "name": "Twitter-RoBERTa-Latest",
        "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "label_mapping": {
            "negative": "negative",
            "neutral": "neutral", 
            "positive": "positive",
            # Also handle numeric labels
            0: "negative",
            1: "neutral",
            2: "positive"
        },
        "preprocess": True,
        "description": "RoBERTa trained on 124M tweets (2018-2021), TweetEval fine-tuned"
    },
    {
        "name": "Twitter-RoBERTa-Original",
        "model_id": "cardiffnlp/twitter-roberta-base-sentiment",
        "label_mapping": {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
            0: "negative",
            1: "neutral",
            2: "positive"
        },
        "preprocess": True,
        "description": "Original RoBERTa trained on 58M tweets, TweetEval fine-tuned"
    },
    {
        "name": "BERTweet-Sentiment",
        "model_id": "finiteautomata/bertweet-base-sentiment-analysis",
        "label_mapping": {
            "NEG": "negative",
            "NEU": "neutral",
            "POS": "positive"
        },
        "preprocess": False,
        "description": "BERTweet fine-tuned for sentiment analysis"
    },
    {
        "name": "RoBERTa-Large-3Class",
        "model_id": "j-hartmann/sentiment-roberta-large-english-3-classes",
        "label_mapping": {
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive"
        },
        "preprocess": False,
        "description": "RoBERTa-large fine-tuned on 5.3K social media posts (86.1% reported)"
    },
    {
        "name": "DistilBERT-Multilingual-3Class",
        "model_id": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "label_mapping": {
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive"
        },
        "preprocess": False,
        "description": "Distilled multilingual model, fast inference"
    }
]

LABELS = ['negative', 'neutral', 'positive']


def preprocess_text(text: str) -> str:
    """Preprocess text for Twitter-RoBERTa models."""
    new_text = []
    for token in text.split(" "):
        token = "@user" if token.startswith("@") and len(token) > 1 else token
        token = "http" if token.startswith("http") else token
        new_text.append(token)
    return " ".join(new_text)


def load_model_pipeline(model_config: dict):
    """Load a HuggingFace sentiment analysis pipeline."""
    from transformers import pipeline, AutoTokenizer
    
    print(f"  Loading {model_config['name']}...")
    
    # Load tokenizer with truncation settings
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
    
    pipe = pipeline(
        "sentiment-analysis",
        model=model_config["model_id"],
        tokenizer=tokenizer,
        truncation=True,
        max_length=512
    )
    return pipe


def predict_with_model(pipe, text: str, model_config: dict) -> Tuple[str, float]:
    """Make prediction with a model pipeline."""
    # Preprocess if needed
    if model_config.get("preprocess"):
        text = preprocess_text(text)
    
    # Get prediction
    try:
        result = pipe(text)[0]
    except Exception as e:
        # Handle potential token length issues
        text = text[:500]  # Truncate as fallback
        result = pipe(text)[0]
    
    # Handle different output formats
    if isinstance(result, list):
        # Multi-label output (top_k=None)
        result = max(result, key=lambda x: x['score'])
    
    label = result['label']
    score = result['score'] * 100
    
    # Map label to our standard format
    label_mapping = model_config.get("label_mapping", {})
    
    if isinstance(label, int):
        sentiment = label_mapping.get(label, "neutral")
    else:
        # Try exact match first, then case-insensitive
        sentiment = label_mapping.get(label, label_mapping.get(label.lower(), label.lower()))
    
    return sentiment, round(score, 2)


def evaluate_model(model_config: dict, test_df: pd.DataFrame) -> Dict:
    """Evaluate a single model on the test dataset."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_config['name']}")
    print(f"Model ID: {model_config['model_id']}")
    print(f"Description: {model_config.get('description', 'N/A')}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Load model
    try:
        pipe = load_model_pipeline(model_config)
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return None
    
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.2f}s")
    
    # Run predictions
    inference_start = time.time()
    print(f"  Running predictions on {len(test_df)} samples...")
    predictions = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        try:
            sentiment, confidence = predict_with_model(pipe, row['text'], model_config)
            predictions.append(sentiment)
            confidences.append(confidence)
        except Exception as e:
            print(f"  Warning: Prediction failed for sample {idx}: {e}")
            predictions.append("neutral")
            confidences.append(50.0)
        
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(test_df)}...")
    
    inference_time = time.time() - inference_start
    
    # Prepare data
    y_true = test_df['expected_sentiment'].str.lower().str.strip()
    y_pred = pd.Series(predictions).str.lower().str.strip()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision_macro = precision_score(y_true, y_pred, labels=LABELS, 
                                     average='macro', zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, labels=LABELS,
                               average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, labels=LABELS,
                       average='macro', zero_division=0) * 100
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    
    class_metrics = {}
    for i, label in enumerate(LABELS):
        support = (y_true == label).sum()
        class_metrics[label] = {
            'precision': round(precision_per_class[i] * 100, 2),
            'recall': round(recall_per_class[i] * 100, 2),
            'f1_score': round(f1_per_class[i] * 100, 2),
            'support': int(support)
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    
    # Print results
    print(f"\n  RESULTS:")
    print(f"  {'-'*50}")
    print(f"  Accuracy:           {accuracy:.2f}%")
    print(f"  Precision (macro):  {precision_macro:.2f}%")
    print(f"  Recall (macro):     {recall_macro:.2f}%")
    print(f"  F1-Score (macro):   {f1_macro:.2f}%")
    print(f"\n  Per-Class Metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
    for label in LABELS:
        m = class_metrics[label]
        print(f"  {label.capitalize():<12} {m['precision']:>9.2f}% {m['recall']:>9.2f}% {m['f1_score']:>9.2f}% {m['support']:>8}")
    
    print(f"\n  Inference time: {inference_time:.2f}s ({inference_time/len(test_df)*1000:.1f}ms per sample)")
    
    return {
        'model_name': model_config['name'],
        'model_id': model_config['model_id'],
        'description': model_config.get('description', ''),
        'accuracy': round(accuracy, 2),
        'precision_macro': round(precision_macro, 2),
        'recall_macro': round(recall_macro, 2),
        'f1_macro': round(f1_macro, 2),
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist(),
        'inference_time': round(inference_time, 2),
        'time_per_sample_ms': round(inference_time / len(test_df) * 1000, 1),
        'predictions': predictions,
        'confidences': confidences
    }


def print_comparison_table(results: List[Dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n\n{'='*100}")
    print("MODEL COMPARISON SUMMARY - ALL 3-CLASS MODELS")
    print(f"{'='*100}")
    
    # Sort by F1-macro (primary metric for imbalanced datasets)
    sorted_results = sorted(results, key=lambda x: x['f1_macro'], reverse=True)
    
    print(f"\n{'Model':<35} {'Accuracy':>10} {'F1-Macro':>10} {'Neg Prec':>10} {'Neu Prec':>10} {'Pos Prec':>10} {'ms/sample':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for i, result in enumerate(sorted_results):
        neg_prec = result['class_metrics']['negative']['precision']
        neu_prec = result['class_metrics']['neutral']['precision']
        pos_prec = result['class_metrics']['positive']['precision']
        
        # Mark best model
        marker = "🏆 " if i == 0 else "   "
        
        print(f"{marker}{result['model_name'][:32]:<32} {result['accuracy']:>9.2f}% "
              f"{result['f1_macro']:>9.2f}% {neg_prec:>9.2f}% {neu_prec:>9.2f}% "
              f"{pos_prec:>9.2f}% {result['time_per_sample_ms']:>9.1f}")
    
    print(f"{'='*100}")
    
    # Print insights
    best = sorted_results[0]
    fastest = min(results, key=lambda x: x['time_per_sample_ms'])
    best_neg = max(results, key=lambda x: x['class_metrics']['negative']['precision'])
    
    print(f"\n📊 INSIGHTS:")
    print(f"  • Best Overall (F1-macro): {best['model_name']} ({best['f1_macro']:.2f}%)")
    print(f"  • Fastest Inference: {fastest['model_name']} ({fastest['time_per_sample_ms']:.1f}ms/sample)")
    print(f"  • Best Negative Precision: {best_neg['model_name']} ({best_neg['class_metrics']['negative']['precision']:.2f}%)")
    
    # Speed vs accuracy tradeoff
    if fastest['model_name'] != best['model_name']:
        speed_ratio = best['time_per_sample_ms'] / fastest['time_per_sample_ms']
        f1_diff = best['f1_macro'] - fastest['f1_macro']
        print(f"\n  ⚡ Speed-Accuracy Tradeoff:")
        print(f"     {fastest['model_name']} is {speed_ratio:.1f}x faster than {best['model_name']}")
        print(f"     but {f1_diff:.2f}% lower F1-score")


def main():
    """Run model comparison."""
    # Load test data
    input_csv = "data/sentiment_test_cases_2025.csv"
    
    # Try multiple paths
    possible_paths = [
        input_csv,
        "../data/sentiment_test_cases_2025.csv",
        "/mnt/user-data/uploads/sentiment_test_cases_2025.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading test data from: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        print("Error: Could not find test data file")
        sys.exit(1)
    
    print(f"Loaded {len(df)} test cases")
    
    # Show class distribution
    dist = df['expected_sentiment'].value_counts()
    print(f"\nClass Distribution:")
    for label, count in dist.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Evaluate all models
    results = []
    for model_config in MODELS_TO_TEST:
        result = evaluate_model(model_config, df)
        if result:
            results.append(result)
            
            # Save individual output CSV
            safe_model_id = model_config['model_id'].replace('/', '_')
            output_csv = f"data/model_comparison/output_{safe_model_id}.csv"
            
            # Create output directory if needed
            os.makedirs("data/model_comparison", exist_ok=True)
            
            output_df = df.copy()
            output_df['model_output'] = result['predictions']
            output_df['confidence_score'] = result['confidences']
            output_df[['text', 'expected_sentiment', 'model_output', 'confidence_score']].to_csv(
                output_csv, index=False
            )
            print(f"  ✅ Saved predictions to: {output_csv}")
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save comparison results
    comparison_file = "data/model_comparison/model_comparison_results.json"
    with open(comparison_file, 'w') as f:
        # Remove predictions/confidences for cleaner JSON
        clean_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items() if k not in ['predictions', 'confidences']}
            clean_results.append(clean_r)
        json.dump(clean_results, f, indent=2)
    
    print(f"\n✅ Comparison results saved to: {comparison_file}")
    
    # Final recommendation
    best_model = max(results, key=lambda x: x['f1_macro'])
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f"🏆 Best Model: {best_model['model_name']}")
    print(f"   Model ID: {best_model['model_id']}")
    print(f"   F1-Score (macro): {best_model['f1_macro']:.2f}%")
    print(f"   Accuracy: {best_model['accuracy']:.2f}%")
    print(f"   Negative Precision: {best_model['class_metrics']['negative']['precision']:.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
