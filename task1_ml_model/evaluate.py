"""
Task 1: ML Foundation - Batch Evaluation Script

Reads the test CSV, runs predictions, and outputs results with comprehensive metrics.

Usage:
    python evaluate.py --input ../data/sentiment_test_cases_2025.csv --output ../data/output_sentiment_test.csv
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task1_ml_model.sentiment_classifier import get_classifier

# Class labels in order
LABELS = ['negative', 'neutral', 'positive']


def print_confusion_matrix(cm: np.ndarray, labels: list) -> None:
    """Print a formatted confusion matrix."""
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 " + "  ".join(f"{l[:3]:>6}" for l in labels))
    print("              +" + "-" * (8 * len(labels)))
    for i, label in enumerate(labels):
        row = "  ".join(f"{val:>6}" for val in cm[i])
        print(f"Actual {label[:3]:>6} | {row}")


def evaluate_model(
    input_csv: str = "data/sentiment_test_cases_2025.csv",
    output_csv: str = "data/output_sentiment_test.csv"
) -> dict:
    """
    Run batch evaluation on the test dataset.
    
    Args:
        input_csv: Path to input CSV with 'text' and 'expected_sentiment' columns
        output_csv: Path to output CSV with predictions
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    # Load test data
    print(f"Loading test data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Validate columns
    required_columns = ['text', 'expected_sentiment']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"Loaded {len(df)} test cases")
    
    # Initialize classifier
    classifier = get_classifier()
    classifier.load_model()
    
    # Run predictions
    print("Running predictions...")
    predictions = []
    confidences = []
    
    for idx, row in df.iterrows():
        text = row['text']
        sentiment, confidence = classifier.predict(text)
        predictions.append(sentiment)
        confidences.append(confidence)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    # Add predictions to dataframe
    df['model_output'] = predictions
    df['confidence_score'] = confidences
    
    # Normalize labels for comparison
    y_true = df['expected_sentiment'].str.lower().str.strip()
    y_pred = df['model_output'].str.lower().str.strip()
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    
    # Macro averages (unweighted mean across classes)
    precision_macro = precision_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0) * 100
    
    # Weighted averages (weighted by support)
    precision_weighted = precision_score(y_true, y_pred, labels=LABELS, average='weighted', zero_division=0) * 100
    recall_weighted = recall_score(y_true, y_pred, labels=LABELS, average='weighted', zero_division=0) * 100
    f1_weighted = f1_score(y_true, y_pred, labels=LABELS, average='weighted', zero_division=0) * 100
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    
    # Build per-class metrics dictionary
    class_metrics = {}
    for i, label in enumerate(LABELS):
        support = (y_true == label).sum()
        class_metrics[label] = {
            'precision': round(precision_per_class[i] * 100, 2),
            'recall': round(recall_per_class[i] * 100, 2),
            'f1_score': round(f1_per_class[i] * 100, 2),
            'support': int(support)
        }
    
    # Save output CSV (only required columns per spec)
    output_df = df[['text', 'expected_sentiment', 'model_output', 'confidence_score']]
    output_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"\n{'OVERALL METRICS':^60}")
    print("-" * 60)
    print(f"  Accuracy:           {accuracy:.2f}%")
    print(f"  Precision (macro):  {precision_macro:.2f}%")
    print(f"  Recall (macro):     {recall_macro:.2f}%")
    print(f"  F1-Score (macro):   {f1_macro:.2f}%")
    print(f"\n  Precision (weighted): {precision_weighted:.2f}%")
    print(f"  Recall (weighted):    {recall_weighted:.2f}%")
    print(f"  F1-Score (weighted):  {f1_weighted:.2f}%")
    
    print(f"\n{'PER-CLASS METRICS':^60}")
    print("-" * 60)
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for label in LABELS:
        m = class_metrics[label]
        print(f"  {label.capitalize():<12} {m['precision']:>9.2f}% {m['recall']:>9.2f}% {m['f1_score']:>9.2f}% {m['support']:>10}")
    
    print_confusion_matrix(cm, LABELS)
    print("=" * 60)
    
    return {
        'total': len(df),
        'accuracy': round(accuracy, 2),
        'precision_macro': round(precision_macro, 2),
        'recall_macro': round(recall_macro, 2),
        'f1_macro': round(f1_macro, 2),
        'precision_weighted': round(precision_weighted, 2),
        'recall_weighted': round(recall_weighted, 2),
        'f1_weighted': round(f1_weighted, 2),
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist()
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate sentiment model on test data")
    parser.add_argument(
        "--input", 
        default="data/sentiment_test_cases_2025.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        default="data/output_sentiment_test.csv", 
        help="Path to output CSV file"
    )
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(args.input, args.output)
        
        # Exit with error if accuracy below threshold
        if results['accuracy'] < 80:
            print(f"\n⚠️  WARNING: Accuracy ({results['accuracy']}%) is below 80% threshold!")
            sys.exit(1)
        else:
            print(f"\n✅ Accuracy ({results['accuracy']}%) meets the 80% threshold!")
            sys.exit(0)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the input CSV file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
