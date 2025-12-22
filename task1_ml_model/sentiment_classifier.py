"""
Task 1: ML Foundation - Sentiment Classification Model

This module provides a sentiment classifier using HuggingFace transformers.
Uses cardiffnlp/twitter-roberta-base-sentiment-latest for 3-class
sentiment classification (negative, neutral, positive).

Usage:
    python sentiment_classifier.py --text "I love this product!"
"""
import re
from typing import Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Label mapping for the cardiffnlp model
LABEL_MAPPING = {
    0: "negative",
    1: "neutral", 
    2: "positive"
}


def preprocess_text(text: str) -> str:
    """
    Preprocess text for the Twitter-RoBERTa sentiment model.
    
    - Replace @mentions with @user
    - Replace URLs with http
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text ready for model inference
    """
    new_text = []
    for token in text.split(" "):
        token = "@user" if token.startswith("@") and len(token) > 1 else token
        token = "http" if token.startswith("http") else token
        new_text.append(token)
    return " ".join(new_text)


class SentimentClassifier:
    """
    Sentiment classifier using HuggingFace transformers.
    
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest for 3-class
    sentiment classification (negative, neutral, positive).
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_name: HuggingFace model identifier (defaults to SENTIMENT_MODEL_NAME env var)
        """
        self.model_name = model_name or os.getenv(
            "SENTIMENT_MODEL_NAME", 
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        if self._is_loaded:
            logger.info("Model already loaded, skipping...")
            return
            
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self._is_loaded = True
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Required dependencies not installed: {e}")
            raise RuntimeError(
                "Model dependencies not available. Please install: "
                "pip install transformers torch scipy"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(
                f"Failed to load sentiment model '{self.model_name}': {e}"
            ) from e
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
            - sentiment_label: "positive", "neutral", or "negative"
            - confidence_score: Float between 0-100
        """
        if not self._is_loaded:
            self.load_model()
        
        try:
            import torch
            from scipy.special import softmax
            import numpy as np
            
            # Preprocess
            processed_text = preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.numpy()[0]
            
            # Get probabilities
            probabilities = softmax(logits)
            
            # Get prediction
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class]) * 100
            
            sentiment_label = LABEL_MAPPING[predicted_class]
            
            return sentiment_label, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed for text: {e}") from e
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded


# Singleton instance
_classifier_instance = None


def get_classifier() -> SentimentClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SentimentClassifier()
    return _classifier_instance


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Sentiment Classification")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    args = parser.parse_args()
    
    classifier = get_classifier()
    classifier.load_model()
    
    sentiment, confidence = classifier.predict(args.text)
    
    result = {
        "model_output": sentiment,
        "confidence_score": confidence
    }
    
    print(json.dumps(result, indent=2))
