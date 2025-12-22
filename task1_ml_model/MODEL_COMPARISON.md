# Sentiment Model Comparison Study

## Executive Summary

This document summarizes a **model comparison study** I use as the reference for why I selected `cardiffnlp/twitter-roberta-base-sentiment-latest` over other comparable 3-class sentiment models.

To keep the comparison fair, I evaluated only models with **native 3-class output** (negative/neutral/positive) on the same 498-sample test set.

Based on the results, `cardiffnlp/twitter-roberta-base-sentiment-latest` provides the best overall balance, with the top **F1-macro (86.60%)** and tied-best **accuracy (86.75%)**.

---

## Models Evaluated

| # | Model                                      | Size   | Training Data           | Classes |
| - | ------------------------------------------ | ------ | ----------------------- | ------- |
| 1 | **Twitter-RoBERTa-Latest** (Current) | ~500MB | 124M tweets (2018-2021) | 3-class |
| 2 | Twitter-RoBERTa-Original                   | ~500MB | 58M tweets              | 3-class |
| 3 | BERTweet-Sentiment                         | ~540MB | 850M tweets             | 3-class |
| 4 | RoBERTa-Large-3Class                       | ~1.4GB | 5.3K social media posts | 3-class |
| 5 | DistilBERT-Multilingual-3Class             | ~540MB | Distilled multilingual  | 3-class |

---

## Performance Comparison

### Overall Metrics

| Model                               | Accuracy         | Precision        | Recall           | F1-Score         | Speed (ms/sample) |
| ----------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| **Twitter-RoBERTa-Latest** ✅ | **86.75%** | **86.87%** | **86.68%** | **86.60%** | 36.1ms            |
| BERTweet-Sentiment                  | 86.75%           | 86.73%           | 86.25%           | 86.36%           | 41.3ms            |
| Twitter-RoBERTa-Original            | 83.53%           | 83.56%           | 83.55%           | 83.38%           | 37.0ms            |
| RoBERTa-Large-3Class                | 81.73%           | 81.92%           | 81.69%           | 81.69%           | 145.2ms           |
| DistilBERT-Multilingual-3Class      | 58.63%           | 53.51%           | 54.34%           | 47.23%           | 21.9ms ⚡         |

### Per-Class Performance

#### Twitter-RoBERTa-Latest (Current Model)

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Negative | 93.59%    | 82.49% | 87.69%   | 177     |
| Neutral  | 82.76%    | 86.33% | 84.51%   | 139     |
| Positive | 84.26%    | 91.21% | 87.60%   | 182     |

#### BERTweet-Sentiment

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Negative | 93.83%    | 85.88% | 89.68%   | 177     |
| Neutral  | 82.35%    | 80.58% | 81.45%   | 139     |
| Positive | 84.00%    | 92.31% | 87.96%   | 182     |

#### Twitter-RoBERTa-Original

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Negative | 90.38%    | 79.66% | 84.68%   | 177     |
| Neutral  | 78.00%    | 84.17% | 80.97%   | 139     |
| Positive | 82.29%    | 86.81% | 84.49%   | 182     |

**Note:** Older version trained on 58M tweets - "Latest" version shows 3.2% F1 improvement

#### RoBERTa-Large-3Class

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Negative | 88.96%    | 81.92% | 85.29%   | 177     |
| Neutral  | 68.06%    | 93.53% | 78.79%   | 139     |
| Positive | 91.67%    | 72.53% | 80.98%   | 182     |

**Note:** Larger model but 4x slower with lower overall performance

#### DistilBERT-Multilingual-3Class

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Negative | 72.04%    | 75.71% | 73.83%   | 177     |
| Neutral  | 37.50%    | 2.16%  | 4.08%    | 139     |
| Positive | 50.99%    | 85.16% | 63.79%   | 182     |

**Note:** Catastrophic neutral detection (2.16% recall) - not suitable for production

---

## Detailed Analysis

### 1. Twitter-RoBERTa-Latest (Current Model) ✅

**Strengths:**

- ✅ **Highest F1-score (86.60%)** - best balanced performance
- ✅ **Best precision on negative class (93.59%)** - critical for escalation
- ✅ Trained on 124M tweets - robust to social media language
- ✅ Handles all 3 classes (negative, neutral, positive)

**Weaknesses:**

- ⚠️ Not the fastest overall (DistilBERT-Multilingual is faster at 21.9ms)
- ⚠️ Not the best negative precision (BERTweet is slightly higher)
- ⚠️ Larger model size (~500MB)

**Verdict:** **BEST CHOICE** - Best overall F1-macro with strong performance across all classes.

---

### 2. BERTweet-Sentiment

**Strengths:**

- ✅ Same accuracy (86.75%)
- ✅ Trained on 850M tweets - even more robust
- ✅ Handles all 3 classes
- ✅ Best negative precision (93.83%)

**Weaknesses:**

- ⚠️ Slightly lower F1-score (86.36% vs 86.60%)
- ⚠️ 14% slower than Twitter-RoBERTa-Latest (41.3ms vs 36.1ms)

**Verdict:** **STRONG ALTERNATIVE** - Nearly identical performance, slightly slower.

---

### 3. Twitter-RoBERTa-Original

**Strengths:**

- ✅ Same architecture as Latest version
- ✅ Fast inference (37.0ms)

**Weaknesses:**

- ❌ **3.2% lower F1-score** (83.38% vs 86.60%)
- ❌ Trained on older dataset (58M tweets)
- ❌ Lower neutral precision (78.00% vs 82.76%)

**Verdict:** **SUPERSEDED** - The "Latest" version is clearly improved.

---

### 4. RoBERTa-Large-3Class

**Strengths:**

- ✅ Best positive precision (91.67%)
- ✅ Larger model capacity

**Weaknesses:**

- ❌ **4x slower** (145.2ms per sample)
- ❌ Lower overall F1-score (81.69%)
- ❌ Poor neutral precision (68.06%)
- ❌ Trained on only 5.3K samples

**Verdict:** **NOT SUITABLE** - Larger model doesn't mean better performance.

---

### 5. DistilBERT-Multilingual-3Class

**Strengths:**

- ✅ **Fastest model** (21.9ms per sample)
- ✅ Multilingual support

**Weaknesses:**

- ❌ **Catastrophic neutral detection** (2.16% recall, 37.50% precision)
- ❌ Very low F1-score (47.23%)
- ❌ Poor overall accuracy (58.63%)

**Verdict:** **NOT SUITABLE** - Speed advantage is meaningless with such poor accuracy.

---

## Speed vs Accuracy Trade-off

```
                    Speed (ms/sample, lower is better)
                    ↑
    DistilBERT-Multi ●  |  21.9ms (58.63% acc) - Too inaccurate
                        |
    RoBERTa-Latest   ●  |  36.1ms (86.75% acc) ← BEST F1
    RoBERTa-Original ●  |  37.0ms (83.53% acc)
    BERTweet         ●  |  41.3ms (86.75% acc)
                        |
    RoBERTa-Large    ●  |  145.2ms (81.73% acc) - Too slow
                        |
                        └─────────────────────────→
                                  Accuracy
```

---

## Why This Model Wins

### 1. Highest F1-Score (86.60%)

F1-score is the **harmonic mean** of precision and recall - a strong single-number summary of balanced performance. This model has the best F1.

### 2. Best Negative Precision (93.59%)

For a support agent, **false escalations are costly**. This model has very high negative precision, meaning when it triggers an escalation, it's usually correct.

Note: BERTweet is slightly higher on negative precision (93.83% vs 93.59%), but this model has the top F1-macro overall.

### 3. Robust to Social Media Language

Trained on 124M tweets, it handles:

- Informal language
- Slang and abbreviations
- @mentions and URLs
- Emojis and emoticons

### 4. Three-Class Support

Unlike some alternatives, this model handles **neutral sentiment**, which is critical for:

- Asking clarifying questions
- Avoiding unnecessary escalations
- Understanding ambiguous messages

---

## When to Consider Alternatives

### Use BERTweet if:

- ✅ You can accept 0.24% lower F1-score
- ✅ You want slightly better negative precision (93.83% vs 93.59%)

### Don't use Twitter-RoBERTa-Original:

- ❌ Superseded by the "Latest" version with 3.2% better F1

### Don't use RoBERTa-Large-3Class:

- ❌ 4x slower with worse performance
- ❌ Larger model size doesn't help

### Don't use DistilBERT-Multilingual:

- ❌ Catastrophic neutral detection failure
- ❌ Unacceptable accuracy for production

---

## Conclusion

This study is the reference for my model selection. `cardiffnlp/twitter-roberta-base-sentiment-latest` is the best overall choice from the evaluated models:

| Metric             | Value  | Rank                      |
| ------------------ | ------ | ------------------------- |
| Accuracy           | 86.75% | 🥇 Tied 1st               |
| F1-Score           | 86.60% | 🥇 1st                    |
| Negative Precision | 93.59% | 🥈 2nd (BERTweet: 93.83%) |
| Speed              | 36.1ms | 🥈 2nd fastest            |

**Bottom line:** It provides the **best balanced performance** for my use case, with strong negative/neutral handling which is critical for escalation decisions.

---

## Files Generated

| File                                                                                              | Description                  |
| ------------------------------------------------------------------------------------------------- | ---------------------------- |
| `data/model_comparison/output_cardiffnlp_twitter-roberta-base-sentiment-latest.csv`             | Current model predictions    |
| `data/model_comparison/output_cardiffnlp_twitter-roberta-base-sentiment.csv`                    | Original RoBERTa predictions |
| `data/model_comparison/output_finiteautomata_bertweet-base-sentiment-analysis.csv`              | BERTweet predictions         |
| `data/model_comparison/output_j-hartmann_sentiment-roberta-large-english-3-classes.csv`         | RoBERTa-Large predictions    |
| `data/model_comparison/output_lxyuan_distilbert-base-multilingual-cased-sentiments-student.csv` | DistilBERT predictions       |
| `data/model_comparison/model_comparison_results.json`                                           | JSON summary of all results  |

---

*Comparison conducted on 498 test cases from `sentiment_test_cases_2025.csv`*
*All models evaluated are native 3-class for fair comparison*
*Evaluation date: December 2025*
