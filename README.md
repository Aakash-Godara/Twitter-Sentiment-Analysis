# Twitter Sentiment Analysis: Classical ML vs. Deep Learning at Scale

## Business Problem

**Can we automatically classify the sentiment of a tweet as positive or negative from raw text alone, cheaply enough to run on a live stream of brand mentions, product feedback, or support tickets?**

This project builds an end-to-end tweet sentiment classifier on 1.6M real tweets, trains and rigorously compares a classical ML pipeline (TF-IDF + Logistic Regression) against a deep learning approach (Bidirectional LSTM), and selects a production model based on accuracy, training/inference cost, and interpretability not by defaulting to the deepest architecture available.

## Dataset
Kaggle — kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Key EDA Findings

- **Classes are almost perfectly balanced:** 798,299 negative(50.02%) vs 797,944 positive(49.98%) no class-imbalance handling needed.
- **Median tweet is 12 words** after cleaning (mean 13.0, IQR 7–18), consistent with the 140-character limit in effect when this dataset was collected (2009).
- Positive tweets skew toward *thank, love, good, haha, awesome*; negative tweets skew toward *want, miss, sad, work, ugh* (see word clouds).

| Chart | So what |
|---|---|
| ![Class Balance](images/class_balance.png) | Confirms the 50/50 split — accuracy is a meaningful metric here, unlike on an imbalanced dataset where it would be misleading. |
| ![Tweet Length Distribution](images/tweet_length_distribution.png) | Both classes have near-identical length distributions — tweet length itself is not a useful feature for sentiment. |
| ![Top Words by Sentiment](images/top_words_by_sentiment.png) | Positive tweets cluster around gratitude/excitement vocabulary; negative tweets cluster around complaints and negation — validates that TF-IDF bigrams (e.g. "do not", "can not") will carry real signal. |
| ![Word Cloud Positive](images/wordcloud_positive.png) | Gratitude and excitement vocabulary dominates. |
| ![Word Cloud Negative](images/wordcloud_negative.png) | Complaints, negation, and loss-related words dominate. |

## Model Performance

Split: **80/20 train/test, stratified on sentiment, `random_state=42`**.

| Metric | Logistic Regression (primary) | Naive Bayes (baseline) | BiLSTM (deep learning) |
|---|---|---|---|
| Training data | 1,276,994 rows (full) | 1,276,994 rows (full) | 240,000 rows (subsample*) |
| Accuracy | **0.8214** | 0.7995 | 0.8051 |
| Precision | 0.8152 | 0.8040 | 0.8105 |
| Recall | **0.8312** | 0.7920 | 0.7962 |
| F1 | **0.8231** | 0.7980 | 0.8033 |
| ROC-AUC | **0.9009** | 0.8810 | 0.8854 |
| Training time | 8.4s | **0.2s** | 102s |


## Model Selection

TF-IDF + Logistic Regression is selected as the production model:

- Best ROC-AUC of all three (0.901)— beats even the BiLSTM (0.885), despite being the simplest architecture and training on 5x less compute-per-row than a neural network requires.
- approax 12x faster to train than the BiLSTM (8.4s vs. 102s) on 5x more data** — a >60x effective throughput advantage that matters directly for how often the model can be retrained as language/slang drifts.
- Interpretable by construction — its coefficients are per-word/bigram log-odds contributions, inspectable directly, unlike the BiLSTM's hidden state which needs extra tooling (LIME/SHAP) to explain a prediction.

The BiLSTM is kept in the repo as a **documented comparison, not discarded** — it shows the classical-vs-deep-learning trade-off honestly rather than assuming deep learning wins by default. With the full 1.6M tweets, more epochs, and pretrained embeddings (e.g. GloVe), the BiLSTM would likely close the gap; that's future work, not a claim made here.
