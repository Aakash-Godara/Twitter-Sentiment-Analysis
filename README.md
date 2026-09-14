# Twitter Sentiment Analysis: Classical ML vs. Deep Learning at Scale

[![CI](https://github.com/Aakash-Godara/twitter-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Aakash-Godara/twitter-sentiment-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Business Problem

**Can we automatically classify the sentiment of a tweet as positive or negative from raw text alone, cheaply enough to run on a live stream of brand mentions, product feedback, or support tickets?**

This project builds an end-to-end tweet sentiment classifier on 1.6M real tweets, trains and rigorously compares a classical ML pipeline (TF-IDF + Logistic Regression) against a deep learning approach (Bidirectional LSTM), and selects a production model based on accuracy, training/inference cost, and interpretability — not by defaulting to the deepest architecture available.

## Dataset

- **Name:** Sentiment140
- **Source:** [Kaggle — kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **How obtained:** Downloaded programmatically via the `kagglehub` Python library (`kagglehub.dataset_download("kazanova/sentiment140")`), no API token required for this public dataset. See [`fetch_data.py`](fetch_data.py).
- **Verification:** Before analysis, the loaded dataframe is asserted to have **exactly 1,600,000 rows and 6 columns**, and the target column is checked to contain only `{0, 4}` (negative/positive). This check runs in `fetch_data.py` and again at the top of the notebook.
- **Note on labels:** Sentiment140 is auto-labeled from emoticons (`:)` → positive, `:(` → negative), not hand-annotated — that's *why* it can exist at 1.6M rows. This is a disclosed property of the dataset, not a hidden assumption.

## Approach

1. **Data cleaning** — HTML-unescape, strip URLs/@mentions, expand contractions, strip punctuation/digits, drop tweets that become empty.
2. **Exploratory analysis** — class balance, tweet-length distribution, top words and word clouds by sentiment.
3. **Feature engineering** — TF-IDF (unigrams + bigrams, 60K features) for the classical models; a Keras `Tokenizer` + padded sequences for the LSTM.
4. **Modelling** — TF-IDF + Logistic Regression (primary) vs. TF-IDF + Multinomial Naive Bayes (fast baseline) vs. Bidirectional LSTM (deep learning comparison), on an 80/20 stratified train/test split (`random_state=42`).
5. **Model selection** — chosen on accuracy, ROC-AUC, training time, and interpretability together, not accuracy alone.
6. **Deployment** — saved model artifacts served three ways (CLI, FastAPI, Streamlit), containerised with Docker, tested and linted automatically via GitHub Actions CI.

Full working code: [`notebooks/01_sentiment_analysis.ipynb`](notebooks/01_sentiment_analysis.ipynb) (executed end-to-end against the real 1.6M-row dataset — every number below is copied directly from its output cells).

## Data Cleaning

Each tweet goes through one shared cleaning function ([`preprocessing.py`](preprocessing.py)), used identically at training and inference time: HTML-unescape → lowercase → strip URLs → strip @mentions → expand contractions (`can't` → `cannot`) → strip punctuation/digits → collapse whitespace.

**1,600,000 → 1,596,243 tweets** survive cleaning (3,757 tweets, 0.23%, became empty strings after removing URLs/mentions and were dropped — e.g. tweets that were only a link or an @mention).

## Key EDA Findings (real numbers, from the notebook output)

- **Classes are almost perfectly balanced:** 798,299 negative (50.02%) vs. 797,944 positive (49.98%) — no class-imbalance handling needed.
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

\* *Training the BiLSTM on the full 1.6M-tweet corpus for multiple epochs is impractical on a single Apple M3 CPU within a reasonable session — this project trains it on a stratified 300K-tweet subsample (240K train / 60K test) instead. This is a disclosed scope decision; see [Methodological Choices](#methodological-choices-every-judgment-call-disclosed).*

![ROC Curve Comparison](images/roc_curve_comparison.png)
![Model Comparison](images/model_comparison.png)
![Confusion Matrices](images/confusion_matrices.png)
![Training Time Comparison](images/training_time_comparison.png)

## Model Selection

**TF-IDF + Logistic Regression is selected as the production model:**

- **Best ROC-AUC of all three (0.901)** — beats even the BiLSTM (0.885), despite being the simplest architecture and training on 5x less compute-per-row than a neural network requires.
- **~12x faster to train than the BiLSTM (8.4s vs. 102s) on 5x more data** — a >60x effective throughput advantage that matters directly for how often the model can be retrained as language/slang drifts.
- **Interpretable by construction** — its coefficients are per-word/bigram log-odds contributions, inspectable directly, unlike the BiLSTM's hidden state which needs extra tooling (LIME/SHAP) to explain a prediction.
- **No GPU required, loads in milliseconds** — cheap to serve in the CLI (`predict.py`) and the [Streamlit demo](app/streamlit_app.py).

The BiLSTM is kept in the repo as a **documented comparison, not discarded** — it shows the classical-vs-deep-learning trade-off honestly rather than assuming deep learning wins by default. With the full 1.6M tweets, more epochs, and pretrained embeddings (e.g. GloVe), the BiLSTM would likely close the gap; that's future work, not a claim made here.

## Explainability

Because the production model is linear, `coefficient × TF-IDF weight` for each word/bigram present in a tweet **is** the model's reasoning — not an approximation the way SHAP/LIME would need to produce for a non-linear model. [`predict.py`](predict.py)'s `explain_prediction()` surfaces this directly:

```
$ python predict.py "Flight got cancelled again and nobody at the desk will help me"
Text:       Flight got cancelled again and nobody at the desk will help me
Sentiment:  Negative
Confidence: 0.043 (probability of Positive)
Top contributing words/bigrams:
  'cancelled'          - negative  (contribution=-1.822)
  'nobody'             - negative  (contribution=-0.971)
  'will help'          + positive  (contribution=+0.368)
  'and nobody'         - negative  (contribution=-0.273)
  'help me'            - negative  (contribution=-0.261)
```

This is exposed in the [FastAPI service](#serving-the-model) (`top_contributions` in the response) and is the kind of per-prediction justification a stakeholder can actually audit — e.g. a support team asking "why did the model flag this tweet as negative?".

## Methodological Choices (every judgment call, disclosed)

| Choice | What it is | Why | Design decision or constraint? |
|---|---|---|---|
| BiLSTM trained on a 300K-tweet subsample, not the full 1.6M | Stratified `train_test_split(train_size=300_000, stratify=target, random_state=42)` | Multiple LSTM epochs over 1.28M rows on a single CPU (no GPU available) would take far longer than fits in one working session | **Hardware constraint**, disclosed — not hidden by only reporting favorable numbers |
| TF-IDF capped at 60,000 features, unigrams + bigrams, `min_df=5` | Vocabulary size cutoff | Keeps the sparse matrix memory-manageable (8GB RAM) while still capturing negation bigrams like "do not" | Design decision |
| LSTM: vocab 20K, sequence length 40, embedding dim 128, 64 LSTM units, early stopping (patience=2, monitor=val_loss) | Standard, reasonable architecture — not hyperparameter-tuned | Keeps the deep-learning comparison a fair, honest baseline rather than an artificially tuned ceiling | Design decision, disclosed |
| Classification threshold = 0.5 | Standard default, not tuned | Tuning it needs a real deployment context (e.g. cost of a false-negative brand complaint) this dataset doesn't supply | Design decision |
| Dropped 3,757 tweets that became empty after cleaning | Tweets that were only a URL or @mention | These carry no sentiment-bearing text once cleaned; keeping them would only add label noise | Data-derived (verified: all had 0 words after cleaning) |
| No stemming step | Words are used as-is after cleaning | TF-IDF bigrams and the LSTM's own embeddings capture enough signal without it | Design decision |

## Serving the Model

Three ways to use the trained model, from quickest to most production-like:

**1. CLI** — one-off predictions with the full explainability breakdown:
```bash
python predict.py "this made my whole week, thank you!"
```

**2. Streamlit demo** — interactive UI for non-technical stakeholders:
```bash
streamlit run app/streamlit_app.py
```

**3. FastAPI service** — a real HTTP API, the form a model actually takes in production:
```bash
uvicorn api.main:app --reload
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "this made my whole week, thank you!"}'
```
Interactive docs (via FastAPI's auto-generated OpenAPI schema) at `http://localhost:8000/docs`.

**Containerised**, both services with one command:
```bash
docker compose up --build
# API:      http://localhost:8000/docs
# Demo:     http://localhost:8501
```
The API image (`Dockerfile`) installs only [`requirements-api.txt`](requirements-api.txt) (scikit-learn, FastAPI, no TensorFlow/Jupyter) — a deliberately lean, fast-to-build production image, separate from the full research environment used for the notebook.

## Model Card: Limitations & Ethical Considerations

- **Label source is a proxy, not ground truth.** Sentiment140's labels come from emoticons in the original tweet, not human annotation. Sarcasm, mixed sentiment, and emoticon-free negative/positive tweets are systematically mislabelled at the source — this ceiling applies to *any* model trained on this data, not just the ones here.
- **Binary only, no neutral class.** A genuinely neutral tweet ("the train arrives at 9am") is forced into Positive or Negative by the model, since Sentiment140 has no neutral label to learn from. This is a real limitation for brand-monitoring use cases, where a large share of mentions are neutral.
- **Temporal and demographic skew.** Collected in 2009, in English, from Twitter's user base at the time. Slang, abbreviations, and platform norms have shifted substantially since; performance on 2026 tweets, other languages, or other platforms (e.g. TikTok comments) is untested and not assumed to transfer.
- **No fairness/bias audit performed.** Twitter text carries no demographic labels, so subgroup performance (e.g. by dialect, such as African-American Vernacular English, which NLP models are documented elsewhere to sometimes misclassify at higher rates) could not be measured here. This is disclosed as an open gap, not asserted as "unbiased."
- **Not validated for high-stakes decisions.** This model is appropriate for aggregate trend-monitoring (e.g. "is sentiment about our product shifting week over week?"), not for decisions about individual people (e.g. moderation, hiring signal, credit) without substantially more validation.
- **Intended use:** exploratory brand/product sentiment monitoring at scale, as a first-pass filter for human review — not an autonomous decision-maker.

## Repository Structure

```
twitter-sentiment-analysis/
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions: lint (ruff) + tests on every push/PR
├── data/
│   └── raw/                          # sentiment140.csv (fetched, not committed)
├── notebooks/
│   └── 01_sentiment_analysis.ipynb   # end-to-end analysis, executed against the real data
├── images/                           # chart PNGs exported from the notebook
├── models/                           # saved TF-IDF vectorizer, LogReg/NB/BiLSTM models, tokenizer
├── api/
│   └── main.py                       # FastAPI service (/predict, /health)
├── app/
│   └── streamlit_app.py              # interactive Streamlit demo
├── tests/
│   ├── test_preprocessing.py         # unit tests for the cleaning function
│   ├── test_predict.py               # unit tests for prediction + explainability
│   └── test_api.py                   # API integration tests (FastAPI TestClient)
├── fetch_data.py                     # downloads + verifies the dataset via kagglehub
├── preprocessing.py                  # shared text-cleaning function (train + inference)
├── predict.py                        # CLI inference + explainability
├── Dockerfile                        # lean image for the FastAPI service
├── Dockerfile.streamlit              # image for the Streamlit demo
├── docker-compose.yml                # runs both services together
├── pyproject.toml                    # ruff lint config
├── .pre-commit-config.yaml           # ruff pre-commit hook
├── requirements.txt                  # full research/notebook environment
├── requirements-api.txt              # minimal runtime deps for serving
├── requirements-dev.txt              # + pytest, ruff, jupyter, etc.
└── README.md
```

## How to Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # full env: notebook + API + test/lint tooling
python fetch_data.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_sentiment_analysis.ipynb
ruff check .
python -m pytest tests/ -v
python predict.py "this made my whole week, thank you!"
streamlit run app/streamlit_app.py        # or: uvicorn api.main:app --reload
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the lint + test steps automatically on every push and pull request.
