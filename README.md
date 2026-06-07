# Financial Earnings Sentiment Pipeline with Temporal Drift Detection

**Type:** Personal Project  
**Timeline:** June 2024 – August 2024 (10 weeks)  
**Domain:** Financial NLP · Sentiment Analysis · Temporal Drift Detection

---

## Overview

End-to-end NLP pipeline ingesting 2,500 earnings call transcripts and analyst reports across 25 S&P 500 companies (3 years, ~180K sentence units). Applies dual-classifier sentiment scoring with temporal drift detection, flagging statistically significant period-over-period sentiment shifts that precede price-relevant events.

Built this independently to extend the NLP pipeline patterns from JPMorgan work into a harder problem: temporal comparison rather than single-document classification.

---

## Problem Statement

Existing earnings sentiment tools have two failure modes:
1. Rule-based keyword matching misses contextual nuance (e.g., "exceeded expectations" can be negative in a guidance-cut context)
2. Single-shot classifiers produce a score without temporal comparison — missing the drift signal practitioners actually need

This pipeline addresses both: contextual BERT-based classification + statistically validated period-over-period drift detection.

---

## System Architecture

```
Earnings Transcripts + Analyst Reports
        │
        ▼
┌─────────────────────────────────┐
│  STAGE 1: INGESTION             │
│  Python parser (plain text/PDF) │
│  PostgreSQL: ticker · period ·  │
│  document type metadata         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  STAGE 2: PREPROCESSING         │
│  NLTK sentence tokenizer        │
│  spaCy NER: ORG · MONEY ·      │
│  DATE · PERSON per sentence     │
│  Pandas intermediate layer      │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3: DUAL CLASSIFICATION                    │
│  Primary: Fine-tuned BERT (FinancialPhraseBank) │
│    → 3-class: positive/neutral/negative          │
│    → Macro F1: 0.86 on earnings test set        │
│  Baseline: TF-IDF + Logistic Regression         │
│    → Macro F1: 0.68 (reference/fallback)        │
│  A/B tracked via MLflow (34 runs)               │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  STAGE 4: DRIFT DETECTION                        │
│  Doc-level score: weighted mean BERT confidence  │
│  Position weights: lead 1.4x · closing 0.9x    │
│  NER entity density as secondary weight          │
│  Drift: (current - prior) / prior               │
│  Threshold ±0.15 → 78% precision on validation  │
│  143 drift events flagged across 25 companies   │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  STAGE 5: TRACKING + DASHBOARD                   │
│  MLflow: experiment tracking (34 runs)           │
│  PostgreSQL: drift events with evidence payload  │
│  Streamlit: per-company trend lines ·            │
│  drift event timeline · sentence-level audit    │
└─────────────────────────────────────────────────┘
```

---

## Results

| Metric | Result |
|--------|--------|
| BERT macro F1 (earnings transcript test set) | **0.86** |
| TF-IDF baseline macro F1 | 0.68 |
| Relative F1 improvement | **26%** |
| Drift detection precision @ ±0.15 threshold | **78%** |
| Documents processed | 2,500 |
| Sentence units classified | ~180,000 |
| Drift events flagged | 143 across 25 companies |
| Full corpus pipeline runtime | **<18 minutes (CPU only)** |
| Streamlit response time | <1.2 seconds |
| MLflow runs | 34 |

---

## Key Technical Decisions

### Drift Threshold Calibration
Built a labeled validation set of 40 earnings periods where analyst consensus had explicitly flagged a sentiment shift. Swept threshold from 0.05 to 0.30 in 0.05 increments. At ±0.15: 78% precision with acceptable recall. Below 0.10: too many noise events. Above 0.20: real shifts missed.

### Position-Weighted Sentiment Aggregation
Lead sentences in earnings calls carry disproportionate signal weight (guidance announcements, executive forward statements). Applied 1.4x weight to lead sentences, 0.9x to closing. NER entity density (presence of monetary values or dates) as secondary amplifier.

### BERT Cross-Domain Transfer
Fine-tuned `bert-base-uncased` on FinancialPhraseBank, then applied directly to earnings transcripts without re-training to evaluate cross-domain generalization. Result: F1 of 0.86 vs 0.88 on FinancialPhraseBank — modest domain shift from formal financial phrases to spoken executive transcript language.

---

## Stack

| Layer | Technology |
|-------|-----------|
| NLP | BERT (`bert-base-uncased`), Hugging Face Transformers, spaCy, NLTK |
| Classification | Fine-tuned BERT (primary), TF-IDF + Logistic Regression (baseline) |
| Data | PostgreSQL, Pandas |
| Experiment Tracking | MLflow (34 runs) |
| Dashboard | Streamlit |

---

## Project Structure

```
4_Financial_Earnings_Sentiment_Pipeline/
├── README.md
├── src/
│   ├── ingestion/
│   │   ├── transcript_parser.py      # PDF + plain text ingestion
│   │   └── postgres_store.py         # Document store with ticker/period metadata
│   ├── preprocessing/
│   │   ├── sentence_tokenizer.py     # NLTK sentence boundary detection
│   │   └── entity_extractor.py       # spaCy NER per sentence
│   ├── classification/
│   │   ├── bert_classifier.py        # Fine-tuned BERT sentiment classifier
│   │   ├── tfidf_baseline.py         # TF-IDF + LR baseline
│   │   └── mlflow_tracker.py         # A/B experiment logging
│   ├── drift/
│   │   ├── drift_detector.py         # Period-over-period drift calculation
│   │   ├── threshold_calibrator.py   # Threshold sweep against validation events
│   │   └── event_logger.py           # Drift event PostgreSQL persistence
│   └── dashboard/
│       └── streamlit_app.py          # Sentiment trend + drift event dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bert_finetuning.ipynb
│   ├── 03_drift_threshold_calibration.ipynb
│   └── 04_results_analysis.ipynb
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Setup & Usage

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run full pipeline on sample data
python src/ingestion/transcript_parser.py --input_dir data/transcripts
python src/classification/bert_classifier.py --mode train
python src/drift/drift_detector.py --threshold 0.15

# Launch Streamlit dashboard
streamlit run src/dashboard/streamlit_app.py
```

---

## Connection to JPMorgan Work

The same NLP toolchain (spaCy NER, NLTK, scikit-learn, MLflow) used here is the foundation of the COiN NLP pipeline at JPMorgan. This project extended the pattern to: (a) a harder temporal comparison problem, (b) end-to-end system ownership including the dashboard layer, and (c) BERT-based contextual classification vs. SVM/TF-IDF.
