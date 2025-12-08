# Psychometric Feature Extraction for Mental‑Health Conversations  
Multitask Transformer Model · Pseudo‑Labeling · Clustering · GCP Deployment  

Authors: Varad Poddar, Gowri Preetam G., Samip Niraula, Bala Anbalagan  

---

## Project Overview

This project builds an end‑to‑end AI system that analyzes mental‑health–related text and extracts interpretable psychosocial signals. These signals are strictly non‑diagnostic and are intended to support therapists by highlighting contextual cues in message-based telehealth environments.

The system predicts three learned dimensions:

1. Sentiment intensity  
2. Trauma‑related linguistic indicators  
3. Family‑history likelihood  

The final system includes:

- A multitask learning model based on XLM‑RoBERTa  
- Weak supervision through large‑scale pseudo‑labeling  
- Clustering to discover narrative archetypes  
- Deployment on Google Cloud Run for real‑time inference  
- A Next.js and Gradio interface for visualization

---

## Model Architecture

We train a multitask model with five prediction heads:

| Dimension | Type | Status | Notes |
|----------|------|--------|-------|
| Sentiment | Regression | Used | Most stable signal |
| Trauma indicators | Regression | Used | Good variance |
| Family‑history probability | Classification | Used | Probabilistic cue |
| Social isolation | Regression | Removed | Low R² |
| Support strength | Regression | Removed | Collapsed output |

Backbone: XLM‑RoBERTa Large (frozen)  
Heads: Linear(1024 → 1)  
Loss: MSELoss for regression, BCEWithLogitsLoss with class imbalance weighting  
Training: 2–3 epochs, LR=1e‑3 for heads only, AMP enabled  

---

## Data Sources

### Mental‑Health‑LongParas  
Long counseling-style messages with annotated psychosocial dimensions.  
Used for supervised training and dimension filtering.

### Reddit Mental Health Classification  
Approximately one million posts from mental‑health subreddits.  
After strict SQL-like filtering, ~100,000 high-quality posts remained.  
Used for large-scale weak supervision and narrative clustering.

---

## Data Cleaning

Applied filters include:

- Remove posts containing survey links or YouTube links  
- Exclude unrelated subreddit labels (e.g., jokes, conspiracy)  
- Remove texts longer than 12,000 characters  
- Lowercase scanning for academic or spam indicators  

This ensured a reliable pseudo‑labeling set.

---

## Training Details

Training setup:

- Backbone frozen for compute efficiency  
- Head-only fine‑tuning  
- LR = 1e‑3  
- Gradient scaling (AMP)  
- Per-head losses tracked  
- Validation performed only at the end to reduce runtime  

Dimension filtering was performed after training. The top three dimensions—sentiment, trauma, and family—were retained based on MAE, R², F1, ROC-AUC, and residual analysis.

---

## Pseudo‑Labeling

The trained model was used to generate:

- pred_sentiment  
- pred_trauma  
- pred_family_prob  

for each Reddit post.  
Values were clipped to valid ranges and standardized.

These pseudo‑labels formed a large-scale psychometric dataset for unsupervised study.

---

## Clustering

Steps taken:

1. Standardize features using StandardScaler  
2. Evaluate K from 2 to 10 using  
   - Silhouette score  
   - Calinski-Harabasz score  
   - Davies-Bouldin score  
   - Adjusted Rand Index for cluster stability  
3. Final choice: **K=2**, providing  
   - Cluster 0: Family‑oriented distress narratives  
   - Cluster 1: Individual-focused distress narratives  
4. Visualization via PCA and UMAP  
5. Interpret clusters via representative examples and centroid values  

---

## Deployment (GCP Cloud Run)

The model was deployed as a Docker container using FastAPI.

End-to-end workflow:

1. Export model (`multitask_model.pt`)  
2. Package with tokenizer and scaler  
3. Build Docker image  
4. Push to Google Artifact Registry  
5. Deploy on Cloud Run  
6. Next.js app connects to the `/predict` endpoint  

Example JSON response:

```json
{
  "sentiment_raw": -0.42,
  "sentiment_bucket": "negative",
  "trauma_raw": 2.87,
  "trauma_bucket": "moderate",
  "family_prob": 0.12,
  "family_bucket": "unlikely",
  "cluster": 1,
  "cluster_label": "individual-focused distress narrative"
}
```

## Gradio Interface

Two modes supported:
Single Message

    Raw model values

    Fine-grained buckets

    Narrative cluster assignment

Multi‑Message

    Per-message table

    Conversation summary

    Sentiment trajectory line plot

Useful for testing, debugging, and demonstration.
Results Summary

    Sentiment and trauma were learnable with moderate accuracy

    Family-history probability provided weak but usable signals

    Isolation and support dimensions were not reliable enough

    Clustering revealed consistent high-level narrative structures

    Real-time inference latency on Cloud Run: ~80–120 ms per message

## Installation

```
git clone <repo-url>
cd project
pip install -r requirements.txt
```

## Run Gradio Demo

```
python app/gradio_demo.py
```

## Run FastAPI Inference Server

```
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Docker Build

```
docker build -t mh-profiles .
docker run -p 8080:8080 mh-profiles
```

## Deploy to Cloud Run

```
gcloud run deploy mh-profiles \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```
