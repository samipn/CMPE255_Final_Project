# CRISP-DM Artifacts

## 1. Business Understanding
- Goal: Extract psychosocial signals (sentiment, trauma, isolation, support, family history) from mental-health text to assist therapists; not for diagnosis.
- Success criteria: Accurate per-head metrics (R²/MAE for regression heads, F1/AUC for family history), low latency (<150ms p50 in prod), interpretable outputs consumed by Bloom app.
- Constraints: Privacy (no data retention), ethical disclaimers, limited trainable parameters (heads-only) for speed, GPU availability.

## 2. Data Understanding
- Sources:
  - `phoenix1803/Mental-Health-LongParas` (supervised training).
  - `solomonk/reddit_mental_health_posts` + `Amod/mental_health_counseling_conversations` (inference/labeling, clustering).
- Explorations/EDA on actual dataset: In `notebooks/Mental_Health_Psychometrics_Training.ipynb` we inspect label distribution, target ranges per psychometric column, and text length stats (informing `max_length=256` and stratified splits). Reddit data checked for removed/deleted markers and length thresholds.

## 3. Data Preparation
- Cleaning: Filter invalid Reddit rows, remove very short texts, combine title+body.
- Splits: 70/15/15 stratified by `label` (train/val/test) in `src/dataset.py` and the notebook.
- Tokenization: XLM-RoBERTa tokenizer, max_length=256, padding/truncation.
- Scaling: Support scores multiplied by 100 to avoid tiny values during training.

## 4. Modeling
- Backbone: XLM-RoBERTa Large (hidden_size=1024), frozen for efficiency.
- Heads: Five linear heads (sentiment, trauma, isolation, support, family history).
- Losses: MSE for regressions; BCEWithLogits with `pos_weight` for family history.
- Optimizer: AdamW on head parameters only; learning rate decay per epoch.
- Alternatives explored: Heads-only vs unfreezing; LR/epoch variants; clustering K selection (silhouette) shown in visuals.

## 5. Evaluation
- Metrics: R²/MAE per regression head; F1/AUC/confusion matrix for family history.
- Visuals: Learning curves, per-head losses, confusion matrix, ROC, clustering; stored in `images/`.
- Splits respected (val/test). Metrics JSON saved with training artifacts; notebook logs per-epoch losses.

## 6. Deployment
- FastAPI (`src/inference.py`) with `/predict` and `/health`.
- Gradio demo (`app/gradio_demo.py`) for interactive showcase.
- Cloud: Vertex AI custom training job → Model Registry → Endpoint; Dockerfile for build.
- Integration: Bloom app consumes endpoint schema (label, confidence, risk_level, psychometrics/all_scores) and health check.

## 7. Monitoring & Next Steps
- Latency targets tracked (local vs Vertex AI ~100ms p50).
- Retrain before go-live to refresh weights on latest data/config; consider unfreezing for higher accuracy.
- Add TensorBoard logging in cloud jobs for per-epoch monitoring; expand bias/fairness checks and add alerting on endpoint health.
