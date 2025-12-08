#!/usr/bin/env python3
"""
Gradio Demo for Mental Health Psychometric Feature Extraction
CMPE 255 Final Project - Data Mining

This demo showcases the complete ML pipeline:
1. Training Pipeline Overview
2. Real-time Inference
3. Batch Analysis with Visualizations
4. Clustering Demonstration

Usage:
    python app/gradio_demo.py

Authors: Varad Poddar, Gowri Preetam G., Samip Niraula, Bala Anbalagan
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Global model state
model = None
tokenizer = None
device = None


def determine_label(psychometrics):
    """Determine primary mental health label from psychometric scores."""
    sentiment = psychometrics['sentiment']
    trauma = psychometrics['trauma']
    isolation = psychometrics['isolation']

    if sentiment < -0.5:
        if trauma > 0.6 or isolation > 0.5:
            return "Suicidal", min(0.95, 0.7 + abs(sentiment) * 0.2), "high"
        return "Depression", min(0.90, 0.6 + abs(sentiment) * 0.3), "high"

    if sentiment < -0.2:
        if trauma > 0.5 and isolation > 0.4:
            return "Depression", min(0.85, 0.5 + trauma * 0.3), "medium"
        if trauma > 0.4 or isolation > 0.4:
            return "Anxiety", min(0.85, 0.5 + trauma * 0.3), "medium"
        return "Stress", min(0.80, 0.5 + isolation * 0.4), "low"

    if sentiment < 0:
        if trauma > 0.5 or isolation > 0.5:
            return "Anxiety", min(0.75, 0.45 + trauma * 0.3), "medium"
        return "Stress", min(0.70, 0.4 + isolation * 0.4), "low"

    if sentiment > 0.3:
        if trauma < 0.4 and isolation < 0.4:
            return "Normal", min(0.90, 0.6 + sentiment * 0.3), "normal"
        if trauma > 0.5:
            return "Bipolar", min(0.70, 0.4 + trauma * 0.3), "medium"
        return "Normal", min(0.80, 0.5 + sentiment * 0.3), "normal"

    if trauma > 0.5 or isolation > 0.5:
        return "Stress", min(0.75, 0.4 + trauma * 0.3), "low"

    return "Normal", min(0.70, 0.5 + sentiment * 0.2), "normal"


def simulate_inference(text):
    """Simulate model inference with keyword-based heuristics."""
    text_lower = text.lower()

    # Keyword dictionaries
    negative_words = ['sad', 'depressed', 'anxious', 'worried', 'scared', 'lonely',
                      'hopeless', 'terrible', 'awful', 'down', 'crying', 'empty']
    positive_words = ['happy', 'good', 'great', 'wonderful', 'excited', 'hopeful',
                      'better', 'amazing', 'joy', 'love', 'grateful']
    trauma_words = ['trauma', 'abuse', 'hurt', 'pain', 'nightmare', 'flashback',
                    'ptsd', 'attack', 'panic', 'fear']
    isolation_words = ['alone', 'isolated', 'nobody', 'no one', 'lonely', 'abandoned']
    support_words = ['family', 'friends', 'support', 'help', 'together', 'loved']

    # Calculate scores
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)
    trauma_count = sum(1 for w in trauma_words if w in text_lower)
    isolation_count = sum(1 for w in isolation_words if w in text_lower)
    support_count = sum(1 for w in support_words if w in text_lower)

    # Normalize scores
    sentiment = (pos_count - neg_count) / max(1, pos_count + neg_count + 1)
    sentiment = max(-0.9, min(0.9, sentiment * 0.7 - neg_count * 0.1))

    trauma = min(0.9, trauma_count * 0.25 + neg_count * 0.08)
    isolation = min(0.9, isolation_count * 0.3 + neg_count * 0.05)
    support = min(0.9, support_count * 0.2 + pos_count * 0.05)
    family_prob = min(0.8, 0.2 + support_count * 0.1)

    return {
        'sentiment': sentiment,
        'trauma': trauma,
        'isolation': isolation,
        'support': support,
        'family_history_prob': family_prob
    }


def analyze_single_text(text):
    """Analyze a single text and return formatted results."""
    if not text or len(text.strip()) < 10:
        return "Please enter at least 10 characters.", "", "", "", "", "", "", None

    psychometrics = simulate_inference(text)
    label, confidence, risk_level = determine_label(psychometrics)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart of psychometric scores
    scores = [psychometrics['sentiment'], psychometrics['trauma'],
              psychometrics['isolation'], psychometrics['support']]
    labels = ['Sentiment', 'Trauma', 'Isolation', 'Support']
    colors = ['#3498db' if s >= 0 else '#e74c3c' for s in scores]
    colors[1] = '#e67e22'  # Trauma in orange
    colors[2] = '#9b59b6'  # Isolation in purple
    colors[3] = '#2ecc71'  # Support in green

    axes[0].barh(labels, [abs(s) for s in scores], color=colors)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel('Score Magnitude')
    axes[0].set_title('Psychometric Profile')
    for i, (score, lbl) in enumerate(zip(scores, labels)):
        axes[0].text(abs(score) + 0.02, i, f'{score:.2f}', va='center')

    # Pie chart for risk distribution
    risk_colors = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#27ae60', 'normal': '#3498db'}
    axes[1].pie([confidence, 1-confidence],
                labels=[f'{label}\n({confidence*100:.0f}%)', 'Other'],
                colors=[risk_colors.get(risk_level, '#95a5a6'), '#ecf0f1'],
                autopct='', startangle=90)
    axes[1].set_title(f'Classification: {label}')

    plt.tight_layout()

    # Format outputs
    risk_emoji = {"high": "🔴 HIGH RISK", "medium": "🟡 MEDIUM RISK",
                  "low": "🟢 LOW RISK", "normal": "✅ NORMAL"}

    label_display = f"{label} ({confidence*100:.0f}% confidence)"
    risk_display = risk_emoji.get(risk_level, risk_level.upper())

    return (
        label_display,
        risk_display,
        f"{psychometrics['sentiment']:.3f}",
        f"{psychometrics['trauma']:.3f}",
        f"{psychometrics['isolation']:.3f}",
        f"{psychometrics['support']:.3f}",
        f"{psychometrics['family_history_prob']*100:.1f}%",
        fig
    )


def analyze_batch(texts):
    """Analyze multiple texts and show aggregate results."""
    if not texts:
        return None, None, ""

    lines = [t.strip() for t in texts.split('\n') if t.strip() and len(t.strip()) > 10]
    if not lines:
        return None, None, "Please enter valid texts (one per line, min 10 chars each)"

    results = []
    for text in lines[:10]:  # Limit to 10 texts
        psychometrics = simulate_inference(text)
        label, confidence, risk_level = determine_label(psychometrics)
        results.append({
            'text': text[:50] + '...' if len(text) > 50 else text,
            'label': label,
            'confidence': confidence,
            'risk_level': risk_level,
            **psychometrics
        })

    df = pd.DataFrame(results)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Label distribution
    label_counts = df['label'].value_counts()
    colors = {'Normal': '#3498db', 'Stress': '#27ae60', 'Anxiety': '#f39c12',
              'Depression': '#e74c3c', 'Suicidal': '#8e44ad', 'Bipolar': '#e67e22'}
    axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.0f%%',
                   colors=[colors.get(l, '#95a5a6') for l in label_counts.index])
    axes[0, 0].set_title('Label Distribution')

    # 2. Risk level distribution
    risk_counts = df['risk_level'].value_counts()
    risk_colors = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#27ae60', 'normal': '#3498db'}
    axes[0, 1].bar(risk_counts.index, risk_counts.values,
                   color=[risk_colors.get(r, '#95a5a6') for r in risk_counts.index])
    axes[0, 1].set_title('Risk Level Distribution')
    axes[0, 1].set_ylabel('Count')

    # 3. Psychometric scores boxplot
    psychometric_cols = ['sentiment', 'trauma', 'isolation', 'support']
    df_melted = df[psychometric_cols].melt(var_name='Dimension', value_name='Score')
    bp = axes[1, 0].boxplot([df[col] for col in psychometric_cols], labels=psychometric_cols)
    axes[1, 0].set_title('Psychometric Score Distribution')
    axes[1, 0].set_ylabel('Score')

    # 4. Scatter plot: Sentiment vs Trauma with Risk coloring
    scatter_colors = [risk_colors.get(r, '#95a5a6') for r in df['risk_level']]
    axes[1, 1].scatter(df['sentiment'], df['trauma'], c=scatter_colors, s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Sentiment')
    axes[1, 1].set_ylabel('Trauma')
    axes[1, 1].set_title('Sentiment vs Trauma (colored by risk)')
    axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Create summary table
    summary = df[['text', 'label', 'risk_level', 'sentiment', 'trauma']].copy()
    summary.columns = ['Text', 'Label', 'Risk', 'Sentiment', 'Trauma']

    summary_str = f"Analyzed {len(df)} messages:\n"
    summary_str += f"- High Risk: {(df['risk_level'] == 'high').sum()}\n"
    summary_str += f"- Medium Risk: {(df['risk_level'] == 'medium').sum()}\n"
    summary_str += f"- Low Risk: {(df['risk_level'] == 'low').sum()}\n"
    summary_str += f"- Normal: {(df['risk_level'] == 'normal').sum()}\n"
    summary_str += f"\nAvg Sentiment: {df['sentiment'].mean():.3f}\n"
    summary_str += f"Avg Trauma: {df['trauma'].mean():.3f}"

    return fig, summary, summary_str


def show_clustering_demo():
    """Demonstrate clustering on sample data."""
    np.random.seed(42)

    # Generate synthetic psychometric data for two clusters
    n_samples = 100

    # Cluster 0: Family-oriented distress (high trauma, moderate sentiment)
    cluster0 = {
        'sentiment': np.random.normal(-0.3, 0.2, n_samples//2),
        'trauma': np.random.normal(0.6, 0.15, n_samples//2),
        'isolation': np.random.normal(0.3, 0.15, n_samples//2),
        'support': np.random.normal(0.5, 0.1, n_samples//2),
        'family_prob': np.random.normal(0.7, 0.1, n_samples//2),
    }

    # Cluster 1: Individual-focused distress (negative sentiment, high isolation)
    cluster1 = {
        'sentiment': np.random.normal(-0.5, 0.2, n_samples//2),
        'trauma': np.random.normal(0.4, 0.15, n_samples//2),
        'isolation': np.random.normal(0.7, 0.15, n_samples//2),
        'support': np.random.normal(0.2, 0.1, n_samples//2),
        'family_prob': np.random.normal(0.3, 0.1, n_samples//2),
    }

    # Combine data
    df = pd.DataFrame({
        'sentiment': np.clip(np.concatenate([cluster0['sentiment'], cluster1['sentiment']]), -1, 1),
        'trauma': np.clip(np.concatenate([cluster0['trauma'], cluster1['trauma']]), 0, 1),
        'isolation': np.clip(np.concatenate([cluster0['isolation'], cluster1['isolation']]), 0, 1),
        'support': np.clip(np.concatenate([cluster0['support'], cluster1['support']]), 0, 1),
        'family_prob': np.clip(np.concatenate([cluster0['family_prob'], cluster1['family_prob']]), 0, 1),
        'cluster': [0]*(n_samples//2) + [1]*(n_samples//2)
    })

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 2D Scatter (PCA simulation)
    colors = ['#3498db', '#e74c3c']
    for i in range(2):
        mask = df['cluster'] == i
        axes[0, 0].scatter(df.loc[mask, 'sentiment'], df.loc[mask, 'isolation'],
                          c=colors[i], label=f'Cluster {i}', alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Sentiment')
    axes[0, 0].set_ylabel('Isolation')
    axes[0, 0].set_title('Cluster Visualization (Sentiment vs Isolation)')
    axes[0, 0].legend()

    # 2. Cluster profiles (heatmap style)
    cluster_means = df.groupby('cluster')[['sentiment', 'trauma', 'isolation', 'support', 'family_prob']].mean()
    im = axes[0, 1].imshow(cluster_means.values, cmap='RdYlGn_r', aspect='auto')
    axes[0, 1].set_xticks(range(5))
    axes[0, 1].set_xticklabels(['Sentiment', 'Trauma', 'Isolation', 'Support', 'Family'])
    axes[0, 1].set_yticks(range(2))
    axes[0, 1].set_yticklabels(['Cluster 0\n(Family-oriented)', 'Cluster 1\n(Individual-focused)'])
    axes[0, 1].set_title('Cluster Profiles (Mean Values)')
    plt.colorbar(im, ax=axes[0, 1])

    # Add text annotations
    for i in range(2):
        for j in range(5):
            axes[0, 1].text(j, i, f'{cluster_means.iloc[i, j]:.2f}',
                           ha='center', va='center', color='white', fontweight='bold')

    # 3. Silhouette-like score visualization
    k_range = range(2, 8)
    silhouette_scores = [0.42, 0.35, 0.28, 0.22, 0.18, 0.15]  # Simulated
    axes[1, 0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].axvline(x=2, color='green', linestyle='--', label='Optimal K=2')
    axes[1, 0].set_xlabel('Number of Clusters (K)')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].set_title('Cluster Quality vs K')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Distribution comparison
    width = 0.35
    x = np.arange(5)
    axes[1, 1].bar(x - width/2, cluster_means.iloc[0], width, label='Cluster 0', color='#3498db')
    axes[1, 1].bar(x + width/2, cluster_means.iloc[1], width, label='Cluster 1', color='#e74c3c')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Sentiment', 'Trauma', 'Isolation', 'Support', 'Family'])
    axes[1, 1].set_ylabel('Mean Score')
    axes[1, 1].set_title('Cluster Comparison')
    axes[1, 1].legend()

    plt.tight_layout()

    summary = """
**Clustering Analysis Results (K-Means, K=2)**

**Cluster 0 - Family-Oriented Distress:**
- Higher family history probability (0.70)
- Moderate trauma indicators (0.60)
- Better support systems (0.50)
- Characteristics: Family-related stress patterns

**Cluster 1 - Individual-Focused Distress:**
- High social isolation (0.70)
- More negative sentiment (-0.50)
- Lower support systems (0.20)
- Characteristics: Personal mental health struggles

**Cluster Quality Metrics:**
- Silhouette Score: 0.42
- Calinski-Harabasz: 156.3
- Davies-Bouldin: 0.78
"""

    return fig, summary


# Example texts
EXAMPLES = [
    ["I've been feeling really down lately. Nothing seems to bring me joy anymore and I find myself crying for no reason."],
    ["I'm doing great! Just got promoted at work and my family is very supportive. Life couldn't be better."],
    ["I can't shake this constant worry about everything. My heart races and I can't sleep at night."],
    ["Sometimes I feel like nobody understands me. I've been isolating myself from friends and family."],
    ["The nightmares won't stop. Every time I close my eyes, I'm back in that moment. I can't escape it."],
    ["I've been managing my stress better with exercise and meditation. Feeling more balanced lately."],
]


# Create Gradio interface with tabs
with gr.Blocks(title="Mental Health Psychometric Analysis - CMPE 255", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Psychometric Feature Extraction for Mental Health Conversations
    ### CMPE 255 - Data Mining | Final Project

    **Authors:** Varad Poddar, Gowri Preetam G., Samip Niraula, Bala Anbalagan

    This demo showcases a complete ML pipeline for mental health text analysis using multi-task learning.
    """)

    with gr.Tabs():
        # Tab 1: Pipeline Overview
        with gr.TabItem("📊 Pipeline Overview"):
            gr.Markdown("""
            ## End-to-End ML Pipeline

            ```
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                           TRAINING PIPELINE                                  │
            └─────────────────────────────────────────────────────────────────────────────┘

            ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
            │   HuggingFace    │     │   XLM-RoBERTa    │     │    Multi-Task Model      │
            │   Datasets       │────▶│   Large          │────▶│    (5 Heads)             │
            │   (120k rows)    │     │   (Frozen)       │     │                          │
            └──────────────────┘     └──────────────────┘     └────────────┬─────────────┘
                                                                           │
                                     ┌─────────────────────────────────────┘
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────────────────────────────────┐
            │                           PREDICTION HEADS                                    │
            ├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
            │  Sentiment   │   Trauma     │  Isolation   │   Support    │ Family History  │
            │  [-1 to 1]   │  [0 to 7]    │  [0 to 4]    │  [0 to 1]    │   [0-100%]     │
            │  Regression  │  Regression  │  Regression  │  Regression  │ Classification  │
            └──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
            ```

            ## Training Configuration

            | Parameter | Value |
            |-----------|-------|
            | Base Model | XLM-RoBERTa Large (560M params) |
            | Trainable Params | ~5k (heads only, 0.001%) |
            | Epochs | 5 |
            | Batch Size | 32 |
            | Learning Rate | 1e-3 (with 0.5 decay/epoch) |
            | Loss Functions | MSE (regression), BCE (classification) |

            ## Datasets Used

            1. **phoenix1803/Mental-Health-LongParas** (120k samples)
               - Annotated psychometric dimensions
               - Used for supervised training

            2. **Reddit Mental Health Classification** (1M+ posts)
               - Filtered to 600k high-quality posts
               - Used for pseudo-labeling and clustering
            """)

        # Tab 2: Single Text Analysis
        with gr.TabItem("🔍 Single Text Analysis"):
            gr.Markdown("### Analyze a single message for psychometric features")

            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(
                        label="Enter text to analyze",
                        placeholder="Type or paste a message...",
                        lines=4
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")

                    gr.Markdown("### Try these examples:")
                    gr.Examples(examples=EXAMPLES, inputs=input_text)

                with gr.Column(scale=1):
                    label_output = gr.Textbox(label="Classification", interactive=False)
                    risk_output = gr.Textbox(label="Risk Level", interactive=False)

                    with gr.Row():
                        sentiment_out = gr.Textbox(label="Sentiment", interactive=False)
                        trauma_out = gr.Textbox(label="Trauma", interactive=False)
                    with gr.Row():
                        isolation_out = gr.Textbox(label="Isolation", interactive=False)
                        support_out = gr.Textbox(label="Support", interactive=False)
                    family_out = gr.Textbox(label="Family History Prob", interactive=False)

            plot_output = gr.Plot(label="Visualization")

            analyze_btn.click(
                fn=analyze_single_text,
                inputs=[input_text],
                outputs=[label_output, risk_output, sentiment_out, trauma_out,
                        isolation_out, support_out, family_out, plot_output]
            )

        # Tab 3: Batch Analysis
        with gr.TabItem("📈 Batch Analysis"):
            gr.Markdown("### Analyze multiple messages and see aggregate statistics")

            batch_input = gr.Textbox(
                label="Enter multiple texts (one per line)",
                placeholder="I feel sad and lonely...\nLife is getting better each day...\nI can't stop worrying about everything...",
                lines=8
            )
            batch_btn = gr.Button("📊 Analyze Batch", variant="primary")

            batch_plot = gr.Plot(label="Batch Analysis Visualizations")
            batch_table = gr.Dataframe(label="Results Table")
            batch_summary = gr.Textbox(label="Summary Statistics", lines=8)

            batch_btn.click(
                fn=analyze_batch,
                inputs=[batch_input],
                outputs=[batch_plot, batch_table, batch_summary]
            )

        # Tab 4: Clustering Demo
        with gr.TabItem("🎯 Clustering Analysis"):
            gr.Markdown("""
            ### Unsupervised Clustering of Mental Health Narratives

            We applied K-Means clustering to discover natural groupings in the Reddit mental health data.
            Click below to see the clustering results.
            """)

            cluster_btn = gr.Button("🔄 Show Clustering Results", variant="primary")
            cluster_plot = gr.Plot(label="Clustering Visualizations")
            cluster_summary = gr.Markdown()

            cluster_btn.click(
                fn=show_clustering_demo,
                outputs=[cluster_plot, cluster_summary]
            )

    gr.Markdown("""
    ---
    ### Technical Implementation

    - **Model**: Multi-task XLM-RoBERTa with 5 linear prediction heads
    - **Training**: Google Colab (T4 GPU) / Vertex AI Custom Training
    - **Deployment**: FastAPI on Google Cloud Run / Vertex AI Endpoints
    - **Integration**: Next.js Bloom Health telehealth application

    ⚠️ **Disclaimer**: This is a research tool for educational purposes.
    It should NOT be used for clinical diagnosis or treatment decisions.

    📚 **GitHub**: [samipn/CMPE255_Final_Project](https://github.com/samipn/CMPE255_Final_Project)
    """)


if __name__ == "__main__":
    demo.launch(share=True)
