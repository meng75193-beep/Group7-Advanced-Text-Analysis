"""
Group 7 - Fake News Detection Web App
Streamlit Deployment for Text Analytics Group Project

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Fake News Detector - Group 7",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS Styling
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #566573;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-fake {
        background-color: #fdecea;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
    }
    .result-true {
        background-color: #eafaf1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9f9;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .info-box {
        background-color: #eaf2f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Helper Functions
# ============================================================

@st.cache_resource
def load_bert_model():
    """Load BERT sentence transformer model (cached for performance)."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource
def load_models():
    """Load trained ML models."""
    models = {}
    if os.path.exists('best_model.pkl'):
        models['best'] = joblib.load('best_model.pkl')
    if os.path.exists('lr_model.pkl'):
        models['lr'] = joblib.load('lr_model.pkl')
    if os.path.exists('svm_model.pkl'):
        models['svm'] = joblib.load('svm_model.pkl')
    return models

def clean_text(text):
    """Clean text using regular expressions."""
    if not text or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text, bert_model, ml_model):
    """Predict if news is fake or true."""
    cleaned = clean_text(text)
    if not cleaned:
        return None, None
    embedding = bert_model.encode([cleaned], normalize_embeddings=True)
    prediction = ml_model.predict(embedding)[0]
    probability = ml_model.predict_proba(embedding)[0]
    return prediction, probability

# ============================================================
# Load Models
# ============================================================
with st.spinner("Loading BERT model and classifiers..."):
    bert_model = load_bert_model()
    ml_models = load_models()

# Determine available model
if 'best' in ml_models:
    default_model = ml_models['best']
    default_model_name = "Best Model (Auto-selected)"
elif 'lr' in ml_models:
    default_model = ml_models['lr']
    default_model_name = "Logistic Regression"
elif 'svm' in ml_models:
    default_model = ml_models['svm']
    default_model_name = "SVM"
else:
    default_model = None
    default_model_name = "No model loaded"

# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("📰 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home / Predict", "📊 Model Performance", "📈 EDA Insights", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose classifier",
    ["Best Model (Auto)", "Logistic Regression", "SVM"]
)

# Select model based on choice
if model_choice == "Logistic Regression" and 'lr' in ml_models:
    active_model = ml_models['lr']
    active_model_name = "Logistic Regression"
elif model_choice == "SVM" and 'svm' in ml_models:
    active_model = ml_models['svm']
    active_model_name = "SVM"
else:
    active_model = default_model
    active_model_name = default_model_name

st.sidebar.markdown(f"**Active:** {active_model_name}")

# ============================================================
# Page 1: Home / Predict
# ============================================================
if page == "🏠 Home / Predict":
    st.markdown('<p class="main-header">📰 Fake News Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Group 7 | BERT Embeddings + Logistic Regression / SVM</p>', unsafe_allow_html=True)

    if default_model is None:
        st.warning("⚠️ No trained model found. Please run the notebook first to generate model files.")
        st.info("Place `best_model.pkl`, `lr_model.pkl`, and `svm_model.pkl` in the same directory as this app.")
    else:
        st.markdown("### Enter News Article Text")
        user_input = st.text_area(
            "Paste the news title or article content below:",
            height=150,
            placeholder="e.g., The Federal Reserve announced a 0.25% interest rate increase on Wednesday..."
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            predict_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            st.caption("The model uses BERT embeddings to analyze semantic patterns in the text.")

        if predict_btn and user_input.strip():
            with st.spinner("Analyzing text..."):
                pred, prob = predict_news(user_input, bert_model, active_model)

            if pred is not None:
                confidence = max(prob) * 100
                is_true = (pred == 1)

                if is_true:
                    st.markdown(f"""
                    <div class="result-true">
                        <h3 style="color:#27ae60; margin:0;">✅ LIKELY TRUE NEWS</h3>
                        <p style="margin:0.5rem 0; font-size:1.1rem;">
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fake">
                        <h3 style="color:#e74c3c; margin:0;">⚠️ LIKELY FAKE NEWS</h3>
                        <p style="margin:0.5rem 0; font-size:1.1rem;">
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability breakdown
                st.markdown("#### Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Category': ['Fake News', 'True News'],
                    'Probability': [prob[0] * 100, prob[1] * 100]
                })
                st.bar_chart(prob_df.set_index('Category'), use_container_width=True)

                # Show cleaned text
                with st.expander("View cleaned text"):
                    st.write(clean_text(user_input)[:500] + "...")

        # Quick examples
        st.markdown("---")
        st.markdown("### 💡 Try These Examples")
        examples = [
            "BREAKING: Scientists discover that drinking bleach can cure all diseases, according to anonymous sources.",
            "The Federal Reserve announced a 0.25% interest rate increase on Wednesday, citing concerns over rising inflation.",
            "SHOCKING: The president is actually a reptilian alien from outer space, claims conspiracy theorist.",
            "According to the latest census data, the population of New York City has grown by 2.3% over the past decade."
        ]
        for i, ex in enumerate(examples, 1):
            if st.button(f"Example {i}: {ex[:60]}...", key=f"ex{i}"):
                pred, prob = predict_news(ex, bert_model, active_model)
                if pred is not None:
                    label = "TRUE" if pred == 1 else "FAKE"
                    st.write(f"**Prediction:** {label} ({max(prob)*100:.1f}% confidence)")

# ============================================================
# Page 2: Model Performance
# ============================================================
elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluation on Test Set (20% holdout)</p>', unsafe_allow_html=True)

    # Check if evaluation data exists
    if not os.path.exists('processed_data.csv') or not os.path.exists('bert_embeddings.npy'):
        st.info("📋 Run the full notebook to generate evaluation results and visualizations.")
        st.markdown("""
        **Expected metrics after training:**
        
        | Metric | Logistic Regression | SVM |
        |--------|-------------------|-----|
        | Accuracy | ~98-99% | ~98-99% |
        | Precision | ~98-99% | ~98-99% |
        | Recall | ~98-99% | ~98-99% |
        | F1-Score | ~98-99% | ~98-99% |
        | ROC-AUC | ~99%+ | ~99%+ |
        """)
    else:
        # Load data and show results
        df = pd.read_csv('processed_data.csv')
        embeddings = np.load('bert_embeddings.npy')

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'].values
        )

        tab1, tab2, tab3 = st.tabs(["📋 Metrics Table", "🔀 Confusion Matrices", "📈 ROC Curves"])

        with tab1:
            results = []
            for name, model in [("Logistic Regression", ml_models.get('lr')), ("SVM", ml_models.get('svm'))]:
                if model is not None:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    results.append({
                        'Model': name,
                        'Accuracy': f"{accuracy_score(y_test, y_pred)*100:.2f}%",
                        'Precision': f"{precision_score(y_test, y_pred)*100:.2f}%",
                        'Recall': f"{recall_score(y_test, y_pred)*100:.2f}%",
                        'F1-Score': f"{f1_score(y_test, y_pred)*100:.2f}%",
                        'ROC-AUC': f"{roc_auc_score(y_test, y_prob)*100:.2f}%"
                    })

            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

                # Bar chart comparison
                st.markdown("#### Visual Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                x = np.arange(len(metrics))
                width = 0.35
                for i, (name, model) in enumerate([("LR", ml_models.get('lr')), ("SVM", ml_models.get('svm'))]):
                    if model:
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)[:, 1]
                        scores = [
                            accuracy_score(y_test, y_pred)*100,
                            precision_score(y_test, y_pred)*100,
                            recall_score(y_test, y_pred)*100,
                            f1_score(y_test, y_pred)*100,
                            roc_auc_score(y_test, y_prob)*100
                        ]
                        ax.bar(x + (i-0.5)*width, scores, width, label=name, alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.set_ylabel('Score (%)')
                ax.set_ylim(80, 102)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)

        with tab2:
            col1, col2 = st.columns(2)
            for col, (name, model) in [(col1, ("Logistic Regression", ml_models.get('lr'))),
                                        (col2, ("SVM", ml_models.get('svm')))]:
                with col:
                    if model is not None:
                        y_pred = model.predict(X_test)
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'], ax=ax)
                        ax.set_title(f'{name}')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots(figsize=(8, 6))
            for name, model, color in [("Logistic Regression", ml_models.get('lr'), '#4a90d9'),
                                        ("SVM", ml_models.get('svm'), '#50c878')]:
                if model is not None:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], 'gray', lw=1, linestyle='--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve Comparison')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# ============================================================
# Page 3: EDA Insights
# ============================================================
elif page == "📈 EDA Insights":
    st.markdown('<p class="main-header">📈 EDA Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Exploratory Data Analysis of the Fake News Dataset</p>', unsafe_allow_html=True)

    if os.path.exists('processed_data.csv'):
        df = pd.read_csv('processed_data.csv')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", f"{len(df):,}")
        with col2:
            st.metric("Fake News", f"{(df['label']==0).sum():,}")
        with col3:
            st.metric("True News", f"{(df['label']==1).sum():,}")

        # Show saved plots if available
        st.markdown("---")
        st.subheader("Visualizations")

        plot_files = [
            ('label_distribution.png', 'Label Distribution'),
            ('text_length_distribution.png', 'Text Length Distribution'),
            ('subject_distribution.png', 'Subject Distribution'),
            ('wordcloud_comparison.png', 'Word Cloud Comparison'),
            ('top_words_comparison.png', 'Top Words Comparison')
        ]

        for fname, title in plot_files:
            if os.path.exists(fname):
                st.markdown(f"#### {title}")
                st.image(fname, use_container_width=True)
            else:
                st.info(f"Run the notebook to generate: {title}")
    else:
        st.info("📋 Run the notebook first to generate EDA visualizations.")
        st.markdown("""
        **Dataset Overview:**
        - ~23,000 Fake News articles
        - ~21,000 True News articles
        - Features: title, text, subject, date
        """)

# ============================================================
# Page 4: About
# ============================================================
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Project Overview

    This is a **Fake News Detection** system developed by **Group 7** for the
    *MSc Business Analytics - Text Analytics* course.

    ### Methodology

    | Component | Details |
    |-----------|---------|
    | **Task** | Binary Classification (Fake vs True News) |
    | **Dataset** | ~44,000 labeled news articles |
    | **Text Cleaning** | Regex, lowercase, URL/HTML removal, stopword removal |
    | **Text Representation** | BERT Sentence Embeddings (all-MiniLM-L6-v2, 384-dim) |
    | **Model 1** | Logistic Regression |
    | **Model 2** | Support Vector Machine (SVM) |
    | **Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC |

    ### Technology Stack

    - **Python** with scikit-learn, PyTorch
    - **Sentence-Transformers** for BERT embeddings
    - **Streamlit** for web deployment
    - **Matplotlib / Seaborn** for visualization

    ### How It Works

    1. User inputs news article text
    2. Text is cleaned using regular expressions
    3. BERT model converts text to 384-dimensional embedding
    4. Trained classifier predicts Fake (0) or True (1)
    5. Confidence score is displayed alongside prediction

    ### Group Members

    | Name | Student ID |
    |------|------------|
    | Meng Zhang | 22424168 |
    | Jubrighter Maame Frimp | 22420884 |
    | Adwoa Serwah Kyei-Baff | 22419517 |
    | Francisca Dede Teye Wo | 22422969 |
    | Enock sayibu | 22419566 |

    ### References

    - Dataset: [Kaggle - Fake News Classification](https://www.kaggle.com/datasets/aadyasingh55/fake-news-classification)
    - BERT Model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    """)

    st.markdown("---")
    st.caption("Group 7 - Text Analytics Group Project | MSc Business Analytics")