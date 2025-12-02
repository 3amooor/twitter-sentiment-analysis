import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json
import sys
from typing import Dict, Any, Optional, Tuple, List, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models() -> Dict[str, Any]:
    """Load all models and vectorizers with error handling"""
    models = {}
    try:
        # Load traditional ML models
        models['logistic_regression'] = joblib.load('models/logisticregression_model.pkl')
        models['random_forest'] = joblib.load('models/randomforest_model.pkl')
        models['svm'] = joblib.load('models/linearsvm_model.pkl')
        models['tfidf'] = joblib.load('models/tfidf_vectorizer.pkl')
        models['label_encoder'] = joblib.load('models/label_encoder.pkl')
        
        # Try to load LSTM if available
        if os.path.exists('models/lstm_model.h5'):
            try:
                from tensorflow.keras.models import load_model
                models['lstm'] = load_model('models/lstm_model.h5')
                if os.path.exists('models/lstm_tokenizer.pkl'):
                    models['lstm_tokenizer'] = joblib.load('models/lstm_tokenizer.pkl')
                if os.path.exists('models/lstm_maxlen.pkl'):
                    models['lstm_maxlen'] = joblib.load('models/lstm_maxlen.pkl')
            except ImportError:
                st.warning("TensorFlow is not installed. LSTM model will not be available.")
            except Exception as e:
                st.warning(f"Error loading LSTM model: {str(e)}")
        
        # Try to load CNN if available
        if os.path.exists('models/cnn_model.h5'):
            try:
                from tensorflow.keras.models import load_model
                models['cnn'] = load_model('models/cnn_model.h5')
                if os.path.exists('models/cnn_tokenizer.pkl'):
                    models['cnn_tokenizer'] = joblib.load('models/cnn_tokenizer.pkl')
                if os.path.exists('models/cnn_maxlen.pkl'):
                    models['cnn_maxlen'] = joblib.load('models/cnn_maxlen.pkl')
            except ImportError:
                st.warning("TensorFlow is not installed. CNN model will not be available.")
            except Exception as e:
                st.warning(f"Error loading CNN model: {str(e)}")
        
        # Load DistilBERT if available
        if os.path.exists('models/distilbert_pt'):
            try:
                models['distilbert_tokenizer'] = DistilBertTokenizer.from_pretrained('models/distilbert_pt')
                models['distilbert'] = DistilBertForSequenceClassification.from_pretrained('models/distilbert_pt')
                models['distilbert'].eval()
                # Move model to GPU if available
                if torch.cuda.is_available():
                    models['distilbert'] = models['distilbert'].to('cuda')
            except Exception as e:
                st.warning(f"Error loading DistilBERT model: {str(e)}")
        
        # Load metrics
        if os.path.exists('results/model_performance.json'):
            try:
                with open('results/model_performance.json', 'r') as f:
                    models['metrics'] = json.load(f)
            except Exception as e:
                st.warning(f"Error loading model metrics: {str(e)}")
                
        if os.path.exists('results/distilbert_metrics.json'):
            try:
                with open('results/distilbert_metrics.json', 'r') as f:
                    models['distilbert_metrics'] = json.load(f)
            except Exception as e:
                st.warning(f"Error loading DistilBERT metrics: {str(e)}")
                
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    return models

def predict_with_keras_model(text: str, model_name: str, models: Dict[str, Any]) -> Tuple[Optional[str], float]:
    """Predict sentiment using LSTM or CNN model"""
    if model_name not in models or f"{model_name}_tokenizer" not in models:
        return None, 0.0
        
    try:
        # Tokenize text
        tokenizer = models[f"{model_name}_tokenizer"]
        maxlen = models.get(f"{model_name}_maxlen", 100)  # Default maxlen if not found
        
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=maxlen)
        
        # Get prediction
        model = models[model_name]
        prediction = model.predict(padded_sequence, verbose=0)
        
        # Get predicted class and confidence
        pred_class = np.argmax(prediction[0])
        confidence = float(prediction[0][pred_class])
        
        # Map to label
        if 'label_encoder' in models:
            try:
                label = models['label_encoder'].inverse_transform([pred_class])[0]
            except:
                label = str(pred_class)
        else:
            label = str(pred_class)
            
        return label, confidence
        
    except Exception as e:
        st.error(f"Error in {model_name} prediction: {str(e)}")
        return None, 0.0

def predict_sentiment(text: str, model_name: str, models: Dict[str, Any]) -> Tuple[Optional[str], float]:
    """Predict sentiment using the selected model"""
    if not text.strip() or model_name not in models:
        return None, 0.0
        
    try:
        # Handle LSTM/CNN models
        if model_name in ['lstm', 'cnn']:
            return predict_with_keras_model(text, model_name, models)
            
        if model_name == 'distilbert' and 'distilbert' in models:
            # Move inputs to the same device as the model
            device = next(models['distilbert'].parameters()).device
            inputs = models['distilbert_tokenizer'](
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                outputs = models['distilbert'](**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            pred_class = np.argmax(probs)
            confidence = float(probs[pred_class])
            
            # Map to label
            if 'label_encoder' in models:
                try:
                    label = models['label_encoder'].inverse_transform([pred_class])[0]
                except:
                    label = str(pred_class)
            else:
                label = str(pred_class)
                
            return label, confidence
            
        elif model_name in models and model_name not in ['tfidf', 'label_encoder']:
            # Traditional ML models
            X = models['tfidf'].transform([text])
            model = models[model_name]
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0].max()
            else:
                pred = model.predict(X)[0]
                proba = 1.0
                
            # Map to label
            if 'label_encoder' in models:
                try:
                    label = models['label_encoder'].inverse_transform([pred])[0]
                except:
                    label = str(pred)
            else:
                label = str(pred)
                
            return label, float(proba)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        
    return None, 0.0

def plot_top_words(model, feature_names, n_top_words, n_topics=5):
    """Plot the top words for each topic"""
    fig, axes = plt.subplots(1, n_topics, figsize=(30, 5), sharex=True)
    if n_topics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for topic_idx, topic in enumerate(model.components_[:n_topics]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={'fontsize': 20})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=15)
        
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
    
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return fig

def display_metrics(metrics, model_name: str) -> None:
    """Display model metrics"""
    if not metrics:
        st.warning("No metrics available for this model.")
        return
        
    if isinstance(metrics, list):
        metrics = metrics[-1]
        
    if 'models' in metrics:
        model_metrics = metrics['models'].get(model_name, {})
        if model_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.2%}")
            with col2:
                st.metric("Loss", f"{model_metrics.get('loss', 0):.4f}")

def main():
    st.title("📊 Twitter Sentiment Analysis Dashboard")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Comparison", "Real-time Analysis", "Visualizations"])
    
    if page == "Home":
        st.header("Welcome to the Sentiment Analysis Dashboard")
        st.markdown("""
        This dashboard allows you to analyze sentiment in Twitter data using various machine learning models.
        
        ### Features:
        - 🏠 **Home**: Overview and quick analysis
        - 📊 **Model Comparison**: Compare performance of different models
        - 🔍 **Real-time Analysis**: Analyze sentiment of custom text
        - 📈 **Visualizations**: Explore data and model insights
        """)
        
        # Quick analysis
        st.subheader("Quick Analysis")
        text = st.text_area("Enter text to analyze:", "I love this product! It's amazing!")
        
        if st.button("Analyze"):
            if text.strip():
                with st.spinner("Analyzing..."):
                    results = []
                    for model_name in ['logistic_regression', 'random_forest', 'svm', 'lstm', 'cnn', 'distilbert']:
                        if model_name in models and model_name not in ['tfidf', 'label_encoder']:
                            try:
                                label, confidence = predict_sentiment(text, model_name, models)
                                if label is not None:
                                    results.append({
                                        'Model': model_name.upper() if model_name in ['lstm', 'cnn'] else model_name.replace('_', ' ').title(),
                                        'Sentiment': label,
                                        'Confidence': f"{confidence:.1%}"
                                    })
                            except Exception as e:
                                st.warning(f"Error with {model_name}: {str(e)}")
                    
                    if results:
                        st.subheader("Analysis Results")
                        df_results = pd.DataFrame(results)
                        st.table(df_results)
                    else:
                        st.warning("No models available for prediction.")
            else:
                st.warning("Please enter some text to analyze.")
    
    elif page == "Model Comparison":
        st.header("Model Comparison")
        
        if 'metrics' in models or 'distilbert_metrics' in models:
            cols = st.columns(2)
            
            with cols[0]:
                st.subheader("Traditional Models")
                if 'metrics' in models:
                    metrics = models['metrics']
                    if isinstance(metrics, list):
                        metrics = metrics[-1]
                    
                    model_names = list(metrics.get('models', {}).keys())
                    model_acc = [m.get('accuracy', 0) for m in metrics.get('models', {}).values()]
                    
                    if model_names:
                        fig, ax = plt.subplots()
                        ax.bar(model_names, model_acc)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Accuracy')
                        ax.set_title('Model Accuracy Comparison')
                        st.pyplot(fig)
                    else:
                        st.warning("No traditional model metrics available.")
                else:
                    st.warning("No metrics available for traditional models.")
            
            with cols[1]:
                st.subheader("Deep Learning Models")
                dl_metrics = {}
                if 'lstm' in models and 'metrics' in models:
                    metrics = models['metrics']
                    if isinstance(metrics, list):
                        metrics = metrics[-1]
                    dl_metrics['LSTM'] = metrics.get('models', {}).get('LSTM', {})
                
                if 'cnn' in models and 'metrics' in models:
                    metrics = models['metrics']
                    if isinstance(metrics, list):
                        metrics = metrics[-1]
                    dl_metrics['CNN'] = metrics.get('models', {}).get('CNN', {})
                
                if 'distilbert_metrics' in models:
                    dl_metrics['DistilBERT'] = models['distilbert_metrics']
                
                if dl_metrics:
                    model_names = list(dl_metrics.keys())
                    accuracies = [m.get('val_accuracy', m.get('accuracy', 0)) for m in dl_metrics.values()]
                    
                    fig, ax = plt.subplots()
                    ax.bar(model_names, accuracies)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Validation Accuracy')
                    ax.set_title('Deep Learning Models Accuracy')
                    st.pyplot(fig)
                else:
                    st.warning("No metrics available for deep learning models.")
        else:
            st.warning("No model metrics available. Please train the models first.")
    
    elif page == "Real-time Analysis":
        st.header("Real-time Sentiment Analysis")
        
        # Model selection - UPDATED as requested
        available_models = ['logistic_regression', 'random_forest', 'svm']
        if 'lstm' in models:
            available_models.append('lstm')
        if 'cnn' in models:
            available_models.append('cnn')
        if 'distilbert' in models:
            available_models.append('distilbert')
            
        model_name = st.selectbox(
            "Select Model:",
            available_models,
            format_func=lambda x: x.upper() if x in ['lstm', 'cnn'] else x.replace('_', ' ').title()
        )
        
        # Text input
        text = st.text_area("Enter text to analyze:", "I'm really happy with this service!")
        
        if st.button("Analyze Sentiment"):
            if text.strip():
                with st.spinner("Analyzing..."):
                    label, confidence = predict_sentiment(text, model_name, models)
                    
                    if label is not None:
                        st.subheader("Analysis Result")
                        
                        # Display sentiment with emoji
                        sentiment_emojis = {
                            'positive': '😊',
                            'negative': '😞',
                            'neutral': '😐'
                        }
                        emoji = sentiment_emojis.get(label.lower(), '')
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Sentiment", f"{emoji} {label.title()}")
                        with col2:
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Confidence bar
                        st.progress(int(confidence * 100))
                        
                        # Show metrics if available
                        if 'metrics' in models:
                            display_metrics(models['metrics'], model_name.upper() if model_name in ['lstm', 'cnn'] else model_name.replace('_', ' ').title())
            else:
                st.warning("Please enter some text to analyze.")
    
    elif page == "Visualizations":
        st.header("Data Visualizations")
        
        # Word Cloud
        st.subheader("Word Cloud")
        if os.path.exists('data/processed/processed_data.csv'):
            try:
                df = pd.read_csv('data/processed/processed_data.csv')
                if 'processed_text' in df.columns:
                    text = ' '.join(df['processed_text'].dropna())
                    
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("'processed_text' column not found in the data.")
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
        else:
            st.warning("Processed data not found at 'data/processed/processed_data.csv'")
        
        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        if os.path.exists('data/processed/processed_data.csv'):
            try:
                df = pd.read_csv('data/processed/processed_data.csv')
                if 'sentiment' in df.columns:
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, x='sentiment', ax=ax)
                    ax.set_title('Distribution of Sentiments')
                    st.pyplot(fig)
                else:
                    st.warning("'sentiment' column not found in the data.")
            except Exception as e:
                st.error(f"Error generating sentiment distribution: {str(e)}")
        
        # Topic Modeling Visualization
        st.subheader("Topic Modeling")
        if os.path.exists('models/lda_model.pkl') and os.path.exists('models/tfidf_vectorizer.pkl'):
            try:
                lda_model = joblib.load('models/lda_model.pkl')
                tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                
                # Get feature names
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                # Plot top words for each topic
                fig = plot_top_words(lda_model, feature_names, 10, min(5, lda_model.n_components))
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating topic visualization: {str(e)}")
        else:
            st.warning("Topic modeling models not found. Please run the topic modeling step first.")

if __name__ == "__main__":
    main()