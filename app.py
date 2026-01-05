"""
Twitter Sentiment Analysis - Interactive Streamlit App
=======================================================
A web application for real-time sentiment prediction on text input.

Usage:
    streamlit run app.py
"""

import os
import re
import json
import pickle
import streamlit as st

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
port_stem = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================
# Custom CSS Styling
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #657786;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .positive-card {
        background: linear-gradient(135deg, #00C853 0%, #69F0AE 100%);
        color: white;
    }
    
    .negative-card {
        background: linear-gradient(135deg, #FF5252 0%, #FF8A80 100%);
        color: white;
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Sample tweet buttons */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #1DA1F2;
        background: white;
        color: #1DA1F2;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #1DA1F2;
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #657786;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Helper Functions
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer."""
    model_path = 'models/sentiment_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    metrics_path = 'models/model_metrics.json'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return model, vectorizer, metrics


def preprocess_text(text):
    """Clean and preprocess input text."""
    # Remove @mentions and URLs
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase and split
    text = text.lower().split()
    
    # Remove stopwords and apply stemming
    text = [port_stem.stem(word) for word in text if word not in STOPWORDS]
    
    return ' '.join(text)


def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for the given text."""
    # Preprocess
    processed_text = preprocess_text(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return prediction, confidence


# ============================================
# Main App
# ============================================
def main():
    # Header
    st.markdown('<p class="header-title">Sentiment Analyzer!</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Discover the sentiment behind any text with the power of Machine Learning!</p>', unsafe_allow_html=True)
    
    # Load model
    model, vectorizer, metrics = load_model()
    
    if model is None:
        st.error("Model not found! Please run `python train_model.py` first to train the model.")
        st.info("""
        **Get started in 3 easy steps:**
        1. Make sure you have the Sentiment140 dataset
        2. Run `python train_model.py` to train and save the model
        3. Refresh this page and you're ready to go!
        """)
        return
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Performance")
        
        if metrics:
            st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
            st.metric("F1 Score", f"{metrics.get('test_f1_score', 0):.1%}")
            st.metric("Training Samples", f"{metrics.get('training_samples', 0):,}")
        
        st.divider()
        
        st.header("About This Model")
        st.markdown("""
        This model was trained on **1.6 million tweets** from the Sentiment140 Twitter dataset!
        
        **Powered by:**
        - TF-IDF Vectorization
        - Logistic Regression
        - NLTK Text Preprocessing
        
        **Works great for:**
        - Social media posts
        - Product reviews
        - Short-form text
        """)
        
        st.divider()
        
        st.markdown("**Created by:** [Hemanth Raj](https://github.com/Hemanthraj09)")
    
    # Main input section
    st.subheader("Enter Your Text!")
    
    user_input = st.text_area(
        label="Type or paste text to analyze",
        placeholder="Try something like: I absolutely love this product! It exceeded all my expectations.",
        height=120,
        label_visibility="collapsed"
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("Analyze Now!", use_container_width=True, type="primary")
    
    # Show result
    if analyze_button and user_input.strip():
        with st.spinner("Analyzing your text..."):
            prediction, confidence = predict_sentiment(user_input, model, vectorizer)
        
        # Display result
        st.markdown("---")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card positive-card">
                <div class="result-text">POSITIVE!</div>
                <div class="confidence-text">Confidence: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.success("This text expresses a positive sentiment!")
        else:
            st.markdown(f"""
            <div class="result-card negative-card">
                <div class="result-text">NEGATIVE</div>
                <div class="confidence-text">Confidence: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.error("This text expresses a negative sentiment.")
        
        # Show preprocessing info
        with st.expander("See how the text was processed"):
            processed = preprocess_text(user_input)
            st.code(f"Original: {user_input}\n\nProcessed: {processed}")
    
    elif analyze_button:
        st.warning("Please enter some text to analyze!")
    
    # Sample tweets section
    st.markdown("---")
    st.subheader("Try These Sample Texts!")
    
    sample_tweets = {
        "Positive": [
            "I love this product! It's amazing and works perfectly!",
            "Best day ever! Just got promoted at work!",
            "This movie was absolutely fantastic, highly recommend it!"
        ],
        "Negative": [
            "Terrible customer service, never buying from them again.",
            "I'm so disappointed with this purchase, waste of money.",
            "Worst experience ever, the product broke after one day!"
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Positive Examples**")
        for tweet in sample_tweets["Positive"]:
            if st.button(tweet[:45] + "...", key=f"pos_{tweet[:20]}"):
                prediction, confidence = predict_sentiment(tweet, model, vectorizer)
                sentiment = "Positive!" if prediction == 1 else "Negative"
                st.success(f"**Result:** {sentiment} ({confidence:.1%})")
    
    with col2:
        st.markdown("**Negative Examples**")
        for tweet in sample_tweets["Negative"]:
            if st.button(tweet[:45] + "...", key=f"neg_{tweet[:20]}"):
                prediction, confidence = predict_sentiment(tweet, model, vectorizer)
                sentiment = "Positive!" if prediction == 1 else "Negative"
                st.error(f"**Result:** {sentiment} ({confidence:.1%})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Built with Streamlit | Trained on 1.6M tweets | Ready for production!</p>
        <p><a href="https://github.com/Hemanthraj09/Twitter-Sentiment-analysis" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
