"""
Twitter Sentiment Analysis - Model Training Script
===================================================
This script trains a Logistic Regression model on the Sentiment140 dataset
and saves the trained model and vectorizer for use in the Streamlit app.

Usage:
    python train_model.py

Output:
    - models/sentiment_model.pkl
    - models/tfidf_vectorizer.pkl
    - models/model_metrics.json
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from zipfile import ZipFile

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
port_stem = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Clean and preprocess tweet text.
    
    Steps:
    1. Remove @mentions and URLs
    2. Remove non-alphabetic characters
    3. Convert to lowercase
    4. Remove stopwords
    5. Apply stemming
    """
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


def load_dataset():
    """Load and prepare the Sentiment140 dataset."""
    
    # Check if dataset exists
    csv_path = 'training.1600000.processed.noemoticon.csv'
    zip_path = 'sentiment140.zip'
    
    if not os.path.exists(csv_path):
        if os.path.exists(zip_path):
            print("📦 Extracting dataset...")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            print("⚠️  Dataset not found!")
            print("Please download from: https://www.kaggle.com/datasets/kazanova/sentiment140")
            print("Or run: kaggle datasets download -d kazanova/sentiment140")
            return None
    
    print("📊 Loading dataset...")
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(csv_path, names=column_names, encoding='ISO-8859-1')
    
    # Convert target: 0 = negative, 4 = positive → 0 = negative, 1 = positive
    df['target'] = df['target'].replace(4, 1)
    
    print(f"✅ Loaded {len(df):,} tweets")
    print(f"   Positive: {(df['target'] == 1).sum():,}")
    print(f"   Negative: {(df['target'] == 0).sum():,}")
    
    return df


def train_model(df, sample_size=None):
    """Train the sentiment analysis model."""
    
    # Optional: Use a sample for faster training
    if sample_size and sample_size < len(df):
        print(f"\n📉 Using sample of {sample_size:,} tweets for training...")
        df = df.sample(n=sample_size, random_state=42)
    
    print("\n🔄 Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Prepare features and target
    X = df['processed_text']
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Vectorize text using TF-IDF
    print("\n📝 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    print("\n🤖 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    train_pred = model.predict(X_train_tfidf)
    test_pred = model.predict(X_test_tfidf)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 RESULTS")
    print(f"{'='*50}")
    print(f"   Training Accuracy: {train_accuracy:.2%}")
    print(f"   Test Accuracy:     {test_accuracy:.2%}")
    print(f"   Test F1-Score:     {test_f1:.2%}")
    print(f"{'='*50}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['Negative', 'Positive']))
    
    # Prepare metrics dictionary
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_f1_score': float(test_f1),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'total_dataset_size': int(len(df)),
        'tfidf_max_features': 5000,
        'model_type': 'Logistic Regression'
    }
    
    return model, vectorizer, metrics


def save_model(model, vectorizer, metrics):
    """Save the trained model, vectorizer, and metrics."""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/sentiment_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"💾 Vectorizer saved to: {vectorizer_path}")
    
    # Save metrics
    metrics_path = 'models/model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to: {metrics_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("🐦 TWITTER SENTIMENT ANALYSIS - MODEL TRAINING")
    print("="*60)
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Train model (use full dataset or sample)
    # For faster training during development, you can set sample_size
    # e.g., sample_size=100000 for 100k samples
    model, vectorizer, metrics = train_model(df, sample_size=None)
    
    # Save model and artifacts
    save_model(model, vectorizer, metrics)
    
    print("\n✅ Training complete! You can now run the Streamlit app:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
