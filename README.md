# Twitter Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://twitter-sentiment-analysis-p3ycuj5h9prjaqbtysahpa.streamlit.app/)

## 🎯 Project Overview
This project performs sentiment analysis on Twitter data using machine learning techniques. It includes both an exploratory Jupyter notebook and an **interactive web application** for real-time sentiment prediction.

### ✨ Features
- **Machine Learning Model**: Logistic Regression with TF-IDF vectorization
- **Interactive Web App**: Real-time sentiment predictions via Streamlit
- **High Accuracy**: ~78% accuracy on test data
- **1.6M Training Samples**: Trained on the Sentiment140 dataset

## 🚀 Live Demo
Try the live web app: [Sentiment Analyzer](https://twitter-sentiment-analysis-p3ycuj5h9prjaqbtysahpa.streamlit.app/)

## 📊 Dataset
- **Source**: [Sentiment140 from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Labels**: Binary (Positive/Negative)

## 🛠️ Technologies Used
| Category | Technologies |
|----------|-------------|
| **ML/Data** | Pandas, NumPy, Scikit-learn |
| **NLP** | NLTK (Tokenization, Stemming, Stopwords) |
| **Web App** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |

## 📈 Project Structure
```
Twitter-Sentiment-Analysis/
├── models/                          # Saved model artifacts
│   ├── sentiment_model.pkl          # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl         # TF-IDF vectorizer
│   └── model_metrics.json           # Model performance metrics
├── Dataset/
│   └── kaggle.json                  # Kaggle API credentials
├── TwitterSentimentanalysis.ipynb   # Exploratory analysis notebook
├── train_model.py                   # Model training script
├── app.py                           # Streamlit web application
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Option 1: Run the Web App (Quick Start)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hemanthraj09/Twitter-Sentiment-analysis.git
   cd Twitter-Sentiment-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (first time only):
   ```bash
   # Download dataset first
   kaggle datasets download -d kazanova/sentiment140
   
   # Train and save model
   python train_model.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

### Option 2: Explore the Notebook

```bash
jupyter notebook TwitterSentimentanalysis.ipynb
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~80% |
| Test Accuracy | ~78% |
| F1-Score | ~78% |
| Model Type | Logistic Regression |
| Vectorization | TF-IDF (5000 features, 1-2 ngrams) |

## 🔍 How It Works

1. **Text Preprocessing**
   - Remove @mentions and URLs
   - Remove special characters
   - Convert to lowercase
   - Remove stopwords
   - Apply Porter Stemming

2. **Feature Extraction**
   - TF-IDF Vectorization with 5000 features
   - Unigrams and Bigrams (ngram_range=(1,2))

3. **Classification**
   - Logistic Regression with L2 regularization

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set the main file path to `app.py`
5. Deploy!

> **Note**: Make sure to include the `models/` folder with trained model files in your repository.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License
This project is licensed under the MIT License.

## 👨‍💻 Author
**Hemanth Raj** - [GitHub Profile](https://github.com/Hemanthraj09)

## 📧 Contact
Feel free to reach out if you have any questions or suggestions!
