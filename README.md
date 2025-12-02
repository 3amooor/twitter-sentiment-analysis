# Twitter Sentiment Analysis Dashboard

A comprehensive sentiment analysis tool that analyzes Twitter data using various machine learning models, including traditional ML, LSTM, CNN, and DistilBERT.

## 📋 Features

- **Multiple Model Support**:
  - Traditional ML: Logistic Regression, Random Forest, SVM
  - Deep Learning: LSTM, CNN, DistilBERT
- **Interactive Dashboard**: Real-time sentiment analysis with visualizations
- **Model Comparison**: Compare performance across different models
- **Data Visualization**: Word clouds, sentiment distribution, and topic modeling

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (if needed)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🏃‍♂️ Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Project Structure
```
├── data/                # Data storage
│   └── processed/      # Processed data files
├── models/             # Saved models
├── app.py              # Main Streamlit application
├── step1_preprocessing.py  # Data preprocessing
├── step2_eda.py        # Exploratory Data Analysis
├── step3_supervised_ml.py  # Traditional ML models
├── step4_unsupervised_ml.py  # Unsupervised learning
├── step5_deeplearning.py   # Deep learning models
├── step5b_distilbert_finetune.py       # DistilBERT model
└── requirements.txt    # Project dependencies
```

## 📊 Model Training

1. **Preprocess the data**
   ```bash
   python step1_preprocessing.py
   ```

2. **Train traditional ML models**
   ```bash
   python step3_supervised_ml.py
   ```

3. **Train deep learning models**
   ```bash
   python step5_deep_learning.py
   ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Streamlit
- Uses Hugging Face Transformers for DistilBERT
- NLTK and spaCy for text processing
- scikit-learn for traditional ML models
