# Twitter Sentiment Analysis

A comprehensive sentiment analysis project that analyzes Twitter data using various machine learning and deep learning techniques. This project includes data preprocessing, exploratory data analysis, multiple model implementations, and an interactive Streamlit dashboard.

## 🚀 Features

- **Data Preprocessing**: Text cleaning, tokenization, lemmatization, and feature extraction
- **Exploratory Data Analysis**: Visualizations including word clouds, sentiment distribution, and n-gram analysis
- **Multiple Model Implementations**:
  - Traditional Machine Learning (Logistic Regression, Random Forest, SVM)
  - Deep Learning (LSTM, CNN)
  - Transformer-based (DistilBERT)
- **Interactive Dashboard**: Streamlit-based UI for model comparison and prediction
- **Model Persistence**: Save and load trained models for inference

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/3amooor/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Project Structure

```
twitter-sentiment-analysis/
├── app.py                  # Streamlit dashboard application
├── requirements.txt        # Project dependencies
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── models/                # Trained models
├── results/               # Analysis results and metrics
├── visualizations/        # Generated visualizations
├── step1_preprocessing.py # Data preprocessing pipeline
├── step2_eda.py          # Exploratory data analysis
├── step3_supervised_ml.py # Traditional ML models
├── step4_unsupervised_ml.py # Unsupervised learning
├── step5_deeplearning.py  # LSTM and CNN models
└── step5b_distilbert_finetune.py # Fine-tuning DistilBERT
```

## � Download Pre-trained Models

Due to the large size of the trained models, please download them from Google Drive and extract them to the `models/` directory:

1. Download the models from [Google Drive](https://drive.google.com/file/d/1ZKCwqeLjiB0bi4LjvSEUFSvC2NXUeuUk/view?usp=sharing)
2. Create a `models` directory in the project root if it doesn't exist:
   ```bash
   mkdir models
   ```
3. Extract the downloaded zip file into the `models` directory

## � Usage

### 1. Data Preprocessing
```bash
python step1_preprocessing.py
```

### 2. Exploratory Data Analysis
```bash
python step2_eda.py
```

### 3. Train Models
- For traditional ML models:
  ```bash
  python step3_supervised_ml.py
  ```
- For deep learning models:
  ```bash
  python step5_deeplearning.py
  ```
- For fine-tuning DistilBERT:
  ```bash
  python step5b_distilbert_finetune.py
  ```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

## 📊 Model Comparison

The project includes multiple models with their respective performance metrics:

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Traditional Models** | | | |
| Logistic Regression | 0.6831 | 0.6824 | 0.6831 | 0.6791 |
| Random Forest | 0.5672 | 0.6005 | 0.5672 | 0.5753 |
| Linear SVM | 0.6982 | 0.6990 | 0.6982 | 0.6980 |
| **Deep Learning** | | | |
| LSTM | 0.8658 | - | - | - |
| CNN | 0.8715 | - | - | - |
| **Transformer** | | | |
| DistilBERT | 0.8488 | - | - | - |

*Note: All metrics are on the test set. Loss values: LSTM (0.6007), CNN (0.5719), DistilBERT (0.4534).*

## 🤖 Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, TensorFlow/Keras, Transformers
- **NLP**: NLTK, spaCy, Gensim
- **Visualization**: Matplotlib, Seaborn, Plotly, WordCloud
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [NLTK](https://www.nltk.org/) for natural language processing


