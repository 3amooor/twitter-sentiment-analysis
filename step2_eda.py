# step2_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import nltk
from nltk.corpus import stopwords
from collections import Counter
import plotly.express as px
from tqdm import tqdm

# Create necessary directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Download NLTK data
nltk.download('stopwords', quiet=True)

def load_data():
    """Load and prepare the preprocessed data"""
    try:
        df = pd.read_csv('data/twitter_preprocessed.csv')
        
        # Convert all text columns to string and handle NaN values
        text_columns = ['processed_text', 'text', 'cleaned_text', 'lemmatized_text']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run step1_preprocessing.py first.")
        exit(1)

def plot_sentiment_distribution(df):
    """Plot and save sentiment distribution"""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='sentiment')
    plt.title('Sentiment Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_dist.png')
    plt.close()

def generate_word_clouds(df):
    """Generate and save word clouds for each sentiment"""
    stop_words = set(stopwords.words('english'))
    
    for sentiment in df['sentiment'].unique():
        # Filter out non-string values and join with space
        texts = df[df['sentiment'] == sentiment]['processed_text']
        text = ' '.join([str(t) for t in texts if isinstance(t, str) and t.strip() != 'nan' and t.strip() != ''])
        
        if not text.strip():
            print(f"Warning: No valid text found for sentiment: {sentiment}")
            continue
            
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            stopwords=stop_words,
            max_words=100
        ).generate(text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment}')
        plt.tight_layout()
        
        # Create a valid filename by removing special characters
        safe_sentiment = "".join(c if c.isalnum() else "_" for c in str(sentiment))
        plt.savefig(f'visualizations/wordcloud_{safe_sentiment}.png')
        plt.close()

def analyze_ngrams(df, n=2):
    """Analyze and save most common n-grams"""
    results = {}
    
    for sentiment in df['sentiment'].unique():
        texts = df[df['sentiment'] == sentiment]['processed_text']
        words = []
        
        for text in texts:
            if not isinstance(text, str) or text.strip() in ['', 'nan']:
                continue
            tokens = str(text).split()
            n_grams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            words.extend(n_grams)
            
        if not words:
            print(f"Warning: No valid text found for n-gram analysis of sentiment: {sentiment}")
            continue
            
        counter = Counter(words)
        most_common = counter.most_common(20)
        results[sentiment] = most_common
        
        # Create a safe filename
        safe_sentiment = "".join(c if c.isalnum() else "_" for c in str(sentiment))
        with open(f'results/{safe_sentiment}_common_{n}grams.txt', 'w', encoding='utf-8') as f:
            for word, count in most_common:
                f.write(f"{word}: {count}\n")
    
    return results

def plot_temporal_trends(df):
    """Plot and save temporal trends if date column exists"""
    date_columns = ['date', 'created_at', 'timestamp']
    date_col = next((col for col in date_columns if col in df.columns), None)
    
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df['date'] = df[date_col].dt.date
            daily_counts = df.groupby(['date', 'sentiment']).size().unstack()
            
            plt.figure(figsize=(12, 6))
            daily_counts.plot(kind='line')
            plt.title('Daily Sentiment Trends')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('visualizations/daily_sentiment_trends.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot temporal trends: {str(e)}")

def main():
    df = load_data()
    
    plot_sentiment_distribution(df)
    generate_word_clouds(df)
    
    analyze_ngrams(df, n=1)  # Unigrams
    analyze_ngrams(df, n=2)  # Bigrams
    
    plot_temporal_trends(df)
    

if __name__ == "__main__":
    main()