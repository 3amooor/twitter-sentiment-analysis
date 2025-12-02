import pandas as pd
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import emoji
import joblib
from tqdm import tqdm
import os

os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Initialize NLTK
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'omw-1.4'], quiet=True)
tqdm.pandas()

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = TweetTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def get_wordnet_pos(self, treebank_tag):
        tag = treebank_tag[0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def clean_text(self, text):
        text = str(text)
        text = emoji.demojize(text)
        text = re.sub(r'http\S+|www\S+|https?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()

    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = self.tokenizer.tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        lemmatized = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) 
                     for word, tag in pos_tags 
                     if word not in self.stop_words and word.isalpha()]
        return ' '.join(lemmatized)

def main():
    df = pd.read_csv('data/twitter_training.csv', 
                    header=None, 
                    names=['id', 'entity', 'sentiment', 'text'])
    
    preprocessor = TextPreprocessor()
    
    # Preprocess text
    df['processed_text'] = df['text'].progress_apply(preprocessor.preprocess)
    
    # Save preprocessed data
    df.to_csv('data/twitter_preprocessed.csv', index=False)
    print("Saved preprocessed data to data/twitter_preprocessed.csv")
    
    print("Creating TF-IDF features...")
    # Initialize and save TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000, 
        ngram_range=(1,2), 
        sublinear_tf=True,
        min_df=5,
        max_df=0.7
    )
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(df['processed_text'])
    
    # Save TF-IDF vectorizer and matrix
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    joblib.dump(tfidf_matrix, 'data/processed/tfidf_matrix.pkl')
    
    # Save feature names
    feature_names = tfidf.get_feature_names_out()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save label encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Save processed data with encodings
    df.to_csv('data/processed/processed_data.csv', index=False)
    
    # Save train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix,
        df['sentiment_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment_encoded']
    )
    

    
if __name__ == "__main__":
    main()