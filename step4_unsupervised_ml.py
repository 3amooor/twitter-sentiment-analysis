# step4_unsupervised_ml.py
from sklearn.decomposition import LatentDirichletAllocation, NMF
import joblib
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

def load_data():
    """Load the TF-IDF matrix and vectorizer"""
    try:
        # Load the TF-IDF matrix
        X = joblib.load('data/processed/tfidf_matrix.pkl')
        
        # Load the TF-IDF vectorizer
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        
        return X, tfidf
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure you've run step1_preprocessing.py first")
        exit(1)

def train_lda(X, n_topics=10, max_iter=10):
    """Train LDA model with progress tracking"""
    print(f"Training LDA with {n_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )
    
    with tqdm(total=max_iter, desc="LDA Training") as pbar:
        for i in range(max_iter):
            lda.partial_fit(X)
            pbar.update(1)
    
    return lda

def train_nmf(X, n_topics=10, max_iter=100):
    """Train NMF model with progress tracking"""
    print(f"Training NMF with {n_topics} topics...")
    nmf = NMF(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=42,
        init='nndsvd'
    )
    
    with tqdm(total=max_iter, desc="NMF Training") as pbar:
        for i in range(10):  # Check more frequently
            nmf.max_iter = (i + 1) * 10
            nmf.fit(X)
            pbar.update(10)
            if i >= max_iter // 10 - 1:
                break
    
    return nmf

def plot_top_words(model, feature_names, n_top_words=10, title="Top Words in Topics"):
    """Plot top words in each topic using matplotlib"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx >= 10:  # Limit to first 10 topics for visualization
            break
            
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={'fontsize': 12})
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.suptitle(title, fontsize=16)
    plt.savefig('visualizations/topic_words.png', bbox_inches='tight')
    plt.close()

def create_word_clouds(model, feature_names, n_top_words=20):
    """Create word clouds for each topic"""
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx >= 5:  # Limit to first 5 topics for word clouds
            break
            
        # Create dictionary of word frequencies
        word_freq = {feature_names[i]: topic[i] for i in topic.argsort()[:-n_top_words-1:-1]}
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white'
        ).generate_from_frequencies(word_freq)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_idx + 1} Word Cloud')
        plt.tight_layout()
        plt.savefig(f'visualizations/topic_{topic_idx + 1}_wordcloud.png')
        plt.close()

def main():
    X, tfidf = load_data()
    
    # Set number of topics
    n_topics = 10
    
    
    # Train LDA
    lda = train_lda(X, n_topics=n_topics)
    joblib.dump(lda, 'models/lda_model.pkl')
    print("LDA model saved to models/lda_model.pkl")
    
    # Train NMF
    nmf = train_nmf(X, n_topics=n_topics)
    joblib.dump(nmf, 'models/nmf_model.pkl')
    print("NMF model saved to models/nmf_model.pkl")
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Create visualizations
    plot_top_words(lda, feature_names, title="Top Words in LDA Topics")
    create_word_clouds(lda, feature_names)
    

if __name__ == "__main__":
    main()