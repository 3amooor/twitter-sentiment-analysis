# step3_supervised_ml.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_data():
    try:
        # Load the TF-IDF matrix
        X = joblib.load('data/processed/tfidf_matrix.pkl')
        
        # Load the processed data with labels
        df = pd.read_csv('data/processed/processed_data.csv')
        y = df['sentiment_encoded'].values
        
        # Reduce the feature space if too large
        if X.shape[1] > 10000:
            from sklearn.feature_selection import SelectKBest, chi2
            print("Reducing feature space...")
            X = SelectKBest(chi2, k=10000).fit_transform(X, y)
            
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure you've run step1_preprocessing.py first")
        exit(1)

def train_models(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models with optimized parameters
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                n_jobs=-1,  # Use all cores
                C=1.0
            )
        },
        'RandomForest': {
            'model': RandomForestClassifier(
                n_estimators=50,  # Reduced number of trees
                max_depth=20,     # Limit tree depth
                random_state=42,
                n_jobs=-1,       # Use all cores
                class_weight='balanced'
            )
        },
        'LinearSVM': {
            'model': LinearSVC(
                max_iter=1000,
                random_state=42,
                dual=False,  # Better for n_samples > n_features
                class_weight='balanced'
            )
        }
    }
    
    results = []
    
    for name, model_info in tqdm(models.items(), desc="Training models"):
        
        try:
            # Train the model
            model = model_info['model']
            model.fit(X_train, y_train)
            
            # Save the model
            joblib.dump(model, f'models/{name.lower()}_model.pkl')
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store results
            results.append({
                'model': name,
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            })
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/supervised_ml_results.csv', index=False)
        print("\nResults saved to results/supervised_ml_results.csv")
        return results_df
    else:
        print("\nNo models were successfully trained.")
        return None

def main():
    X, y = load_data()
    
    print(f"\nData shape: {X.shape}")
    results = train_models(X, y)
    
    if results is not None:
        print("\nTraining complete! Models saved to models/ directory")
        print(results[['model', 'accuracy', 'f1']].to_string(index=False))

if __name__ == "__main__":
    main()