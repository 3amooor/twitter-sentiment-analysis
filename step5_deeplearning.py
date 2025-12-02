# step5_deeplearning.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_data():
    try:
        df = pd.read_csv('data/processed/processed_data.csv')
        X = df['processed_text'].fillna('').astype(str).values
        y = df['sentiment_encoded'].values
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please run step1_preprocessing.py first")
        exit(1)

def create_lstm_model(vocab_size, max_length, num_classes):
    """Create LSTM model for text classification"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length, mask_zero=True),
        LSTM(128, dropout=0.2, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_cnn_model(vocab_size, max_length, num_classes):
    """Create CNN model for text classification"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_models():
    X, y = load_data()
    
    # Get number of classes
    num_classes = len(np.unique(y))
    print(f"Found {num_classes} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    max_length = 100  # Adjust based on your data
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Save tokenizer
    joblib.dump(tokenizer, 'models/tokenizer.pkl')
    
    # Train LSTM
    print("\n" + "="*50)
    print("Training LSTM model...")
    print("="*50)
    lstm_model = create_lstm_model(vocab_size, max_length, num_classes)
    lstm_model.summary()
    
    lstm_history = lstm_model.fit(
        X_train_pad, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
            ModelCheckpoint(
                'models/lstm_best.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ],
        verbose=1
    )
    
    # Save final LSTM model
    lstm_model.save('models/lstm_model.h5')
    print("LSTM model saved to models/lstm_model.h5")
    
    # Train CNN
    print("\n" + "="*50)
    print("Training CNN model...")
    print("="*50)
    cnn_model = create_cnn_model(vocab_size, max_length, num_classes)
    cnn_model.summary()
    
    cnn_history = cnn_model.fit(
        X_train_pad, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
            ModelCheckpoint(
                'models/cnn_best.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ],
        verbose=1
    )
    
    # Save final CNN model
    cnn_model.save('models/cnn_model.h5')
    print("CNN model saved to models/cnn_model.h5")
    
    # Evaluate models
    lstm_loss, lstm_acc = lstm_model.evaluate(X_test_pad, y_test, verbose=0)
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test_pad, y_test, verbose=0)
    
    print("\n" + "="*50)
    print("Model Performance:")
    print("="*50)
    print(f"LSTM - Test Accuracy: {lstm_acc:.4f}, Test Loss: {lstm_loss:.4f}")
    print(f"CNN  - Test Accuracy: {cnn_acc:.4f}, Test Loss: {cnn_loss:.4f}")

if __name__ == "__main__":
    # Set memory growth to prevent OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    train_models()