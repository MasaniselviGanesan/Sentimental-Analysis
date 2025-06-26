# shared_preprocessing.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------
# Clean Text Function
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)             # Remove URLs
    text = re.sub(r"@\w+|#", '', text)                     # Remove mentions, hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)             # Remove special characters
    text = re.sub(r"\d+", '', text)                        # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()               # Remove extra spaces
    return text

# -----------------------------
# Unified Preprocessing Method
# -----------------------------
def preprocess_data(df, text_col='text', label_col='airline_sentiment', max_words=10000, max_len=50):
    # Keep only relevant columns and drop nulls
    df = df[[text_col, label_col]].dropna()

    # Clean text column
    df[text_col] = df[text_col].apply(clean_text)

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df[label_col])

    # Tokenize text
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[text_col])

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df[text_col])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    return padded, df['label'].values, tokenizer, label_encoder

# --------------------------
# Optional: Save/Load Tools
# --------------------------
import pickle

def save_tokenizer_and_encoder(tokenizer, label_encoder, tokenizer_path='tokenizer.pkl', encoder_path='label_encoder.pkl'):
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_tokenizer_and_encoder(tokenizer_path='tokenizer.pkl', encoder_path='label_encoder.pkl'):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return tokenizer, label_encoder
