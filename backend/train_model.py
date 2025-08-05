import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === LOAD & SAMPLE DATA ===

# Load 10k Sentiment140
sentiment140 = pd.read_csv(
    '../data/sentiment140/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    nrows=10000
)
sentiment140.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
sentiment140 = sentiment140[['target', 'text']]

# Load 1k Airline
airline = pd.read_csv('../data/twitter-airline/Tweets.csv')
airline = airline[['airline_sentiment', 'text']].sample(1000, random_state=42)
airline['target'] = airline['airline_sentiment'].map({'negative': 0, 'neutral': 2, 'positive': 4})

# Combine
combined = pd.concat([sentiment140, airline[['target', 'text']]]).reset_index(drop=True)

# === CLEAN TEXT ===

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

combined['text'] = combined['text'].apply(clean_text)
combined['label'] = combined['target'].map({0: 0, 2: 1, 4: 2})

# === SPLIT ===

X_train, X_test, y_train, y_test = train_test_split(
    combined['text'], combined['label'], test_size=0.2, random_state=42
)

# === TOKENIZER ===

tokenizer = Tokenizer(num_words=3000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

maxlen = 30
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

# === CNN MODEL ===

cnn_model = Sequential([
    Embedding(input_dim=3000, output_dim=16, input_length=maxlen),
    Conv1D(16, 3, activation='relu'),
    MaxPooling1D(pool_size=2),
    GRU(16),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

cnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

cnn_model.fit(X_train_pad, y_train, epochs=2, batch_size=64, validation_data=(X_test_pad, y_test))

# === SAVE CNN & TOKENIZER ===

os.makedirs('model', exist_ok=True)
os.makedirs('preprocess', exist_ok=True)

cnn_model.save("model/cnn_model.keras", save_format="keras")
joblib.dump(tokenizer, 'preprocess/tokenizer.pkl')

# === RANDOM FOREST ===

vectorizer = TfidfVectorizer(max_features=1500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf = RandomForestClassifier(n_estimators=25, random_state=42)
rf.fit(X_train_tfidf, y_train)

joblib.dump(rf, 'model/rf_model.pkl')
joblib.dump(vectorizer, 'preprocess/tfidf_vectorizer.pkl')

print("✅ Models trained and saved!")
