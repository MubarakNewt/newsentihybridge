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

print("🔄 Starting COMPREHENSIVE model training with real datasets...")

# === LOAD REAL DATASETS ===

print("📊 Loading Sentiment140 dataset...")
# Load Sentiment140 (1.6M tweets)
sentiment140 = pd.read_csv(
    '../data/sentiment140/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    nrows=50000  # Load 50k samples for training
)
sentiment140.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
sentiment140 = sentiment140[['target', 'text']]

print("📊 Loading Twitter Airline dataset...")
# Load Twitter Airline dataset
airline = pd.read_csv('../data/twitter-airline/Tweets.csv')
airline = airline[['airline_sentiment', 'text']].dropna()

print(f"📊 Sentiment140 samples: {len(sentiment140)}")
print(f"📊 Airline samples: {len(airline)}")

# === CLEAN AND PREPROCESS ===

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("🧹 Cleaning text data...")
sentiment140['text'] = sentiment140['text'].apply(clean_text)
airline['text'] = airline['text'].apply(clean_text)

# Remove empty texts
sentiment140 = sentiment140[sentiment140['text'].str.len() > 0]
airline = airline[airline['text'].str.len() > 0]

# === FIX LABEL MAPPING ===

# Sentiment140: 0=negative, 4=positive
# Airline: negative=0, neutral=1, positive=2

# Map Sentiment140 labels properly
sentiment140['label'] = sentiment140['target'].map({0: 0, 4: 2})  # 0=negative, 4=positive

# Map Airline labels
airline['label'] = airline['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# === CREATE BALANCED DATASET ===

print("⚖️ Creating balanced dataset...")

# Sample equal numbers from each class
def sample_balanced(df, label_col, samples_per_class=5000):
    balanced_data = []
    for label in [0, 1, 2]:  # negative, neutral, positive
        class_data = df[df[label_col] == label]
        if len(class_data) > 0:
            sampled = class_data.sample(min(samples_per_class, len(class_data)), random_state=42)
            balanced_data.append(sampled)
    return pd.concat(balanced_data, ignore_index=True)

# Sample from Sentiment140 (only has negative and positive)
sentiment140_balanced = sample_balanced(sentiment140, 'label', 5000)

# Sample from Airline (has all three classes)
airline_balanced = sample_balanced(airline, 'label', 5000)

# Combine datasets
combined = pd.concat([sentiment140_balanced, airline_balanced], ignore_index=True)

print(f"📊 Final balanced dataset: {len(combined)} samples")
print(f"📊 Label distribution:")
print(combined['label'].value_counts().sort_index())

# === SPLIT DATA ===

X_train, X_test, y_train, y_test = train_test_split(
    combined['text'], combined['label'], 
    test_size=0.2, random_state=42, stratify=combined['label']
)

print(f"📊 Training set: {len(X_train)}")
print(f"📊 Test set: {len(X_test)}")

# === TRAIN TOKENIZER ===

print("🔤 Training tokenizer...")
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

maxlen = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

# === TRAIN CNN MODEL ===

print("🧠 Training CNN model...")
cnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=maxlen),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, 3, activation='relu'),
    GRU(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

cnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

cnn_history = cnn_model.fit(X_train_pad, y_train, epochs=10, batch_size=64, 
                           validation_data=(X_test_pad, y_test), verbose=1)

# === TRAIN RANDOM FOREST ===

print("🌲 Training Random Forest model...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
rf_model.fit(X_train_tfidf, y_train)

# Evaluate models
cnn_score = cnn_model.evaluate(X_test_pad, y_test, verbose=0)[1]
rf_score = rf_model.score(X_test_tfidf, y_test)

print(f"🧠 CNN Accuracy: {cnn_score:.4f}")
print(f"🌲 RF Accuracy: {rf_score:.4f}")

# === SAVE MODELS ===

os.makedirs('model', exist_ok=True)
os.makedirs('preprocess', exist_ok=True)

cnn_model.save("model/cnn_model.keras")
joblib.dump(tokenizer, 'preprocess/tokenizer.pkl')
joblib.dump(rf_model, 'model/rf_model.pkl')
joblib.dump(vectorizer, 'preprocess/tfidf_vectorizer.pkl')

print("✅ Models trained and saved with comprehensive real data!")

# === TEST MODELS ===

print("\n🧪 Testing models with real examples...")
test_texts = [
    "i love this movie", "i hate this food", "the weather is okay",
    "this is amazing", "this is terrible", "neutral opinion",
    "wonderful experience", "awful service", "good quality"
]

label_map = {0: "negative", 1: "neutral", 2: "positive"}

for text in test_texts:
    cleaned_text = clean_text(text)
    
    # CNN prediction
    seq = tokenizer.texts_to_sequences([cleaned_text])
    pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    cnn_probs = cnn_model.predict(pad, verbose=0)[0]
    cnn_class = int(np.argmax(cnn_probs))
    cnn_conf = float(np.max(cnn_probs))
    
    # RF prediction
    tfidf = vectorizer.transform([cleaned_text])
    rf_probs = rf_model.predict_proba(tfidf)[0]
    rf_class = int(np.argmax(rf_probs))
    rf_conf = float(np.max(rf_probs))
    
    print(f"📝 '{text}' -> CNN: {label_map[cnn_class]} ({cnn_conf:.3f}), RF: {label_map[rf_class]} ({rf_conf:.3f})")

print("\n🎯 Models trained with comprehensive real data!") 