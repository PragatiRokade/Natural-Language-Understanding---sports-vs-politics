import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Data
try:
    df = pd.read_csv('final_dataset.csv')
    df.dropna(subset=['text', 'label'], inplace=True)
    print(f"Loaded dataset with {len(df)} articles.")
except FileNotFoundError:
    print("Error: 'final_dataset.csv' not found.")
    exit()

# 2. Define Feature Representations
# We create a dictionary of vectorizers to loop through
feature_extractors = {
    "Bag of Words (BoW)": CountVectorizer(stop_words='english', max_features=5000),
    "TF-IDF": TfidfVectorizer(stop_words='english', max_features=5000),
    "N-Grams (Bi-grams)": CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
}

# 3. Define Models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# 4. Master Loop: Features -> Models
print("\n" + "="*80)
print(f"{'Feature':<20} | {'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
print("="*80)

for feature_name, vectorizer in feature_extractors.items():
    # Transform text using current method
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for model_name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"{feature_name:<20} | {model_name:<20} | {acc:.3f}  | {prec:.3f}  | {rec:.3f}  | {f1:.3f}")

print("="*80)