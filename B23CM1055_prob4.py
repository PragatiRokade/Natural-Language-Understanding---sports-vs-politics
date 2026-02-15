"""
NLU Assignment 1 - Problem 4: Sports vs Politics Classifier
Student Name: [Your Name]
Roll Number: B23CM1055

This script implements a complete pipeline for text classification:
1. Data Processing: Filtering Kaggle data and merging it with scraped content.
2. Feature Representation: Comparing Bag of Words, TF-IDF, and N-Grams.
3. Machine Learning: Comparing Naive Bayes, SVM, and Logistic Regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- SECTION 1: DATA PREPARATION ---

def prepare_dataset():
    """
    Loads the processed Kaggle data and the custom scraped Indian context data.
    Merges them into a single balanced dataset for training.
    """
    try:
        # Load the BBC Kaggle data (assumes it was filtered by filter_kaggle.py)
        # Note: If running for the first time, ensure BBC_News_processed.csv is present
        kaggle_df = pd.read_csv('BBC_News_processed.csv')
        kaggle_df = kaggle_df[kaggle_df['Category'].isin(['sport', 'politics'])]
        kaggle_df = kaggle_df[['Text', 'Category']].rename(columns={'Text': 'text', 'Category': 'label'})
        
        # Load your custom scraped Indian context data
        scraped_df = pd.read_csv('B23CM1055_dataset.csv')
        
        # Combine historical and modern real-time data
        final_df = pd.concat([kaggle_df, scraped_df], ignore_index=True)
        
        # Shuffle to ensure classes are mixed during training
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return final_df
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

# --- SECTION 2: MODEL TRAINING AND EVALUATION ---

def run_experiments(df):
    """
    Evaluates 3 feature representation techniques across 3 ML models.
    """
    # Define our three feature extraction methods
    # 1. BoW: Simple counts
    # 2. TF-IDF: Weighted frequency to highlight unique terms
    # 3. N-Grams: Capturing word pairs (Bi-grams) for context
    feature_extractors = {
        "Bag of Words": CountVectorizer(stop_words='english', max_features=5000),
        "TF-IDF": TfidfVectorizer(stop_words='english', max_features=5000),
        "N-Grams (1,2)": CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    }

    # Define our three ML techniques for comparison
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": SVC(kernel='linear'),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    print("\n" + "="*80)
    print(f"{'Feature':<15} | {'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("="*80)

    for feat_name, vectorizer in feature_extractors.items():
        # Convert text into numerical feature vectors
        X = vectorizer.fit_transform(df['text'])
        y = df['label']
        
        # Split into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in models.items():
            # Training the classifier
            model.fit(X_train, y_train)
            
            # Generating predictions
            y_pred = model.predict(X_test)
            
            # Calculating quantitative metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            print(f"{feat_name:<15} | {model_name:<20} | {acc:.3f}  | {prec:.3f}  | {rec:.3f}  | {f1:.3f}")

    print("="*80)

if __name__ == "__main__":
    dataset = prepare_dataset()
    if dataset is not None:
        print(f"Dataset successfully compiled with {len(dataset)} articles.")
        run_experiments(dataset)