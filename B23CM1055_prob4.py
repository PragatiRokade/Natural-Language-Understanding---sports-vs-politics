"""
NLU Assignment 1 - Problem 4: Sports vs Politics Classifier

Final Submission Script:
- Merges BBC, Scraped, and 20 Newsgroups datasets.
- Implements Regex-based splitting to extract thousands of individual articles.
- Compares 3 feature representations and 3 ML models.
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- SECTION 1: DATA PREPARATION ---

def load_combined_20news(file_path):
    """
    Advanced Regex Parser: Splits the combined master file into individual 
    articles based on newsgroup headers and document IDs.
    """
    data = []
    category_map = {
        'rec.sport.baseball.txt': 'sport',
        'rec.sport.hockey.txt': 'sport',
        'talk.politics.guns.txt': 'politics',
        'talk.politics.mideast.txt': 'politics',
        'talk.politics.misc.txt': 'politics'
    }

    if not os.path.exists(file_path):
        print(f"Warning: '{file_path}' not found. 20 Newsgroups data skipped.")
        return pd.DataFrame(columns=['text', 'label'])

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        full_content = f.read()
        
    # Split the file into the 5 original newsgroup blocks
    file_blocks = full_content.split('--- START OF FILE: ')
    
    for block in file_blocks[1:]:
        try:
            header, body = block.split(' ---', 1)
            filename = header.strip()
            label = category_map.get(filename)
            
            # Use Regex to split by the article start marker found in your files
            # This identifies individual articles within the master text file
            articles = re.split(r'\nNewsgroup:.*\n[Dd]ocument_id:', body)
            
            for article in articles:
                if len(article.strip()) > 100:  # to ensure we aren't picking up fragments
                    # Strip remaining email headers to prevent model 'cheating'
                    if '\n\n' in article:
                        clean_text = article.split('\n\n', 1)[1]
                    else:
                        clean_text = article
                    
                    data.append({'text': clean_text.strip(), 'label': label})
        except Exception:
            continue
            
    return pd.DataFrame(data)

def prepare_dataset():
    """
    Consolidates all data sources into a single structured DataFrame.
    """
    try:
        # 1. Load BBC Kaggle data
        kaggle_df = pd.read_csv('BBC_News_processed.csv')
        kaggle_df = kaggle_df[kaggle_df['Category'].isin(['sport', 'politics'])]
        kaggle_df = kaggle_df[['Text', 'Category']].rename(columns={'Text': 'text', 'Category': 'label'})
        
        # 2. Load custom scraped Indian context data
        scraped_df = pd.read_csv('B23CM1055_dataset.csv')
        scraped_df['label'] = scraped_df['label'].replace({'sports': 'sport'})
        
        # 3. Load Expanded 20 Newsgroups data
        newsgroups_df = load_combined_20news('combined_20news_data.txt')
        
        # Combine all sources
        final_df = pd.concat([kaggle_df, scraped_df, newsgroups_df], ignore_index=True)
        final_df = final_df.dropna(subset=['text', 'label'])
        
        # Shuffle for unbiased training
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("\n" + "="*40)
        print("FINAL DATASET STATISTICS")
        print("="*40)
        print(final_df['label'].value_counts())
        print(f"Total Combined Samples: {len(final_df)}")
        print("="*40)
        
        return final_df
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return None

# --- SECTION 2: TRAINING AND EVALUATION ---

def run_experiments(df):
    """
    Executes a grid-search style evaluation of 3 features x 3 models.
    """
    # 
    feature_extractors = {
        "Bag of Words": CountVectorizer(stop_words='english', max_features=5000),
        "TF-IDF": TfidfVectorizer(stop_words='english', max_features=5000),
        "N-Grams (1,2)": TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    }

    # 
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": SVC(kernel='linear'),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    print("\n" + "="*95)
    print(f"{'Feature Representation':<25} | {'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("="*95)

    for feat_name, vectorizer in feature_extractors.items():
        X = vectorizer.fit_transform(df['text'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            print(f"{feat_name:<25} | {model_name:<20} | {acc:.3f}  | {prec:.3f}  | {rec:.3f}  | {f1:.3f}")

    print("="*95)

if __name__ == "__main__":
    dataset = prepare_dataset()
    if dataset is not None and len(dataset) > 0:
        run_experiments(dataset)