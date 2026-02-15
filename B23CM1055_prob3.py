import math
import sys
from collections import Counter

def tokenize(text):
    """
    Standard whitespace tokenization and lowercasing as per assignment constraints.
    """
    # Simply lowercase and split by any whitespace
    return text.lower().split()

def train():
    """
    Reads the provided pos.txt and neg.txt files to build the Naive Bayes model.
    """
    try:
        # Open and read training files
        with open('pos.txt', 'r', encoding='utf-8') as f:
            pos_lines = f.readlines()
        with open('neg.txt', 'r', encoding='utf-8') as f:
            neg_lines = f.readlines()
            
        # Create a flat list of all words for each class
        pos_words = []
        for line in pos_lines:
            pos_words.extend(tokenize(line))
            
        neg_words = []
        for line in neg_lines:
            neg_words.extend(tokenize(line))
        
        # Determine the total unique vocabulary size for Laplace smoothing
        vocab = set(pos_words + neg_words)
        v_size = len(vocab)
        
        # Count occurrences of each word per class
        pos_counts = Counter(pos_words)
        neg_counts = Counter(neg_words)
        
        # Store all necessary data for the prediction phase
        return {
            "pos_counts": pos_counts,
            "neg_counts": neg_counts,
            "v_size": v_size,
            "pos_total_words": len(pos_words),
            "neg_total_words": len(neg_words),
            "pos_docs": len(pos_lines),
            "neg_docs": len(neg_lines)
        }
    except FileNotFoundError:
        print("Error: 'pos.txt' and 'neg.txt' files must be in the current directory.")
        sys.exit(1)

def predict(sentence, model):
    """
    Predicts sentiment using Naive Bayes logic with Laplace smoothing.
    """
    words = tokenize(sentence)
    
    # Calculate log priors: P(class)
    total_docs = model["pos_docs"] + model["neg_docs"]
    log_prior_pos = math.log(model["pos_docs"] / total_docs)
    log_prior_neg = math.log(model["neg_docs"] / total_docs)
    
    log_likelihood_pos = 0
    log_likelihood_neg = 0
    
    for word in words:
        # Laplace Smoothing: (count + 1) / (total_words + vocabulary_size)
        # Positive Likelihood
        p_word_pos = (model["pos_counts"].get(word, 0) + 1) / (model["pos_total_words"] + model["v_size"])
        log_likelihood_pos += math.log(p_word_pos)
        
        # Negative Likelihood
        p_word_neg = (model["neg_counts"].get(word, 0) + 1) / (model["neg_total_words"] + model["v_size"])
        log_likelihood_neg += math.log(p_word_neg)
        
    # Final score = log(prior) + sum(log(likelihoods))
    score_pos = log_prior_pos + log_likelihood_pos
    score_neg = log_prior_neg + log_likelihood_neg
    
    return "POSITIVE" if score_pos > score_neg else "NEGATIVE"

if __name__ == "__main__":
    # 1. Train the model using the provided text files
    model_data = train()
    print("Naive Bayes model trained successfully.")
    
    # 2. Enter interactive mode as required by the assignment
    print("Enter a sentence to predict sentiment (or 'exit' to quit):")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'exit':
            break
            
        sentiment = predict(user_input, model_data)
        print(f"Prediction: {sentiment}")