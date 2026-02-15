import sys
import re
from collections import Counter, defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python RollNumber_prob2.py k corpus.txt")
        sys.exit(1)
        
    k = int(sys.argv[1])
    corpus_path = sys.argv[2]
    
    with open(corpus_path, 'r') as f:
        text = f.read().split()
        
    vocab = Counter([' '.join(list(word)) + ' </w>' for word in text])
    
    for i in range(k):
        pairs = get_stats(vocab)
        if not pairs: break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"Step {i+1}: Merged {best}")

    print("\nFinal Vocabulary:")
    unique_tokens = set()
    for word in vocab:
        for token in word.split():
            unique_tokens.add(token)
    print(sorted(list(unique_tokens)))