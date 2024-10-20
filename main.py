# This Program is part of the Tutorial Article - https://dev.to/santhoshvijayabaskar/build-your-own-language-model-a-simple-guide-with-python-and-numpy-1k3l
# Author: Santhosh Vijayabaskar

import numpy as np

# Sample dataset: A small text corpus
corpus = """Artificial Intelligence is the new electricity.
Machine learning is the future of AI.
AI is transforming industries and shaping the future."""

# Tokenize the corpus into words
words = corpus.lower().split()

# Create a vocabulary of unique words
vocab = list(set(words))
vocab_size = len(vocab)

print(f"Vocabulary: {vocab}")
print(f"Vocabulary size: {vocab_size}")

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Convert the words in the corpus to indices
corpus_indices = [word_to_idx[word] for word in words]

# Initialize bigram counts matrix
bigram_counts = np.zeros((vocab_size, vocab_size))

# Count occurrences of each bigram in the corpus
for i in range(len(corpus_indices) - 1):
    current_word = corpus_indices[i]
    next_word = corpus_indices[i + 1]
    bigram_counts[current_word, next_word] += 1

# Apply Laplace smoothing by adding 1 to all bigram counts
bigram_counts += 0.01

# Normalize the counts to get probabilities
bigram_probabilities = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)

print("Bigram probabilities matrix: ", bigram_probabilities)


def predict_next_word(current_word, bigram_probabilities):
    word_idx = word_to_idx[current_word]
    next_word_probs = bigram_probabilities[word_idx]
    next_word_idx = np.random.choice(range(vocab_size), p=next_word_probs)
    return idx_to_word[next_word_idx]


# Test the model with a word
current_word = "ai"
next_word = predict_next_word(current_word, bigram_probabilities)
print(f"Given '{current_word}', the model predicts '{next_word}'.")


def generate_sentence(start_word, bigram_probabilities, length=5):
    sentence = [start_word]
    current_word = start_word

    for _ in range(length):
        next_word = predict_next_word(current_word, bigram_probabilities)
        sentence.append(next_word)
        current_word = next_word

    return ' '.join(sentence)


# Generate a sentence starting with "artificial"
generated_sentence = generate_sentence(
    "artificial", bigram_probabilities, length=10)
print(f"Generated sentence: {generated_sentence}")
