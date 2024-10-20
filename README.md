# Bigram Language Model

This project demonstrates a simple bigram-based language model built from scratch using Python and NumPy. The model predicts the next word in a sequence based on the current word and uses Laplace smoothing to handle unseen word pairs.

## Features
- Tokenizes a small text corpus into words
- Counts occurrences of word pairs (bigrams)
- Applies Laplace smoothing to avoid zero probabilities
- Generates a sentence based on predictions

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SanthoshVijayabaskar/bigram-language-model.git
  ```

2. Navigate to the project directory:
  ```bash
  cd bigram-language-model
```

3. Install the required dependencies:
  ```bash
  pip install numpy

## Running the Program

1. Run the main program:
  ```bash
  python3 main.py
```

2. The program will print:
    - Vocabulary and vocabulary size
    - Bigram probabilities matrix
    - Predicted next word for a given word
    - A generated sentence using the bigram model

## How the Model Works
Laplace Smoothing: We apply smoothing (bigram_counts += 0.01) to avoid zero probabilities when a word pair doesn't appear in the corpus, ensuring every word pair has a non-zero probability.

Word Prediction: The model predicts the next word based on the probability distribution of bigrams.

## Example Output
```bash
  Vocabulary: ['electricity.', 'artificial', 'future', ...]
  Bigram probabilities matrix: [[0.055... 0.055...], [...]]
  Given 'ai', the model predicts 'is'.
  Generated sentence: artificial intelligence is transforming industries...
```

## License
This project is licensed under the MIT License.

